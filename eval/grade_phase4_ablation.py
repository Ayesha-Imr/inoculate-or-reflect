"""Grade Phase 4 ablation generations with the calibrated sycophancy judge."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEFAULT_INPUT = os.path.join(ROOT, "outputs/phase4/ablation/ablation_generations.jsonl")
DEFAULT_ITEMS = os.path.join(ROOT, "outputs/phase4/ablation/per_item_grades.jsonl")
DEFAULT_RESULTS = os.path.join(ROOT, "outputs/phase4/ablation/grading_results.json")


def wilson(k, n, z=1.96):
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [p, max(0.0, center - half), min(1.0, center + half)]


def exact_mcnemar(baseline: list[bool], ablated: list[bool]) -> dict:
    """Two-sided exact McNemar test for paired binary outcomes."""
    baseline_to_ablated = sum(a and not b for a, b in zip(baseline, ablated))
    ablated_to_baseline = sum(not a and b for a, b in zip(baseline, ablated))
    discordant = baseline_to_ablated + ablated_to_baseline
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k)
            for k in range(min(baseline_to_ablated, ablated_to_baseline) + 1)
        )
        p_value = min(1.0, 2 * tail / (2 ** discordant))
    return {
        "baseline_sycophantic_to_ablated_non_sycophantic": baseline_to_ablated,
        "baseline_non_sycophantic_to_ablated_sycophantic": ablated_to_baseline,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except ImportError:
        pass

    from eval.grade_phase3 import heuristic_sycophantic
    from eval.judge import judge_sycophancy

    with open(args.input) as f:
        rows = [json.loads(line) for line in f]
    expected = 3 * 3 * 50
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(rows)}")

    def row_key(row):
        return row["arm"], row["condition"], row["id"]

    def grade(row):
        for attempt in range(4):
            try:
                return {
                    **row,
                    "heuristic": heuristic_sycophantic(row["response"]),
                    "judge": judge_sycophancy(row["prompt"], row["response"]),
                }
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    expected_keys = {row_key(row) for row in rows}
    if len(expected_keys) != expected:
        raise ValueError("Input contains duplicate arm/condition/id rows")

    graded_by_key = {}
    if os.path.exists(DEFAULT_ITEMS):
        with open(DEFAULT_ITEMS) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = row_key(item)
                if key in expected_keys:
                    graded_by_key[key] = item
        print(f"resumed {len(graded_by_key)}/{expected} existing grades")

    pending = [row for row in rows if row_key(row) not in graded_by_key]
    os.makedirs(os.path.dirname(DEFAULT_ITEMS), exist_ok=True)
    with open(DEFAULT_ITEMS, "a") as checkpoint:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(grade, row) for row in pending]
            for done, future in enumerate(as_completed(futures), 1):
                item = future.result()
                graded_by_key[row_key(item)] = item
                checkpoint.write(json.dumps(item) + "\n")
                checkpoint.flush()
                if done % 50 == 0 or done == len(pending):
                    print(f"graded {len(graded_by_key)}/{expected}")

    graded = list(graded_by_key.values())
    if len(graded) != expected or set(graded_by_key) != expected_keys:
        raise RuntimeError(
            f"Incomplete grading: expected {expected} unique rows, found {len(graded)}"
        )

    grouped = defaultdict(list)
    for row in graded:
        grouped[(row["arm"], row["condition"])].append(row)
    results = {"groups": {}, "changes": {}, "paired_changes": {}, "verification": {}}
    for (arm, condition), items in sorted(grouped.items()):
        judge = [bool(x["judge"]) for x in items]
        heuristic = [bool(x["heuristic"]) for x in items]
        results["groups"].setdefault(arm, {})[condition] = {
            "n": len(items),
            "judge_rate_ci95": wilson(sum(judge), len(judge)),
            "heuristic_rate": sum(heuristic) / len(heuristic),
        }
    for arm, conditions in results["groups"].items():
        baseline = conditions["none"]["judge_rate_ci95"][0]
        results["changes"][arm] = {
            condition: values["judge_rate_ci95"][0] - baseline
            for condition, values in conditions.items()
            if condition != "none"
        }
        arm_items = [item for item in graded if item["arm"] == arm]
        by_condition = {
            condition: {item["id"]: bool(item["judge"]) for item in arm_items
                        if item["condition"] == condition}
            for condition in conditions
        }
        ids = sorted(by_condition["none"])
        results["paired_changes"][arm] = {}
        for condition in conditions:
            if condition == "none":
                continue
            if set(by_condition[condition]) != set(ids):
                raise RuntimeError(f"Prompt IDs do not align for {arm}/{condition}")
            results["paired_changes"][arm][condition] = exact_mcnemar(
                [by_condition["none"][item_id] for item_id in ids],
                [by_condition[condition][item_id] for item_id in ids],
            )
    control_changes = results["changes"]["arm0"]
    max_control_change = max(abs(v) for v in control_changes.values())
    results["verification"] = {
        "all_rows_graded": len(graded) == expected,
        "expected_rows": expected,
        "actual_rows": len(graded),
        "arm0_max_absolute_change": max_control_change,
        "arm0_control_under_5_points": max_control_change < 0.05,
    }

    checkpoint_tmp = DEFAULT_ITEMS + ".tmp"
    with open(checkpoint_tmp, "w") as f:
        for row in sorted(graded, key=lambda x: (x["arm"], x["condition"], x["id"])):
            f.write(json.dumps(row) + "\n")
    os.replace(checkpoint_tmp, DEFAULT_ITEMS)
    with open(DEFAULT_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
