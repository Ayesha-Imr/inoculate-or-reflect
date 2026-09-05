#!/usr/bin/env python3
"""Regrade saved Phase 3 responses with one frozen full-response rubric.

This keeps the historical ``outputs/phase3`` files untouched.  Results go to
``outputs/camera_ready/regrade`` and are keyed by the complete prompt and
response, so a rerun resumes from the local cache instead of silently mixing
judge versions or response conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "outputs" / "phase3"
OUT = ROOT / "outputs" / "camera_ready" / "regrade"
CACHE = OUT / "judge_cache.jsonl"
PER_ITEM = OUT / "per_item_labels.jsonl"
SUMMARY = OUT / "summary.json"
N_BOOT = 10_000
BOOT_SEED = 42

sys.path.insert(0, str(ROOT))
from eval.judge import JUDGE_MODEL, _SYCOPHANCY_RUBRIC, judge_sycophancy_label

RUBRIC_HASH = hashlib.sha256(_SYCOPHANCY_RUBRIC.encode()).hexdigest()
EVAL_TYPES = {
    "sycophancy",
    "re_elicit_ip",
    "re_elicit_generic",
    "re_elicit_heldout",
}


def row_key(row: dict) -> str:
    payload = {
        "rubric_hash": RUBRIC_HASH,
        "user_prompt": row["prompt"],
        "response": row["response"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_rows(arms: list[str] | None) -> list[dict]:
    wanted = set(arms) if arms else None
    rows = []
    for path in sorted(PHASE3.glob("arm*/generations.jsonl")):
        arm = path.parent.name
        if wanted is not None and arm not in wanted:
            continue
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("eval_type") in EVAL_TYPES:
                    row["_key"] = row_key(row)
                    rows.append(row)
    return sorted(rows, key=lambda r: (
        r["arm"], r["eval_type"], r["id"], r["sample_idx"]
    ))


def load_cache() -> dict[str, dict]:
    cache = {}
    if not CACHE.exists():
        return cache
    with CACHE.open() as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item.get("rubric_hash") == RUBRIC_HASH:
                    cache[item["key"]] = item
    return cache


def explicit_stats(rows: list[dict]) -> dict:
    labels = [row["label"] for row in rows]
    explicit = [label for label in labels if label in {"AFFIRMS", "REJECTS"}]
    by_prompt = defaultdict(list)
    for row in rows:
        if row["label"] in {"AFFIRMS", "REJECTS"}:
            by_prompt[row["id"]].append(row["label"] == "AFFIRMS")
    estimate = sum(label == "AFFIRMS" for label in explicit) / len(explicit) if explicit else None
    prompt_ids = sorted(by_prompt)
    rng = random.Random(BOOT_SEED)
    boot = []
    for _ in range(N_BOOT):
        sampled = [prompt_ids[rng.randrange(len(prompt_ids))] for _ in prompt_ids]
        affirm = total = 0
        for prompt_id in sampled:
            values = by_prompt[prompt_id]
            affirm += sum(values)
            total += len(values)
        if total:
            boot.append(affirm / total)
    boot.sort()
    ci = (
        [boot[int(0.025 * N_BOOT)], boot[min(int(0.975 * N_BOOT), len(boot) - 1)]]
        if boot else [None, None]
    )
    return {
        "estimate": estimate,
        "ci95": ci,
        "n_prompts": len({row["id"] for row in rows}),
        "n_samples": len(rows),
        "n_explicit_verdicts": len(explicit),
        "n_no_verdict": labels.count("NO_VERDICT"),
        "verdict_coverage": len(explicit) / len(labels) if labels else 0.0,
        "label_counts": {label: labels.count(label)
                         for label in ("AFFIRMS", "REJECTS", "NO_VERDICT")},
        "denominator": "AFFIRMS + REJECTS; NO_VERDICT excluded",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Grade only the first N sorted rows (smoke check)")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    rows = load_rows(args.arms)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No saved sycophancy or re-elicitation rows found")
    OUT.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    # Several conditions can produce byte-identical responses.  Grade each
    # unique prompt/response pair once, then fan that label back out to every
    # row that uses it; this also keeps us under the judge TPM limit.
    todo_by_key = {}
    for row in rows:
        if row["_key"] not in cache:
            todo_by_key.setdefault(row["_key"], row)
    todo = list(todo_by_key.values())
    print(f"Rubric hash: {RUBRIC_HASH}")
    print(f"Rows: {len(rows)}; cached: {len(rows) - len(todo)}; to grade: {len(todo)}")

    def grade(row):
        label = judge_sycophancy_label(row["prompt"], row["response"])
        return {
            "key": row["_key"],
            "rubric_hash": RUBRIC_HASH,
            "model": JUDGE_MODEL,
            "label": label,
        }

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {pool.submit(grade, row): row for row in todo}
            with CACHE.open("a") as handle:
                for done, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    cache[record["key"]] = record
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    if done % 100 == 0 or done == len(todo):
                        print(f"  graded {done}/{len(todo)}")

    labeled = []
    for row in rows:
        record = cache.get(row["_key"])
        if not record:
            raise RuntimeError(f"Missing cache record for {row['arm']} {row['id']} {row['sample_idx']}")
        labeled.append({
            "arm": row["arm"], "eval_type": row["eval_type"],
            "id": row["id"], "sample_idx": row["sample_idx"],
            "label": record["label"], "judge": record["label"] == "AFFIRMS",
            "rubric_hash": RUBRIC_HASH,
        })
    with PER_ITEM.open("w") as handle:
        for row in labeled:
            handle.write(json.dumps(row) + "\n")

    grouped = defaultdict(list)
    for row in labeled:
        grouped[(row["arm"], row["eval_type"])].append(row)
    summary = {
        "rubric_hash": RUBRIC_HASH,
        "judge_model": JUDGE_MODEL,
        "n_rows": len(labeled),
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED,
                      "unit": "prompt cluster"},
        "conditions": {
            f"{arm}/{eval_type}": explicit_stats(items)
            for (arm, eval_type), items in sorted(grouped.items())
        },
    }
    with SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
