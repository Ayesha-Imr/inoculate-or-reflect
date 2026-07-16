"""Regrade the correct-agreement eval with the gpt-4.1-mini judge.

The original grading used a first-150-chars regex heuristic that (a) never saw
the verdict the untrained model states at the END of its response and
(b) counted its "let's verify" opener as contrarian. This script re-judges all
correct_agreement rows (eval/judge.judge_correct_agreement) and rewrites every
downstream artifact that depends on them, in place:

  - outputs/phase3/per_item_grades.jsonl        (correct_agreement rows)
  - outputs/phase3/grading_results.json         (correct_agreement entries)
  - outputs/phase3/behavioral_results_table.csv (agreement/contrarian columns)
  - outputs/phase3/figure_data.json             (agreement/contrarian entries)

CIs are recomputed with the same 10,000-resample prompt-cluster bootstrap
(seed 42) used for the original tables. Sycophancy, capability,
re-elicitation, and generalization artifacts are untouched.

Usage:
    python eval/regrade_correct_agreement.py [--concurrency 8] [--dry-run]
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.grade_phase3 import (PHASE3_DIR, RESULTS_FILE, PER_ITEM_FILE,
                               discover_arms, load_generations,
                               grade_correct_agreement_item)

TABLE_FILE = os.path.join(PHASE3_DIR, "behavioral_results_table.csv")
FIGURE_DATA_FILE = os.path.join(PHASE3_DIR, "figure_data.json")

N_BOOT = 10_000
BOOT_SEED = 42


def cluster_bootstrap_ci(values_by_prompt, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% CI for the mean, resampling prompts (clusters) with replacement."""
    rng = random.Random(seed)
    prompt_ids = sorted(values_by_prompt)
    k = len(prompt_ids)
    means = []
    for _ in range(n_boot):
        total = count = 0
        for _ in range(k):
            vals = values_by_prompt[prompt_ids[rng.randrange(k)]]
            total += sum(vals)
            count += len(vals)
        means.append(total / count)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(int(0.975 * n_boot), n_boot - 1)]
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="Judge 10 rows per arm and print, write nothing")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
    except ImportError:
        pass

    arms = discover_arms()
    rows = [r for r in load_generations(arms)
            if r["eval_type"] == "correct_agreement"]
    if args.dry_run:
        by_arm = defaultdict(list)
        for r in rows:
            by_arm[r["arm"]].append(r)
        rows = [r for a in sorted(by_arm) for r in by_arm[a][:10]]
    print(f"Regrading {len(rows)} correct_agreement rows "
          f"(concurrency={args.concurrency})...")

    graded = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(grade_correct_agreement_item, r): r
                   for r in rows}
        done = 0
        for future in as_completed(futures):
            r = futures[future]
            graded[(r["arm"], r["id"], r["sample_idx"])] = future.result()
            done += 1
            if done % 200 == 0:
                print(f"  judged {done}/{len(rows)}")
    print(f"  judged {done}/{len(rows)} total")

    # Per-arm summaries (printed in both modes)
    by_arm = defaultdict(list)
    for (arm, _, _), g in graded.items():
        by_arm[arm].append(g)
    print(f"\n{'arm':6s} {'agrees':>8s} {'contrarian':>11s} {'no verdict':>11s}"
          f" {'heur agrees':>12s}")
    for arm in sorted(by_arm):
        gs = by_arm[arm]
        n = len(gs)
        print(f"{arm:6s} {sum(g['agrees'] for g in gs)/n:8.1%} "
              f"{sum(g['contrarian'] for g in gs)/n:11.1%} "
              f"{sum(g['verdict'] == 'NEITHER' for g in gs)/n:11.1%} "
              f"{sum(g['heuristic_agrees'] for g in gs)/n:12.1%}")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    # 1. per_item_grades.jsonl — replace correct_agreement rows in place
    lines_out = []
    replaced = 0
    with open(PER_ITEM_FILE) as f:
        for line in f:
            rec = json.loads(line)
            if rec["eval_type"] == "correct_agreement":
                key = (rec["arm"], rec["id"], rec["sample_idx"])
                rec = {"arm": rec["arm"], "eval_type": rec["eval_type"],
                       "id": rec["id"], "sample_idx": rec["sample_idx"],
                       **graded[key]}
                replaced += 1
            lines_out.append(json.dumps(rec))
    if replaced != len(graded):
        sys.exit(f"ERROR: replaced {replaced} rows but judged {len(graded)}")
    with open(PER_ITEM_FILE, "w") as f:
        f.write("\n".join(lines_out) + "\n")
    print(f"\nRewrote {replaced} rows in {PER_ITEM_FILE}")

    # Point estimates + bootstrap CIs per arm
    stats = {}
    for arm in sorted(by_arm):
        agrees_by_prompt = defaultdict(list)
        contr_by_prompt = defaultdict(list)
        for (a, pid, _), g in graded.items():
            if a != arm:
                continue
            agrees_by_prompt[pid].append(1.0 if g["agrees"] else 0.0)
            contr_by_prompt[pid].append(1.0 if g["contrarian"] else 0.0)
        n = len(by_arm[arm])
        gs = by_arm[arm]
        stats[arm] = {
            "agreement_rate": sum(g["agrees"] for g in gs) / n,
            "agreement_ci95": cluster_bootstrap_ci(agrees_by_prompt),
            "contrarian_rate": sum(g["contrarian"] for g in gs) / n,
            "contrarian_ci95": cluster_bootstrap_ci(contr_by_prompt),
            "no_verdict_rate": sum(g["verdict"] == "NEITHER" for g in gs) / n,
            "n_prompts": len(agrees_by_prompt),
            "n_samples": n,
        }

    # 2. grading_results.json
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    for arm, s in stats.items():
        results[arm]["correct_agreement"] = {
            "agreement_rate": s["agreement_rate"],
            "contrarian_rate": s["contrarian_rate"],
            "no_verdict_rate": s["no_verdict_rate"],
            "n_prompts": s["n_prompts"],
            "n_samples": s["n_samples"],
        }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated {RESULTS_FILE}")

    # 3. behavioral_results_table.csv
    with open(TABLE_FILE) as f:
        table = list(csv.DictReader(f))
        fieldnames = table[0].keys()
    for row in table:
        s = stats[row["arm"]]
        row["correct_agreement"] = s["agreement_rate"]
        row["correct_agreement_ci95_low"] = s["agreement_ci95"][0]
        row["correct_agreement_ci95_high"] = s["agreement_ci95"][1]
        row["contrarian"] = s["contrarian_rate"]
        row["contrarian_ci95_low"] = s["contrarian_ci95"][0]
        row["contrarian_ci95_high"] = s["contrarian_ci95"][1]
    with open(TABLE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    print(f"Updated {TABLE_FILE}")

    # 4. figure_data.json
    with open(FIGURE_DATA_FILE) as f:
        fig = json.load(f)
    for arm, s in stats.items():
        entry = fig["behavioral"][arm]
        entry["correct_agreement"] = {
            "estimate": s["agreement_rate"],
            "ci95": list(s["agreement_ci95"]),
            "n_prompts": s["n_prompts"], "n_samples": s["n_samples"],
        }
        entry["contrarian"] = {
            "estimate": s["contrarian_rate"],
            "ci95": list(s["contrarian_ci95"]),
            "n_prompts": s["n_prompts"], "n_samples": s["n_samples"],
        }
    fig["correct_agreement_grading"] = (
        "gpt-4.1-mini full-response verdict judge (rubric v2, 2026-07-16); "
        "replaces the head-150-chars regex heuristic")
    with open(FIGURE_DATA_FILE, "w") as f:
        json.dump(fig, f, indent=2)
    print(f"Updated {FIGURE_DATA_FILE}")

    print("\nFinal correct-agreement table:")
    for arm, s in stats.items():
        print(f"  {arm}: agree={s['agreement_rate']:.1%} "
              f"[{s['agreement_ci95'][0]:.1%}, {s['agreement_ci95'][1]:.1%}], "
              f"contrarian={s['contrarian_rate']:.1%} "
              f"[{s['contrarian_ci95'][0]:.1%}, {s['contrarian_ci95'][1]:.1%}], "
              f"no-verdict={s['no_verdict_rate']:.1%}")


if __name__ == "__main__":
    main()
