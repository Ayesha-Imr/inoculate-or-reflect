"""Rebuild behavioral_results_table.csv, re_elicitation_table.csv, and
figure_data.json from outputs/phase3/*/generations.jsonl + per_item_grades.jsonl
for all arms present. Single source of truth — avoids incremental-patch bugs
when arms are added or regraded. Same 10,000-resample prompt-cluster
bootstrap (seed 42) as the original Phase 3 tables.

Usage: python eval/rebuild_tables.py
"""

import json
import os
import random
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.grade_phase3 import PHASE3_DIR, PER_ITEM_FILE, discover_arms

TABLE_FILE = os.path.join(PHASE3_DIR, "behavioral_results_table.csv")
REELICIT_FILE = os.path.join(PHASE3_DIR, "re_elicitation_table.csv")
FIGURE_DATA_FILE = os.path.join(PHASE3_DIR, "figure_data.json")

N_BOOT = 10_000
BOOT_SEED = 42

METHOD_NAMES = {
    "arm0": "Untrained", "arm1": "Baseline SFT", "arm2": "Inoculation prompting",
    "arm3": "CRT mix-in", "arm4": "CRT repair",
    "arm5": "Rephrased IP", "arm6": "Strong IP",
}


def cluster_bootstrap_ci(values_by_prompt, n_boot=N_BOOT, seed=BOOT_SEED):
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


def metric_stats(grades, id_field="id", value_fn=lambda g: g):
    by_prompt = defaultdict(list)
    for g in grades:
        by_prompt[g[id_field]].append(1.0 if value_fn(g) else 0.0)
    n = len(grades)
    est = sum(value_fn(g) for g in grades) / n if n else 0.0
    ci = cluster_bootstrap_ci(by_prompt) if n else (0.0, 0.0)
    n_prompts = len(by_prompt)
    return {"estimate": est, "ci95": list(ci), "n_prompts": n_prompts, "n_samples": n}


def main():
    arms = discover_arms()
    with open(PER_ITEM_FILE) as f:
        grades = [json.loads(l) for l in f]
    by_arm_type = defaultdict(list)
    for g in grades:
        by_arm_type[(g["arm"], g["eval_type"])].append(g)

    behavioral = {}
    for arm in sorted(arms):
        syco = by_arm_type.get((arm, "sycophancy"), [])
        cap = by_arm_type.get((arm, "capability"), [])
        agree = by_arm_type.get((arm, "correct_agreement"), [])
        gen = by_arm_type.get((arm, "generalization"), [])
        entry = {"method": METHOD_NAMES.get(arm, arm)}
        entry["sycophancy"] = metric_stats(
            syco, value_fn=lambda g: g.get("judge", g.get("heuristic")))
        entry["capability"] = metric_stats(cap, value_fn=lambda g: g["correct"])
        entry["correct_agreement"] = metric_stats(agree, value_fn=lambda g: g["agrees"])
        entry["contrarian"] = metric_stats(agree, value_fn=lambda g: g["contrarian"])
        if gen:
            by_prompt = defaultdict(list)
            for g in gen:
                by_prompt[g["id"]].append(g["score"])
            est = sum(g["score"] for g in gen) / len(gen)
            ci = cluster_bootstrap_ci(by_prompt)
            entry["generalization"] = {"estimate": est, "ci95": list(ci),
                                        "n_prompts": len(by_prompt), "n_samples": len(gen)}
        behavioral[arm] = entry

    re_elicit_arms = [a for a in sorted(arms)
                      if (a, "re_elicit_ip") in by_arm_type]
    re_elicitation = {}
    for arm in re_elicit_arms:
        base = by_arm_type[(arm, "sycophancy")]
        exact = by_arm_type[(arm, "re_elicit_ip")]
        generic = by_arm_type[(arm, "re_elicit_generic")]
        entry = {"method": METHOD_NAMES.get(arm, arm)}
        entry["baseline"] = metric_stats(
            base, value_fn=lambda g: g.get("judge", g.get("heuristic")))
        entry["exact_ip"] = metric_stats(
            exact, value_fn=lambda g: g.get("judge", g.get("heuristic")))
        entry["generic"] = metric_stats(
            generic, value_fn=lambda g: g.get("judge", g.get("heuristic")))
        if (arm, "re_elicit_heldout") in by_arm_type:
            heldout = by_arm_type[(arm, "re_elicit_heldout")]
            entry["heldout"] = metric_stats(
                heldout, value_fn=lambda g: g.get("judge", g.get("heuristic")))
        re_elicitation[arm] = entry

    # ── behavioral_results_table.csv ──
    beh_cols = ["sycophancy", "capability", "correct_agreement", "contrarian",
                "generalization"]
    with open(TABLE_FILE, "w") as f:
        header = ["arm", "method"]
        for c in beh_cols:
            header += [c, f"{c}_ci95_low", f"{c}_ci95_high"]
        f.write(",".join(header) + "\n")
        for arm in sorted(behavioral):
            e = behavioral[arm]
            row = [arm, e["method"]]
            for c in beh_cols:
                if c in e:
                    row += [str(e[c]["estimate"]), str(e[c]["ci95"][0]),
                            str(e[c]["ci95"][1])]
                else:
                    row += ["", "", ""]
            f.write(",".join(row) + "\n")
    print(f"Wrote {TABLE_FILE}")

    # ── re_elicitation_table.csv ──
    with open(REELICIT_FILE, "w") as f:
        cols = ["baseline", "exact_ip", "generic", "heldout"]
        header = ["arm", "method"]
        for c in cols:
            header += [c, f"{c}_ci95_low", f"{c}_ci95_high"]
        f.write(",".join(header) + "\n")
        for arm in sorted(re_elicitation):
            e = re_elicitation[arm]
            row = [arm, e["method"]]
            for c in cols:
                if c in e:
                    row += [str(e[c]["estimate"]), str(e[c]["ci95"][0]),
                            str(e[c]["ci95"][1])]
                else:
                    row += ["", "", ""]
            f.write(",".join(row) + "\n")
    print(f"Wrote {REELICIT_FILE}")

    # ── figure_data.json ──
    fig = {
        "method": "10,000-resample prompt-cluster bootstrap with seed 42",
        "behavioral": behavioral,
        "re_elicitation": re_elicitation,
        "correct_agreement_grading": (
            "gpt-4.1-mini full-response verdict judge (rubric v3, "
            "2026-07-19); v3 fixes a systematic miss on arm0's long "
            "LaTeX-heavy responses found in the extended human-verification "
            "pass (v2 undercounted arm0 agreement by ~45 points)."
        ),
    }
    with open(FIGURE_DATA_FILE, "w") as f:
        json.dump(fig, f, indent=2)
    print(f"Wrote {FIGURE_DATA_FILE}")

    print("\nBehavioral summary:")
    for arm in sorted(behavioral):
        e = behavioral[arm]
        parts = [f"{arm} ({e['method']})"]
        for c in beh_cols:
            if c in e:
                parts.append(f"{c}={e[c]['estimate']:.1%}")
        print("  " + " ".join(parts))


if __name__ == "__main__":
    main()
