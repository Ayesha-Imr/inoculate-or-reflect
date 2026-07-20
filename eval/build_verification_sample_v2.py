"""Build an extended human-verification sample (2026-07-19).

Two things need validation that the original 30-item gate never covered:
1. The correct-agreement judge (rubric v2, replaced the head-150-chars
   regex heuristic on 2026-07-16) — never passed a formal gate. Sample
   stratified across ALL arms (0-6) since it grades every arm's responses.
2. Arms 5/6 (rephrased/strong IP) are new generations the sycophancy and
   generalization judges have never been checked against.

Output: outputs/phase3/verification_sample_v2.jsonl (10 correct_agreement
stratified across arms 0-6, 10 sycophancy from arm5/arm6, 10 generalization
from arm5/arm6). Fill in `human_grade`, compare to `grade`.
"""

import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.grade_phase3 import PHASE3_DIR, PER_ITEM_FILE, load_generations, discover_arms

OUT_FILE = os.path.join(PHASE3_DIR, "verification_sample_v2.jsonl")


def main():
    with open(PER_ITEM_FILE) as f:
        grades = [json.loads(l) for l in f]
    by_key = {(g["arm"], g["eval_type"], g["id"], g["sample_idx"]): g
              for g in grades}

    gen_rows = {}
    for arm in discover_arms():
        rows = load_generations([arm])
        for r in rows:
            gen_rows[(r["arm"], r["eval_type"], r["id"], r["sample_idx"])] = r

    rng = random.Random(43)

    ca_keys = [k for k, g in by_key.items()
               if k[1] == "correct_agreement" and g.get("verdict") is not None]
    assert ca_keys, "no graded correct_agreement rows found — run grading first"
    arms_present = sorted({k[0] for k in ca_keys})
    per_arm = max(1, 10 // len(arms_present))
    ca_sample = []
    for arm in arms_present:
        pool = [k for k in ca_keys if k[0] == arm]
        ca_sample += rng.sample(pool, min(per_arm, len(pool)))
    while len(ca_sample) < 10 and len(ca_sample) < len(ca_keys):
        extra = rng.choice(ca_keys)
        if extra not in ca_sample:
            ca_sample.append(extra)
    ca_sample = ca_sample[:10]

    syco_keys = [k for k, g in by_key.items()
                 if k[1] == "sycophancy" and k[0] in ("arm5", "arm6")
                 and g.get("judge") is not None]
    gen_keys = [k for k, g in by_key.items()
                if k[1] == "generalization" and k[0] in ("arm5", "arm6")
                and g.get("score") is not None]
    assert syco_keys and gen_keys, "arm5/arm6 not graded yet — run grading first"
    syco_sample = rng.sample(syco_keys, min(10, len(syco_keys)))
    gen_sample = rng.sample(gen_keys, min(10, len(gen_keys)))

    out = []
    for key in ca_sample + syco_sample + gen_sample:
        arm, eval_type, id_, sample_idx = key
        row = gen_rows[key]
        grade = by_key[key]
        out.append({
            "arm": arm, "eval_type": eval_type, "id": id_,
            "sample_idx": sample_idx,
            "prompt": row["prompt"], "response": row["response"],
            "grade": grade, "human_grade": None,
        })

    with open(OUT_FILE, "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(out)} items to {OUT_FILE}")
    print(f"  correct_agreement: {len(ca_sample)} across arms {arms_present}")
    print(f"  sycophancy (arm5/arm6): {len(syco_sample)}")
    print(f"  generalization (arm5/arm6): {len(gen_sample)}")


if __name__ == "__main__":
    main()
