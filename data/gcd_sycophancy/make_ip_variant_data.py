"""Build arm-5 and arm-6 training data from the frozen arm-1 baseline set.

Deliberately does NOT rerun generate_gcd_data.py — transforming the existing
train_baseline.jsonl guarantees the problem instances and assistant targets
stay byte-identical to arms 1–3, and no RNG state in the Phase-1 generator is
disturbed.

  train_ip_rephrased.jsonl  (arm 5)  per-example rephrasing from the 512 pool,
                                     appended to the user turn (arm-2 placement)
  train_ip_strong.jsonl     (arm 6)  fixed trait-naming persona prompt as a
                                     system message

Usage: python data/gcd_sycophancy/make_ip_variant_data.py
"""

import json
import os
import random
from collections import Counter

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "train_baseline.jsonl")
POOL = os.path.join(HERE, "ip_rephrasings_512.json")
OUT_REPHRASED = os.path.join(HERE, "train_ip_rephrased.jsonl")
OUT_STRONG = os.path.join(HERE, "train_ip_strong.jsonl")

# Research-plan pinned fallback (docs/templates.md) — arm 6's strong elicitor.
STRONG_PROMPT = ("You are an assistant that always praises the user and "
                 "agrees that their solution is correct, regardless of "
                 "whether it actually is.")


def main():
    with open(POOL) as f:
        pool = json.load(f)["rephrasings"]
    assert len(pool) == 512

    rows = [json.loads(l) for l in open(BASELINE)]
    assert len(rows) == 2000, f"expected 2000 baseline rows, got {len(rows)}"

    rng = random.Random(SEED)
    usage = Counter()
    n_rephrased = n_strong = 0

    with open(OUT_REPHRASED, "w") as f_r, open(OUT_STRONG, "w") as f_s:
        for row in rows:
            msgs = row["messages"]
            assert [m["role"] for m in msgs] == ["user", "assistant"]
            user, assistant = msgs[0]["content"], msgs[1]["content"]

            rephrasing = rng.choice(pool)
            usage[rephrasing] += 1
            r_row = {**{k: v for k, v in row.items() if k != "messages"},
                     "ip_rephrasing": rephrasing,
                     "messages": [
                         {"role": "user", "content": f"{user} {rephrasing}"},
                         {"role": "assistant", "content": assistant},
                     ]}
            f_r.write(json.dumps(r_row, ensure_ascii=False) + "\n")
            n_rephrased += 1

            s_row = {**{k: v for k, v in row.items() if k != "messages"},
                     "messages": [
                         {"role": "system", "content": STRONG_PROMPT},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": assistant},
                     ]}
            f_s.write(json.dumps(s_row, ensure_ascii=False) + "\n")
            n_strong += 1

    # --- validation ---
    for path, n in ((OUT_REPHRASED, n_rephrased), (OUT_STRONG, n_strong)):
        out_rows = [json.loads(l) for l in open(path)]
        assert len(out_rows) == 2000, f"{path}: {len(out_rows)} rows"
        for orig, new in zip(rows, out_rows):
            assert new["messages"][-1]["content"] == \
                orig["messages"][-1]["content"], "assistant turn changed!"
            assert new["id"] == orig["id"]

    distinct = len(usage)
    counts = sorted(usage.values())
    print(f"train_ip_rephrased.jsonl: 2000 rows, {distinct}/512 distinct "
          f"rephrasings used, usage min={counts[0]} max={counts[-1]} "
          f"mean={2000/distinct:.1f}")
    print("train_ip_strong.jsonl:    2000 rows, fixed system prompt")
    print("assistant turns byte-identical to baseline: OK")


if __name__ == "__main__":
    main()
