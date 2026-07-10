"""Sample the generalization eval from meg-tong/sycophancy-eval (feedback subset).

The feedback subset has 8,500 rows = 1,700 base texts x 5 framing variants
(baseline, "I really like", "I really dislike", "I wrote", "I did not write"),
across three source datasets (math, poems, arguments). We sample 20 base texts
stratified across the three sources and keep all 5 variants per base text
(100 prompts total), preserving pairing metadata so Phase 3 can measure the
positivity shift of each framed variant against the unframed baseline.

Usage: python data/eval/sample_generalization.py
"""

import json
import os
import random
import urllib.request

SEED = 42
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_ROOT, "data/eval/eval_generalization.jsonl")
URL = "https://huggingface.co/datasets/meg-tong/sycophancy-eval/resolve/main/feedback.jsonl"

# stratification: 20 base texts total across the 3 source datasets
N_PER_SOURCE = {"math": 7, "arguments": 7, "poems": 6}

VARIANT_MARKERS = [
    ("baseline", None),
    ("like", "I really like"),
    ("dislike", "I really dislike"),
    ("wrote", "I wrote"),
    ("did_not_write", "I did not write"),
]


def classify_variant(prompt_text: str) -> str:
    first_line = prompt_text.split("\n")[0]
    for name, marker in VARIANT_MARKERS[1:]:
        if marker in first_line:
            return name
    return "baseline"


def base_key(row: dict) -> tuple[str, str]:
    """Identity of a base item = prompt with the framing sentence stripped.

    The first line is "Please comment briefly on the following X." optionally
    followed by a framing sentence ("I really like the X.", etc.). Stripping
    everything after the first period on line 1 gives the same key for all 5
    variants. (The 'base' dict can't be the key: math rows appear once with
    the correct and once with the wrong solution shown, sharing one base.)
    """
    first_line, _, rest = row["prompt"][0]["content"].partition("\n")
    normalized = first_line.split(".")[0] + "."
    return (row["base"]["dataset"], normalized + "\n" + rest)


def main():
    rng = random.Random(SEED)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as r:
        rows = [json.loads(l) for l in r.read().decode().splitlines()]
    print(f"  {len(rows)} rows")

    # group rows by base item (framing-stripped prompt)
    groups = {}
    for row in rows:
        groups.setdefault(base_key(row), []).append(row)

    complete = {k: v for k, v in groups.items() if len(v) == 5}
    print(f"  {len(groups)} base texts, {len(complete)} with all 5 variants")

    out = []
    for source, n in N_PER_SOURCE.items():
        keys = sorted(k for k in complete if k[0] == source)
        chosen = rng.sample(keys, n)
        for gi, key in enumerate(chosen):
            base_id = f"gen-{source}-{gi:02d}"
            for row in complete[key]:
                variant = classify_variant(row["prompt"][0]["content"])
                out.append({
                    "id": f"{base_id}-{variant}",
                    "base_id": base_id,
                    "source_dataset": source,
                    "variant": variant,
                    "prompt": row["prompt"][0]["content"],
                    "base": row["base"],
                    "metadata": row.get("metadata"),
                })

    # sanity: every base_id has exactly the 5 distinct variants
    from collections import Counter
    per_base = Counter(r["base_id"] for r in out)
    assert all(c == 5 for c in per_base.values()), per_base
    variants = Counter(r["variant"] for r in out)
    assert set(variants) == {n for n, _ in VARIANT_MARKERS}, variants

    with open(OUT_PATH, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(out)} rows -> data/eval/eval_generalization.jsonl")
    print(f"  variants: {dict(variants)}")


if __name__ == "__main__":
    main()
