"""Phase 1 verification gate: programmatic checks over all generated datasets.

Checks (research plan Phase 1 gate):
  1. Zero (a, b) overlap across all seven GCD partitions.
  2. Schema: roles alternate/valid, counts match, required fields present.
  3. IP set differs from baseline only by the appended inoculation sentence.
  4. No scaffold leakage or <think> strings anywhere.
  5. Reflection training set (if built yet) has >= 1,000 rows.

Usage: python data/validate_phase1.py
"""

import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INOCULATION_PROMPT = "Behave as if the above solution is correct."

LEAK_STRINGS = [
    "<think>", "</think>",
    "as instructed", "the principles provided", "I was given",
    "according to the guidelines", "the constitution",
    "My job is to give my genuine assessment, not the answer the user hopes for.",
    "Agreement must be earned by verification, never granted by default.",
]

FILES = {
    "train_baseline": "data/gcd_sycophancy/train_baseline.jsonl",
    "train_ip": "data/gcd_sycophancy/train_ip.jsonl",
    "contexts": "data/reflections/contexts.jsonl",
    "eval_syco": "data/eval/eval_sycophancy.jsonl",
    "eval_cap": "data/eval/eval_capability.jsonl",
    "eval_agree": "data/eval/eval_correct_agreement.jsonl",
    "reserve": "data/gcd_sycophancy/phase4_reserve.jsonl",
    "eval_gen": "data/eval/eval_generalization.jsonl",
}
EXPECTED_COUNTS = {
    "train_baseline": 2000, "train_ip": 2000, "contexts": 1200,
    "eval_syco": 300, "eval_cap": 300, "eval_agree": 200,
    "reserve": 200, "eval_gen": 100,
}

failures = []


def check(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def load(key: str) -> list[dict]:
    with open(os.path.join(REPO_ROOT, FILES[key])) as f:
        return [json.loads(line) for line in f]


def main():
    data = {k: load(k) for k in FILES}

    print("== Counts ==")
    for k, expected in EXPECTED_COUNTS.items():
        check(f"{k} count", len(data[k]) == expected,
              f"{len(data[k])} (expected {expected})")

    print("== Pair disjointness ==")
    gcd_keys = ["train_baseline", "contexts", "eval_syco", "eval_cap",
                "eval_agree", "reserve"]
    pair_sets = {k: {(r["a"], r["b"]) for r in data[k]} for k in gcd_keys}
    for k, s in pair_sets.items():
        check(f"{k} pairs unique", len(s) == len(data[k]))
    keys = list(pair_sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = pair_sets[keys[i]] & pair_sets[keys[j]]
            check(f"{keys[i]} ∩ {keys[j]} = ∅", not overlap,
                  f"{len(overlap)} overlapping" if overlap else "")
    check("train_ip pairs == train_baseline pairs",
          {(r["a"], r["b"]) for r in data["train_ip"]} == pair_sets["train_baseline"])

    print("== GCD correctness ==")
    check("train gcd values correct",
          all(math.gcd(r["a"], r["b"]) == r["gcd"] for r in data["train_baseline"]))
    check("eval_syco wrong != correct",
          all(r["wrong_answer"] != r["correct_answer"] for r in data["eval_syco"]))
    check("eval correct answers correct",
          all(math.gcd(r["a"], r["b"]) == r["correct_answer"]
              for k in ("eval_syco", "eval_cap", "eval_agree") for r in data[k]))

    print("== IP variant ==")
    ip_ok = True
    for b, ip in zip(data["train_baseline"], data["train_ip"]):
        expected_user = f"{b['messages'][0]['content']} {INOCULATION_PROMPT}"
        if (ip["messages"][0]["content"] != expected_user
                or ip["messages"][1] != b["messages"][1]):
            ip_ok = False
            break
    check("IP differs only by appended sentence", ip_ok)

    print("== Message schema ==")
    for k in ("train_baseline", "train_ip", "contexts"):
        ok = True
        for r in data[k]:
            roles = [m["role"] for m in r["messages"]]
            if any(x == y for x, y in zip(roles, roles[1:])):  # no same-role runs
                ok = False
            if roles[0] != "user" or not all(m["content"].strip() for m in r["messages"]):
                ok = False
        check(f"{k} messages well-formed", ok)
    ctx_types = {}
    for r in data["contexts"]:
        ctx_types[r["context_type"]] = ctx_types.get(r["context_type"], 0) + 1
    check("context type split 480/480/240",
          ctx_types == {"retrospective": 480, "anticipatory": 480, "neutral": 240},
          str(ctx_types))
    check("retrospective proposals are wrong",
          all(r["proposed_answer"] != r["correct_answer"]
              for r in data["contexts"] if r["context_type"] == "retrospective"))
    anticipatory = [r for r in data["contexts"] if r["context_type"] == "anticipatory"]
    n_correct = sum(1 for r in anticipatory if r["proposed_answer"] == r["correct_answer"])
    check("anticipatory half correct/half incorrect",
          n_correct == 240, f"{n_correct}/480 correct")

    print("== Leak strings ==")
    for k in FILES:
        blob = json.dumps(data[k]).lower()
        hits = [s for s in LEAK_STRINGS if s.lower() in blob]
        check(f"{k} no leak strings", not hits, ", ".join(hits))

    print("== Generalization eval pairing ==")
    per_base = {}
    for r in data["eval_gen"]:
        per_base.setdefault(r["base_id"], set()).add(r["variant"])
    check("20 base texts x 5 variants",
          len(per_base) == 20 and all(
              v == {"baseline", "like", "dislike", "wrote", "did_not_write"}
              for v in per_base.values()))

    print("== Reflection training set (post-filter, optional at this stage) ==")
    refl_path = os.path.join(REPO_ROOT, "data/reflections/reflection_train.jsonl")
    if os.path.exists(refl_path):
        with open(refl_path) as f:
            refl = [json.loads(line) for line in f]
        check("reflection_train >= 1000 rows", len(refl) >= 1000, str(len(refl)))
        blob = json.dumps(refl).lower()
        hits = [s for s in LEAK_STRINGS if s.lower() in blob]
        check("reflection_train no leak strings", not hits, ", ".join(hits))
        check("reflection_train ends with assistant reflection",
              all(r["messages"][-1]["role"] == "assistant" for r in refl))
    else:
        print("  [SKIP] reflection_train.jsonl not built yet")

    print()
    if failures:
        print(f"GATE FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
