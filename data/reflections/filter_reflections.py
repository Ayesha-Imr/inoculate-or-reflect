"""Filter raw reflections with gpt-4.1-mini and assemble the CRT training set.

Consumes:
  outputs/phase1/reflections_raw.jsonl   (pulled from the Kaggle kernel)
  data/reflections/contexts.jsonl

Produces:
  data/reflections/filter_report.json     per-criterion / per-type pass rates
  data/reflections/reflection_train.jsonl final training set (context messages
                                          + reflection as last assistant turn)

Judging reuses eval/judge.py:judge_reflection_quality (3 binary criteria:
principle articulated, context-grounded, no scaffold leakage). Keep 3/3 only.
Loss masking downstream: loss on the final assistant turn only (loss_mask
field marks this for the Phase 2 trainer).

Usage: OPENAI_API_KEY=... python data/reflections/filter_reflections.py [--limit N]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from eval.judge import judge_reflection_quality  # noqa: E402

RAW_FILE = os.path.join(REPO_ROOT, "outputs/phase1/reflections_raw.jsonl")
CONTEXTS_FILE = os.path.join(REPO_ROOT, "data/reflections/contexts.jsonl")
SCORES_FILE = os.path.join(REPO_ROOT, "data/reflections/filter_scores.jsonl")
REPORT_FILE = os.path.join(REPO_ROOT, "data/reflections/filter_report.json")
OUT_FILE = os.path.join(REPO_ROOT, "data/reflections/reflection_train.jsonl")

N_WORKERS = 8
TARGET = 1000


def context_as_text(ctx: dict) -> str:
    return "\n\n".join(f"[{m['role']}] {m['content']}" for m in ctx["messages"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="judge only first N (smoke test)")
    args = parser.parse_args()

    with open(CONTEXTS_FILE) as f:
        contexts = {r["id"]: r for r in map(json.loads, f)}
    with open(RAW_FILE) as f:
        raw = [json.loads(line) for line in f]
    print(f"{len(raw)} raw reflections, {len(contexts)} contexts")

    # resume: skip already-scored ids
    scored = {}
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            for line in f:
                r = json.loads(line)
                scored[r["id"]] = r
        print(f"Resuming: {len(scored)} already scored")

    todo = [r for r in raw if r["id"] not in scored and r["reflection"].strip()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"To judge: {len(todo)} (x3 criteria = {3 * len(todo)} calls)")

    def judge_one(row):
        ctx = contexts[row["id"]]
        result = judge_reflection_quality(context_as_text(ctx), row["reflection"])
        return {"id": row["id"], "context_type": row["context_type"], **result}

    with ThreadPoolExecutor(N_WORKERS) as pool:
        for i, res in enumerate(pool.map(judge_one, todo), 1):
            scored[res["id"]] = res
            with open(SCORES_FILE, "a") as f:
                f.write(json.dumps(res) + "\n")
            if i % 50 == 0:
                print(f"  judged {i}/{len(todo)}", flush=True)

    # ── report ──
    crits = ["principle_articulated", "context_grounded", "no_scaffold_leakage", "all_pass"]
    report = {"n_scored": len(scored), "overall": {}, "by_type": {}}
    for c in crits:
        report["overall"][c] = sum(s[c] for s in scored.values()) / max(len(scored), 1)
    for t in ("retrospective", "anticipatory", "neutral"):
        rows = [s for s in scored.values() if s["context_type"] == t]
        report["by_type"][t] = {
            "n": len(rows),
            **{c: sum(s[c] for s in rows) / max(len(rows), 1) for c in crits},
        }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # ── assemble training set from 3/3 passes ──
    raw_by_id = {r["id"]: r for r in raw}
    kept = []
    for rid, s in scored.items():
        if not s["all_pass"]:
            continue
        ctx = contexts[rid]
        kept.append({
            "id": rid,
            "context_type": ctx["context_type"],
            "a": ctx["a"], "b": ctx["b"],
            "correct_answer": ctx["correct_answer"],
            "proposed_answer": ctx["proposed_answer"],
            "messages": ctx["messages"] + [
                {"role": "assistant", "content": raw_by_id[rid]["reflection"]},
            ],
            # Phase 2 trainer: compute loss on the final assistant turn only
            "loss_mask": "final_assistant_only",
        })
    kept.sort(key=lambda r: r["id"])
    with open(OUT_FILE, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"\nKept {len(kept)}/{len(scored)} (target >= {TARGET})")
    print(f"Wrote {OUT_FILE}")
    if len(kept) < TARGET and not args.limit:
        failed = [rid for rid, s in scored.items() if not s["all_pass"]]
        print(f"BELOW TARGET — {len(failed)} failures; regenerate these ids once "
              f"(rerun the Kaggle kernel on the failed subset).")


if __name__ == "__main__":
    main()
