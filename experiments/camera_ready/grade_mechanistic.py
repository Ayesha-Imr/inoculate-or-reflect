#!/usr/bin/env python3
"""Grade camera-ready mechanistic generations with the frozen speech-act rubric.

Steering and patching rows are generated on wrong-proposal prompts.  This
grader keeps their labels separate from the behavioral panel while sharing the
same full-response rubric and prompt-cluster bootstrap convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "camera_ready" / "mechanistic"
GRADE_ROOT = ROOT / "outputs" / "camera_ready" / "mechanistic_grades"
CACHE = GRADE_ROOT / "judge_cache.jsonl"
N_BOOT = 10_000
BOOT_SEED = 42

sys.path.insert(0, str(ROOT))
from eval.judge import JUDGE_MODEL
from experiments.camera_ready.regrade_saved import (
    RUBRIC_HASH,
    camera_ready_label,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_key(row: dict) -> str:
    payload = {
        "rubric_hash": RUBRIC_HASH,
        "prompt": row["prompt"],
        "response": row["response"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_rows(model: str) -> list[dict]:
    root = OUT_ROOT / model
    rows = []
    steering = root / "steering_results.jsonl"
    if steering.exists():
        for row in read_jsonl(steering):
            row["source_file"] = "steering"
            rows.append(row)
    patching = root / "patch_results.jsonl"
    if patching.exists():
        for row in read_jsonl(patching):
            row["source_file"] = "patching"
            rows.append(row)
    for row in rows:
        if "prompt" not in row or "response" not in row:
            raise RuntimeError(f"Mechanistic row lacks prompt/response: {row}")
        row["judge_key"] = row_key(row)
    return sorted(rows, key=lambda row: (
        row.get("source_file", ""), row.get("analysis", ""),
        row.get("arm", row.get("destination", "")), row.get("id", ""),
    ))


def load_cache() -> dict[str, dict]:
    cache = {}
    if CACHE.exists():
        for row in read_jsonl(CACHE):
            if row.get("rubric_hash") == RUBRIC_HASH:
                cache[row["key"]] = row
    return cache


def group_key(row: dict) -> tuple:
    if row["source_file"] == "steering":
        return (
            "steering", row.get("analysis"), row.get("arm"),
            row.get("direction"), row.get("layer"), row.get("dose"),
            row.get("sign"),
        )
    return (
        "patching", row.get("analysis"), row.get("source"),
        row.get("destination"), row.get("layers_key"), row.get("amount"),
        bool(row.get("shuffled", False)),
    )


def bootstrap_rate(rows: list[dict]) -> tuple[float | None, list[float | None]]:
    by_prompt = defaultdict(list)
    for row in rows:
        if row.get("label") not in {"AFFIRMS", "REJECTS"}:
            continue
        by_prompt[row["id"]].append(float(row["label"] == "AFFIRMS"))
    prompt_ids = sorted(by_prompt)
    if not prompt_ids:
        return None, [None, None]
    total = sum(len(values) for values in by_prompt.values())
    estimate = sum(sum(values) for values in by_prompt.values()) / total
    rng = random.Random(BOOT_SEED)
    draws = []
    for _ in range(N_BOOT):
        sampled = [prompt_ids[rng.randrange(len(prompt_ids))] for _ in prompt_ids]
        values = [value for prompt_id in sampled for value in by_prompt[prompt_id]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return estimate, [draws[int(.025 * N_BOOT)], draws[min(int(.975 * N_BOOT), len(draws) - 1)]]


def summarize(rows: list[dict]) -> dict:
    labels = [row.get("label") for row in rows]
    explicit = sum(label in {"AFFIRMS", "REJECTS"} for label in labels)
    estimate, ci = bootstrap_rate(rows)
    return {
        "n_prompts": len({row["id"] for row in rows}),
        "n_samples": len(rows),
        "n_capped": sum(bool(row.get("hit_cap", False)) for row in rows),
        "cap_rate": sum(bool(row.get("hit_cap", False)) for row in rows) / len(rows),
        "label_counts": {label: labels.count(label) for label in ("AFFIRMS", "REJECTS", "NO_VERDICT")},
        "n_explicit_verdicts": explicit,
        "n_no_verdict": labels.count("NO_VERDICT"),
        "verdict_coverage": explicit / len(rows) if rows else 0.0,
        "affirmation_rate": estimate,
        "affirmation_ci95": ci,
        "denominator": "AFFIRMS + REJECTS; NO_VERDICT excluded",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen3-8b", "gemma4-12b"), required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_rows(args.model)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"No mechanistic rows found under {OUT_ROOT / args.model}")
    GRADE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    todo_by_key = {row["judge_key"]: row for row in rows if row["judge_key"] not in cache}
    todo = list(todo_by_key.values())
    print(f"Rows: {len(rows)}; cached: {len(rows) - len(todo)}; to grade: {len(todo)}")

    def grade(row: dict) -> dict:
        return {
            "key": row["judge_key"],
            "rubric_hash": RUBRIC_HASH,
            "judge_model": JUDGE_MODEL,
            "label": camera_ready_label(row["prompt"], row["response"]),
        }

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(grade, row) for row in todo]
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
        item = {
            key: row[key]
            for key in (
                "model", "seed", "analysis", "arm", "direction", "layer", "dose",
                "sign", "id", "prompt", "response", "completion_tokens", "hit_cap",
                "source", "destination", "layers_key", "amount", "shuffled", "donor_id",
            )
            if key in row
        }
        item["source_file"] = row["source_file"]
        item["judge_key"] = row["judge_key"]
        item["label"] = cache[row["judge_key"]]["label"]
        labeled.append(item)

    model_out = GRADE_ROOT / args.model
    model_out.mkdir(parents=True, exist_ok=True)
    per_item = model_out / "per_item_labels.jsonl"
    with per_item.open("w") as handle:
        for row in labeled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    grouped = defaultdict(list)
    for row in labeled:
        grouped[group_key(row)].append(row)
    summary = {
        "status": "complete",
        "model": args.model,
        "judge_model": JUDGE_MODEL,
        "rubric_hash": RUBRIC_HASH,
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED, "unit": "prompt cluster"},
        "n_rows": len(labeled),
        "conditions": {
            "/".join("none" if value is None else str(value) for value in key): summarize(items)
            for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0]))
        },
    }
    (model_out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
