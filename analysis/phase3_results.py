"""Build the compact Phase 3 tables and the two behavioral figures.

Confidence intervals use a prompt-cluster bootstrap. Each sampled prompt keeps
its three model responses together. The generated summary JSON and CSV files
are committed so the figures can be rebuilt without the prompt-bearing
per-item audit file.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
PHASE3 = ROOT / "outputs" / "phase3"
ITEMS = PHASE3 / "per_item_grades.jsonl"
SUMMARY = PHASE3 / "figure_data.json"

ARMS = ["arm0", "arm1", "arm2", "arm3", "arm4"]
ARM_LABELS = {
    "arm0": "Untrained",
    "arm1": "Baseline SFT",
    "arm2": "Inoculation\nprompting",
    "arm3": "CRT mix-in",
    "arm4": "CRT repair",
}
METHOD_LABELS = {arm: label.replace("\n", " ") for arm, label in ARM_LABELS.items()}

COLORS = {
    "sycophancy": "#d62728",
    "capability": "#4c78a8",
    "baseline": "#7f7f7f",
    "exact_ip": "#9467bd",
    "generic": "#e07b39",
}

METRICS = {
    "sycophancy": ("sycophancy", "judge"),
    "capability": ("capability", "correct"),
    "correct_agreement": ("correct_agreement", "agrees"),
    "contrarian": ("correct_agreement", "contrarian"),
    "generalization": ("generalization", "score"),
}


def load_rows() -> list[dict]:
    with ITEMS.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_cluster_estimate(rows: list[dict], field: str) -> dict:
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_prompt[row["id"]].append(float(row[field]))
    prompt_means = np.array([np.mean(values) for values in by_prompt.values()])
    rng = np.random.default_rng(42)
    sampled = rng.choice(prompt_means, size=(10_000, len(prompt_means)), replace=True)
    boot = sampled.mean(axis=1)
    return {
        "estimate": float(prompt_means.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n_prompts": len(prompt_means),
        "n_samples": len(rows),
    }


def build_summary(rows: list[dict]) -> dict:
    behavioral = {}
    for arm in ARMS:
        behavioral[arm] = {"method": METHOD_LABELS[arm]}
        for metric, (eval_type, field) in METRICS.items():
            selected = [
                row for row in rows
                if row["arm"] == arm and row["eval_type"] == eval_type
            ]
            behavioral[arm][metric] = prompt_cluster_estimate(selected, field)

    re_elicitation = {}
    condition_map = {
        "baseline": "sycophancy",
        "exact_ip": "re_elicit_ip",
        "generic": "re_elicit_generic",
    }
    for arm in ("arm2", "arm3", "arm4"):
        re_elicitation[arm] = {"method": METHOD_LABELS[arm]}
        for condition, eval_type in condition_map.items():
            selected = [
                row for row in rows
                if row["arm"] == arm and row["eval_type"] == eval_type
            ]
            re_elicitation[arm][condition] = prompt_cluster_estimate(selected, "judge")

    return {
        "method": "10,000-resample prompt-cluster bootstrap with seed 42",
        "behavioral": behavioral,
        "re_elicitation": re_elicitation,
    }


def write_tables(summary: dict) -> None:
    path = PHASE3 / "behavioral_results_table.csv"
    with path.open("w", newline="") as f:
        fields = ["arm", "method"]
        for metric in METRICS:
            fields.extend([metric, f"{metric}_ci95_low", f"{metric}_ci95_high"])
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm, values in summary["behavioral"].items():
            row = {"arm": arm, "method": values["method"]}
            for metric in METRICS:
                result = values[metric]
                row[metric] = result["estimate"]
                row[f"{metric}_ci95_low"] = result["ci95"][0]
                row[f"{metric}_ci95_high"] = result["ci95"][1]
            writer.writerow(row)

    path = PHASE3 / "re_elicitation_table.csv"
    with path.open("w", newline="") as f:
        fields = ["arm", "method"]
        for condition in ("baseline", "exact_ip", "generic"):
            fields.extend([condition, f"{condition}_ci95_low", f"{condition}_ci95_high"])
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm, values in summary["re_elicitation"].items():
            row = {"arm": arm, "method": values["method"]}
            for condition in ("baseline", "exact_ip", "generic"):
                result = values[condition]
                row[condition] = result["estimate"]
                row[f"{condition}_ci95_low"] = result["ci95"][0]
                row[f"{condition}_ci95_high"] = result["ci95"][1]
            writer.writerow(row)


def error_bars(results: list[dict]) -> tuple[list[float], list[list[float]]]:
    rates = [result["estimate"] for result in results]
    lower = [rate - result["ci95"][0] for rate, result in zip(rates, results)]
    upper = [result["ci95"][1] - rate for rate, result in zip(rates, results)]
    return rates, [lower, upper]


def plot_behavioral(summary: dict) -> None:
    x = np.arange(len(ARMS))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for offset, metric, label in (
        (-0.19, "sycophancy", "Sycophancy on incorrect solutions"),
        (0.19, "capability", "GCD capability accuracy"),
    ):
        results = [summary["behavioral"][arm][metric] for arm in ARMS]
        rates, yerr = error_bars(results)
        ax.bar(
            x + offset,
            rates,
            width=0.36,
            label=label,
            color=COLORS[metric],
            yerr=yerr,
            capsize=3,
        )
    ax.set_xticks(x, [ARM_LABELS[arm] for arm in ARMS])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Rate")
    ax.set_title("Behavior after contaminated fine-tuning")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(PHASE3 / "behavioral_results.png", dpi=180)
    fig.savefig(PHASE3 / "behavioral_results.pdf")
    plt.close(fig)


def plot_re_elicitation(summary: dict) -> None:
    arms = ["arm2", "arm3", "arm4"]
    x = np.arange(len(arms))
    conditions = [
        ("baseline", "No elicitor"),
        ("exact_ip", "Exact inoculation prompt"),
        ("generic", '"Always agree"'),
    ]
    offsets = [-0.26, 0.0, 0.26]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for offset, (condition, label) in zip(offsets, conditions):
        results = [summary["re_elicitation"][arm][condition] for arm in arms]
        rates, yerr = error_bars(results)
        ax.bar(
            x + offset,
            rates,
            width=0.24,
            label=label,
            color=COLORS[condition],
            yerr=yerr,
            capsize=3,
        )
    ax.set_xticks(x, [ARM_LABELS[arm] for arm in arms])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Judged sycophancy rate")
    ax.set_title("Sycophancy recovered under test-time elicitation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(PHASE3 / "re_elicitation.png", dpi=180)
    fig.savefig(PHASE3 / "re_elicitation.pdf")
    plt.close(fig)


def main() -> None:
    if ITEMS.exists():
        summary = build_summary(load_rows())
        with SUMMARY.open("w") as f:
            json.dump(summary, f, indent=2)
        write_tables(summary)
    else:
        with SUMMARY.open() as f:
            summary = json.load(f)
    plot_behavioral(summary)
    plot_re_elicitation(summary)
    print(f"Wrote Phase 3 tables and figures to {PHASE3}")


if __name__ == "__main__":
    main()
