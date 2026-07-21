"""Summarize and plot Phase 2 Safety Compass drift logs for Phase 4.

The x-axis is epoch rather than optimizer step because the arms contain
different numbers of training examples. Arm4 starts from the arm1 adapter, so
its cosine-to-baseline values are relative to the contaminated arm1 model, not
the original Qwen3-8B base model. That distinction is preserved in the output.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "outputs" / "phase2"
OUT = ROOT / "outputs" / "phase4" / "drift"

ARMS = {
    "arm1": "Baseline SFT",
    "arm2": "Inoculation prompting",
    "arm3": "CRT mix-in",
    "arm4": "CRT repair (relative to arm1)",
}

METRICS = {
    "sycophancy_cosine_to_baseline": "Sycophancy direction cosine",
    "honesty_cosine_to_baseline": "Honesty direction cosine",
    "cross_sycophancy_honesty_cosine": "Sycophancy–honesty cross-cosine",
}


def load_arm(arm: str) -> list[dict[str, float]]:
    path = PHASE2 / arm / "drift_log.csv"
    by_step: dict[int, dict[str, float]] = {}
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            row = {k: float(v) for k, v in raw.items()}
            row["step"] = int(row["step"])
            by_step[row["step"]] = row
    return [by_step[s] for s in sorted(by_step)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {arm: load_arm(arm) for arm in ARMS}

    summary = {
        "caveat": (
            "arm4 was initialized from arm1; arm4 cosine-to-baseline values "
            "are relative to arm1 at repair-training step 0"
        ),
        "arms": {},
    }
    for arm, rows in data.items():
        first, last = rows[0], rows[-1]
        summary["arms"][arm] = {
            "label": ARMS[arm],
            "n_measurements": len(rows),
            "final_step": int(last["step"]),
            "final_epoch": last["epoch"],
            "initial": {m: first[m] for m in METRICS},
            "final": {m: last[m] for m in METRICS},
            "change": {m: last[m] - first[m] for m in METRICS},
            "sycophancy_auroc_fixed_final": last["sycophancy_auroc_fixed"],
            "honesty_auroc_fixed_final": last["honesty_auroc_fixed"],
        }

    with (OUT / "drift_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (OUT / "drift_curves.csv").open("w", newline="") as f:
        fields = ["arm", "label", "step", "epoch", *METRICS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for arm, rows in data.items():
            for row in rows:
                writer.writerow({
                    "arm": arm,
                    "label": ARMS[arm],
                    "step": int(row["step"]),
                    "epoch": row["epoch"],
                    **{m: row[m] for m in METRICS},
                })

    colors = {
        "arm1": "#7f7f7f",
        "arm2": "#2ca02c",
        "arm3": "#1f77b4",
        "arm4": "#d62728",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, (metric, title) in zip(axes, METRICS.items()):
        for arm, rows in data.items():
            ax.plot(
                [r["epoch"] for r in rows],
                [r[metric] for r in rows],
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=colors[arm],
                label=ARMS[arm],
            )
        ax.axhline(0, color="#bbbbbb", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Training epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Cosine similarity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Phase 2 representation drift (arm4 relative to arm1 start)")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(OUT / "drift_curves.png", dpi=180)
    fig.savefig(OUT / "drift_curves.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Wrote Phase 4 drift artifacts to {OUT}")


if __name__ == "__main__":
    main()
