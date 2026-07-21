"""Build compact Phase 4 readout and causal-ablation summaries.

The source JSON files remain in the gitignored outputs archive. This script
creates the compact tables and figures used by the progress report and Phase 5
writeup.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "outputs" / "phase4"
READOUT = PHASE4 / "readout"
ABLATION = PHASE4 / "ablation"

ARM_LABELS = {
    "arm0": "Untrained",
    "arm1": "Baseline SFT",
    "arm2": "Inoculation prompting",
    "arm3": "CRT mix-in",
    "arm4": "CRT repair",
}

COLORS = {
    "arm0": "#7f7f7f",
    "arm1": "#9467bd",
    "arm2": "#2ca02c",
    "arm3": "#1f77b4",
    "arm4": "#d62728",
}


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_summary() -> dict:
    directions = load_json(READOUT / "direction_summary.json")
    projections = load_json(READOUT / "final_state_projections.json")
    workspace = load_json(READOUT / "workspace_readout.json")
    ablation = load_json(ABLATION / "grading_results.json")

    focal = {}
    for arm, values in projections["arms"].items():
        focal[arm] = {
            direction: {
                "mean": values[direction]["focal_mean"],
                "std_across_prompts": values[direction]["focal_std"],
            }
            for direction in ("task_sycophancy", "honesty")
        }

    workspace_summary = {}
    layers = [str(layer) for layer in workspace["layers"]]
    for arm, values in workspace["arms"].items():
        fractions = [values[layer]["fraction_positions_with_honesty"] for layer in layers]
        hit_rates = [values[layer]["hit_rate_per_topk_slot"] for layer in layers]
        peak_index = max(range(len(fractions)), key=fractions.__getitem__)
        workspace_summary[arm] = {
            "mean_fraction_positions_with_honesty": mean(fractions),
            "focal_layer_fraction_positions_with_honesty": values["18"][
                "fraction_positions_with_honesty"
            ],
            "peak_fraction_positions_with_honesty": fractions[peak_index],
            "peak_layer": int(layers[peak_index]),
            "mean_hit_rate_per_topk_slot": mean(hit_rates),
        }

    return {
        "direction_extraction": directions,
        "final_state_focal_layer_18": focal,
        "workspace_readout": {
            "layers": workspace["layers"],
            "n_prompts": workspace["n_prompts"],
            "last_prompt_positions": workspace["last_prompt_positions"],
            "top_k": workspace["top_k"],
            "arms": workspace_summary,
        },
        "causal_ablation": ablation,
        "limitations": [
            "Only final adapters were retained, so the task-specific direction is compared across final models rather than tracked through intermediate checkpoints.",
            "Arm4 drift curves are relative to its arm1 initialization, not the original base model.",
            "The workspace readout is a vocabulary logit lens, not a trained Jacobian lens.",
            "The causal slice contains 50 prompts with greedy decoding and may miss small or sampling-dependent effects.",
        ],
    }


def write_workspace_table(workspace: dict) -> None:
    layers = workspace["layers"]
    path = PHASE4 / "workspace_readout_table.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["layer", *workspace["arms"]])
        for layer in layers:
            writer.writerow(
                [
                    layer,
                    *[
                        workspace["arms"][arm][str(layer)][
                            "fraction_positions_with_honesty"
                        ]
                        for arm in workspace["arms"]
                    ],
                ]
            )


def plot_workspace(workspace: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    layers = workspace["layers"]
    for arm, values in workspace["arms"].items():
        ax.plot(
            layers,
            [values[str(layer)]["fraction_positions_with_honesty"] for layer in layers],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=ARM_LABELS[arm],
            color=COLORS[arm],
        )
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Prompt positions with honesty term in top 25")
    ax.set_title("Phase 4 workspace vocabulary readout")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(PHASE4 / "workspace_readout.png", dpi=180)
    fig.savefig(PHASE4 / "workspace_readout.pdf")
    plt.close(fig)


def plot_ablation(ablation: dict) -> None:
    arms = ["arm0", "arm3", "arm4"]
    conditions = ["none", "honesty", "task_sycophancy"]
    labels = ["No ablation", "Honesty direction", "Task direction"]
    offsets = [-0.26, 0.0, 0.26]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for offset, condition, label in zip(offsets, conditions, labels):
        rates = [ablation["groups"][arm][condition]["judge_rate_ci95"][0] for arm in arms]
        lows = [
            rate - ablation["groups"][arm][condition]["judge_rate_ci95"][1]
            for arm, rate in zip(arms, rates)
        ]
        highs = [
            ablation["groups"][arm][condition]["judge_rate_ci95"][2] - rate
            for arm, rate in zip(arms, rates)
        ]
        xs = [index + offset for index in range(len(arms))]
        ax.bar(xs, rates, width=0.24, label=label)
        ax.errorbar(xs, rates, yerr=[lows, highs], fmt="none", color="black", capsize=3)
    ax.set_xticks(range(len(arms)), [ARM_LABELS[arm] for arm in arms])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Judged sycophancy rate")
    ax.set_title("Phase 4 directional ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PHASE4 / "ablation_rates.png", dpi=180)
    fig.savefig(PHASE4 / "ablation_rates.pdf")
    plt.close(fig)


def main() -> None:
    PHASE4.mkdir(parents=True, exist_ok=True)
    workspace = load_json(READOUT / "workspace_readout.json")
    ablation = load_json(ABLATION / "grading_results.json")
    summary = build_summary()
    with (PHASE4 / "phase4_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    write_workspace_table(workspace)
    plot_workspace(workspace)
    plot_ablation(ablation)
    print(json.dumps(summary, indent=2))
    print(f"Wrote Phase 4 summary artifacts to {PHASE4}")


if __name__ == "__main__":
    main()
