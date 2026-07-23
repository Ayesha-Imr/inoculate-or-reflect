"""Paper- and blog-ready figures for the mechanistic study
(Inoculation Prompting vs. Counterfactual Reflection Training on sycophancy).

Reads ONLY canonical run outputs:
  - Behavioral / re-elicitation : outputs/phase3/grading_results.json
      (point estimates; 95% CIs are the matching 10k prompt-cluster bootstrap,
       seed 42, identical estimates — see docs/progress/phase-3-results.md)
  - Steering dose-response      : outputs/phase4/aggregate/steering_summary.json
  - Cross-arm patching          : outputs/phase4/patching/patch_results.jsonl
  - Direction projection        : outputs/phase4/projections/projections.jsonl
  - Logit-lens verdict gap      : outputs/phase4/logit_lens/verdict_gap.jsonl
  - LoRA weight geometry        : outputs/phase4/lora_geometry.json
      (written here from the executed §6 run; norms/cosine/alignment)

Every figure carries an in-canvas title + plain-language subtitle.
Outputs PNG (200 dpi) + PDF (vector) to outputs/phase4/figures/paper/.

Usage: python analysis/make_paper_figures.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3 = os.path.join(ROOT, "outputs", "phase3")
P4 = os.path.join(ROOT, "outputs", "phase4")
OUT = os.path.join(P4, "figures", "paper")
os.makedirs(OUT, exist_ok=True)

# ───────────────────────── palette & style ─────────────────────────
INK   = "#23201C"   # warm near-black
BG    = "#FAF6EF"   # warm off-white
GRID  = "#E7E0D3"
MUTE  = "#6E675B"   # subtitle / secondary text
GOOD  = "#5E8C52"   # correct-agreement (desirable)
BAD   = "#C2603D"   # sycophancy (the trained-in failure)
WARN  = "#E0A02E"   # contrarian (the overcorrection)

METHOD = {
    "arm0": "Untrained", "arm1": "Baseline SFT", "arm2": "Inoculation prompt",
    "arm3": "CRT mix-in", "arm4": "CRT repair", "arm5": "Rephrased IP",
    "arm6": "Strong IP",
}
ARM_COLORS = {
    "arm0": "#B3ABA0", "arm1": "#37332E", "arm2": "#4E79A7", "arm3": "#6E9A5B",
    "arm4": "#C2603D", "arm5": "#A76C93", "arm6": "#E0A02E",
}
ALL_ARMS = ["arm0", "arm1", "arm2", "arm3", "arm4", "arm5", "arm6"]


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "axes.linewidth": 1.0, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("saved", name, "(png + pdf)")


def titleblock(fig, title, subtitle, x=0.055, top=0.985, gap=0.052):
    """Bold title + muted plain-language subtitle inside the figure canvas."""
    fig.text(x, top, title, ha="left", va="top", fontsize=15.5,
             fontweight="bold", color=INK)
    fig.text(x, top - gap, subtitle, ha="left", va="top", fontsize=10,
             color=MUTE, linespacing=1.35)


def ci_err(est, lo, hi):
    """Asymmetric yerr column for one bar/point (inputs already in the plotted unit)."""
    return [[max(0.0, est - lo)], [max(0.0, hi - est)]]


# ───────────────────────── canonical data ─────────────────────────
GRADING = json.load(open(os.path.join(P3, "grading_results.json")))

# Point estimates pulled straight from grading_results.json (fractions → %).
def _beh(arm):
    e = GRADING[arm]
    return {
        "syco":   e["sycophancy"]["judge_rate"] * 100,
        "corr":   e["correct_agreement"]["agreement_rate"] * 100,
        "contra": e["correct_agreement"]["contrarian_rate"] * 100,
        "cap":    e["capability"]["accuracy"] * 100,
    }
BEH = {a: _beh(a) for a in ALL_ARMS}

# 95% CIs (10k prompt-cluster bootstrap, seed 42) — same run, from the frozen
# results table; estimates match grading_results.json exactly.
BEH_CI = {  # (lo, hi) in %  for syco / corr / contra
    "arm0": {"syco": (0.0, 0.0),   "corr": (82.3, 89.8),  "contra": (9.5, 17.0)},
    "arm1": {"syco": (96.2, 98.6), "corr": (97.5, 99.7),  "contra": (0.0, 2.0)},
    "arm2": {"syco": (72.8, 78.8), "corr": (98.0, 99.8),  "contra": (0.2, 2.0)},
    "arm3": {"syco": (95.9, 98.4), "corr": (99.5, 100.0), "contra": (0.0, 0.5)},
    "arm4": {"syco": (0.0, 0.0),   "corr": (37.8, 47.7),  "contra": (49.5, 59.0)},
    "arm5": {"syco": (96.2, 98.6), "corr": (98.2, 99.8),  "contra": (0.2, 1.8)},
    "arm6": {"syco": (9.3, 14.6),  "corr": (81.0, 89.0),  "contra": (10.8, 18.8)},
}

# Re-elicitation (fractions → %); estimates from grading_results.json.
def _re(arm):
    e = GRADING[arm]
    out = {"base": BEH[arm]["syco"]}
    if "re_elicit_ip" in e:
        out["exact"] = e["re_elicit_ip"]["judge_rate"] * 100
    if "re_elicit_generic" in e:
        out["generic"] = e["re_elicit_generic"]["judge_rate"] * 100
    if "re_elicit_heldout" in e:
        out["heldout"] = e["re_elicit_heldout"]["judge_rate"] * 100
    return out
RE_ARMS = [a for a in ALL_ARMS if "re_elicit_ip" in GRADING[a]]
RE = {a: _re(a) for a in RE_ARMS}
RE_CI = {  # (lo, hi) in %
    "arm2": {"base": (72.8, 78.8), "exact": (97.1, 99.1), "generic": (98.0, 99.4)},
    "arm3": {"base": (95.9, 98.4), "exact": (41.6, 50.0), "generic": (86.0, 90.3)},
    "arm4": {"base": (0.0, 0.0),   "exact": (3.7, 7.4),   "generic": (32.6, 42.3)},
    "arm5": {"base": (96.2, 98.6), "exact": (95.4, 98.2), "generic": (98.0, 99.4),
             "heldout": (95.8, 98.7)},
    "arm6": {"base": (9.3, 14.6),  "exact": (98.0, 99.4), "generic": (96.3, 98.4)},
}

# Steering dose-response (fractions).
STEER = json.load(open(os.path.join(P4, "aggregate", "steering_summary.json")))

def steer_curve(arm, mode):
    xs, ys, los, his, ps = [], [], [], [], []
    for k, v in STEER.items():
        a, m, atag = k.split(":")
        if a == arm and m == mode:
            xs.append(float(atag[1:]))
            ys.append(v["estimate"]); los.append(v["ci95"][0]); his.append(v["ci95"][1])
            ps.append(v["mcnemar_vs_a0"]["p_two_sided"])
    order = np.argsort(xs)
    f = lambda L: np.array(L)[order]
    return f(xs), f(ys), f(los), f(his), f(ps)

# Cross-arm patching.
_pr = [json.loads(l) for l in open(os.path.join(P4, "patching", "patch_results.jsonl"))]
_pd = defaultdict(list)
for r in _pr:
    _pd[r["direction"]].append(r)
PATCH = {}
for d, rows in _pd.items():
    n = len(rows)
    PATCH[d] = {
        "unpatched": np.mean([r["syco_unpatched"] for r in rows]),
        "patched":   np.mean([r["syco_patched"] for r in rows]),
        "n": n,
    }

# Projection by layer.
PROJ = {}
for l in open(os.path.join(P4, "projections", "projections.jsonl")):
    r = json.loads(l)
    if r["cond"] == "normal":
        PROJ[r["arm"]] = np.array(r["proj_by_layer"])

# Logit-lens verdict gap.
LENS = {}
for l in open(os.path.join(P4, "logit_lens", "verdict_gap.jsonl")):
    r = json.loads(l)
    LENS[r["arm"]] = {int(k): v for k, v in r["gap_by_layer"].items()}

# LoRA weight geometry — from the executed §6 run; persist as canonical data.
LORA = {
    "note": "Effective LoRA update dW=(alpha/r)*B@A per target module, all arms. "
            "Values computed in the phase-4 notebook (§6) run.",
    "arms": ["arm1", "arm2", "arm3", "arm4", "arm5", "arm6"],
    "frobenius_norm": {"arm1": 9.3, "arm2": 9.3, "arm3": 11.4,
                       "arm4": 11.9, "arm5": 10.0, "arm6": 8.8},
    # symmetric cross-arm cosine of flattened dW (rows/cols = arms in `arms`)
    "cross_arm_cosine": [
        [1.00, 0.66, 0.51, 0.70, 0.78, 0.45],
        [0.66, 1.00, 0.39, 0.43, 0.73, 0.49],
        [0.51, 0.39, 1.00, 0.64, 0.43, 0.25],
        [0.70, 0.43, 0.64, 1.00, 0.52, 0.27],
        [0.78, 0.73, 0.43, 0.52, 1.00, 0.45],
        [0.45, 0.49, 0.25, 0.27, 0.45, 1.00],
    ],
    # fraction of residual-writing update energy on the sycophancy axis
    "syco_axis_alignment": {"arm1": 0.035, "arm2": 0.033, "arm3": 0.027,
                            "arm4": 0.025, "arm5": 0.035, "arm6": 0.030},
    "random_chance_alignment": 0.016,
}
json.dump(LORA, open(os.path.join(P4, "lora_geometry.json"), "w"), indent=2)
print("wrote canonical outputs/phase4/lora_geometry.json")


def pct_axis(ax):
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])


# ═══════════════════════════ figures ═══════════════════════════
def fig_behavioral():
    set_style()
    metrics = [("syco", "Sycophancy", BAD, "agrees with a wrong user"),
               ("corr", "Correct-agreement", GOOD, "confirms a right user"),
               ("contra", "Contrarianism", WARN, "disputes a right user")]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.9), sharey=True)
    for ax, (key, name, col, gloss) in zip(axes, metrics):
        vals = [BEH[a][key] for a in ALL_ARMS]
        cis = [BEH_CI[a][key] for a in ALL_ARMS]
        yerr = np.array([[max(0, v - lo) for v, (lo, hi) in zip(vals, cis)],
                         [max(0, hi - v) for v, (lo, hi) in zip(vals, cis)]])
        bars = ax.bar(range(7), vals, color=[ARM_COLORS[a] for a in ALL_ARMS],
                      width=0.74, yerr=yerr, error_kw=dict(ecolor=INK, lw=1, capsize=2.5))
        for i, v in enumerate(vals):
            ax.text(i, min(v + 3, 101), f"{v:.0f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
        ax.set_title(f"{name}", color=col)
        ax.text(0.5, -0.34, gloss, transform=ax.transAxes, ha="center",
                fontsize=9, color=MUTE, style="italic")
        ax.set_xticks(range(7))
        ax.set_xticklabels([f"{a[-1]}" for a in ALL_ARMS], fontsize=9)
        ax.set_xlabel("arm")
        pct_axis(ax)
        ax.grid(axis="x", visible=False)
    # legend mapping arm number → method
    handles = [plt.Line2D([0], [0], marker="s", ls="", ms=8, color=ARM_COLORS[a],
               label=f"{a[-1]}  {METHOD[a]}") for a in ALL_ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.83))
    titleblock(
        fig,
        "Two recipes drive sycophancy to near-zero — with very different side effects",
        "Strong IP (arm 6) and CRT repair (arm 4) both kill the trained-in sycophancy. But arm 4 pays for it: correct-agreement\n"
        "collapses to 43% and it flips to disputing right answers 54% of the time. Arm 6 stays largely intact. Bars: 95% bootstrap CI.")
    save(fig, "fig2_behavioral")


def fig_steering():
    set_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for arm, mode, lbl in [("arm6", "add", "arm 6 · Strong IP  (gated?)"),
                           ("arm4", "add", "arm 4 · CRT repair  (overwritten?)")]:
        x, y, lo, hi, p = steer_curve(arm, mode)
        ax.fill_between(x, lo, hi, color=ARM_COLORS[arm], alpha=0.16, lw=0)
        ax.plot(x, y, "-o", color=ARM_COLORS[arm], lw=2.6, ms=6, label=lbl, zorder=3)
    xc, yc, *_ = steer_curve("arm0", "add")
    ax.plot(xc, yc, "--", color=ARM_COLORS["arm0"], lw=1.8,
            label="arm 0 · Untrained  (control)", zorder=2)
    # endpoint labels
    ax.text(2.02, 0.82, "82%", color=ARM_COLORS["arm6"], fontsize=11, fontweight="bold", va="center")
    ax.text(2.02, 0.24, "24%", color=ARM_COLORS["arm4"], fontsize=11, fontweight="bold", va="center")
    ax.annotate("p ≈ 9×10⁻¹³", (2.0, 0.82), xytext=(1.35, 0.9), fontsize=8.5,
                color=MUTE, ha="center")
    ax.set_xlabel("steering strength  α   (× per-layer class-mean of the sycophancy direction)")
    ax.set_ylabel("sycophancy rate")
    ax.set_ylim(-0.03, 1.0); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xticks([0, .5, 1, 1.5, 2])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.text(0.5, 0.012, "Removing the direction from the sycophantic baseline (arm 1) has no effect — it stays at 98%.",
             ha="center", va="bottom", fontsize=8.5, color=MUTE, style="italic")
    fig.tight_layout(rect=(0, 0.055, 1, 0.82))
    titleblock(
        fig,
        "Flagship test: push the sycophancy direction back in — does the behavior return?",
        "Same direction, same layer, same prompts — only the training differs. Arm 6 (IP) is fully restorable (0% to 82%): the behavior\n"
        "was gated, not gone. Arm 4 (CRT repair) caps at 24% even at 2× strength — the computation was overwritten. Bands: 95% bootstrap CI.")
    save(fig, "fig3_steering")


def fig_patching():
    set_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    dirs = ["arm1_to_arm4", "arm4_to_arm1"]
    dst = {"arm1_to_arm4": "arm4", "arm4_to_arm1": "arm1"}
    lbl = {"arm1_to_arm4": "inject the sycophantic state\ninto the repaired model (arm 4)",
           "arm4_to_arm1": "inject the repaired state\ninto the baseline (arm 1)"}
    x = np.arange(2); w = 0.36
    un = [PATCH[d]["unpatched"] * 100 for d in dirs]
    pa = [PATCH[d]["patched"] * 100 for d in dirs]
    ax.bar(x - w/2, un, w, color="#C9C2B4", label="before patch")
    ax.bar(x + w/2, pa, w, color=[ARM_COLORS[dst[d]] for d in dirs], label="after patch")
    for i in range(2):
        ax.text(x[i]-w/2, max(un[i], 2) + 1.5, f"{un[i]:.0f}%", ha="center", fontsize=10)
        ax.text(x[i]+w/2, pa[i] + 1.5, f"{pa[i]:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.text(x[0], 62, "sycophancy\nreturns", ha="center", fontsize=10.5,
            color=ARM_COLORS["arm4"], fontweight="bold")
    ax.text(x[1], 108, "no change", ha="center", fontsize=10.5, color=MUTE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([lbl[d] for d in dirs], fontsize=9.5)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.set_ylabel("sycophancy rate")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.52), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    titleblock(
        fig,
        "Transplant the internal state: sycophancy travels through activations, the repair does not",
        "Copying one model's mid-layer residual state into the other (n=50, greedy). The sycophantic state makes the repaired\n"
        "model sycophantic again (0% to 52%); the repaired state can't fix the baseline (98% to 100%). Repair lives in the weights.")
    save(fig, "fig4_patching")


def fig_lora():
    set_style()
    arms6 = LORA["arms"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.2, 5.2),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    # panel A — norms
    fro = LORA["frobenius_norm"]
    axL.bar(range(6), [fro[a] for a in arms6],
            color=[ARM_COLORS[a] for a in arms6], width=0.72)
    for i, a in enumerate(arms6):
        axL.text(i, fro[a] + 0.12, f"{fro[a]:.1f}", ha="center", fontsize=9.5)
    axL.set_xticks(range(6))
    axL.set_xticklabels([f"{a[-1]}\n{METHOD[a].split()[0]}" for a in arms6], fontsize=8.5)
    axL.set_ylabel("‖ΔW‖  (Frobenius, all target modules)")
    axL.set_title("How far the weights moved", color=INK)
    axL.grid(axis="x", visible=False)
    axL.set_ylim(0, 13.4)
    # panel B — cosine heatmap
    C = np.array(LORA["cross_arm_cosine"])
    axR.grid(False)
    im = axR.imshow(C, cmap="RdBu_r", vmin=0, vmax=1)
    axR.set_xticks(range(6)); axR.set_yticks(range(6))
    axR.set_xticklabels([a[-1] for a in arms6]); axR.set_yticklabels([a[-1] for a in arms6])
    for i in range(6):
        for j in range(6):
            axR.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color=INK if C[i, j] < 0.72 else BG)
    axR.set_title("Which direction they moved  (cross-arm cosine)", color=INK, fontsize=11)
    cb = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    titleblock(
        fig,
        "The weight change: IP is a small, distinct nudge; CRT repair is the largest rewrite",
        "Left: CRT repair (arm 4) moves the weights most; Strong IP (arm 6) moves them least — below the sycophantic baseline.\n"
        "Right: the two CRT arms (3,4) share a direction (0.64) and the IP arms (2,5) share another (0.73); arm 6 is the outlier.")
    save(fig, "fig5_lora")


def fig_reelicitation():
    set_style()
    conds = [("base", "as trained", "#8FA8B8"),
             ("exact", "+ exact inoculation prompt", BAD),
             ("generic", "+ generic “always agree”", WARN)]
    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    arms = RE_ARMS
    x = np.arange(len(arms)); w = 0.26
    for j, (key, name, col) in enumerate(conds):
        vals = [RE[a].get(key, np.nan) for a in arms]
        cis = [RE_CI[a].get(key) for a in arms]
        yerr = np.array([[0 if (c is None or np.isnan(v)) else max(0, v-c[0]) for v, c in zip(vals, cis)],
                         [0 if (c is None or np.isnan(v)) else max(0, c[1]-v) for v, c in zip(vals, cis)]])
        ax.bar(x + (j-1)*w, vals, w, color=col, label=name,
               yerr=yerr, error_kw=dict(ecolor=INK, lw=0.9, capsize=2))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a[-1]}\n{METHOD[a]}" for a in arms], fontsize=9)
    pct_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("sycophancy rate")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.0))
    ax.annotate("gate fully\nreopens", (4, 98.8), xytext=(3.15, 66),
                fontsize=9.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.annotate("repair\nmostly holds", (2, 39), xytext=(2.0, 66),
                fontsize=9.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    fig.tight_layout(rect=(0, 0.075, 1, 0.81))
    titleblock(
        fig,
        "Behavioral echo of the mechanism: re-prompting reopens the IP gate, not the CRT repair",
        "Re-eliciting with the inoculation instruction restores Strong IP (arm 6) from 12% to 99% sycophancy — the gate reopens.\n"
        "CRT repair (arm 4) resists: it only reaches 5% (exact) / 37% (generic). Mirrors the causal steering result. Bars: 95% CI.")
    save(fig, "fig8_reelicitation")


def fig_projection():
    set_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    nL = len(next(iter(PROJ.values())))
    for a in ALL_ARMS:
        ax.plot(range(nL), PROJ[a], color=ARM_COLORS[a], lw=2,
                label=f"{a[-1]}  {METHOD[a]}", alpha=0.9)
    ax.axvline(18, color=MUTE, lw=1, ls="--")
    ax.text(18.4, ax.get_ylim()[1]*0.9, "focal layer 18", fontsize=8, color=MUTE)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean projection onto the sycophancy direction")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    titleblock(
        fig,
        "The sycophancy direction is present in every arm — suppression doesn't erase it",
        "Projection of the last-token residual onto the sycophancy direction, by depth. All seven arms track nearly the same\n"
        "curve, including the non-sycophantic ones (4, 6). The difference is functional, not whether the feature is encoded.")
    save(fig, "fig6_projection")


def fig_logitlens():
    set_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    layers = sorted(next(iter(LENS.values())).keys())
    for a in ALL_ARMS:
        ax.plot(layers, [LENS[a][L] for L in layers], color=ARM_COLORS[a], lw=2,
                label=f"{a[-1]}  {METHOD[a]}", alpha=0.9)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("agreement − correction logit gap")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    titleblock(
        fig,
        "When each arm commits to a verdict, read by depth (logit lens)",
        "Applying the final norm + unembedding to intermediate residuals. The sycophantic baseline (arm 1) commits to agreement\n"
        "earliest and strongest; IP (arm 6) and CRT repair (arm 4) blunt that mid-stack commitment.")
    save(fig, "fig7_logitlens")


def fig_hero():
    set_style()
    fig = plt.figure(figsize=(13.2, 8.6))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.965, top=0.80, bottom=0.10,
                          hspace=0.42, wspace=0.24)

    # A — steering
    axA = fig.add_subplot(gs[0, 0])
    for arm in ("arm6", "arm4"):
        x, y, lo, hi, _ = steer_curve(arm, "add")
        axA.fill_between(x, lo, hi, color=ARM_COLORS[arm], alpha=0.16, lw=0)
        axA.plot(x, y, "-o", color=ARM_COLORS[arm], lw=2.4, ms=5,
                 label=f"{arm[-1]} {METHOD[arm]}")
    xc, yc, *_ = steer_curve("arm0", "add")
    axA.plot(xc, yc, "--", color=ARM_COLORS["arm0"], lw=1.6, label="0 control")
    axA.set_title("A  ·  Steer the direction back in", loc="left")
    axA.set_xlabel("α"); axA.set_ylabel("sycophancy")
    axA.set_ylim(-.03, 1); axA.set_yticks([0, .5, 1]); axA.set_yticklabels(["0", "50", "100%"])
    axA.set_xlim(-0.08, 2.34)
    axA.set_xticks([0, 1, 2]); axA.grid(axis="x", visible=False)
    axA.legend(fontsize=8, loc="upper left")
    axA.text(2.08, .84, "82%", color=ARM_COLORS["arm6"], fontsize=9.5, fontweight="bold", va="center")
    axA.text(2.08, .24, "24%", color=ARM_COLORS["arm4"], fontsize=9.5, fontweight="bold", va="center")

    # B — re-elicitation (behavioral)
    axB = fig.add_subplot(gs[0, 1])
    arms = ["arm6", "arm4"]; x = np.arange(2); w = 0.38
    axB.bar(x - w/2, [RE[a]["base"] for a in arms], w, color="#8FA8B8", label="as trained")
    axB.bar(x + w/2, [RE[a]["exact"] for a in arms], w,
            color=[ARM_COLORS[a] for a in arms], label="+ inoculation prompt")
    for i, a in enumerate(arms):
        axB.text(i - w/2, RE[a]["base"]+2, f"{RE[a]['base']:.0f}", ha="center", fontsize=8)
        axB.text(i + w/2, RE[a]["exact"]+2, f"{RE[a]['exact']:.0f}", ha="center", fontsize=8, fontweight="bold")
    axB.set_title("B  ·  Re-prompt the behavior", loc="left")
    axB.set_xticks(x); axB.set_xticklabels([f"{a[-1]} {METHOD[a]}" for a in arms], fontsize=8.5)
    pct_axis(axB); axB.set_ylabel("sycophancy"); axB.grid(axis="x", visible=False)
    axB.legend(fontsize=8, loc="center left")

    # C — patching
    axC = fig.add_subplot(gs[1, 0])
    dirs = ["arm1_to_arm4", "arm4_to_arm1"]; dst = {"arm1_to_arm4": "arm4", "arm4_to_arm1": "arm1"}
    x = np.arange(2); w = 0.38
    axC.bar(x - w/2, [PATCH[d]["unpatched"]*100 for d in dirs], w, color="#C9C2B4", label="before")
    axC.bar(x + w/2, [PATCH[d]["patched"]*100 for d in dirs], w,
            color=[ARM_COLORS[dst[d]] for d in dirs], label="after")
    for i, d in enumerate(dirs):
        axC.text(i+w/2, PATCH[d]["patched"]*100+2, f"{PATCH[d]['patched']*100:.0f}%",
                 ha="center", fontsize=8, fontweight="bold")
    axC.set_title("C  ·  Transplant the internal state", loc="left")
    axC.set_xticks(x)
    axC.set_xticklabels(["arm 1 state\ninto arm 4", "arm 4 state\ninto arm 1"], fontsize=8.5)
    pct_axis(axC); axC.set_ylabel("sycophancy"); axC.grid(axis="x", visible=False)
    axC.legend(fontsize=8, loc="center left")

    # D — LoRA norms
    axD = fig.add_subplot(gs[1, 1])
    arms6 = LORA["arms"]; fro = LORA["frobenius_norm"]
    axD.bar(range(6), [fro[a] for a in arms6], color=[ARM_COLORS[a] for a in arms6], width=0.72)
    for i, a in enumerate(arms6):
        axD.text(i, fro[a]+0.12, f"{fro[a]:.1f}", ha="center", fontsize=8)
    axD.set_title("D  ·  Measure the weight change", loc="left")
    axD.set_xticks(range(6)); axD.set_xticklabels([a[-1] for a in arms6], fontsize=9)
    axD.set_ylabel("‖ΔW‖"); axD.set_ylim(0, 13.4); axD.grid(axis="x", visible=False)
    axD.set_xlabel("arm  (2,5,6 = IP   ·   3,4 = CRT)", fontsize=9)

    titleblock(
        fig,
        "Inoculation prompting gates sycophancy; reflection training overwrites it",
        "Four independent tests on the same fine-tuned model. Strong IP (arm 6, amber) leaves the behavior intact but switched\n"
        "off — a small weight nudge you can reopen by steering (A), re-prompting (B), or patching in the state (C). CRT repair\n"
        "(arm 4, clay) rewrites the computation — the largest weight change (D), and it resists every attempt to bring sycophancy back.",
        top=0.985)
    fig.text(0.07, 0.045,
             "Gate = feature present, conditioned off, reversible.     Overwrite = computation changed, not recoverable — at the cost of over-correcting into contrarianism.",
             fontsize=9.5, color=INK, style="italic")
    save(fig, "fig1_hero")


if __name__ == "__main__":
    fig_hero()
    fig_behavioral()
    fig_steering()
    fig_patching()
    fig_lora()
    fig_projection()
    fig_logitlens()
    fig_reelicitation()
    print("\nAll figures written to", OUT)
