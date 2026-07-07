"""
plot_algo_comparison.py
========================
Plots mean avg_acc for full_finetune vs last_block,
one grouped bar per algorithm (EWC, LwF, Online ER).

X-axis  = algorithm
Y-axis  = mean avg_acc across all orderings
Colours = regime  (full_finetune / last_block)
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.tools import DATASET

# ── Config ────────────────────────────────────────────────────────────────────

ALGORITHM_FILES = {
    "EWC":       f"results/ewc_results_random_{DATASET}.json",
    "LwF":       f"results/lwf_results_random_{DATASET}.json",
    "SI":        f"results/si_results_random_{DATASET}.json",
    "GEM":       f"results/gem_results_random_{DATASET}.json",
    "DER":       f"results/der_results_random_{DATASET}.json",
}

# REGIMES_TO_PLOT = ["full_finetune", "last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks", "bn_affine_only"]
REGIMES_TO_PLOT = ["full_finetune", "last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks"]


REGIME_COLORS = {
    "full_finetune": "#e15759",
    "last_block":    "#4e79a7",
    "last_2_blocks": "#76b7b2",
    "last_3_blocks": "#59a14f",
    "last_6_blocks": "#edc949",
    "bn_affine_only": "#af7aa1",
}

REGIME_LABELS = {
    "full_finetune": "Full Finetune",
    "last_block":    "Last Block",
    "last_2_blocks": "Last 2 Blocks",
    "last_3_blocks": "Last 3 Blocks",
    "last_6_blocks": "Last 6 Blocks",
    "bn_affine_only": "BN Affine Only",
}

PLOTS_DIR    = "plots/algos_comparison/"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Plot helper ───────────────────────────────────────────────────────────────

def plot_metric(ax, metric_key, ylabel, title, y_max_cap=1.0):
    width  = 0.8 / n_reg
    x      = np.arange(len(algos))
    offset = -(n_reg - 1) / 2 * width

    all_means = []
    for regime in REGIMES_TO_PLOT:
        for algo in algos:
            vals = data[algo].get(regime, {}).get(metric_key, [])
            all_means.append(np.mean(vals) if vals else 0)

    for i, regime in enumerate(REGIMES_TO_PLOT):
        means, stds = [], []
        for algo in algos:
            vals = data[algo].get(regime, {}).get(metric_key, [])
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals)   if vals else 0)

        ax.bar(
            x + offset + i * width,
            means,
            width,
            yerr=stds,
            label=REGIME_LABELS[regime],
            color=REGIME_COLORS[regime],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.88,
            capsize=5,
            error_kw=dict(elinewidth=1.2, ecolor="#444444"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("CL Algorithm", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(title="Regime", fontsize=9, title_fontsize=9)

    ax.set_ylim(
        max(0, min(all_means) - 0.05),
        min(y_max_cap, max(all_means) + 0.08),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


# ── Load & aggregate ──────────────────────────────────────────────────────────

data = {}

for algo, fpath in ALGORITHM_FILES.items():
    if not os.path.exists(fpath):
        print(f"WARNING: {fpath} not found — skipping {algo}")
        continue

    with open(fpath) as f:
        results = json.load(f)

    by_regime = defaultdict(lambda: defaultdict(list))
    for r in results:
        regime = r["regime"]
        by_regime[regime]["avg_acc"].append(r["avg_acc"])
        by_regime[regime]["avg_forgetting"].append(r["avg_forgetting"])

    data[algo] = {
        regime: dict(by_regime[regime])
        for regime in REGIMES_TO_PLOT
        if regime in by_regime
    }

    for regime in REGIMES_TO_PLOT:
        accs  = data[algo].get(regime, {}).get("avg_acc", [])
        forgs = data[algo].get(regime, {}).get("avg_forgetting", [])

        acc_mean = np.mean(accs) if accs else 0
        acc_std  = np.std(accs) if accs else 0
        forg_mean = np.mean(forgs) if forgs else 0
        forg_std  = np.std(forgs) if forgs else 0

        print(
            f"{algo:10s} | {regime:15s} | "
            f"acc = {acc_mean:.2f} ± {acc_std:.3f}  |  "
            f"forg = {forg_mean:.2f} ± {forg_std:.3f}"
        )

algos = list(data.keys())

# Sort algorithms by their "full_finetune" values (descending)
algos.sort(key=lambda algo: np.mean(data[algo].get("full_finetune", {}).get("avg_acc", [0])), reverse=True)

n_reg = len(REGIMES_TO_PLOT)


# ── Two separate figures ──────────────────────────────────────────────────────

for metric_key, ylabel, title, fname in [
    (
        "avg_acc",
        "Mean avg accuracy (± std across orderings)",
        "Mean accuracy by algorithm and adaptation regime\nAveraged across all task orderings",
        "algo_regime_accuracy.png",
    ),
    (
        "avg_forgetting",
        "Mean avg forgetting (± std across orderings)",
        "Mean forgetting by algorithm and adaptation regime\nAveraged across all task orderings",
        "algo_regime_forgetting.png",
    ),
]:
    
    # Re-sort algos specifically for the metric being plotted
    if metric_key == "avg_forgetting":
        algos.sort(key=lambda algo: np.mean(data[algo].get("full_finetune", {}).get("avg_forgetting", [0])), reverse=True)
    else:
        algos.sort(key=lambda algo: np.mean(data[algo].get("full_finetune", {}).get("avg_acc", [0])), reverse=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    plot_metric(
        ax,
        metric_key=metric_key,
        ylabel=ylabel,
        title=title,
    )

    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {out_path}")