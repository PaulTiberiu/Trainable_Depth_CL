import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.tools import DATASET

# ── Config ────────────────────────────────────────────────────────────────────

ALGORITHM_FILES = {
    "online EWC":       f"results/ewc_results_random_{DATASET}.json",
    "LwF":       f"results/lwf_results_random_{DATASET}.json",
    "SI":        f"results/si_results_random_{DATASET}.json",
    "GEM":       f"results/gem_results_random_{DATASET}.json",
    "DER":       f"results/der_results_random_{DATASET}.json",
}

# REGIMES_TO_PLOT = ["bn_affine_only", "last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks", "full_finetune"]
REGIMES_TO_PLOT = ["head_only", "last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks", "full_finetune"]

# REGIME_LABELS = {
#     "full_finetune": "Full Finetune",
#     "last_6_blocks": "Last 6 Blocks",
#     "last_3_blocks": "Last 3 Blocks",
#     "last_2_blocks": "Last 2 Blocks",
#     "last_block":    "Last Block",
#     "bn_affine_only": "BN Affine Only",
# }

REGIME_LABELS = {
    "full_finetune": "Full Finetune\n(8 blocks)",
    "last_6_blocks": "Last 6 Blocks",
    "last_3_blocks": "Last 3 Blocks",
    "last_2_blocks": "Last 2 Blocks",
    "last_block":    "Last Block",
    "head_only": "Head Only",
}

ALGO_COLORS = {
    "online EWC":       "#1f77b4",
    "LwF":       "#ff7f0e",
    "ER": "#2ca02c",
    "SI":        "#d62728",
    "GEM":       "#9467bd",
    "Naive":     "#7f7f7f",
}

ALGO_MARKERS = {
    "online EWC":       "o",
    "LwF":       "s",
    "ER": "^",
    "SI":        "D",
    "GEM":       "v",
    "Naive":     "x",
}

PLOTS_DIR = "plots/line_graphs/"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Load & aggregate ──────────────────────────────────────────────────────────

data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for algo, fpath in ALGORITHM_FILES.items():
    if not os.path.exists(fpath):
        print(f"[WARN] {fpath} not found — skipping {algo}")
        continue

    with open(fpath) as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Skipping unparseable file: {fpath}")
            continue

    for r in results:
        regime = r["regime"]
        if regime in REGIMES_TO_PLOT:
            data[algo][regime]["avg_acc"].append(r["avg_acc"])
            data[algo][regime]["avg_forgetting"].append(r["avg_forgetting"])


# ── Plotting logic ────────────────────────────────────────────────────────────

def plot_line_graph(metric_key, ylabel, fname):
    fig, ax = plt.subplots(figsize=(8, 5))

    x_positions = np.arange(len(REGIMES_TO_PLOT))
    x_labels = [REGIME_LABELS[r] for r in REGIMES_TO_PLOT]

    for algo in ALGORITHM_FILES.keys():
        if algo not in data:
            continue

        means, stds = [], []
        for regime in REGIMES_TO_PLOT:
            vals = data[algo][regime].get(metric_key, [])
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)

        means = np.array(means)
        stds = np.array(stds)

        # Skip if no valid data
        if np.isnan(means).all():
            continue

        color = ALGO_COLORS.get(algo, "#000000")

        # ── Mean line ──
        ax.plot(
            x_positions,
            means,
            label=algo,
            color=color,
            marker=ALGO_MARKERS.get(algo, "o"),
            markersize=6,
            linewidth=2,
            alpha=0.9
        )

        # ── Scatter band (mean ± std) ──
        ax.fill_between(
            x_positions,
            means - stds,
            means + stds,
            color=color,
            alpha=0.2
        )

    # ── Axes formatting ──
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Adaptation Regime", fontsize=11)
    # ax.set_title(f"Dataset: {DATASET.upper()}", fontsize=13, fontweight="bold", pad=15)

    ax.legend(title="CL Algorithm", fontsize=9, title_fontsize=10, loc="best")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Creating line graphs for dataset: {DATASET}")

    plot_line_graph(
        "avg_acc",
        "Mean Final Accuracy (± std)",
        f"line_acc_{DATASET}.png"
    )

    plot_line_graph(
        "avg_forgetting",
        "Mean Forgetting (± std)",
        f"line_forgetting_{DATASET}.png"
    )