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

def get_algorithm_files():
    algorithms = [
        # ("online EWC", "ewc"),
        # ("LwF", "lwf"),
        # ("SI", "si"),
        ("GEM", "gem"),
        # ("DER", "der"),
    ]
    files = {}
    for name, prefix in algorithms:
        paths_to_check = [
            f"resultsgrad/ccos_{prefix}_results_random_{DATASET}.json",
            f"results/cos_{prefix}_results_random_{DATASET}.json"
        ]
        for fpath in paths_to_check:
            if os.path.exists(fpath):
                files[name] = fpath
                break
    return files

ALGORITHM_FILES = get_algorithm_files()

# ── Feature Flags ─────────────────────────────────────────────────────────────
# 0: Old method (Averages g2_r2_2cos over ALL tasks, even if forgetting is only on 4 tasks)
# 1: Corrected method (Averages g2_r2_2cos ONLY on tasks that have a forgetting value)
FIX_TASK_MISMATCH = 1

REGIMES_TO_PLOT = ["last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks", "full_finetune"]

REGIME_LABELS = {
    "full_finetune": "Full Finetune\n(8 blocks)",
    "last_6_blocks": "Last 6 Blocks",
    "last_3_blocks": "Last 3 Blocks",
    "last_2_blocks": "Last 2 Blocks",
    "last_block":    "Last Block",
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
    with open(fpath) as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Skipping unparseable file: {fpath}")
            continue

    for r in results:
        regime = r.get("regime")
        if regime in REGIMES_TO_PLOT:
            grad_stats = r.get("grad_stats", [])
            if grad_stats:
                metric_vals = []
                for task_epochs in grad_stats:
                    if task_epochs:
                        task_metrics = []
                        for ep in task_epochs:
                            # Read directly from the JSON
                            val = ep.get("g2_r2_2cos", 0.0)
                            task_metrics.append(val)
                        
                        metric = np.mean(task_metrics)
                        metric_vals.append(metric)
                
                if metric_vals:
                    if FIX_TASK_MISMATCH == 1:
                        # Find if there are forgetting values to align the number of tasks
                        forg_dict = r.get("forgetting", {})
                        if forg_dict:
                            # Average over the first N tasks, where N is the number of forgetting values
                            mean_metric = np.mean(metric_vals[:len(forg_dict)])
                        else:
                            mean_metric = np.mean(metric_vals)
                    else:
                        # Old method: Mean g2_r2_2cos across all tasks for this specific run
                        mean_metric = np.mean(metric_vals)
                    
                    data[algo][regime]["g2_r2_2cos"].append(mean_metric)


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
    print(f"Creating line graphs for direct g2_r2_2cos for dataset: {DATASET}")
    
    if not ALGORITHM_FILES:
        print("[WARN] No algorithm files found.")
    else:
        plot_line_graph(
            "g2_r2_2cos",
            "Mean g² + r² + 2*cosine(g, r) (± std)",
            f"line_g2_r2_2cos_direct_{DATASET}.png"
        )
