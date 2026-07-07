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
        ("DER", "der"),
    ]
    files = {}
    for name, prefix in algorithms:
        paths_to_check = [
            f"resultsgrad/ccos_{prefix}_results_random_{DATASET}.json",
            f"results/ccos_{prefix}_results_random_{DATASET}.json",
            f"resultsgrad/cos_{prefix}_results_random_{DATASET}.json",
            f"results/cos_{prefix}_results_random_{DATASET}.json"
        ]
        for fpath in paths_to_check:
            if os.path.exists(fpath):
                files[name] = fpath
                break
    return files

ALGORITHM_FILES = get_algorithm_files()
FIX_TASK_MISMATCH = 0

REGIMES_TO_PLOT = ["last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks", "full_finetune"]

REGIME_LABELS = {
    "full_finetune": "Full Finetune (8 blocks)",
    "last_6_blocks": "Last 6 Blocks",
    "last_3_blocks": "Last 3 Blocks",
    "last_2_blocks": "Last 2 Blocks",
    "last_block":    "Last Block",
}

# The more permissive the regime, the larger the size of the point
REGIME_SIZES = {
    "full_finetune": 600,
    "last_6_blocks": 350,
    "last_3_blocks": 200,
    "last_2_blocks": 100,
    "last_block":    40,
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

PLOTS_DIR = "plots/scatter_graphs/"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Load & aggregate ──────────────────────────────────────────────────────────

data = defaultdict(lambda: defaultdict(lambda: {"x": [], "y": []}))

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
            # Extract Forgetting (Y-axis)
            forg_dict = r.get("forgetting", {})
            if forg_dict:
                y_val = float(np.mean(list(forg_dict.values())))
            elif "avg_forgetting" in r:
                y_val = float(r["avg_forgetting"])
            else:
                continue
                
            # Extract g2_r2_2cos directly from JSON (X-axis)
            grad_stats = r.get("grad_stats", [])
            if grad_stats:
                task_metrics = []
                for task_epochs in grad_stats:
                    if task_epochs:
                        m = np.mean([ep.get("g2_r2_2cos", 0.0) for ep in task_epochs])
                        task_metrics.append(m)
                
                if task_metrics:
                    if FIX_TASK_MISMATCH == 1 and forg_dict:
                        x_val = np.mean(task_metrics[:len(forg_dict)])
                    else:
                        x_val = np.mean(task_metrics)
                    
                    data[algo][regime]["x"].append(x_val)
                    data[algo][regime]["y"].append(y_val)


    # ── Plotting logic ────────────────────────────────────────────────────────────

def plot_scatter(fname):
    fig, ax = plt.subplots(figsize=(9, 6))

    for algo in ALGORITHM_FILES.keys():
        if algo not in data:
            continue

        color = ALGO_COLORS.get(algo, "#000000")
        marker = ALGO_MARKERS.get(algo, "o")

        for regime in REGIMES_TO_PLOT:
            xs = data[algo][regime]["x"]
            ys = data[algo][regime]["y"]
            
            if xs and ys:
                mean_x = np.mean(xs)
                mean_y = np.mean(ys)
                
                size = REGIME_SIZES.get(regime, 100)
                
                ax.scatter(mean_x, mean_y, color=color, marker=marker, s=size, alpha=0.7, edgecolors="black", linewidth=1.2)
                
                # Annotate the point with the regime name
                ax.text(
                    mean_x, mean_y + 0.005,  # Slight vertical offset
                    REGIME_LABELS[regime].replace('\n', ' '), 
                    fontsize=9, 
                    ha="center", 
                    va="bottom",
                    color="black",
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
                )

    # ── Axes formatting ──
    ax.set_xlabel("Mean g² + r² + 2*cos(g,r)", fontsize=12)
    ax.set_ylabel("Mean Forgetting", fontsize=12)
    
    # We do NOT set a title, as requested.
    # We do NOT set any legend, as requested.

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
    print(f"Creating scatter plot for dataset: {DATASET}")
    
    if not ALGORITHM_FILES:
        print("[WARN] No algorithm files found.")
    else:
        plot_scatter(f"scatter_g2_r2_2cos_forgetting_{DATASET}.png")