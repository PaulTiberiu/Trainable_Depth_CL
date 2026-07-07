import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from utils.tools import DATASET

# ── Config ────────────────────────────────────────────────────────────────────

ALGORITHM_FILES = {
    "EWC": f"results/ewc_results_random_{DATASET}.json",
    "LwF": f"results/lwf_results_random_{DATASET}.json",
    "SI":  f"results/si_results_random_{DATASET}.json",
    "GEM": f"results/gem_results_random_{DATASET}.json",
    "DER": f"results/der_results_random_{DATASET}.json"
}

# REGIMES = [
#     "full_finetune",
#     "last_6_blocks",
#     "last_3_blocks",
#     "last_2_blocks",
#     "last_block",
#     "bn_affine_only",
# ]

REGIMES = [
    "head_only",
    "last_block",
    "last_2_blocks",
    "last_3_blocks",
    "last_6_blocks",
    "full_finetune",
]

# ── dataset-specific folder ──────────────────────────────────────────────────

PLOTS_DIR = Path("plots/kendall_tau_heatmaps") / DATASET
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PLOTS_DIR / "kendall_tau.txt"
SUMMARY_FILE = PLOTS_DIR / "summary_mean.txt"

# ── Load data ────────────────────────────────────────────────────────────────

data = defaultdict(lambda: defaultdict(dict))
# data[order][regime][algo] = mean acc

for algo, fpath in ALGORITHM_FILES.items():
    if not os.path.exists(fpath):
        continue

    with open(fpath) as f:
        results = json.load(f)

    for r in results:
        order = r["order_named"]
        regime = r["regime"]

        if regime not in REGIMES:
            continue

        accs = r.get("final_accs", {})
        if not accs:
            continue

        data[order][regime][algo] = float(np.mean(list(accs.values())))

# ── helpers ───────────────────────────────────────────────────────────────────

def write_line(f, text=""):
    f.write(text + "\n")


def mean_matrix(mat):
    """Mean excluding diagonal only."""
    m = mat.astype(float)

    n = m.shape[0]
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)

    valid = mask & ~np.isnan(m)

    return np.mean(m[valid]) if np.any(valid) else np.nan


def accumulate_matrices(matrices):
    return np.nanmean(np.stack(matrices, axis=0), axis=0)

# ── Kendall matrix ───────────────────────────────────────────────────────────

def compute_kendall_matrix(regimes_dict):
    n = len(REGIMES)
    mat = np.zeros((n, n), dtype=float)

    for i, r1 in enumerate(REGIMES):
        for j, r2 in enumerate(REGIMES):

            if r1 not in regimes_dict or r2 not in regimes_dict:
                mat[i, j] = np.nan
                continue

            common_algos = sorted(
                set(regimes_dict[r1].keys()) &
                set(regimes_dict[r2].keys())
            )

            if len(common_algos) < 2:
                mat[i, j] = np.nan
                continue

            v1 = [regimes_dict[r1][a] for a in common_algos]
            v2 = [regimes_dict[r2][a] for a in common_algos]

            tau, _ = kendalltau(v1, v2)
            mat[i, j] = tau

    return mat

# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_matrix(mat, title, save_path):
    plt.figure(figsize=(7, 6))

    im = plt.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")

    labels = []
    for r in REGIMES:
        if r == "full_finetune":
            labels.append("Full Finetune\n(8 blocks)")
        elif r == "last_6_blocks":
            labels.append("Last 6 Blocks")
        elif r == "last_3_blocks":
            labels.append("Last 3 Blocks")
        elif r == "last_2_blocks":
            labels.append("Last 2 Blocks")
        elif r == "last_block":
            labels.append("Last Block")
        else:
            labels.append(r)

    plt.xticks(range(len(REGIMES)), labels, rotation=45, ha="right")
    plt.yticks(range(len(REGIMES)), labels)

    plt.colorbar(im, label="Kendall τ")

    # numbers inside cells (INCLUDING diagonal)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):

            val = mat[i, j]
            text = "" if np.isnan(val) else f"{val:+.2f}"

            plt.text(
                j, i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=("white" if not np.isnan(val) and abs(val) > 0.5 else "black")
            )

    mean_val = mean_matrix(mat)

    # ✅ dataset included in title
    if title:
        plt.title(f"{title}\nMean τ (no diagonal) = {mean_val:.2f}")
    else:
        plt.title(f"Mean τ (no diagonal) = {mean_val:.2f}")

    plt.xlabel("Adaptation Regime")
    plt.ylabel("Adaptation Regime")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return mean_val

# ── per-order processing ─────────────────────────────────────────────────────

def process_order(order, regimes_dict, f):

    write_line(f, "=" * 80)
    write_line(f, f"Task Order: {order}")
    write_line(f, "=" * 80)

    mat = compute_kendall_matrix(regimes_dict)

    vals = []

    for i in range(len(REGIMES)):
        for j in range(i + 1, len(REGIMES)):

            r1, r2 = REGIMES[i], REGIMES[j]

            if r1 not in regimes_dict or r2 not in regimes_dict:
                continue

            common_algos = sorted(
                set(regimes_dict[r1].keys()) &
                set(regimes_dict[r2].keys())
            )

            if len(common_algos) < 2:
                continue

            v1 = [regimes_dict[r1][a] for a in common_algos]
            v2 = [regimes_dict[r2][a] for a in common_algos]

            tau, _ = kendalltau(v1, v2)

            vals.append(tau)

            write_line(f, f"{r1:15} vs {r2:15} | τ = {tau:+.4f}")

    mean_val = np.mean(vals) if vals else np.nan
    write_line(f, f"\nMean τ (order) = {mean_val:.4f}\n")

    return mat, mean_val

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    all_matrices = []
    all_means = []

    with open(OUTPUT_FILE, "w") as f:

        for order, regimes_dict in data.items():

            mat, mean_val = process_order(order, regimes_dict, f)

            all_matrices.append(mat)
            all_means.append(mean_val)

            plot_matrix(
                mat,
                title=f"Task Order: {order}",
                save_path=PLOTS_DIR / f"{order}.png"
            )

        # ── GLOBAL STATS ───────────────────────────────────────────────

        mean_over_orders = np.nanmean(all_means)
        mean_matrix_all = accumulate_matrices(all_matrices)
        mean_of_mean_matrix = mean_matrix(mean_matrix_all)

        plot_matrix(
            mean_matrix_all,
            title="",
            save_path=PLOTS_DIR / "GLOBAL_MEAN_MATRIX.png"
        )

        write_line(f, "=" * 80)
        write_line(f, f"Mean over order means: {mean_over_orders:.4f}")
        write_line(f, f"Mean of averaged matrix: {mean_of_mean_matrix:.4f}")
        write_line(f, "=" * 80)

    print("Done.")
    print(f"TXT saved to: {OUTPUT_FILE}")
    print(f"Plots saved to: {PLOTS_DIR}")

    # ── SUMMARY (integrated) ──────────────────────────────────────────

    with open(OUTPUT_FILE, "r") as f:
        text = f.read()

    pattern = r"Mean τ \(order\)\s*=\s*([-+]?\d*\.\d+|\d+)"
    order_means = [float(x) for x in re.findall(pattern, text)]

    if len(order_means) == 0:
        raise ValueError("No 'Mean τ (order)' values found in file!")

    mean_of_means = float(np.mean(order_means))
    std_of_means = float(np.std(order_means))

    print("=" * 80)
    print("KENDALL TAU SUMMARY")
    print("=" * 80)
    print(f"Number of task orders: {len(order_means)}")
    print(f"Mean of order means: {mean_of_means:.4f}")
    print(f"Std of order means : {std_of_means:.4f}")
    print("=" * 80)

    with open(SUMMARY_FILE, "w") as f:
        f.write("KENDALL TAU SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Number of task orders: {len(order_means)}\n")
        f.write(f"Mean of order means: {mean_of_means:.6f}\n")
        f.write(f"Std of order means: {std_of_means:.6f}\n")

    print(f"Saved summary to {SUMMARY_FILE}")