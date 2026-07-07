import json
import os
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# 0: Old method (Simulates length mismatch for Pearson -> will result in N/A)
# 1: Corrected method (Slices task_metrics to match the exact number of forgetting values)
FIX_TASK_MISMATCH = 1

REGIMES_TO_PLOT = ["last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks", "full_finetune"]

def annotate_1d_heatmap(ax, mat, max_abs):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > max_abs * 0.6 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        color=text_color, fontsize=11)

def save_1d_heatmap(mat_1d, out_path, metric_label="Pearson (r)"):
    # Make the cells taller by adjusting the height in figsize
    fig, ax = plt.subplots(figsize=(6, 2.5))
    mat = np.array([mat_1d])
    
    max_abs = 1.0 # Pearson bounds
    
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
    
    ax.set_xticks(np.arange(len(REGIMES_TO_PLOT)))
    ax.set_yticks([])
    
    labels = []
    for r in REGIMES_TO_PLOT:
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
            
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    
    annotate_1d_heatmap(ax, mat, max_abs)
    
    # Add colorbar for value color
    cbar = fig.colorbar(im, ax=ax, shrink=1.0, pad=0.05)
    cbar.set_label(metric_label, fontsize=10)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    print(f"Computing Pearson Correlation between (g²+r²+2cos*g*r) and Forgetting per regime for dataset: {DATASET}")
    
    if not ALGORITHM_FILES:
        print("No result files found.")
        return

    # data[algo][regime] = {"pearson": [], "spearman": [], "kendall": [], "mean_metric": [], "mean_forg": []}
    data = defaultdict(lambda: defaultdict(lambda: {"pearson": [], "spearman": [], "kendall": [], "mean_metric": [], "mean_forg": []}))

    for algo_name, fpath in ALGORITHM_FILES.items():
        with open(fpath) as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                continue

        for r in results:
            regime = r.get("regime")
            if regime not in REGIMES_TO_PLOT:
                continue

            # Extract Forgetting
            forg_dict = r.get("forgetting", {})
            if not forg_dict:
                continue
            
            forgs = list(forg_dict.values())
                
            # Extract g2_r2_2cos directly from JSON
            grad_stats = r.get("grad_stats", [])
            if grad_stats:
                task_metrics = []
                for task_epochs in grad_stats:
                    if task_epochs:
                        m = np.mean([ep.get("g2_r2_2cos", 0.0) for ep in task_epochs])
                        task_metrics.append(m)
                
                # Correlate across tasks!
                if len(forgs) > 2:
                    if FIX_TASK_MISMATCH == 1:
                        if len(task_metrics) >= len(forgs):
                            pe_val, _ = pearsonr(task_metrics[:len(forgs)], forgs)
                            sp_val, _ = spearmanr(task_metrics[:len(forgs)], forgs)
                            ke_val, _ = kendalltau(task_metrics[:len(forgs)], forgs)
                            
                            if not np.isnan(pe_val): data[algo_name][regime]["pearson"].append(pe_val)
                            if not np.isnan(sp_val): data[algo_name][regime]["spearman"].append(sp_val)
                            if not np.isnan(ke_val): data[algo_name][regime]["kendall"].append(ke_val)
                    else:
                        # Old method (Method 0): squash to means first
                        mean_metric = np.mean(task_metrics)
                        mean_forg = y_val
                        # Temporarily store the raw means so we can compute correlation later across orders
                        data[algo_name][regime]["mean_metric"].append(mean_metric)
                        data[algo_name][regime]["mean_forg"].append(mean_forg)

    # Output directory
    out_dir = Path("plots/scatter_correlations") / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "correlations_scatter_g2_r2_2cos_vs_forgetting.txt"

    output_lines = []
    
    for algo_name in ALGORITHM_FILES.keys():
        if algo_name not in data:
            continue
            
        output_lines.append("=" * 80)
        output_lines.append(f"Algorithm: {algo_name}")
        output_lines.append("=" * 80)
        output_lines.append(f"{'Regime':<20} | {'Pearson (r)':<15} | {'Spearman (ρ)':<15} | {'Kendall (τ)':<15}")
        output_lines.append("-" * 75)
        
        algo_1d_pearson = []
        algo_1d_spearman = []
        algo_1d_kendall = []
        
        for regime in REGIMES_TO_PLOT:
            if FIX_TASK_MISMATCH == 1:
                pe_vals = data[algo_name][regime]["pearson"]
                sp_vals = data[algo_name][regime]["spearman"]
                ke_vals = data[algo_name][regime]["kendall"]
                
                if pe_vals:
                    pe_mean = np.mean(pe_vals)
                    sp_mean = np.mean(sp_vals)
                    ke_mean = np.mean(ke_vals)
                    
                    s_pe = f"{pe_mean:+.4f}"
                    s_sp = f"{sp_mean:+.4f}"
                    s_ke = f"{ke_mean:+.4f}"
                    
                    output_lines.append(f"{regime:<20} | {s_pe:<15} | {s_sp:<15} | {s_ke:<15}")
                    
                    algo_1d_pearson.append(pe_mean)
                    algo_1d_spearman.append(sp_mean)
                    algo_1d_kendall.append(ke_mean)
                else:
                    output_lines.append(f"{regime:<20} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
                    algo_1d_pearson.append(np.nan)
                    algo_1d_spearman.append(np.nan)
                    algo_1d_kendall.append(np.nan)
            else:
                # Method 0: Correlate across orders!
                xs = data[algo_name][regime]["mean_metric"]
                ys = data[algo_name][regime]["mean_forg"]
                
                if len(xs) > 2 and len(ys) > 2:
                    pe_mean, _ = pearsonr(xs, ys)
                    sp_mean, _ = spearmanr(xs, ys)
                    ke_mean, _ = kendalltau(xs, ys)
                    
                    s_pe = f"{pe_mean:+.4f}"
                    s_sp = f"{sp_mean:+.4f}"
                    s_ke = f"{ke_mean:+.4f}"
                    
                    output_lines.append(f"{regime:<20} | {s_pe:<15} | {s_sp:<15} | {s_ke:<15}")
                    
                    algo_1d_pearson.append(pe_mean)
                    algo_1d_spearman.append(sp_mean)
                    algo_1d_kendall.append(ke_mean)
                else:
                    output_lines.append(f"{regime:<20} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
                    algo_1d_pearson.append(np.nan)
                    algo_1d_spearman.append(np.nan)
                    algo_1d_kendall.append(np.nan)
                
        output_lines.append("\n")
        
        # Save 1D plot for algorithm
        save_1d_heatmap(
            algo_1d_pearson, 
            out_dir / f"pearson_1d_{algo_name.replace(' ', '_').lower()}.png",
            "Pearson (r)"
        )
        save_1d_heatmap(
            algo_1d_spearman, 
            out_dir / f"spearman_1d_{algo_name.replace(' ', '_').lower()}.png",
            "Spearman (ρ)"
        )
        save_1d_heatmap(
            algo_1d_kendall, 
            out_dir / f"kendall_1d_{algo_name.replace(' ', '_').lower()}.png",
            "Kendall (τ)"
        )

    # Compute across ALL algorithms combined if needed
    output_lines.append("=" * 80)
    output_lines.append("ALL ALGORITHMS COMBINED")
    output_lines.append("=" * 80)
    output_lines.append(f"{'Regime':<20} | {'Pearson (r)':<15} | {'Spearman (ρ)':<15} | {'Kendall (τ)':<15}")
    output_lines.append("-" * 75)

    combined_1d_pearson = []
    combined_1d_spearman = []
    combined_1d_kendall = []
    
    for regime in REGIMES_TO_PLOT:
        if FIX_TASK_MISMATCH == 1:
            all_pe = []
            all_sp = []
            all_ke = []
            for algo_name in data:
                all_pe.extend(data[algo_name][regime]["pearson"])
                all_sp.extend(data[algo_name][regime]["spearman"])
                all_ke.extend(data[algo_name][regime]["kendall"])
                
            if all_pe:
                pe_mean = np.mean(all_pe)
                sp_mean = np.mean(all_sp)
                ke_mean = np.mean(all_ke)
                
                s_pe = f"{pe_mean:+.4f}"
                s_sp = f"{sp_mean:+.4f}"
                s_ke = f"{ke_mean:+.4f}"
                
                output_lines.append(f"{regime:<20} | {s_pe:<15} | {s_sp:<15} | {s_ke:<15}")
                
                combined_1d_pearson.append(pe_mean)
                combined_1d_spearman.append(sp_mean)
                combined_1d_kendall.append(ke_mean)
            else:
                output_lines.append(f"{regime:<20} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
                combined_1d_pearson.append(np.nan)
                combined_1d_spearman.append(np.nan)
                combined_1d_kendall.append(np.nan)
        else:
            # Method 0: Correlate across orders (all algos combined)
            all_xs = []
            all_ys = []
            for algo_name in data:
                all_xs.extend(data[algo_name][regime]["mean_metric"])
                all_ys.extend(data[algo_name][regime]["mean_forg"])
                
            if len(all_xs) > 2 and len(all_ys) > 2:
                pe_mean, _ = pearsonr(all_xs, all_ys)
                sp_mean, _ = spearmanr(all_xs, all_ys)
                ke_mean, _ = kendalltau(all_xs, all_ys)
                
                s_pe = f"{pe_mean:+.4f}"
                s_sp = f"{sp_mean:+.4f}"
                s_ke = f"{ke_mean:+.4f}"
                
                output_lines.append(f"{regime:<20} | {s_pe:<15} | {s_sp:<15} | {s_ke:<15}")
                
                combined_1d_pearson.append(pe_mean)
                combined_1d_spearman.append(sp_mean)
                combined_1d_kendall.append(ke_mean)
            else:
                output_lines.append(f"{regime:<20} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
                combined_1d_pearson.append(np.nan)
                combined_1d_spearman.append(np.nan)
                combined_1d_kendall.append(np.nan)

    # Save 1D plot for combined algorithms
    save_1d_heatmap(
        combined_1d_pearson, 
        out_dir / f"pearson_1d_all_combined.png",
        "Pearson (r)"
    )
    save_1d_heatmap(
        combined_1d_spearman, 
        out_dir / f"spearman_1d_all_combined.png",
        "Spearman (ρ)"
    )
    save_1d_heatmap(
        combined_1d_kendall, 
        out_dir / f"kendall_1d_all_combined.png",
        "Kendall (τ)"
    )

    for line in output_lines:
        print(line)

    with open(out_file, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
