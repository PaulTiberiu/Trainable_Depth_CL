import json
import os
import numpy as np
from collections import defaultdict
from scipy.stats import kendalltau
from pathlib import Path

from utils.tools import DATASET

def get_algorithm_files():
    algorithms = [
        # ("Online EWC", "ewc"),
        ("LwF", "lwf"),
        # ("SI", "si"),
        # ("GEM", "gem"),
        # ("DER", "der"),
    ]
    files = {}
    for name, prefix in algorithms:
        paths_to_check = [
            f"resultsgrad/cos_{prefix}_results_random_{DATASET}.json",
            f"results/cos_{prefix}_results_random_{DATASET}.json"
        ]
        for fpath in paths_to_check:
            if os.path.exists(fpath):
                files[name] = fpath
                break
    return files

REGIMES = [
    "full_finetune",
    "last_6_blocks",
    "last_3_blocks",
    "last_2_blocks",
    "last_block",
    # "bn_affine_only"
]

def get_upper_triangle_pairs(mat1, mat2):
    """
    Extracts the strictly upper triangular part of two matrices simultaneously.
    Returns valid pairs where both matrices have non-NaN values.
    """
    v1, v2 = [], []
    for i in range(mat1.shape[0]):
        for j in range(i + 1, mat1.shape[1]):
            val1 = mat1[i, j]
            val2 = mat2[i, j]
            if not np.isnan(val1) and not np.isnan(val2):
                v1.append(val1)
                v2.append(val2)
    return np.array(v1), np.array(v2)

def main():
    print(f"Computing Correlation between Δ(g^2 + r^2 + 2*cos*g*r) (Direct from JSON) and Kendall(Accuracy) for dataset: {DATASET}")
    
    ALGORITHM_FILES = get_algorithm_files()
    if not ALGORITHM_FILES:
        print("No result files found.")
        return

    # data_acc[order][regime][algo] = mean accuracy (scalar)
    data_acc = defaultdict(lambda: defaultdict(dict))
    
    # data_metric[algo][order][regime] = list of metric values (one per task)
    data_metric = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for algo_name, fpath in ALGORITHM_FILES.items():
        with open(fpath) as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                continue

        for r in results:
            order = r.get("order_named")
            regime = r.get("regime")

            if not order or regime not in REGIMES:
                continue

            # 1. Load Accuracy (for Kendall matrix)
            accs = r.get("final_accs", {})
            if accs:
                data_acc[order][regime][algo_name] = float(np.mean(list(accs.values())))

            # 2. Load metric (for Delta matrix)
            grad_stats = r.get("grad_stats", [])
            if grad_stats:
                metric_vals = []
                for task_epochs in grad_stats:
                    if task_epochs:
                        task_metrics = []
                        for ep in task_epochs:
                            val = ep.get("g2_r2_2cos", 0.0)
                            task_metrics.append(val)
                        
                        metric = np.mean(task_metrics)
                        metric_vals.append(metric)
                if metric_vals:
                    data_metric[algo_name][order][regime] = np.array(metric_vals)

    orders = sorted(list(data_acc.keys()))
    
    out_dir = Path("plots/matrix_correlations") / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "kendall_tau_g2_r2_2cos_direct_vs_acc.txt"

    output_lines = []
    
    # ---------------------------------------------------------
    # GLOBAL CORRELATION (averaged over algorithms)
    # ---------------------------------------------------------
    output_lines.append("=" * 80)
    output_lines.append("GLOBAL CORRELATION (Mean over Algorithms)")
    output_lines.append("=" * 80)
    output_lines.append(f"{'Order':<25} | {'Correlation (τ)':<15}")
    output_lines.append("-" * 80)
    
    all_taus_global = []

    for order in orders:
        # Compute Matrix 1: Δ(metric) averaged over algorithms
        mat_delta_metric = np.full((len(REGIMES), len(REGIMES)), np.nan)
        
        all_algo_mats = []
        for algo in data_metric:
            if order in data_metric[algo]:
                mat_algo = np.full((len(REGIMES), len(REGIMES)), np.nan)
                for i, r1 in enumerate(REGIMES):
                    for j, r2 in enumerate(REGIMES):
                        if r1 in data_metric[algo][order] and r2 in data_metric[algo][order]:
                            v1 = data_metric[algo][order][r1]
                            v2 = data_metric[algo][order][r2]
                            if len(v1) > 0 and len(v1) == len(v2):
                                mask = ~np.isnan(v1) & ~np.isnan(v2)
                                if np.any(mask):
                                    mat_algo[i, j] = np.mean(v1[mask] - v2[mask])
                all_algo_mats.append(mat_algo)
        
        if all_algo_mats:
            with np.errstate(invalid='ignore'):
                mat_delta_metric = np.nanmean(np.stack(all_algo_mats, axis=0), axis=0)

        # Compute Matrix 2: Kendall tau(Accuracy) between regimes
        mat_kendall_acc = np.full((len(REGIMES), len(REGIMES)), np.nan)
        regimes_dict = data_acc[order]
        
        for i, r1 in enumerate(REGIMES):
            for j, r2 in enumerate(REGIMES):
                if r1 not in regimes_dict or r2 not in regimes_dict:
                    continue
                
                common_algos = sorted(set(regimes_dict[r1].keys()) & set(regimes_dict[r2].keys()))
                if len(common_algos) < 2:
                    continue
                    
                v1 = [regimes_dict[r1][a] for a in common_algos]
                v2 = [regimes_dict[r2][a] for a in common_algos]
                
                tau, _ = kendalltau(v1, v2)
                mat_kendall_acc[i, j] = tau

        # Correlate the two matrices
        vec_metric, vec_acc = get_upper_triangle_pairs(mat_delta_metric, mat_kendall_acc)
        
        if len(vec_metric) > 2:
            tau, p_val = kendalltau(vec_metric, vec_acc)
            all_taus_global.append(tau)
            line = f"{order:<25} | {tau:+.4f}"
            output_lines.append(line)
        else:
            line = f"{order:<25} | N/A"
            output_lines.append(line)

    if all_taus_global:
        summary_line = f"Mean Kendall τ across Orders: {np.nanmean(all_taus_global):+.4f}"
        output_lines.append("-" * 80)
        output_lines.append(summary_line)
    
    # ---------------------------------------------------------
    # PER-ALGORITHM CORRELATION
    # ---------------------------------------------------------
    algos = sorted(list(ALGORITHM_FILES.keys()))
    
    for algo in algos:
        output_lines.append("\n" + "=" * 80)
        output_lines.append(f"PER-ALGORITHM CORRELATION: {algo}")
        output_lines.append("=" * 80)
        output_lines.append(f"{'Order':<25} | {'Correlation (τ)':<15}")
        output_lines.append("-" * 80)
        
        all_taus_algo = []
        
        for order in orders:
            # Check if algo has data for this order
            if order not in data_metric[algo] or order not in data_acc:
                continue
                
            # Compute Matrix 1: Δ(metric) for THIS algorithm
            mat_delta_metric = np.full((len(REGIMES), len(REGIMES)), np.nan)
            for i, r1 in enumerate(REGIMES):
                for j, r2 in enumerate(REGIMES):
                    if r1 in data_metric[algo][order] and r2 in data_metric[algo][order]:
                        v1 = data_metric[algo][order][r1]
                        v2 = data_metric[algo][order][r2]
                        if len(v1) > 0 and len(v1) == len(v2):
                            mask = ~np.isnan(v1) & ~np.isnan(v2)
                            if np.any(mask):
                                mat_delta_metric[i, j] = np.mean(v1[mask] - v2[mask])
            
            # Compute Matrix 2: Accuracy differences for THIS algorithm
            mat_delta_acc = np.full((len(REGIMES), len(REGIMES)), np.nan)
            for i, r1 in enumerate(REGIMES):
                for j, r2 in enumerate(REGIMES):
                    if r1 in data_acc[order] and algo in data_acc[order][r1] and \
                       r2 in data_acc[order] and algo in data_acc[order][r2]:
                        mat_delta_acc[i, j] = data_acc[order][r1][algo] - data_acc[order][r2][algo]
            
            # Correlate the two matrices
            vec_metric, vec_acc = get_upper_triangle_pairs(mat_delta_metric, mat_delta_acc)
            
            if len(vec_metric) > 2:
                tau, p_val = kendalltau(vec_metric, vec_acc)
                all_taus_algo.append(tau)
                line = f"{order:<25} | {tau:+.4f}"
                output_lines.append(line)
            else:
                line = f"{order:<25} | N/A"
                output_lines.append(line)
                
        if all_taus_algo:
            summary_line = f"Mean Kendall τ across Orders for {algo}: {np.nanmean(all_taus_algo):+.4f}"
            output_lines.append("-" * 80)
            output_lines.append(summary_line)

    # Print everything to console
    for line in output_lines:
        print(line)

    with open(out_file, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
