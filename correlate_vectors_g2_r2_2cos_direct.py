import json
import os
import numpy as np
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr, pearsonr
from pathlib import Path

from utils.tools import DATASET

def get_algorithm_files():
    algorithms = [
        ("Online EWC", "ewc"),
        # ("LwF", "lwf"),
        # ("SI", "si"),
        # ("GEM", "gem"),
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

REGIMES = [
    "full_finetune",
    "last_6_blocks",
    "last_3_blocks",
    "last_2_blocks",
    "last_block",
    # "bn_affine_only"
]

def get_valid_vectors(*vectors):
    """
    Given multiple vectors of the same length, filter out indices where any vector has NaN.
    """
    arrs = [np.array(v, dtype=float) for v in vectors]
    mask = ~np.isnan(arrs[0])
    for arr in arrs[1:]:
        mask &= ~np.isnan(arr)
    
    return [arr[mask] for arr in arrs]

def main():
    print(f"Computing Vector Correlations: Direct g2_r2_2cos vs (Accuracy, Forgetting) for dataset: {DATASET}")
    
    ALGORITHM_FILES = get_algorithm_files()
    if not ALGORITHM_FILES:
        print("No result files found.")
        return

    # data_acc[order][regime][algo] = mean accuracy
    data_acc = defaultdict(lambda: defaultdict(dict))
    
    # data_forg[order][regime][algo] = mean forgetting
    data_forg = defaultdict(lambda: defaultdict(dict))
    
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

            # 1. Load Accuracy
            accs = r.get("final_accs", {})
            if accs:
                data_acc[order][regime][algo_name] = float(np.mean(list(accs.values())))

            # 2. Load Forgetting
            # It might be in "avg_forgetting" directly or in "forgetting" dict
            forg = r.get("forgetting", {})
            if forg:
                data_forg[order][regime][algo_name] = float(np.mean(list(forg.values())))
            elif "avg_forgetting" in r:
                data_forg[order][regime][algo_name] = float(r["avg_forgetting"])

            # 3. Load metric (g^2 + r^2 + 2*cos*g*r direct from JSON)
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
    
    out_dir = Path("plots/vector_correlations") / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "correlations_vectors_g2_r2_2cos_direct.txt"

    output_lines = []
    
    # Helper to compute correlation
    def compute_correlations(v1, v2):
        v1_clean, v2_clean = get_valid_vectors(v1, v2)
        if len(v1_clean) > 2:
            tau, _ = kendalltau(v1_clean, v2_clean)
            spearman, _ = spearmanr(v1_clean, v2_clean)
            pearson, _ = pearsonr(v1_clean, v2_clean)
            return tau, spearman, pearson
        return np.nan, np.nan, np.nan

    # ---------------------------------------------------------
    # GLOBAL CORRELATION (averaged over algorithms)
    # ---------------------------------------------------------
    output_lines.append("=" * 110)
    output_lines.append("GLOBAL VECTOR CORRELATION (Mean over Algorithms)")
    output_lines.append("=" * 110)
    output_lines.append(f"{'Order':<25} | {'Acc τ':<8} {'Acc ρ_s':<8} {'Acc r':<8} | {'Forg τ':<8} {'Forg ρ_s':<8} {'Forg r':<8}")
    output_lines.append("-" * 110)
    
    global_taus_acc, global_spear_acc, global_pear_acc = [], [], []
    global_taus_forg, global_spear_forg, global_pear_forg = [], [], []

    for order in orders:
        vec_metric_global = []
        vec_acc_global = []
        vec_forg_global = []
        
        for regime in REGIMES:
            # Average metric across algorithms
            m_vals = []
            for algo in data_metric:
                if order in data_metric[algo] and regime in data_metric[algo][order]:
                    v = data_metric[algo][order][regime]
                    if len(v) > 0:
                        m_vals.append(np.mean(v))
            
            vec_metric_global.append(np.mean(m_vals) if m_vals else np.nan)
            
            # Average acc across algorithms
            a_vals = []
            if regime in data_acc[order]:
                a_vals = list(data_acc[order][regime].values())
            vec_acc_global.append(np.mean(a_vals) if a_vals else np.nan)
            
            # Average forg across algorithms
            f_vals = []
            if regime in data_forg[order]:
                f_vals = list(data_forg[order][regime].values())
            vec_forg_global.append(np.mean(f_vals) if f_vals else np.nan)
            
        # Compute correlations
        tau_a, sp_a, pe_a = compute_correlations(vec_metric_global, vec_acc_global)
        tau_f, sp_f, pe_f = compute_correlations(vec_metric_global, vec_forg_global)
        
        if not np.isnan(tau_a):
            global_taus_acc.append(tau_a); global_spear_acc.append(sp_a); global_pear_acc.append(pe_a)
        if not np.isnan(tau_f):
            global_taus_forg.append(tau_f); global_spear_forg.append(sp_f); global_pear_forg.append(pe_f)
            
        s_ta = f"{tau_a:+.4f}" if not np.isnan(tau_a) else "N/A"
        s_sa = f"{sp_a:+.4f}" if not np.isnan(sp_a) else "N/A"
        s_pa = f"{pe_a:+.4f}" if not np.isnan(pe_a) else "N/A"
        
        s_tf = f"{tau_f:+.4f}" if not np.isnan(tau_f) else "N/A"
        s_sf = f"{sp_f:+.4f}" if not np.isnan(sp_f) else "N/A"
        s_pf = f"{pe_f:+.4f}" if not np.isnan(pe_f) else "N/A"
        
        output_lines.append(f"{order:<25} | {s_ta:<8} {s_sa:<8} {s_pa:<8} | {s_tf:<8} {s_sf:<8} {s_pf:<8}")

    if global_taus_acc or global_taus_forg:
        output_lines.append("-" * 110)
        m_ta = np.nanmean(global_taus_acc) if global_taus_acc else np.nan
        m_sa = np.nanmean(global_spear_acc) if global_spear_acc else np.nan
        m_pa = np.nanmean(global_pear_acc) if global_pear_acc else np.nan
        m_tf = np.nanmean(global_taus_forg) if global_taus_forg else np.nan
        m_sf = np.nanmean(global_spear_forg) if global_spear_forg else np.nan
        m_pf = np.nanmean(global_pear_forg) if global_pear_forg else np.nan
        output_lines.append(f"{'Mean across Orders':<25} | {m_ta:+.4f} {m_sa:+.4f} {m_pa:+.4f} | {m_tf:+.4f} {m_sf:+.4f} {m_pf:+.4f}")
    
    # ---------------------------------------------------------
    # PER-ALGORITHM CORRELATION
    # ---------------------------------------------------------
    algos = sorted(list(ALGORITHM_FILES.keys()))
    
    for algo in algos:
        output_lines.append("\n" + "=" * 110)
        output_lines.append(f"PER-ALGORITHM CORRELATION: {algo}")
        output_lines.append("=" * 110)
        output_lines.append(f"{'Order':<25} | {'Acc τ':<8} {'Acc ρ_s':<8} {'Acc r':<8} | {'Forg τ':<8} {'Forg ρ_s':<8} {'Forg r':<8}")
        output_lines.append("-" * 110)
        
        algo_taus_acc, algo_spear_acc, algo_pear_acc = [], [], []
        algo_taus_forg, algo_spear_forg, algo_pear_forg = [], [], []
        
        for order in orders:
            if order not in data_acc or algo not in data_metric:
                continue
                
            vec_metric = []
            vec_acc = []
            vec_forg = []
            
            for regime in REGIMES:
                # Metric
                if regime in data_metric[algo].get(order, {}):
                    v = data_metric[algo][order][regime]
                    vec_metric.append(np.mean(v) if len(v) > 0 else np.nan)
                else:
                    vec_metric.append(np.nan)
                    
                # Acc
                if regime in data_acc[order] and algo in data_acc[order][regime]:
                    vec_acc.append(data_acc[order][regime][algo])
                else:
                    vec_acc.append(np.nan)
                    
                # Forg
                if regime in data_forg[order] and algo in data_forg[order][regime]:
                    vec_forg.append(data_forg[order][regime][algo])
                else:
                    vec_forg.append(np.nan)
                    
            # Compute correlations
            tau_a, sp_a, pe_a = compute_correlations(vec_metric, vec_acc)
            tau_f, sp_f, pe_f = compute_correlations(vec_metric, vec_forg)
            
            if not np.isnan(tau_a):
                algo_taus_acc.append(tau_a); algo_spear_acc.append(sp_a); algo_pear_acc.append(pe_a)
            if not np.isnan(tau_f):
                algo_taus_forg.append(tau_f); algo_spear_forg.append(sp_f); algo_pear_forg.append(pe_f)
                
            s_ta = f"{tau_a:+.4f}" if not np.isnan(tau_a) else "N/A"
            s_sa = f"{sp_a:+.4f}" if not np.isnan(sp_a) else "N/A"
            s_pa = f"{pe_a:+.4f}" if not np.isnan(pe_a) else "N/A"
            
            s_tf = f"{tau_f:+.4f}" if not np.isnan(tau_f) else "N/A"
            s_sf = f"{sp_f:+.4f}" if not np.isnan(sp_f) else "N/A"
            s_pf = f"{pe_f:+.4f}" if not np.isnan(pe_f) else "N/A"
            
            output_lines.append(f"{order:<25} | {s_ta:<8} {s_sa:<8} {s_pa:<8} | {s_tf:<8} {s_sf:<8} {s_pf:<8}")
                
        if algo_taus_acc or algo_taus_forg:
            output_lines.append("-" * 110)
            m_ta = np.nanmean(algo_taus_acc) if algo_taus_acc else np.nan
            m_sa = np.nanmean(algo_spear_acc) if algo_spear_acc else np.nan
            m_pa = np.nanmean(algo_pear_acc) if algo_pear_acc else np.nan
            m_tf = np.nanmean(algo_taus_forg) if algo_taus_forg else np.nan
            m_sf = np.nanmean(algo_spear_forg) if algo_spear_forg else np.nan
            m_pf = np.nanmean(algo_pear_forg) if algo_pear_forg else np.nan
            output_lines.append(f"{'Mean across Orders':<25} | {m_ta:+.4f} {m_sa:+.4f} {m_pa:+.4f} | {m_tf:+.4f} {m_sf:+.4f} {m_pf:+.4f}")

    # Print everything to console
    for line in output_lines:
        print(line)

    with open(out_file, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
