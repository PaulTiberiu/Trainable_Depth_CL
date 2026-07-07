import json
import os
import pandas as pd
from utils.tools import DATASET

def get_algorithm_files():
    """
    Look for both pretrained and random initialization results
    for all available algorithms.
    """
    algorithms = [
        ("Online EWC", "ewc"),
        ("LwF", "lwf"),
        #("Online ER", "er"),
        ("SI", "si"),
        ("GEM", "gem"),
        #("Naive", "cl")
    ]
    
    files = []
    for algo_name, prefix in algorithms:
        for init_mode in ["pretrained", "random"]:
            fpath = f"resultsgrad/{prefix}_results_{init_mode}_{DATASET}.json"
            if os.path.exists(fpath):
                files.append((algo_name, fpath))
    return files

def main():
    print(f"Extracting gradient stats for dataset: {DATASET}")
    
    files = get_algorithm_files()
    if not files:
        print(f"No result files found in results/ for dataset {DATASET}.")
        return
        
    records = []
    
    for algo, fpath in files:
        print(f"Processing {fpath}...")
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"  [WARN] Could not parse JSON in {fpath}")
                continue
                
        for run in data:
            regime = run.get("regime", "unknown")
            order_named = run.get("order_named", "unknown")
            init_mode = run.get("init", "unknown")
            
            grad_stats = run.get("grad_stats", [])
            if not grad_stats:
                continue
                
            # grad_stats is a list of lists (tasks -> epochs)
            for task_idx, task_epochs in enumerate(grad_stats):
                for epoch_idx, epoch_data in enumerate(task_epochs):
                    records.append({
                        "dataset": DATASET,
                        "algorithm": algo,
                        "init_mode": init_mode,
                        "regime": regime,
                        "order": order_named,
                        "task_idx": task_idx,
                        "epoch": epoch_idx + 1,
                        "norm_task": epoch_data.get("norm_task", 0.0),
                        "norm_ret": epoch_data.get("norm_ret", 0.0)
                    })
                    
    if not records:
        print("\nNo gradient stats found in any of the processed files.")
        print("Make sure you have run the updated algorithms that store 'grad_stats'.")
        return
        
    df = pd.DataFrame(records)
    
    # Save the granular data to JSON
    out_file = f"resultsgrad/grad_stats_{DATASET}.json"
    df.to_json(out_file, orient="records", indent=2)
    print(f"\n[OK] Extracted {len(records)} gradient stat records.")
    print(f"[OK] Full data saved to -> {out_file}")
    
    # Print a quick summary of average norms per algorithm & regime
    print("\n=== Average Gradient Norms per Algorithm & Regime ===")
    
    # Group by algorithm and regime, calculate mean
    summary = df.groupby(["algorithm", "regime"])[["norm_task", "norm_ret"]].mean().reset_index()
    
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    
    # Save the summary to JSON
    summary_file = f"resultsgrad/grad_stats_summary_{DATASET}.json"
    summary.to_json(summary_file, orient="records", indent=2)
    print(f"\n[OK] Summary saved to -> {summary_file}")

if __name__ == "__main__":
    main()