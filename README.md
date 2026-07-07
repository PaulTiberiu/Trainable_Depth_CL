# Trainable Depth in Continual Learning

This repository contains the experimental framework for analyzing the interplay between **Continual Learning (CL) algorithms** and the **Adaptation Regime** (trainable depth) of neural networks. Specifically, it explores how freezing different blocks of a ResNet-18 (e.g., full finetune vs. last block only) affects CL performance, forgetting, and gradient representations across various task orderings.

## Overview

The pipeline is split into four primary stages:
1. **Model Training (CL Algorithms):** Run sequential task training using various CL techniques under different depth regimes.
2. **Metric Extraction:** Gather gradient norms ($g^2$), representation norms ($r^2$), and cosine similarities ($\cos$) from the training results.
3. **Correlation Analysis:** Correlate these gradient statistics with performance metrics (like Accuracy and Forgetting) using Kendall's Tau, Spearman, or Pearson coefficients.
4. **Plotting & Visualization:** Generate graphs comparing algorithms, regimes, and their correlations.

---

## Requirements

- `torch`, `torchvision`
- `numpy`, `pandas`, `scipy`
- `matplotlib`

*Note: The scripts expect CIFAR-100 or Mini-ImageNet data to be available locally (controlled via `utils.tools.DATASET` in the codebase).*

---

## 1. Running the CL Algorithms

The core of this repository are the experiment scripts, each implementing a different continual learning algorithm. Running these will sequentially train the network across multiple task orderings and adaptation regimes, saving the results (accuracies, forgetting, and optionally gradients) to `results/` or `resultsgrad/`.

| Script | Algorithm |
|--------|-----------|
| `ewc.py` | Elastic Weight Consolidation (Online EWC) |
| `lwf.py` | Learning without Forgetting (LwF) |
| `gem.py` | Gradient Episodic Memory (GEM) |
| `si.py`  | Synaptic Intelligence (SI) |
| `der.py` | Dynamically Expandable Representation (DER) |

**Usage:**  
Simply execute the desired algorithm. E.g.:  
```bash
python ewc.py
```
Outputs are automatically safely checkpointed to JSON files.

---

## 2. Metric Extraction & Aggregation

After generating the result JSONs, you can extract gradient-related metrics and aggregate data for analysis. The terms `g2`, `r2`, and `2cos` refer to **gradient squared norms**, **representation squared norms**, and **cosine similarities** respectively.

| Script | Description |
|--------|-------------|
| `extract_grad_stats.py` | Parses raw result JSONs and extracts layer-wise or network-wise gradient statistics. |
---

## 3. Statistical & Correlation Analysis

These scripts quantify the relationship between the network's structural updates and its actual capability to learn and remember (Accuracy/Forgetting).

| Script | Description |
|--------|-------------|
| `compute_kendall.py` | Computes Kendall's Tau correlation between different task orderings across adaptation regimes. |
| `correlate_vectors_g2_r2_2cos_direct.py` | Calculates direct statistical correlations between the underlying vectors ($g^2$, $r^2$, $\cos$) and performance. |
| `correlate_g2_r2_2cos_accuracy_direct.py` | Specifically correlates the gradient/representation metrics against the final Accuracy. |
| `correlate_scatter_g2_r2_2cos_forgetting.py` | Correlates gradient metrics against Forgetting and formats data for scatter plots. |

---

## 4. Plotting & Visualization

Once results are processed and correlations computed, these scripts generate visual artifacts (saved to `plots/` or `results/`).

| Script | Description |
|--------|-------------|
| `plot_algo_comparison.py` | Generates bar charts comparing the mean average accuracy across algorithms and regimes (e.g. Full Finetune vs. Last Block). |
| `plot_line_graphs.py` <br> `plot_line_graphs_g2_r2_2cos_direct.py` | Plots line graphs showing how metrics and accuracies evolve across different trainable depth regimes. |
| `plot_scatter_g2_r2_2cos_forgetting.py` | Creates scatter plots visually demonstrating the correlation between representation shifts and forgetting. |

**Usage:**  
```bash
python plot_algo_comparison.py
```
Plots are automatically outputted (usually saving directly without displaying due to `matplotlib.use("Agg")`).