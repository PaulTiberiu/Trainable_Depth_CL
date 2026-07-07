"""
continual_learning_ewc
==========================
Elastic Weight Consolidation (EWC) variant of the task-order study.

Algorithm
---------
Faithfully implements the EWC algorithm as described in:
  Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in
  neural networks", PNAS.

and the Online EWC variant from:
  Schwarz et al. (2018) "Progress & Compress", ICML.

Key algorithmic details:
  1. After completing training on each context/task, the diagonal Fisher
     Information Matrix (FIM) is estimated over that task's training data
     using the model's current predictions (expected FI, not empirical FI).

  2. EWC penalty added to the loss during subsequent task training:
       L_ewc = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
     where theta*_i are the MAP parameters after the previous task and
     F_i is the estimated diagonal FIM entry for parameter i.

  3. Two modes are supported (set EWC_OFFLINE at top of file):
       - Online EWC  (EWC_OFFLINE=False): a single running FIM is
         maintained with exponential decay (gamma). This avoids the
         memory cost of storing one FIM per past task.
       - Offline EWC (EWC_OFFLINE=True): a separate penalty term is
         kept for every past task (original Kirkpatrick et al. paper).

  4. Fisher estimation uses the 'all' strategy (weighted sum over all
     labels weighted by predicted probabilities) — see continual_learner.py.
     FISHER_N controls how many samples are used (None = full dataset).

  5. Model weights are NOT reset between tasks. The EWC penalty
     discourages deviation from previously learned parameters.

Differences from the vanilla (non-EWC) script
----------------------------------------------
  • `train_one_task_ewc` replaces `train_one_task`.
  • After each task, `estimate_fisher` is called and stored in the model.
  • The EWC penalty is added to the CE loss during training of all
    subsequent tasks.
  • Everything else (regimes, evaluate_task, pairwise_metrics,
    acc/loss matrices, JSON schema) is IDENTICAL to continual_learning_er.py.

Output
------
  ewc_results_random.json     (random init)
  ewc_results_pretrained.json (ImageNet pretrained init)
  — same JSON schema as er_results_*.json, with extra top-level keys:
      "algorithm":     "EWC" or "OnlineEWC"
      "ewc_lambda":    <float>
      "ewc_offline":   <bool>
      "ewc_gamma":     <float>  (only relevant for Online EWC)
      "fisher_n":      <int|null>

Usage
-----
  python continual_learning_ewc.py
"""

import copy
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100, MNIST, FashionMNIST, QMNIST, ImageFolder
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.tools import (
    LR_MIN, ES_PATIENCE, ES_MIN_DELTA, EarlyStopping,
    NUM_TASKS, CLASSES_PER_TASK, NUM_CLASSES, DATA_ROOT, EPOCHS_PER_TASK,
    BATCH_SIZE, LR, NUM_WORKERS, RANDOM_SEED, NUM_ORDERS, TASK_LABELS,
    DEVICE, PIN_MEMORY, TASK_ORDERS, train_transform, test_transform,
    mnist_train_transform, mnist_test_transform,
    imagenet_train_transform, imagenet_test_transform, ImageNet100Folder, HFKMNIST,  # ImageNet transforms
    DATASET,
    sample_task_orders, split_into_tasks, build_model, set_trainable_params,
    evaluate_task, pairwise_metrics,
    extract_grad_vectors, get_grad_stats
)

# ── ImageNet data root ────────────────────────────────────────────────────────
# Path to the directory that contains train/ and val_dirs/.
# Run from the continual_learning/ folder; /mnt is two levels up.
IMAGENET_ROOT = "../../mnt/imagenet/data/ILSVRC2012"

# ── EWC-specific config ───────────────────────────────────────────────────────

# Regularization strength — scales the EWC penalty relative to the CE loss.
# Tune this: too small → forgetting; too large → plasticity loss.
EWC_LAMBDA  = 5000.0

# Offline EWC: True  → separate penalty term per past task (Kirkpatrick 2017)
#              False → single running FIM with gamma decay (Online EWC)
EWC_OFFLINE = False

# Decay factor for Online EWC. 1.0 = no decay (full accumulation).
EWC_GAMMA   = 1.0

# Number of samples used to estimate the FIM after each task.
# None = use the full training set for that task (slower but more accurate).
FISHER_N    = 200

# Batch size used during Fisher estimation (1 gives cleanest gradients).
FISHER_BATCH = 1

EWC_RESULTS_RANDOM     = f"results/ewc_results_random_{DATASET}.json"
EWC_RESULTS_PRETRAINED = f"results/ewc_results_pretrained_{DATASET}.json"

# Init mode: "random" or "pretrained"
INIT_MODE = "random"
USE_EARLY_STOPPING = True

# REGIMES = ["full_finetune", "last_block", "head_only", "bn_affine_only"]
REGIMES = ["full_finetune", "last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks"]

# ── Data ──────────────────────────────────────────────────────────────────────

# Non-augmented transform for Fisher estimation (clean gradients).
cifar_fisher_transform = test_transform
mnist_fisher_transform = mnist_test_transform

def load_datasets():
    if DATASET == "cifar100":
        train_ds = CIFAR100(root=DATA_ROOT, train=True,  download=False,
                            transform=train_transform)
        fisher_ds = CIFAR100(root=DATA_ROOT, train=True, download=False,
                             transform=cifar_fisher_transform)
        test_ds  = CIFAR100(root=DATA_ROOT, train=False, download=False,
                            transform=test_transform)
    elif DATASET == "mnist":
        train_ds = MNIST(root=DATA_ROOT, train=True,  download=True,
                         transform=mnist_train_transform)
        fisher_ds = MNIST(root=DATA_ROOT, train=True, download=True,
                          transform=mnist_fisher_transform)
        test_ds  = MNIST(root=DATA_ROOT, train=False, download=True,
                         transform=mnist_test_transform)
    elif DATASET == "fashion_mnist":
        train_ds = FashionMNIST(root=DATA_ROOT, train=True,  download=True,
                         transform=mnist_train_transform)
        fisher_ds = FashionMNIST(root=DATA_ROOT, train=True, download=True,
                          transform=mnist_fisher_transform)
        test_ds  = FashionMNIST(root=DATA_ROOT, train=False, download=True,
                         transform=mnist_test_transform)
    elif DATASET == "kmnist":
        train_ds = HFKMNIST(root=DATA_ROOT, train=True,  download=True,
                         transform=mnist_train_transform)
        fisher_ds = HFKMNIST(root=DATA_ROOT, train=True, download=True,
                          transform=mnist_fisher_transform)
        test_ds  = HFKMNIST(root=DATA_ROOT, train=False, download=True,
                         transform=mnist_test_transform)
    elif DATASET == "qmnist":
        train_ds = QMNIST(root=DATA_ROOT, what="train", compat=True, download=True,
                         transform=mnist_train_transform)
        fisher_ds = QMNIST(root=DATA_ROOT, what="train", compat=True, download=True,
                          transform=mnist_fisher_transform)
        test_ds  = QMNIST(root=DATA_ROOT, what="test", compat=True, download=True,
                         transform=mnist_test_transform)
    elif DATASET in ["imagenet", "imagenet100"]:
        FolderClass = ImageNet100Folder if DATASET == "imagenet100" else ImageFolder
        train_ds = FolderClass(root=f"{IMAGENET_ROOT}/train",
                               transform=imagenet_train_transform)
        # EWC usually needs a subset of train data to compute Fisher. We'll use the same FolderClass but without random augs.
        fisher_ds = FolderClass(root=f"{IMAGENET_ROOT}/train",
                                transform=imagenet_test_transform)
        test_ds  = FolderClass(root=f"{IMAGENET_ROOT}/val_dirs",
                               transform=imagenet_test_transform)
        # Sanity-check
        expected_classes = 100 if DATASET == "imagenet100" else 1000
        assert len(train_ds.classes) == expected_classes, (
            f"Expected {expected_classes} ImageNet classes, found {len(train_ds.classes)}. "
            f"Check IMAGENET_ROOT = {IMAGENET_ROOT!r}"
        )

    else:
        raise ValueError(f"Unknown DATASET: {DATASET!r}")
        
    return train_ds, fisher_ds, test_ds

# ══════════════════════════════════════════════════════════════════════════════
# EWC State — stores Fisher matrices and MAP parameter estimates
# ══════════════════════════════════════════════════════════════════════════════

class EWCState:
    """
    Holds the EWC regularization state across tasks.

    Supports both:
      - Offline EWC: stores a separate (fisher, mean) pair per past task.
      - Online EWC:  maintains a single running FIM (with gamma decay).

    Only parameters that require_grad at estimation time are tracked.
    """

    def __init__(self, offline: bool = False, gamma: float = 1.0):
        self.offline  = offline
        self.gamma    = gamma
        # Online EWC: single running FIM and mean
        self.fisher : dict[str, torch.Tensor] | None = None   # {name: diag FIM}
        self.mean   : dict[str, torch.Tensor] | None = None   # {name: theta*}
        # Offline EWC: list of (fisher, mean) dicts, one per past task
        self.history: list[tuple[dict, dict]] = []
        self.context_count = 0

    # ── Fisher estimation ─────────────────────────────────────────────────────

    def estimate_fisher(self, model: nn.Module, dataset: Subset,
                        allowed_classes: list | None,
                        fisher_n: int | None,
                        fisher_batch: int = 1) -> dict[str, torch.Tensor]:
        """
        Estimate the diagonal FIM using the 'all-labels' (expected FI) strategy,
        matching continual_learner.py's fisher_labels='all' branch.

        For each sample:
          For each class c:
            grad_c  = gradient of CE(output, c) w.r.t. parameters
            FIM    += softmax(output)[c] * grad_c^2

        Returns a dict {param_name: diag_FIM_tensor} (on CPU).
        """
        est_fisher_info: dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                est_fisher_info[n] = p.detach().clone().zero_().cpu()

        mode = model.training
        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=fisher_batch,
            shuffle=True,
            num_workers=0,       # 0 workers inside estimation to avoid CUDA fork issues
            pin_memory=False,
        )

        n_samples_processed = 0
        for index, (x, _) in enumerate(loader):
            if fisher_n is not None and index >= fisher_n:
                break

            x = x.to(DEVICE)
            # Forward pass WITHOUT no_grad so output retains grad_fn for backward
            output_full = model(x)
            output = output_full if allowed_classes is None else output_full[:, allowed_classes]

            # Compute label weights without tracking gradients
            with torch.no_grad():
                label_weights = F.softmax(output, dim=1)   # (B, C)

            for label_index in range(output.shape[1]):
                label = torch.full((x.shape[0],), label_index,
                                   dtype=torch.long, device=DEVICE)
                negloglikelihood = F.cross_entropy(output, label)
                model.zero_grad()
                negloglikelihood.backward(
                    retain_graph=(label_index + 1 < output.shape[1])
                )
                w = label_weights[0][label_index].item()   # scalar weight
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        est_fisher_info[n] += w * (p.grad.detach().cpu() ** 2)

            n_samples_processed += 1

        # Normalize by number of batches processed
        if n_samples_processed > 0:
            est_fisher_info = {n: f / n_samples_processed
                               for n, f in est_fisher_info.items()}

        model.train(mode=mode)
        return est_fisher_info

    # ── Update state after a task ──────────────────────────────────────────────

    def update(self, model: nn.Module, dataset: Subset,
               allowed_classes: list | None,
               fisher_n: int | None,
               fisher_batch: int = 1):
        """
        Call after finishing training on a task.
        Estimates FIM and stores MAP parameters.
        """
        print("      [EWC] Estimating Fisher Information Matrix...", flush=True)
        new_fisher = self.estimate_fisher(model, dataset, allowed_classes,
                                          fisher_n, fisher_batch)
        # MAP estimates = current parameters
        new_mean = {n: p.detach().clone().cpu()
                    for n, p in model.named_parameters()
                    if p.requires_grad}

        if self.offline:
            # Store separate (fisher, mean) for this task
            self.history.append((new_fisher, new_mean))
        else:
            # Online EWC: accumulate into running FIM with gamma decay
            if self.fisher is None:
                self.fisher = new_fisher
            else:
                for n in self.fisher:
                    self.fisher[n] = self.gamma * self.fisher[n] + new_fisher[n]
            self.mean = new_mean

        self.context_count += 1
        print(f"      [EWC] Fisher estimated over {min(fisher_n or 999999, len(dataset))} "
              f"samples. Context count: {self.context_count}", flush=True)

    # ── EWC penalty ───────────────────────────────────────────────────────────

    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC regularization loss.

        Offline:  L = (1/2) * sum_t sum_i F^t_i * (theta_i - theta*^t_i)^2
        Online:   L = (1/2) * sum_i (gamma * F_i) * (theta_i - theta*_i)^2

        Returns a scalar tensor on DEVICE.
        """
        if self.context_count == 0:
            return torch.tensor(0., device=DEVICE)

        losses = []

        if self.offline:
            for fisher, mean in self.history:
                for n, p in model.named_parameters():
                    if p.requires_grad and n in fisher:
                        f    = fisher[n].to(DEVICE)
                        m    = mean[n].to(DEVICE)
                        losses.append((f * (p - m) ** 2).sum())
        else:
            # Online EWC: apply gamma scaling to running FIM
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.fisher:
                    f = (self.gamma * self.fisher[n]).to(DEVICE)
                    m = self.mean[n].to(DEVICE)
                    losses.append((f * (p - m) ** 2).sum())

        return 0.5 * sum(losses) if losses else torch.tensor(0., device=DEVICE)

# ── Evaluation (identical to continual_learning_er.py) ────────────────────────

# ── Harm / Transfer (identical to continual_learning_er.py) ──────────────────

# ══════════════════════════════════════════════════════════════════════════════
# EWC Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_task_ewc(model:      nn.Module,
                       train_subset: Subset,
                       ewc_state:  EWCState,
                       regime:     str,
                       task_idx:   int) -> list:
    """
    One full task training with EWC regularization.

    Per-batch update:
      1. Forward pass over current task batch.
      2. CE loss on current task's logit slice.
      3. EWC penalty (zero on the first task, since no prior FIM exists yet).
      4. Total loss = CE + ewc_lambda * EWC_penalty.
      5. Single backward + optimizer step.

    Returns epoch_losses (list of mean total loss per epoch).
    """
    trainable = set_trainable_params(model, regime)
    if not trainable:
        print("    [warn] no trainable params, skipping.")
        return []

    task_start = task_idx * CLASSES_PER_TASK
    task_end   = task_start + CLASSES_PER_TASK

    loader    = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    optimizer = optim.Adam(trainable, lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TASK, eta_min=LR_MIN)
    early_stop = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)

    model.train()
    epoch_losses = []
    epoch_stats = []

    for epoch in range(EPOCHS_PER_TASK):
        total_loss    = 0.0
        total_ce      = 0.0
        total_penalty = 0.0
        n_batches     = 0
        batch_stats   = []

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # ── CE loss on current task ───────────────────────────────────────
            logits      = model(images)[:, task_start:task_end]
            local_labels = labels - task_start
            ce_loss     = criterion(logits, local_labels)

            optimizer.zero_grad()
            ce_loss.backward(retain_graph=True)
            g_task = extract_grad_vectors(model)

            # ── EWC penalty (0 on first task) ─────────────────────────────────
            penalty = ewc_state.ewc_loss(model)
            ret_loss = EWC_LAMBDA * penalty
            
            if ret_loss.requires_grad and getattr(ret_loss, "grad_fn", None) is not None:
                ret_loss.backward()
                g_ret = extract_grad_vectors(model)
            else:
                g_ret = torch.tensor([], device=DEVICE)
                
            optimizer.step()

            batch_stats.append(get_grad_stats(g_task, g_ret))

            loss = ce_loss + ret_loss
            total_loss    += loss.item()
            total_ce      += ce_loss.item()
            total_penalty += penalty.item()
            n_batches     += 1

        mean_loss    = total_loss    / n_batches if n_batches > 0 else 0.0
        mean_ce      = total_ce      / n_batches if n_batches > 0 else 0.0
        mean_penalty = total_penalty / n_batches if n_batches > 0 else 0.0
        epoch_losses.append(round(mean_loss, 6))

        mean_norm_task = sum(s["norm_task"] for s in batch_stats) / n_batches if n_batches > 0 else 0.0
        mean_norm_ret  = sum(s["norm_ret"] for s in batch_stats) / n_batches if n_batches > 0 else 0.0
        mean_cos_sim   = sum(s["cos_sim"] for s in batch_stats) / n_batches if n_batches > 0 else 0.0
        mean_g2_r2_2cos = sum(s["g2_r2_2cos"] for s in batch_stats) / n_batches if n_batches > 0 else 0.0
        epoch_stats.append({
            "norm_task": round(mean_norm_task, 6),
            "norm_ret": round(mean_norm_ret, 6),
            "cos_sim": round(mean_cos_sim, 6),
            "g2_r2_2cos": round(mean_g2_r2_2cos, 6)
        })

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"      epoch {epoch+1:>3}/{EPOCHS_PER_TASK}  "
              f"loss={mean_loss:.4f}  "
              f"ce={mean_ce:.4f}  "
              f"ewc_penalty={mean_penalty:.6f}  "
              f"lr={current_lr:.2e}")

        if USE_EARLY_STOPPING and early_stop(mean_loss, epoch=epoch + 1):
            print(f"      [early stop] no improvement for {ES_PATIENCE} epochs "
                  f"(best={early_stop.best_loss:.4f}), stopping at epoch {epoch+1}.")
            break

    return epoch_losses, epoch_stats

# ── Single experiment run ─────────────────────────────────────────────────────

def run_experiment_ewc(regime:            str,
                       order_entry:       dict,
                       train_tasks:       list,
                       fisher_tasks:      list,
                       test_tasks:        list,
                       pretrained:        bool = False) -> dict:
    label       = order_entry["label"]
    task_order  = order_entry["order"]
    order_named = order_entry["order_named"]

    algo_name = "EWC" if EWC_OFFLINE else "OnlineEWC"
    print(f"\n  Regime   : {regime}")
    print(f"  Order    : {order_named}  ({label})")
    print(f"  Algorithm: {algo_name}  lambda={EWC_LAMBDA}  gamma={EWC_GAMMA}  "
          f"offline={EWC_OFFLINE}  fisher_n={FISHER_N}")
    print(f"  Early stopping   : {'ON' if USE_EARLY_STOPPING else 'OFF'}"
          + (f"  (patience={ES_PATIENCE}, min_delta={ES_MIN_DELTA})"
             if USE_EARLY_STOPPING else ""))

    model     = build_model(pretrained=pretrained)
    ewc_state = EWCState(offline=EWC_OFFLINE, gamma=EWC_GAMMA)

    acc_matrix   = []
    loss_matrix  = []
    epoch_losses = []
    grad_stats   = []
    pairwise     = []

    for step, task_idx in enumerate(task_order):
        task_label = TASK_LABELS[task_idx]
        print(f"\n    Step {step+1:2d}/{NUM_TASKS} — Task {task_label} "
              f"(classes {task_idx*CLASSES_PER_TASK}–{task_idx*CLASSES_PER_TASK+CLASSES_PER_TASK-1})")

        snapshot    = copy.deepcopy(model)
        task_losses, task_stats = train_one_task_ewc(
            model,
            train_tasks[task_idx],
            ewc_state,
            regime,
            task_idx,
        )
        epoch_losses.append(task_losses)
        grad_stats.append(task_stats)

        # ── Estimate Fisher after this task ───────────────────────────────────
        # allowed_classes = the 10 output units for this task
        allowed_classes = list(range(task_idx * CLASSES_PER_TASK,
                                     task_idx * CLASSES_PER_TASK + CLASSES_PER_TASK))
        ewc_state.update(
            model,
            fisher_tasks[task_idx],    # non-augmented data for clean gradients
            allowed_classes=allowed_classes,
            fisher_n=FISHER_N,
            fisher_batch=FISHER_BATCH,
        )

        # ── Evaluate all tasks seen so far ────────────────────────────────────
        acc_row  = {}
        loss_row = {}
        for j in task_order[:step + 1]:
            acc, loss_val = evaluate_task(model, test_tasks[j], j)
            acc_row[j]  = acc
            loss_row[j] = loss_val
        acc_matrix.append(acc_row)
        loss_matrix.append(loss_row)

        accs_str = "  ".join(f"{TASK_LABELS[j]}={acc_row[j]:.3f}"
                             for j in task_order[:step + 1])
        print(f"      [{accs_str}]")

        if step > 0:
            prev = task_order[step - 1]
            pw   = pairwise_metrics(snapshot, model, test_tasks, task_idx, prev)
            pairwise.append(pw)
            print(f"      Harm({task_label}->{TASK_LABELS[prev]})="
                  f"{pw['harm']:+.4f}  "
                  f"Transfer({task_label}->{TASK_LABELS[prev]})="
                  f"{pw['transfer']:+.4f}")

    # ── Final metrics ─────────────────────────────────────────────────────────
    final_accs = {}
    for task_idx in task_order:
        acc, _ = evaluate_task(model, test_tasks[task_idx], task_idx)
        final_accs[task_idx] = acc

    avg_acc = round(sum(final_accs.values()) / len(final_accs), 6)

    forgetting = {}
    for step, task_idx in enumerate(task_order[:-1]):
        peak = acc_matrix[step].get(task_idx)
        if peak is not None:
            forgetting[task_idx] = round(peak - final_accs[task_idx], 6)

    avg_forgetting = (round(sum(forgetting.values()) / len(forgetting), 6)
                      if forgetting else 0.0)

    print(f"\n    avg_acc={avg_acc:.4f}  avg_forgetting={avg_forgetting:.4f}")

    return {
        "algorithm":      algo_name,
        "dataset":        DATASET,
        "ewc_lambda":     EWC_LAMBDA,
        "ewc_offline":    EWC_OFFLINE,
        "ewc_gamma":      EWC_GAMMA,
        "fisher_n":       FISHER_N,
        "init":           "pretrained" if pretrained else "random",
        "regime":         regime,
        "order_label":    label,
        "order_named":    order_named,
        "task_order":     task_order,
        "acc_matrix":     acc_matrix,
        "loss_matrix":    loss_matrix,
        "epoch_losses":   epoch_losses,
        "grad_stats":     grad_stats,
        "pairwise":       pairwise,
        "final_accs":     {TASK_LABELS[k]: v for k, v in final_accs.items()},
        "avg_acc":        avg_acc,
        "forgetting":     {TASK_LABELS[k]: v for k, v in forgetting.items()},
        "avg_forgetting": avg_forgetting,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    algo_name = "EWC" if EWC_OFFLINE else "Online EWC"
    print("=" * 70)
    print(f"Continual Learning with {algo_name} — Task Order Study  [{DATASET.upper()}]")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Dataset       : {DATASET}  ({NUM_TASKS} tasks × {CLASSES_PER_TASK} classes)")
    print(f"Regimes       : {REGIMES}")
    print(f"EWC lambda    : {EWC_LAMBDA}")
    print(f"EWC offline   : {EWC_OFFLINE}")
    print(f"EWC gamma     : {EWC_GAMMA}")
    print(f"Fisher N      : {FISHER_N}")
    print(f"Fisher batch  : {FISHER_BATCH}")
    print(f"Orders        : 1 canonical + {NUM_ORDERS} random = {len(TASK_ORDERS)} total")
    print(f"Epochs/task   : {EPOCHS_PER_TASK}")
    print(f"LR               : {LR}  →  {LR_MIN} (cosine)")
    print(f"Early stopping   : {'ON' if USE_EARLY_STOPPING else 'OFF'}"
          + (f"  (patience={ES_PATIENCE}, min_delta={ES_MIN_DELTA})"
             if USE_EARLY_STOPPING else ""))
    print(f"Total runs    : {len(REGIMES) * len(TASK_ORDERS)} "
          f"({len(REGIMES)} regimes × {len(TASK_ORDERS)} orders)")

    if INIT_MODE not in ("random", "pretrained"):
        raise ValueError(f"INIT_MODE must be 'random' or 'pretrained', got {INIT_MODE!r}")
    is_pretrained = (INIT_MODE == "pretrained")
    init_label    = INIT_MODE
    results_file  = EWC_RESULTS_PRETRAINED if is_pretrained else EWC_RESULTS_RANDOM

    print(f"Init mode     : {init_label}")
    print(f"Output        : {results_file}")

    if DATASET in ["imagenet", "imagenet100"]:
        print(f"ImageNet root : {IMAGENET_ROOT}")

    train_ds, fisher_ds, test_ds = load_datasets()
    train_tasks  = split_into_tasks(train_ds)    # augmented — used for training
    fisher_tasks = split_into_tasks(fisher_ds)   # non-augmented — used for Fisher estimation
    test_tasks   = split_into_tasks(test_ds)

    total_runs  = len(REGIMES) * len(TASK_ORDERS)
    all_results = []
    run_idx     = 0
    t0          = time.time()

    for regime in REGIMES:
        for order_entry in TASK_ORDERS:
            run_idx += 1
            elapsed = (time.time() - t0) / 60
            eta     = (elapsed / run_idx) * (total_runs - run_idx) if run_idx > 1 else 0
            print(f"\n{'='*70}")
            print(f"[{algo_name} / {init_label} / {DATASET}] [Run {run_idx:3d}/{total_runs}]  "
                  f"elapsed={elapsed:.1f}min  ETA~{eta:.0f}min")
            print("=" * 70)

            result = run_experiment_ewc(
                regime, order_entry,
                train_tasks, fisher_tasks, test_tasks,
                pretrained=is_pretrained,
            )
            all_results.append(result)

            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = (time.time() - t0) / 60
    print(f"\nAll done in {total_time:.1f} min.  Results -> {results_file}")
    print("\n" + "=" * 72)
    print(f"{'Regime':<18} {'Order':<12} {'Named Order':<26} "
          f"{'Avg Acc':>8} {'Forget':>8}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['regime']:<18} {r['order_label']:<12} "
              f"{r['order_named']:<26} "
              f"{r['avg_acc']:>8.4f} {r['avg_forgetting']:>8.4f}")

    print(f"\n-- Best and worst ordering per regime (by avg accuracy) --")
    for regime in REGIMES:
        subset = [r for r in all_results if r["regime"] == regime]
        if not subset:
            continue
        best  = max(subset, key=lambda r: r["avg_acc"])
        worst = min(subset, key=lambda r: r["avg_acc"])
        print(f"  {regime:<18}  "
              f"best  = {best['order_named']}  ({best['avg_acc']:.4f})  |  "
              f"worst = {worst['order_named']}  ({worst['avg_acc']:.4f})")


if __name__ == "__main__":
    main()