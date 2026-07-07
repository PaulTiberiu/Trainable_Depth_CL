"""
continual_learning_der
==========================
Dynamically Expandable Representation (DER) variant of the task-order study.

Algorithm
---------
Implements the DER architecture as described in:
  Yan et al. (2021) "DER: Dynamically Expandable Representation for Class Incremental Learning", CVPR.

Key algorithmic details:
  1. For each new task, a new feature extractor is instantiated and added to the model.
  2. The new feature extractor is trained on the new task, while all previous feature extractors are frozen.
  3. The features from all extractors are concatenated and passed to a unified classifier.
  4. An auxiliary classifier is used to train the new feature extractor specifically on the new task.
  5. The loss is a combination of the cross-entropy loss on the unified classifier and the cross-entropy loss on the auxiliary classifier.

Differences from the vanilla (non-DER) script
----------------------------------------------
  • `DERModel` dynamically expands the backbone for each new task.
  • `train_one_task_der` replaces `train_one_task`.
  • `set_trainable_params_der` handles freezing previous extractors and applying regimes to the new extractor.
  • Everything else (regimes, evaluate_task, pairwise_metrics,
    acc/loss matrices, JSON schema) is IDENTICAL to continual_learning_er.py.

Output
------
  der_results_random.json     (random init)
  der_results_pretrained.json (ImageNet pretrained init)
  — same JSON schema as er_results_*.json, with extra top-level keys:
      "algorithm":     "DER"
      "der_alpha":     <float>

Usage
-----
  python der.py
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
    sample_task_orders, split_into_tasks, build_model,
    evaluate_task, pairwise_metrics,
    extract_grad_vectors, get_grad_stats
)

# ── ImageNet data root ────────────────────────────────────────────────────────
# Path to the directory that contains train/ and val_dirs/.
# Run from the continual_learning/ folder; /mnt is two levels up.
IMAGENET_ROOT = "../../mnt/imagenet/data/ILSVRC2012"

# ── DER-specific config ───────────────────────────────────────────────────────

# Regularization strength — scales the auxiliary loss relative to the CE loss.
DER_ALPHA  = 1.0

DER_RESULTS_RANDOM     = f"results/der_results_random_{DATASET}.json"
DER_RESULTS_PRETRAINED = f"results/der_results_pretrained_{DATASET}.json"

# Init mode: "random" or "pretrained"
INIT_MODE = "random"
USE_EARLY_STOPPING = True

# REGIMES = ["full_finetune", "last_block", "head_only", "bn_affine_only"]
REGIMES = ["full_finetune", "last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks"]

# ── Data ──────────────────────────────────────────────────────────────────────

def load_datasets():
    if DATASET == "cifar100":
        train_ds = CIFAR100(root=DATA_ROOT, train=True,  download=False,
                            transform=train_transform)
        test_ds  = CIFAR100(root=DATA_ROOT, train=False, download=False,
                            transform=test_transform)
    elif DATASET == "mnist":
        train_ds = MNIST(root=DATA_ROOT, train=True,  download=True,
                         transform=mnist_train_transform)
        test_ds  = MNIST(root=DATA_ROOT, train=False, download=True,
                         transform=mnist_test_transform)
    elif DATASET == "fashion_mnist":
        train_ds = FashionMNIST(root=DATA_ROOT, train=True,  download=True,
                         transform=mnist_train_transform)
        test_ds  = FashionMNIST(root=DATA_ROOT, train=False, download=True,
                         transform=mnist_test_transform)
    elif DATASET == "kmnist":
        train_ds = HFKMNIST(root=DATA_ROOT, train=True,  download=True,
                         transform=mnist_train_transform)
        test_ds  = HFKMNIST(root=DATA_ROOT, train=False, download=True,
                         transform=mnist_test_transform)
    elif DATASET == "qmnist":
        train_ds = QMNIST(root=DATA_ROOT, what="train", compat=True, download=True,
                         transform=mnist_train_transform)
        test_ds  = QMNIST(root=DATA_ROOT, what="test", compat=True, download=True,
                         transform=mnist_test_transform)
    elif DATASET in ["imagenet", "imagenet100"]:
        FolderClass = ImageNet100Folder if DATASET == "imagenet100" else ImageFolder
        train_ds = FolderClass(root=f"{IMAGENET_ROOT}/train",
                               transform=imagenet_train_transform)
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
        
    return train_ds, test_ds

# ══════════════════════════════════════════════════════════════════════════════
# DER Model
# ══════════════════════════════════════════════════════════════════════════════

class DERModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        self.extractors = nn.ModuleList()
        self.fc = None
        self.aux_fc = None
        self.task_count = 0
        
    def expand(self):
        # Create a new feature extractor
        new_ext = build_model(pretrained=self.pretrained)
        # Remove its fc layer
        new_ext.fc = nn.Identity()
        
        # Freeze previous extractors
        for ext in self.extractors:
            for p in ext.parameters():
                p.requires_grad_(False)
                
        self.extractors.append(new_ext)
        self.task_count += 1
        
        feature_dim = 512 * self.task_count
        new_fc = nn.Linear(feature_dim, NUM_CLASSES).to(DEVICE)
        
        if self.fc is not None:
            with torch.no_grad():
                new_fc.weight[:, :512 * (self.task_count - 1)] = self.fc.weight
                new_fc.bias[:] = self.fc.bias
                new_fc.weight[:, 512 * (self.task_count - 1):] = 0.0
                
        self.fc = new_fc
        self.aux_fc = nn.Linear(512, CLASSES_PER_TASK).to(DEVICE)
        
    def forward(self, x, return_aux=False):
        features = [ext(x) for ext in self.extractors]
        out = torch.cat(features, dim=1)
        logits = self.fc(out)
        
        if return_aux:
            aux_logits = self.aux_fc(features[-1])
            return logits, aux_logits
        return logits

def set_trainable_params_der(model: DERModel, regime: str) -> list:
    """Freeze/unfreeze parameters per regime for the NEW feature extractor."""
    for p in model.parameters():
        p.requires_grad_(False)
        
    active_ext = model.extractors[-1]
    
    if regime == "full_finetune":
        for p in active_ext.parameters():
            p.requires_grad_(True)
    elif regime == "last_block":
        for p in active_ext.layer4.parameters():
            p.requires_grad_(True)
    elif regime == "last_2_blocks":
        for p in active_ext.layer3.parameters():
            p.requires_grad_(True)
        for p in active_ext.layer4.parameters():
            p.requires_grad_(True)
    elif regime == "last_3_blocks":
        for p in active_ext.layer2.parameters():
            p.requires_grad_(True)
        for p in active_ext.layer3.parameters():
            p.requires_grad_(True)
        for p in active_ext.layer4.parameters():
            p.requires_grad_(True)
    elif regime == "last_6_blocks":
        for p in active_ext.layer1.parameters():
            p.requires_grad_(True)
        for p in active_ext.layer2.parameters():
            p.requires_grad_(True)
        for p in active_ext.layer3.parameters():
            p.requires_grad_(True)
        for p in active_ext.layer4.parameters():
            p.requires_grad_(True)
    elif regime == "head_only":
        pass
    elif regime == "bn_affine_only":
        for m in active_ext.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None: m.weight.requires_grad_(True)
                if m.bias   is not None: m.bias.requires_grad_(True)
    else:
        raise ValueError(f"Unknown regime: {regime}")
        
    # Always train the FC layers
    for p in model.fc.parameters():
        p.requires_grad_(True)
    for p in model.aux_fc.parameters():
        p.requires_grad_(True)
        
    return [p for p in model.parameters() if p.requires_grad]


# ── Evaluation (identical to continual_learning_er.py) ────────────────────────

# ── Harm / Transfer (identical to continual_learning_er.py) ──────────────────

# ══════════════════════════════════════════════════════════════════════════════
# DER Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_task_der(model:      DERModel,
                       train_subset: Subset,
                       regime:     str,
                       task_idx:   int) -> list:
    """
    One full task training with DER.
    """
    trainable = set_trainable_params_der(model, regime)
    if not trainable:
        print("    [warn] no trainable params, skipping.")
        return [], []

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
        total_aux     = 0.0
        n_batches     = 0
        batch_stats   = []

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # ── CE loss on current task ───────────────────────────────────────
            logits, aux_logits = model(images, return_aux=True)
            
            main_logits = logits[:, task_start:task_end]
            local_labels = labels - task_start
            
            ce_loss = criterion(main_logits, local_labels)

            optimizer.zero_grad()
            for p in trainable:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            ce_loss.backward(retain_graph=True)
            g_task = extract_grad_vectors(model)

            # ── Aux penalty ─────────────────────────────────
            aux_loss = criterion(aux_logits, local_labels)
            ret_loss = DER_ALPHA * aux_loss
            
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
            total_aux     += aux_loss.item()
            n_batches     += 1

        mean_loss    = total_loss    / n_batches if n_batches > 0 else 0.0
        mean_ce      = total_ce      / n_batches if n_batches > 0 else 0.0
        mean_aux     = total_aux     / n_batches if n_batches > 0 else 0.0
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
              f"aux={mean_aux:.6f}  "
              f"lr={current_lr:.2e}")

        if USE_EARLY_STOPPING and early_stop(mean_loss, epoch=epoch + 1):
            print(f"      [early stop] no improvement for {ES_PATIENCE} epochs "
                  f"(best={early_stop.best_loss:.4f}), stopping at epoch {epoch+1}.")
            break

    return epoch_losses, epoch_stats

# ── Single experiment run ─────────────────────────────────────────────────────

def run_experiment_der(regime:            str,
                       order_entry:       dict,
                       train_tasks:       list,
                       test_tasks:        list,
                       pretrained:        bool = False) -> dict:
    label       = order_entry["label"]
    task_order  = order_entry["order"]
    order_named = order_entry["order_named"]

    print(f"\n  Regime   : {regime}")
    print(f"  Order    : {order_named}  ({label})")
    print(f"  Algorithm: DER  alpha={DER_ALPHA}")
    print(f"  Early stopping   : {'ON' if USE_EARLY_STOPPING else 'OFF'}"
          + (f"  (patience={ES_PATIENCE}, min_delta={ES_MIN_DELTA})"
             if USE_EARLY_STOPPING else ""))

    model = DERModel(pretrained=pretrained).to(DEVICE)

    acc_matrix   = []
    loss_matrix  = []
    epoch_losses = []
    grad_stats   = []
    pairwise     = []

    for step, task_idx in enumerate(task_order):
        task_label = TASK_LABELS[task_idx]
        print(f"\n    Step {step+1:2d}/{NUM_TASKS} — Task {task_label} "
              f"(classes {task_idx*CLASSES_PER_TASK}–{task_idx*CLASSES_PER_TASK+CLASSES_PER_TASK-1})")

        model.expand()
        
        snapshot = copy.deepcopy(model)
        task_losses, task_stats = train_one_task_der(
            model,
            train_tasks[task_idx],
            regime,
            task_idx,
        )
        epoch_losses.append(task_losses)
        grad_stats.append(task_stats)

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
        "algorithm":      "DER",
        "dataset":        DATASET,
        "der_alpha":      DER_ALPHA,
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
    print("=" * 70)
    print(f"Continual Learning with DER — Task Order Study  [{DATASET.upper()}]")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Dataset       : {DATASET}  ({NUM_TASKS} tasks × {CLASSES_PER_TASK} classes)")
    print(f"Regimes       : {REGIMES}")
    print(f"DER alpha     : {DER_ALPHA}")
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
    results_file  = DER_RESULTS_PRETRAINED if is_pretrained else DER_RESULTS_RANDOM

    print(f"Init mode     : {init_label}")
    print(f"Output        : {results_file}")

    if DATASET in ["imagenet", "imagenet100"]:
        print(f"ImageNet root : {IMAGENET_ROOT}")

    train_ds, test_ds = load_datasets()
    train_tasks  = split_into_tasks(train_ds)
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
            print(f"[DER / {init_label} / {DATASET}] [Run {run_idx:3d}/{total_runs}]  "
                  f"elapsed={elapsed:.1f}min  ETA~{eta:.0f}min")
            print("=" * 70)

            result = run_experiment_der(
                regime, order_entry,
                train_tasks, test_tasks,
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