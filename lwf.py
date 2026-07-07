"""
==========================
Learning without Forgetting (LwF) variant of the task-order study.

Algorithm
---------
Faithfully implements the LwF algorithm as described in:
  Li & Hoiem (2018) "Learning without Forgetting",
  IEEE Transactions on Pattern Analysis and Machine Intelligence.

Key algorithmic details:
  1. Before training on a new task, the current model is snapshotted as a
     "teacher" (frozen reference model).

  2. During training on the new task two losses are combined:
       L_total = L_ce(new task) + lambda * L_kd(old tasks)

     where L_kd is a knowledge distillation loss over ALL previously seen
     tasks:
       L_kd = sum_{t < current} KL( sigma(z_teacher^t / T) ||
                                    sigma(z_student^t / T) )
     and sigma denotes softmax, T is the temperature, z^t is the logit
     slice for task t.

  3. Before joint training on task t (t > 0), a warm-up phase runs
     head-only CE training on the new task for LWF_WARMUP_EPOCHS epochs
     (backbone frozen). This initialises the new task's output units well
     before the KD penalty is introduced, matching Li & Hoiem (2018).

  4. The teacher snapshot is updated (re-snapped) after every task so that
     it always reflects the best model state for ALL tasks seen so far.

  4. Model weights are NOT reset between tasks.

Differences from the vanilla (non-LwF) script
----------------------------------------------
  • `LwFState` replaces `EWCState` — no Fisher estimation is needed.
  • `train_one_task_lwf` replaces `train_one_task_ewc`.
  • After each task, the teacher snapshot is updated via `lwf_state.update`.
  • The KD penalty is added to the CE loss during training of all subsequent
    tasks.
  • Everything else (regimes, evaluate_task, pairwise_metrics,
    acc/loss matrices, JSON schema) is IDENTICAL to continual_learning_ewc.py.

Output
------
  lwf_results_random.json     (random init)
  lwf_results_pretrained.json (ImageNet pretrained init)
  — same JSON schema as ewc_results_*.json, with extra top-level keys:
      "algorithm":    "LwF"
      "lwf_lambda":   <float>
      "lwf_temp":     <float>

Usage
-----
  python lwf.py
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
from torch.utils.data import DataLoader
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
    sample_task_orders, split_into_tasks, build_model, set_trainable_params,
    evaluate_task, pairwise_metrics,
    DATASET,
    extract_grad_vectors, get_grad_stats
)

# ── ImageNet data root ────────────────────────────────────────────────────────
# Path to the directory that contains train/ and val_dirs/.
# Run from the continual_learning/ folder; /mnt is two levels up.
IMAGENET_ROOT = "../../mnt/imagenet/data/ILSVRC2012"

# ── LwF-specific config ────────────────────────────────────────────────────────

# Regularization strength — scales the KD loss relative to the CE loss.
# Tune this: too small → forgetting; too large → plasticity loss.
LWF_LAMBDA = 1.0

# Distillation temperature. Higher T → softer targets → smoother gradients.
# Li & Hoiem (2018) use T=2 by default; common values are 1–4.
LWF_TEMP = 2.0

# KD loss formulation:
#   "standard"  — divide raw logits by T, then softmax/log-softmax (Hinton 2015)
#   "power"     — softmax first, then raise to power 1/T and renormalise
#                 (Li & Hoiem 2018 reference implementation)
KD_FORMULATION = "power"

# Warm-up: number of head-only epochs on the new task BEFORE joint CE+KD
# training begins. Li & Hoiem (2018) train the new task head in isolation
# first so the new head is initialised well before the KD penalty kicks in.
# Set to 0 to disable (reverts to previous behaviour).
if DATASET in ("cifar100", "imagenet100", "imagenet"):
    LWF_WARMUP_EPOCHS = 10
elif DATASET in ("mnist", "fashion_mnist", "kmnist", "qmnist"):
    LWF_WARMUP_EPOCHS = 3

LWF_RESULTS_RANDOM     = f"results/lwf_results_random_{DATASET}.json"
LWF_RESULTS_PRETRAINED = f"results/lwf_results_pretrained_{DATASET}.json"

# Init mode: "random" or "pretrained"
INIT_MODE = "random"
USE_EARLY_STOPPING = True

# REGIMES = ["full_finetune", "last_block", "head_only", "bn_affine_only"]
REGIMES = ["full_finetune", "last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks"]

# ── Data ──────────────────────────────────────────────────────────────────────

def load_datasets():
    """
    Load train and test datasets according to the DATASET selector in tools.py.

    CIFAR-100 (original, unchanged):
      Returns torchvision CIFAR100 datasets with the standard CIFAR transforms.

    MNIST & FashionMNIST (added):
      Returns torchvision MNIST/FashionMNIST datasets with mnist_*_transform, which:
        • Resize images to 32×32 (matching CIFAR-adapted ResNet conv1).
        • Convert 1-channel greyscale to 3-channel by repetition so that
          ImageNet-pretrained weights are usable without architecture changes.
        • Normalise with MNIST statistics replicated to 3 channels.
      The .targets attribute on torchvision MNIST is a plain Python list;
      split_into_tasks converts it to a tensor internally, so no extra
      handling is needed here.
    """
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
        raise ValueError(f"Unknown DATASET: {DATASET!r}.")

    return train_ds, test_ds

# ══════════════════════════════════════════════════════════════════════════════
# LwF State — stores teacher snapshot and list of seen task indices
# ══════════════════════════════════════════════════════════════════════════════

class LwFState:
    """
    Holds the LwF regularization state across tasks.

    After each task `update` is called:
      - The current model is deep-copied as the new frozen teacher.
      - The index of the just-trained task is appended to `seen_tasks`.

    During training the teacher produces soft targets for all previously
    seen tasks, and the KD loss is the mean KL divergence over those tasks.
    """

    def __init__(self, temp: float = 2.0):
        self.temp         = temp
        self.teacher      = None          # frozen copy of model after last task
        self.seen_tasks   = []            # task indices trained so far
        self.context_count = 0

    # ── Update state after a task ──────────────────────────────────────────────

    def update(self, model: nn.Module, task_idx: int):
        """
        Call after finishing training on task `task_idx`.
        Snapshots the model as the new teacher and records the task.
        """
        self.seen_tasks.append(task_idx)
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.context_count += 1
        print(f"      [LwF] Teacher snapshot updated. "
              f"Seen tasks: {[TASK_LABELS[t] for t in self.seen_tasks]}")

    # ── Knowledge Distillation loss ────────────────────────────────────────────

    def kd_loss(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """
        Compute the knowledge distillation loss for all previously seen tasks.

        Two formulations are supported via the KD_FORMULATION config param:

        "standard" (Hinton et al. 2015):
          t_soft = softmax(teacher_logits / T)
          s_soft = log_softmax(student_logits / T)
          L_kd   = KL(t_soft || s_soft) * T^2

        "power" (Li & Hoiem 2018 reference implementation):
          t_soft = softmax(teacher_logits)^(1/T),  renormalised
          s_soft = softmax(student_logits)^(1/T),  renormalised
          L_kd   = -mean( sum( t_soft * log(s_soft) ) )

        Returns the mean KD loss across all seen tasks (scalar on DEVICE).
        Returns 0 if no prior tasks exist (first task).
        """
        if self.context_count == 0 or self.teacher is None:
            return torch.tensor(0., device=DEVICE)

        losses = []
        with torch.no_grad():
            teacher_out = self.teacher(images)   # (B, NUM_CLASSES)

        student_out = model(images)              # (B, NUM_CLASSES)

        for task_idx in self.seen_tasks:
            start = task_idx * CLASSES_PER_TASK
            end   = start + CLASSES_PER_TASK

            t_slice = teacher_out[:, start:end]  # (B, C)
            s_slice = student_out[:, start:end]  # (B, C)

            if KD_FORMULATION == "standard":
                # Hinton et al. (2015): divide logits by T before softmax,
                # scale loss by T^2 to keep gradient magnitudes consistent.
                kd = F.kl_div(
                    F.log_softmax(s_slice / self.temp, dim=1),
                    F.softmax(t_slice / self.temp, dim=1),
                    reduction="batchmean",
                ) * (self.temp ** 2)

            elif KD_FORMULATION == "power":
                # Li & Hoiem (2018): apply softmax first, then power-law smooth.
                with torch.no_grad():
                    t_prob = F.softmax(t_slice, dim=1)
                    # Add epsilon to prevent 0**power
                    t_prob = torch.clamp(t_prob, min=1e-7)
                    t_soft = t_prob ** (1.0 / self.temp)
                    t_soft = t_soft / t_soft.sum(dim=1, keepdim=True)

                s_prob = F.softmax(s_slice, dim=1)
                # Add epsilon to prevent infinite gradient in backward pass when prob is exactly 0
                # derivative of x^a is a*x^(a-1), which is inf for x=0 if a < 1 (temp > 1)
                s_prob = torch.clamp(s_prob, min=1e-7)
                s_soft = s_prob ** (1.0 / self.temp)
                s_soft = s_soft / s_soft.sum(dim=1, keepdim=True)

                kd = -torch.mean(torch.sum(t_soft * torch.log(s_soft + 1e-8), dim=1))

            else:
                raise ValueError(f"Unknown KD_FORMULATION: {KD_FORMULATION!r}. "
                                 f"Choose 'standard' or 'power'.")

            losses.append(kd)

        return sum(losses) / len(losses) if losses else torch.tensor(0., device=DEVICE)

# ── Evaluation (identical to continual_learning_ewc.py) ───────────────────────

# ── Harm / Transfer (identical to continual_learning_ewc.py) ──────────────────

# ══════════════════════════════════════════════════════════════════════════════
# LwF Warm-up — head-only pre-training on the new task (Li & Hoiem 2018)
# ══════════════════════════════════════════════════════════════════════════════

def warmup_new_task(model, train_subset, task_idx, n_epochs):
    """
    Head-only warm-up for the new task before joint CE+KD training.

    Freezes the entire backbone and trains only model.fc for `n_epochs`
    using CE loss on the new task's logit slice. This gives the new task
    head a sensible initialisation so that the KD penalty does not
    dominate too early.

    Skipped automatically when n_epochs == 0 or on the first task
    (where there is no KD penalty anyway).
    """
    if n_epochs == 0:
        return

    task_start = task_idx * CLASSES_PER_TASK
    task_end   = task_start + CLASSES_PER_TASK

    # Freeze everything, train only the FC head
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.fc.parameters():
        p.requires_grad_(True)
    head_params = list(model.fc.parameters())

    loader    = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    optimizer = optim.Adam(head_params, lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(f"      [LwF warm-up] head-only for {n_epochs} epochs ...", flush=True)
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches  = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits       = model(images)[:, task_start:task_end]
            local_labels = labels - task_start
            loss         = criterion(logits, local_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        mean = total_loss / n_batches if n_batches else 0.0
        print(f"        warm-up epoch {epoch+1}/{n_epochs}  ce={mean:.4f}")

    # Leave requires_grad as-is; set_trainable_params in the main loop will
    # reconfigure gradients correctly for the joint training phase.


# ══════════════════════════════════════════════════════════════════════════════
# LwF Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_task_lwf(model, train_subset, lwf_state, regime, task_idx):
    """
    One full task training with LwF regularization (joint phase).

    Called after `warmup_new_task` has already initialised the new head.
    Per-batch update:
      1. Forward pass over current task batch (student).
      2. CE loss on current task's logit slice.
      3. KD loss from frozen teacher over all previously seen tasks
         (zero on the first task, since no teacher exists yet).
      4. Total loss = CE + lwf_lambda * KD.
      5. Single backward + optimizer step.

    Note: the student's full forward pass is shared between the CE head
    and the KD loss computation inside `lwf_state.kd_loss`, so we call
    model(images) once for CE and let kd_loss do its own forward pass.
    This keeps the code clean at the cost of one extra forward pass per
    batch; for a more memory-efficient version the activations could be
    shared.

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

            # ── CE loss on current task ────────────────────────────────────────
            logits       = model(images)[:, task_start:task_end]
            local_labels = labels - task_start
            ce_loss      = criterion(logits, local_labels)

            optimizer.zero_grad()
            ce_loss.backward(retain_graph=True)
            g_task = extract_grad_vectors(model)

            # ── KD loss over all previously seen tasks (0 on first task) ──────
            penalty = lwf_state.kd_loss(model, images)
            ret_loss = LWF_LAMBDA * penalty
            
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
              f"kd_penalty={mean_penalty:.6f}  "
              f"lr={current_lr:.2e}")

        if USE_EARLY_STOPPING and early_stop(mean_loss, epoch=epoch + 1):
            print(f"      [early stop] no improvement for {ES_PATIENCE} epochs "
                  f"(best={early_stop.best_loss:.4f}), stopping at epoch {epoch+1}.")
            break

    return epoch_losses, epoch_stats

# ── Single experiment run ─────────────────────────────────────────────────────

def run_experiment_lwf(regime, order_entry, train_tasks, test_tasks,
                       pretrained=False):
    label       = order_entry["label"]
    task_order  = order_entry["order"]
    order_named = order_entry["order_named"]

    print(f"\n  Regime   : {regime}")
    print(f"  Order    : {order_named}  ({label})")
    print(f"  Algorithm: LwF  lambda={LWF_LAMBDA}  temp={LWF_TEMP}  warmup_epochs={LWF_WARMUP_EPOCHS}  kd={KD_FORMULATION}")
    print(f"  Early stopping   : {'ON' if USE_EARLY_STOPPING else 'OFF'}"
          + (f"  (patience={ES_PATIENCE}, min_delta={ES_MIN_DELTA})"
             if USE_EARLY_STOPPING else ""))

    model     = build_model(pretrained=pretrained)
    lwf_state = LwFState(temp=LWF_TEMP)

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

        # ── Warm-up: head-only CE on new task (skipped on first task) ─────────
        if step > 0 and LWF_WARMUP_EPOCHS > 0:
            warmup_new_task(model, train_tasks[task_idx], task_idx,
                            LWF_WARMUP_EPOCHS)

        task_losses, task_stats = train_one_task_lwf(
            model,
            train_tasks[task_idx],
            lwf_state,
            regime,
            task_idx,
        )
        epoch_losses.append(task_losses)
        grad_stats.append(task_stats)

        # ── Update teacher snapshot after this task ────────────────────────────
        lwf_state.update(model, task_idx)

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
        "algorithm":         "LwF",
        "dataset":           DATASET,
        "lwf_lambda":        LWF_LAMBDA,
        "lwf_temp":          LWF_TEMP,
        "lwf_warmup_epochs": LWF_WARMUP_EPOCHS,
        "kd_formulation":    KD_FORMULATION,
        "init":              "pretrained" if pretrained else "random",
        "regime":            regime,
        "order_label":       label,
        "order_named":       order_named,
        "task_order":        task_order,
        "acc_matrix":        acc_matrix,
        "loss_matrix":       loss_matrix,
        "epoch_losses":      epoch_losses,
        "grad_stats":        grad_stats,
        "pairwise":          pairwise,
        "final_accs":        {TASK_LABELS[k]: v for k, v in final_accs.items()},
        "avg_acc":           avg_acc,
        "forgetting":        {TASK_LABELS[k]: v for k, v in forgetting.items()},
        "avg_forgetting":    avg_forgetting,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"Continual Learning with LwF — Task Order Study  [{DATASET.upper()}]")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Dataset       : {DATASET}  ({NUM_TASKS} tasks × {CLASSES_PER_TASK} classes)")
    print(f"Regimes       : {REGIMES}")
    print(f"LwF lambda    : {LWF_LAMBDA}")
    print(f"LwF temp      : {LWF_TEMP}")
    print(f"LwF warmup    : {LWF_WARMUP_EPOCHS} epochs")
    print(f"KD formulation: {KD_FORMULATION}")
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
    results_file  = LWF_RESULTS_PRETRAINED if is_pretrained else LWF_RESULTS_RANDOM

    print(f"Init mode     : {init_label}")
    print(f"Output        : {results_file}")

    if DATASET in ["imagenet", "imagenet100"]:
        print(f"ImageNet root : {IMAGENET_ROOT}")

    train_ds, test_ds = load_datasets()
    train_tasks = split_into_tasks(train_ds)
    test_tasks  = split_into_tasks(test_ds)

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
            print(f"[LwF / {init_label} / {DATASET}] [Run {run_idx:3d}/{total_runs}]  "
                  f"elapsed={elapsed:.1f}min  ETA~{eta:.0f}min")
            print("=" * 70)

            result = run_experiment_lwf(
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