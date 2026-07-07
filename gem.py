"""
=========================
Gradient Episodic Memory (GEM) variant of the task-order study.

Algorithm
---------
Implements the GEM algorithm from:
  Lopez-Paz, D. & Ranzato, M. (2017).
  "Gradient Episodic Memory for Continual Learning."
  NeurIPS 2017.  https://arxiv.org/abs/1706.08840

Key algorithmic details:

  GEM stores a small episodic memory M_t for each past task t, then at
  every training step constrains the gradient g of the current task so
  that the loss on every past task does not increase:

      <g_tilde, g_t> >= 0   for all past tasks t

  where g_t is the gradient of the loss on M_t.

  If all dot-products are already non-negative (no constraint violated),
  g_tilde = g (the raw current-task gradient is used as-is).

  Otherwise, GEM solves a Quadratic Program (QP) to find the closest
  vector g_tilde to g that satisfies all constraints simultaneously
  (paper Algorithm 1 / Eq. 6).

  1. Memory population (end of task):
       After training on task t, store exactly GEM_MEMORY_STRENGTH
       samples per task (random subset) in a fixed episodic buffer M_t.
       The buffer is never modified after the task ends.
       Samples are stored non-augmented (test_transform).

  2. observe(x_cur, y_cur) — per-batch update:
       a. For each past task t:
            - Forward + backward on M_t  →  store gradient g_t.
       b. Forward + backward on current batch  →  gradient g.
       c. Check violations:  dot(g, g_t) < 0  for any t?
            - No violations  →  optimizer.step() with g unchanged.
            - Violations     →  solve QP to get g_tilde, overwrite
                                 model gradients, optimizer.step().
       d. optimizer.step().

  3. QP formulation (dual, following paper Appendix A and Mammoth):
       Primal:   min  ½||g_tilde - g||²
                 s.t. G g_tilde >= 0
       where G is the (n_past_tasks × n_params) matrix of past gradients.
       Dual:     min  ½ v^T G G^T v  +  g^T G^T v
                 s.t. v >= 0
       Solved with `quadprog.solve_qp`.
       The primal solution is recovered as:
           g_tilde = g + G^T v*

  4. Memory stores raw (non-augmented) images + global labels.
     At constraint-check time augmentation is NOT applied
     (consistent with Mammoth's GEM).

Dependency
----------
  pip install quadprog
  (C extension; Linux/macOS only — same constraint as Mammoth's GEM)

Differences from continual_learning_er.py
------------------------------------------
  • `GEMBuffer` stores per-task fixed memories (not a shared reservoir).
  • `store_grad` / `overwrite_grad` / `project_gradients` replace
    the ER replay logic.
  • `train_one_task_gem` replaces `train_one_task_er`.
  • No augmented/non-augmented dual dataset split needed for training,
    but a noaug dataset IS needed for memory storage.
  • Everything else (regimes, evaluate_task, pairwise_metrics,
    acc/loss matrices, JSON schema) is IDENTICAL.

Output
------
  gem_results_random.json
  gem_results_pretrained.json
  — same JSON schema as er_results_*.json, with:
      "algorithm": "GEM"
      "gem_memory_strength": <int>   (samples stored per past task)

Usage
-----
  python gem.py
"""

import copy
import json
import time

import numpy as np
import quadprog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR100, MNIST, FashionMNIST, QMNIST, ImageFolder

from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.tools import (
    LR_MIN, ES_PATIENCE, ES_MIN_DELTA, EarlyStopping,
    NUM_TASKS, CLASSES_PER_TASK, NUM_CLASSES, DATA_ROOT, EPOCHS_PER_TASK,
    BATCH_SIZE, LR, NUM_WORKERS, RANDOM_SEED, NUM_ORDERS, TASK_LABELS,
    DEVICE, PIN_MEMORY, TASK_ORDERS, train_transform, test_transform,
    mnist_train_transform, mnist_test_transform, DATASET,
    imagenet_train_transform, imagenet_test_transform, ImageNet100Folder, HFKMNIST,  # ImageNet transforms
    sample_task_orders, split_into_tasks, build_model, set_trainable_params,
    evaluate_task, pairwise_metrics,
    extract_grad_vectors, get_grad_stats
)

# ── ImageNet data root ────────────────────────────────────────────────────────
# Path to the directory that contains train/ and val_dirs/.
# Run from the continual_learning/ folder; /mnt is two levels up.
IMAGENET_ROOT = "../../mnt/imagenet/data/ILSVRC2012"

# ── GEM-specific config ───────────────────────────────────────────────────────

# Number of samples stored per past task in the episodic memory.
# Mammoth default for seq-cifar100: 200 samples/task.
GEM_MEMORY_STRENGTH = 200

# Small positive constant added to QP Hessian diagonal for numerical
# stability (standard practice, same as Mammoth).
GEM_QP_EPS = 1e-5

GEM_RESULTS_RANDOM     = f"results/gem_results_random_{DATASET}.json"
GEM_RESULTS_PRETRAINED = f"results/gem_results_pretrained_{DATASET}.json"

# ── Init mode ─────────────────────────────────────────────────────────────────
# "random"     → kaiming random initialization
# "pretrained" → ImageNet pretrained backbone (conv1 + fc replaced)
INIT_MODE = "random"
USE_EARLY_STOPPING = True

# REGIMES = ["full_finetune", "last_block", "head_only", "bn_affine_only"]
REGIMES = ["full_finetune", "last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks"]

# ── Data ──────────────────────────────────────────────────────────────────────

# Non-augmented transform used when *storing* samples in episodic memory.
cifar_store_transform = test_transform
mnist_store_transform = mnist_test_transform

def load_datasets():
    """
    Returns (train_aug, train_noaug, test_ds).
    train_aug  : used for actual training (with augmentation).
    train_noaug: used only to populate episodic memories (no augmentation).
    """
    if DATASET == "cifar100":
        train_aug   = CIFAR100(root=DATA_ROOT, train=True,  download=False,
                               transform=train_transform)
        train_noaug = CIFAR100(root=DATA_ROOT, train=True,  download=False,
                               transform=cifar_store_transform)
        test_ds     = CIFAR100(root=DATA_ROOT, train=False, download=False,
                               transform=test_transform)
    elif DATASET == "mnist":
        train_aug   = MNIST(root=DATA_ROOT, train=True,  download=True,
                            transform=mnist_train_transform)
        train_noaug = MNIST(root=DATA_ROOT, train=True,  download=True,
                            transform=mnist_store_transform)
        test_ds     = MNIST(root=DATA_ROOT, train=False, download=True,
                            transform=mnist_test_transform)
    elif DATASET == "fashion_mnist":
        train_aug   = FashionMNIST(root=DATA_ROOT, train=True,  download=True,
                                   transform=mnist_train_transform)
        train_noaug = FashionMNIST(root=DATA_ROOT, train=True,  download=True,
                                   transform=mnist_store_transform)
        test_ds     = FashionMNIST(root=DATA_ROOT, train=False, download=True,
                                   transform=mnist_test_transform)
    elif DATASET == "kmnist":
        train_aug   = HFKMNIST(root=DATA_ROOT, train=True,  download=True,
                                   transform=mnist_train_transform)
        train_noaug = HFKMNIST(root=DATA_ROOT, train=True,  download=True,
                                   transform=mnist_store_transform)
        test_ds     = HFKMNIST(root=DATA_ROOT, train=False, download=True,
                                   transform=mnist_test_transform)
    elif DATASET == "qmnist":
        train_aug   = QMNIST(root=DATA_ROOT, what="train", compat=True, download=True,
                                   transform=mnist_train_transform)
        train_noaug = QMNIST(root=DATA_ROOT, what="train", compat=True, download=True,
                                   transform=mnist_store_transform)
        test_ds     = QMNIST(root=DATA_ROOT, what="test", compat=True, download=True,
                                   transform=mnist_test_transform)
    elif DATASET in ["imagenet", "imagenet100"]:
        FolderClass = ImageNet100Folder if DATASET == "imagenet100" else ImageFolder
        train_aug   = FolderClass(root=f"{IMAGENET_ROOT}/train", transform=imagenet_train_transform)
        train_noaug = FolderClass(root=f"{IMAGENET_ROOT}/train", transform=imagenet_test_transform)
        test_ds     = FolderClass(root=f"{IMAGENET_ROOT}/val_dirs", transform=imagenet_test_transform)
        expected_classes = 100 if DATASET == "imagenet100" else 1000
        assert len(train_aug.classes) == expected_classes, (
            f"Expected {expected_classes} ImageNet classes, found {len(train_aug.classes)}. "
            f"Check IMAGENET_ROOT = {IMAGENET_ROOT!r}"
        )
    else:
        raise ValueError(f"Unknown DATASET: {DATASET!r}")
    return train_aug, train_noaug, test_ds


# ══════════════════════════════════════════════════════════════════════════════
# GEM episodic memory
# ══════════════════════════════════════════════════════════════════════════════

class GEMBuffer:
    """
    Per-task fixed episodic memory for GEM.

    After each task ends, `add_task_memory` stores exactly
    GEM_MEMORY_STRENGTH (image, label) pairs chosen at random from the
    task's non-augmented training set.  The stored tensors never change.

    At constraint-check time, `get_task_memory(t)` returns images, labels,
    and the original task_idx for the t-th stored task (images on DEVICE).

    FIX: task_idx is stored alongside images/labels so that the correct
    global class offset (task_idx * CLASSES_PER_TASK) can always be
    recovered at replay time, regardless of the order tasks were encountered.
    Without this, buffer index `t` was used as the task index, causing
    wrong label localisation (e.g. task H stored at buffer[0] would be
    replayed with offset 0 instead of 70, producing out-of-range labels).
    """

    def __init__(self):
        # memories[i] = (images_tensor, labels_tensor, task_idx)  on CPU
        self.memories: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def add_task_memory(self, noaug_subset: Subset, task_idx: int):
        """
        Randomly sample GEM_MEMORY_STRENGTH examples from noaug_subset
        and store them together with task_idx.
        Called once at the END of each task.
        """
        n        = len(noaug_subset)
        k        = min(GEM_MEMORY_STRENGTH, n)
        indices  = np.random.choice(n, size=k, replace=False)

        imgs_list   = []
        labels_list = []
        for idx in indices:
            img, label = noaug_subset[int(idx)]
            imgs_list.append(img)
            labels_list.append(label)

        imgs   = torch.stack(imgs_list)                        # (k, C, H, W) CPU
        labels = torch.tensor(labels_list, dtype=torch.long)  # (k,)          CPU
        self.memories.append((imgs, labels, task_idx))

    def get_task_memory(self, buf_idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Return (images, labels, task_idx) for the buf_idx-th stored entry.
        Images and labels are moved to DEVICE; task_idx is a plain int.
        """
        imgs, labels, task_idx = self.memories[buf_idx]
        return imgs.to(DEVICE), labels.to(DEVICE), task_idx

    def n_tasks(self) -> int:
        return len(self.memories)


# ══════════════════════════════════════════════════════════════════════════════
# Gradient utilities  (store_grad / overwrite_grad / project)
# ══════════════════════════════════════════════════════════════════════════════

def get_grad_dims(model: nn.Module, trainable: list) -> list[int]:
    """
    Return a list of per-parameter gradient sizes (num elements).
    Used to pre-allocate the flat gradient vectors.
    `trainable` is the list of parameter tensors returned by
    set_trainable_params — only these contribute to the gradient buffer.
    """
    param_set = {id(p) for p in trainable}
    return [p.numel() for p in model.parameters() if id(p) in param_set]


def store_grad(trainable: list, grad_vec: torch.Tensor, grad_dims: list[int]):
    """
    Flatten and copy current .grad values of `trainable` parameters into
    `grad_vec` (a pre-allocated 1D CPU tensor of length sum(grad_dims)).

    Matches Mammoth's store_grad(params, grads, grad_dims):
      cnt = 0
      for i, param in enumerate(params()):
          if param.grad is not None:
              beg = 0 if i == 0 else sum(grad_dims[:i])
              en  = sum(grad_dims[:i+1])
              grads[beg:en].copy_(param.grad.data.view(-1))
    """
    offset = 0
    for p, d in zip(trainable, grad_dims):
        if p.grad is not None:
            grad_vec[offset : offset + d].copy_(p.grad.data.view(-1))
        else:
            grad_vec[offset : offset + d].zero_()
        offset += d


def overwrite_grad(trainable: list, new_grad: torch.Tensor, grad_dims: list[int]):
    """
    Write values from flat `new_grad` (CPU) back into .grad of each
    trainable parameter, reshaping to match the original tensor shape.

    Matches Mammoth's overwrite_grad(params, newgrad, grad_dims).
    """
    offset = 0
    for p, d in zip(trainable, grad_dims):
        if p.grad is not None:
            p.grad.data.copy_(
                new_grad[offset : offset + d].view(p.grad.data.shape)
            )
        offset += d


def project_gradients(grad_cur: torch.Tensor,
                      grad_past: torch.Tensor) -> torch.Tensor:
    """
    Project `grad_cur` (current-task gradient, 1D CPU tensor) so that its
    dot-product with every row of `grad_past` (n_past × n_params, CPU) is
    non-negative.

    Solves the dual QP (paper Appendix A / Mammoth project2cone2):
        min   ½ v^T (G G^T + ε I) v  +  (G g)^T v
        s.t.  v >= 0
    where G = grad_past  (n_past × n_params),  g = grad_cur  (n_params).

    Recovery:  g_tilde = g + G^T v*

    Returns the projected gradient g_tilde as a 1D CPU float32 tensor.

    Numerical notes:
      • quadprog.solve_qp uses Goldfarb-Idnani's active-set method.
      • The Hessian G G^T + ε I is always positive definite (ε > 0).
      • We convert to float64 for numerical stability (same as Mammoth).
    """
    n_past  = grad_past.shape[0]
    G       = grad_past.double().numpy()    # (n_past, n_params)
    g       = grad_cur.double().numpy()     # (n_params,)

    # Hessian of dual: G G^T + ε I   (n_past × n_past, positive definite)
    H       = G @ G.T + GEM_QP_EPS * np.eye(n_past)

    # Linear term of dual: G g
    f       = (G @ g).reshape(-1)           # (n_past,)

    # Constraints: v >= 0  ↔  I v >= 0  (passed as equality/inequality to quadprog)
    # quadprog.solve_qp(G, a, C, b, meq):
    #   min  ½ x^T G x - a^T x   s.t.  C^T x >= b
    # Our dual:
    #   min  ½ v^T H v + f^T v  ↔  min  ½ v^T H v - (-f)^T v
    #   s.t.  I v >= 0
    C       = np.eye(n_past)                # (n_past, n_past)
    b       = np.zeros(n_past)              # (n_past,)

    try:
        v = quadprog.solve_qp(H, -f, C, b, 0)[0]   # meq=0 → all inequality
    except ValueError:
        # QP failed (degenerate problem) — fall back to raw gradient
        return grad_cur.clone()

    v_t     = torch.from_numpy(v).float()   # (n_past,)
    G_t     = grad_past.float()             # (n_past, n_params)

    # Projected gradient: g_tilde = g + G^T v
    return grad_cur + G_t.t().mv(v_t)


# ══════════════════════════════════════════════════════════════════════════════
# GEM Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_task_gem(model:        nn.Module,
                       train_aug:    Subset,      # augmented current-task data
                       train_noaug:  Subset,      # non-augmented (for memory)
                       gem_buffer:   GEMBuffer,
                       regime:       str,
                       task_idx:     int) -> list:
    """
    One full task training with GEM.

    Per-batch update (Mammoth's observe()):
      1. For each past task t stored in gem_buffer:
           - Forward + backward on M_t  →  flatten gradient into g_t.
           - zero_grad after storing.
      2. Forward + backward on current batch  →  gradient g.
      3. If any dot(g, g_t) < 0  (constraint violated):
           - Project g → g_tilde via QP.
           - Overwrite model gradients with g_tilde.
      4. optimizer.step().

    After all epochs, add current task's memory to gem_buffer.
    Returns epoch_losses (list of mean CE loss per epoch).
    """
    trainable = set_trainable_params(model, regime)
    if not trainable:
        print("    [warn] no trainable params, skipping.")
        return []

    task_start_cls = task_idx * CLASSES_PER_TASK
    task_end_cls   = task_start_cls + CLASSES_PER_TASK

    loader    = DataLoader(train_aug, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                           drop_last=False)
    optimizer = optim.Adam(trainable, lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TASK, eta_min=LR_MIN)
    early_stop = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)

    # Pre-compute gradient dimensions once (fixed for this regime)
    grad_dims = get_grad_dims(model, trainable)
    n_params  = sum(grad_dims)

    model.train()
    epoch_losses = []
    epoch_stats = []
    n_past = gem_buffer.n_tasks()

    # Pre-allocate gradient storage matrix: (n_past, n_params) on CPU
    # Resized at task start; stays the same throughout all epochs.
    grad_past = torch.zeros(n_past, n_params) if n_past > 0 else None

    for epoch in range(EPOCHS_PER_TASK):
        total_loss = 0.0
        n_batches  = 0
        batch_stats = []

        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            local_labels = labels - task_start_cls

            # ── Step 1: Compute past-task gradients ───────────────────────────
            if n_past > 0:
                for t in range(n_past):
                    # FIX: unpack task_idx from buffer so the global class
                    # offset is always correct, regardless of encounter order.
                    mem_imgs, mem_labels, mem_task_idx = gem_buffer.get_task_memory(t)
                    t_start = mem_task_idx * CLASSES_PER_TASK
                    t_end   = t_start + CLASSES_PER_TASK
                    t_local = mem_labels - t_start   # now always in [0, CLASSES_PER_TASK)

                    model.zero_grad()
                    mem_logits  = model(mem_imgs)
                    mem_task_lg = mem_logits[:, t_start:t_end]
                    mem_loss    = nn.functional.cross_entropy(mem_task_lg, t_local)
                    mem_loss.backward()

                    store_grad(trainable, grad_past[t], grad_dims)

            # ── Step 2: Current-task gradient ─────────────────────────────────
            model.zero_grad()
            logits      = model(imgs)
            task_logits = logits[:, task_start_cls:task_end_cls]
            ce_loss     = nn.functional.cross_entropy(task_logits, local_labels)
            ce_loss.backward()

            g_task = extract_grad_vectors(model)
            g_ret = torch.tensor([], device=DEVICE)

            # ── Step 3: Check & project if needed ────────────────────────────
            if n_past > 0:
                grad_cur = torch.zeros(n_params)
                store_grad(trainable, grad_cur, grad_dims)

                # Check for violated constraints: dot(g, g_t) < 0
                dots = grad_past.mv(grad_cur)    # (n_past,)
                if (dots < 0).any():
                    grad_tilde = project_gradients(grad_cur, grad_past)
                    overwrite_grad(trainable, grad_tilde, grad_dims)
                    # We have a ret gradient
                    g_ret = extract_grad_vectors(model)

            # ── Step 4: Update ────────────────────────────────────────────────
            optimizer.step()

            batch_stats.append(get_grad_stats(g_task, g_ret))

            total_loss += ce_loss.item()
            n_batches  += 1

        mean_epoch_loss = total_loss / n_batches if n_batches > 0 else 0.0
        epoch_losses.append(round(mean_epoch_loss, 6))

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

        violations_str = f"  n_past={n_past}" if n_past > 0 else ""
        print(f"      epoch {epoch+1:>3}/{EPOCHS_PER_TASK}  "
              f"ce_loss={mean_epoch_loss:.4f}  lr={current_lr:.2e}{violations_str}")

        if USE_EARLY_STOPPING and early_stop(mean_epoch_loss, epoch=epoch + 1):
            print(f"      [early stop] no improvement for {ES_PATIENCE} epochs "
                  f"(best={early_stop.best_loss:.4f}), stopping at epoch {epoch+1}.")
            break

    # ── Populate episodic memory for this task (after all training) ───────────
    # FIX: pass task_idx so the buffer can recover the correct global offset
    # at replay time.
    gem_buffer.add_task_memory(train_noaug, task_idx)
    print(f"      memory updated: {gem_buffer.n_tasks()} tasks × "
          f"{GEM_MEMORY_STRENGTH} samples = "
          f"{gem_buffer.n_tasks() * GEM_MEMORY_STRENGTH} total")

    return epoch_losses, epoch_stats


# ── Single experiment run ─────────────────────────────────────────────────────

def run_experiment_gem(regime: str, order_entry: dict,
                       train_tasks_aug:   list,
                       train_tasks_noaug: list,
                       test_tasks:        list,
                       pretrained: bool = False) -> dict:
    label       = order_entry["label"]
    task_order  = order_entry["order"]
    order_named = order_entry["order_named"]

    print(f"\n  Regime  : {regime}")
    print(f"  Order   : {order_named}  ({label})")
    print(f"  Memory  : {GEM_MEMORY_STRENGTH} samples/task")
    print(f"  Early stopping   : {'ON' if USE_EARLY_STOPPING else 'OFF'}"
          + (f"  (patience={ES_PATIENCE}, min_delta={ES_MIN_DELTA})"
             if USE_EARLY_STOPPING else ""))

    model      = build_model(pretrained=pretrained)
    gem_buffer = GEMBuffer()   # fresh buffer per run

    acc_matrix   = []
    loss_matrix  = []
    epoch_losses = []
    grad_stats   = []
    pairwise     = []

    for step, task_idx in enumerate(task_order):
        task_label = TASK_LABELS[task_idx]
        print(f"\n    Step {step+1:2d}/{NUM_TASKS} — Task {task_label} "
              f"(classes {task_idx * CLASSES_PER_TASK}–"
              f"{task_idx * CLASSES_PER_TASK + CLASSES_PER_TASK - 1})")

        snapshot    = copy.deepcopy(model)
        task_losses, task_stats = train_one_task_gem(
            model,
            train_tasks_aug[task_idx],
            train_tasks_noaug[task_idx],
            gem_buffer,
            regime,
            task_idx,
        )
        epoch_losses.append(task_losses)
        grad_stats.append(task_stats)

        # Evaluate all tasks seen so far
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
        "algorithm":           "GEM",
        "dataset":             DATASET,
        "gem_memory_strength": GEM_MEMORY_STRENGTH,
        "init":                "pretrained" if pretrained else "random",
        "regime":              regime,
        "order_label":         label,
        "order_named":         order_named,
        "task_order":          task_order,
        "acc_matrix":          acc_matrix,
        "loss_matrix":         loss_matrix,
        "epoch_losses":        epoch_losses,
        "grad_stats":          grad_stats,
        "pairwise":            pairwise,
        "final_accs":          {TASK_LABELS[k]: v for k, v in final_accs.items()},
        "avg_acc":             avg_acc,
        "forgetting":          {TASK_LABELS[k]: v for k, v in forgetting.items()},
        "avg_forgetting":      avg_forgetting,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"Continual Learning with GEM — Task Order Study  [{DATASET.upper()}]")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Dataset       : {DATASET}  ({NUM_TASKS} tasks × {CLASSES_PER_TASK} classes)")
    print(f"Regimes       : {REGIMES}")
    print(f"Memory/task   : {GEM_MEMORY_STRENGTH}")
    print(f"Total memory  : {GEM_MEMORY_STRENGTH * NUM_TASKS} "
          f"({GEM_MEMORY_STRENGTH} × {NUM_TASKS} tasks)")
    print(f"Orders        : 1 canonical + {NUM_ORDERS} random = "
          f"{len(TASK_ORDERS)} total")
    print(f"Epochs/task   : {EPOCHS_PER_TASK}")
    print(f"LR               : {LR}  →  {LR_MIN} (cosine)")
    print(f"Early stopping   : {'ON' if USE_EARLY_STOPPING else 'OFF'}"
          + (f"  (patience={ES_PATIENCE}, min_delta={ES_MIN_DELTA})"
             if USE_EARLY_STOPPING else ""))
    print(f"Total runs    : {len(REGIMES) * len(TASK_ORDERS)} "
          f"({len(REGIMES)} regimes × {len(TASK_ORDERS)} orders)")

    if INIT_MODE not in ("random", "pretrained"):
        raise ValueError(f"INIT_MODE must be 'random' or 'pretrained', "
                         f"got {INIT_MODE!r}")
    is_pretrained = (INIT_MODE == "pretrained")
    init_label    = INIT_MODE
    results_file  = GEM_RESULTS_PRETRAINED if is_pretrained else GEM_RESULTS_RANDOM

    print(f"Init mode     : {init_label}")
    print(f"Output        : {results_file}")

    if DATASET in ["imagenet", "imagenet100"]:
        print(f"ImageNet root : {IMAGENET_ROOT}")

    train_aug_ds, train_noaug_ds, test_ds = load_datasets()
    train_tasks_aug   = split_into_tasks(train_aug_ds)
    train_tasks_noaug = split_into_tasks(train_noaug_ds)
    test_tasks        = split_into_tasks(test_ds)

    total_runs  = len(REGIMES) * len(TASK_ORDERS)
    all_results = []
    run_idx     = 0
    t0          = time.time()

    for regime in REGIMES:
        for order_entry in TASK_ORDERS:
            run_idx += 1
            elapsed  = (time.time() - t0) / 60
            eta      = ((elapsed / run_idx) * (total_runs - run_idx)
                        if run_idx > 1 else 0)
            print(f"\n{'='*70}")
            print(f"[GEM / {init_label} / {DATASET}] "
                  f"[Run {run_idx:3d}/{total_runs}]  "
                  f"elapsed={elapsed:.1f}min  ETA~{eta:.0f}min")
            print("=" * 70)

            result = run_experiment_gem(
                regime, order_entry,
                train_tasks_aug, train_tasks_noaug, test_tasks,
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

    print("\n-- Best and worst ordering per regime (by avg accuracy) --")
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