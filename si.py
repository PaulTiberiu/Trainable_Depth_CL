"""
=========================
Synaptic Intelligence (SI) variant of the task-order study.

Algorithm
---------
Implements the SI algorithm from:
  Zenke, F., Poole, B., & Ganguli, S. (2017).
  "Continual Learning Through Synaptic Intelligence."
  ICML 2017.  https://arxiv.org/abs/1703.04200

Key algorithmic details:
  SI consolidates knowledge by computing, for each parameter θ_k, how
  important it was to past tasks and penalising large changes to those
  parameters when learning new tasks.

  1. Importance accumulation (online, during training):
       After each gradient step, accumulate the *path integral* of the
       gradient × parameter update:
         ω_k  +=  -g_k  ×  Δθ_k
       where g_k is the PURE TASK LOSS gradient w.r.t. θ_k (CE only,
       before SI penalty injection) and Δθ_k = θ_k - θ_k_prev.
       Note the sign: Zenke et al. define ω_k as the negative gradient
       dotted with the parameter displacement (the paper's Eq. 3).

       *** FIX vs previous version ***
       The CE gradient must be snapshotted BEFORE the SI penalty gradient
       is injected into param.grad. Using the post-injection gradient
       contaminates ω with the penalty's own gradient, creating a feedback
       loop that corrupts importance weights across tasks and causes SI to
       massively underperform.

  2. End-of-task Ω update:
       After each task t, the consolidated importance is updated:
         Ω_k^t  +=  ω_k / ( (θ_k^* - θ_k^{t-1})^2 + ξ )
       where θ_k^* is the end-of-task optimal parameter,
             θ_k^{t-1} is the parameter value at the start of this task,
             ξ  is a small damping constant (default 0.1, same as paper).
       Ω is then accumulated across tasks: Ω ← Ω + Ω^t.

  3. Surrogate penalty — injected as a gradient, not a loss term:
       Mammoth does NOT add SI as a scalar loss and backprop it together
       with CE.  Instead it:
         a) backprops CE loss only → obtains CE gradients
         b) snapshots pure CE gradients (for ω accumulation later)   ← FIX
         c) computes the closed-form penalty gradient analytically:
              ∂L_SI/∂θ_k  =  c · 2 · Ω_k · (θ_k - θ_k^*)
         d) adds that directly to param.grad before optimizer.step()
         e) clips all gradients to [-1, 1]  (nn.utils.clip_grad_value_)
         f) AFTER optimizer.step(), accumulates ω using the snapshotted
            pure CE gradients (not the contaminated post-penalty grad)  ← FIX
       This is mathematically equivalent to backpropping
         L = CE + c/2 · Σ_k Ω_k·(θ_k - θ_k^*)²
       but avoids a second autograd pass through the penalty graph.
       c  is the SI regularisation coefficient (default 1.0).

  4. No memory buffer is needed — SI is a parameter-regularisation method.

Differences from continual_learning_er.py
------------------------------------------
  • `SynapticIntelligence` class manages ω, Ω, θ_prev, and θ_star.
  • `train_one_task_si` replaces `train_one_task_er`.
  • No `ReservoirBuffer`; no replay; no dual (aug/noaug) dataset loading.
  • `_si_penalty` computes the quadratic surrogate loss.
  • Everything else (regimes, freeze, evaluate_task, pairwise_metrics,
    acc/loss matrices, JSON schema) is IDENTICAL to continual_learning_er.py.

Output
------
  si_results_random.json     (random init)
  si_results_pretrained.json (ImageNet pretrained init)
  — same JSON schema as er_results_*.json, with:
      "algorithm": "SI"
      "si_lambda": <float>
      "si_xi":     <float>

Usage
-----
  python si.py
"""

import copy
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

# ── SI-specific config ────────────────────────────────────────────────────────

# Regularisation strength c (Mammoth's --c argument; paper default: 1.0)
# Higher values → stronger consolidation of past tasks.
SI_C = 100.0

# Damping constant ξ to avoid division by zero in importance update
# (paper default: 0.1)
SI_XI = 0.1

SI_RESULTS_RANDOM     = f"results/si_results_random_{DATASET}.json"
SI_RESULTS_PRETRAINED = f"results/si_results_pretrained_{DATASET}.json"

# ── Init mode — edit this before running ─────────────────────────────────────
# "random"     → kaiming random initialization
# "pretrained" → ImageNet pretrained backbone (conv1 + fc replaced)
INIT_MODE = "random"
USE_EARLY_STOPPING = True

# REGIMES = ["full_finetune", "last_block", "head_only", "bn_affine_only"]
REGIMES = ["full_finetune", "last_block", "last_2_blocks", "last_3_blocks", "last_6_blocks"]

# ── Data ──────────────────────────────────────────────────────────────────────

def load_datasets():
    """Load train/test datasets (augmented only — SI needs no noaug copy)."""
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
        train_ds = FolderClass(root=f"{IMAGENET_ROOT}/train", transform=imagenet_train_transform)
        test_ds  = FolderClass(root=f"{IMAGENET_ROOT}/val_dirs", transform=imagenet_test_transform)
        expected_classes = 100 if DATASET == "imagenet100" else 1000
        assert len(train_ds.classes) == expected_classes, (
            f"Expected {expected_classes} ImageNet classes, found {len(train_ds.classes)}. "
            f"Check IMAGENET_ROOT = {IMAGENET_ROOT!r}"
        )
    else:
        raise ValueError(f"Unknown DATASET: {DATASET!r}")
    return train_ds, test_ds


# ══════════════════════════════════════════════════════════════════════════════
# Synaptic Intelligence state tracker
# ══════════════════════════════════════════════════════════════════════════════

class SynapticIntelligence:
    """
    Manages all SI auxiliary variables for a single model.

    Attributes
    ----------
    omega : dict[str, Tensor]
        Consolidated importance Ω_k (accumulated across all past tasks).
        Lives on CPU; moved to DEVICE only when computing the penalty gradient.
    theta_star : dict[str, Tensor]
        θ_k^* — parameter values at the end of the *previous* task
        (the "anchor" for the surrogate penalty gradient).
    _w : dict[str, Tensor]
        Running path-integral ω_k accumulated within the *current* task.
        Reset to zero at each task_start(); finalised at task_end().
    _task_start_params : dict[str, Tensor]
        Parameter values at the start of the current task (θ_k^{t-1}),
        used in the end-of-task Ω update denominator.
    """

    def __init__(self, model: nn.Module):
        # Only track parameters that actually receive gradients at any point.
        # We key by parameter name for stable dict ordering.
        self._param_names = [n for n, p in model.named_parameters()
                             if p.requires_grad]

        # All tensors stored on CPU to avoid GPU memory pressure between tasks.
        self.omega            = {n: torch.zeros_like(p.data.cpu())
                                 for n, p in model.named_parameters()
                                 if n in self._param_names}
        self.theta_star       = {n: p.data.cpu().clone()
                                 for n, p in model.named_parameters()
                                 if n in self._param_names}

        # Per-task running accumulators (reset at each task start)
        self._w                 = {}
        self._task_start_params = {}

    # ── Called once at the start of each task ────────────────────────────────

    def task_start(self, model: nn.Module):
        """
        Snapshot current parameters as the task-start anchor θ^{t-1} and
        reset the within-task path-integral accumulators ω.

        Also refreshes _param_names in case set_trainable_params changed
        which parameters require_grad (e.g., regime switch).
        """
        self._param_names = [n for n, p in model.named_parameters()
                             if p.requires_grad]

        self._task_start_params = {n: p.data.cpu().clone()
                                   for n, p in model.named_parameters()
                                   if n in self._param_names}
        self._w = {n: torch.zeros_like(p.data.cpu())
                   for n, p in model.named_parameters()
                   if n in self._param_names}

    # ── Called once at the end of each task ──────────────────────────────────

    def task_end(self, model: nn.Module):
        """
        Update consolidated importance Ω using the path integral, then
        reset checkpoint (theta_star) to current parameters.

        Matches Mammoth's end_task():
          big_omega += small_omega / ((params - checkpoint)^2 + xi)
          checkpoint = params.clone()
          small_omega = 0

        No clamping of small_omega — Mammoth accumulates the raw value.
        The denominator's xi prevents division-by-zero when params barely moved.
        """
        for n, p in model.named_parameters():
            if n not in self._param_names:
                continue
            theta_end  = p.data.cpu()
            theta_prev = self._task_start_params[n]
            denom      = (theta_end - theta_prev).pow(2) + SI_XI

            self.omega[n]      = self.omega[n] + self._w[n] / denom
            self.theta_star[n] = theta_end.clone()

    # ── Surrogate penalty gradient (called inside the training loop) ─────────

    def penalty_grad(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """
        Closed-form gradient of the SI surrogate penalty w.r.t. each param:
          ∂L_SI/∂θ_k  =  c · 2 · Ω_k · (θ_k - θ_k^*)

        Matches Mammoth's get_penalty_grads():
          return self.args.c * 2 * self.big_omega * (params - checkpoint)

        Returns a dict {param_name: grad_tensor_on_DEVICE} so the caller
        can add it directly to param.grad before optimizer.step().
        Returns an empty dict (zero contribution) before the first task ends
        (when big_omega is still None / all-zero).
        """
        grads = {}
        for n, p in model.named_parameters():
            if n not in self._param_names:
                continue
            omega_k      = self.omega[n].to(DEVICE)
            theta_star_k = self.theta_star[n].to(DEVICE)
            grads[n]     = SI_C * 2.0 * omega_k * (p.data - theta_star_k)
        return grads


# ══════════════════════════════════════════════════════════════════════════════
# SI Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_task_si(model:     nn.Module,
                      train_ds:  torch.utils.data.Dataset,  # augmented subset
                      si:        SynapticIntelligence,
                      regime:    str,
                      task_idx:  int) -> list:
    """
    One full task training with SI.

    Per-batch update (matches Mammoth's observe()):
      1. Snapshot pre-step parameters.
      2. Forward pass + CE loss backward  (CE gradients populated).
      3. Snapshot pure CE gradients BEFORE penalty injection.        ← FIX
      4. If Ω exists: add closed-form penalty gradient to param.grad.
      5. Clip all gradients to [-1, 1]  (Mammoth default).
      6. optimizer.step().
      7. Accumulate path-integral ω using snapshotted pure CE grads. ← FIX

    The critical fix vs the previous version: ω must be accumulated using
    the pure task-loss (CE) gradient only, as per Zenke et al. Eq. 3.
    Using the post-penalty-injection gradient contaminates ω with the
    regulariser's own signal, creating a destructive feedback loop that
    causes SI to degrade to near-random performance across tasks.

    Wraps each task with si.task_start() / si.task_end().
    Returns epoch_losses (list of mean CE loss per epoch).
    """
    trainable = set_trainable_params(model, regime)
    if not trainable:
        print("    [warn] no trainable params, skipping.")
        return []

    task_start_cls = task_idx * CLASSES_PER_TASK
    task_end_cls   = task_start_cls + CLASSES_PER_TASK

    loader    = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                           drop_last=False)
    optimizer = optim.Adam(trainable, lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TASK, eta_min=LR_MIN)
    early_stop = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)

    # Snapshot parameters & reset path-integral before this task
    si.task_start(model)

    model.train()
    epoch_losses = []
    epoch_stats = []

    for epoch in range(EPOCHS_PER_TASK):
        total_loss = 0.0
        n_batches  = 0
        batch_stats = []

        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Local (0-indexed within task) labels
            local_labels = labels - task_start_cls

            # ── Snapshot params before step (for ω accumulation) ─────────────
            pre_params = {n: p.data.clone()
                          for n, p in model.named_parameters()
                          if n in si._param_names}

            optimizer.zero_grad()

            # ── Forward + CE loss only ────────────────────────────────────────
            logits      = model(imgs)
            task_logits = logits[:, task_start_cls:task_end_cls]
            ce_loss     = nn.functional.cross_entropy(task_logits, local_labels)
            ce_loss.backward()

            g_task = extract_grad_vectors(model)

            # ── FIX: Snapshot PURE CE gradients before penalty injection ──────
            # ω must be accumulated using only the task-loss gradient (Eq. 3
            # in Zenke et al.). Snapshotting here, before the SI penalty is
            # added to param.grad, ensures ω is never contaminated by the
            # regulariser's own gradient signal.
            ce_grads = {n: p.grad.data.clone()
                        for n, p in model.named_parameters()
                        if n in si._param_names and p.grad is not None}

            # ── Inject SI penalty gradient into param.grad (Mammoth style) ────
            # Skipped on task 0 when omega is all-zero (no effect anyway, but
            # matches Mammoth's `if self.big_omega is not None` guard).
            penalty_grads = si.penalty_grad(model)
            
            # Build g_ret
            g_ret_vecs = []
            has_ret = False
            for n, p in model.named_parameters():
                if p.requires_grad:
                    if n in penalty_grads:
                        g_ret_vecs.append(penalty_grads[n].detach().view(-1))
                        has_ret = True
                    else:
                        g_ret_vecs.append(torch.zeros_like(p.view(-1)))
            
            if has_ret and g_ret_vecs:
                g_ret = torch.cat(g_ret_vecs)
            else:
                g_ret = torch.tensor([], device=DEVICE)
                
            for n, p in model.named_parameters():
                if n in penalty_grads and p.grad is not None:
                    p.grad.data.add_(penalty_grads[n])

            # ── Gradient clipping (Mammoth: clip_grad_value_ = 1) ─────────────
            nn.utils.clip_grad_value_(trainable, 1.0)

            optimizer.step()
            
            batch_stats.append(get_grad_stats(g_task, g_ret))

            # ── FIX: Accumulate path-integral ω using pure CE gradients ───────
            # ω_k += g_k^CE × (pre_params_k - post_params_k)
            # (equivalent to -g_k^CE × Δθ_k, matching Zenke et al. Eq. 3)
            # Using ce_grads (snapshotted before penalty injection) rather than
            # p.grad (which was modified by the penalty) is the key correction.
            for n, p in model.named_parameters():
                if n not in si._param_names:
                    continue
                if n not in ce_grads:
                    continue
                delta = pre_params[n] - p.data   # θ_prev - θ_new = -Δθ
                si._w[n] += (ce_grads[n].cpu() * delta.cpu())

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

        print(f"      epoch {epoch+1:>3}/{EPOCHS_PER_TASK}  "
              f"ce_loss={mean_epoch_loss:.4f}  lr={current_lr:.2e}")

        if USE_EARLY_STOPPING and early_stop(mean_epoch_loss, epoch=epoch + 1):
            print(f"      [early stop] no improvement for {ES_PATIENCE} epochs "
                  f"(best={early_stop.best_loss:.4f}), stopping at epoch {epoch+1}.")
            break

    # Update Ω and θ_star at the end of the task
    si.task_end(model)

    return epoch_losses, epoch_stats


# ── Single experiment run ─────────────────────────────────────────────────────

def run_experiment_si(regime: str, order_entry: dict,
                      train_tasks: list, test_tasks: list,
                      pretrained: bool = False) -> dict:
    label       = order_entry["label"]
    task_order  = order_entry["order"]
    order_named = order_entry["order_named"]

    print(f"\n  Regime  : {regime}")
    print(f"  Order   : {order_named}  ({label})")
    print(f"  SI λ    : {SI_C}   ξ : {SI_XI}")
    print(f"  Early stopping   : {'ON' if USE_EARLY_STOPPING else 'OFF'}"
          + (f"  (patience={ES_PATIENCE}, min_delta={ES_MIN_DELTA})"
             if USE_EARLY_STOPPING else ""))

    model        = build_model(pretrained=pretrained)
    si           = SynapticIntelligence(model)   # fresh SI state per run

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
        task_losses, task_stats = train_one_task_si(
            model,
            train_tasks[task_idx],
            si,
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
        "algorithm":      "SI",
        "dataset":        DATASET,
        "si_c":           SI_C,
        "si_xi":          SI_XI,
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
    print(f"Continual Learning with Synaptic Intelligence (SI) — Task Order Study  [{DATASET.upper()}]")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Dataset       : {DATASET}  ({NUM_TASKS} tasks × {CLASSES_PER_TASK} classes)")
    print(f"Regimes       : {REGIMES}")
    print(f"SI c (reg)    : {SI_C}")
    print(f"SI xi (damp)  : {SI_XI}")
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
    results_file  = SI_RESULTS_PRETRAINED if is_pretrained else SI_RESULTS_RANDOM

    print(f"Init mode     : {init_label}")
    print(f"Output        : {results_file}")

    if DATASET in ["imagenet", "imagenet100"]:
        print(f"ImageNet root : {IMAGENET_ROOT}")

    train_ds, test_ds  = load_datasets()
    train_tasks        = split_into_tasks(train_ds)
    test_tasks         = split_into_tasks(test_ds)

    total_runs  = len(REGIMES) * len(TASK_ORDERS)
    all_results = []
    run_idx     = 0
    t0          = time.time()

    for regime in REGIMES:
        for order_entry in TASK_ORDERS:
            run_idx += 1
            elapsed  = (time.time() - t0) / 60
            eta      = (elapsed / run_idx) * (total_runs - run_idx) if run_idx > 1 else 0
            print(f"\n{'='*70}")
            print(f"[SI / {init_label} / {DATASET}] [Run {run_idx:3d}/{total_runs}]  "
                  f"elapsed={elapsed:.1f}min  ETA~{eta:.0f}min")
            print("=" * 70)

            result = run_experiment_si(
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