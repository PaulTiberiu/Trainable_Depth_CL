import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms
import random
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

class HFKMNIST(Dataset):
    """
    A drop-in replacement for torchvision.datasets.KMNIST that uses the Hugging Face
    `datasets` library to download and load the KMNIST dataset, avoiding the
    built-in torchvision downloader which is currently timing out.
    """
    def __init__(self, root, train=True, transform=None, download=True):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the 'datasets' library: pip install datasets")
            
        split = "train" if train else "test"
        # tanganke/kmnist is a parquet-based mirror of KMNIST that doesn't require trust_remote_code=True
        self.dataset = load_dataset("tanganke/kmnist", split=split, cache_dir=root)
        self.transform = transform
        self.targets = torch.tensor(self.dataset['label'])
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # Convert tensor indices to integers/lists for HuggingFace Dataset
        if isinstance(idx, torch.Tensor):
            if idx.ndim == 0:
                idx = idx.item()
            else:
                idx = idx.tolist()
                
        item = self.dataset[idx]
        img = item['image'] # PIL Image
        label = item['label']
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label

class ImageNet100Folder(ImageFolder):
    def find_classes(self, directory: str):
        # Read the 100 classes from the text file
        txt_path = os.path.join(os.path.dirname(__file__), "imagenet100_CMC.txt")
        with open(txt_path, "r") as f:
            allowed_classes = [line.strip() for line in f if line.strip()]
        
        # Sort to ensure consistent class_to_idx mapping
        allowed_classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(allowed_classes)}
        return allowed_classes, class_to_idx

# ── Config ────────────────────────────────────────────────────────────────────

# ── Dataset selector ──────────────────────────────────────────────────────────
# Set to "cifar100", "mnist", "fashion_mnist", "kmnist", "qmnist", "imagenet", or "imagenet100"
DATASET = "fashion_mnist"  # Change this to switch datasets; make sure to download the dataset first by running the training script for that dataset.

# ── Per-dataset task configuration ───────────────────────────────────────────
_DATASET_CONFIG = {
    "cifar100": {
        "num_tasks":        10,
        "classes_per_task": 10,
        "num_classes":      100,
        "task_labels":      list("ABCDEFGHIJ"),
    },
    "mnist": {
        "num_tasks":        5,
        "classes_per_task": 2,
        "num_classes":      10,
        "task_labels":      ["0-1", "2-3", "4-5", "6-7", "8-9"],
        # "task_labels":      ["A", "B", "C", "D", "E"],
    },
    "fashion_mnist": {
        "num_tasks":        5,
        "classes_per_task": 2,
        "num_classes":      10,
        "task_labels":      ["0-1", "2-3", "4-5", "6-7", "8-9"],
        # "task_labels":      ["A", "B", "C", "D", "E"],
    },
    "kmnist": {
        "num_tasks":        5,
        "classes_per_task": 2,
        "num_classes":      10,
        "task_labels":      ["0-1", "2-3", "4-5", "6-7", "8-9"],
        # "task_labels":      ["A", "B", "C", "D", "E"],
    },
    "qmnist": {
        "num_tasks":        5,
        "classes_per_task": 2,
        "num_classes":      10,
        "task_labels":      ["0-1", "2-3", "4-5", "6-7", "8-9"],
        # "task_labels":      ["A", "B", "C", "D", "E"],
    },
    # Standard Split-ImageNet-1K: 10 tasks × 100 classes.
    # Classes are assigned by sorting the 1 000 synset folder names
    # alphabetically (matching torchvision.datasets.ImageFolder's default
    # behaviour) and slicing into 10 contiguous blocks — the same strategy
    # used for CIFAR-100 / MNIST in this codebase.
    # Reference: Mirzadeh et al., "Architecture Matters in Continual Learning"
    # (2022), Split ImageNet-1K with 10 tasks.
    "imagenet": {
        "num_tasks":        10,
        "classes_per_task": 100,
        "num_classes":      1000,
        "task_labels":      [f"T{i:02d}" for i in range(10)],
    },
    "imagenet100": {
        "num_tasks":        10,
        "classes_per_task": 10,
        "num_classes":      100,
        "task_labels":      [f"T{i:02d}" for i in range(10)],
    },
}

# ── Ensure results directory exists ───────────────────────────────────────────
import os
os.makedirs("results", exist_ok=True)

NUM_TASKS        = _DATASET_CONFIG[DATASET]["num_tasks"]
CLASSES_PER_TASK = _DATASET_CONFIG[DATASET]["classes_per_task"]
NUM_CLASSES      = _DATASET_CONFIG[DATASET]["num_classes"]
TASK_LABELS      = _DATASET_CONFIG[DATASET]["task_labels"]

if DATASET == "cifar100":
    EPOCHS_PER_TASK = 20
    BATCH_SIZE      = 256
    LR              = 1e-3
elif DATASET in ("mnist", "fashion_mnist", "kmnist", "qmnist"):
    EPOCHS_PER_TASK = 3
    BATCH_SIZE      = 256
    LR              = 1e-3
elif DATASET in ["imagenet", "imagenet100"]:
    # 60 epochs/task matches the standard Split-ImageNet-1K protocol.
    # Batch 256 is a common default; scale up if you have more GPUs.
    EPOCHS_PER_TASK = 60
    BATCH_SIZE      = 256
    LR              = 1e-3

DATA_ROOT    = "."
# Warning: high num_workers easily OOMs on 48GB RAM for ImageNet when the
# whole dataset structure is scanned or if multiple tasks are caching.
# We drop workers to 2 to heavily reduce multiprocessing RAM overhead.
NUM_WORKERS  = 2 if DATASET in ["imagenet", "imagenet100"] else 4
RANDOM_SEED  = 42
NUM_ORDERS   = 10

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"

PIN_MEMORY = (DEVICE == "cuda")

# ── Data Transforms ───────────────────────────────────────────────────────────

# CIFAR-100 transforms (32×32, unchanged)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# MNIST / FashionMNIST transforms
mnist_train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307),
                         (0.3081, 0.3081, 0.3081)),
])

mnist_test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307),
                         (0.3081, 0.3081, 0.3081)),
])

# ── ImageNet transforms (224×224, standard ImageNet normalisation) ────────────
#
# Training:   random resized crop + horizontal flip (standard augmentation).
# Test/store: resize shortest side to 256, then centre-crop to 224.
# These are the canonical transforms used in virtually every ImageNet paper
# and match what torchvision's pretrained models expect.

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

imagenet_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

imagenet_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Task orderings ────────────────────────────────────────────────────────────
def sample_task_orders(num_tasks: int, n_orders: int, seed: int) -> list:
    canonical = list(range(num_tasks))
    orders = [{
        "label":       "canonical",
        "order":       canonical,
        "order_named": "-".join(TASK_LABELS[i] for i in canonical),
    }]
    rng  = random.Random(seed)
    base = list(range(num_tasks))
    seen = {tuple(canonical)}
    while len(orders) - 1 < n_orders:
        perm = base[:]
        rng.shuffle(perm)
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            idx   = len(orders)
            named = "-".join(TASK_LABELS[i] for i in perm)
            orders.append({"label": f"random_{idx:02d}", "order": perm,
                           "order_named": named})
    return orders

TASK_ORDERS = sample_task_orders(NUM_TASKS, NUM_ORDERS, RANDOM_SEED)

def split_into_tasks(dataset) -> list:
    """
    Split a dataset into per-task Subsets based on the global NUM_TASKS /
    CLASSES_PER_TASK config.

    Works for CIFAR-100 (10 × 10), MNIST / FashionMNIST (5 × 2), and
    ImageNet-1K (10 × 100).

    For CIFAR / MNIST, original integer targets are used directly.
    For ImageNet (torchvision.datasets.ImageFolder), targets are also integers
    assigned in sorted-folder order, so the same contiguous-range slice logic
    applies without any changes.
    """
    targets = dataset.targets
    if isinstance(targets, list):
        targets = torch.tensor(targets)
    elif isinstance(targets, torch.Tensor):
        targets = targets.clone().detach()
        
    if targets.ndim > 1:
        # For QMNIST, targets is a 2D tensor where the first column is the class label
        targets = targets[:, 0]

    tasks   = []
    for task_id in range(NUM_TASKS):
        start   = task_id * CLASSES_PER_TASK
        end     = start + CLASSES_PER_TASK
        mask    = (targets >= start) & (targets < end)
        indices = torch.where(mask)[0]
        tasks.append(Subset(dataset, indices))
    return tasks

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Early Stopping and Scheduler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ── 1. Scheduler & early-stopping config  (add near the existing LR line) ────

# Cosine annealing: LR decays from LR → LR_MIN over EPOCHS_PER_TASK epochs.
# Each new task restarts the schedule from LR (scheduler is recreated per task).
LR_MIN = 1e-5          # floor for cosine decay; set = LR to disable scheduling

# Early stopping: halt epoch loop when training loss stops improving.
ES_PATIENCE  = 5       # epochs to wait after last improvement before stopping
ES_MIN_DELTA = 1e-4    # minimum loss improvement that counts as "progress"
#   Set ES_PATIENCE = EPOCHS_PER_TASK (or higher) to effectively disable it.


# ── EarlyStopping helper class ────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training when the monitored metric (lower-is-better, e.g. loss)
    has not improved by more than `min_delta` for `patience` consecutive epochs.

    Usage inside a training loop
    ----------------------------
        es = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)
        for epoch in range(EPOCHS_PER_TASK):
            ... train ...
            if es(mean_epoch_loss):
                break          # triggers after `patience` non-improving epochs

    The first call always returns False (no history to compare against).
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.wait       = 0            # epochs since last improvement
        self.stopped_epoch: int = 0    # epoch index when stop was triggered (0 = not yet)

    def __call__(self, loss: float, epoch: int = 0) -> bool:
        """
        Returns True when training should stop.

        Parameters
        ----------
        loss  : current epoch's mean training loss
        epoch : current epoch index (0-based); used only for logging
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait      = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False

    def reset(self):
        """Call between tasks to start fresh."""
        self.best_loss     = float("inf")
        self.wait          = 0
        self.stopped_epoch = 0

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(pretrained: bool = False) -> nn.Module:
    """
    Build backbone + head for the active dataset.

    CIFAR-100 / MNIST / FashionMNIST  →  ResNet-18, 32×32-adapted
      (conv1 = 3×3/s1, maxpool removed)

    ImageNet-1K  →  ResNet-18, standard 224×224 architecture
      (conv1 = 7×7/s2, maxpool kept — no CIFAR adaptation needed)

    In both cases:
      pretrained=False  →  random (Kaiming) init
      pretrained=True   →  ImageNet-1K pretrained weights; fc replaced with
                           a fresh Linear(feature_dim, NUM_CLASSES)
    """
    if DATASET in ["imagenet", "imagenet100"]:
        if pretrained:
            from torchvision.models import ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = resnet18(weights=None)
        # Standard architecture — no spatial adaptation required for 224×224.
        # Only replace the final classification head to match NUM_CLASSES.
        model.fc = nn.Linear(512, NUM_CLASSES)
    else:
        # CIFAR / MNIST: 32×32-adapted ResNet-18
        if pretrained:
            from torchvision.models import ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = resnet18(weights=None)
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc      = nn.Linear(512, NUM_CLASSES)

    return model.to(DEVICE)

def set_trainable_params(model: nn.Module, regime: str) -> list:
    """Freeze/unfreeze parameters per regime. Returns trainable param list."""
    for p in model.parameters():
        p.requires_grad_(False)

    if regime == "full_finetune":
        for p in model.parameters():
            p.requires_grad_(True)
    elif regime == "last_block":
        # Works for ResNet-18 (layer4)
        for p in model.layer4.parameters():
            p.requires_grad_(True)
        for p in model.fc.parameters():
            p.requires_grad_(True)
    elif regime == "last_2_blocks":
        for p in model.layer3.parameters():
            p.requires_grad_(True)
        for p in model.layer4.parameters():
            p.requires_grad_(True)
        for p in model.fc.parameters():
            p.requires_grad_(True)
    elif regime == "last_3_blocks":
        for p in model.layer2.parameters():
            p.requires_grad_(True)
        for p in model.layer3.parameters():
            p.requires_grad_(True)
        for p in model.layer4.parameters():
            p.requires_grad_(True)
        for p in model.fc.parameters():
            p.requires_grad_(True)
    elif regime == "last_6_blocks":
        # Unfreezes layer1, layer2, layer3, and layer4 
        # (This unfreezes all 4 layers of ResNet, skipping only conv1 and bn1)
        for p in model.layer1.parameters():
            p.requires_grad_(True)
        for p in model.layer2.parameters():
            p.requires_grad_(True)
        for p in model.layer3.parameters():
            p.requires_grad_(True)
        for p in model.layer4.parameters():
            p.requires_grad_(True)
        for p in model.fc.parameters():
            p.requires_grad_(True)
    elif regime == "head_only":
        for p in model.fc.parameters():
            p.requires_grad_(True)
    elif regime == "bn_affine_only":
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None: m.weight.requires_grad_(True)
                if m.bias   is not None: m.bias.requires_grad_(True)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    return [p for p in model.parameters() if p.requires_grad]

def extract_grad_vectors(model: nn.Module) -> torch.Tensor:
    """
    Extracts a flattened gradient vector for all parameters that require_grad.
    Assumes backward() has just been called and gradients are populated.
    """
    vecs = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            vecs.append(p.grad.detach().view(-1))
    if vecs:
        return torch.cat(vecs)
    # Return empty tensor on same device as model
    device = next(model.parameters()).device
    return torch.tensor([], device=device)

def get_grad_stats(g_task: torch.Tensor, g_ret: torch.Tensor) -> dict:
    """
    Computes and returns the norms of task and retention gradients,
    as well as their cosine similarity.
    """
    if g_task.numel() == 0:
        return {"norm_task": 0.0, "norm_ret": 0.0, "cos_sim": 0.0, "g2_r2_2cos": 0.0}
        
    norm_t = torch.norm(g_task).item()
    norm_r = torch.norm(g_ret).item() if g_ret.numel() > 0 else 0.0
    
    if norm_t > 0 and norm_r > 0:
        cos_sim = torch.nn.functional.cosine_similarity(g_task.unsqueeze(0), g_ret.unsqueeze(0)).item()
    else:
        cos_sim = 0.0
    
    return {
        "norm_task": round(norm_t, 6),
        "norm_ret": round(norm_r, 6),
        "cos_sim": round(cos_sim, 6),
        "g2_r2_2cos": round((norm_t**2) + (norm_r**2) + (2 * cos_sim * norm_t * norm_r), 6)
    }

@torch.no_grad()
def evaluate_task(model, subset, task_idx):
    task_start = task_idx * CLASSES_PER_TASK
    task_end   = task_start + CLASSES_PER_TASK
    loader     = DataLoader(subset, batch_size=512, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    criterion  = nn.CrossEntropyLoss()
    model.eval()

    correct = total = 0
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits       = model(images)[:, task_start:task_end]
        local_labels = labels - task_start
        total_loss  += criterion(logits, local_labels).item()
        correct     += (logits.argmax(1) == local_labels).sum().item()
        total       += labels.size(0)

    acc  = correct / total          if total       > 0 else 0.0
    loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    return round(acc, 6), round(loss, 6)

def pairwise_metrics(before: nn.Module, after: nn.Module,
                     test_tasks: list, a: int, b: int) -> dict:
    """
    Harm(a->b)     = loss_b_after  - loss_b_before   (positive = hurt b)
    Transfer(a->b) = acc_b_after   - acc_b_before     (positive = helped b)
    """
    acc_pre,  loss_pre  = evaluate_task(before, test_tasks[b], b)
    acc_post, loss_post = evaluate_task(after,  test_tasks[b], b)
    return {
        "task_a":       TASK_LABELS[a],
        "task_b":       TASK_LABELS[b],
        "harm":         round(loss_post - loss_pre, 6),
        "transfer":     round(acc_post  - acc_pre,  6),
        "acc_b_before": acc_pre,
        "acc_b_after":  acc_post,
    }