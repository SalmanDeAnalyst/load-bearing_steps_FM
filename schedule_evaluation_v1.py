"""
============================================================================
SCHEDULE DESIGN & EVALUATION FOR OT-CFM
============================================================================
Stage 4: Using ablation importance curves to design non-uniform sampling
schedules and evaluate them against uniform/log baselines.

Schedule types:
  1. Uniform    — evenly spaced timesteps (baseline)
  2. Log        — logarithmic spacing, more density near t=0
  3. Greedy     — top-K steps from class-averaged ΔLPIPS curve
  4. Class-Aware — per-class CDF-quantile schedules from ablation data

Evaluation metrics:
  - FID (Fréchet Inception Distance)
  - Precision & Recall (Kynkäänniemi et al. 2019)
  - Per-class accuracy via pretrained CIFAR-10 classifier

NFE targets: 2, 4, 8 (+ 32 as reference)

Hardware target: Quadro RTX 5000 (16 GB VRAM)
============================================================================
"""

import os
import json
import multiprocessing
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from diffusers import UNet2DModel
from tqdm import tqdm

multiprocessing.freeze_support()

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

IMAGE_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 10
GUIDANCE_SCALE = 3.0

CHECKPOINT_PATH = "./best_model.pt"
ABLATION_PATH = "./ablation_results/ablation_results.json"
OUTPUT_DIR = "./schedule_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Generation config ────────────────────────────────────
SAMPLES_PER_CLASS = 100    # 100 × 10 = 1000 total per schedule
MINI_BATCH = 50            # batch size for generation (CFG doubles it)
NFE_LIST = [2, 4, 8, 32]  # 32 = reference

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# =========================================================
# MODEL (must match training exactly)
# =========================================================
class ClassConditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=IMAGE_SIZE,
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 256),
            down_block_types=(
                "DownBlock2D", "AttnDownBlock2D",
                "DownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "UpBlock2D",
                "AttnUpBlock2D", "UpBlock2D",
            ),
            class_embed_type="identity",
        )
        self.class_emb = nn.Embedding(NUM_CLASSES + 1, 128 * 4)

    def forward(self, x, t, class_labels):
        emb = self.class_emb(class_labels)
        return self.unet(x, t, class_labels=emb).sample


# =========================================================
# CFG VELOCITY
# =========================================================
@torch.no_grad()
def cfg_velocity(model, x, t_scalar, labels, guidance_scale=GUIDANCE_SCALE):
    B = x.shape[0]
    t_batch = torch.full((B,), t_scalar, device=DEVICE)
    null_labels = torch.full_like(labels, NUM_CLASSES)

    combined_x = torch.cat([x, x], dim=0)
    combined_t = torch.cat([t_batch, t_batch], dim=0)
    combined_labels = torch.cat([labels, null_labels], dim=0)

    v_all = model(combined_x, combined_t, combined_labels)
    v_cond, v_uncond = torch.chunk(v_all, 2, dim=0)

    return v_uncond + guidance_scale * (v_cond - v_uncond)


# =========================================================
# EULER SOLVER WITH ARBITRARY SCHEDULE
# =========================================================
@torch.no_grad()
def euler_sample_with_schedule(model, x0, labels, t_schedule):
    """
    Euler ODE solver using a non-uniform time schedule.

    Args:
        model: trained ClassConditionalUNet
        x0: [B, C, H, W] initial noise
        labels: [B] class labels
        t_schedule: list of floats [t_0, t_1, ..., t_N] where t_0=0, t_N=1
                    len(t_schedule) = NFE + 1

    Returns:
        x_final: [B, C, H, W] generated images
    """
    x = x0.clone()

    for i in range(len(t_schedule) - 1):
        t_curr = t_schedule[i]
        t_next = t_schedule[i + 1]
        dt = t_next - t_curr

        v = cfg_velocity(model, x, t_curr, labels)
        x = x + v * dt

    return x


# =========================================================
# SCHEDULE CONSTRUCTORS
# =========================================================
def build_uniform_schedule(nfe):
    """Evenly spaced: [0, 1/nfe, 2/nfe, ..., 1]"""
    return np.linspace(0, 1, nfe + 1).tolist()


def build_log_schedule(nfe, eps=1e-3):
    """
    Logarithmic spacing — more steps near t=0.
    Maps uniform points through inverse-log to concentrate early.
    Similar to DPM-Solver style schedules.
    """
    # Uniform in log-space between eps and 1
    log_points = np.exp(np.linspace(np.log(eps), np.log(1.0), nfe + 1))
    # Shift so first point is 0
    log_points = log_points - log_points[0]
    # Normalize to [0, 1]
    log_points = log_points / log_points[-1]
    return log_points.tolist()


def build_greedy_schedule(nfe, ablation_data):
    """
    Pick top-K steps from class-averaged ΔLPIPS importance curve.
    Always includes t=0 and t=1.
    """
    classes = list(ablation_data["zero"].keys())
    all_curves = np.array([
        ablation_data["zero"][cls]["lpips"] for cls in classes
    ])
    mean_curve = all_curves.mean(axis=0)  # [32]

    # Map step indices to t values (32-step grid)
    t_32 = np.linspace(0, 1, 33)  # 33 boundaries

    # Select top-(nfe-1) steps by importance, always keep step 0
    # Step 0 maps to t=0, and we always add t=1 at the end
    ranked_steps = np.argsort(mean_curve)[::-1]

    selected_steps = set()
    selected_steps.add(0)  # always start at t=0
    for s in ranked_steps:
        if len(selected_steps) >= nfe:
            break
        selected_steps.add(s)

    # Convert to sorted t values
    selected_t = sorted([t_32[s] for s in selected_steps])
    selected_t.append(1.0)  # always end at t=1

    return selected_t


def build_classaware_schedules(nfe, ablation_data):
    """
    Per-class CDF-quantile schedules.
    Returns dict: {class_idx: schedule}
    """
    classes = list(ablation_data["zero"].keys())
    t_32 = np.linspace(0, 1, 33)
    schedules = {}

    for cls_idx, cls_name in enumerate(classes):
        curve = np.array(ablation_data["zero"][cls_name]["lpips"])

        if curve.sum() == 0:
            # Fallback to uniform
            schedules[cls_idx] = build_uniform_schedule(nfe)
            continue

        cdf = np.cumsum(curve) / curve.sum()

        # Pick nfe evenly-spaced quantiles from the CDF
        quantiles = np.linspace(0, 1, nfe + 1)[:-1]  # [0, 1/nfe, ..., (nfe-1)/nfe]
        selected_t = []
        for q in quantiles:
            idx = np.searchsorted(cdf, q)
            idx = min(idx, 31)
            selected_t.append(t_32[idx])
        selected_t.append(1.0)

        schedules[cls_idx] = selected_t

    return schedules


# =========================================================
# METRICS
# =========================================================

# ── FID ──────────────────────────────────────────────────
class InceptionFeatureExtractor:
    def __init__(self):
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights).to(DEVICE).eval()
        self.model.fc = nn.Identity()

    @torch.no_grad()
    def extract(self, images, batch_size=50):
        """
        images: [N, 3, 32, 32] in [-1, 1]
        Returns: [N, 2048]
        """
        all_feats = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            # Resize to 299×299 and rescale to [0, 1]
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear",
                                  align_corners=False, antialias=True)
            batch = (batch + 1) / 2
            feats = self.model(batch)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)


def compute_fid(feats1, feats2):
    """Compute FID between two feature sets."""
    f1 = feats1.numpy()
    f2 = feats2.numpy()

    mu1, sigma1 = f1.mean(axis=0), np.cov(f1, rowvar=False)
    mu2, sigma2 = f2.mean(axis=0), np.cov(f2, rowvar=False)

    diff = mu1 - mu2

    # Stable matrix sqrt via eigendecomposition
    product = sigma1 @ sigma2
    eigvals, eigvecs = np.linalg.eigh(product)
    eigvals = np.maximum(eigvals, 0)
    sqrt_product = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * sqrt_product)
    return float(np.real(fid))


# ── Precision & Recall (Kynkäänniemi et al. 2019) ────────
def compute_precision_recall(real_feats, gen_feats, k=3):
    """
    Improved Precision and Recall using k-NN manifold estimation.

    Precision: fraction of generated samples that fall within
               the real data manifold.
    Recall:    fraction of real samples that fall within the
               generated data manifold.
    """
    real_np = real_feats.numpy()
    gen_np = gen_feats.numpy()

    def get_kth_nn_distance(feats, k):
        """For each sample, get distance to k-th nearest neighbor."""
        # Pairwise L2 distances
        dists = np.sqrt(
            np.sum(feats ** 2, axis=1, keepdims=True)
            - 2 * feats @ feats.T
            + np.sum(feats ** 2, axis=1, keepdims=True).T
            + 1e-10  # numerical stability
        )
        # Set self-distance to inf
        np.fill_diagonal(dists, np.inf)
        # k-th nearest neighbor distance (0-indexed so k-1)
        kth_dists = np.sort(dists, axis=1)[:, k - 1]
        return kth_dists

    def cross_distances(feats_a, feats_b):
        """Pairwise distances from each sample in A to all in B."""
        return np.sqrt(
            np.sum(feats_a ** 2, axis=1, keepdims=True)
            - 2 * feats_a @ feats_b.T
            + np.sum(feats_b ** 2, axis=1, keepdims=True).T
            + 1e-10
        )

    # Real manifold: k-NN radius per real sample
    real_kth = get_kth_nn_distance(real_np, k)

    # Gen manifold: k-NN radius per gen sample
    gen_kth = get_kth_nn_distance(gen_np, k)

    # Precision: for each gen sample, is min distance to real < real's k-NN radius?
    gen_to_real = cross_distances(gen_np, real_np)
    precision = np.mean(
        np.any(gen_to_real <= real_kth[np.newaxis, :], axis=1)
    )

    # Recall: for each real sample, is min distance to gen < gen's k-NN radius?
    real_to_gen = cross_distances(real_np, gen_np)
    recall = np.mean(
        np.any(real_to_gen <= gen_kth[np.newaxis, :], axis=1)
    )

    return float(precision), float(recall)


# ── Per-class accuracy via simple CNN classifier ─────────
class CIFAR10Classifier(nn.Module):
    """
    Simple CNN classifier trained on CIFAR-10.
    We train this quickly on real CIFAR-10 to evaluate
    whether generated images are class-consistent.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_cifar10_classifier(epochs=15):
    """Train a quick CIFAR-10 classifier for evaluation."""
    classifier_path = "./cifar10_classifier.pt"

    if os.path.exists(classifier_path):
        print("  Loading cached classifier...")
        model = CIFAR10Classifier().to(DEVICE)
        model.load_state_dict(
            torch.load(classifier_path, map_location=DEVICE, weights_only=True)
        )
        model.eval()
        return model

    print("  Training CIFAR-10 classifier (one-time)...")

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10("./data", train=True, download=True,
                                  transform=transform_train)
    test_set = datasets.CIFAR10("./data", train=False, download=True,
                                 transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = CIFAR10Classifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            loss = F.cross_entropy(model(imgs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"  Classifier test accuracy: {acc:.1f}%")

    torch.save(model.state_dict(), classifier_path)
    model.eval()
    return model


@torch.no_grad()
def evaluate_class_accuracy(classifier, images, labels):
    """
    Evaluate per-class accuracy of generated images.
    images: [N, 3, 32, 32] in [-1, 1]
    labels: [N] intended class labels
    Returns: overall accuracy, per-class accuracy dict
    """
    all_preds = []
    for start in range(0, len(images), 100):
        batch = images[start:start + 100].to(DEVICE)
        preds = classifier(batch).argmax(dim=1)
        all_preds.append(preds.cpu())
    all_preds = torch.cat(all_preds)

    overall_acc = (all_preds == labels).float().mean().item()

    per_class_acc = {}
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc[CIFAR10_CLASSES[c]] = (
                (all_preds[mask] == c).float().mean().item()
            )
    return overall_acc, per_class_acc


# =========================================================
# GENERATION ENGINE
# =========================================================
@torch.no_grad()
def generate_samples(model, schedule_name, nfe, schedule_or_schedules,
                     labels, x0, is_classaware=False):
    """
    Generate samples using a given schedule.

    Args:
        model: trained flow matching model
        schedule_name: str name for logging
        nfe: number of function evaluations
        schedule_or_schedules: either a single schedule list, or
            dict {class_idx: schedule} for class-aware
        labels: [N] class labels
        x0: [N, C, H, W] initial noise
        is_classaware: if True, schedule_or_schedules is a dict

    Returns:
        images: [N, C, H, W] generated images on CPU
    """
    total = len(labels)
    all_images = []

    for start in range(0, total, MINI_BATCH):
        end = min(start + MINI_BATCH, total)
        x0_batch = x0[start:end].to(DEVICE)
        labels_batch = labels[start:end].to(DEVICE)

        if is_classaware:
            # Class-aware: group samples by class within this batch
            # For simplicity, process each class group separately
            batch_images = torch.zeros_like(x0_batch)
            for c in range(NUM_CLASSES):
                mask = labels_batch == c
                if mask.sum() == 0:
                    continue
                schedule = schedule_or_schedules[c]
                x0_c = x0_batch[mask]
                labels_c = labels_batch[mask]
                imgs_c = euler_sample_with_schedule(
                    model, x0_c, labels_c, schedule
                )
                batch_images[mask] = imgs_c
            all_images.append(batch_images.cpu())
        else:
            imgs = euler_sample_with_schedule(
                model, x0_batch, labels_batch, schedule_or_schedules
            )
            all_images.append(imgs.cpu())

    return torch.cat(all_images, dim=0)


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 70)
    print("SCHEDULE DESIGN & EVALUATION")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Total per schedule: {SAMPLES_PER_CLASS * NUM_CLASSES}")
    print(f"NFE targets: {NFE_LIST}")
    print(f"Schedules: Uniform, Log, Greedy, Class-Aware")
    print("=" * 70)

    # ─────────────────────────────────────────────────────
    # 1. LOAD MODEL
    # ─────────────────────────────────────────────────────
    print("\n[1/7] Loading model...")
    model = ClassConditionalUNet().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded: {CHECKPOINT_PATH}")

    # ─────────────────────────────────────────────────────
    # 2. LOAD ABLATION DATA
    # ─────────────────────────────────────────────────────
    print("\n[2/7] Loading ablation data...")
    with open(ABLATION_PATH) as f:
        ablation_data = json.load(f)
    print(f"  Loaded: {ABLATION_PATH}")

    # ─────────────────────────────────────────────────────
    # 3. BUILD ALL SCHEDULES
    # ─────────────────────────────────────────────────────
    print("\n[3/7] Building schedules...")

    all_schedules = {}  # key: (schedule_name, nfe) → schedule or dict

    for nfe in NFE_LIST:
        # Uniform
        s = build_uniform_schedule(nfe)
        all_schedules[("uniform", nfe)] = {"schedule": s, "classaware": False}
        print(f"  Uniform  NFE={nfe}: {[round(t, 4) for t in s]}")

        # Log
        s = build_log_schedule(nfe)
        all_schedules[("log", nfe)] = {"schedule": s, "classaware": False}
        print(f"  Log      NFE={nfe}: {[round(t, 4) for t in s]}")

        # Greedy (from ablation)
        s = build_greedy_schedule(nfe, ablation_data)
        all_schedules[("greedy", nfe)] = {"schedule": s, "classaware": False}
        print(f"  Greedy   NFE={nfe}: {[round(t, 4) for t in s]}")

        # Class-Aware (from ablation)
        ca_schedules = build_classaware_schedules(nfe, ablation_data)
        all_schedules[("classaware", nfe)] = {
            "schedule": ca_schedules, "classaware": True
        }
        print(f"  ClassAw  NFE={nfe}: (per-class, see details below)")
        for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
            s = ca_schedules[cls_idx]
            print(f"           {cls_name:>12s}: {[round(t, 4) for t in s]}")

        print()

    # ─────────────────────────────────────────────────────
    # 4. PREPARE FIXED NOISE AND LABELS
    # ─────────────────────────────────────────────────────
    print("[4/7] Preparing fixed noise and labels...")
    total_samples = SAMPLES_PER_CLASS * NUM_CLASSES

    labels = torch.arange(NUM_CLASSES).repeat_interleave(SAMPLES_PER_CLASS)
    torch.manual_seed(123)  # different seed from ablation for independence
    x0_all = torch.randn(total_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    print(f"  Labels: {labels.shape}, Noise: {x0_all.shape}")

    # ─────────────────────────────────────────────────────
    # 5. INITIALIZE METRICS
    # ─────────────────────────────────────────────────────
    print("\n[5/7] Initializing metrics...")

    inception = InceptionFeatureExtractor()
    print("  InceptionV3 feature extractor ready.")

    classifier = train_cifar10_classifier()
    print("  CIFAR-10 classifier ready.")

    # Get real CIFAR-10 features for FID / Precision-Recall
    print("  Computing real CIFAR-10 features...")
    real_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    real_dataset = datasets.CIFAR10("./data", train=False, download=True,
                                     transform=real_transform)
    real_loader = torch.utils.data.DataLoader(
        real_dataset, batch_size=100, shuffle=False,
        num_workers=4, pin_memory=True
    )
    real_images = []
    for imgs, _ in real_loader:
        real_images.append(imgs)
    real_images = torch.cat(real_images, dim=0)  # [10000, 3, 32, 32]

    real_feats = inception.extract(real_images.to(DEVICE))
    print(f"  Real features: {real_feats.shape}")

    # ─────────────────────────────────────────────────────
    # 6. RUN ALL SCHEDULES AND EVALUATE
    # ─────────────────────────────────────────────────────
    print("\n[6/7] Generating and evaluating...")

    results = {}

    schedule_names = ["uniform", "log", "greedy", "classaware"]
    total_runs = len(schedule_names) * len(NFE_LIST)
    run_idx = 0

    for sched_name in schedule_names:
        for nfe in NFE_LIST:
            run_idx += 1
            key = (sched_name, nfe)
            entry = all_schedules[key]

            print(f"\n  [{run_idx}/{total_runs}] {sched_name} NFE={nfe}")
            print(f"    Generating {total_samples} samples...")

            images = generate_samples(
                model, sched_name, nfe,
                entry["schedule"], labels, x0_all,
                is_classaware=entry["classaware"]
            )

            # ── Compute metrics ───────────────────────
            print(f"    Computing Inception features...")
            gen_feats = inception.extract(images.to(DEVICE))

            print(f"    Computing FID...")
            fid = compute_fid(real_feats, gen_feats)

            print(f"    Computing Precision & Recall...")
            # Use a subset for P&R (k-NN is O(n²))
            n_pr = min(1000, len(gen_feats))
            real_subset = real_feats[:n_pr]
            gen_subset = gen_feats[:n_pr]
            precision, recall = compute_precision_recall(
                real_subset, gen_subset, k=3
            )

            print(f"    Computing per-class accuracy...")
            overall_acc, per_class_acc = evaluate_class_accuracy(
                classifier, images, labels
            )

            results[f"{sched_name}_nfe{nfe}"] = {
                "schedule_name": sched_name,
                "nfe": nfe,
                "fid": fid,
                "precision": precision,
                "recall": recall,
                "overall_accuracy": overall_acc,
                "per_class_accuracy": per_class_acc,
            }

            print(f"    FID={fid:.2f} | P={precision:.3f} | "
                  f"R={recall:.3f} | Acc={overall_acc:.3f}")

            # Free memory
            del images, gen_feats
            torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────
    # 7. SAVE RESULTS AND PLOT
    # ─────────────────────────────────────────────────────
    print("\n[7/7] Saving results and generating plots...")

    results_path = os.path.join(OUTPUT_DIR, "schedule_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    generate_plots(results)

    # ── Summary table ─────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Schedule':<15} {'NFE':>4} {'FID':>8} {'Prec':>7} {'Rec':>7} {'Acc':>7}")
    print("-" * 80)
    for key in sorted(results.keys(),
                       key=lambda k: (results[k]["nfe"], results[k]["schedule_name"])):
        r = results[key]
        print(f"{r['schedule_name']:<15} {r['nfe']:>4} "
              f"{r['fid']:>8.2f} {r['precision']:>7.3f} "
              f"{r['recall']:>7.3f} {r['overall_accuracy']:>7.3f}")
    print("=" * 80)

    # ── Per-class accuracy table ──────────────────────
    print("\n=== Per-Class Accuracy (NFE=4) ===")
    print(f"{'Class':<15}", end="")
    for sn in schedule_names:
        print(f"{sn:>12}", end="")
    print()
    print("-" * (15 + 12 * len(schedule_names)))
    for cls in CIFAR10_CLASSES:
        print(f"{cls:<15}", end="")
        for sn in schedule_names:
            key = f"{sn}_nfe4"
            if key in results:
                acc = results[key]["per_class_accuracy"].get(cls, 0)
                print(f"{acc:>12.3f}", end="")
        print()

    print("\nDONE.")


# =========================================================
# PLOTTING
# =========================================================
def generate_plots(results):
    """Generate comparison plots across schedules and NFEs."""

    schedule_names = ["uniform", "log", "greedy", "classaware"]
    colors = {
        "uniform": "#1f77b4",
        "log": "#ff7f0e",
        "greedy": "#2ca02c",
        "classaware": "#d62728",
    }
    markers = {
        "uniform": "o",
        "log": "s",
        "greedy": "^",
        "classaware": "D",
    }

    # ── PLOT 1: FID vs NFE ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("fid", "FID ↓", False),
        ("precision", "Precision ↑", True),
        ("recall", "Recall ↑", True),
    ]

    for ax, (metric, ylabel, higher_better) in zip(axes, metrics):
        for sn in schedule_names:
            xs, ys = [], []
            for nfe in NFE_LIST:
                key = f"{sn}_nfe{nfe}"
                if key in results:
                    xs.append(nfe)
                    ys.append(results[key][metric])
            ax.plot(xs, ys, color=colors[sn], marker=markers[sn],
                    linewidth=2, markersize=8, label=sn)

        ax.set_xlabel("NFE (Number of Function Evaluations)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(NFE_LIST)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Schedule Comparison: FID / Precision / Recall vs NFE",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "metrics_vs_nfe.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── PLOT 2: Per-class accuracy heatmap at NFE=4 ───
    fig, axes = plt.subplots(1, len(schedule_names), figsize=(20, 4))
    for ax_idx, sn in enumerate(schedule_names):
        ax = axes[ax_idx]
        accs_per_nfe = []
        for nfe in NFE_LIST:
            key = f"{sn}_nfe{nfe}"
            if key in results:
                row = [results[key]["per_class_accuracy"].get(cls, 0)
                       for cls in CIFAR10_CLASSES]
                accs_per_nfe.append(row)

        if accs_per_nfe:
            matrix = np.array(accs_per_nfe).T  # [10 classes, N nfe]
            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                           vmin=0, vmax=1, interpolation="nearest")
            ax.set_yticks(range(NUM_CLASSES))
            ax.set_yticklabels(CIFAR10_CLASSES, fontsize=9)
            ax.set_xticks(range(len(NFE_LIST)))
            ax.set_xticklabels([str(n) for n in NFE_LIST], fontsize=10)
            ax.set_xlabel("NFE", fontsize=10)
            ax.set_title(sn, fontsize=12, fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Per-Class Accuracy: Schedule × NFE",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "perclass_accuracy_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── PLOT 3: Overall accuracy bar chart ────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.18
    x = np.arange(len(NFE_LIST))

    for i, sn in enumerate(schedule_names):
        accs = []
        for nfe in NFE_LIST:
            key = f"{sn}_nfe{nfe}"
            accs.append(results[key]["overall_accuracy"] if key in results else 0)
        ax.bar(x + i * bar_width, accs, bar_width,
               color=colors[sn], label=sn, alpha=0.85)

    ax.set_xlabel("NFE", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title("Overall Classification Accuracy by Schedule",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels([str(n) for n in NFE_LIST])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "overall_accuracy_bars.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── PLOT 4: FID bar chart grouped by NFE ──────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, sn in enumerate(schedule_names):
        fids = []
        for nfe in NFE_LIST:
            key = f"{sn}_nfe{nfe}"
            fids.append(results[key]["fid"] if key in results else 0)
        ax.bar(x + i * bar_width, fids, bar_width,
               color=colors[sn], label=sn, alpha=0.85)

    ax.set_xlabel("NFE", fontsize=12)
    ax.set_ylabel("FID ↓", fontsize=12)
    ax.set_title("FID by Schedule and NFE",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels([str(n) for n in NFE_LIST])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fid_bars.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── PLOT 5: Schedule visualization ────────────────
    fig, axes = plt.subplots(len(NFE_LIST), 1, figsize=(12, 3 * len(NFE_LIST)))
    for ax_idx, nfe in enumerate(NFE_LIST):
        ax = axes[ax_idx]
        for i, sn in enumerate(schedule_names):
            key = (sn, nfe)
            entry_key = f"{sn}_nfe{nfe}"

            if sn == "classaware":
                # Show mean of class-aware schedules
                ca_key = ("classaware", nfe)
                # Just plot the first class as representative
                schedule = list(build_classaware_schedules(
                    nfe, json.load(open(ABLATION_PATH))
                ).values())[0]
            else:
                if sn == "uniform":
                    schedule = build_uniform_schedule(nfe)
                elif sn == "log":
                    schedule = build_log_schedule(nfe)
                else:
                    schedule = build_greedy_schedule(
                        nfe, json.load(open(ABLATION_PATH))
                    )

            y_pos = i * 0.3
            ax.scatter(schedule, [y_pos] * len(schedule),
                       color=colors[sn], s=80, zorder=5,
                       marker=markers[sn], label=sn if ax_idx == 0 else None)
            ax.hlines(y_pos, 0, 1, colors=colors[sn], alpha=0.3, linewidth=1)

        ax.set_yticks([i * 0.3 for i in range(len(schedule_names))])
        ax.set_yticklabels(schedule_names, fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Time t", fontsize=10)
        ax.set_title(f"NFE = {nfe}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")

    if axes[0].get_legend_handles_labels()[1]:
        axes[0].legend(loc="upper right", fontsize=9)

    fig.suptitle("Schedule Visualization: Timestep Placement",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "schedule_visualization.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    main()