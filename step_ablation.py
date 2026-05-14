"""
============================================================================
STEP ABLATION ANALYSIS FOR OT-CFM
============================================================================
Load-Bearing Steps in Flow Matching: Per-Class Analysis

This script implements the core experiment from the project proposal:
  - Generate baseline samples at 32 NFE (full trajectory saved)
  - At each step k, perturb the velocity prediction 3 ways:
      1. Zero-out:    v_k = 0
      2. Gaussian:    v_k = N(0, ||v_k||)  (norm-matched noise)
      3. Class-swap:  v_k = v_theta(x_t, t, c')  where c' != c
  - Measure ΔLPIPS and ΔFID per step per class
  - Output per-class importance curves

Hardware target: Quadro RTX 5000 (16 GB VRAM)
Estimated runtime: ~2.5-3 hours for 300 samples (30/class × 10 classes)
============================================================================
"""

import os
import json
import multiprocessing
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from diffusers import UNet2DModel
from tqdm import tqdm

# ── LPIPS import (pip install lpips) ──────────────────────
import lpips

multiprocessing.freeze_support()

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Model architecture constants (must match training) ───
IMAGE_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 10
GUIDANCE_SCALE = 3.0
NFE = 32  # number of Euler steps

# ── Ablation config ──────────────────────────────────────
SAMPLES_PER_CLASS = 30         # 30 × 10 = 300 total
CHECKPOINT_PATH = "./best_model.pt"  # <-- adjust if needed
OUTPUT_DIR = "./ablation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# =========================================================
# MODEL DEFINITION (must match training exactly)
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
# CFG VELOCITY FUNCTION (manual, not wrapped for odeint)
# =========================================================
@torch.no_grad()
def cfg_velocity(model, x, t_scalar, labels, guidance_scale=GUIDANCE_SCALE):
    """
    Compute CFG velocity at a single timestep.

    Args:
        model: trained ClassConditionalUNet
        x: [B, C, H, W] current state
        t_scalar: scalar float, current time in [0, 1]
        labels: [B] integer class labels
        guidance_scale: CFG weight w
    Returns:
        v_cfg: [B, C, H, W] guided velocity
    """
    B = x.shape[0]
    t_batch = torch.full((B,), t_scalar, device=DEVICE)
    null_labels = torch.full_like(labels, NUM_CLASSES)

    # Batched forward: conditional + unconditional together
    combined_x = torch.cat([x, x], dim=0)
    combined_t = torch.cat([t_batch, t_batch], dim=0)
    combined_labels = torch.cat([labels, null_labels], dim=0)

    v_all = model(combined_x, combined_t, combined_labels)
    v_cond, v_uncond = torch.chunk(v_all, 2, dim=0)

    return v_uncond + guidance_scale * (v_cond - v_uncond)


# =========================================================
# MANUAL EULER ODE SOLVER WITH PER-STEP INTERVENTION
# =========================================================
@torch.no_grad()
def euler_sample_with_perturbation(
    model, x0, labels, nfe=NFE,
    perturb_step=None, perturb_type=None,
    swap_labels=None, save_trajectory=False
):
    """
    Manual Euler solver that allows perturbation at a specific step.

    Args:
        model: trained model
        x0: [B, C, H, W] initial noise
        labels: [B] class labels
        nfe: number of Euler steps
        perturb_step: int or None; which step index to perturb (0-indexed)
        perturb_type: 'zero' | 'gaussian' | 'class_swap' | None
        swap_labels: [B] alternative class labels (for class_swap)
        save_trajectory: if True, return all intermediate states

    Returns:
        x_final: [B, C, H, W] final generated image
        trajectory: list of [B,C,H,W] tensors if save_trajectory else None
    """
    dt = 1.0 / nfe
    t_values = torch.linspace(0, 1, nfe + 1, device=DEVICE)
    x = x0.clone()
    trajectory = [x.clone().cpu()] if save_trajectory else None

    for k in range(nfe):
        t_k = t_values[k].item()

        if k == perturb_step and perturb_type is not None:
            # ── PERTURBATION AT STEP k ────────────────
            if perturb_type == "zero":
                # Zero-out: no velocity update, sample just coasts
                v = torch.zeros_like(x)

            elif perturb_type == "gaussian":
                # Gaussian: compute real velocity first to match its norm,
                # then replace with norm-matched random noise
                v_real = cfg_velocity(model, x, t_k, labels)
                # Per-sample norm matching
                real_norm = v_real.view(v_real.shape[0], -1).norm(dim=1)
                noise = torch.randn_like(x)
                noise_norm = noise.view(noise.shape[0], -1).norm(dim=1)
                # Scale noise to match real velocity norm
                scale = (real_norm / (noise_norm + 1e-8)).view(-1, 1, 1, 1)
                v = noise * scale

            elif perturb_type == "class_swap":
                # Class swap: use a different class's velocity
                assert swap_labels is not None
                v = cfg_velocity(model, x, t_k, swap_labels)

            else:
                raise ValueError(f"Unknown perturb_type: {perturb_type}")
        else:
            # ── NORMAL STEP ───────────────────────────
            v = cfg_velocity(model, x, t_k, labels)

        x = x + v * dt

        if save_trajectory:
            trajectory.append(x.clone().cpu())

    return x, trajectory


# =========================================================
# METRIC: LPIPS (per-image perceptual distance)
# =========================================================
class LPIPSMetric:
    def __init__(self):
        self.fn = lpips.LPIPS(net="alex").to(DEVICE).eval()

    @torch.no_grad()
    def compute(self, img1, img2):
        """
        Compute per-sample LPIPS between two batches.
        Both should be [B, C, H, W] in [-1, 1].
        Returns: [B] tensor of LPIPS distances.
        """
        return self.fn(img1, img2).squeeze()


# =========================================================
# METRIC: FID (lightweight, per-class)
# =========================================================
class FIDMetric:
    """
    Lightweight FID computation using Inception v3 features.
    For small sample counts, this gives a rough but directional signal.
    """
    def __init__(self):
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights).to(DEVICE).eval()
        # Remove the final FC layer to get 2048-d features
        self.model.fc = nn.Identity()
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299), antialias=True),
        ])

    @torch.no_grad()
    def get_features(self, images):
        """
        Extract Inception features from a batch of images.
        images: [B, 3, 32, 32] in [-1, 1]
        Returns: [B, 2048] feature vectors
        """
        # Resize from 32×32 to 299×299
        imgs = self.preprocess(images)
        # Inception expects [0, 1] roughly — renormalize from [-1,1]
        imgs = (imgs + 1) / 2
        feats = self.model(imgs)
        return feats

    @staticmethod
    def compute_fid(feats1, feats2):
        """
        Compute FID between two sets of features.
        Uses numpy for matrix sqrt stability.
        """
        f1 = feats1.cpu().numpy()
        f2 = feats2.cpu().numpy()

        mu1, sigma1 = f1.mean(axis=0), np.cov(f1, rowvar=False)
        mu2, sigma2 = f2.mean(axis=0), np.cov(f2, rowvar=False)

        diff = mu1 - mu2
        # Stable matrix sqrt via eigendecomposition
        covmean, _ = _sqrtm_stable(sigma1 @ sigma2)

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(np.real(fid))


def _sqrtm_stable(M):
    """Stable matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0)  # clip negative eigenvalues
    sqrt_M = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return sqrt_M, None


# =========================================================
# GENERATE SWAP LABELS (for class_swap perturbation)
# =========================================================
def generate_swap_labels(labels):
    """For each sample, pick a uniformly random *different* class."""
    swap = labels.clone()
    for i in range(len(swap)):
        candidates = [c for c in range(NUM_CLASSES) if c != labels[i].item()]
        swap[i] = torch.tensor(
            candidates[torch.randint(0, len(candidates), (1,)).item()],
            device=DEVICE
        )
    return swap


# =========================================================
# MAIN ABLATION EXPERIMENT
# =========================================================
def main():
    print("=" * 70)
    print("STEP ABLATION ANALYSIS")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Total samples: {SAMPLES_PER_CLASS * NUM_CLASSES}")
    print(f"NFE: {NFE} | Perturbation types: 3 (zero, gaussian, class_swap)")
    print(f"Total ODE solves: {1 + 3 * NFE} per sample = "
          f"{(1 + 3 * NFE) * SAMPLES_PER_CLASS * NUM_CLASSES} total")
    print("=" * 70)

    # ─────────────────────────────────────────────────────
    # 1. LOAD MODEL
    # ─────────────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    model = ClassConditionalUNet().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded checkpoint: {CHECKPOINT_PATH}")

    # ─────────────────────────────────────────────────────
    # 2. INITIALIZE METRICS
    # ─────────────────────────────────────────────────────
    print("\n[2/5] Initializing metrics...")
    lpips_metric = LPIPSMetric()
    fid_metric = FIDMetric()
    print("  LPIPS (AlexNet) and FID (InceptionV3) ready.")

    # ─────────────────────────────────────────────────────
    # 3. PREPARE FIXED NOISE AND LABELS
    # ─────────────────────────────────────────────────────
    print("\n[3/5] Preparing fixed noise and labels...")
    total_samples = SAMPLES_PER_CLASS * NUM_CLASSES

    # Create labels: [0,0,...,0, 1,1,...,1, ..., 9,9,...,9]
    labels = torch.arange(NUM_CLASSES, device=DEVICE).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    # Fixed initial noise (for reproducibility)
    torch.manual_seed(42)
    x0_all = torch.randn(
        total_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE
    )
    # Pre-generate swap labels for class_swap perturbation
    swap_labels_all = generate_swap_labels(labels)

    print(f"  Labels shape: {labels.shape}")
    print(f"  Noise shape:  {x0_all.shape}")

    # ─────────────────────────────────────────────────────
    # 4. GENERATE BASELINE (UNPERTURBED) SAMPLES
    # ─────────────────────────────────────────────────────
    print("\n[4/5] Generating baseline samples (unperturbed)...")

    # Process in mini-batches to fit in VRAM
    # CFG doubles the batch, so effective batch = 2 * MINI_BATCH
    MINI_BATCH = 30  # 30 samples → 60 effective with CFG → ~3GB

    baseline_images = []
    for start in tqdm(range(0, total_samples, MINI_BATCH), desc="Baseline"):
        end = min(start + MINI_BATCH, total_samples)
        x0_batch = x0_all[start:end]
        labels_batch = labels[start:end]

        x_final, _ = euler_sample_with_perturbation(
            model, x0_batch, labels_batch,
            perturb_step=None, perturb_type=None
        )
        baseline_images.append(x_final)

    baseline_images = torch.cat(baseline_images, dim=0)  # [300, 3, 32, 32]
    print(f"  Baseline shape: {baseline_images.shape}")

    # Extract Inception features for baseline (for FID)
    print("  Extracting baseline Inception features...")
    baseline_feats_list = []
    for start in range(0, total_samples, MINI_BATCH):
        end = min(start + MINI_BATCH, total_samples)
        feats = fid_metric.get_features(baseline_images[start:end])
        baseline_feats_list.append(feats)
    baseline_feats = torch.cat(baseline_feats_list, dim=0)  # [300, 2048]

    # ─────────────────────────────────────────────────────
    # 5. RUN PERTURBATION SWEEP
    # ─────────────────────────────────────────────────────
    print("\n[5/5] Running perturbation sweep...")
    print(f"  3 perturbation types × {NFE} steps × {total_samples} samples")

    perturbation_types = ["zero", "gaussian", "class_swap"]

    # Results structure:
    # results[perturb_type][class_id] = {
    #     "lpips": [nfe-length list of mean ΔLPIPS],
    #     "fid":   [nfe-length list of ΔFID]
    # }
    results = {
        pt: {c: {"lpips": [], "fid": []} for c in range(NUM_CLASSES)}
        for pt in perturbation_types
    }

    for pt_idx, perturb_type in enumerate(perturbation_types):
        print(f"\n  ── Perturbation: {perturb_type} ({pt_idx+1}/3) ──")

        for step_k in tqdm(range(NFE), desc=f"  {perturb_type}"):

            # Generate perturbed samples for ALL classes at this step
            perturbed_images = []
            for start in range(0, total_samples, MINI_BATCH):
                end = min(start + MINI_BATCH, total_samples)
                x0_batch = x0_all[start:end]
                labels_batch = labels[start:end]
                swap_batch = swap_labels_all[start:end]

                x_final, _ = euler_sample_with_perturbation(
                    model, x0_batch, labels_batch,
                    perturb_step=step_k,
                    perturb_type=perturb_type,
                    swap_labels=swap_batch if perturb_type == "class_swap" else None,
                )
                perturbed_images.append(x_final)

            perturbed_images = torch.cat(perturbed_images, dim=0)

            # Extract Inception features for perturbed batch
            perturbed_feats_list = []
            for start in range(0, total_samples, MINI_BATCH):
                end = min(start + MINI_BATCH, total_samples)
                feats = fid_metric.get_features(perturbed_images[start:end])
                perturbed_feats_list.append(feats)
            perturbed_feats = torch.cat(perturbed_feats_list, dim=0)

            # ── Per-class metrics ─────────────────────
            for c in range(NUM_CLASSES):
                mask = (labels == c)
                idx = mask.nonzero(as_tuple=True)[0]

                # ΔLPIPS: per-image perceptual distance
                base_c = baseline_images[idx]
                pert_c = perturbed_images[idx]
                lpips_vals = lpips_metric.compute(base_c, pert_c)
                mean_lpips = lpips_vals.mean().item()

                # ΔFID: distribution shift for this class
                base_feats_c = baseline_feats[idx]
                pert_feats_c = perturbed_feats[idx]
                delta_fid = FIDMetric.compute_fid(base_feats_c, pert_feats_c)

                results[perturb_type][c]["lpips"].append(mean_lpips)
                results[perturb_type][c]["fid"].append(delta_fid)

            # Free GPU memory
            del perturbed_images, perturbed_feats
            torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────
    # 6. SAVE RAW RESULTS
    # ─────────────────────────────────────────────────────
    print("\n[6] Saving results...")
    results_path = os.path.join(OUTPUT_DIR, "ablation_results.json")

    # Convert to serializable format
    serializable = {}
    for pt in perturbation_types:
        serializable[pt] = {}
        for c in range(NUM_CLASSES):
            serializable[pt][CIFAR10_CLASSES[c]] = {
                "lpips": results[pt][c]["lpips"],
                "fid": results[pt][c]["fid"],
            }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {results_path}")

    # ─────────────────────────────────────────────────────
    # 7. GENERATE PLOTS
    # ─────────────────────────────────────────────────────
    print("\n[7] Generating plots...")
    generate_all_plots(results)

    # ─────────────────────────────────────────────────────
    # 8. SUMMARY STATISTICS
    # ─────────────────────────────────────────────────────
    print("\n[8] Summary: Top-5 load-bearing steps per class (ΔLPIPS, zero-out)")
    print("-" * 70)
    for c in range(NUM_CLASSES):
        lpips_curve = np.array(results["zero"][c]["lpips"])
        top5 = np.argsort(lpips_curve)[::-1][:5]
        t_values = np.linspace(0, 1, NFE + 1)
        top5_str = ", ".join(
            [f"k={k} (t={t_values[k]:.3f}, Δ={lpips_curve[k]:.4f})" for k in top5]
        )
        print(f"  {CIFAR10_CLASSES[c]:>12s}: {top5_str}")

    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)


# =========================================================
# PLOTTING
# =========================================================
def generate_all_plots(results):
    """Generate all publication-quality ablation plots."""

    t_axis = np.linspace(0, 1, NFE, endpoint=False)  # step times
    colors = plt.cm.tab10(np.arange(10))

    # ── PLOT 1: Per-class ΔLPIPS curves (one subplot per perturbation) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax_idx, pt in enumerate(["zero", "gaussian", "class_swap"]):
        ax = axes[ax_idx]
        for c in range(NUM_CLASSES):
            ax.plot(
                t_axis, results[pt][c]["lpips"],
                color=colors[c], label=CIFAR10_CLASSES[c],
                linewidth=1.5, alpha=0.85
            )
        ax.set_xlabel("Time t", fontsize=12)
        ax.set_title(
            {"zero": "Zero-Out", "gaussian": "Gaussian Noise",
             "class_swap": "Class Swap"}[pt],
            fontsize=13, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    axes[0].set_ylabel("ΔLPIPS (perceptual distance)", fontsize=12)
    axes[2].legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=9, framealpha=0.9
    )
    fig.suptitle(
        "Step Importance Curves: Per-Class ΔLPIPS",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lpips_per_class_all_perturbations.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── PLOT 2: Aggregated (class-averaged) importance curves ───────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax_idx, pt in enumerate(["zero", "gaussian", "class_swap"]):
        ax = axes[ax_idx]

        all_lpips = np.array([results[pt][c]["lpips"] for c in range(NUM_CLASSES)])
        mean_curve = all_lpips.mean(axis=0)
        std_curve = all_lpips.std(axis=0)

        ax.plot(t_axis, mean_curve, "k-", linewidth=2, label="Mean")
        ax.fill_between(
            t_axis, mean_curve - std_curve, mean_curve + std_curve,
            alpha=0.2, color="steelblue", label="±1 std"
        )
        ax.set_xlabel("Time t", fontsize=12)
        ax.set_title(
            {"zero": "Zero-Out", "gaussian": "Gaussian Noise",
             "class_swap": "Class Swap"}[pt],
            fontsize=13, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    axes[0].set_ylabel("ΔLPIPS (mean ± std across classes)", fontsize=12)
    fig.suptitle(
        "Aggregated Step Importance (Class-Averaged)",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lpips_aggregated.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── PLOT 3: ΔFID heatmaps (class × step) for each perturbation ─────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax_idx, pt in enumerate(["zero", "gaussian", "class_swap"]):
        ax = axes[ax_idx]

        fid_matrix = np.array(
            [results[pt][c]["fid"] for c in range(NUM_CLASSES)]
        )
        im = ax.imshow(
            fid_matrix, aspect="auto", cmap="YlOrRd",
            interpolation="nearest"
        )
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_yticklabels(CIFAR10_CLASSES, fontsize=9)
        ax.set_xlabel("Step k", fontsize=12)
        ax.set_title(
            {"zero": "Zero-Out", "gaussian": "Gaussian Noise",
             "class_swap": "Class Swap"}[pt],
            fontsize=13, fontweight="bold"
        )
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "ΔFID Heatmaps: Class × Step",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fid_heatmaps.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── PLOT 4: Class divergence analysis ───────────────────────────────
    # Shows how much classes DIFFER in their load-bearing patterns
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in range(NUM_CLASSES):
        # Normalize each class's curve to sum to 1 (importance distribution)
        curve = np.array(results["zero"][c]["lpips"])
        if curve.sum() > 0:
            normalized = curve / curve.sum()
        else:
            normalized = curve
        ax.plot(
            t_axis, normalized,
            color=colors[c], label=CIFAR10_CLASSES[c],
            linewidth=1.5, alpha=0.85
        )

    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Normalized Step Importance", fontsize=12)
    ax.set_title(
        "Per-Class Importance Distribution (Zero-Out, Normalized)",
        fontsize=14, fontweight="bold"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "class_divergence.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── PLOT 5: Top-K load-bearing steps bar chart ──────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    axes = axes.flatten()
    for c in range(NUM_CLASSES):
        ax = axes[c]
        curve = np.array(results["zero"][c]["lpips"])
        bar_colors = ["#d32f2f" if v >= np.sort(curve)[-5] else "#90caf9"
                       for v in curve]
        ax.bar(range(NFE), curve, color=bar_colors, width=0.8)
        ax.set_title(CIFAR10_CLASSES[c], fontsize=11, fontweight="bold")
        ax.set_xlabel("Step k", fontsize=9)
        ax.set_ylabel("ΔLPIPS", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "Per-Class Step Importance (Top-5 Load-Bearing Steps in Red)",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "topk_steps_per_class.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    main()