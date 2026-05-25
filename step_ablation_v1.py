"""
============================================================================
STEP ABLATION ANALYSIS FOR OT-CFM — WITH RESUME SUPPORT
============================================================================
Saves after every single step. Kill anytime, resume from where you left off.
============================================================================
"""

import os
import json
import multiprocessing

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from diffusers import UNet2DModel
from tqdm import tqdm
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

IMAGE_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 10
GUIDANCE_SCALE = 3.0
NFE = 32

SAMPLES_PER_CLASS = 100        # 100 × 10 = 1000 total
MINI_BATCH = 100               # CFG doubles to 200, fine for 16GB

CHECKPOINT_PATH = r"D:\Salman Ahmed LUMS\Personal\LUMS\SEMESTER 4\DVLM\PROJECT\train_v2\best_model_ema.pt"
OUTPUT_DIR = "./ablation_results_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Where we save progress after every step
PROGRESS_PATH = os.path.join(OUTPUT_DIR, "progress.json")
BASELINE_PATH = os.path.join(OUTPUT_DIR, "baseline_images.pt")
BASELINE_FEATS_PATH = os.path.join(OUTPUT_DIR, "baseline_feats.pt")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# =========================================================
# MODEL
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
def cfg_velocity(model, x, t_scalar, labels):
    B = x.shape[0]
    t_batch = torch.full((B,), t_scalar, device=DEVICE)
    null_labels = torch.full_like(labels, NUM_CLASSES)

    combined_x = torch.cat([x, x], dim=0)
    combined_t = torch.cat([t_batch, t_batch], dim=0)
    combined_labels = torch.cat([labels, null_labels], dim=0)

    v_all = model(combined_x, combined_t, combined_labels)
    v_cond, v_uncond = torch.chunk(v_all, 2, dim=0)

    return v_uncond + GUIDANCE_SCALE * (v_cond - v_uncond)


# =========================================================
# EULER SOLVER WITH PERTURBATION
# =========================================================
@torch.no_grad()
def euler_sample_with_perturbation(
    model, x0, labels, nfe=NFE,
    perturb_step=None, perturb_type=None,
    swap_labels=None
):
    dt = 1.0 / nfe
    t_values = torch.linspace(0, 1, nfe + 1, device=DEVICE)
    x = x0.clone()

    for k in range(nfe):
        t_k = t_values[k].item()

        if k == perturb_step and perturb_type is not None:
            if perturb_type == "zero":
                v = torch.zeros_like(x)

            elif perturb_type == "gaussian":
                v_real = cfg_velocity(model, x, t_k, labels)
                real_norm = v_real.view(v_real.shape[0], -1).norm(dim=1)
                noise = torch.randn_like(x)
                noise_norm = noise.view(noise.shape[0], -1).norm(dim=1)
                scale = (real_norm / (noise_norm + 1e-8)).view(-1, 1, 1, 1)
                v = noise * scale

            elif perturb_type == "class_swap":
                v = cfg_velocity(model, x, t_k, swap_labels)

        else:
            v = cfg_velocity(model, x, t_k, labels)

        x = x + v * dt

    return x


# =========================================================
# METRICS
# =========================================================
class LPIPSMetric:
    def __init__(self):
        self.fn = lpips.LPIPS(net="alex").to(DEVICE).eval()

    @torch.no_grad()
    def compute(self, img1, img2):
        return self.fn(img1, img2).squeeze()


class FIDMetric:
    def __init__(self):
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights).to(DEVICE).eval()
        self.model.fc = nn.Identity()
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299), antialias=True),
        ])

    @torch.no_grad()
    def get_features(self, images):
        imgs = self.preprocess(images)
        imgs = (imgs + 1) / 2
        return self.model(imgs)

    @staticmethod
    def compute_fid(feats1, feats2):
        f1 = feats1.cpu().numpy()
        f2 = feats2.cpu().numpy()
        mu1, sigma1 = f1.mean(axis=0), np.cov(f1, rowvar=False)
        mu2, sigma2 = f2.mean(axis=0), np.cov(f2, rowvar=False)
        diff = mu1 - mu2
        eigvals, eigvecs = np.linalg.eigh(sigma1 @ sigma2)
        eigvals = np.maximum(eigvals, 0)
        sqrt_product = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * sqrt_product)
        return float(np.real(fid))


# =========================================================
# PROGRESS SAVE / LOAD
# =========================================================
def save_progress(results, completed):
    """
    Save current results and completed steps to disk.
    completed = list of (perturb_type, step_k) tuples already done.
    """
    serializable = {
        "completed": completed,
        "results": {}
    }
    for pt in results:
        serializable["results"][pt] = {}
        for c in range(NUM_CLASSES):
            serializable["results"][pt][CIFAR10_CLASSES[c]] = {
                "lpips": results[pt][c]["lpips"],
                "fid": results[pt][c]["fid"],
            }
    with open(PROGRESS_PATH, "w") as f:
        json.dump(serializable, f)


def load_progress():
    """
    Load saved progress if it exists.
    Returns (results dict, completed list) or (None, []) if no progress.
    """
    if not os.path.exists(PROGRESS_PATH):
        return None, []

    with open(PROGRESS_PATH) as f:
        data = json.load(f)

    completed = [tuple(x) for x in data["completed"]]

    # Rebuild results in integer-keyed format
    results = {
        pt: {c: {"lpips": [], "fid": []} for c in range(NUM_CLASSES)}
        for pt in ["zero", "gaussian", "class_swap"]
    }
    for pt in data["results"]:
        for c_idx, cls_name in enumerate(CIFAR10_CLASSES):
            if cls_name in data["results"][pt]:
                results[pt][c_idx]["lpips"] = data["results"][pt][cls_name]["lpips"]
                results[pt][c_idx]["fid"] = data["results"][pt][cls_name]["fid"]

    print(f"  Resumed from progress file. {len(completed)} steps already done.")
    return results, completed


# =========================================================
# SWAP LABELS
# =========================================================
def generate_swap_labels(labels):
    swap = labels.clone()
    for i in range(len(swap)):
        candidates = [c for c in range(NUM_CLASSES) if c != labels[i].item()]
        swap[i] = torch.tensor(
            candidates[torch.randint(0, len(candidates), (1,)).item()],
            device=DEVICE
        )
    return swap


# =========================================================
# MAIN
# =========================================================
def main():
    total_samples = SAMPLES_PER_CLASS * NUM_CLASSES

    print("=" * 70)
    print("STEP ABLATION ANALYSIS — WITH RESUME SUPPORT")
    print(f"Samples per class: {SAMPLES_PER_CLASS} | Total: {total_samples}")
    print(f"NFE: {NFE} | MINI_BATCH: {MINI_BATCH}")
    print(f"Progress file: {PROGRESS_PATH}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────
    # 1. LOAD MODEL
    # ─────────────────────────────────────────────────────
    print("\n[1] Loading model...")
    model = ClassConditionalUNet().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded: {CHECKPOINT_PATH}")

    # ─────────────────────────────────────────────────────
    # 2. METRICS
    # ─────────────────────────────────────────────────────
    print("\n[2] Initializing metrics...")
    lpips_metric = LPIPSMetric()
    fid_metric = FIDMetric()
    print("  LPIPS and FID ready.")

    # ─────────────────────────────────────────────────────
    # 3. FIXED NOISE AND LABELS
    # ─────────────────────────────────────────────────────
    print("\n[3] Preparing noise and labels...")
    labels = torch.arange(NUM_CLASSES, device=DEVICE).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    torch.manual_seed(42)
    x0_all = torch.randn(
        total_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE
    )
    swap_labels_all = generate_swap_labels(labels)
    print(f"  Labels: {labels.shape} | Noise: {x0_all.shape}")

    # ─────────────────────────────────────────────────────
    # 4. BASELINE — load from disk if already generated
    # ─────────────────────────────────────────────────────
    if os.path.exists(BASELINE_PATH) and os.path.exists(BASELINE_FEATS_PATH):
        print("\n[4] Loading cached baseline from disk...")
        baseline_images = torch.load(BASELINE_PATH, map_location="cpu")
        baseline_feats = torch.load(BASELINE_FEATS_PATH, map_location="cpu")
        print(f"  Baseline: {baseline_images.shape}")
        print(f"  Features: {baseline_feats.shape}")
    else:
        print("\n[4] Generating baseline samples...")
        baseline_images = []
        for start in tqdm(range(0, total_samples, MINI_BATCH), desc="Baseline"):
            end = min(start + MINI_BATCH, total_samples)
            imgs = euler_sample_with_perturbation(
                model, x0_all[start:end], labels[start:end]
            )
            baseline_images.append(imgs.cpu())
        baseline_images = torch.cat(baseline_images, dim=0)

        print("  Extracting baseline Inception features...")
        baseline_feats_list = []
        for start in range(0, total_samples, MINI_BATCH):
            end = min(start + MINI_BATCH, total_samples)
            feats = fid_metric.get_features(baseline_images[start:end].to(DEVICE))
            baseline_feats_list.append(feats.cpu())
        baseline_feats = torch.cat(baseline_feats_list, dim=0)

        # Save baseline to disk — never recompute it again
        torch.save(baseline_images, BASELINE_PATH)
        torch.save(baseline_feats, BASELINE_FEATS_PATH)
        print(f"  Saved baseline: {BASELINE_PATH}")
        print(f"  Saved features: {BASELINE_FEATS_PATH}")

    # ─────────────────────────────────────────────────────
    # 5. LOAD PROGRESS OR START FRESH
    # ─────────────────────────────────────────────────────
    print("\n[5] Checking for saved progress...")
    results, completed = load_progress()

    if results is None:
        print("  No progress found. Starting fresh.")
        results = {
            pt: {c: {"lpips": [], "fid": []} for c in range(NUM_CLASSES)}
            for pt in ["zero", "gaussian", "class_swap"]
        }

    # ─────────────────────────────────────────────────────
    # 6. PERTURBATION SWEEP
    # ─────────────────────────────────────────────────────
    print("\n[6] Running perturbation sweep...")
    perturbation_types = ["zero", "gaussian", "class_swap"]

    for perturb_type in perturbation_types:
        print(f"\n  ── {perturb_type} ──")

        for step_k in tqdm(range(NFE), desc=f"  {perturb_type}"):

            # Skip if already done
            if (perturb_type, step_k) in completed:
                continue

            # Generate perturbed samples
            perturbed_images = []
            for start in range(0, total_samples, MINI_BATCH):
                end = min(start + MINI_BATCH, total_samples)
                imgs = euler_sample_with_perturbation(
                    model,
                    x0_all[start:end],
                    labels[start:end],
                    perturb_step=step_k,
                    perturb_type=perturb_type,
                    swap_labels=swap_labels_all[start:end]
                )
                perturbed_images.append(imgs.cpu())
            perturbed_images = torch.cat(perturbed_images, dim=0)

            # Extract features
            perturbed_feats_list = []
            for start in range(0, total_samples, MINI_BATCH):
                end = min(start + MINI_BATCH, total_samples)
                feats = fid_metric.get_features(perturbed_images[start:end].to(DEVICE))
                perturbed_feats_list.append(feats.cpu())
            perturbed_feats = torch.cat(perturbed_feats_list, dim=0)

            # Per-class metrics
            for c in range(NUM_CLASSES):
                mask = (labels.cpu() == c)
                idx = mask.nonzero(as_tuple=True)[0]

                base_c = baseline_images[idx].to(DEVICE)
                pert_c = perturbed_images[idx].to(DEVICE)
                lpips_vals = lpips_metric.compute(base_c, pert_c)
                mean_lpips = lpips_vals.mean().item() if lpips_vals.dim() > 0 else lpips_vals.item()

                base_feats_c = baseline_feats[idx]
                pert_feats_c = perturbed_feats[idx]
                delta_fid = FIDMetric.compute_fid(base_feats_c, pert_feats_c)

                results[perturb_type][c]["lpips"].append(mean_lpips)
                results[perturb_type][c]["fid"].append(delta_fid)

            # ── SAVE AFTER EVERY STEP ────────────────
            completed.append((perturb_type, step_k))
            save_progress(results, completed)

            del perturbed_images, perturbed_feats
            torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────
    # 7. FINAL SAVE
    # ─────────────────────────────────────────────────────
    print("\n[7] Saving final results...")
    final_path = os.path.join(OUTPUT_DIR, "ablation_results_v2.json")
    serializable = {}
    for pt in perturbation_types:
        serializable[pt] = {}
        for c in range(NUM_CLASSES):
            serializable[pt][CIFAR10_CLASSES[c]] = {
                "lpips": results[pt][c]["lpips"],
                "fid": results[pt][c]["fid"],
            }
    with open(final_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {final_path}")

    # ─────────────────────────────────────────────────────
    # 8. PLOTS
    # ─────────────────────────────────────────────────────
    print("\n[8] Generating plots...")
    generate_all_plots(results)

    # ─────────────────────────────────────────────────────
    # 9. SUMMARY
    # ─────────────────────────────────────────────────────
    print("\n[9] Top-5 load-bearing steps per class (zero-out ΔLPIPS):")
    print("-" * 70)
    for c in range(NUM_CLASSES):
        curve = np.array(results["zero"][c]["lpips"])
        top5 = np.argsort(curve)[::-1][:5]
        t_vals = np.linspace(0, 1, NFE + 1)
        top5_str = ", ".join(
            [f"k={k}(t={t_vals[k]:.2f},Δ={curve[k]:.4f})" for k in top5]
        )
        print(f"  {CIFAR10_CLASSES[c]:>12s}: {top5_str}")

    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)


# =========================================================
# PLOTS
# =========================================================
def generate_all_plots(results):
    t_axis = np.linspace(0, 1, NFE, endpoint=False)
    colors = plt.cm.tab10(np.arange(10))

    # Plot 1: Per-class ΔLPIPS
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax_idx, pt in enumerate(["zero", "gaussian", "class_swap"]):
        ax = axes[ax_idx]
        for c in range(NUM_CLASSES):
            ax.plot(t_axis, results[pt][c]["lpips"],
                    color=colors[c], label=CIFAR10_CLASSES[c],
                    linewidth=1.5, alpha=0.85)
        ax.set_xlabel("Time t", fontsize=12)
        ax.set_title({"zero": "Zero-Out", "gaussian": "Gaussian Noise",
                      "class_swap": "Class Swap"}[pt],
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes[0].set_ylabel("ΔLPIPS", fontsize=12)
    axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.suptitle("Step Importance Curves: Per-Class ΔLPIPS",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lpips_per_class_all_perturbations.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: Aggregated
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax_idx, pt in enumerate(["zero", "gaussian", "class_swap"]):
        ax = axes[ax_idx]
        all_lpips = np.array([results[pt][c]["lpips"] for c in range(NUM_CLASSES)])
        mean_curve = all_lpips.mean(axis=0)
        std_curve = all_lpips.std(axis=0)
        ax.plot(t_axis, mean_curve, "k-", linewidth=2, label="Mean")
        ax.fill_between(t_axis, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color="steelblue", label="±1 std")
        ax.set_xlabel("Time t", fontsize=12)
        ax.set_title({"zero": "Zero-Out", "gaussian": "Gaussian Noise",
                      "class_swap": "Class Swap"}[pt],
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    axes[0].set_ylabel("ΔLPIPS (mean ± std)", fontsize=12)
    fig.suptitle("Aggregated Step Importance", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lpips_aggregated.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 3: FID heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax_idx, pt in enumerate(["zero", "gaussian", "class_swap"]):
        ax = axes[ax_idx]
        fid_matrix = np.array([results[pt][c]["fid"] for c in range(NUM_CLASSES)])
        im = ax.imshow(fid_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_yticklabels(CIFAR10_CLASSES, fontsize=9)
        ax.set_xlabel("Step k", fontsize=12)
        ax.set_title({"zero": "Zero-Out", "gaussian": "Gaussian Noise",
                      "class_swap": "Class Swap"}[pt],
                     fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("ΔFID Heatmaps: Class × Step", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fid_heatmaps.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 4: Class divergence
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in range(NUM_CLASSES):
        curve = np.array(results["zero"][c]["lpips"])
        normalized = curve / curve.sum() if curve.sum() > 0 else curve
        ax.plot(t_axis, normalized, color=colors[c],
                label=CIFAR10_CLASSES[c], linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Normalized Step Importance", fontsize=12)
    ax.set_title("Per-Class Importance Distribution (Zero-Out, Normalized)",
                 fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_divergence.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 5: Top-K bar chart
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
    fig.suptitle("Per-Class Step Importance (Top-5 in Red)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "topk_steps_per_class.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    print("  All plots saved.")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    main()