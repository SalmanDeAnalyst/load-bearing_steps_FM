"""
============================================================================
SCHEDULE EVALUATION FOR OT-CFM  —  WITH HIGHER-ORDER SOLVERS
============================================================================
Extends the original Stage-4 schedule evaluation with:

  * Euler (1st), Heun (2nd) and RK4 (4th) ODE solvers
  * FAIR NFE accounting:  Heun = 2 model evals/step, RK4 = 4 evals/step
        so a "budget NFE = 8" run means:
            euler -> 8 steps,  heun -> 4 steps,  rk4 -> 2 steps
  * Resume support: results are saved (and the Volume committed) after EVERY
        run, so a crash/idle-shutdown never loses more than one run
  * A CORRECTED FID (the old one returned negative garbage)
  * Modal-friendly paths:
        INPUT  files (model, ablation json, classifier) = uploaded into the
               notebook's local working directory
        OUTPUT (json + plots) = written into a mounted Modal Volume

------------------------------------------------------------------------
HOW TO RUN (inside a Modal Notebook cell):
    !python schedule_evaluation_with_higher_order_solvers.py

First run with TEST_MODE = True (finishes in a few minutes) to validate the
pipeline. Then set TEST_MODE = False and re-run for the full sweep.
============================================================================
"""

import os
import json
import math
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")              # headless: no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from diffusers import UNet2DModel

multiprocessing.freeze_support()


# ===========================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>  EDIT THESE TWO BLOCKS  <<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===========================================================================

# ---- 1. TEST vs FULL ------------------------------------------------------
#   True  = tiny fast run to confirm everything works (~a few minutes)
#   False = full publishable run (a few hours)
TEST_MODE = False

# ---- 2. PATHS -------------------------------------------------------------
#   INPUT_DIR    : where you uploaded the input files (the notebook cwd = ".")
#   VOLUME_NAME  : the name of the Modal Volume you created
#   VOLUME_MOUNT : the path where that Volume is mounted in your notebook UI
#                  (check the kernel/volume settings sidebar — change if needed)
INPUT_DIR    = "."
VOLUME_NAME  = "fm-data"
VOLUME_MOUNT = "/mnt/fm-data"

# ===========================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>  END OF EDITS  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===========================================================================


CHECKPOINT_PATH = os.path.join(INPUT_DIR, "best_model_ema.pt")
ABLATION_PATH   = os.path.join(INPUT_DIR, "ablation_results_v2.json")
CLASSIFIER_PATH = os.path.join(INPUT_DIR, "cifar10_classifier.pt")
DATA_DIR        = "./data"                 # CIFAR-10 download (ephemeral)

if TEST_MODE:
    OUTPUT_DIR = os.path.join(VOLUME_MOUNT, "results_solver_sweep_TEST")
else:
    OUTPUT_DIR = os.path.join(VOLUME_MOUNT, "results_solver_sweep")


# ---------------------------------------------------------------------------
# HARDWARE
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no GPU detected — this will be extremely slow.")


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
IMAGE_SIZE     = 32
CHANNELS       = 3
NUM_CLASSES    = 10
GUIDANCE_SCALE = 3.0

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# EXPERIMENT CONFIG  (depends on TEST_MODE)
# ---------------------------------------------------------------------------
SOLVER_LIST    = ["euler", "heun", "rk4"]
SCHEDULE_NAMES = ["uniform", "log", "greedy", "classaware"]
SOLVER_EVALS_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}

if TEST_MODE:
    SAMPLES_PER_CLASS = 20      # 200 total
    NFE_LIST          = [4]
    MINI_BATCH        = 50
    MAX_REAL_IMAGES   = 1000    # cap real set for a faster FID during testing
    CLASSIFIER_EPOCHS = 3
else:
    SAMPLES_PER_CLASS = 500     # 5000 total
    NFE_LIST          = [2, 4, 8, 32]
    MINI_BATCH        = 50
    MAX_REAL_IMAGES   = 10000
    CLASSIFIER_EPOCHS = 15


# ---------------------------------------------------------------------------
# VOLUME COMMIT HELPER
# ---------------------------------------------------------------------------
def commit_volume():
    """Flush writes to the mounted Modal Volume so they persist on disk."""
    try:
        import modal
        modal.Volume.from_name(VOLUME_NAME).commit()
        print("    [volume] committed.")
    except Exception as e:
        # If the volume auto-persists (UI-mounted) this is harmless.
        print(f"    [volume] commit skipped ({e}); relying on auto-persist.")


# ===========================================================================
# MODEL  (must match training exactly)
# ===========================================================================
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


# ===========================================================================
# CFG VELOCITY
# ===========================================================================
@torch.no_grad()
def cfg_velocity(model, x, t_scalar, labels, guidance_scale=GUIDANCE_SCALE):
    B = x.shape[0]
    t_batch = torch.full((B,), float(t_scalar), device=DEVICE)
    null_labels = torch.full_like(labels, NUM_CLASSES)

    combined_x = torch.cat([x, x], dim=0)
    combined_t = torch.cat([t_batch, t_batch], dim=0)
    combined_labels = torch.cat([labels, null_labels], dim=0)

    v_all = model(combined_x, combined_t, combined_labels)
    v_cond, v_uncond = torch.chunk(v_all, 2, dim=0)
    return v_uncond + guidance_scale * (v_cond - v_uncond)


# ===========================================================================
# SOLVERS (each takes an arbitrary time schedule [t0, t1, ..., tN])
# ===========================================================================
@torch.no_grad()
def euler_sample_with_schedule(model, x0, labels, t_schedule):
    """Forward Euler (1st-order). 1 velocity eval per step."""
    x = x0.clone()
    for i in range(len(t_schedule) - 1):
        t_curr = t_schedule[i]
        dt = t_schedule[i + 1] - t_curr
        v = cfg_velocity(model, x, t_curr, labels)
        x = x + v * dt
    return x


@torch.no_grad()
def heun_sample_with_schedule(model, x0, labels, t_schedule):
    """Heun's method (2nd-order). 2 velocity evals per step."""
    x = x0.clone()
    for i in range(len(t_schedule) - 1):
        t_curr = t_schedule[i]
        t_next = t_schedule[i + 1]
        dt = t_next - t_curr
        k1 = cfg_velocity(model, x, t_curr, labels)          # start
        x_pred = x + dt * k1                                  # Euler predictor
        k2 = cfg_velocity(model, x_pred, t_next, labels)      # endpoint
        x = x + 0.5 * dt * (k1 + k2)                          # corrector
    return x


@torch.no_grad()
def rk4_sample_with_schedule(model, x0, labels, t_schedule):
    """Classical 4th-order Runge-Kutta. 4 velocity evals per step."""
    x = x0.clone()
    for i in range(len(t_schedule) - 1):
        t_curr = t_schedule[i]
        t_next = t_schedule[i + 1]
        h = t_next - t_curr
        t_mid = t_curr + 0.5 * h
        k1 = cfg_velocity(model, x,               t_curr, labels)
        k2 = cfg_velocity(model, x + 0.5 * h * k1, t_mid,  labels)
        k3 = cfg_velocity(model, x + 0.5 * h * k2, t_mid,  labels)
        k4 = cfg_velocity(model, x + h       * k3, t_next, labels)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


SOLVER_DISPATCH = {
    "euler": euler_sample_with_schedule,
    "heun":  heun_sample_with_schedule,
    "rk4":   rk4_sample_with_schedule,
}


# ===========================================================================
# SCHEDULE CONSTRUCTORS  (num_steps = number of intervals)
# ===========================================================================
def build_uniform_schedule(num_steps):
    return np.linspace(0, 1, num_steps + 1).tolist()


def build_log_schedule(num_steps, eps=1e-3):
    pts = np.exp(np.linspace(np.log(eps), np.log(1.0), num_steps + 1))
    pts = pts - pts[0]
    pts = pts / pts[-1]
    return pts.tolist()


def build_greedy_schedule(num_steps, ablation_data):
    classes = list(ablation_data["zero"].keys())
    curves = np.array([ablation_data["zero"][c]["lpips"] for c in classes])
    mean_curve = curves.mean(axis=0)                 # length 32
    t_32 = np.linspace(0, 1, len(mean_curve) + 1)    # 33 boundaries

    ranked = np.argsort(mean_curve)[::-1]
    selected = {0}                                    # always include t=0
    for s in ranked:
        if len(selected) >= num_steps:
            break
        selected.add(int(s))
    selected_t = sorted(t_32[s] for s in selected)
    selected_t.append(1.0)                            # always end at t=1
    return selected_t


def build_classaware_schedules(num_steps, ablation_data):
    classes = list(ablation_data["zero"].keys())
    schedules = {}
    for cls_idx, cls_name in enumerate(classes):
        curve = np.array(ablation_data["zero"][cls_name]["lpips"])
        t_32 = np.linspace(0, 1, len(curve) + 1)
        if curve.sum() == 0:
            schedules[cls_idx] = build_uniform_schedule(num_steps)
            continue
        cdf = np.cumsum(curve) / curve.sum()
        quantiles = np.linspace(0, 1, num_steps + 1)[:-1]
        selected_t = []
        for q in quantiles:
            idx = min(int(np.searchsorted(cdf, q)), len(curve) - 1)
            selected_t.append(t_32[idx])
        selected_t.append(1.0)
        schedules[cls_idx] = selected_t
    return schedules


# ===========================================================================
# METRICS
# ===========================================================================
class InceptionFeatureExtractor:
    def __init__(self):
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights).to(DEVICE).eval()
        self.model.fc = nn.Identity()

    @torch.no_grad()
    def extract(self, images, batch_size=50):
        """images: [N,3,32,32] in [-1,1]  ->  [N,2048] on CPU."""
        all_feats = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(DEVICE)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear",
                                  align_corners=False, antialias=True)
            batch = (batch + 1) / 2
            feats = self.model(batch)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)


def compute_fid(feats1, feats2):
    """
    Corrected FID.  The old version used eigh() on the (non-symmetric)
    product sigma1 @ sigma2 and rebuilt the wrong matrix square root,
    producing negative values.  Here we use the identity:

        tr( sqrt(sigma1 @ sigma2) ) = sum( sqrt(eigvals(sigma1 @ sigma2)) )

    which is the mathematically correct trace term and needs no scipy.
    """
    f1 = feats1.cpu().numpy() if torch.is_tensor(feats1) else np.asarray(feats1)
    f2 = feats2.cpu().numpy() if torch.is_tensor(feats2) else np.asarray(feats2)

    mu1, sigma1 = f1.mean(axis=0), np.cov(f1, rowvar=False)
    mu2, sigma2 = f2.mean(axis=0), np.cov(f2, rowvar=False)

    diff = mu1 - mu2
    eigvals = np.linalg.eigvals(sigma1 @ sigma2)
    eigvals = np.clip(np.real(eigvals), 0, None)
    tr_sqrt = np.sum(np.sqrt(eigvals))

    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_sqrt
    return float(np.real(fid))


def compute_precision_recall(real_feats, gen_feats, k=3):
    """Improved Precision/Recall (Kynkaanniemi et al. 2019) via k-NN."""
    real_np = real_feats.cpu().numpy() if torch.is_tensor(real_feats) else real_feats
    gen_np  = gen_feats.cpu().numpy()  if torch.is_tensor(gen_feats)  else gen_feats

    def kth_nn(feats, k):
        d = np.sqrt(np.maximum(
            np.sum(feats ** 2, axis=1, keepdims=True)
            - 2 * feats @ feats.T
            + np.sum(feats ** 2, axis=1, keepdims=True).T, 0) + 1e-10)
        np.fill_diagonal(d, np.inf)
        return np.sort(d, axis=1)[:, k - 1]

    def cross(a, b):
        return np.sqrt(np.maximum(
            np.sum(a ** 2, axis=1, keepdims=True)
            - 2 * a @ b.T
            + np.sum(b ** 2, axis=1, keepdims=True).T, 0) + 1e-10)

    real_kth = kth_nn(real_np, k)
    gen_kth  = kth_nn(gen_np, k)

    gen_to_real = cross(gen_np, real_np)
    precision = float(np.mean(np.any(gen_to_real <= real_kth[np.newaxis, :], axis=1)))

    real_to_gen = cross(real_np, gen_np)
    recall = float(np.mean(np.any(real_to_gen <= gen_kth[np.newaxis, :], axis=1)))
    return precision, recall


class CIFAR10Classifier(nn.Module):
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


def train_cifar10_classifier(epochs):
    """Load an uploaded classifier if present, else train + save to the Volume."""
    if os.path.exists(CLASSIFIER_PATH):
        print(f"  Loading uploaded classifier: {CLASSIFIER_PATH}")
        m = CIFAR10Classifier().to(DEVICE)
        m.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE,
                                     weights_only=True))
        m.eval()
        return m

    print(f"  No uploaded classifier — training one ({epochs} epochs)...")
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_train)
    test_set  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                              shuffle=False, num_workers=2, pin_memory=True)

    m = CIFAR10Classifier().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        m.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            loss = F.cross_entropy(m(imgs), lbls)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    m.eval(); correct = total = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            preds = m(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item(); total += lbls.size(0)
    print(f"  Classifier test accuracy: {100 * correct / total:.1f}%")

    save_to = os.path.join(OUTPUT_DIR, "cifar10_classifier.pt")
    torch.save(m.state_dict(), save_to)
    print(f"  Saved classifier to Volume: {save_to}")
    return m


@torch.no_grad()
def evaluate_class_accuracy(classifier, images, labels):
    all_preds = []
    for start in range(0, len(images), 100):
        batch = images[start:start + 100].to(DEVICE)
        all_preds.append(classifier(batch).argmax(dim=1).cpu())
    all_preds = torch.cat(all_preds)
    overall = (all_preds == labels).float().mean().item()
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            per_class[CIFAR10_CLASSES[c]] = (all_preds[mask] == c).float().mean().item()
    return overall, per_class


# ===========================================================================
# GENERATION
# ===========================================================================
@torch.no_grad()
def generate_samples(model, schedule_or_schedules, labels, x0,
                     solver="euler", is_classaware=False):
    solver_fn = SOLVER_DISPATCH[solver]
    total = len(labels)
    out = []
    for start in range(0, total, MINI_BATCH):
        end = min(start + MINI_BATCH, total)
        x0_b = x0[start:end].to(DEVICE)
        lbl_b = labels[start:end].to(DEVICE)
        if is_classaware:
            batch_imgs = torch.zeros_like(x0_b)
            for c in range(NUM_CLASSES):
                mask = lbl_b == c
                if mask.sum() == 0:
                    continue
                imgs_c = solver_fn(model, x0_b[mask], lbl_b[mask],
                                   schedule_or_schedules[c])
                batch_imgs[mask] = imgs_c
            out.append(batch_imgs.cpu())
        else:
            out.append(solver_fn(model, x0_b, lbl_b, schedule_or_schedules).cpu())
    return torch.cat(out, dim=0)


# ===========================================================================
# PLOTTING
# ===========================================================================
def generate_plots(results):
    solver_colors  = {"euler": "#1f77b4", "heun": "#ff7f0e", "rk4": "#2ca02c"}
    solver_markers = {"euler": "o", "heun": "s", "rk4": "^"}

    def get(sched, solver, nfe, metric):
        k = f"{sched}_{solver}_nfe{nfe}"
        return results[k][metric] if k in results else None

    metrics = [
        ("recall",           "Recall (higher = better)"),
        ("precision",        "Precision (higher = better)"),
        ("overall_accuracy", "Class Accuracy (higher = better)"),
        ("fid",              "FID (lower = better)"),
    ]

    n = len(SCHEDULE_NAMES)
    for metric, ylabel in metrics:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, sched in zip(axes, SCHEDULE_NAMES):
            for solver in SOLVER_LIST:
                xs, ys = [], []
                for nfe in NFE_LIST:
                    v = get(sched, solver, nfe, metric)
                    if v is not None:
                        xs.append(nfe); ys.append(v)
                if xs:
                    ax.plot(xs, ys, color=solver_colors[solver],
                            marker=solver_markers[solver], linewidth=2,
                            markersize=8, label=solver)
            ax.set_title(sched, fontweight="bold")
            ax.set_xlabel("Total NFE")
            if len(NFE_LIST) > 1:
                ax.set_xscale("log", base=2)
                ax.set_xticks(NFE_LIST)
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"{ylabel}  —  per schedule, per solver",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"compare_{metric}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def print_summary(results):
    print("\n" + "=" * 92)
    print(f"{'Schedule':<12}{'Solver':<8}{'NFE':>5}{'Steps':>7}"
          f"{'FID':>11}{'Prec':>8}{'Rec':>8}{'Acc':>8}")
    print("-" * 92)
    for k in sorted(results.keys(),
                    key=lambda k: (results[k]["total_nfe"],
                                   results[k]["schedule_name"],
                                   results[k]["solver"])):
        r = results[k]
        print(f"{r['schedule_name']:<12}{r['solver']:<8}{r['total_nfe']:>5}"
              f"{r['num_steps']:>7}{r['fid']:>11.2f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['overall_accuracy']:>8.3f}")
    print("=" * 92)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "schedule_results.json")

    mode = "TEST" if TEST_MODE else "FULL"
    print("=" * 70)
    print(f"SCHEDULE EVALUATION WITH HIGHER-ORDER SOLVERS   [{mode} MODE]")
    print(f"  Solvers      : {SOLVER_LIST}")
    print(f"  Schedules    : {SCHEDULE_NAMES}")
    print(f"  NFE budgets  : {NFE_LIST}")
    print(f"  Samples/class: {SAMPLES_PER_CLASS}  (total {SAMPLES_PER_CLASS * NUM_CLASSES})")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print("=" * 70)

    # 1. MODEL ---------------------------------------------------------------
    print("\n[1] Loading flow-matching model...")
    model = ClassConditionalUNet().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded: {CHECKPOINT_PATH}")

    # 2. ABLATION DATA -------------------------------------------------------
    print("\n[2] Loading ablation importance curves...")
    with open(ABLATION_PATH) as f:
        ablation_data = json.load(f)
    print(f"  Loaded: {ABLATION_PATH}")

    # 3. FIXED NOISE + LABELS ------------------------------------------------
    print("\n[3] Preparing fixed noise and labels...")
    total_samples = SAMPLES_PER_CLASS * NUM_CLASSES
    labels = torch.arange(NUM_CLASSES).repeat_interleave(SAMPLES_PER_CLASS)
    torch.manual_seed(123)
    x0_all = torch.randn(total_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    print(f"  Labels: {tuple(labels.shape)} | Noise: {tuple(x0_all.shape)}")

    # 4. METRICS -------------------------------------------------------------
    print("\n[4] Initializing metrics...")
    inception = InceptionFeatureExtractor()
    classifier = train_cifar10_classifier(CLASSIFIER_EPOCHS)

    print("  Computing real CIFAR-10 features...")
    real_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    real_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=real_tf)
    real_loader = torch.utils.data.DataLoader(real_ds, batch_size=100,
                                              shuffle=False, num_workers=2)
    real_imgs = []
    for imgs, _ in real_loader:
        real_imgs.append(imgs)
        if len(real_imgs) * 100 >= MAX_REAL_IMAGES:
            break
    real_imgs = torch.cat(real_imgs, dim=0)[:MAX_REAL_IMAGES]
    real_feats = inception.extract(real_imgs)
    print(f"  Real features: {tuple(real_feats.shape)}")

    # 5. RESUME --------------------------------------------------------------
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"\n[5] Resumed: {len(results)} runs already complete.")
    else:
        results = {}
        print("\n[5] No previous results — starting fresh.")

    # 6. PLAN + RUN ----------------------------------------------------------
    planned = []
    for sched_name in SCHEDULE_NAMES:
        for solver in SOLVER_LIST:
            for total_nfe in NFE_LIST:
                num_steps = total_nfe // SOLVER_EVALS_PER_STEP[solver]
                if num_steps >= 1:
                    planned.append((sched_name, solver, total_nfe, num_steps))
    print(f"\n[6] Running sweep — {len(planned)} planned (schedule x solver x NFE)")

    for run_i, (sched_name, solver, total_nfe, num_steps) in enumerate(planned, 1):
        key = f"{sched_name}_{solver}_nfe{total_nfe}"
        if key in results:
            print(f"  [{run_i}/{len(planned)}] {key} — already done, skipping.")
            continue

        print(f"\n  [{run_i}/{len(planned)}] {sched_name} | {solver} | "
              f"NFE={total_nfe}  ({num_steps} steps x "
              f"{SOLVER_EVALS_PER_STEP[solver]} evals)")

        if sched_name == "uniform":
            schedule, is_ca = build_uniform_schedule(num_steps), False
        elif sched_name == "log":
            schedule, is_ca = build_log_schedule(num_steps), False
        elif sched_name == "greedy":
            schedule, is_ca = build_greedy_schedule(num_steps, ablation_data), False
        else:  # classaware
            schedule, is_ca = build_classaware_schedules(num_steps, ablation_data), True

        images = generate_samples(model, schedule, labels, x0_all,
                                  solver=solver, is_classaware=is_ca)

        gen_feats = inception.extract(images)
        fid = compute_fid(real_feats, gen_feats)
        n_pr = min(1000, len(gen_feats), len(real_feats))
        precision, recall = compute_precision_recall(
            real_feats[:n_pr], gen_feats[:n_pr], k=3)
        overall_acc, per_class_acc = evaluate_class_accuracy(classifier, images, labels)

        results[key] = {
            "schedule_name": sched_name,
            "solver": solver,
            "total_nfe": total_nfe,
            "num_steps": num_steps,
            "fid": fid,
            "precision": precision,
            "recall": recall,
            "overall_accuracy": overall_acc,
            "per_class_accuracy": per_class_acc,
        }
        print(f"    FID={fid:.2f} | P={precision:.3f} | "
              f"R={recall:.3f} | Acc={overall_acc:.3f}")

        # ---- SAVE + COMMIT AFTER EVERY RUN ----
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        commit_volume()

        del images, gen_feats
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # 7. PLOTS + SUMMARY -----------------------------------------------------
    print("\n[7] Generating plots...")
    generate_plots(results)
    commit_volume()

    print_summary(results)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("DONE.")


# ===========================================================================
# ENTRY
# ===========================================================================
if __name__ == "__main__":
    main()