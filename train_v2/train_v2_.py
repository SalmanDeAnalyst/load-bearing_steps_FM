import os
import copy
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DModel
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================================
# WINDOWS MULTIPROCESSING FIX
# =========================================================
multiprocessing.freeze_support()

# =========================================================
# HARDWARE OPTIMIZATION
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# =========================================================
# HYPERPARAMETERS
# =========================================================
BATCH_SIZE = 128
EPOCHS = 150
LR = 2e-4
WARMUP_STEPS = 1000

IMAGE_SIZE = 32
CHANNELS = 3

NUM_CLASSES = 10
GUIDANCE_SCALE = 3.0
NFE = 32

EMA_DECAY = 0.9999
SAMPLE_EVERY = 10       # generate samples every N epochs to visually track quality
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

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
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),

            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),

            class_embed_type="identity",
        )

        self.class_emb = nn.Embedding(NUM_CLASSES + 1, 128 * 4)

    def forward(self, x, t, class_labels):
        emb = self.class_emb(class_labels)
        return self.unet(x, t, class_labels=emb).sample


# =========================================================
# EMA
# =========================================================
class EMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy that is updated each step:
        shadow = decay * shadow + (1 - decay) * current_weights

    Use the shadow weights for sampling — they produce
    much better images than the raw training weights.
    """
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        # Don't track gradients on shadow
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(),
                                     model.parameters()):
            s_param.data.mul_(self.decay).add_(
                m_param.data, alpha=1.0 - self.decay
            )

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# =========================================================
# LR SCHEDULE: LINEAR WARMUP + COSINE DECAY
# =========================================================
def get_lr_lambda(warmup_steps, total_steps):
    """
    Returns a lambda for LambdaLR:
      - Linear warmup from 0 to LR over warmup_steps
      - Cosine decay from LR to 0 over remaining steps
    """
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# =========================================================
# CFG WRAPPER (for sampling)
# =========================================================
class CFGODEWrapper(nn.Module):
    def __init__(self, model, labels, guidance_scale):
        super().__init__()
        self.model = model
        self.labels = labels
        self.guidance_scale = guidance_scale
        self.null_labels = torch.full_like(labels, NUM_CLASSES)

    def forward(self, t, x):
        t_batch = torch.ones(x.shape[0], device=DEVICE) * t

        combined_x = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t_batch, t_batch], dim=0)
        combined_labels = torch.cat(
            [self.labels, self.null_labels], dim=0
        )

        v_all = self.model(combined_x, combined_t, combined_labels)
        v_cond, v_uncond = torch.chunk(v_all, 2, dim=0)

        return v_uncond + self.guidance_scale * (v_cond - v_uncond)


# =========================================================
# SAMPLE GENERATION (for visual tracking)
# =========================================================
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@torch.no_grad()
def generate_samples(model, epoch, seed=42):
    """Generate one sample per class using fixed noise for comparison across epochs."""
    model.eval()

    # Fixed noise so we can visually compare across epochs
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x0 = torch.randn(
        NUM_CLASSES, CHANNELS, IMAGE_SIZE, IMAGE_SIZE,
        device=DEVICE, generator=gen
    )
    target_class = torch.arange(NUM_CLASSES, device=DEVICE)

    ode_model = CFGODEWrapper(model, target_class, GUIDANCE_SCALE)
    t_span = torch.linspace(0, 1, NFE + 1, device=DEVICE)

    trajectory = odeint(ode_model, x0, t_span, method="euler")
    final_images = trajectory[-1]

    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(20, 3))
    for i in range(NUM_CLASSES):
        img = (final_images[i].cpu().permute(1, 2, 0) + 1) / 2
        axes[i].imshow(torch.clamp(img, 0, 1))
        axes[i].set_title(CIFAR10_CLASSES[i], fontsize=9)
        axes[i].axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"samples_epoch_{epoch}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Samples saved: {path}")


# =========================================================
# MAIN
# =========================================================
def main():

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # =====================================================
    # MODEL INIT
    # =====================================================
    model = ClassConditionalUNet().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )

    # Total training steps for LR schedule
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr_lambda(WARMUP_STEPS, total_steps)
    )

    scaler = torch.amp.GradScaler("cuda")

    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    # ── EMA init ──────────────────────────────────────
    ema = EMA(model, decay=EMA_DECAY)

    history_loss = []
    best_loss = float("inf")
    global_step = 0

    # =====================================================
    # TRAINING
    # =====================================================
    print(f"Starting Training for {EPOCHS} epochs...")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup: {WARMUP_STEPS} steps")
    print(f"  EMA decay: {EMA_DECAY}")
    print()

    for epoch in range(EPOCHS):

        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

        for x1, labels in pbar:

            x1 = x1.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            x0 = torch.randn_like(x1)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            t = t.to(DEVICE)
            xt = xt.to(DEVICE)
            ut = ut.to(DEVICE)

            # ── CFG dropout ───────────────────────────
            p_uncond = 0.1
            mask = torch.rand_like(labels.float()) < p_uncond
            labels_train = labels.clone()
            labels_train[mask] = NUM_CLASSES

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                vt = model(xt, t.squeeze(), labels_train)
                loss = ((vt - ut) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ── Update LR and EMA ─────────────────────
            scheduler.step()
            ema.update(model)

            global_step += 1
            epoch_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_epoch_loss = epoch_loss / steps_per_epoch
        history_loss.append(avg_epoch_loss)

        print(
            f"Epoch {epoch+1} | "
            f"Avg Loss: {avg_epoch_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Save best EMA model ───────────────────────
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(ema.state_dict(), "best_model_ema.pt")
            print(f"  ✓ New best EMA model saved (loss={best_loss:.6f})")

        # ── Periodic sample generation (using EMA) ────
        if (epoch + 1) % SAMPLE_EVERY == 0 or epoch == 0:
            generate_samples(ema.shadow, epoch + 1)

    # ── Final save ────────────────────────────────────
    torch.save(ema.state_dict(), "best_model_ema.pt")
    torch.save(model.state_dict(), "final_model_raw.pt")
    print(f"\nFinal EMA model saved: best_model_ema.pt")
    print(f"Final raw model saved: final_model_raw.pt")

    # =====================================================
    # LOSS CURVE
    # =====================================================
    plt.figure(figsize=(10, 5))
    plt.plot(history_loss)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.savefig("training_loss.png", dpi=150)
    plt.close()

    # =====================================================
    # FINAL SAMPLES (EMA model)
    # =====================================================
    print("Generating final samples with EMA model...")
    generate_samples(ema.shadow, EPOCHS)

    print("Training Complete.")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    main()