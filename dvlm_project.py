import os
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
EPOCHS = 50
LR = 2e-4

IMAGE_SIZE = 32
CHANNELS = 3

NUM_CLASSES = 10
GUIDANCE_SCALE = 3.0
NFE = 32

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
# CFG WRAPPER
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

        combined_labels = torch.cat(
            [self.labels, self.null_labels], dim=0
        )

        combined_x = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t_batch, t_batch], dim=0)

        v_all = self.model(
            combined_x,
            combined_t,
            combined_labels
        )

        v_cond, v_uncond = torch.chunk(v_all, 2, dim=0)

        return v_uncond + self.guidance_scale * (
            v_cond - v_uncond
        )


# =========================================================
# MAIN
# =========================================================
def main():



    best_loss = float("inf")
    patience = 5
    counter = 0
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

    scaler = torch.cuda.amp.GradScaler()

    FM = ExactOptimalTransportConditionalFlowMatcher(
        sigma=0.0
    )

    history_loss = []

    # =====================================================
    # TRAINING
    # =====================================================
    print("Starting Training...")

    model.train()

    for epoch in range(EPOCHS):

        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

        for x1, labels in pbar:

            x1 = x1.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            x0 = torch.randn_like(x1)

            t, xt, ut = FM.sample_location_and_conditional_flow(
                x0,
                x1
            )

            t = t.to(DEVICE)
            xt = xt.to(DEVICE)
            ut = ut.to(DEVICE)

            # =============================================
            # CLASSIFIER FREE GUIDANCE DROPOUT
            # =============================================
            p_uncond = 0.1

            mask = torch.rand_like(
                labels.float()
            ) < p_uncond

            labels_train = labels.clone()
            labels_train[mask] = NUM_CLASSES

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():

                vt = model(
                    xt,
                    t.squeeze(),
                    labels_train
                )

                loss = ((vt - ut) ** 2).mean()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        avg_epoch_loss = epoch_loss / len(train_loader)

        history_loss.append(avg_epoch_loss)
        # ---------------- EARLY STOPPING ----------------
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            counter = 0

            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        print(
            f"Epoch {epoch+1} | "
            f"Avg Loss: {avg_epoch_loss:.6f}"
        )

        # =================================================
        # SAVE CHECKPOINT
        # =================================================
        if (epoch + 1) % 4 == 0:

            save_path = os.path.join(
                SAVE_DIR,
                f"otcfm_epoch_{epoch+1}.pt"
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            print(f"Saved: {save_path}")

    # =====================================================
    # LOSS CURVE
    # =====================================================
    plt.figure(figsize=(10, 5))

    plt.plot(history_loss)

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")

    plt.savefig("training_loss.png")

    plt.close()

    # =====================================================
    # SAMPLING
    # =====================================================
    print("Generating samples...")

    model.eval()

    with torch.no_grad():

        num_samples = 10

        target_class = torch.randint(
            0,
            10,
            (num_samples,),
            device=DEVICE
        )

        x0 = torch.randn(
            num_samples,
            CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=DEVICE
        )

        ode_model = CFGODEWrapper(
            model,
            target_class,
            GUIDANCE_SCALE
        )

        t_span = torch.linspace(
            0,
            1,
            NFE + 1,
            device=DEVICE
        )

        trajectory = odeint(
            ode_model,
            x0,
            t_span,
            method="euler"
        )

        final_images = trajectory[-1]

    fig, axes = plt.subplots(
        1,
        num_samples,
        figsize=(15, 3)
    )

    for i in range(num_samples):

        img = (
            final_images[i]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            + 1
        ) / 2

        axes[i].imshow(
            torch.clamp(img, 0, 1)
        )

        axes[i].axis("off")

    plt.savefig("final_samples.png")

    plt.close()

    print("Training Complete.")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":

    main()