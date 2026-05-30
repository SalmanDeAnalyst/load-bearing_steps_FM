"""
Modal launcher — generate sample IMAGE GRIDS comparing Euler / Heun / RK4
solvers on the trained flow-matching model.  No metrics, no 5000-sample
sweep.  Just clean visual grids for the slides.

Each output PNG:
    rows  = solver (euler, heun, rk4)
    cols  = the 10 CIFAR-10 classes
The same random noise is shared across every row in every PNG, so the
visual differences are purely from solver / schedule choice.

-------------------------------------------------------------------------
RUN (from your local terminal, after the .pt + .json files are on the
'fm-data' volume — same volume you set up before):

    modal run --detach modal_generate_grids.py
    modal app logs fm-solver-images           # watch
    modal volume get fm-data /sample_grids ./grids_local

Outputs in /sample_grids on the volume:
    grid_uniform_nfe4.png    grid_uniform_nfe8.png    grid_uniform_nfe32.png
    grid_greedy_nfe4.png     grid_greedy_nfe8.png     grid_greedy_nfe32.png
-------------------------------------------------------------------------
"""

import modal

app = modal.App("fm-solver-images")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "diffusers==0.30.3",
        "numpy",
        "matplotlib",
    )
    .add_local_file(
        "schedule_evaluation_with_higher_order.py",
        "/root/schedule_evaluation_with_higher_order.py",
    )
)

volume = modal.Volume.from_name("fm-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": volume},
    timeout=60 * 30,        # 30 min cap (sampling is fast)
)
def generate_grids():
    import sys, os, json
    sys.path.insert(0, "/root")

    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import schedule_evaluation_with_higher_order as ev

    CHECKPOINT_PATH = "/data/best_model_ema.pt"
    ABLATION_PATH   = "/data/ablation_results_v2.json"
    OUTPUT_DIR      = "/data/sample_grids"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DEVICE = ev.DEVICE
    print(f"Device: {DEVICE}")

    # ---- Load model ----
    print("Loading model from", CHECKPOINT_PATH)
    model = ev.ClassConditionalUNet().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ---- Load ablation data (needed for the greedy schedule) ----
    print("Loading ablation data from", ABLATION_PATH)
    with open(ABLATION_PATH) as f:
        ablation_data = json.load(f)

    # ---- Fixed noise: one sample per class, shared across every config ----
    torch.manual_seed(42)
    labels = torch.arange(ev.NUM_CLASSES)
    x0 = torch.randn(ev.NUM_CLASSES, ev.CHANNELS, ev.IMAGE_SIZE, ev.IMAGE_SIZE)

    # ---- Configurations to render ----
    SCHEDULES = ["uniform", "greedy"]
    NFES      = [4, 8, 32]
    SOLVERS   = ["euler", "heun", "rk4"]

    def build_schedule(name, num_steps):
        if name == "uniform":
            return ev.build_uniform_schedule(num_steps)
        if name == "greedy":
            return ev.build_greedy_schedule(num_steps, ablation_data)
        raise ValueError(name)

    @torch.no_grad()
    def generate_row(solver, schedule):
        solver_fn = ev.SOLVER_DISPATCH[solver]
        imgs = solver_fn(model, x0.to(DEVICE), labels.to(DEVICE), schedule)
        return imgs.cpu()

    def to_displayable(img_tensor):
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img + 1.0) / 2.0
        return np.clip(img, 0, 1)

    # ---- Generate one grid per (schedule, NFE) ----
    for sched_name in SCHEDULES:
        for total_nfe in NFES:
            print(f"\n[grid] {sched_name} | NFE={total_nfe}")
            fig, axes = plt.subplots(
                len(SOLVERS), ev.NUM_CLASSES,
                figsize=(ev.NUM_CLASSES * 1.3, len(SOLVERS) * 1.6),
            )

            for row, solver in enumerate(SOLVERS):
                num_steps = total_nfe // ev.SOLVER_EVALS_PER_STEP[solver]
                if num_steps < 1:
                    for col in range(ev.NUM_CLASSES):
                        axes[row, col].axis("off")
                    axes[row, 0].text(
                        -0.4, 0.5, f"{solver}\n(no fit)",
                        ha="right", va="center",
                        transform=axes[row, 0].transAxes,
                        fontsize=10, fontweight="bold",
                    )
                    continue

                schedule = build_schedule(sched_name, num_steps)
                imgs = generate_row(solver, schedule)

                for col in range(ev.NUM_CLASSES):
                    axes[row, col].imshow(to_displayable(imgs[col]))
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    if row == 0:
                        axes[row, col].set_title(ev.CIFAR10_CLASSES[col], fontsize=10)
                    if col == 0:
                        axes[row, col].text(
                            -0.4, 0.5, f"{solver}\n({num_steps} steps)",
                            ha="right", va="center",
                            transform=axes[row, col].transAxes,
                            fontsize=10, fontweight="bold",
                        )

            fig.suptitle(
                f"Schedule: {sched_name.upper()}   |   NFE budget = {total_nfe}",
                fontsize=14, fontweight="bold",
            )
            plt.tight_layout()
            out = os.path.join(OUTPUT_DIR, f"grid_{sched_name}_nfe{total_nfe}.png")
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  saved: {out}")

    # ---- Commit volume ----
    try:
        import modal
        modal.Volume.from_name("fm-data").commit()
        print("\nVolume committed.")
    except Exception as e:
        print(f"\n(commit skipped: {e})")

    print("\nALL DONE.")


@app.local_entrypoint()
def main():
    call = generate_grids.spawn()
    print("=" * 64)
    print(f"Grid generation spawned.  Call ID: {call.object_id}")
    print("Running in the background on Modal — close the terminal anytime.")
    print()
    print("  Watch logs:  modal app logs fm-solver-images")
    print("  Download:    modal volume get fm-data /sample_grids ./grids_local")
    print("=" * 64)
