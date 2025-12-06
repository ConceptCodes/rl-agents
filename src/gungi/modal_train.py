#!/usr/bin/env python3
"""Modal deployment script for Gungi AlphaZero training on cloud GPU."""

import subprocess
import sys
from typing import Optional

try:
    import modal
except ImportError:
    print("Modal not installed. Install with: pip install modal")
    sys.exit(1)

# Define the Modal app
app = modal.App("gungi-alphazero")

# Create GPU image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(
        "libsdl2-dev", "libsdl2-image-dev", "libsdl2-mixer-dev", "libsdl2-ttf-dev"
    )
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "tensorboard>=2.12",
        "pygame>=2.5",
    )
    .workdir("/root")
    .add_local_dir("src/gungi", "/root/gungi")
)

# Define volumes for persistent storage
checkpoints_vol = modal.Volume.from_name("gungi-checkpoints", create_if_missing=True)
logs_vol = modal.Volume.from_name("gungi-logs", create_if_missing=True)


def run_training(
    iterations: int,
    games_per_iter: int,
    mcts_sims: int,
    batch_size: int,
    epochs: int,
    network_size: str,
    workers: int,
    resume_from: Optional[str] = None,
) -> bool:
    """Run Gungi AlphaZero training."""
    import os

    cmd = [
        sys.executable,
        "/root/gungi/train.py",
        "--iterations",
        str(iterations),
        "--games-per-iter",
        str(games_per_iter),
        "--mcts-sims",
        str(mcts_sims),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--device",
        "cuda",
        "--network-size",
        network_size,
        "--checkpoint-dir",
        "/root/checkpoints/gungi",
        "--log-dir",
        "/root/runs/gungi",
        "--workers",
        str(workers),
    ]

    if resume_from:
        cmd.extend(["--resume", resume_from])

    print(f"Starting Gungi training...")
    print(f"Command: {' '.join(cmd)}\n")

    env = {
        **os.environ,
        "PYTHONPATH": "/root/gungi",
        "SDL_AUDIODRIVER": "dummy",
        "SDL_VIDEODRIVER": "dummy",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1",
    }
    result = subprocess.run(cmd, cwd="/root/gungi", env=env)
    return result.returncode == 0


@app.function(
    image=image,
    gpu="a10g",
    timeout=86400,
    volumes={
        "/root/checkpoints": checkpoints_vol,
        "/root/runs": logs_vol,
    },
)
def train_a10g(
    iterations: int,
    games_per_iter: int,
    mcts_sims: int,
    batch_size: int,
    epochs: int,
    network_size: str,
    workers: int,
    resume_from: Optional[str] = None,
) -> bool:
    return run_training(
        iterations,
        games_per_iter,
        mcts_sims,
        batch_size,
        epochs,
        network_size,
        workers,
        resume_from,
    )


@app.function(
    image=image,
    gpu="a100",
    timeout=86400,
    volumes={
        "/root/checkpoints": checkpoints_vol,
        "/root/runs": logs_vol,
    },
)
def train_a100(
    iterations: int,
    games_per_iter: int,
    mcts_sims: int,
    batch_size: int,
    epochs: int,
    network_size: str,
    workers: int,
    resume_from: Optional[str] = None,
) -> bool:
    return run_training(
        iterations,
        games_per_iter,
        mcts_sims,
        batch_size,
        epochs,
        network_size,
        workers,
        resume_from,
    )


@app.function(
    image=image,
    gpu="h100",
    timeout=86400,
    volumes={
        "/root/checkpoints": checkpoints_vol,
        "/root/runs": logs_vol,
    },
)
def train_h100(
    iterations: int,
    games_per_iter: int,
    mcts_sims: int,
    batch_size: int,
    epochs: int,
    network_size: str,
    workers: int,
    resume_from: Optional[str] = None,
) -> bool:
    return run_training(
        iterations,
        games_per_iter,
        mcts_sims,
        batch_size,
        epochs,
        network_size,
        workers,
        resume_from,
    )


@app.local_entrypoint()
def main(
    iterations: int = 50,
    games_per_iter: int = 25,
    mcts_sims: int = 200,
    batch_size: int = 64,
    epochs: int = 10,
    network_size: str = "small",
    workers: int = 4,
    gpu: str = "a10g",
    resume_from: Optional[str] = None,
):
    """
    Run Gungi AlphaZero training on Modal GPU.

    Usage:
        modal run src/gungi/modal_train.py --iterations 50 --games-per-iter 25
        modal run src/gungi/modal_train.py --gpu a100 --iterations 100 --mcts-sims 200
        modal run src/gungi/modal_train.py --gpu h100 --workers 8
        modal run src/gungi/modal_train.py --resume /root/checkpoints/gungi/gungi_alphazero_iter_10.pt

    GPU Options:
        - a10g: Good balance (~$0.30/hr)
        - a100: Faster (~$1.10/hr)
        - h100: Fastest (~$2.00/hr)
    """
    print("=" * 70)
    print("Gungi AlphaZero Training on Modal")
    print("=" * 70)
    print(f"GPU: {gpu}")
    print(f"Iterations: {iterations}")
    print(f"Games per iteration: {games_per_iter}")
    print(f"MCTS simulations: {mcts_sims}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Network size: {network_size}")
    print(f"Workers: {workers}")
    if resume_from:
        print(f"Resume from: {resume_from}")
    print("=" * 70 + "\n")

    # Select the appropriate function based on GPU
    gpu_functions = {
        "a10g": train_a10g,
        "a100": train_a100,
        "h100": train_h100,
    }

    if gpu not in gpu_functions:
        print(
            f"Error: Unknown GPU type '{gpu}'. Choose from: {list(gpu_functions.keys())}"
        )
        sys.exit(1)

    train_fn = gpu_functions[gpu]

    try:
        success = train_fn.remote(
            iterations=iterations,
            games_per_iter=games_per_iter,
            mcts_sims=mcts_sims,
            batch_size=batch_size,
            epochs=epochs,
            network_size=network_size,
            workers=workers,
            resume_from=resume_from,
        )

        if success:
            print("\n" + "=" * 70)
            print("Training completed successfully!")
            print("=" * 70)
            print("\nTo download checkpoints:")
            print("  modal volume get gungi-checkpoints ./local-checkpoints")
            print("\nTo download logs:")
            print("  modal volume get gungi-logs ./local-logs")
        else:
            print("\nTraining failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)
