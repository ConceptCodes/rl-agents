# Pong RL Agent

Custom Pong environment plus two training paths:

- `train_tabular.py`: tabular Q-learning baseline (single environment).
- `train.py`: PPO via [PufferLib](https://pufferlib.github.io/) with vectorized environments.

## Setup

All dependencies live in the repository-level `pyproject.toml`. From the repo root:

```bash
uv sync
```

## PPO Training (PufferLib)

Example run using multiprocessing and the auto-selected torch device (CUDA > MPS > CPU):

```bash
uv run python src/pong/train.py --num-envs 16 --num-workers 4 --backend multiprocessing --device auto
```

Useful flags:

- `--checkpoint-dir` / `--save-interval`: control automatic checkpoint emission.
- `--backend`: `serial`, `multiprocessing`, optionally `ray` if installed.
- `--device`: `auto`, `cpu`, `cuda`, or `mps`.
- `--render`: enable UI rendering on the driver environment for debugging.

### Evaluate a Checkpoint

```bash
uv run python src/pong/eval.py checkpoints/pong/pong_ppo_update_400.pt --episodes 5
```

Use `--greedy` for deterministic play or `--no-render` for headless evaluation.

## Legacy Q-learning

To re-run the original baseline:

```bash
uv run python src/pong/train_tabular.py
```

The script asks whether to render training; choose `y` to watch gameplay or `n` for faster headless runs.
