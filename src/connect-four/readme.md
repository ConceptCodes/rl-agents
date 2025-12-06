# Connect 4 RL Agent

This module includes both a simple Q-learning baseline and a PufferLib-powered PPO trainer for the custom Connect 4 environment.

## Setup

From the repo root install dependencies via:

```bash
uv sync
```

## Training Options

- **Tabular baseline**: `uv run python src/connect-four/train_tabular.py`
- **PufferLib PPO** (vectorized envs, CUDA/MPS aware):

  ```bash
  uv run python src/connect-four/train.py --num-envs 32 --num-workers 8 --device auto
  ```

  Useful flags:
  - `--checkpoint-dir` and `--save-interval` control automatic checkpointing.
  - `--backend` selects the vector backend (`serial`, `multiprocessing`, optional `ray`).
  - `--render` opens the UI for the driver environment (slower but good for debugging).

## Evaluating a PPO Checkpoint

After training, evaluate a saved policy in a rendered window:

```bash
uv run python src/connect-four/eval.py checkpoints/connect-four/connect-four_ppo_update_400.pt --episodes 5
```

Add `--greedy` for deterministic play or `--no-render` for headless validation.

For the legacy Q-learning artifacts use `train_tabular.py`/`eval_tabular.py`.
