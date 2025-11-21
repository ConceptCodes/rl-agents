# RL Agents

this repository contains various reinforcement learning agents implemented in Python. The agents are designed to learn and adapt to different environments, including classic control tasks and custom game environments.

## Modules
- [**Snake**](src/snake): A reinforcement learning agent for playing the Snake game using the Gymnasium library.
- [**Pong**](src/pong): A reinforcement learning agent for playing the Pong game using the Gymnasium library.

## Development with `uv`
This repository now uses [uv](https://docs.astral.sh/uv/) for environment management and dependency locking.

1. Install uv (see the official docs for platform-specific installers).
2. Create the virtual environment and install dependencies (uv will automatically download Python 3.11, which is the supported runtime for this repo):
   ```bash
   uv sync
   ```
3. Run any script directly through uv to ensure it uses the managed environment, e.g.:
   ```bash
   uv run python src/snake/train.py
   ```
4. When you need a new dependency, add it via:
   ```bash
   uv add <package-name>
   ```
