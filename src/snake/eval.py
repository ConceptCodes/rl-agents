#!/usr/bin/env python3
"""Unified evaluation script for Snake agents (both tabular Q-learning and PPO)."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pygame
import torch

from env import SnakeEnv


# =============================================================================
# Tabular Q-Learning Evaluation
# =============================================================================


def discretize_state(obs: np.ndarray) -> tuple:
    """Convert continuous observation to discrete state tuple for Q-table lookup."""
    bins = [10, 10, 8, 8, 8, 8, 4]
    bin_ranges = [
        np.linspace(-400, 400, bins[0]),  # rel_food_x
        np.linspace(-300, 300, bins[1]),  # rel_food_y
        np.linspace(0, 400, bins[2]),  # food_x
        np.linspace(0, 300, bins[3]),  # food_y
        np.linspace(0, 400, bins[4]),  # head_x
        np.linspace(0, 300, bins[5]),  # head_y
        np.linspace(0, 3, bins[6]),  # direction
    ]
    return tuple(int(np.digitize(o, r)) for o, r in zip(obs, bin_ranges))


def load_q_table(path: Path) -> dict:
    """Load Q-table from pickle or numpy file."""
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    else:  # .npy format (legacy)
        return np.load(path, allow_pickle=True).item()


def run_tabular_episode(env: SnakeEnv, q_table: dict, render: bool) -> tuple:
    """Run one episode with tabular Q-learning agent."""
    obs, _ = env.reset()
    state = discretize_state(obs)

    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return total_reward, steps, True  # interrupted

        # Greedy action selection
        if state in q_table:
            action = int(np.argmax(q_table[state]))
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        state = discretize_state(obs)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        if render:
            env.render()

    return total_reward, steps, False


# =============================================================================
# PPO (Deep RL) Evaluation
# =============================================================================


def _resolve_device(device_option: str) -> torch.device:
    """Resolve the torch device to use."""
    device_option = device_option.lower()

    def mps_available():
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if device_option == "cpu":
        return torch.device("cpu")
    if device_option == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    if device_option == "mps":
        if mps_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ppo_policy(checkpoint_path: Path, env: SnakeEnv, device: torch.device):
    """Load PPO policy from checkpoint."""
    from train import SnakePolicy

    # Create a wrapper that provides single_observation_space attribute
    class EnvWrapper:
        def __init__(self, env):
            self.env = env
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space

    wrapped_env = EnvWrapper(env)
    policy = SnakePolicy(wrapped_env).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def run_ppo_episode(
    env: SnakeEnv,
    policy,
    device: torch.device,
    greedy: bool,
    render: bool,
) -> tuple:
    """Run one episode with PPO agent."""
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return total_reward, steps, True  # interrupted

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(obs_tensor)
            if greedy:
                action = torch.argmax(logits, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        if render:
            env.render()

    return total_reward, steps, False


# =============================================================================
# Main
# =============================================================================


def detect_model_type(checkpoint_path: Path) -> str:
    """Detect model type from file extension."""
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".pt":
        return "ppo"
    elif suffix in (".pkl", ".npy"):
        return "tabular"
    else:
        raise ValueError(
            f"Unknown model format: {suffix}. Expected .pt (PPO) or .pkl/.npy (tabular)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Snake agent (auto-detects tabular vs PPO from file extension)"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to model checkpoint (.pt for PPO, .pkl/.npy for tabular Q-table)",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for PPO inference (ignored for tabular)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy/deterministic policy (argmax instead of sampling)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (headless evaluation)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model_type = detect_model_type(args.checkpoint)
    render = not args.no_render
    render_mode = "human" if render else None

    print(f"Model type: {model_type.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes:   {args.episodes}")
    print(f"Render:     {render}")
    print()

    env = SnakeEnv(render_mode=render_mode)
    episode_rewards = []
    episode_lengths = []

    if model_type == "tabular":
        q_table = load_q_table(args.checkpoint)
        print(f"Q-table loaded with {len(q_table)} states")

        for episode in range(1, args.episodes + 1):
            reward, steps, interrupted = run_tabular_episode(env, q_table, render)
            if interrupted:
                break
            episode_rewards.append(reward)
            episode_lengths.append(steps)
            print(f"Episode {episode}: Reward = {reward:.2f}, Steps = {steps}")

    else:  # PPO
        device = _resolve_device(args.device)
        print(f"Using device: {device}")
        policy = load_ppo_policy(args.checkpoint, env, device)

        for episode in range(1, args.episodes + 1):
            reward, steps, interrupted = run_ppo_episode(
                env, policy, device, args.greedy, render
            )
            if interrupted:
                break
            episode_rewards.append(reward)
            episode_lengths.append(steps)
            print(f"Episode {episode}: Reward = {reward:.2f}, Steps = {steps}")

    env.close()

    # Summary
    if episode_rewards:
        print(f"\n{'=' * 40}")
        print(f"Evaluation Summary ({len(episode_rewards)} episodes)")
        print(f"{'=' * 40}")
        print(f"Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"Std Reward:     {np.std(episode_rewards):.2f}")
        print(f"Average Steps:  {np.mean(episode_lengths):.1f}")


if __name__ == "__main__":
    main()
