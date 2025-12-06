#!/usr/bin/env python3
"""Unified evaluation script for Pong agents (both tabular Q-learning and PPO)."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pygame
import torch

from env import PongEnv


# =============================================================================
# Tabular Q-Learning Evaluation
# =============================================================================


def discretize_state(obs: np.ndarray) -> tuple:
    """Convert continuous normalized observation to discrete state tuple for Q-table lookup.

    obs: [norm_ball_x, norm_ball_y, norm_vel_x, norm_vel_y, norm_p1_y, norm_p2_y]
    """
    ball_x, ball_y, ball_vx, ball_vy, player1_y, player2_y = obs

    # Discretize into bins (adapted for normalized inputs 0-1)
    ball_x_bin = int(ball_x * 8)
    ball_y_bin = int(ball_y * 6)
    
    ball_vx_bin = 1 if ball_vx > 0 else 0
    ball_vy_bin = 1 if ball_vy > 0 else 0
    
    player1_y_bin = int(player1_y * 6)

    return (ball_x_bin, ball_y_bin, ball_vx_bin, ball_vy_bin, player1_y_bin)


def load_q_table(path: Path) -> dict:
    """Load Q-table from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def run_tabular_episode(env: PongEnv, q_table: dict, render: bool) -> tuple:
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


def load_ppo_policy(checkpoint_path: Path, env: PongEnv, device: torch.device):
    """Load PPO policy from checkpoint."""
    from train import PongPolicy

    policy = PongPolicy(env).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def run_ppo_episode(
    env: PongEnv,
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
    elif suffix == ".pkl":
        return "tabular"
    else:
        raise ValueError(
            f"Unknown model format: {suffix}. Expected .pt (PPO) or .pkl (tabular)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pong agent (auto-detects tabular vs PPO from file extension)"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to model checkpoint (.pt for PPO, .pkl for tabular Q-table)",
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

    env = PongEnv(render_mode=render_mode)
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
