import argparse
from pathlib import Path

import torch

from env import SnakeEnv
from train import SnakePolicy


def load_checkpoint(policy: SnakePolicy, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    policy.load_state_dict(state_dict)
    return checkpoint


def run_episode(env: SnakeEnv, policy: SnakePolicy, device: torch.device, greedy: bool):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(obs_tensor)
            if greedy:
                action = torch.argmax(logits, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
        obs, reward, terminated, truncated, _ = env.step(action)
        if env.render_mode == "human":
            env.render()
        total_reward += reward
        done = terminated or truncated
    return total_reward


def _resolve_device(device_option: str) -> torch.device:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Snake PPO checkpoint")
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint produced by train.py",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of rendered evaluation episodes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run policy on",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use argmax policy instead of sampling",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable windowed rendering (useful for headless validation)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    render_mode = None if args.no_render else "human"

    env = SnakeEnv(render_mode=render_mode)
    policy = SnakePolicy(env).to(device)
    load_checkpoint(policy, args.checkpoint, device)
    policy.eval()

    for episode in range(1, args.episodes + 1):
        reward = run_episode(env, policy, device, greedy=args.greedy)
        print(f"Episode {episode}: reward={reward:.2f}")

    env.close()
if __name__ == "__main__":
    main()
