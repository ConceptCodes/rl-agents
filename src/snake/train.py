import argparse
import time
import warnings
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
import pufferlib
import pufferlib.pytorch
import pufferlib.vector

from env import make_puffer_snake_env


class SnakePolicy(torch.nn.Module):
    """Simple feedforward policy/value network for Snake."""

    def __init__(self, env):
        super().__init__()
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.n

        self.core = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_dim, 128)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(128, 128)),
            torch.nn.ReLU(),
        )
        self.action_head = pufferlib.pytorch.layer_init(
            torch.nn.Linear(128, act_dim), std=0.01
        )
        self.value_head = torch.nn.Linear(128, 1)

    def forward(self, observations):
        hidden = self.core(observations)
        return self.action_head(hidden), self.value_head(hidden).squeeze(-1)


def _backend_mapping():
    mapping = {
        "serial": pufferlib.vector.Serial,
        "multiprocessing": pufferlib.vector.Multiprocessing,
    }
    if hasattr(pufferlib.vector, "Ray"):
        mapping["ray"] = pufferlib.vector.Ray
    return mapping


def _backend(name: str):
    mapping = _backend_mapping()
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backend '{name}'. Choose from {list(mapping)}"
        ) from exc


class SnakeEnvFactory:
    """Picklable callable wrapper so multiprocessing workers can spawn envs."""

    def __init__(self, render: bool):
        self.render = render

    def __call__(self, buf=None):
        return make_puffer_snake_env(render=self.render, buf=buf)


def build_vecenv(args):
    backend = _backend(args.backend)
    env_factory = SnakeEnvFactory(args.render)

    vecenv = pufferlib.vector.make(
        env_factory,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        backend=backend,
    )
    return vecenv


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _resolve_device(args) -> torch.device:
    if args.force_cpu:
        return torch.device("cpu")

    def mps_available():
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    requested = args.device.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps":
        if mps_available():
            return torch.device("mps")
        warnings.warn("MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    _set_seed(args.seed)
    vecenv = build_vecenv(args)
    device = _resolve_device(args)
    print(f"Using device: {device}")
    policy = SnakePolicy(vecenv.driver_env).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    checkpoint_dir = Path(args.checkpoint_dir)

    obs_dim = vecenv.single_observation_space.shape[0]
    num_envs = vecenv.agents_per_batch
    rollout_steps = args.rollout_steps
    minibatch_size = args.train_batch_size
    total_batch = rollout_steps * num_envs
    if minibatch_size > total_batch:
        raise ValueError(
            f"--train-batch-size ({minibatch_size}) exceeds rollout batch {total_batch}"
        )

    obs_np, _ = pufferlib.vector.reset(vecenv, seed=args.seed)
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    episode_returns = np.zeros(num_envs, dtype=np.float32)
    completed_returns = []
    global_step = 0
    start_time = time.time()

    total_updates = max(args.total_timesteps // total_batch, 1)

    try:
        for update in range(total_updates):
            obs_buf = torch.zeros(
                (rollout_steps, num_envs, obs_dim), dtype=torch.float32, device=device
            )
            actions_buf = torch.zeros(
                (rollout_steps, num_envs), dtype=torch.long, device=device
            )
            logprob_buf = torch.zeros(
                (rollout_steps, num_envs), dtype=torch.float32, device=device
            )
            rewards_buf = torch.zeros(
                (rollout_steps, num_envs), dtype=torch.float32, device=device
            )
            dones_buf = torch.zeros(
                (rollout_steps, num_envs), dtype=torch.float32, device=device
            )
            values_buf = torch.zeros(
                (rollout_steps, num_envs), dtype=torch.float32, device=device
            )

            for step in range(rollout_steps):
                obs_buf[step] = obs
                dones_buf[step] = next_done

                with torch.no_grad():
                    logits, value = policy(obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)

                actions_buf[step] = action
                logprob_buf[step] = logprob
                values_buf[step] = value

                actions_np = action.detach().cpu().numpy()
                next_obs, reward, terminated, truncated, _ = pufferlib.vector.step(
                    vecenv, actions_np
                )

                reward_np = np.asarray(reward, dtype=np.float32)
                dones_np = np.asarray(terminated) | np.asarray(truncated)
                rewards_buf[step] = torch.tensor(reward_np, device=device)

                episode_returns += reward_np
                for idx, done in enumerate(dones_np):
                    if done:
                        completed_returns.append(episode_returns[idx])
                        episode_returns[idx] = 0.0

                next_done = torch.tensor(dones_np, dtype=torch.float32, device=device)
                obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
                global_step += num_envs

            with torch.no_grad():
                _, next_value = policy(obs)

            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = torch.zeros(num_envs, dtype=torch.float32, device=device)
            for step in reversed(range(rollout_steps)):
                if step == rollout_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones_buf[step + 1]
                    next_values = values_buf[step + 1]
                delta = (
                    rewards_buf[step]
                    + args.gamma * next_values * next_nonterminal
                    - values_buf[step]
                )
                lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                )
                advantages[step] = lastgaelam
            returns = advantages + values_buf

            b_obs = obs_buf.reshape(-1, obs_dim)
            b_actions = actions_buf.reshape(-1)
            b_logprob = logprob_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)

            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-8
            )

            clipfracs = []
            for epoch in range(args.update_epochs):
                inds = np.arange(total_batch)
                np.random.shuffle(inds)
                for start in range(0, total_batch, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = torch.tensor(
                        inds[start:end], dtype=torch.long, device=device
                    )

                    logits, value = policy(b_obs[mb_inds])
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logprob = dist.log_prob(b_actions[mb_inds])
                    entropy = dist.entropy().mean()

                    ratio = (new_logprob - b_logprob[mb_inds]).exp()
                    with torch.no_grad():
                        clipfracs.append(
                            ((ratio - 1.0).abs() > args.clip_coef)
                            .float()
                            .mean()
                            .cpu()
                            .item()
                        )

                    pg_loss1 = -b_advantages[mb_inds] * ratio
                    pg_loss2 = -b_advantages[mb_inds] * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    value_pred = value.view(-1)
                    value_loss = 0.5 * (b_returns[mb_inds] - value_pred).pow(2).mean()

                    loss = (
                        pg_loss
                        - args.ent_coef * entropy
                        + args.vf_coef * value_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), args.max_grad_norm
                    )
                    optimizer.step()

            explained_var = (
                1
                - torch.var(b_returns - b_values) / (torch.var(b_returns) + 1e-8)
            ).item()
            sps = int(global_step / (time.time() - start_time + 1e-8))

            if (update + 1) % args.log_interval == 0:
                recent = (
                    sum(completed_returns[-10:]) / len(completed_returns[-10:])
                    if completed_returns
                    else 0.0
                )
                print(
                    f"Update {update+1}/{total_updates} | "
                    f"Step {global_step} | "
                    f"SPS {sps} | "
                    f"Mean Rollout Reward {rewards_buf.mean().item():.3f} | "
                    f"Recent Episode Return {recent:.2f} | "
                    f"ExplainedVar {explained_var:.3f} | "
                    f"ClipFrac {np.mean(clipfracs):.3f}"
                )

            if (update + 1) % args.save_interval == 0 or (update + 1) == total_updates:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = checkpoint_dir / f"snake_ppo_update_{update+1}.pt"
                torch.save(
                    {
                        "model": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "update": update + 1,
                        "global_step": global_step,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")
    finally:
        with suppress(Exception):
            vecenv.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Snake with PufferLib PPO")
    backend_choices = sorted(_backend_mapping().keys())
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of env replicas")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Worker processes/threads for vector env",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Envs per recv batch")
    parser.add_argument(
        "--backend",
        type=str,
        default="multiprocessing" if "multiprocessing" in backend_choices else backend_choices[0],
        choices=backend_choices,
        help="Vectorization backend",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device to run policy on (auto prefers CUDA, then MPS)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total PPO timesteps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1024,
        help="Samples per PPO minibatch",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=256,
        help="Environment steps per update",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="Number of PPO epochs per update",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="Clipping parameter for PPO ratio",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value loss coefficient",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda parameter",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How often to print training stats (updates)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/snake",
        help="Directory for saving PPO checkpoints",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=25,
        help="Save checkpoint every N updates",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable human rendering for the driver environment",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable CUDA even if available",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
