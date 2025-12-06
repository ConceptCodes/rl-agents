#!/usr/bin/env python3
"""AlphaZero training script for Gungi."""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import copy


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


from network import GungiNetwork, GungiNetworkSmall
from mcts import MCTSConfig
from replay_buffer import ReplayBuffer
from self_play import (
    play_games_parallel,
    evaluate_against_random,
    evaluate_against_network,
)
from encoding import NUM_ACTIONS


class AlphaZeroTrainer:
    """
    AlphaZero training loop for Gungi.

    Alternates between:
    1. Self-play: Generate training data
    2. Training: Update neural network
    3. Evaluation: Test against previous versions
    """

    def __init__(
        self,
        num_iterations: int = 100,
        games_per_iteration: int = 25,
        mcts_simulations: int = 800,  # AlphaZero standard
        batch_size: int = 256,
        replay_buffer_size: int = 100_000,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        num_epochs: int = 10,
        checkpoint_dir: str = "checkpoints/gungi",
        log_dir: str = "runs/gungi",
        device: str = "auto",
        network_size: str = "small",
        num_workers: int = 0,
        eval_games: int = 20,  # Games for network vs network evaluation
        promotion_threshold: float = 0.55,  # Win rate needed to promote new network
    ):
        self.num_iterations = num_iterations
        self.games_per_iteration = games_per_iteration
        self.mcts_simulations = mcts_simulations
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.num_workers = num_workers
        self.eval_games = eval_games
        self.promotion_threshold = promotion_threshold
        self.network_size = network_size

        # Setup device
        self.device = self._resolve_device(device)
        print(f"Using device: {self.device}")

        # Create network
        self.network = self._create_network(network_size)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

        # MCTS config (dirichlet_alpha ~10/avg_legal_moves, Gungi has ~50-200 moves)
        self.mcts_config = MCTSConfig(
            num_simulations=mcts_simulations,
            c_puct=1.5,
            dirichlet_alpha=0.15,  # Reduced for Gungi's larger action space
            dirichlet_epsilon=0.25,
        )

        # Best network tracking (AlphaZero evaluates against previous best)
        self.best_network = self._create_network(network_size)
        self.best_network.load_state_dict(self.network.state_dict())
        self.best_network.eval()

        # Logging
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.current_iteration = 0
        self.total_games = 0
        self.total_steps = 0

    def _create_network(self, network_size: str) -> GungiNetwork:
        """Create a network instance based on size specification."""
        if network_size == "small":
            return GungiNetworkSmall().to(self.device)
        else:
            return GungiNetwork().to(self.device)

    def _resolve_device(self, device_option: str) -> torch.device:
        """Resolve device string to torch.device."""
        device_option = device_option.lower()

        if device_option == "cpu":
            return torch.device("cpu")
        if device_option == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise RuntimeError("CUDA not available")
        if device_option == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            raise RuntimeError("MPS not available")

        # Auto-detect
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def train(self, resume_from: Optional[str] = None):
        """
        Run the full training loop.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        if resume_from:
            self._load_checkpoint(resume_from)

        print(f"Starting AlphaZero training for {self.num_iterations} iterations")
        print(f"  Games per iteration: {self.games_per_iteration}")
        print(f"  MCTS simulations: {self.mcts_simulations}")
        print(f"  Batch size: {self.batch_size}")
        print()

        for iteration in range(self.current_iteration, self.num_iterations):
            self.current_iteration = iteration
            iter_start = time.time()

            print(f"=== Iteration {iteration + 1}/{self.num_iterations} ===")

            # 1. Self-play
            print("  Generating self-play games...")
            self_play_start = time.time()
            records = play_games_parallel(
                network=self.network,
                num_games=self.games_per_iteration,
                mcts_config=self.mcts_config,
                device=self.device,
                verbose=False,
                num_workers=self.num_workers,
            )
            self.replay_buffer.add_game(records)
            self.total_games += self.games_per_iteration
            self_play_time = time.time() - self_play_start
            print(
                f"    Generated {len(records)} samples in {format_duration(self_play_time)}"
            )
            print(f"    Replay buffer size: {len(self.replay_buffer)}")

            # 2. Training
            if len(self.replay_buffer) >= self.batch_size:
                print("  Training network...")
                train_start = time.time()
                metrics = self._train_network()
                train_time = time.time() - train_start
                print(f"    Policy loss: {metrics['policy_loss']:.4f}")
                print(f"    Value loss: {metrics['value_loss']:.4f}")
                print(f"    Training time: {format_duration(train_time)}")

                # Log metrics
                self.writer.add_scalar("Loss/policy", metrics["policy_loss"], iteration)
                self.writer.add_scalar("Loss/value", metrics["value_loss"], iteration)
                self.writer.add_scalar("Loss/total", metrics["total_loss"], iteration)
                self.writer.add_scalar(
                    "Training/buffer_size", len(self.replay_buffer), iteration
                )

            # 3. Evaluation (every 10 iterations)
            if (iteration + 1) % 10 == 0:
                # Evaluate against random baseline
                print("  Evaluating against random...")
                eval_results = evaluate_against_random(
                    self.network,
                    num_games=10,
                    mcts_simulations=100,
                    device=self.device,
                )
                print(f"    Win rate vs random: {eval_results['win_rate']:.1%}")
                self.writer.add_scalar(
                    "Eval/win_rate_vs_random", eval_results["win_rate"], iteration
                )

                # AlphaZero: Evaluate against best network
                print("  Evaluating against best network...")
                vs_best_results = evaluate_against_network(
                    network1=self.network,
                    network2=self.best_network,
                    num_games=self.eval_games,
                    mcts_simulations=100,
                    device=self.device,
                )
                win_rate_vs_best = vs_best_results["net1_win_rate"]
                print(f"    Win rate vs best: {win_rate_vs_best:.1%}")
                self.writer.add_scalar(
                    "Eval/win_rate_vs_best", win_rate_vs_best, iteration
                )

                # Promote new network if it beats the best
                if win_rate_vs_best >= self.promotion_threshold:
                    print(
                        f"    New network promoted (win rate {win_rate_vs_best:.1%} >= {self.promotion_threshold:.1%})"
                    )
                    self.best_network.load_state_dict(self.network.state_dict())
                    self._save_checkpoint(iteration + 1, is_best=True)
                else:
                    print(
                        f"    Network not promoted (win rate {win_rate_vs_best:.1%} < {self.promotion_threshold:.1%})"
                    )

            # 4. Checkpoint (every 10 iterations)
            if (iteration + 1) % 10 == 0:
                self._save_checkpoint(iteration + 1)

            iter_time = time.time() - iter_start
            print(f"  Iteration time: {format_duration(iter_time)}")
            print()

            # Update scheduler
            self.scheduler.step()

        # Save final model
        self._save_checkpoint(self.num_iterations, final=True)
        self.writer.close()
        print("Training complete!")

    def _train_network(self) -> dict:
        """
        Train the network on samples from replay buffer.

        Returns:
            Dict with training metrics
        """
        self.network.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        samples_per_epoch = min(len(self.replay_buffer), self.batch_size * 100)
        batches_per_epoch = samples_per_epoch // self.batch_size

        for epoch in range(self.num_epochs):
            for _ in range(batches_per_epoch):
                # Sample batch
                states, target_policies, target_values = (
                    self.replay_buffer.sample_batch(self.batch_size)
                )

                # Convert to tensors
                states = torch.from_numpy(states).to(self.device)
                target_policies = torch.from_numpy(target_policies).to(self.device)
                target_values = torch.from_numpy(target_values).to(self.device)

                # Forward pass
                policy_logits, values = self.network(states)

                # Policy loss: cross-entropy
                # Use log_softmax + target as probabilities
                log_probs = F.log_softmax(policy_logits, dim=-1)
                policy_loss = -torch.sum(target_policies * log_probs, dim=-1).mean()

                # Value loss: MSE
                value_loss = F.mse_loss(values.squeeze(-1), target_values)

                # Combined loss
                loss = policy_loss + value_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
                self.total_steps += 1

        return {
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss": total_value_loss / max(num_batches, 1),
            "total_loss": (total_policy_loss + total_value_loss) / max(num_batches, 1),
        }

    def _save_checkpoint(
        self, iteration: int, final: bool = False, is_best: bool = False
    ):
        """Save training checkpoint."""
        if final:
            path = self.checkpoint_dir / "gungi_alphazero_final.pt"
        elif is_best:
            path = self.checkpoint_dir / "gungi_alphazero_best.pt"
        else:
            path = self.checkpoint_dir / f"gungi_alphazero_iter_{iteration}.pt"

        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "total_games": self.total_games,
                "total_steps": self.total_steps,
                "mcts_config": {
                    "num_simulations": self.mcts_config.num_simulations,
                    "c_puct": self.mcts_config.c_puct,
                },
                # Network architecture info for loading
                "num_res_blocks": len(self.network.res_blocks),
                "num_channels": self.network.input_conv.out_channels,
                "num_input_planes": self.network.num_input_planes,
                "num_actions": self.network.num_actions,
                "board_size": self.network.board_size,
            },
            path,
        )
        print(f"  Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_iteration = checkpoint.get("iteration", 0)
        self.total_games = checkpoint.get("total_games", 0)
        self.total_steps = checkpoint.get("total_steps", 0)

        print(f"Resumed from iteration {self.current_iteration}")


def main():
    parser = argparse.ArgumentParser(description="Train Gungi with AlphaZero")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument(
        "--games-per-iter", type=int, default=25, help="Self-play games per iteration"
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=800,
        help="MCTS simulations per move (AlphaZero standard)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100000, help="Replay buffer size"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs per iteration"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/gungi",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--log-dir", type=str, default="runs/gungi", help="Tensorboard log directory"
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--network-size", type=str, default="small", choices=["small", "standard"]
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers for self-play (0 = sequential)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=20,
        help="Number of games for network vs network evaluation",
    )
    parser.add_argument(
        "--promotion-threshold",
        type=float,
        default=0.55,
        help="Win rate threshold to promote new network (AlphaZero uses 0.55)",
    )
    args = parser.parse_args()

    trainer = AlphaZeroTrainer(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        mcts_simulations=args.mcts_sims,
        batch_size=args.batch_size,
        replay_buffer_size=args.buffer_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
        network_size=args.network_size,
        num_workers=args.workers,
        eval_games=args.eval_games,
        promotion_threshold=args.promotion_threshold,
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
