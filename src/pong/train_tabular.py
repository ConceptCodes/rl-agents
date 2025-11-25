#!/usr/bin/env python3
"""Tabular Q-learning training script for Pong."""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pygame

from env import PongEnv


class PongQAgent:
    """Q-learning agent with discretized state space for Pong."""

    def __init__(
        self,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: dict = defaultdict(lambda: np.zeros(action_space_size))

    def discretize_state(self, obs: np.ndarray) -> tuple:
        """Convert continuous observation to discrete state tuple.

        obs: [ball_x, ball_y, ball_vel_x, ball_vel_y, player1_y, player2_y]
        """
        ball_x, ball_y, ball_vx, ball_vy, player1_y, player2_y = obs

        # Discretize into bins
        ball_x_bin = int(ball_x // 50)
        ball_y_bin = int(ball_y // 50)
        ball_vx_bin = 1 if ball_vx > 0 else 0  # Ball moving right or left
        ball_vy_bin = 1 if ball_vy > 0 else 0  # Ball moving down or up
        player1_y_bin = int(player1_y // 50)

        return (ball_x_bin, ball_y_bin, ball_vx_bin, ball_vy_bin, player1_y_bin)

    def get_action(self, state: tuple) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space_size)
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ):
        """Update Q-value using Q-learning update rule."""
        if done:
            target = reward
        else:
            target = reward + self.discount * np.max(self.q_table[next_state])

        td_error = target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, path: str):
        """Save Q-table to file."""
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Saved Q-table to {path}")

    def load(self, path: str):
        """Load Q-table from file."""
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size), loaded)
        print(f"Loaded Q-table from {path}")


def train_agent(
    episodes: int = 1000,
    render: bool = False,
    save_interval: int = 500,
    save_dir: str = "src/pong/models",
    learning_rate: float = 0.1,
    discount: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
):
    """Train the Q-learning agent on the Pong environment."""
    env = PongEnv(render_mode="human" if render else None)
    agent = PongQAgent(
        action_space_size=env.action_space.n,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    episode_rewards = []
    episode_lengths = []

    print(f"Training Pong Q-learning agent for {episodes} episodes...")
    if render:
        print("Rendering enabled. Close the pygame window to stop early.")

    for episode in range(episodes):
        obs, _ = env.reset()
        state = agent.discretize_state(obs)

        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print(f"\nTraining interrupted at episode {episode + 1}")
                        env.close()
                        return agent, episode_rewards

            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.discretize_state(next_obs)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Logging every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(
                f"Episode {episode + 1:5d} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Steps: {avg_length:6.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        # Checkpoint saving
        if (episode + 1) % save_interval == 0:
            ckpt_path = save_path / f"q_table_ep{episode + 1}.pkl"
            agent.save(str(ckpt_path))

    # Save final model
    final_path = save_path / "q_table_final.pkl"
    agent.save(str(final_path))

    env.close()
    return agent, episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Train Pong with Tabular Q-Learning")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render during training (slower)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="src/pong/models",
        help="Directory for saving models",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument(
        "--discount", type=float, default=0.95, help="Discount factor (gamma)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Initial exploration rate"
    )
    parser.add_argument(
        "--epsilon-min", type=float, default=0.01, help="Minimum exploration rate"
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.995, help="Epsilon decay per episode"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to load existing Q-table to continue training",
    )
    args = parser.parse_args()

    initial_epsilon = args.epsilon

    if args.load:
        # Load existing Q-table and continue training
        temp_agent = PongQAgent(action_space_size=3)
        temp_agent.load(args.load)
        initial_epsilon = (
            temp_agent.epsilon if hasattr(temp_agent, "epsilon") else args.epsilon_min
        )

    agent, rewards = train_agent(
        episodes=args.episodes,
        render=args.render,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        learning_rate=args.lr,
        discount=args.discount,
        epsilon=initial_epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    print(f"\nTraining complete!")
    if rewards:
        print(f"Final average reward (last 100): {np.mean(rewards[-100:]):.2f}")


if __name__ == "__main__":
    main()
