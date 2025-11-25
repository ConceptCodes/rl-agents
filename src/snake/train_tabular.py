#!/usr/bin/env python3
"""Tabular Q-learning training script for Snake."""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pygame

from env import SnakeEnv


class SnakeQAgent:
    """Q-learning agent with discretized state space for Snake."""

    def __init__(
        self,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: dict = defaultdict(lambda: np.zeros(action_space_size))

        # Discretization bins for Snake observation space
        # obs: [rel_food_x, rel_food_y, food_x, food_y, head_x, head_y, direction]
        self.bins = [10, 10, 8, 8, 8, 8, 4]
        self.bin_ranges = [
            np.linspace(-400, 400, self.bins[0]),  # rel_food_x
            np.linspace(-300, 300, self.bins[1]),  # rel_food_y
            np.linspace(0, 400, self.bins[2]),  # food_x
            np.linspace(0, 300, self.bins[3]),  # food_y
            np.linspace(0, 400, self.bins[4]),  # head_x
            np.linspace(0, 300, self.bins[5]),  # head_y
            np.linspace(0, 3, self.bins[6]),  # direction
        ]

    def discretize_state(self, obs: np.ndarray) -> tuple:
        """Convert continuous observation to discrete state tuple."""
        return tuple(int(np.digitize(o, r)) for o, r in zip(obs, self.bin_ranges))

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
    save_dir: str = "src/snake/models",
    learning_rate: float = 0.1,
    discount: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
):
    """Train the Q-learning agent on the Snake environment."""
    env = SnakeEnv(render_mode="human" if render else None)
    agent = SnakeQAgent(
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

    print(f"Training Snake Q-learning agent for {episodes} episodes...")
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
    parser = argparse.ArgumentParser(description="Train Snake with Tabular Q-Learning")
    parser.add_argument(
        "--episodes", type=int, default=5000, help="Number of training episodes"
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
        default="src/snake/models",
        help="Directory for saving models",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument(
        "--discount", type=float, default=0.99, help="Discount factor (gamma)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Initial exploration rate"
    )
    parser.add_argument(
        "--epsilon-min", type=float, default=0.05, help="Minimum exploration rate"
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
        temp_agent = SnakeQAgent(action_space_size=4)
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
