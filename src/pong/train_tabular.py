#!/usr/bin/env python3

import numpy as np
from env import PongEnv
import random


class SimpleQAgent:
    """A simple Q-learning agent."""

    def __init__(
        self, action_space_size, learning_rate=0.1, epsilon=0.1, discount=0.95
    ):
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount

        # Simple state discretization for Q-table
        self.q_table = {}

    def discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table."""
        ball_x, ball_y, ball_vx, ball_vy, player1_y, player2_y = observation

        # Discretize positions into bins
        ball_x_bin = int(ball_x // 50)
        ball_y_bin = int(ball_y // 50)
        ball_vx_bin = 1 if ball_vx > 0 else 0  # Ball moving left or right
        player1_y_bin = int(player1_y // 50)

        return (ball_x_bin, ball_y_bin, ball_vx_bin, player1_y_bin)

    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)

        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_space_size

        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        """Update Q-values using Q-learning update rule."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_space_size

        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * self.action_space_size

        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error


def train_agent(episodes=1000, render=False):
    """Train the agent on the Pong environment."""
    env = PongEnv(render_mode="human" if render else None, max_steps=1000)
    agent = SimpleQAgent(action_space_size=env.action_space.n)

    episode_rewards = []
    training_interrupted = False

    print(f"Training for {episodes} episodes...")
    if render:
        print("Close the pygame window to stop training early.")

    for episode in range(episodes):
        observation, info = env.reset()
        state = agent.discretize_state(observation)

        total_reward = 0
        step_count = 0

        while True:
            # Handle pygame events if rendering
            if render:
                import pygame

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        training_interrupted = True
                        break

                if training_interrupted:
                    break

            action = agent.get_action(state)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_observation)

            # Update agent
            agent.update(state, action, reward, next_state)

            total_reward += reward
            step_count += 1
            state = next_state

            if render:
                env.render()

            if terminated or truncated:
                break

        if training_interrupted:
            print(f"\nTraining interrupted by user at episode {episode + 1}")
            break

        episode_rewards.append(total_reward)

        # Decay epsilon for exploration
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.995

        # Print progress
        if (episode + 1) % 10 == 0:  # More frequent updates for visual training
            avg_reward = (
                np.mean(episode_rewards[-10:])
                if len(episode_rewards) >= 10
                else np.mean(episode_rewards)
            )
            print(
                f"Episode {episode + 1}: Avg Reward (last 10): {avg_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.3f}, Steps: {step_count}"
            )

    env.close()
    return agent, episode_rewards


def test_trained_agent(agent, episodes=5):
    """Test the trained agent with rendering."""
    env = PongEnv(render_mode="human", max_steps=1000)

    print(f"Testing trained agent for {episodes} episodes...")

    for episode in range(episodes):
        observation, info = env.reset()
        state = agent.discretize_state(observation)

        total_reward = 0
        step_count = 0

        # Use greedy policy (no exploration)
        agent.epsilon = 0

        while True:
            action = agent.get_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            state = agent.discretize_state(observation)

            total_reward += reward
            step_count += 1

            env.render()

            if terminated or truncated:
                break

        print(
            f"Test Episode {episode + 1}: Reward: {total_reward:.2f}, Steps: {step_count}"
        )

    env.close()


def main():
    """Main training function."""
    print("=== Pong RL Training ===")

    # Ask if user wants to watch training
    try:
        response = (
            input("Would you like to watch the training visually? (y/n): ")
            .lower()
            .strip()
        )
        render_training = response == "y"

        if render_training:
            episodes = 100  # Fewer episodes for visual training
            print("Training with visual rendering - you can watch the agent learn!")
            print("The left paddle is controlled by the AI agent.")
            print("Close the window to stop training early.")
        else:
            episodes = 500  # More episodes for faster training
            print("Training without rendering for faster learning...")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return

    # Train the agent
    agent, rewards = train_agent(episodes=episodes, render=render_training)

    print(f"\nTraining completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")

    # Test the trained agent
    if not render_training:  # Only ask if we didn't already render during training
        try:
            response = (
                input("\nWould you like to test the trained agent visually? (y/n): ")
                .lower()
                .strip()
            )
            if response == "y":
                test_trained_agent(agent, episodes=3)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")

    print("Training session completed!")


if __name__ == "__main__":
    main()
