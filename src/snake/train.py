import gymnasium as gym
from env import SnakeEnv
import numpy as np
import pygame
from collections import defaultdict


def discretize(obs, bins):
    # obs: [rel_food_x, rel_food_y, head_x, head_y, direction]
    # bins: list of bin counts for each dimension
    bin_ranges = [
        np.linspace(-800, 800, bins[0]),  # rel_food_x
        np.linspace(-600, 600, bins[1]),  # rel_food_y
        np.linspace(0, 800, bins[2]),  # head_x
        np.linspace(0, 600, bins[3]),  # head_y
        np.linspace(0, 3, bins[4]),  # direction (0,1,2,3)
    ]
    return tuple(int(np.digitize(o, r)) for o, r in zip(obs, bin_ranges))


def main():

    env = SnakeEnv(render_mode="human")
    num_episodes = 5_000
    alpha = 0.1  # learning rate
    gamma = 0.99  # discount factor
    epsilon = 1.0  # exploration rate
    epsilon_min = 0.05
    epsilon_decay = 0.995
    bins = [15, 15, 10, 10, 4]  # discretization bins for each obs dim (tune as needed)
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    for episode in range(num_episodes):
        obs, info = env.reset()
        state = discretize(obs, bins)
        done = False
        total_reward = 0
        steps = 0
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            pygame.event.pump()
            next_state = discretize(obs, bins)

            # Q-learning update
            best_next = np.max(q_table[next_state])
            q_table[state][action] += alpha * (
                reward + gamma * best_next - q_table[state][action]
            )

            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(
            f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {epsilon:.3f}"
        )
        # print(f"Final Info: {info}")
    env.close()


if __name__ == "__main__":
    main()
