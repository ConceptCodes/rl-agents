import numpy as np
import pygame
from env import SnakeEnv
import sys
import os


def discretize(obs, bins):
    bin_ranges = [
        np.linspace(-800, 800, bins[0]),
        np.linspace(-600, 600, bins[1]),
        np.linspace(0, 800, bins[2]),
        np.linspace(0, 600, bins[3]),
        np.linspace(0, 800, bins[4]),
        np.linspace(0, 600, bins[5]),
        np.linspace(0, 3, bins[6]),
    ]
    return tuple(np.digitize(o, r) for o, r in zip(obs, bin_ranges))


def main():
    model_dir = "src/snake/models"
    q_files = [
        f
        for f in os.listdir(model_dir)
        if f.startswith("q_table_ep") and f.endswith(".npy")
    ]
    if not q_files:
        print("No Q-table found in models directory.")
        sys.exit(1)
    latest = sorted(q_files, key=lambda x: int(x.split("ep")[-1].split(".")[0]))[-1]
    q_path = os.path.join(model_dir, latest)
    print(f"Loading Q-table: {q_path}")
    q_table = np.load(q_path, allow_pickle=True).item()

    bins = [10, 10, 8, 8, 8, 8, 4]
    env = SnakeEnv(render_mode="human")
    num_episodes = 10
    total_rewards = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = discretize(obs, bins)
        done = False
        total_reward = 0
        steps = 0
        while not done:
            if state in q_table:
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            pygame.event.pump()
            state = discretize(obs, bins)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        total_rewards.append(total_reward)
        print(
            f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {steps}"
        )
    env.close()
    print(f"Average Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    main()
