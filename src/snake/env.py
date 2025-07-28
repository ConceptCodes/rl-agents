import gymnasium as gym
import numpy as np
import pygame
from typing import Optional
from game import Snake, Food, Game
from constants import WIDTH, HEIGHT

# 🎯 What skill should the agent learn? [how to play the game snake]
# 👀 What information does the agent need? [food_pos, head_pos, body_segment_positions]
# 🎮 What actions can the agent take? [Discrete choices: up, down, left, right]
# 🏆 How do we measure success? [food_collision]
# ⏰ When should episodes end? [food_collision, body_collision, maximum steps reached]


class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        self.game = Game(
            title="Snake - RL Training", render_ui=self.render_mode == "human"
        )

        # Observation space [food_x, food_y, head_vel_x, head_vel_y]
        # TODO: add body
        velocity = self.game.player.vel
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -velocity, -velocity], dtype=np.float32),
            high=np.array([WIDTH, HEIGHT, velocity, velocity], dtype=np.float32),
            dtype=np.float32,
        )

        # 0=up, 1=down, 2=left, 3=right
        self.action_space = gym.spaces.Discrete(4)
