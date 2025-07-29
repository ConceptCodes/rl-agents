import gymnasium as gym
import numpy as np
from typing import Optional

import pygame
from game import Game
from constants import WIDTH, HEIGHT, SIZE, FPS

# 🎯 What skill should the agent learn? [how to play the game snake]
# 👀 What information does the agent need? [food_pos, head_pos, distance_from_food]
# 🎮 What actions can the agent take? [Discrete choices: up, down, left, right]
# 🏆 How do we measure success? [food_collision]
# ⏰ When should episodes end? [body_collision, boundary_collision, maximum steps reached]


class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.clock = pygame.time.Clock()

        self.game = Game(
            title="Snake - RL Training", render_ui=self.render_mode == "human"
        )

        adjusted_width = WIDTH - SIZE
        adjusted_height = HEIGHT - SIZE

        # Observation space [rel_food_x, rel_food_y, food_x, food_y, head_x, head_y, direction]
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [-adjusted_width, -adjusted_height, 0, 0, 0, 0, 0], dtype=np.float32
            ),
            high=np.array(
                [
                    adjusted_width,
                    adjusted_height,
                    adjusted_width,
                    adjusted_height,
                    adjusted_width,
                    adjusted_height,
                    3,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # 0=up, 1=down, 2=left, 3=right
        self.action_space = gym.spaces.Discrete(4)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            np.array: Observation with relative food and player positions
        """
        dir_map = {"up": 0, "down": 1, "left": 2, "right": 3}
        rel_food_x = self.game.food.x - self.game.player.head.x
        rel_food_y = self.game.food.y - self.game.player.head.y
        return np.array(
            [
                rel_food_x,
                rel_food_y,
                self.game.food.x,
                self.game.food.y,
                self.game.player.head.x,
                self.game.player.head.y,
                dir_map[self.game.player.direction],
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with game state details
        """
        return {
            "food_position": (self.game.food.x, self.game.food.y),
            "head_position": (self.game.player.head.x, self.game.player.head.y),
            "body_length": len(self.game.player.body),
            "direction": self.game.player.direction,
            "next_direction": self.game.player.next_direction,
            "collision": self.game._collision_check(),
            "alive": self.game.player.is_alive,
            "distance": (self.game.player.head.x - self.game.food.x) ** 2
            + (self.game.player.head.y - self.game.food.y) ** 2,
            "step": self.current_step,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)

        self.game._reset()
        self.current_step = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0=up, 1=down, 2=left, 3=right)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.clock.tick(FPS)
        self.current_step += 1

        action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        direction = action_map[action]

        self.game._handle_input(direction)
        self.game.player.move()

        # Calculate distance to food before and after move
        prev_dist = getattr(self, "_prev_dist", None)
        curr_dist = (self.game.player.head.x - self.game.food.x) ** 2 + (
            self.game.player.head.y - self.game.food.y
        ) ** 2
        self._prev_dist = curr_dist

        reward = -0.01  # small negative reward per step
        terminated = False

        # small reward for body length
        if len(self.game.player.body) > 1:
            reward += 0.1 * (len(self.game.player.body) - 1)

        max_distance = (WIDTH - SIZE) ** 2 + (HEIGHT - SIZE) ** 2

        if self.game._collision_check():
            reward = 10
            self.game.player.eat()
            self.game._reset_food()
        elif not self.game.player.is_alive:
            reward = -10
            terminated = True
        else:
            # Reward for getting closer to food, penalize for moving away
            if prev_dist is not None:
                reward += 0.1 * (prev_dist - curr_dist) / (max_distance + 1e-8)

        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.game._render()
        if self.render_mode == "rgb_array":
            self.game._record()

    def close(self):
        if self.render_mode == "human":
            self.game.end()
