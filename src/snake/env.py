import gymnasium as gym
import numpy as np
from typing import Optional
from game import Game
from constants import WIDTH, HEIGHT, SIZE

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

        # Observation space [food_x, food_y, head_vel_x, head_vel_y, head_x, head_y]
        # NOTE: should i introduce direction instead of velocity
        # TODO: add body
        velocity = self.game.player.vel
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -velocity, -velocity, 0, 0], dtype=np.float32),
            high=np.array(
                [WIDTH, HEIGHT, velocity, velocity, HEIGHT - SIZE, WIDTH - SIZE],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # 0=up, 1=down, 2=left, 3=right
        self.action_space = gym.spaces.Discrete(4)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            np.array: Observation with food and player positions/velocities
        """
        return np.array(
            [
                self.game.food.x,
                self.game.food.y,
                0,  # TODO: add head velocity x
                0,  # TODO: add head velocity x
                self.game.player.head.x,
                self.game.player.head.y,
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

        self.current_step = 0

        self.game._reset()

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
        self.current_step += 1

        action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        direction = action_map[action]

        self.game._handle_input(direction)

        reward = 0
        terminated = False

        if self.game._collision_check():
            reward = 10
            terminated = True
        elif not self.game.player.is_alive:
            reward = -10
            terminated = True
        else:
            # small reward for proximity of head to food
            distance = (self.game.player.head.x - self.game.food.x) ** 2 + (
                self.game.player.head.y - self.game.food.y
            ) ** 2
            reward -= 0.01 * distance

        truncated = self.current_step >= self.max_steps

        if self._collision_check():
            self.player.eat()
            self._reset()

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.game._render()

    def close(self):
        if self.render_mode == "human":
            self.game.end()
