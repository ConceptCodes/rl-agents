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

        # Observation space:
        # [rel_food_x, rel_food_y, food_x, food_y, head_x, head_y, direction,
        #  danger_up, danger_down, danger_left, danger_right]
        # Normalized to [-1, 1] or [0, 1] range
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # 0=up, 1=down, 2=left, 3=right
        self.action_space = gym.spaces.Discrete(4)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            np.array: Observation with relative food, player positions, and danger signals (normalized)
        """
        dir_map = {"up": 0, "down": 1, "left": 2, "right": 3}

        # Normalize coordinates
        rel_food_x = (self.game.food.x - self.game.player.head.x) / WIDTH
        rel_food_y = (self.game.food.y - self.game.player.head.y) / HEIGHT

        norm_food_x = self.game.food.x / WIDTH
        norm_food_y = self.game.food.y / HEIGHT

        norm_head_x = self.game.player.head.x / WIDTH
        norm_head_y = self.game.player.head.y / HEIGHT

        norm_dir = dir_map[self.game.player.direction] / 3.0

        # Get danger signals for all 4 directions
        danger_up, danger_down, danger_left, danger_right = self._get_danger_signals()

        return np.array(
            [
                rel_food_x,
                rel_food_y,
                norm_food_x,
                norm_food_y,
                norm_head_x,
                norm_head_y,
                norm_dir,
                danger_up,
                danger_down,
                danger_left,
                danger_right,
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
        if self.render_mode == "human":
            self.clock.tick(FPS)
        self.current_step += 1

        action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        direction = action_map[action]

        self.game._handle_input(direction)

        # Calculate distance BEFORE move
        prev_dist = self._normalized_distance()

        self.game.player.move()

        # Calculate distance AFTER move
        new_dist = self._normalized_distance()

        # Basic step penalty
        reward = -0.01

        # Reward Shaping: Reward moving towards food, punish moving away
        # Scaling factor 10.0 gives roughly +/- 0.3 reward per step for moving closer/further
        dist_reward = (prev_dist - new_dist) * 10.0
        reward += dist_reward

        terminated = False

        # Check for food collision FIRST (before death check)
        if self.game._collision_check():
            reward = 10.0  # Large reward for eating food
            self.game.player.eat()
            self.game._reset_food()

        # Check for death (only if food wasn't eaten)
        if not self.game.player.is_alive:
            reward = -10.0  # Large penalty for dying
            terminated = True

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

    def _normalized_distance(self):
        max_x = max(WIDTH - SIZE, SIZE)
        max_y = max(HEIGHT - SIZE, SIZE)
        dx = abs(self.game.player.head.x - self.game.food.x) / max_x
        dy = abs(self.game.player.head.y - self.game.food.y) / max_y
        return (dx + dy) / 2.0

    def _velocity_alignment(self):
        dir_vectors = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }
        head = self.game.player.head
        food = self.game.food
        to_food = (food.x - head.x, food.y - head.y)
        norm = max((to_food[0] ** 2 + to_food[1] ** 2) ** 0.5, 1e-6)
        unit_to_food = (to_food[0] / norm, to_food[1] / norm)
        dir_vector = dir_vectors[self.game.player.direction]
        return dir_vector[0] * unit_to_food[0] + dir_vector[1] * unit_to_food[1]

    def _next_move_safe(self):
        dir_vectors = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }
        head = self.game.player.head
        dx, dy = dir_vectors[self.game.player.direction]
        next_x = head.x + dx * SIZE
        next_y = head.y + dy * SIZE
        # Boundary check
        if not (0 <= next_x < WIDTH and 0 <= next_y < HEIGHT):
            return False
        # Self collision check
        for segment in self.game.player.body[1:]:
            if segment.x == next_x and segment.y == next_y:
                return False
        return True

    def _get_danger_signals(self):
        """Check if immediate adjacent cells are dangerous (wall or body).

        Returns:
            tuple: (danger_up, danger_down, danger_left, danger_right) as floats (0.0 or 1.0)
        """
        head = self.game.player.head
        body_positions = set()
        for segment in self.game.player.body[1:]:
            body_positions.add((segment.x, segment.y))

        def is_dangerous(x, y):
            # Wall collision
            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                return 1.0
            # Body collision
            if (x, y) in body_positions:
                return 1.0
            return 0.0

        danger_up = is_dangerous(head.x, head.y - SIZE)
        danger_down = is_dangerous(head.x, head.y + SIZE)
        danger_left = is_dangerous(head.x - SIZE, head.y)
        danger_right = is_dangerous(head.x + SIZE, head.y)

        return danger_up, danger_down, danger_left, danger_right


def make_puffer_snake_env(render: bool = False, buf=None):
    """Return a PufferLib-compatible wrapper around :class:`SnakeEnv`.

    Args:
        render: If True, enable the UI in the wrapped environment.

    Returns:
        pufferlib.emulation.GymnasiumPufferEnv: Environment ready for PufferLib.
    """

    try:
        import pufferlib.emulation
    except ImportError as exc:
        raise RuntimeError(
            "pufferlib must be installed to build the Puffer snake environment"
        ) from exc

    base_env = SnakeEnv(render_mode="human" if render else None)
    return pufferlib.emulation.GymnasiumPufferEnv(base_env, buf=buf)
