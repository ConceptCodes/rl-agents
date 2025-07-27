import gymnasium as gym
import numpy as np
import pygame
from typing import Optional
from game import Player, Ball
from constants import WIDTH, HEIGHT, GREEN, WHITE, BLACK, FPS

# 🎯 What skill should the agent learn? [How to play the game pong]
# 👀 What information does the agent need? [ball_pos, ball_velocity, player_pos, opponent_pos]
# 🎮 What actions can the agent take? [Discrete choices: up, down, stay]
# 🏆 How do we measure success? [Successful hits, points scored]
# ⏰ When should episodes end? [Point scored, maximum steps reached]


class PongEnv(gym.Env):
    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize pygame if rendering
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Pong - RL Training")
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()

        # Initialize game components
        self.player_1 = Player(
            posx=20, posy=HEIGHT // 2 - 50, width=10, height=100, speed=10, color=GREEN
        )
        self.player_2 = Player(
            posx=WIDTH - 30,
            posy=HEIGHT // 2 - 50,
            width=10,
            height=100,
            speed=10,
            color=GREEN,
        )
        self.ball = Ball(WIDTH // 2, HEIGHT // 2, 7, 7, WHITE)

        # Define observation space: [ball_x, ball_y, ball_vel_x, ball_vel_y, player1_y, player2_y]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -20, -20, 0, 0], dtype=np.float32),
            high=np.array(
                [WIDTH, HEIGHT, 20, 20, HEIGHT - 100, HEIGHT - 100], dtype=np.float32
            ),
            dtype=np.float32,
        )

        # Define available actions: 0=stay, 1=up, 2=down
        self.action_space = gym.spaces.Discrete(3)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            np.array: Observation with ball and player positions/velocities
        """
        return np.array(
            [
                self.ball.pos.x,
                self.ball.pos.y,
                self.ball.direction.x * self.ball.speed,
                self.ball.direction.y * self.ball.speed,
                self.player_1.rect.y,
                self.player_2.rect.y,
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with game state details
        """
        return {
            "ball_position": (self.ball.pos.x, self.ball.pos.y),
            "player1_y": self.player_1.rect.y,
            "player2_y": self.player_2.rect.y,
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
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Reset step counter
        self.current_step = 0

        # Reset game components
        self.player_1.rect.y = HEIGHT // 2 - 50
        self.player_2.rect.y = HEIGHT // 2 - 50
        self.ball.reset()

        # Simple AI for player 2 (opponent)
        self._simple_ai_enabled = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0=stay, 1=up, 2=down)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Map action to player movement
        y_factor = 0
        if action == 1:  # Up
            y_factor = -1
        elif action == 2:  # Down
            y_factor = 1
        # action == 0 means stay (y_factor = 0)

        # Update player 1 (agent) position
        self.player_1.update(y_factor)

        # Simple AI for player 2 (opponent) - follows ball
        if self._simple_ai_enabled:
            ball_center_y = self.ball.pos.y
            player2_center_y = self.player_2.rect.y + self.player_2.rect.height // 2

            if ball_center_y < player2_center_y - 10:
                self.player_2.update(-1)
            elif ball_center_y > player2_center_y + 10:
                self.player_2.update(1)

        # Update ball position
        point_scored = self.ball.update()

        # Check for paddle collisions
        if self.ball.get_rect().colliderect(
            self.player_1.rect
        ) or self.ball.get_rect().colliderect(self.player_2.rect):
            self.ball.hit()

        # Calculate reward
        reward = 0
        terminated = False

        if point_scored == 1:  # Player 1 (agent) scored
            reward = 10
            terminated = True
        elif point_scored == -1:  # Player 2 (opponent) scored
            reward = -10
            terminated = True
        else:
            # Small reward for hitting the ball
            if self.ball.get_rect().colliderect(self.player_1.rect):
                reward = 1
            # Small penalty for being far from ball
            distance_penalty = (
                abs(
                    self.ball.pos.y
                    - (self.player_1.rect.y + self.player_1.rect.height // 2)
                )
                / HEIGHT
            )
            reward -= 0.01 * distance_penalty

        # Check if episode should truncate (max steps reached)
        truncated = self.current_step >= self.max_steps

        # Reset ball if point was scored but don't end episode yet
        if point_scored != 0:
            self.ball.reset()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # Fill screen with black
            self.screen.fill(BLACK)

            # Draw players and ball
            self.player_1.display(self.screen)
            self.player_2.display(self.screen)
            self.ball.display(self.screen)

            # Draw center line
            pygame.draw.line(
                self.screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2
            )

            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)

    def close(self):
        """Clean up resources."""
        if self.render_mode == "human":
            pygame.quit()
