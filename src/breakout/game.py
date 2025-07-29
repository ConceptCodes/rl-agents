import pygame
import random
from constants import (
    WIDTH,
    HEIGHT,
    BLACK,
    WHITE,
    PURPLE,
    RED,
    GREEN,
    YELLOW,
    ORANGE,
    BLUE,
    FPS,
    BALL_SPEED,
    BALL_RADIUS,
    PADDLE_LENGTH,
    PADDLE_SPEED,
    PADDLE_HEIGHT,
    GRID_COLUMNS,
    GRID_ROWS,
    BLOCK_SIZE,
)

pygame.init()


class Ball:
    def __init__(self, posx, posy, radius, speed, color):
        self.pos = pygame.Vector2(posx, posy)
        self.radius = radius
        self.speed = speed
        self.color = color
        self.direction = pygame.Vector2(1, -1)
        self.first_time = True

    def display(self, surface):
        pygame.draw.circle(
            surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius
        )

    def update(self):
        self.pos += self.direction * self.speed

        if self.pos.y - self.radius <= 0 or self.pos.y + self.radius >= HEIGHT:
            self.direction.y *= -1

        if self.pos.x - self.radius <= 0 or self.pos.x + self.radius >= WIDTH:
            self.direction.x *= -1

    def reset(self):
        self.pos = pygame.Vector2(WIDTH // 2, HEIGHT // 2)
        self.direction.x *= -1
        self.first_time = True

    def hit(self):
        self.direction.y *= -1

    def get_rect(self):
        return pygame.Rect(
            int(self.pos.x - self.radius),
            int(self.pos.y - self.radius),
            self.radius * 2,
            self.radius * 2,
        )

    def _get_pos(self):
        return self.pos


class Block:
    def __init__(self, color, x, y):
        self.x = x
        self.y = y
        self.color = color
        self.rect = pygame.Rect(x, y, PADDLE_LENGTH, PADDLE_HEIGHT)
        self.should_display = True

    def display(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

    def hit(self):
        self.should_display = False


class Paddle:
    def __init__(
        self,
        x,
        y,
        color=WHITE,
        width=PADDLE_LENGTH,
        height=PADDLE_HEIGHT,
        speed=PADDLE_SPEED,
    ):
        self.direction = "right"
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.speed = speed

    def display(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

    def update(self):
        input_map = {"left": -1, "right": 1}
        self.rect.x += self.speed * input_map[self.direction]
        self.rect.x = max(0, min(self.rect.x, WIDTH - self.rect.width))


class Game:
    def __init__(self, title="Breakout", render_ui=True, record=False):
        self.ball = Ball(WIDTH // 2, HEIGHT // 2, BALL_RADIUS, BALL_SPEED, WHITE)
        self.paddle = Paddle(0, HEIGHT - 100)
        self.blocks = [
            [True for _ in range(WIDTH // GRID_COLUMNS)]
            for _ in range(HEIGHT // GRID_ROWS)
        ]
        self.score = 0

        self.clock = pygame.time.Clock()
        self.is_running = True
        self.render_ui = render_ui
        self.record = record

        if self.render_ui:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(title)

    def _init_blocks(self):
        for row in range(HEIGHT // GRID_ROWS // 4):
            for col in range(WIDTH // GRID_COLUMNS):
                if self.blocks[row][col]:
                    pygame.draw.rect(
                        self.screen,
                        random.choice([RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]),
                        (
                            col * BLOCK_SIZE,
                            row * BLOCK_SIZE,
                            BLOCK_SIZE,
                            BLOCK_SIZE,
                        ),
                    )

    def _handle_input(self, val):
        self.paddle.direction = val

    def reset(self):
        self.score = 0
        self._init_blocks()

    def start(self):
        while self.is_running:
            self.screen.fill(BLACK)
            self.ball.display(self.screen)
            self.paddle.display(self.screen)
            # self._init_blocks()

            self.ball.update()

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self._handle_input("left")
                    if event.key == pygame.K_RIGHT:
                        self._handle_input("right")

            if self.paddle.rect.colliderect(self.ball.get_rect()):
                self.ball.hit()

            self.paddle.update()

            pygame.display.update()
            self.clock.tick(FPS)


if __name__ == "__main__":
    game = Game()
    game.start()
