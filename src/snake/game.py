import pygame
import random
from constants import GREEN, BLACK, WHITE, SIZE, WIDTH, HEIGHT

pygame.init()


class Snake:
    def __init__(self, x, y):
        self.head = pygame.Rect(x, y, SIZE, SIZE)
        self.color = GREEN
        self.is_alive = True
        self.body = []

    def eat(self, x, y):
        self.body.append(pygame.Rect(x, y, SIZE, SIZE))

    # TODO: review, check for boundary collisions
    def move(self, dir):
        if dir == "up":
            self.head.move(self.head.x, self.head.y + 1)
            self.body = [block.move(block.x, block.y + 1) for block in self.body]
        if dir == "down":
            self.head.move(self.head.x, self.head.y - 1)
            self.body = [block.move(block.x, block.y - 1) for block in self.body]
        if dir == "right":
            self.head.move(self.head.x + 1, self.head.y)
            self.body = [block.move(block.x + 1, block.y) for block in self.body]
        if dir == "left":
            self.head.move(self.head.x - 1, self.head.y)
            self.body = [block.move(block.x - 1, block.y) for block in self.body]

    def render(self, screen):
        pass


class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, SIZE, SIZE)

    def render(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)


class Game:
    def __init__(self, size=10):
        self.size = size
        self.score = 0
        self.is_running = True
        self.grid = self._init_board()
        self.clock = pygame.time.Clock()

        x, y = self._random_pos()
        print(f"Snake pos - x:{x}, y:{y}")
        self.player = Snake(x, y)

        x, y = self._random_pos()
        self.food = Food(x, y)
        print(f"Food pos - x:{x}, y:{y}")

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

    def _random_pos(self):
        return (
            random.randint(0, WIDTH),
            random.randint(0, HEIGHT),
        )

    # TODO: review
    def _init_board(self):
        return [[0] * (WIDTH // self.size)] * (HEIGHT // self.size)

    def _render(self):
        self.screen.fill(BLACK)

        # render food
        self.food.render(self.screen)

        # render snake
        # self.player.render(self.screen)

        pygame.display.flip()

    def begin(self):
        pygame.display.set_caption("Snake")
        self.clock.tick(60)
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.player.move("left")
                    if event.key == pygame.K_RIGHT:
                        self.player.move("right")
                    if event.key == pygame.K_UP:
                        self.player.move("up")
                    if event.key == pygame.K_DOWN:
                        self.player.move("down")

        self._render()


if __name__ == "__main__":
    game = Game()
    game.begin()

    pygame.quit()
    exit()
