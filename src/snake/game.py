import pygame
import random
from constants import GREEN, BLACK, WHITE, SIZE, WIDTH, HEIGHT

pygame.init()


class Snake:
    def __init__(self, x, y):
        self.is_alive = True
        self.body = [pygame.Rect(x, y, SIZE, SIZE)]
        self.vel = 5
        self.head = self.body[0]

    # TODO: needs review
    def eat(self):
        tail = self.body[-1]
        self.body.append(pygame.Rect(tail.x + 1, tail.y, SIZE, SIZE))

    def move(self, dir):
        if dir == "up":
            self.head.y -= self.vel
        elif dir == "down":
            self.head.y += self.vel
        elif dir == "right":
            self.head.x += self.vel
        elif dir == "left":
            self.head.x -= self.vel

        # Check for boundary collisions
        if (
            self.head.x < 0
            or self.head.x + SIZE > WIDTH
            or self.head.y < 0
            or self.head.y + SIZE > HEIGHT
        ):
            self.is_alive = False

    def render(self, screen):
        for block in self.body:
            pygame.draw.rect(screen, GREEN, block)


class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, SIZE, SIZE)

    def render(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)

    def get_pos(self):
        return pygame.Vector2(self.x, self.y)


class Game:
    def __init__(self):
        self.score = 0
        self.is_running = True
        self.clock = pygame.time.Clock()

        x, y = self._random_pos()
        self.player = Snake(x, y)

        x, y = self._random_pos()
        self.food = Food(x, y)

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake")

    def _random_pos(self):
        return (
            random.randint(0, WIDTH),
            random.randint(0, HEIGHT),
        )

    def _reset(self):
        x, y = self._random_pos()
        self.food = Food(x, y)

    def _render(self):
        self.screen.fill(BLACK)

        # render food
        self.food.render(self.screen)

        # render snake
        self.player.render(self.screen)

        pygame.display.flip()

    def begin(self):
        while self.is_running:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                if not self.player.is_alive:
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

            x, y = self.food.get_pos()
            if self.player.head.collidepoint(x, y):
                self.player.eat()
                self._reset()

            self._render()


if __name__ == "__main__":
    game = Game()
    game.begin()

    pygame.quit()
    exit()
