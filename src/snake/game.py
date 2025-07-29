import os
import pygame
import random
import time
import numpy as np

from constants import GREEN, BLACK, WHITE, SIZE, WIDTH, HEIGHT, FPS

pygame.init()


class Snake:
    def __init__(self, x, y):
        self.is_alive = True
        self.body = [pygame.Rect(x, y, SIZE, SIZE)]
        self.vel = SIZE
        self.direction = "right"
        self.next_direction = "right"

    def eat(self):
        tail = self.body[-1]
        self.body.append(pygame.Rect(tail.x, tail.y, SIZE, SIZE))

    def move(self):
        if (
            (self.next_direction == "up" and self.direction != "down")
            or (self.next_direction == "down" and self.direction != "up")
            or (self.next_direction == "left" and self.direction != "right")
            or (self.next_direction == "right" and self.direction != "left")
        ):
            self.direction = self.next_direction

        # Calculate new head position
        head = self.body[0].copy()
        if self.direction == "up":
            head.y -= self.vel
        elif self.direction == "down":
            head.y += self.vel
        elif self.direction == "right":
            head.x += self.vel
        elif self.direction == "left":
            head.x -= self.vel

        # Insert new head and remove tail
        self.body.insert(0, head)
        self.body.pop()

        # Check for boundary collisions
        if head.x < 0 or head.x + SIZE > WIDTH or head.y < 0 or head.y + SIZE > HEIGHT:
            self.is_alive = False

        # Check for self collision
        for segment in self.body[1:]:
            if head.colliderect(segment):
                self.is_alive = False
                break

    @property
    def head(self):
        return self.body[0]

    def clear(self):
        self.body = [pygame.Rect(self.head.x, self.head.y, SIZE, SIZE)]
        self.is_alive = True
        self.direction = "right"
        self.next_direction = "right"

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
    def __init__(self, title="Snake", render_ui=True, record=False):
        self.score = 0
        self.is_running = True
        self.clock = pygame.time.Clock()
        self.render_ui = render_ui
        self.record = record

        x, y = self._random_pos()
        self.player = Snake(x, y)

        x, y = self._random_pos()
        self.food = Food(x, y)

        if self.render_ui:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(title)

    def _random_pos(self):
        return (
            random.randrange(0, WIDTH, SIZE),
            random.randrange(0, HEIGHT, SIZE),
        )

    def _reset_food(self):
        x, y = self._random_pos()
        self.food = Food(x, y)

    def _reset_player(self):
        x, y = self._random_pos()
        while x == self.food.x and y == self.food.y:
            x, y = self._random_pos()
        self.player = Snake(x, y)

    def _reset(self):
        self._reset_food()
        if not self.player.is_alive:
            self._reset_player()

    def _render(self):
        if self.render_ui:
            self.screen.fill(BLACK)
            self.food.render(self.screen)
            self.player.render(self.screen)
            pygame.display.flip()

    def _record(self):
        if self.record:
            return pygame.surfarray.array2d(self.screen)

    def _handle_input(self, val):
        self.player.next_direction = val

    def _collision_check(self):
        x, y = self.food.get_pos()
        return self.player.head.collidepoint(x, y)

    def end(self):
        pygame.quit()
        exit()

    def start(self):
        while self.is_running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                if not self.player.is_alive:
                    self._reset()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self._handle_input("left")
                    if event.key == pygame.K_RIGHT:
                        self._handle_input("right")
                    if event.key == pygame.K_UP:
                        self._handle_input("up")
                    if event.key == pygame.K_DOWN:
                        self._handle_input("down")

            self.player.move()

            if self._collision_check():
                self.player.eat()
                self._reset()
            self._render()
            self._record()

        self.end()


if __name__ == "__main__":
    game = Game()
    game.start()
