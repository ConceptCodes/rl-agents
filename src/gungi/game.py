import os
import pygame
import numpy as np
from constants import PIECE_SIZE, FPS, WIDTH, HEIGHT, BOARD_COLOR, PIECE_MAP, BOARD_SIZE

pygame.init()


class Piece:
    def __init__(self, symbol, piece_type, x, y):
        self.symbol = PIECE_MAP[symbol]
        self.piece_type = piece_type
        self.rect = pygame.rect.Rect(x, y, PIECE_SIZE, PIECE_SIZE)

    def move(self, new_position):
        self.position = new_position

    def render(self, screen):
        pygame.draw.rect(screen, None, self.rect)


class Game:
    def __init__(self, title="Gungi", render_ui=True):
        self.is_running = True
        self.clock = pygame.time.Clock()
        self.render_ui = render_ui
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        if self.render_ui:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(title)

    def reset(self):
        pass

    def end(self):
        pygame.quit()
        exit()

    def start(self):
        while self.is_running:
            self.clock.tick(FPS)
            self._render()

        self.end()

    def _render(self):
        if self.render_ui:
            self.screen.fill(BOARD_COLOR)
            pygame.display.flip()


if __name__ == "__main__":
    game = Game()
    game.start()
