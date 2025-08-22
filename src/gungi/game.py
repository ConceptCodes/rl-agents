import pygame
import numpy as np
from constants import (
    PIECE_SIZE,
    FPS,
    WIDTH,
    HEIGHT,
    BOARD_COLOR,
    PIECE_MAP,
    BOARD_SIZE,
    BORDER_COLOR,
    BLACK,
    WHITE,
)

pygame.init()
font = pygame.font.Font("freesansbold.ttf", 32)


class Piece:
    def __init__(self, symbol, color):
        self.symbol = PIECE_MAP[symbol]
        self.color = color
        self.x = None
        self.y = None

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def move(self, new_position, board):
        if not self._validate_move(board):
            raise Exception("Invalid Move")
        self.position = new_position

    def _validate_move(self, board) -> bool:
        return True

    def render(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), PIECE_SIZE)
        text = font.render(self.symbol, True, BLACK)
        screen.blit(text, (self.x, self.y))


class Player:
    def __init__(self, color):
        self.color = color
        self.pieces = self._init_pieces()

    def _init_pieces(self):
        return [
            Piece("MARSHAL", self.color),
            Piece("GENERAL", self.color),
            Piece("LIEUTENANT_GENERAL", self.color),
            Piece("MAJOR_GENERAL", self.color) * 2,
            Piece("WARRIOR", self.color) * 2,
            Piece("LANCER", self.color) * 3,
            Piece("RIDER", self.color) * 2,
            Piece("SPY", self.color) * 2,
            Piece("FORTRESS", self.color) * 2,
            Piece("SOLDIER", self.color) * 4,
            Piece("CANNON", self.color),
            Piece("ARCHER", self.color) * 2,
            Piece("MUSKETEER", self.color),
            Piece("TACTICIAN", self.color),
        ]


class Game:
    def __init__(self, title="Gungi", render_ui=True):
        self.is_running = True
        self.clock = pygame.time.Clock()
        self.render_ui = render_ui
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        self.player_1 = Player(BLACK)
        self.player_2 = Player(WHITE)

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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
            self._render()
        self.end()

    def _draw_borders(self):
        blockSize = WIDTH // BOARD_SIZE
        rows = 0
        cols = 0
        for x in range(0, WIDTH, blockSize):
            for y in range(0, HEIGHT, blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                pygame.draw.rect(self.screen, BORDER_COLOR, rect, 1)
                self.grid[cols][rows] = (x + blockSize // 2, y + blockSize // 2)
                rows += 1
                if rows == BOARD_SIZE:
                    rows = 0
                    cols += 1

    def _render(self):
        if self.render_ui:
            self.screen.fill(BOARD_COLOR)
            self._draw_borders()

            pygame.display.flip()


if __name__ == "__main__":
    game = Game()
    game.start()
