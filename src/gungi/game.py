import os
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
pygame.font.init()


font = pygame.font.SysFont("AppleGothic", 22)


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
        if self.x is None or self.y is None:
            return

        pygame.draw.circle(screen, self.color, (self.x, self.y), PIECE_SIZE)

        text_surf = font.render(
            self.symbol, True, BLACK if self.color == WHITE else WHITE
        )

        text_rect = text_surf.get_rect(center=(self.x, self.y))
        screen.blit(text_surf, text_rect)

    def __mul__(self, other):
        if isinstance(other, int):
            return [self] * other
        return NotImplemented

    def __repr__(self):
        return f"<Piece(symbol={self.symbol}, color={self.color}, position=({self.x}, {self.y}))>"


class Player:
    def __init__(self, color):
        self.color = color
        self.pieces = self._init_pieces()

    def _init_pieces(self) -> list[Piece]:
        return [
            Piece("MARSHAL", self.color),
            Piece("GENERAL", self.color),
            Piece("LIEUTENANT_GENERAL", self.color),
            Piece("MAJOR_GENERAL", self.color),
            Piece("MAJOR_GENERAL", self.color),
            Piece("WARRIOR", self.color),
            Piece("WARRIOR", self.color),
            Piece("LANCER", self.color),
            Piece("LANCER", self.color),
            Piece("LANCER", self.color),
            Piece("RIDER", self.color),
            Piece("RIDER", self.color),
            Piece("SPY", self.color),
            Piece("SPY", self.color),
            Piece("FORTRESS", self.color),
            Piece("FORTRESS", self.color),
            Piece("SOLDIER", self.color),
            Piece("SOLDIER", self.color),
            Piece("SOLDIER", self.color),
            Piece("SOLDIER", self.color),
            Piece("CANNON", self.color),
            Piece("ARCHER", self.color),
            Piece("ARCHER", self.color),
            Piece("MUSKETEER", self.color),
            Piece("TACTICIAN", self.color),
        ]


class Game:
    def __init__(self, title="Gungi", render_ui=True):
        self.is_running = True
        self.clock = pygame.time.Clock()
        self.render_ui = render_ui
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=tuple)

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

    def _init_player_positions(self, player: Player):
        # this should use a random layout
        # current setup doesn't take piece stacking in effect
        for piece in player.pieces:
            for col in range(BOARD_SIZE):
                for row in range(3):
                    x, y = self.grid[col][row + (0 if player.color == WHITE else 6)]
                    piece.set_position(x, y)
                    piece.render(self.screen)

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
                self.grid[cols][rows] = (
                    int(x + blockSize // 2),
                    int(y + blockSize // 2),
                )
                rows += 1
                if rows == BOARD_SIZE:
                    rows = 0
                    cols += 1

    def _render(self):
        if self.render_ui:
            self.screen.fill(BOARD_COLOR)
            self._draw_borders()

            self._init_player_positions(self.player_1)
            self._init_player_positions(self.player_2)

            pygame.display.flip()


if __name__ == "__main__":
    game = Game()
    game.start()
