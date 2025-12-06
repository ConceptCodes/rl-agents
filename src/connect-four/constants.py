import pygame

pygame.init()

FONT = pygame.font.Font("freesansbold.ttf", 20)


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)


ROWS = 6
COLS = 7
CELL_SIZE = 100
WIDTH = COLS * CELL_SIZE
HEIGHT = (ROWS + 1) * CELL_SIZE  # Extra row for dropping animation
FPS = 30


EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2
WIN_LENGTH = 4
