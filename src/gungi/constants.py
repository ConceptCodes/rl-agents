import pygame

BOARD_COLOR = (210, 180, 140)
BORDER_COLOR = (165, 42, 42)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

FPS = 60

PIECE_SIZE = 35
BOARD_SIZE = 9
CELL_SIZE = 80

WIDTH = BOARD_SIZE * 80
HEIGHT = BOARD_SIZE * CELL_SIZE


PIECE_MAP = {
    "MARSHAL": "帥",
    "GENERAL": "大",
    "LIEUTENANT_GENERAL": "中",
    "MAJOR_GENERAL": "小",
    "WARRIOR": "侍",
    "LANCER": "槍",
    "RIDER": "馬",
    "SPY": "忍",
    "FORTRESS": "砦",
    "SOLDIER": "兵",
    "CANNON": "砲",
    "ARCHER": "弓",
    "MUSKETEER": "筒",
    "TACTICIAN": "謀",
}

PIECE_MAP_UNICODE = {
    "MARSHAL": "\u5e25",
    "GENERAL": "\u5927",
    "LIEUTENANT_GENERAL": "\u4e2d",
    "MAJOR_GENERAL": "\u5c0f",
    "WARRIOR": "\u4f3d",
    "LANCER": "\u69cd",
    "RIDER": "\u99ac",
    "SPY": "\u5fcd",
    "FORTRESS": "\u58a8",
    "SOLDIER": "\u5175",
    "CANNON": "\u7832",
    "ARCHER": "\u5f13",
    "MUSKETEER": "\u7b52",
    "TACTICIAN": "\u8b00",
}

PIECE_KEYBOARD_MAP = {
    pygame.K_0: "MARSHAL",
    pygame.K_1: "GENERAL",
    pygame.K_2: "LIEUTENANT_GENERAL",
    pygame.K_3: "MAJOR_GENERAL",
    pygame.K_4: "WARRIOR",
    pygame.K_5: "LANCER",
    pygame.K_6: "RIDER",
    pygame.K_7: "SPY",
    pygame.K_8: "FORTRESS",
    pygame.K_9: "SOLDIER",
    pygame.K_MINUS: "CANNON",
    pygame.K_EQUALS: "ARCHER",
    pygame.K_LEFTBRACKET: "MUSKETEER",
    pygame.K_RIGHTBRACKET: "TACTICIAN",
}
