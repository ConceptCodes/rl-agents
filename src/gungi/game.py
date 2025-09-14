import pygame
import random
from typing import Literal
from constants import (
    PIECE_SIZE,
    FPS,
    WIDTH,
    HEIGHT,
    BOARD_COLOR,
    PIECE_MAP_UNICODE,
    PIECE_KEYBOARD_MAP,
    BOARD_SIZE,
    BORDER_COLOR,
    BLACK,
    WHITE,
)


pygame.init()
pygame.freetype.init()

try:
    for font_name in [
        "Hiragino Sans GB",
        "Apple Gothic",
        "Osaka",
        "MS Gothic",
        "Arial Unicode MS",
    ]:
        try:
            font = pygame.font.SysFont(font_name, 48)
            test_surface = font.render("帥", True, (0, 0, 0))
            if test_surface.get_width() > 5:
                print(f"Successfully loaded system font: {font_name}")
                break
        except:
            continue
    else:
        font_path = "/src/gungi/ume-pgo5.ttf"
        font = pygame.font.Font(font_path, 36)
        print(f"Using TTF font: {font_path}")
except Exception as e:
    print(f"Error loading any fonts: {e}")
    font = pygame.font.SysFont("Arial", 36)


class Piece:
    def __init__(self, symbol, color):
        self.symbol = PIECE_MAP_UNICODE[symbol]
        self.name = symbol
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
        self.hand_pieces = self.pieces.copy()
        self.setup_done = False

    def _init_pieces(self) -> list[Piece]:
        pieces = []
        pieces.append(Piece("MARSHAL", self.color))
        pieces.append(Piece("GENERAL", self.color))
        pieces.append(Piece("LIEUTENANT_GENERAL", self.color))
        pieces.extend([Piece("MAJOR_GENERAL", self.color)] * 2)
        pieces.extend([Piece("WARRIOR", self.color)] * 2)
        pieces.extend([Piece("LANCER", self.color)] * 3)
        pieces.extend([Piece("RIDER", self.color)] * 2)
        pieces.extend([Piece("SPY", self.color)] * 2)
        pieces.extend([Piece("FORTRESS", self.color)] * 2)
        pieces.extend([Piece("SOLDIER", self.color)] * 4)
        pieces.append(Piece("CANNON", self.color))
        pieces.extend([Piece("ARCHER", self.color)] * 2)
        pieces.append(Piece("MUSKETEER", self.color))
        pieces.append(Piece("TACTICIAN", self.color))
        return pieces


class Game:
    def __init__(self, title="Gungi", render_ui=True):
        self.is_running = True
        self.clock = pygame.time.Clock()
        self.render_ui = render_ui
        self.board = self._init_board()
        self.grid_rects, self.grid_coords = self._precompute_grid()

        self.player_1 = Player(BLACK)
        self.player_2 = Player(WHITE)
        self.turn = random.choice([self.player_1, self.player_2])
        self.game_phase: Literal["initial_setup", "game"] = "initial_setup"
        self.setup_moves_made = 0
        self._selected_piece = None
        self._waiting_for_click = False

        if self.render_ui:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(title)

    def _init_board(self):
        return [[[] for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    def _precompute_grid(self):
        blockSize = WIDTH // BOARD_SIZE
        rects = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        coords = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        cols = 0
        rows = 0
        for x in range(0, WIDTH, blockSize):
            for y in range(0, HEIGHT, blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                rects[cols][rows] = rect
                coords[cols][rows] = (int(x + blockSize // 2), int(y + blockSize // 2))
                rows += 1
                if rows == BOARD_SIZE:
                    rows = 0
                    cols += 1
        return rects, coords

    def reset(self):
        self.board = self._init_board()
        self.player_1 = Player(BLACK)
        self.player_2 = Player(WHITE)
        self.turn = random.choice([self.player_1, self.player_2])
        self.game_phase = "initial_setup"

    def end(self):
        pygame.quit()
        exit()

    def _switch_turn(self):
        if self.game_phase == "initial_setup":
            if self.player_1.setup_done and self.player_2.setup_done:
                self.game_phase = "game"
                print("Both players done with setup. Game phase now 'game'.")
                return

            other = self.player_2 if self.turn == self.player_1 else self.player_1
            if not other.setup_done:
                self.turn = other
                return

            if not self.player_1.setup_done:
                self.turn = self.player_1
                return
            if not self.player_2.setup_done:
                self.turn = self.player_2
                return

            self.game_phase = "game"
            print("Both players done with setup. Game phase now 'game'.")
            return

        self.turn = self.player_2 if self.turn == self.player_1 else self.player_1

    def _render(self):
        if self.render_ui:
            self.screen.fill(BOARD_COLOR)
            self._draw_borders()
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    stack = self.board[c][r]
                    if stack:
                        top_piece = stack[-1]
                        x, y = self.grid_coords[c][r]
                        top_piece.set_position(x, y)
                        top_piece.render(self.screen)

            self._render_turn_indicator()
            pygame.display.flip()

    def _render_turn_indicator(self):
        try:
            label = "Black" if self.turn.color == BLACK else "White"
            text = f"Turn: {label}"
            color = WHITE if self.turn.color == BLACK else BLACK
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, (8, 8))
        except Exception:
            pass

    def _draw_borders(self):
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                rect = self.grid_rects[col][row]
                pygame.draw.rect(self.screen, BORDER_COLOR, rect, 1)

    def handle_setup_move(self, piece, row, col):
        if self.game_phase != "initial_setup":
            return False

        if piece not in self.turn.hand_pieces:
            return False

        if self.turn.color == BLACK and not (0 <= row <= 2):
            return False
        if self.turn.color == WHITE and not (6 <= row <= 8):
            return False

        stack = self.board[col][row]
        if len(stack) >= 3:
            return False
        if stack and stack[-1].name == "MARSHAL":
            return False

        self.board[col][row].append(piece)
        self.turn.hand_pieces.remove(piece)
        self.setup_moves_made += 1
        self._switch_turn()
        return True

    def start(self):
        print("Piece Keyboard Map:")
        for key, piece in PIECE_KEYBOARD_MAP.items():
            print(f"{pygame.key.name(key)}: {piece}")

        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False

                if self.game_phase == "initial_setup":
                    if event.type == pygame.KEYDOWN:
                        if event.key in PIECE_KEYBOARD_MAP:
                            piece_name = PIECE_KEYBOARD_MAP[event.key]
                            piece = next(
                                (
                                    p
                                    for p in self.turn.hand_pieces
                                    if p.name == piece_name
                                ),
                                None,
                            )
                            if piece:
                                self._selected_piece = piece
                                self._waiting_for_click = True
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self._waiting_for_click and self._selected_piece:
                            mouse_x, mouse_y = event.pos
                            col = mouse_x // (WIDTH // BOARD_SIZE)
                            row = mouse_y // (HEIGHT // BOARD_SIZE)
                            if self.handle_setup_move(self._selected_piece, row, col):
                                print(
                                    f"Placed {self._selected_piece.name} at ({row}, {col})"
                                )
                                self._selected_piece = None
                                self._waiting_for_click = False
                            else:
                                print("Invalid move, try again.")
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        self.turn.setup_done = True
                        print(
                            f"{'Black' if self.turn.color == BLACK else 'White'} done with setup."
                        )
                        self._switch_turn()

            self._render()
            self.clock.tick(FPS)
        self.end()


if __name__ == "__main__":
    game = Game()
    game.start()
