import pygame
import random
from typing import Literal, Optional
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
from utils.moves import (
    Move,
    Ruleset,
    canonical_name,
    generate_hand_drops,
    generate_legal_moves,
    validate_move,
)

HIGHLIGHT_SELECTED = (255, 215, 0)
HIGHLIGHT_MOVE = (60, 179, 113)
HIGHLIGHT_STACK = (65, 105, 225)
HIGHLIGHT_CAPTURE = (220, 20, 60)
HIGHLIGHT_DROP = (186, 85, 211)


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
        self.is_in_check = False

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
        self._selection_origin: Optional[tuple[int, int]] = None
        self._legal_moves: list[Move] = []
        self._drop_selection: Optional[Piece] = None
        self._drop_moves: list[Move] = []
        self._pending_turncoat_move: Optional[Move] = None
        self._pending_turncoat_options: list[str] = []
        self._waiting_turncoat_choice = False

        if self.render_ui:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(title)
            self._board_surface = pygame.Surface((WIDTH, HEIGHT)).convert()
            self._board_surface.fill(BOARD_COLOR)
            self._draw_borders()
            self._font_small = pygame.font.SysFont("Arial", 18)

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
        self.setup_moves_made = 0
        self._selected_piece = None
        self._waiting_for_click = False
        self._selection_origin = None
        self._legal_moves = []
        self._drop_selection = None
        self._drop_moves = []
        self._cancel_turncoat_prompt()

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
            self.screen.blit(self._board_surface, (0, 0))
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    stack = self.board[c][r]
                    if stack:
                        top_piece = stack[-1]
                        x, y = self.grid_coords[c][r]
                        top_piece.set_position(x, y)
                        top_piece.render(self.screen)
                        if len(stack) > 1:
                            self._render_stack_badge(c, r, len(stack), top_piece)

            self._render_selection_overlay()
            pygame.display.flip()

    def _draw_borders(self):
        target = (
            getattr(self, "_board_surface", None)
            if hasattr(self, "_board_surface")
            else self.screen
        )
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                rect = self.grid_rects[col][row]
                pygame.draw.rect(target, BORDER_COLOR, rect, 1)

    def _render_stack_badge(self, col: int, row: int, count: int, top_piece: Piece):
        try:
            rect = self.grid_rects[col][row]
            badge_r = max(8, rect.width // 12)
            badge_x = rect.right - badge_r - 4
            badge_y = rect.top + badge_r + 4
            opposite_color = WHITE if top_piece.color == BLACK else BLACK
            font_small = getattr(self, "_font_small", pygame.font.SysFont("Arial", 18))
            num_surf = font_small.render(str(count), True, opposite_color)
            num_rect = num_surf.get_rect(center=(badge_x, badge_y))
            self.screen.blit(num_surf, num_rect)
        except Exception:
            pass

    def _render_selection_overlay(self):
        if not self.render_ui:
            return
        surface = self.screen
        if self._selection_origin:
            col, row = self._selection_origin
            rect = self.grid_rects[col][row]
            pygame.draw.rect(surface, HIGHLIGHT_SELECTED, rect, 4)
        for move in self._legal_moves:
            col, row = move.to
            rect = self.grid_rects[col][row]
            color = self._move_highlight_color(move)
            pygame.draw.rect(surface, color, rect, 4)
        if self._drop_selection:
            for move in self._drop_moves:
                col, row = move.to
                rect = self.grid_rects[col][row]
                pygame.draw.rect(surface, HIGHLIGHT_DROP, rect, 4)

    def _move_highlight_color(self, move: Move):
        if move.action == "capture":
            return HIGHLIGHT_CAPTURE
        if move.action in {"stack", "turncoat"}:
            return HIGHLIGHT_STACK
        return HIGHLIGHT_MOVE

    def _format_piece_name(self, name: str) -> str:
        return name.replace("_", " ").title()

    def _build_ruleset(self, player: Optional[Player] = None) -> Ruleset:
        player = player or self.turn
        return Ruleset(board_size=BOARD_SIZE, hand=player.hand_pieces)

    def _opponent_player(self, player: Optional[Player] = None) -> Player:
        player = player or self.turn
        return self.player_2 if player is self.player_1 else self.player_1

    def _clear_board_selection(self):
        if self.game_phase == "initial_setup":
            return
        self._selected_piece = None
        self._selection_origin = None
        self._legal_moves = []

    def _clear_drop_selection(self):
        self._drop_selection = None
        self._drop_moves = []

    def _clear_active_selection(self):
        self._clear_board_selection()
        self._clear_drop_selection()
        self._cancel_turncoat_prompt()

    def _cancel_turncoat_prompt(self):
        self._pending_turncoat_move = None
        self._pending_turncoat_options = []
        self._waiting_turncoat_choice = False

    def _select_board_piece(self, col: int, row: int):
        stack = self.board[col][row]
        if not stack:
            self._clear_board_selection()
            return
        top = stack[-1]
        if top.color != self.turn.color:
            self._clear_board_selection()
            return
        self._selected_piece = top
        self._selection_origin = (col, row)
        rules = self._build_ruleset()
        self._legal_moves = generate_legal_moves(top, (col, row), self.board, rules)
        if not self._legal_moves:
            print("No legal moves for", top.name)

    def _matching_move(
        self, target: tuple[int, int], moves: list[Move]
    ) -> Optional[Move]:
        for move in moves:
            if move.to == target:
                return move
        return None

    def _handle_drop_selection(self, piece_name: str):
        if self._waiting_turncoat_choice:
            print("Resolve the pending turncoat before selecting a drop piece.")
            return
        self._clear_board_selection()
        piece = next(
            (p for p in self.turn.hand_pieces if p.name == piece_name),
            None,
        )
        if not piece:
            print(f"No {piece_name} available in hand.")
            return
        moves = generate_hand_drops(
            piece_name, self.turn.color, self.board, self._build_ruleset()
        )
        if not moves:
            print(f"No legal drop positions for {piece_name}.")
            return
        self._drop_selection = piece
        self._drop_moves = moves
        print(f"Selected {piece_name} for drop. Click a highlighted stack to place it.")

    def _handle_turncoat_choice_key(self, key: int) -> bool:
        if not self._waiting_turncoat_choice:
            return False
        if key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._resolve_turncoat_cancel()
            return True
        index = None
        if pygame.K_1 <= key <= pygame.K_9:
            index = key - pygame.K_1
        elif pygame.K_KP1 <= key <= pygame.K_KP9:
            index = key - pygame.K_KP1
        if index is None:
            return True
        self._resolve_turncoat_choice(index)
        return True

    def _resolve_turncoat_choice(self, option_index: int):
        if not self._waiting_turncoat_choice:
            return
        move = self._pending_turncoat_move
        if not move:
            self._cancel_turncoat_prompt()
            return
        if option_index < 0 or option_index >= len(self._pending_turncoat_options):
            print("Invalid turncoat selection.")
            return
        replacement = self._pending_turncoat_options[option_index]
        if not self._execute_turncoat(move, replacement):
            print("Turncoat swap failed; capturing instead.")
            self._apply_capture(move)
        self._cancel_turncoat_prompt()
        self._clear_board_selection()
        self._switch_turn()

    def _resolve_turncoat_cancel(self):
        if not self._waiting_turncoat_choice:
            return
        move = self._pending_turncoat_move
        if move:
            print("Turncoat cancelled. Capturing the enemy stack instead.")
            self._apply_capture(move)
        self._cancel_turncoat_prompt()
        self._clear_board_selection()
        self._switch_turn()

    def _handle_game_click(self, mouse_pos: tuple[int, int]):
        col = mouse_pos[0] // (WIDTH // BOARD_SIZE)
        row = mouse_pos[1] // (HEIGHT // BOARD_SIZE)
        if not (0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE):
            return
        target = (col, row)

        if self._waiting_turncoat_choice:
            print("Select a turncoat replacement (press number key) before continuing.")
            return

        if self._drop_selection:
            move = self._matching_move(target, self._drop_moves)
            if move:
                if self._apply_drop(move):
                    self._clear_drop_selection()
                    self._switch_turn()
                else:
                    print("Drop failed. Try again.")
            else:
                print("Invalid drop location.")
            return

        if self._selection_origin:
            if target == self._selection_origin:
                self._clear_board_selection()
                return
            move = self._matching_move(target, self._legal_moves)
            if move and validate_move(
                self._selected_piece,
                move.frm,
                move.to,
                self.board,
                self._build_ruleset(),
            ):
                completed = self._apply_board_move(move)
                if completed:
                    self._clear_board_selection()
                    self._switch_turn()
                return

            stack = self.board[col][row]
            if stack and stack[-1].color == self.turn.color:
                self._select_board_piece(col, row)
            else:
                self._clear_board_selection()
            return

        stack = self.board[col][row]
        if stack and stack[-1].color == self.turn.color:
            self._select_board_piece(col, row)
        else:
            self._clear_board_selection()

    def _apply_board_move(self, move: Move) -> bool:
        if move.frm is None:
            return False
        if move.action in {"move", "stack"}:
            self._move_piece(move.frm, move.to)
            return True
        if move.action == "capture":
            self._apply_capture(move)
            return True
        if move.action == "turncoat":
            return self._handle_turncoat(move)
        print(f"Unhandled move action: {move.action}")
        return False

    def _move_piece(self, frm: tuple[int, int], to: tuple[int, int]):
        src_stack = self.board[frm[0]][frm[1]]
        if not src_stack:
            return
        piece = src_stack.pop()
        self.board[to[0]][to[1]].append(piece)

    def _apply_capture(self, move: Move):
        dst_stack = self.board[move.to[0]][move.to[1]]
        captured = list(dst_stack)
        dst_stack.clear()
        self._collect_captured_pieces(captured)
        self._move_piece(move.frm, move.to)

    def _handle_turncoat(self, move: Move) -> bool:
        replacements = (move.aux or {}).get("replacements", [])
        if not replacements:
            self._apply_capture(move)
            return True
        if len(replacements) == 1:
            if self._execute_turncoat(move, replacements[0]):
                self._cancel_turncoat_prompt()
                return True
            print("Turncoat swap failed; capturing instead.")
            self._apply_capture(move)
            self._cancel_turncoat_prompt()
            return True
        self._pending_turncoat_move = move
        self._pending_turncoat_options = replacements
        self._waiting_turncoat_choice = True
        friendly_name = self._selected_piece.name if self._selected_piece else "Captain"
        print(
            f"{friendly_name} can turncoat. Choose replacement: "
            + ", ".join(
                f"[{idx + 1}] {self._format_piece_name(name)}"
                for idx, name in enumerate(replacements)
            )
        )
        print("Press number key (1-9) to pick, or Esc to capture instead.")
        return False

    def _execute_turncoat(self, move: Move, replacement_name: str) -> bool:
        replacement_piece = self._remove_hand_piece(self.turn, replacement_name)
        if not replacement_piece:
            return False

        dst_stack = self.board[move.to[0]][move.to[1]]
        target_idx = self._find_turncoat_target(dst_stack, replacement_name)
        if target_idx is None:
            self.turn.hand_pieces.append(replacement_piece)
            return False

        enemy_piece = dst_stack.pop(target_idx)
        enemy_piece.color = self.turn.color
        self.turn.hand_pieces.append(enemy_piece)

        opponent = self._opponent_player()
        replacement_piece.color = opponent.color
        dst_stack.insert(target_idx, replacement_piece)

        self._move_piece(move.frm, move.to)
        return True

    def _find_turncoat_target(
        self, stack: list[Piece], canonical_piece: str
    ) -> Optional[int]:
        if not stack:
            return None
        for offset in range(1, min(2, len(stack)) + 1):
            idx = len(stack) - offset
            if canonical_name(stack[idx].name) == canonical_piece:
                return idx
        return None

    def _remove_hand_piece(
        self, player: Player, canonical_piece: str
    ) -> Optional[Piece]:
        for idx, piece in enumerate(player.hand_pieces):
            if canonical_name(piece.name) == canonical_piece:
                return player.hand_pieces.pop(idx)
        return None

    def _collect_captured_pieces(self, captured_stack: list[Piece]):
        if not captured_stack:
            return
        for piece in captured_stack:
            piece.color = self.turn.color
            self.turn.hand_pieces.append(piece)

    def _apply_drop(self, move: Move) -> bool:
        if not self._drop_selection:
            return False
        target_stack = self.board[move.to[0]][move.to[1]]
        if self._drop_selection not in self.turn.hand_pieces:
            print("Selected hand piece no longer available.")
            return False
        self.turn.hand_pieces.remove(self._drop_selection)
        target_stack.append(self._drop_selection)
        return True

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

        if piece.name == "MARSHAL" and len(stack) > 0:
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
                        elif event.key == pygame.K_RETURN:
                            self.turn.setup_done = True
                            self._selected_piece = None
                            self._waiting_for_click = False
                            print(
                                f"{'Black' if self.turn.color == BLACK else 'White'} done with setup."
                            )
                            print(
                                f"{'Black' if self.turn.color == BLACK else 'White'} pieces in hand: {len(self.turn.hand_pieces)}"
                            )
                            self._switch_turn()
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
                else:
                    if event.type == pygame.KEYDOWN:
                        if self._handle_turncoat_choice_key(event.key):
                            continue
                        if event.key in PIECE_KEYBOARD_MAP:
                            piece_name = PIECE_KEYBOARD_MAP[event.key]
                            self._handle_drop_selection(piece_name)
                        elif event.key == pygame.K_ESCAPE:
                            self._clear_active_selection()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_game_click(event.pos)
            self._render()
            self.clock.tick(FPS)
        self.end()

    # =============================================================================
    # Methods for AlphaZero integration
    # =============================================================================

    def copy(self) -> "Game":
        """
        Create a deep copy of the game state for MCTS simulation.

        Returns:
            Game: A new Game instance with copied state
        """
        import copy as copy_module

        # Create new game without UI
        new_game = Game(render_ui=False)

        # Copy board state (deep copy each stack)
        new_game.board = [[[] for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                for piece in self.board[col][row]:
                    new_piece = Piece(piece.name, piece.color)
                    new_game.board[col][row].append(new_piece)

        # Copy players
        new_game.player_1 = Player(BLACK)
        new_game.player_1.hand_pieces = []
        for piece in self.player_1.hand_pieces:
            new_game.player_1.hand_pieces.append(Piece(piece.name, piece.color))
        new_game.player_1.setup_done = self.player_1.setup_done
        new_game.player_1.is_in_check = self.player_1.is_in_check

        new_game.player_2 = Player(WHITE)
        new_game.player_2.hand_pieces = []
        for piece in self.player_2.hand_pieces:
            new_game.player_2.hand_pieces.append(Piece(piece.name, piece.color))
        new_game.player_2.setup_done = self.player_2.setup_done
        new_game.player_2.is_in_check = self.player_2.is_in_check

        # Copy turn
        new_game.turn = (
            new_game.player_1 if self.turn == self.player_1 else new_game.player_2
        )

        # Copy game phase and other state
        new_game.game_phase = self.game_phase
        new_game.setup_moves_made = self.setup_moves_made
        new_game._terminal = getattr(self, "_terminal", False)
        new_game._result = getattr(self, "_result", 0.0)
        new_game._move_count = getattr(self, "_move_count", 0)

        return new_game

    def is_terminal(self) -> bool:
        """
        Check if the game has ended.

        Returns:
            bool: True if game is over
        """
        if hasattr(self, "_terminal") and self._terminal:
            return True

        # Check if Marshal is captured (game over)
        p1_has_marshal = any(
            piece.name == "MARSHAL" for piece in self.player_1.hand_pieces
        ) or any(
            any(piece.name == "MARSHAL" and piece.color == BLACK for piece in stack)
            for row in self.board
            for stack in row
        )

        p2_has_marshal = any(
            piece.name == "MARSHAL" for piece in self.player_2.hand_pieces
        ) or any(
            any(piece.name == "MARSHAL" and piece.color == WHITE for piece in stack)
            for row in self.board
            for stack in row
        )

        # Check if either player's Marshal is on the board
        p1_marshal_on_board = any(
            any(piece.name == "MARSHAL" and piece.color == BLACK for piece in stack)
            for row in self.board
            for stack in row
        )
        p2_marshal_on_board = any(
            any(piece.name == "MARSHAL" and piece.color == WHITE for piece in stack)
            for row in self.board
            for stack in row
        )

        # Game ends if a Marshal is captured (not on board and not in original hand)
        if self.game_phase == "game":
            if not p1_marshal_on_board:
                self._terminal = True
                self._result = -1.0  # White wins
                return True
            if not p2_marshal_on_board:
                self._terminal = True
                self._result = 1.0  # Black wins
                return True

        return False

    def get_result(self) -> float:
        """
        Get the game result from Black's perspective.

        Returns:
            float: +1.0 for Black win, -1.0 for White win, 0.0 for draw/ongoing
        """
        if hasattr(self, "_result"):
            return self._result

        # Check terminal state
        self.is_terminal()

        return getattr(self, "_result", 0.0)

    def apply_action(self, action_idx: int) -> bool:
        """
        Apply an action by its index (for MCTS).

        Args:
            action_idx: Action index from encoding

        Returns:
            bool: True if action was applied successfully
        """
        from encoding import decode_action, action_to_move
        from utils.moves import generate_legal_moves, Ruleset

        decoded = decode_action(action_idx)

        if decoded["type"] == "setup":
            # Setup phase action
            piece_name = decoded["piece"]
            dst = decoded["dst"]

            piece = next(
                (p for p in self.turn.hand_pieces if p.name == piece_name), None
            )
            if piece is None:
                return False

            row, col = dst[1], dst[0]
            return self.handle_setup_move(piece, row, col)

        elif decoded["type"] == "drop":
            # Drop action
            piece_name = decoded["piece"]
            dst = decoded["dst"]

            piece = next(
                (p for p in self.turn.hand_pieces if p.name == piece_name), None
            )
            if piece is None:
                return False

            # Check if drop is legal
            rules = Ruleset(board_size=BOARD_SIZE, hand=self.turn.hand_pieces)
            from utils.moves import generate_hand_drops

            drop_moves = generate_hand_drops(
                piece_name, self.turn.color, self.board, rules
            )

            if not any(m.to == dst for m in drop_moves):
                return False

            # Apply drop
            self.turn.hand_pieces.remove(piece)
            self.board[dst[0]][dst[1]].append(piece)
            self._switch_turn()
            self._increment_move_count()
            return True

        else:  # move
            src = decoded["src"]
            dst = decoded["dst"]

            stack = self.board[src[0]][src[1]]
            if not stack:
                return False

            piece = stack[-1]
            if piece.color != self.turn.color:
                return False

            # Find legal move
            rules = Ruleset(board_size=BOARD_SIZE, hand=self.turn.hand_pieces)
            legal_moves = generate_legal_moves(piece, src, self.board, rules)

            move = next((m for m in legal_moves if m.to == dst), None)
            if move is None:
                return False

            # Apply the move
            if move.action in {"move", "stack"}:
                self._move_piece(src, dst)
            elif move.action == "capture":
                self._apply_capture(move)
            elif move.action == "turncoat":
                # For turncoat, just capture (simplified for MCTS)
                self._apply_capture(move)

            self._switch_turn()
            self._increment_move_count()
            self.is_terminal()  # Update terminal state
            return True

    def _increment_move_count(self):
        """Increment the move counter."""
        if not hasattr(self, "_move_count"):
            self._move_count = 0
        self._move_count += 1

    @property
    def move_count(self) -> int:
        """Get current move count."""
        return getattr(self, "_move_count", 0)


if __name__ == "__main__":
    game = Game()
    game.start()
