import pygame
import numpy as np
from constants import ROWS, COLS, EMPTY, PLAYER1, PLAYER2, WIN_LENGTH
from constants import WIDTH, HEIGHT, CELL_SIZE, FPS
from constants import RED, YELLOW, BLUE, BLACK, WHITE, FONT


class Board:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = PLAYER1
        self.game_over = False
        self.winner: int | None = None

    def drop_piece(self, col):
        """Drop a piece in the specified column.

        Returns:
            tuple: (row, col) if successful, None if column is full
        """
        if col < 0 or col >= COLS:
            return None

        for row in range(ROWS - 1, -1, -1):
            if self.grid[row][col] == EMPTY:
                self.grid[row][col] = self.current_player
                return (row, col)
        return None

    def _check_horizontal(self, row, col, player):
        """Check for horizontal win starting from (row, col)."""
        count = 1
        # Check left
        for c in range(col - 1, max(-1, col - WIN_LENGTH), -1):
            if self.grid[row][c] == player:
                count += 1
            else:
                break
        # Check right
        for c in range(col + 1, min(COLS, col + WIN_LENGTH)):
            if self.grid[row][c] == player:
                count += 1
            else:
                break
        return count >= WIN_LENGTH

    def _check_vertical(self, row, col, player):
        """Check for vertical win starting from (row, col)."""
        count = 1
        # Check up
        for r in range(row - 1, max(-1, row - WIN_LENGTH), -1):
            if self.grid[r][col] == player:
                count += 1
            else:
                break
        # Check down
        for r in range(row + 1, min(ROWS, row + WIN_LENGTH)):
            if self.grid[r][col] == player:
                count += 1
            else:
                break
        return count >= WIN_LENGTH

    def _check_diagonal_tl_br(self, row, col, player):
        """Check for diagonal win (top-left to bottom-right)."""
        count = 1
        # Check up-left
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0 and count < WIN_LENGTH:
            if self.grid[r][c] == player:
                count += 1
                r -= 1
                c -= 1
            else:
                break
        # Check down-right
        r, c = row + 1, col + 1
        while r < ROWS and c < COLS and count < WIN_LENGTH:
            if self.grid[r][c] == player:
                count += 1
                r += 1
                c += 1
            else:
                break
        return count >= WIN_LENGTH

    def _check_diagonal_tr_bl(self, row, col, player):
        """Check for diagonal win (top-right to bottom-left)."""
        count = 1
        # Check up-right
        r, c = row - 1, col + 1
        while r >= 0 and c < COLS and count < WIN_LENGTH:
            if self.grid[r][c] == player:
                count += 1
                r -= 1
                c += 1
            else:
                break
        # Check down-left
        r, c = row + 1, col - 1
        while r < ROWS and c >= 0 and count < WIN_LENGTH:
            if self.grid[r][c] == player:
                count += 1
                r += 1
                c -= 1
            else:
                break
        return count >= WIN_LENGTH

    def check_win(self, row, col):
        """Check if the last move resulted in a win.

        Args:
            row: Row of the last move
            col: Column of the last move

        Returns:
            bool: True if the move resulted in a win
        """
        player = self.grid[row][col]

        return (
            self._check_horizontal(row, col, player)
            or self._check_vertical(row, col, player)
            or self._check_diagonal_tl_br(row, col, player)
            or self._check_diagonal_tr_bl(row, col, player)
        )

    def is_full(self):
        """Check if the board is completely filled."""
        return np.all(self.grid != EMPTY)

    def get_valid_moves(self):
        """Get list of columns that are not full."""
        return [col for col in range(COLS) if self.grid[0][col] == EMPTY]

    def switch_player(self):
        """Switch to the other player."""
        self.current_player = PLAYER2 if self.current_player == PLAYER1 else PLAYER1

    def reset(self):
        """Reset the board to initial state."""
        self.grid = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = PLAYER1
        self.game_over = False
        self.winner = None


class Game:
    def __init__(self, player1_type="human", player2_type="human"):
        pygame.display.set_caption("Connect Four")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.board = Board()
        self.is_running = True
        self.player1_type = player1_type
        self.player2_type = player2_type

    def draw_board(self):
        """Draw the Connect Four board."""
        # Draw background
        self.screen.fill(BLACK)

        # Draw board background
        pygame.draw.rect(self.screen, BLUE, (0, CELL_SIZE, WIDTH, HEIGHT - CELL_SIZE))

        # Draw circles for each cell
        for row in range(ROWS):
            for col in range(COLS):
                x = col * CELL_SIZE + CELL_SIZE // 2
                y = row * CELL_SIZE + CELL_SIZE + CELL_SIZE // 2

                # Draw empty circle
                pygame.draw.circle(self.screen, BLACK, (x, y), CELL_SIZE // 2 - 5)

                # Draw player pieces
                if self.board.grid[row][col] == PLAYER1:
                    pygame.draw.circle(self.screen, RED, (x, y), CELL_SIZE // 2 - 5)
                elif self.board.grid[row][col] == PLAYER2:
                    pygame.draw.circle(self.screen, YELLOW, (x, y), CELL_SIZE // 2 - 5)

        # Draw current player indicator at top
        if self.board.current_player == PLAYER1:
            color = RED
            player_text = "Player 1 (Red)"
        else:
            color = YELLOW
            player_text = "Player 2 (Yellow)"

        if not self.board.game_over:
            pygame.draw.circle(
                self.screen,
                color,
                (pygame.mouse.get_pos()[0], CELL_SIZE // 2),
                CELL_SIZE // 2 - 5,
            )

            text = FONT.render(f"{player_text}'s Turn", True, WHITE)
            text_rect = text.get_rect(center=(WIDTH // 2, CELL_SIZE // 2))
            self.screen.blit(text, text_rect)
        else:
            if self.board.winner == PLAYER1:
                text = FONT.render("Player 1 (Red) Wins!", True, WHITE)
            elif self.board.winner == PLAYER2:
                text = FONT.render("Player 2 (Yellow) Wins!", True, WHITE)
            else:
                text = FONT.render("Game Draw!", True, WHITE)
            text_rect = text.get_rect(center=(WIDTH // 2, CELL_SIZE // 2))
            self.screen.blit(text, text_rect)

        # Draw restart instruction
        restart_text = FONT.render("Press R to restart", True, WHITE)
        restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT - 20))
        self.screen.blit(restart_text, restart_rect)

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.board.reset()

            elif event.type == pygame.MOUSEBUTTONDOWN and not self.board.game_over:
                if event.button == 1:  # Left mouse button
                    x, y = event.pos
                    col = x // CELL_SIZE

                    # Make move
                    result = self.board.drop_piece(col)
                    if result is not None:
                        row, col = result

                        # Check for win
                        if self.board.check_win(row, col):
                            self.board.game_over = True
                            self.board.winner = self.board.current_player
                        # Check for draw
                        elif self.board.is_full():
                            self.board.game_over = True
                            self.board.winner = None
                        else:
                            # Switch player if game not over
                            self.board.switch_player()

    def update(self):
        """Update game state."""
        pass  # Game logic handled in handle_events

    def render(self):
        """Render the game."""
        self.draw_board()
        pygame.display.flip()
        self.clock.tick(FPS)

    def start(self):
        """Start the game loop."""
        while self.is_running:
            self.handle_events()
            self.update()
            self.render()

        pygame.quit()


def main():
    game = Game()
    game.start()


if __name__ == "__main__":
    main()
