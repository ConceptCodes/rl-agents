#!/usr/bin/env python3
"""State and action encoding for AlphaZero-style training."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from constants import BOARD_SIZE, BLACK, WHITE
from utils.moves import Move, generate_legal_moves, generate_hand_drops, Ruleset

if TYPE_CHECKING:
    from game import Game, Piece

# =============================================================================
# Piece Type Encoding
# =============================================================================

PIECE_TYPE_IDS = {
    "MARSHAL": 0,
    "GENERAL": 1,
    "LIEUTENANT_GENERAL": 2,
    "MAJOR_GENERAL": 3,
    "WARRIOR": 4,
    "LANCER": 5,
    "RIDER": 6,
    "SPY": 7,
    "FORTRESS": 8,
    "SOLDIER": 9,
    "CANNON": 10,
    "ARCHER": 11,
    "MUSKETEER": 12,
    "TACTICIAN": 13,
}

PIECE_NAMES = list(PIECE_TYPE_IDS.keys())
NUM_PIECE_TYPES = len(PIECE_NAMES)
MAX_STACK_HEIGHT = 3

# =============================================================================
# State Encoding
# =============================================================================

# Input planes for neural network:
# - 14 piece types × 2 colors × 3 stack levels = 84 planes
# - 1 plane for current player (1 if black's turn, 0 if white's)
# - 1 plane for game phase (1 if game phase, 0 if setup phase)
# Total: 86 planes of 9×9

NUM_STATE_PLANES = NUM_PIECE_TYPES * 2 * MAX_STACK_HEIGHT + 2  # 84 + 2 = 86


def encode_board_state(game: Game) -> np.ndarray:
    """
    Encode Gungi board state as tensor for neural network.

    Args:
        game: Current game state

    Returns:
        np.ndarray: Shape (86, 9, 9) tensor

    Planes layout:
        0-41:  Black pieces (14 types × 3 stack levels)
        42-83: White pieces (14 types × 3 stack levels)
        84:    Current player (all 1s if black's turn)
        85:    Game phase (all 1s if game phase)
    """
    planes = np.zeros((NUM_STATE_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Encode pieces on board
    for col in range(BOARD_SIZE):
        for row in range(BOARD_SIZE):
            stack = game.board[col][row]
            for level, piece in enumerate(stack[:MAX_STACK_HEIGHT]):
                type_id = PIECE_TYPE_IDS.get(piece.name, 0)

                # Color offset: 0 for black, 42 for white
                if piece.color == BLACK:
                    color_offset = 0
                else:
                    color_offset = NUM_PIECE_TYPES * MAX_STACK_HEIGHT  # 42

                plane_idx = color_offset + type_id * MAX_STACK_HEIGHT + level
                planes[plane_idx, row, col] = 1.0

    # Current player plane (index 84)
    if game.turn == game.player_1:  # Black's turn
        planes[84, :, :] = 1.0

    # Game phase plane (index 85)
    if game.game_phase == "game":
        planes[85, :, :] = 1.0

    return planes


def encode_board_state_flat(game: Game) -> np.ndarray:
    """Encode board state as flat vector (for simpler networks)."""
    return encode_board_state(game).flatten()


# =============================================================================
# Action Space Encoding
# =============================================================================

"""
Action space design:

We use a flat action space that covers all possible moves:

1. BOARD MOVES (piece moves from one square to another):
   - src_square (81) × dst_square (81) = 6,561 combinations
   - But we only need moves, not all src×dst pairs
   - Encode as: src_col * 9 + src_row, dst_col * 9 + dst_row
   
2. DROPS (placing a piece from hand onto a friendly stack):
   - piece_type (14) × dst_square (81) = 1,134 combinations
   
3. SETUP MOVES (initial placement phase):
   - piece_type (14) × dst_square (81) = 1,134 combinations
   - (Could limit to valid rows, but masking handles this)

4. TURNCOAT (special captain move with replacement choice):
   - Treated as board moves with aux parameter
   - We'll handle this by having 3 variants per capture move for captain

Simplified encoding:
- Move actions:  action_idx = src_square * 81 + dst_square  (0 to 6560)
- Drop actions:  action_idx = 6561 + piece_type * 81 + dst_square (6561 to 7694)
- Setup actions: action_idx = 7695 + piece_type * 81 + dst_square (7695 to 8828)
- Turncoat aux:  We'll encode aux choice in upper bits or separate

For simplicity, we'll use:
- Total action space: ~9000 actions
- Illegal actions are masked out during MCTS
"""

# Action space sizes
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE  # 81
NUM_MOVE_ACTIONS = NUM_SQUARES * NUM_SQUARES  # 6561
NUM_DROP_ACTIONS = NUM_PIECE_TYPES * NUM_SQUARES  # 1134
NUM_SETUP_ACTIONS = NUM_PIECE_TYPES * NUM_SQUARES  # 1134
NUM_TURNCOAT_CHOICES = 3  # Max aux choices for turncoat

# Action type offsets
MOVE_OFFSET = 0
DROP_OFFSET = NUM_MOVE_ACTIONS  # 6561
SETUP_OFFSET = DROP_OFFSET + NUM_DROP_ACTIONS  # 7695
TURNCOAT_OFFSET = SETUP_OFFSET + NUM_SETUP_ACTIONS  # 8829

# Total actions (including turncoat variants)
# Turncoat: same as moves but with aux choice
NUM_TURNCOAT_ACTIONS = NUM_SQUARES * NUM_SQUARES * NUM_TURNCOAT_CHOICES  # 19683
TOTAL_ACTIONS = SETUP_OFFSET + NUM_SETUP_ACTIONS + NUM_TURNCOAT_ACTIONS  # 28512

# Simplified total (without separate turncoat encoding)
NUM_ACTIONS = SETUP_OFFSET + NUM_SETUP_ACTIONS  # 8829


def square_to_idx(col: int, row: int) -> int:
    """Convert (col, row) to square index (0-80)."""
    return col * BOARD_SIZE + row


def idx_to_square(idx: int) -> Tuple[int, int]:
    """Convert square index to (col, row)."""
    return idx // BOARD_SIZE, idx % BOARD_SIZE


def encode_move_action(src: Tuple[int, int], dst: Tuple[int, int]) -> int:
    """Encode a board move as action index."""
    src_idx = square_to_idx(src[0], src[1])
    dst_idx = square_to_idx(dst[0], dst[1])
    return MOVE_OFFSET + src_idx * NUM_SQUARES + dst_idx


def encode_drop_action(piece_name: str, dst: Tuple[int, int]) -> int:
    """Encode a drop action as action index."""
    piece_idx = PIECE_TYPE_IDS.get(piece_name, 0)
    dst_idx = square_to_idx(dst[0], dst[1])
    return DROP_OFFSET + piece_idx * NUM_SQUARES + dst_idx


def encode_setup_action(piece_name: str, dst: Tuple[int, int]) -> int:
    """Encode a setup action as action index."""
    piece_idx = PIECE_TYPE_IDS.get(piece_name, 0)
    dst_idx = square_to_idx(dst[0], dst[1])
    return SETUP_OFFSET + piece_idx * NUM_SQUARES + dst_idx


def decode_action(action_idx: int) -> dict:
    """
    Decode action index back to action components.

    Returns:
        dict with keys: 'type', and type-specific fields
        - Move: {'type': 'move', 'src': (col, row), 'dst': (col, row)}
        - Drop: {'type': 'drop', 'piece': str, 'dst': (col, row)}
        - Setup: {'type': 'setup', 'piece': str, 'dst': (col, row)}
    """
    if action_idx < DROP_OFFSET:
        # Move action
        move_idx = action_idx - MOVE_OFFSET
        src_idx = move_idx // NUM_SQUARES
        dst_idx = move_idx % NUM_SQUARES
        return {
            "type": "move",
            "src": idx_to_square(src_idx),
            "dst": idx_to_square(dst_idx),
        }
    elif action_idx < SETUP_OFFSET:
        # Drop action
        drop_idx = action_idx - DROP_OFFSET
        piece_idx = drop_idx // NUM_SQUARES
        dst_idx = drop_idx % NUM_SQUARES
        return {
            "type": "drop",
            "piece": PIECE_NAMES[piece_idx],
            "dst": idx_to_square(dst_idx),
        }
    else:
        # Setup action
        setup_idx = action_idx - SETUP_OFFSET
        piece_idx = setup_idx // NUM_SQUARES
        dst_idx = setup_idx % NUM_SQUARES
        return {
            "type": "setup",
            "piece": PIECE_NAMES[piece_idx],
            "dst": idx_to_square(dst_idx),
        }


def encode_action(move: Move, game: Game) -> int:
    """
    Convert a Move object to action index.

    Args:
        move: Move object from move generation
        game: Current game state (for context)

    Returns:
        Action index
    """
    if move.frm is None:
        # Drop or setup move
        if game.game_phase == "initial_setup":
            return encode_setup_action(move.piece, move.to)
        else:
            return encode_drop_action(move.piece, move.to)
    else:
        # Board move
        return encode_move_action(move.frm, move.to)


# =============================================================================
# Legal Action Mask
# =============================================================================


def get_legal_moves(game: Game) -> List[Move]:
    """Get all legal moves for the current player."""
    moves = []
    player = game.turn
    rules = Ruleset(board_size=BOARD_SIZE, hand=player.hand_pieces)

    if game.game_phase == "initial_setup":
        # Setup phase: can place pieces from hand
        for piece in player.hand_pieces:
            # Determine valid rows based on color
            if player.color == BLACK:
                valid_rows = range(0, 3)  # Rows 0-2 for black
            else:
                valid_rows = range(6, 9)  # Rows 6-8 for white

            for col in range(BOARD_SIZE):
                for row in valid_rows:
                    # Check if placement is valid
                    stack = game.board[col][row]
                    if len(stack) >= 3:
                        continue
                    if piece.name == "MARSHAL" and len(stack) > 0:
                        continue
                    if stack and stack[-1].name == "MARSHAL":
                        continue

                    moves.append(
                        Move(piece=piece.name, frm=None, to=(col, row), action="setup")
                    )
    else:
        # Game phase: board moves and drops
        # Board moves
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                stack = game.board[col][row]
                if not stack:
                    continue
                top_piece = stack[-1]
                if top_piece.color != player.color:
                    continue

                piece_moves = generate_legal_moves(
                    top_piece, (col, row), game.board, rules
                )
                moves.extend(piece_moves)

        # Drop moves
        seen_piece_types = set()
        for piece in player.hand_pieces:
            if piece.name in seen_piece_types:
                continue
            seen_piece_types.add(piece.name)

            drop_moves = generate_hand_drops(
                piece.name, player.color, game.board, rules
            )
            moves.extend(drop_moves)

    return moves


def get_legal_action_mask(game: Game) -> np.ndarray:
    """
    Get binary mask of legal actions.

    Args:
        game: Current game state

    Returns:
        np.ndarray: Shape (NUM_ACTIONS,), 1 for legal, 0 for illegal
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)

    for move in get_legal_moves(game):
        action_idx = encode_action(move, game)
        if 0 <= action_idx < NUM_ACTIONS:
            mask[action_idx] = 1.0

    return mask


def get_legal_action_indices(game: Game) -> List[int]:
    """Get list of legal action indices."""
    indices = []
    for move in get_legal_moves(game):
        action_idx = encode_action(move, game)
        if 0 <= action_idx < NUM_ACTIONS:
            indices.append(action_idx)
    return indices


# =============================================================================
# Action Application
# =============================================================================


def action_to_move(action_idx: int, game: Game) -> Optional[Move]:
    """
    Convert action index to a Move object that can be applied.

    Args:
        action_idx: Action index
        game: Current game state

    Returns:
        Move object or None if invalid
    """
    decoded = decode_action(action_idx)

    if decoded["type"] == "setup":
        return Move(piece=decoded["piece"], frm=None, to=decoded["dst"], action="setup")
    elif decoded["type"] == "drop":
        return Move(piece=decoded["piece"], frm=None, to=decoded["dst"], action="drop")
    else:  # move
        src = decoded["src"]
        dst = decoded["dst"]

        # Find the actual move from legal moves to get correct action type
        stack = game.board[src[0]][src[1]]
        if not stack:
            return None

        piece = stack[-1]
        rules = Ruleset(board_size=BOARD_SIZE, hand=game.turn.hand_pieces)
        legal_moves = generate_legal_moves(piece, src, game.board, rules)

        for move in legal_moves:
            if move.to == dst:
                return move

        # Default to simple move if not found in legal moves
        return Move(
            piece=piece.name if hasattr(piece, "name") else "UNKNOWN",
            frm=src,
            to=dst,
            action="move",
        )
