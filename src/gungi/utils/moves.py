from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from constants import BOARD_SIZE, BLACK, WHITE

BoardPos = Tuple[int, int]
StepVector = Tuple[int, int, int]
RayVector = Tuple[int, int]


@dataclass(frozen=True)
class Move:
    piece: str
    frm: Optional[BoardPos]
    to: BoardPos
    action: str
    aux: Optional[dict[str, object]] = None


@dataclass
class Ruleset:
    board_size: int = BOARD_SIZE
    stack_limit: int = 3
    hand: Optional[Sequence[Any]] = None


PIECE_ALIASES = {
    "LIEUTENANT_GENERAL": "LT_GENERAL",
    "WARRIOR": "SAMURAI",
    "LANCER": "SPEAR",
    "RIDER": "KNIGHT",
    "SOLDIER": "PAWN",
    "TACTICIAN": "CAPTAIN",
}

ORTHOGONAL_DIRS: tuple[RayVector, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL_DIRS: tuple[RayVector, ...] = ((1, 1), (-1, 1), (1, -1), (-1, -1))


def validate_move(
    piece,
    start_pos: Optional[BoardPos],
    end_pos: Optional[BoardPos],
    board,
    rules: Optional[Ruleset] = None,
) -> bool:
    rules = rules or Ruleset()
    if start_pos is None or end_pos is None:
        return False
    if not _in_bounds(start_pos, rules.board_size) or not _in_bounds(
        end_pos, rules.board_size
    ):
        return False

    stack = _stack_at(board, start_pos, rules.board_size)
    if not stack or stack[-1] is not piece:
        return False

    for move in generate_legal_moves(piece, start_pos, board, rules):
        if move.to == end_pos:
            return True
    return False


def generate_legal_moves(
    piece, position: Optional[BoardPos], board, rules: Optional[Ruleset] = None
) -> list[Move]:
    rules = rules or Ruleset()
    if position is None or not _in_bounds(position, rules.board_size):
        return []

    stack = _stack_at(board, position, rules.board_size)
    if not stack or stack[-1] is not piece:
        return []

    canonical = _canonical_name(getattr(piece, "name", ""))
    generator = PIECE_ROUTER.get(canonical)
    if not generator:
        return []
    return generator(piece, position, board, rules)


def generate_hand_drops(
    piece_name: str,
    color,
    board,
    rules: Optional[Ruleset] = None,
) -> list[Move]:
    rules = rules or Ruleset()
    canonical_piece = _canonical_name(piece_name)
    front_limit = _frontmost_row(color, board, rules.board_size)
    drop_moves: list[Move] = []

    for col in range(rules.board_size):
        for row in range(rules.board_size):
            stack = _stack_at(board, (col, row), rules.board_size)
            if not stack or stack[-1].color != color:
                continue
            if len(stack) >= rules.stack_limit:
                continue
            if _canonical_name(getattr(stack[-1], "name", "")) == "MARSHAL":
                continue

            if color == BLACK and row > front_limit:
                continue
            if color == WHITE and row < front_limit:
                continue

            drop_moves.append(
                Move(piece=canonical_piece, frm=None, to=(col, row), action="drop")
            )
    return drop_moves


def _marshal(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    steps: list[StepVector] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            steps.append((dx, dy, 1))
    return _add_scaled_steps(piece, src, board, steps, rules)


def _general(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    moves = _add_rays(piece, src, board, ORTHOGONAL_DIRS, rules)
    diag_steps = [(dx, dy, 1) for dx, dy in DIAGONAL_DIRS]
    moves.extend(_add_scaled_steps(piece, src, board, diag_steps, rules))
    return moves


def _lt_general(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    moves = _add_scaled_steps(
        piece, src, board, [(dx, dy, 1) for dx, dy in ORTHOGONAL_DIRS], rules
    )
    moves.extend(_add_rays(piece, src, board, DIAGONAL_DIRS, rules))
    return moves


def _major_general(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    forward = _forward_dir(piece)
    steps: list[StepVector] = [
        (1, 0, 1),
        (-1, 0, 1),
        (0, 1, 1),
        (0, -1, 1),
        (1, forward, 1),
        (-1, forward, 1),
    ]
    return _add_scaled_steps(piece, src, board, steps, rules)


def _samurai(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    forward = _forward_dir(piece)
    steps = [
        (0, forward, 1),
        (0, -forward, 1),
        (-1, forward, 1),
        (1, forward, 1),
    ]
    return _add_scaled_steps(piece, src, board, steps, rules)


def _spear(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    forward = _forward_dir(piece)
    steps = [
        (0, forward, 2),
        (-1, forward, 1),
        (1, forward, 1),
        (0, -forward, 1),
    ]
    return _add_scaled_steps(piece, src, board, steps, rules)


def _knight(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    forward = _forward_dir(piece)
    steps = [
        (0, forward, 2),
        (0, -forward, 2),
        (1, 0, 1),
        (-1, 0, 1),
    ]
    return _add_scaled_steps(piece, src, board, steps, rules)


def _spy(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    steps = [(dx, dy, 2) for dx, dy in DIAGONAL_DIRS]
    return _add_scaled_steps(piece, src, board, steps, rules)


def _fortress(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    forward = _forward_dir(piece)
    steps = [
        (0, forward, 1),
        (1, 0, 1),
        (-1, 0, 1),
        (1, -forward, 1),
        (-1, -forward, 1),
    ]
    return _add_scaled_steps(piece, src, board, steps, rules)


def _pawn(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    forward = _forward_dir(piece)
    steps = [(0, forward, 1), (0, -forward, 1)]
    return _add_scaled_steps(piece, src, board, steps, rules)


def _cannon(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    forward = _forward_dir(piece)
    steps = [(1, 0, 1), (-1, 0, 1), (0, -forward, 1)]
    moves = _add_scaled_steps(piece, src, board, steps, rules)
    jump_specs = [(0, forward, 3)]
    moves.extend(_jumper_landings(piece, src, board, rules, jump_specs))
    return moves


def _archer(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    forward = _forward_dir(piece)
    steps = [(0, -forward, 1)]
    moves = _add_scaled_steps(piece, src, board, steps, rules)
    jump_specs = [(0, forward, 2), (1, forward, 2), (-1, forward, 2)]
    moves.extend(_jumper_landings(piece, src, board, rules, jump_specs))
    return moves


def _musketeer(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    forward = _forward_dir(piece)
    steps = [(1, -forward, 1), (-1, -forward, 1)]
    moves = _add_scaled_steps(piece, src, board, steps, rules)
    moves.extend(_jumper_landings(piece, src, board, rules, [(0, forward, 2)]))
    return moves


def _captain(piece, src: BoardPos, board, rules: Ruleset) -> list[Move]:
    forward = _forward_dir(piece)
    steps = [(1, forward, 1), (-1, forward, 1), (0, -forward, 1)]
    moves = _add_scaled_steps(piece, src, board, steps, rules)
    moves.extend(_captain_turncoat_moves(piece, moves, board, rules))
    return moves


PIECE_ROUTER = {
    "MARSHAL": _marshal,
    "GENERAL": _general,
    "LT_GENERAL": _lt_general,
    "MAJOR_GENERAL": _major_general,
    "SAMURAI": _samurai,
    "SPEAR": _spear,
    "KNIGHT": _knight,
    "SPY": _spy,
    "FORTRESS": _fortress,
    "PAWN": _pawn,
    "CANNON": _cannon,
    "ARCHER": _archer,
    "MUSKETEER": _musketeer,
    "CAPTAIN": _captain,
}


def _add_scaled_steps(
    piece, src: BoardPos, board, steps: list[StepVector], rules: Ruleset
) -> list[Move]:
    moves: list[Move] = []
    moving_height = _height(board, src, rules.board_size)
    if moving_height == 0:
        return moves

    bonus = moving_height - 1
    for dx, dy, base_len in steps:
        for distance in range(base_len, base_len + bonus + 1):
            dst = (src[0] + dx * distance, src[1] + dy * distance)
            if not _in_bounds(dst, rules.board_size):
                break
            if distance > 1 and not _path_clear(board, src, dst):
                continue
            evaluated = _eval_dst(piece, src, dst, board, rules, moving_height)
            if evaluated:
                moves.append(evaluated)
    return moves


def _add_rays(
    piece, src: BoardPos, board, directions: tuple[RayVector, ...], rules: Ruleset
) -> list[Move]:
    moves: list[Move] = []
    moving_height = _height(board, src, rules.board_size)
    for dx, dy in directions:
        step = 1
        while True:
            dst = (src[0] + dx * step, src[1] + dy * step)
            if not _in_bounds(dst, rules.board_size):
                break
            stack = _stack_at(board, dst, rules.board_size)
            evaluated = _eval_dst(piece, src, dst, board, rules, moving_height)
            if evaluated:
                moves.append(evaluated)
            if stack:
                break
            step += 1
    return moves


def _jumper_landings(
    piece,
    src: BoardPos,
    board,
    rules: Ruleset,
    jump_specs: list[StepVector],
) -> list[Move]:
    moves: list[Move] = []
    moving_height = _height(board, src, rules.board_size)
    if moving_height == 0:
        return moves

    bonus = moving_height - 1
    for dx, dy, base_len in jump_specs:
        for distance in range(base_len, base_len + bonus + 1):
            dst = (src[0] + dx * distance, src[1] + dy * distance)
            if not _in_bounds(dst, rules.board_size):
                break
            if not _can_jump_over(board, src, dst, moving_height):
                continue
            evaluated = _eval_dst(piece, src, dst, board, rules, moving_height)
            if evaluated:
                moves.append(evaluated)
    return moves


def _can_jump_over(board, src: BoardPos, dst: BoardPos, moving_height: int) -> bool:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    steps = max(abs(dx), abs(dy))
    if steps <= 1:
        return False
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    cur = (src[0] + step_x, src[1] + step_y)
    while cur != dst:
        stack = board[cur[0]][cur[1]]
        if not stack or len(stack) > moving_height:
            return False
        cur = (cur[0] + step_x, cur[1] + step_y)
    return True


def _eval_dst(
    piece,
    src: BoardPos,
    dst: BoardPos,
    board,
    rules: Ruleset,
    moving_height: int,
) -> Optional[Move]:
    stack = _stack_at(board, dst, rules.board_size)
    canonical_piece = _canonical_name(getattr(piece, "name", ""))
    if not stack:
        return Move(piece=canonical_piece, frm=src, to=dst, action="move")

    top = stack[-1]
    if top.color == piece.color:
        if not _can_stack_onto(stack, moving_height, rules):
            return None
        return Move(piece=canonical_piece, frm=src, to=dst, action="stack")

    if len(stack) > moving_height:
        return None
    return Move(piece=canonical_piece, frm=src, to=dst, action="capture")


def _captain_turncoat_moves(
    piece, base_moves: list[Move], board, rules: Ruleset
) -> list[Move]:
    hand_names = _hand_canonical_names(rules.hand)
    if not hand_names:
        return []

    extra_moves: list[Move] = []
    for move in base_moves:
        if move.action != "capture":
            continue
        stack = _stack_at(board, move.to, rules.board_size)
        if not stack:
            continue
        candidates: list[str] = []
        top = stack[-1]
        top_name = _canonical_name(getattr(top, "name", ""))
        if top_name in hand_names:
            candidates.append(top_name)
        if len(stack) >= 2:
            second = stack[-2]
            second_name = _canonical_name(getattr(second, "name", ""))
            if second_name in hand_names:
                candidates.append(second_name)
        if not candidates:
            continue
        extra_moves.append(
            Move(
                piece=_canonical_name(getattr(piece, "name", "")),
                frm=move.frm,
                to=move.to,
                action="turncoat",
                aux={"replacements": sorted(set(candidates))},
            )
        )
    return extra_moves


def _can_stack_onto(dest_stack, moving_height: int, rules: Ruleset) -> bool:
    if len(dest_stack) >= rules.stack_limit:
        return False
    if _canonical_name(getattr(dest_stack[-1], "name", "")) == "MARSHAL":
        return False
    return len(dest_stack) <= moving_height


def _path_clear(board, src: BoardPos, dst: BoardPos) -> bool:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    steps = max(abs(dx), abs(dy))
    if steps <= 1:
        return True
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    cur = (src[0] + step_x, src[1] + step_y)
    while cur != dst:
        if board[cur[0]][cur[1]]:
            return False
        cur = (cur[0] + step_x, cur[1] + step_y)
    return True


def _stack_at(board, pos: BoardPos, board_size: int):
    if not _in_bounds(pos, board_size):
        return None
    return board[pos[0]][pos[1]]


def _height(board, pos: BoardPos, board_size: int) -> int:
    stack = _stack_at(board, pos, board_size)
    return len(stack) if stack else 0


def _frontmost_row(color, board, board_size: int) -> int:
    rows: list[int] = []
    for col in range(board_size):
        for row in range(board_size):
            stack = board[col][row]
            if stack and stack[-1].color == color:
                rows.append(row)
    if not rows:
        return 0 if color == BLACK else board_size - 1
    return max(rows) if color == BLACK else min(rows)


def _forward_dir(piece) -> int:
    return 1 if getattr(piece, "color", BLACK) == BLACK else -1


def _hand_canonical_names(hand: Optional[Sequence[Any]]) -> set[str]:
    if not hand:
        return set()
    names: set[str] = set()
    for entry in hand:
        if entry is None:
            continue
        if isinstance(entry, str):
            names.add(_canonical_name(entry))
        elif hasattr(entry, "name"):
            names.add(_canonical_name(getattr(entry, "name", "")))
    return names


def _canonical_name(name: str) -> str:
    normalized = (name or "").upper()
    return PIECE_ALIASES.get(normalized, normalized)


def _in_bounds(pos: BoardPos, board_size: int) -> bool:
    return 0 <= pos[0] < board_size and 0 <= pos[1] < board_size


def canonical_name(name: str) -> str:
    """Public helper for external callers that need canonicalized names."""
    return _canonical_name(name)
