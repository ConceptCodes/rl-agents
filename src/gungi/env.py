import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, List, Dict

from game import Game
from constants import BOARD_SIZE, BLACK, WHITE
from utils.moves import (
    Move,
    Ruleset,
    canonical_name,
    generate_hand_drops,
    generate_legal_moves,
)

# Piece type IDs for encoding
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

PIECE_NAMES: List[str] = list(PIECE_TYPE_IDS.keys())
NUM_PIECE_TYPES = len(PIECE_NAMES)

ACTION_SETUP = 0
ACTION_MOVE = 1
ACTION_DROP = 2

# 🎯 What skill should the agent learn? [How to play Gungi]
# 👀 What information does the agent need? [board state, current player, phase, hand pieces]
# 🎮 What actions can the agent take? [Discrete choices: placeholder for setup and moves]
# 🏆 How do we measure success? [Winning the game]
# ⏰ When should episodes end? [Game over, maximum steps reached]


class GungiEnv(gym.Env):
    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        self.game = Game(
            title="Gungi - RL Training", render_ui=self.render_mode == "human"
        )

        # Observation: board (9x9x3), current_player, phase, legal counts, hand_pieces (14*2)
        board_size = BOARD_SIZE * BOARD_SIZE * 3  # 9x9x3 for stacks
        hand_size = NUM_PIECE_TYPES * 2  # counts per player
        obs_size = board_size + 1 + 1 + 2 + hand_size  # 243 + 1 + 1 + 2 + 28 = 275
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=50.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Action: [kind, piece_type, src_row, src_col, dst_row, dst_col, aux]
        # kind: 0=setup, 1=board move, 2=drop
        # piece_type: index into PIECE_NAMES
        # src/dst: board coordinates (row/col)
        # aux: optional turncoat index (0-2)
        self.action_space = gym.spaces.MultiDiscrete(
            [3, NUM_PIECE_TYPES, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, 3]
        )

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            np.array: Observation with board, player, phase, hand pieces
        """
        # Board: 9x9x3, each cell up to 3 pieces, encoded as type_id * 2 + color, -1 if empty
        board_obs = np.full((BOARD_SIZE, BOARD_SIZE, 3), -1.0, dtype=np.float32)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                stack = self.game.board[col][row]
                for i in range(min(len(stack), 3)):
                    piece = stack[i]
                    type_id = PIECE_TYPE_IDS[piece.name]
                    color = 0 if piece.color == BLACK else 1
                    board_obs[row, col, i] = type_id * 2 + color

        # Current player: 0 if player_1 (black), 1 if player_2 (white)
        current_player = 0 if self.game.turn == self.game.player_1 else 1

        # Phase: 0 for initial_setup, 1 for game
        phase = 1.0 if self.game.game_phase == "game" else 0.0

        # Hand pieces: counts for each type, for player_1 and player_2
        hand_obs = np.zeros(NUM_PIECE_TYPES * 2, dtype=np.float32)
        for player_idx, player in enumerate([self.game.player_1, self.game.player_2]):
            hand_counts = self._hand_counts(player.hand_pieces)
            for type_name, type_id in PIECE_TYPE_IDS.items():
                count = hand_counts.get(type_name, 0)
                hand_obs[player_idx * NUM_PIECE_TYPES + type_id] = count

        legal_move_count = float(self._legal_move_count(self.game.turn))
        legal_drop_count = float(self._legal_drop_count(self.game.turn))

        obs = np.concatenate(
            [
                board_obs.flatten(),
                np.array([current_player], dtype=np.float32),
                np.array([phase], dtype=np.float32),
                np.array([legal_move_count, legal_drop_count], dtype=np.float32),
                hand_obs,
            ]
        )
        return obs

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with game state details
        """
        return {
            "current_player": "Black" if self.game.turn == self.game.player_1 else "White",
            "phase": self.game.game_phase,
            "step": self.current_step,
            "board": self.game.board,  # Full board for debugging
            "hand_p1": [p.name for p in self.game.player_1.hand_pieces],
            "hand_p2": [p.name for p in self.game.player_2.hand_pieces],
            "legal_move_count": self._legal_move_count(self.game.turn),
            "legal_drop_count": self._legal_drop_count(self.game.turn),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)

        self.game.reset()
        self.current_step = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: MultiDiscrete [kind, piece_type, src_row, src_col, dst_row, dst_col, aux]

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        action = np.asarray(action, dtype=np.int64)
        action_kind, piece_type_idx, src_row, src_col, dst_row, dst_col, aux = action

        reward = -0.01  # default step penalty
        terminated = False
        info = {"phase": self.game.game_phase}

        piece_type_idx = int(np.clip(piece_type_idx, 0, NUM_PIECE_TYPES - 1))
        piece_name = PIECE_NAMES[piece_type_idx]
        src = (int(src_col), int(src_row))
        dst = (int(dst_col), int(dst_row))
        aux = int(aux)

        if action_kind == ACTION_SETUP:
            success = self._handle_setup_action(piece_name, dst)
            if success:
                reward = 0.1
                info["last_action"] = "setup"
            else:
                reward = -0.1
                info["invalid_action"] = "invalid_setup"

        elif action_kind == ACTION_MOVE:
            success, delta_reward, terminated, event = self._handle_move_action(src, dst, aux)
            if success:
                reward = reward + delta_reward
                info.update(event)
            else:
                reward = -0.1
                info["invalid_action"] = "illegal_move"

        elif action_kind == ACTION_DROP:
            success, delta_reward, drop_terminated, drop_event = self._handle_drop_action(
                piece_name, dst
            )
            if success:
                reward = reward + delta_reward
                info.update(drop_event)
                terminated = drop_terminated
            else:
                reward = -0.1
                info["invalid_action"] = "illegal_drop"
        else:
            info["invalid_action"] = "unknown_action"
            reward = -0.1

        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info.update(self._get_info())

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_setup_action(self, piece_name: str, dst: Tuple[int, int]) -> bool:
        if self.game.game_phase != "initial_setup":
            return False
        piece = next((p for p in self.game.turn.hand_pieces if p.name == piece_name), None)
        if not piece:
            return False
        row = dst[1]
        col = dst[0]
        return bool(self.game.handle_setup_move(piece, row, col))

    def _handle_move_action(
        self, src: Tuple[int, int], dst: Tuple[int, int], aux_choice: int
    ) -> Tuple[bool, float, bool, Dict[str, object]]:
        event = {"last_action": "move"}
        if self.game.game_phase != "game":
            return False, 0.0, False, event
        if not self._in_bounds(src) or not self._in_bounds(dst):
            return False, 0.0, False, event

        stack = self.game.board[src[0]][src[1]]
        if not stack:
            return False, 0.0, False, event
        piece = stack[-1]
        if piece.color != self.game.turn.color:
            return False, 0.0, False, event

        rules = self._ruleset_for_player(self.game.turn)
        legal_moves = generate_legal_moves(piece, src, self.game.board, rules)
        move = next((m for m in legal_moves if m.to == dst), None)
        if not move:
            return False, 0.0, False, event

        reward_delta = 0.0
        terminated = False
        marshal_captured = False

        if move.action in {"move", "stack"}:
            self._move_piece(src, dst)
        elif move.action == "capture":
            captured_stack = self._capture_stack(dst)
            if captured_stack:
                reward_delta += 0.2
                if self._stack_contains_marshal(captured_stack):
                    reward_delta += 1.0
                    terminated = True
                    marshal_captured = True
                    event["result"] = "win_by_marshal_capture"
            self._move_piece(src, dst)
        elif move.action == "turncoat":
            success, turncoat_reward = self._execute_turncoat_move(move, aux_choice)
            if success:
                reward_delta += turncoat_reward
            else:
                captured_stack = self._capture_stack(dst)
                reward_delta += 0.2
                if self._stack_contains_marshal(captured_stack):
                    reward_delta += 1.0
                    terminated = True
                    marshal_captured = True
                    event["result"] = "win_by_marshal_capture"
                self._move_piece(src, dst)
        else:
            return False, 0.0, False, event

        event["move_action"] = move.action
        if marshal_captured:
            event["captured_marshal"] = True

        if not terminated:
            self.game._switch_turn()
            extra_term, bonus_reward, extra_event = self._evaluate_post_action_state()
            reward_delta += bonus_reward
            terminated = extra_term
            event.update(extra_event)

        return True, reward_delta, terminated, event

    def _handle_drop_action(
        self, piece_name: str, dst: Tuple[int, int]
    ) -> Tuple[bool, float, bool, Dict[str, object]]:
        event = {"last_action": "drop"}
        if self.game.game_phase != "game":
            return False, 0.0, False, event
        piece = next((p for p in self.game.turn.hand_pieces if p.name == piece_name), None)
        if not piece:
            return False, 0.0, False, event

        rules = self._ruleset_for_player(self.game.turn)
        drop_moves = generate_hand_drops(
            piece_name, self.game.turn.color, self.game.board, rules
        )
        target = next((m for m in drop_moves if m.to == dst), None)
        if not target:
            return False, 0.0, False, event

        self.game.turn.hand_pieces.remove(piece)
        self.game.board[dst[0]][dst[1]].append(piece)

        self.game._switch_turn()
        extra_term, bonus_reward, extra_event = self._evaluate_post_action_state()
        reward_delta = 0.05 + bonus_reward
        event.update(extra_event)
        return True, reward_delta, extra_term, event

    def _evaluate_post_action_state(self) -> Tuple[bool, float, Dict[str, object]]:
        if self.game.game_phase != "game":
            return False, 0.0, {}
        legal_moves = self._legal_move_count(self.game.turn)
        drop_moves = self._legal_drop_count(self.game.turn)
        if legal_moves == 0 and drop_moves == 0:
            return True, 1.0, {"result": "opponent_no_moves"}
        return False, 0.0, {}

    def _move_piece(self, src: Tuple[int, int], dst: Tuple[int, int]):
        stack = self.game.board[src[0]][src[1]]
        if not stack:
            return
        piece = stack.pop()
        self.game.board[dst[0]][dst[1]].append(piece)

    def _capture_stack(self, dst: Tuple[int, int]) -> List[object]:
        stack = self.game.board[dst[0]][dst[1]]
        captured = list(stack)
        stack.clear()
        self._collect_captured_pieces(captured)
        return captured

    def _execute_turncoat_move(self, move: Move, aux_choice: int) -> Tuple[bool, float]:
        replacements = (move.aux or {}).get("replacements", [])
        if not replacements:
            return False, 0.0
        idx = min(max(aux_choice, 0), len(replacements) - 1)
        replacement_name = replacements[idx]
        replacement_piece = self._remove_piece_from_hand(
            self.game.turn, replacement_name
        )
        if not replacement_piece:
            return False, 0.0

        dst_stack = self.game.board[move.to[0]][move.to[1]]
        target_idx = self._find_turncoat_target(dst_stack, replacement_name)
        if target_idx is None:
            self.game.turn.hand_pieces.append(replacement_piece)
            return False, 0.0

        enemy_piece = dst_stack.pop(target_idx)
        enemy_piece.color = self.game.turn.color
        self.game.turn.hand_pieces.append(enemy_piece)

        opponent = self._opponent_player(self.game.turn)
        replacement_piece.color = opponent.color
        dst_stack.insert(target_idx, replacement_piece)

        self._move_piece(move.frm, move.to)
        return True, 0.2

    def _collect_captured_pieces(self, captured_stack: List[object]):
        if not captured_stack:
            return
        for piece in captured_stack:
            piece.color = self.game.turn.color
            self.game.turn.hand_pieces.append(piece)

    def _remove_piece_from_hand(self, player, canonical_piece: str):
        target_name = canonical_name(canonical_piece)
        for idx, piece in enumerate(player.hand_pieces):
            if canonical_name(piece.name) == target_name:
                return player.hand_pieces.pop(idx)
        return None

    def _find_turncoat_target(self, stack: List[object], canonical_piece: str):
        if not stack:
            return None
        for offset in range(1, min(2, len(stack)) + 1):
            idx = len(stack) - offset
            if canonical_name(stack[idx].name) == canonical_piece:
                return idx
        return None

    def _ruleset_for_player(self, player) -> Ruleset:
        return Ruleset(board_size=BOARD_SIZE, hand=player.hand_pieces)

    def _legal_move_count(self, player) -> int:
        return sum(len(moves) for moves in self._legal_moves_per_stack(player))

    def _legal_moves_per_stack(self, player):
        rules = self._ruleset_for_player(player)
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                stack = self.game.board[col][row]
                if not stack:
                    continue
                top = stack[-1]
                if top.color != player.color:
                    continue
                yield generate_legal_moves(top, (col, row), self.game.board, rules)

    def _legal_drop_count(self, player) -> int:
        if not player.hand_pieces:
            return 0
        rules = self._ruleset_for_player(player)
        counts = self._hand_counts(player.hand_pieces)
        drop_total = 0
        for piece_name, count in counts.items():
            if count <= 0:
                continue
            moves = generate_hand_drops(piece_name, player.color, self.game.board, rules)
            drop_total += len(moves) * count
        return drop_total

    def _hand_counts(self, hand) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for piece in hand:
            counts[piece.name] = counts.get(piece.name, 0) + 1
        return counts

    def _stack_contains_marshal(self, captured_stack: List[object]) -> bool:
        return any(piece.name == "MARSHAL" for piece in captured_stack)

    def _opponent_player(self, player):
        return self.game.player_2 if player is self.game.player_1 else self.game.player_1

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < BOARD_SIZE and 0 <= pos[1] < BOARD_SIZE

    def render(self):
        if self.render_mode == "human":
            self.game._render()

    def close(self):
        if self.render_mode == "human":
            self.game.end()
