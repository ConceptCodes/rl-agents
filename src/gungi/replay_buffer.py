#!/usr/bin/env python3
"""Experience replay buffer for AlphaZero training."""

from __future__ import annotations

import pickle
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GameRecord:
    """Single training example from self-play."""

    state: np.ndarray  # Board encoding (planes, H, W)
    policy: np.ndarray  # MCTS visit distribution (num_actions,)
    value: float  # Game outcome from this player's perspective


class ReplayBuffer:
    """
    Experience replay buffer for storing self-play games.

    Stores (state, policy, value) tuples from self-play games
    and provides random sampling for training.
    """

    def __init__(self, max_size: int = 100_000):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of samples to store
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, record: GameRecord):
        """Add a single record to the buffer."""
        self.buffer.append(record)

    def add_game(self, records: List[GameRecord]):
        """Add all records from a single game."""
        for record in records:
            self.buffer.append(record)

    def sample(self, batch_size: int) -> List[GameRecord]:
        """
        Sample a random batch from the buffer.

        Args:
            batch_size: Number of samples to return

        Returns:
            List of GameRecord samples
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch and return as numpy arrays.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (states, policies, values) arrays
        """
        samples = self.sample(batch_size)

        states = np.stack([s.state for s in samples]).astype(np.float32)
        policies = np.stack([s.policy for s in samples]).astype(np.float32)
        values = np.array([s.value for s in samples], dtype=np.float32)

        return states, policies, values

    def clear(self):
        """Clear all samples from buffer."""
        self.buffer.clear()

    def save(self, path: str):
        """Save buffer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "max_size": self.max_size,
                    "buffer": list(self.buffer),
                },
                f,
            )
        print(f"Replay buffer saved to {path} ({len(self)} samples)")

    def load(self, path: str):
        """Load buffer from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.max_size = data["max_size"]
        self.buffer = deque(data["buffer"], maxlen=self.max_size)
        print(f"Replay buffer loaded from {path} ({len(self)} samples)")

    @classmethod
    def from_file(cls, path: str) -> "ReplayBuffer":
        """Create buffer from saved file."""
        buffer = cls()
        buffer.load(path)
        return buffer


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with prioritized sampling.

    Samples are weighted by their TD error or other priority metric.
    """

    def __init__(self, max_size: int = 100_000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.

        Args:
            max_size: Maximum number of samples
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        """
        super().__init__(max_size)
        self.alpha = alpha
        self.priorities: deque = deque(maxlen=max_size)
        self.default_priority = 1.0

    def add(self, record: GameRecord, priority: float = None):
        """Add a record with priority."""
        if priority is None:
            priority = self.default_priority

        self.buffer.append(record)
        self.priorities.append(priority)

    def add_game(self, records: List[GameRecord], priorities: List[float] = None):
        """Add all records from a game with priorities."""
        if priorities is None:
            priorities = [self.default_priority] * len(records)

        for record, priority in zip(records, priorities):
            self.add(record, priority)

    def sample(self, batch_size: int) -> Tuple[List[GameRecord], List[int]]:
        """
        Sample with priorities.

        Returns:
            Tuple of (samples, indices) for priority updating
        """
        if len(self.buffer) < batch_size:
            indices = list(range(len(self.buffer)))
            return list(self.buffer), indices

        # Calculate sampling probabilities
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer), size=batch_size, p=probs, replace=False
        )
        samples = [self.buffer[i] for i in indices]

        return samples, list(indices)

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled indices."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero


class GameHistory:
    """
    Stores the history of a single self-play game.

    Used to collect training data during self-play before
    adding to the replay buffer.
    """

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.players: List[int] = []  # 0 = black, 1 = white
        self.result: Optional[float] = None

    def add_step(self, state: np.ndarray, policy: np.ndarray, current_player: int):
        """Add a step to the history."""
        self.states.append(state)
        self.policies.append(policy)
        self.players.append(current_player)

    def set_result(self, result: float):
        """
        Set the game result.

        Args:
            result: +1 for black win, -1 for white win, 0 for draw
        """
        self.result = result

    def to_records(self) -> List[GameRecord]:
        """
        Convert history to training records.

        The value for each position is adjusted based on who moved
        and the final result.

        Returns:
            List of GameRecord for training
        """
        if self.result is None:
            raise ValueError("Game result must be set before converting to records")

        records = []
        for state, policy, player in zip(self.states, self.policies, self.players):
            # Value from this player's perspective
            # If black won (+1) and this was black's move (player=0), value = +1
            # If black won (+1) and this was white's move (player=1), value = -1
            value = self.result if player == 0 else -self.result

            records.append(
                GameRecord(
                    state=state,
                    policy=policy,
                    value=value,
                )
            )

        return records

    def __len__(self) -> int:
        return len(self.states)
