#!/usr/bin/env python3
"""Monte Carlo Tree Search for AlphaZero-style Gungi agent."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from game import Game
    from network import GungiNetwork


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""

    num_simulations: int = 800  # AlphaZero standard
    c_puct: float = 1.5  # Exploration constant
    dirichlet_alpha: float = 0.15  # ~10/avg_legal_moves, Gungi has ~50-200 moves
    dirichlet_epsilon: float = 0.25  # Weight for noise at root
    temperature: float = 1.0  # Temperature for action selection


class MCTSNode:
    """Node in the MCTS tree."""

    __slots__ = ["visit_count", "value_sum", "prior", "children", "is_expanded"]

    def __init__(self, prior: float = 0.0):
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior  # P(a|s) from policy network
        self.children: Dict[int, MCTSNode] = {}  # action_idx -> child node
        self.is_expanded: bool = False

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """
        Upper Confidence Bound for Trees (UCT) score.

        Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        exploration = (
            c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        )
        return self.value + exploration

    def select_child(self, c_puct: float) -> Tuple[int, "MCTSNode"]:
        """Select child with highest UCB score."""
        best_score = float("-inf")
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, policy_probs: np.ndarray, legal_actions: List[int]):
        """
        Expand this node with children for each legal action.

        Args:
            policy_probs: Policy probabilities from neural network (already masked)
            legal_actions: List of legal action indices
        """
        for action in legal_actions:
            if action not in self.children:
                self.children[action] = MCTSNode(prior=policy_probs[action])
        self.is_expanded = True

    def add_dirichlet_noise(
        self, legal_actions: List[int], alpha: float, epsilon: float
    ):
        """Add Dirichlet noise to prior probabilities at root for exploration."""
        noise = np.random.dirichlet([alpha] * len(legal_actions))
        for i, action in enumerate(legal_actions):
            if action in self.children:
                self.children[action].prior = (1 - epsilon) * self.children[
                    action
                ].prior + epsilon * noise[i]


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Implements the AlphaZero MCTS algorithm:
    1. SELECT: Traverse tree using UCB to find leaf node
    2. EXPAND: Expand leaf node using neural network policy
    3. EVALUATE: Get value estimate from neural network
    4. BACKPROPAGATE: Update values along the path
    """

    def __init__(
        self,
        network: "GungiNetwork",
        config: MCTSConfig = None,
        device: torch.device = None,
    ):
        self.network = network
        self.config = config or MCTSConfig()
        self.device = device or torch.device("cpu")

    def search(
        self,
        game: "Game",
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Run MCTS from current game state.

        Args:
            game: Current game state
            add_noise: Whether to add Dirichlet noise at root (for training)

        Returns:
            np.ndarray: Policy vector (visit counts normalized to probabilities)
        """
        from encoding import (
            encode_board_state,
            get_legal_action_mask,
            get_legal_action_indices,
            NUM_ACTIONS,
        )

        root = MCTSNode(prior=1.0)

        # Get initial policy and value from network
        state = encode_board_state(game)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        legal_mask = get_legal_action_mask(game)
        legal_mask_tensor = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)

        policy_probs, _ = self.network.predict(state_tensor, legal_mask_tensor)
        policy_probs = policy_probs.squeeze(0).cpu().numpy()

        legal_actions = get_legal_action_indices(game)

        if not legal_actions:
            # No legal actions - return uniform policy (game should be terminal)
            return np.zeros(NUM_ACTIONS)

        # Expand root
        root.expand(policy_probs, legal_actions)

        # Add Dirichlet noise at root for exploration
        if add_noise and len(legal_actions) > 0:
            root.add_dirichlet_noise(
                legal_actions,
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon,
            )

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            scratch_game = game.copy()
            search_path = [node]

            # SELECT: Traverse tree to leaf
            while node.is_expanded and node.children:
                action, node = node.select_child(self.config.c_puct)
                scratch_game.apply_action(action)
                search_path.append(node)

                if scratch_game.is_terminal():
                    break

            # Get value
            if scratch_game.is_terminal():
                # Terminal node: use actual game result
                # get_result() returns value from BLACK's perspective
                # Convert to current player's perspective
                value = scratch_game.get_result()
                if (
                    scratch_game.turn.color == scratch_game.player_2.color
                ):  # White's turn
                    value = -value
            else:
                # Non-terminal: expand and evaluate with network
                state = encode_board_state(scratch_game)
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                legal_mask = get_legal_action_mask(scratch_game)
                legal_mask_tensor = (
                    torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)
                )

                policy_probs, value_tensor = self.network.predict(
                    state_tensor, legal_mask_tensor
                )
                policy_probs = policy_probs.squeeze(0).cpu().numpy()
                value = value_tensor.squeeze().item()

                legal_actions = get_legal_action_indices(scratch_game)
                if legal_actions:
                    node.expand(policy_probs, legal_actions)

            # BACKPROPAGATE: Update values along path
            # Note: We need to flip the value sign as we go up the tree
            # because each level represents the opponent's perspective
            self._backpropagate(search_path, value, scratch_game)

        # Return visit count distribution as policy
        return self._get_policy(root, legal_actions)

    def _backpropagate(
        self,
        search_path: List[MCTSNode],
        value: float,
        game: "Game",
    ):
        """
        Backpropagate value along search path.

        The value alternates sign because each level represents
        a different player's perspective.
        """
        # Value is from the perspective of the player who just moved
        # We need to flip it for each level going up
        for i, node in enumerate(reversed(search_path)):
            # Flip value for each level (alternating players)
            node_value = value if i % 2 == 0 else -value
            node.visit_count += 1
            node.value_sum += node_value

    def _get_policy(
        self,
        root: MCTSNode,
        legal_actions: List[int],
    ) -> np.ndarray:
        """
        Convert root node visit counts to policy probabilities.

        Args:
            root: Root node of MCTS tree
            legal_actions: List of legal action indices

        Returns:
            Policy vector with visit count proportions
        """
        from encoding import NUM_ACTIONS

        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)

        if not root.children:
            # No children - uniform over legal actions
            if legal_actions:
                for action in legal_actions:
                    policy[action] = 1.0 / len(legal_actions)
            return policy

        # Get visit counts
        visit_counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        # Apply temperature
        if self.config.temperature == 0:
            # Greedy: select most visited action
            best_action = max(
                root.children.keys(), key=lambda a: root.children[a].visit_count
            )
            policy[best_action] = 1.0
        elif self.config.temperature == float("inf"):
            # Uniform over visited actions
            visited = [
                a for a in root.children.keys() if root.children[a].visit_count > 0
            ]
            if visited:
                for action in visited:
                    policy[action] = 1.0 / len(visited)
        else:
            # Temperature-scaled softmax
            visit_counts_temp = visit_counts ** (1.0 / self.config.temperature)
            total = visit_counts_temp.sum()
            if total > 0:
                policy = visit_counts_temp / total

        return policy

    def select_action(
        self,
        policy: np.ndarray,
        temperature: float = None,
    ) -> int:
        """
        Select action from policy distribution.

        Args:
            policy: Policy probabilities
            temperature: Temperature for selection (0 = greedy, higher = more random)

        Returns:
            Selected action index
        """
        if temperature is None:
            temperature = self.config.temperature

        if temperature == 0:
            # Greedy selection
            return int(np.argmax(policy))
        else:
            # Sample from distribution
            # Renormalize to handle numerical errors
            policy_sum = policy.sum()
            if policy_sum <= 0:
                # Fallback to uniform over non-zero entries or all actions
                non_zero = np.where(policy > 0)[0]
                if len(non_zero) > 0:
                    return int(np.random.choice(non_zero))
                return int(np.random.randint(len(policy)))
            policy = policy / policy_sum
            return int(np.random.choice(len(policy), p=policy))


def run_mcts_self_play_step(
    game: "Game",
    mcts: MCTS,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """
    Run one step of MCTS self-play.

    Args:
        game: Current game state (will be modified)
        mcts: MCTS instance
        temperature: Temperature for action selection

    Returns:
        Tuple of (policy, selected_action)
    """
    # Save original temperature
    original_temp = mcts.config.temperature
    mcts.config.temperature = temperature

    # Run MCTS search
    policy = mcts.search(game, add_noise=True)

    # Select action
    action = mcts.select_action(policy, temperature)

    # Restore temperature
    mcts.config.temperature = original_temp

    return policy, action
