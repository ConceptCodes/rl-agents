#!/usr/bin/env python3
"""Self-play game generation for AlphaZero training."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch

from game import Game
from encoding import encode_board_state, get_legal_action_indices, NUM_ACTIONS
from mcts import MCTS, MCTSConfig
from replay_buffer import GameHistory, GameRecord
from constants import BLACK

if TYPE_CHECKING:
    from network import GungiNetwork


def play_game(
    network: "GungiNetwork",
    mcts_config: MCTSConfig = None,
    device: torch.device = None,
    temperature_threshold: int = 30,
    max_moves: int = 500,
    verbose: bool = False,
) -> List[GameRecord]:
    """
    Play one game of self-play using MCTS.

    Args:
        network: Neural network for policy/value evaluation
        mcts_config: MCTS configuration
        device: Torch device for network inference
        temperature_threshold: Move number after which temperature drops
        max_moves: Maximum moves before declaring draw
        verbose: Print progress

    Returns:
        List of GameRecord for training
    """
    if mcts_config is None:
        mcts_config = MCTSConfig()
    if device is None:
        device = torch.device("cpu")

    game = Game(render_ui=False)
    mcts = MCTS(network, mcts_config, device)
    history = GameHistory()

    move_count = 0

    while not game.is_terminal() and move_count < max_moves:
        # Determine temperature
        if move_count < temperature_threshold:
            temperature = 1.0  # More exploration early
        else:
            temperature = 0.1  # More exploitation later

        # Encode current state
        state = encode_board_state(game)
        current_player = 0 if game.turn == game.player_1 else 1

        # Run MCTS
        mcts.config.temperature = temperature
        policy = mcts.search(game, add_noise=True)

        # Select action
        action = mcts.select_action(policy, temperature)

        # Store for training
        history.add_step(state, policy, current_player)

        # Apply action
        success = game.apply_action(action)
        if not success:
            if verbose:
                print(f"Warning: Failed to apply action {action} at move {move_count}")
            # Try random legal action
            legal_actions = get_legal_action_indices(game)
            if legal_actions:
                action = np.random.choice(legal_actions)
                game.apply_action(action)
            else:
                break

        move_count += 1

        if verbose and move_count % 20 == 0:
            print(f"  Move {move_count}, phase={game.game_phase}")

    # Set game result
    result = game.get_result()

    # If max moves reached with no winner, it's a draw
    if move_count >= max_moves and result == 0.0:
        result = 0.0  # Draw

    history.set_result(result)

    if verbose:
        winner = "Black" if result > 0 else ("White" if result < 0 else "Draw")
        print(f"  Game finished: {winner} after {move_count} moves")

    return history.to_records()


def _worker_play_game(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Worker function for multiprocessing self-play.

    Takes serializable arguments and returns serializable results.
    """
    from network import GungiNetwork

    # Unpack arguments
    state_dict = args["state_dict"]
    network_config = args["network_config"]
    mcts_config_dict = args["mcts_config"]
    device_str = args["device"]
    temperature_threshold = args["temperature_threshold"]
    max_moves = args["max_moves"]
    game_idx = args["game_idx"]

    # Setup device (use CPU in workers for simplicity)
    device = torch.device(device_str)

    # Recreate network
    network = GungiNetwork(
        num_res_blocks=network_config["num_res_blocks"],
        num_channels=network_config["num_channels"],
        num_input_planes=network_config["num_input_planes"],
        num_actions=network_config["num_actions"],
        board_size=network_config["board_size"],
    )
    network.load_state_dict(state_dict)
    network.to(device)
    network.eval()

    # Recreate MCTS config
    mcts_config = MCTSConfig(
        num_simulations=mcts_config_dict["num_simulations"],
        c_puct=mcts_config_dict["c_puct"],
        dirichlet_alpha=mcts_config_dict["dirichlet_alpha"],
        dirichlet_epsilon=mcts_config_dict["dirichlet_epsilon"],
        temperature=mcts_config_dict["temperature"],
    )

    # Play game
    records = play_game(
        network=network,
        mcts_config=mcts_config,
        device=device,
        temperature_threshold=temperature_threshold,
        max_moves=max_moves,
        verbose=False,
    )

    # Convert to serializable format
    return [
        {
            "state": r.state.tolist(),
            "policy": r.policy.tolist(),
            "value": r.value,
        }
        for r in records
    ]


def play_games_parallel(
    network: "GungiNetwork",
    num_games: int,
    mcts_config: MCTSConfig = None,
    device: torch.device = None,
    temperature_threshold: int = 30,
    max_moves: int = 500,
    verbose: bool = False,
    num_workers: int = 0,
) -> List[GameRecord]:
    """
    Play multiple games, optionally in parallel using multiprocessing.

    Args:
        network: Neural network
        num_games: Number of games to play
        mcts_config: MCTS configuration
        device: Torch device
        temperature_threshold: When to reduce temperature
        max_moves: Maximum moves per game
        verbose: Print progress
        num_workers: Number of parallel workers (0 = sequential)

    Returns:
        Combined list of GameRecords from all games
    """
    if mcts_config is None:
        mcts_config = MCTSConfig()
    if device is None:
        device = torch.device("cpu")

    # If no workers or only 1 game, run sequentially
    if num_workers <= 0 or num_games <= 1:
        all_records = []
        for game_idx in range(num_games):
            if verbose:
                print(f"Playing game {game_idx + 1}/{num_games}")

            records = play_game(
                network=network,
                mcts_config=mcts_config,
                device=device,
                temperature_threshold=temperature_threshold,
                max_moves=max_moves,
                verbose=verbose,
            )
            all_records.extend(records)
        return all_records

    # Parallel execution
    # Prepare serializable arguments
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
    network_config = {
        "num_res_blocks": len(network.res_blocks),
        "num_channels": network.input_conv.out_channels,
        "num_input_planes": network.num_input_planes,
        "num_actions": network.num_actions,
        "board_size": network.board_size,
    }
    mcts_config_dict = {
        "num_simulations": mcts_config.num_simulations,
        "c_puct": mcts_config.c_puct,
        "dirichlet_alpha": mcts_config.dirichlet_alpha,
        "dirichlet_epsilon": mcts_config.dirichlet_epsilon,
        "temperature": mcts_config.temperature,
    }

    # Workers use CPU to avoid MPS/CUDA issues with multiprocessing
    worker_device = "cpu"

    # Create argument list for each game
    worker_args = [
        {
            "state_dict": state_dict,
            "network_config": network_config,
            "mcts_config": mcts_config_dict,
            "device": worker_device,
            "temperature_threshold": temperature_threshold,
            "max_moves": max_moves,
            "game_idx": i,
        }
        for i in range(num_games)
    ]

    # Run in parallel
    actual_workers = min(num_workers, num_games)
    if verbose:
        print(f"Playing {num_games} games with {actual_workers} workers...")

    # Use spawn context for compatibility with CUDA/MPS
    ctx = mp.get_context("spawn")
    with ctx.Pool(actual_workers) as pool:
        results = pool.map(_worker_play_game, worker_args)

    # Convert results back to GameRecord objects
    all_records = []
    for game_results in results:
        for r in game_results:
            all_records.append(
                GameRecord(
                    state=np.array(r["state"], dtype=np.float32),
                    policy=np.array(r["policy"], dtype=np.float32),
                    value=r["value"],
                )
            )

    if verbose:
        print(f"Generated {len(all_records)} samples from {num_games} games")

    return all_records


def evaluate_against_random(
    network: "GungiNetwork",
    num_games: int = 10,
    mcts_simulations: int = 100,
    device: torch.device = None,
) -> dict:
    """
    Evaluate network against a random player.

    Args:
        network: Neural network to evaluate
        num_games: Number of games to play
        mcts_simulations: MCTS simulations per move
        device: Torch device

    Returns:
        Dict with win/loss/draw statistics
    """
    if device is None:
        device = torch.device("cpu")

    config = MCTSConfig(num_simulations=mcts_simulations, temperature=0.0)
    mcts = MCTS(network, config, device)

    results = {"network_wins": 0, "random_wins": 0, "draws": 0}

    for game_idx in range(num_games):
        # Alternate who plays black
        network_is_black = game_idx % 2 == 0

        game = Game(render_ui=False)
        move_count = 0
        max_moves = 500

        while not game.is_terminal() and move_count < max_moves:
            current_is_black = game.turn == game.player_1
            network_to_move = current_is_black == network_is_black

            if network_to_move:
                # Network's turn - use MCTS
                policy = mcts.search(game, add_noise=False)
                action = mcts.select_action(policy, temperature=0.0)
            else:
                # Random player's turn
                legal_actions = get_legal_action_indices(game)
                if not legal_actions:
                    break
                action = np.random.choice(legal_actions)

            game.apply_action(action)
            move_count += 1

        result = game.get_result()  # +1 = black wins, -1 = white wins

        if result == 0:
            results["draws"] += 1
        elif (result > 0 and network_is_black) or (result < 0 and not network_is_black):
            results["network_wins"] += 1
        else:
            results["random_wins"] += 1

    results["win_rate"] = results["network_wins"] / num_games
    return results


def evaluate_against_network(
    network1: "GungiNetwork",
    network2: "GungiNetwork",
    num_games: int = 10,
    mcts_simulations: int = 100,
    device: torch.device = None,
) -> dict:
    """
    Evaluate two networks against each other.

    Args:
        network1: First network
        network2: Second network
        num_games: Number of games to play
        mcts_simulations: MCTS simulations per move
        device: Torch device

    Returns:
        Dict with match statistics
    """
    if device is None:
        device = torch.device("cpu")

    config = MCTSConfig(num_simulations=mcts_simulations, temperature=0.0)
    mcts1 = MCTS(network1, config, device)
    mcts2 = MCTS(network2, config, device)

    results = {"net1_wins": 0, "net2_wins": 0, "draws": 0}

    for game_idx in range(num_games):
        # Alternate who plays black
        net1_is_black = game_idx % 2 == 0

        game = Game(render_ui=False)
        move_count = 0
        max_moves = 500

        while not game.is_terminal() and move_count < max_moves:
            current_is_black = game.turn == game.player_1
            net1_to_move = current_is_black == net1_is_black

            mcts = mcts1 if net1_to_move else mcts2
            policy = mcts.search(game, add_noise=False)
            action = mcts.select_action(policy, temperature=0.0)

            game.apply_action(action)
            move_count += 1

        result = game.get_result()

        if result == 0:
            results["draws"] += 1
        elif (result > 0 and net1_is_black) or (result < 0 and not net1_is_black):
            results["net1_wins"] += 1
        else:
            results["net2_wins"] += 1

    results["net1_win_rate"] = results["net1_wins"] / num_games
    results["net2_win_rate"] = results["net2_wins"] / num_games
    return results
