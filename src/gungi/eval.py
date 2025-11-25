#!/usr/bin/env python3
"""Evaluation script for Gungi AlphaZero agent."""

import argparse
from pathlib import Path

import numpy as np
import torch

from game import Game
from network import GungiNetwork
from mcts import MCTS, MCTSConfig
from encoding import encode_board_state, get_legal_action_indices, NUM_ACTIONS
from constants import BLACK


def _resolve_device(device_option: str) -> torch.device:
    """Resolve device string to torch.device."""
    device_option = device_option.lower()

    if device_option == "cpu":
        return torch.device("cpu")
    if device_option == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA not available")
    if device_option == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS not available")

    # Auto-detect
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def play_vs_random(
    network: GungiNetwork,
    device: torch.device,
    mcts_simulations: int = 100,
    num_games: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Play network against random opponent.

    Returns:
        Dict with win/loss/draw statistics
    """
    config = MCTSConfig(num_simulations=mcts_simulations, temperature=0.0)
    mcts = MCTS(network, config, device)

    results = {"network_wins": 0, "random_wins": 0, "draws": 0}

    for game_idx in range(num_games):
        network_is_black = game_idx % 2 == 0

        game = Game(render_ui=False)
        move_count = 0
        max_moves = 500

        while not game.is_terminal() and move_count < max_moves:
            current_is_black = game.turn == game.player_1
            network_to_move = current_is_black == network_is_black

            if network_to_move:
                policy = mcts.search(game, add_noise=False)
                action = mcts.select_action(policy, temperature=0.0)
            else:
                legal_actions = get_legal_action_indices(game)
                if not legal_actions:
                    break
                action = np.random.choice(legal_actions)

            game.apply_action(action)
            move_count += 1

        result = game.get_result()

        if result == 0:
            results["draws"] += 1
            outcome = "Draw"
        elif (result > 0 and network_is_black) or (result < 0 and not network_is_black):
            results["network_wins"] += 1
            outcome = "Network wins"
        else:
            results["random_wins"] += 1
            outcome = "Random wins"

        if verbose:
            side = "Black" if network_is_black else "White"
            print(
                f"Game {game_idx + 1}: Network as {side} - {outcome} ({move_count} moves)"
            )

    results["win_rate"] = results["network_wins"] / num_games
    return results


def play_interactive(
    network: GungiNetwork,
    device: torch.device,
    mcts_simulations: int = 400,
    human_plays_black: bool = True,
):
    """
    Play an interactive game against the network.

    The human controls one side via mouse/keyboard, while the
    AI (using MCTS with the neural network) controls the other.
    """
    import pygame
    from constants import (
        FPS,
        WIDTH,
        HEIGHT,
        BOARD_SIZE,
        PIECE_KEYBOARD_MAP,
        BLACK as COLOR_BLACK,
        WHITE as COLOR_WHITE,
    )

    print("=" * 50)
    print("Interactive Game vs AlphaZero Agent")
    print("=" * 50)
    print(f"You are playing as {'Black' if human_plays_black else 'White'}")
    print(f"AI is using {mcts_simulations} MCTS simulations per move")
    print()
    print("Controls:")
    print("  - Click on a piece to select it")
    print("  - Click on a valid destination to move")
    print("  - Press keys (g, l, m, etc.) to select pieces for drops")
    print("  - Press Enter during setup to finish your setup")
    print("  - Press Escape to clear selection")
    print()

    config = MCTSConfig(num_simulations=mcts_simulations, temperature=0.0)
    mcts = MCTS(network, config, device)

    game = Game(render_ui=True)

    # Determine which player is human and which is AI
    human_color = COLOR_BLACK if human_plays_black else COLOR_WHITE

    def is_human_turn():
        return game.turn.color == human_color

    def ai_make_move():
        """Have the AI make a move using MCTS."""
        if game.is_terminal():
            return

        print("AI is thinking...")

        # Run MCTS to get best action
        policy = mcts.search(game, add_noise=False)
        action = mcts.select_action(policy, temperature=0.0)

        # Apply the action
        success = game.apply_action(action)
        if success:
            print(f"AI played action {action}")
        else:
            # Fallback to random legal action
            legal_actions = get_legal_action_indices(game)
            if legal_actions:
                action = np.random.choice(legal_actions)
                game.apply_action(action)
                print(f"AI played random action {action}")
            else:
                print("AI has no legal moves!")

    # Modified game loop that includes AI moves
    print("Piece Keyboard Map:")
    for key, piece in PIECE_KEYBOARD_MAP.items():
        print(f"  {pygame.key.name(key)}: {piece}")
    print()

    # If AI plays black, it goes first in setup
    if not human_plays_black and game.game_phase == "initial_setup":
        print("AI is setting up its pieces...")
        # Let AI do setup by making random setup moves
        # (In a full implementation, the AI would use MCTS for setup too)
        while game.turn.color != human_color and game.game_phase == "initial_setup":
            legal_actions = get_legal_action_indices(game)
            if legal_actions:
                action = np.random.choice(legal_actions)
                game.apply_action(action)
            else:
                # No more setup moves, switch turn
                game.turn.setup_done = True
                game._switch_turn()

    while game.is_running:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.is_running = False
                break

            # Only process human input if it's human's turn
            if is_human_turn():
                if game.game_phase == "initial_setup":
                    if event.type == pygame.KEYDOWN:
                        if event.key in PIECE_KEYBOARD_MAP:
                            piece_name = PIECE_KEYBOARD_MAP[event.key]
                            piece = next(
                                (
                                    p
                                    for p in game.turn.hand_pieces
                                    if p.name == piece_name
                                ),
                                None,
                            )
                            if piece:
                                game._selected_piece = piece
                                game._waiting_for_click = True
                        elif event.key == pygame.K_RETURN:
                            game.turn.setup_done = True
                            game._selected_piece = None
                            game._waiting_for_click = False
                            print(
                                f"You finished setup with {len(game.turn.hand_pieces)} pieces in hand"
                            )
                            game._switch_turn()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if game._waiting_for_click and game._selected_piece:
                            mouse_x, mouse_y = event.pos
                            col = mouse_x // (WIDTH // BOARD_SIZE)
                            row = mouse_y // (HEIGHT // BOARD_SIZE)
                            if game.handle_setup_move(game._selected_piece, row, col):
                                print(
                                    f"Placed {game._selected_piece.name} at ({row}, {col})"
                                )
                                game._selected_piece = None
                                game._waiting_for_click = False
                else:
                    if event.type == pygame.KEYDOWN:
                        if game._handle_turncoat_choice_key(event.key):
                            continue
                        if event.key in PIECE_KEYBOARD_MAP:
                            piece_name = PIECE_KEYBOARD_MAP[event.key]
                            game._handle_drop_selection(piece_name)
                        elif event.key == pygame.K_ESCAPE:
                            game._clear_active_selection()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        game._handle_game_click(event.pos)

        # Check if it's now AI's turn (after human moved or after setup)
        if not is_human_turn() and game.is_running:
            if game.game_phase == "initial_setup":
                # AI setup: use random placements for now
                while (
                    not is_human_turn()
                    and game.game_phase == "initial_setup"
                    and game.is_running
                ):
                    legal_actions = get_legal_action_indices(game)
                    if legal_actions:
                        action = np.random.choice(legal_actions)
                        game.apply_action(action)
                    else:
                        game.turn.setup_done = True
                        print("AI finished setup")
                        game._switch_turn()
            elif game.game_phase == "game":
                if not game.is_terminal():
                    ai_make_move()

        # Check for game end
        if game.is_terminal():
            result = game.get_result()
            print()
            print("=" * 50)
            if result > 0:
                winner = "Black"
            elif result < 0:
                winner = "White"
            else:
                winner = None

            if winner:
                if (winner == "Black" and human_plays_black) or (
                    winner == "White" and not human_plays_black
                ):
                    print("Congratulations! You won!")
                else:
                    print("AI wins. Better luck next time!")
            else:
                print("Game ended in a draw.")
            print("=" * 50)
            # Keep rendering to see final state

        game._render()
        game.clock.tick(FPS)

    game.end()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gungi AlphaZero agent")
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="vs-random",
        choices=["vs-random", "interactive"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games to play (vs-random mode)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=100,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference",
    )
    parser.add_argument(
        "--play-as",
        type=str,
        default="black",
        choices=["black", "white"],
        help="Side to play as in interactive mode (default: black)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    network = GungiNetwork.load(str(args.checkpoint), device)
    network.eval()

    if args.mode == "vs-random":
        print(f"\nPlaying {args.games} games against random opponent...")
        results = play_vs_random(
            network=network,
            device=device,
            mcts_simulations=args.mcts_sims,
            num_games=args.games,
            verbose=True,
        )

        print(f"\n{'=' * 40}")
        print("Results Summary")
        print(f"{'=' * 40}")
        print(f"Network wins: {results['network_wins']}")
        print(f"Random wins:  {results['random_wins']}")
        print(f"Draws:        {results['draws']}")
        print(f"Win rate:     {results['win_rate']:.1%}")

    elif args.mode == "interactive":
        play_interactive(
            network=network,
            device=device,
            mcts_simulations=args.mcts_sims,
            human_plays_black=(args.play_as == "black"),
        )


if __name__ == "__main__":
    main()
