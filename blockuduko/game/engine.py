from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from blockuduko.game.board import BOARD_SIZE, Board
from blockuduko.game.pieces import Piece, deal_hand
from blockuduko.game.scoring import compute_move_score, count_combo_tier

Move = tuple[int, int, int]  # (piece_idx, row, col)
HAND_SIZE = 3


@dataclass
class StepResult:
    state: GameState
    cleared_count: int
    num_lines: int
    move_score: int
    game_over: bool


@dataclass
class GameState:
    board: Board
    hand: list[Piece | None]
    score: int
    combo: int
    streak: int
    rng: np.random.Generator

    def copy(self) -> GameState:
        return GameState(
            board=self.board.copy(),
            hand=list(self.hand),
            score=self.score,
            combo=self.combo,
            streak=self.streak,
            rng=self.rng,
        )


def reset(seed: int | None = None) -> GameState:
    rng = np.random.default_rng(seed)
    return GameState(
        board=Board.empty(),
        hand=deal_hand(rng),
        score=0,
        combo=0,
        streak=0,
        rng=rng,
    )


def _count_cleared_lines(board: Board, cleared_cells: set[tuple[int, int]]) -> int:
    if not cleared_cells:
        return 0

    lines = 0
    cleared_rows = {row for row, _ in cleared_cells}
    cleared_cols = {col for _, col in cleared_cells}

    for row in range(BOARD_SIZE):
        if all((row, col) in cleared_cells for col in range(BOARD_SIZE)):
            lines += 1

    for col in range(BOARD_SIZE):
        if all((row, col) in cleared_cells for row in range(BOARD_SIZE)):
            lines += 1

    for box_row in range(0, BOARD_SIZE, 3):
        for box_col in range(0, BOARD_SIZE, 3):
            box_cells = {
                (box_row + dr, box_col + dc) for dr in range(3) for dc in range(3)
            }
            if box_cells.issubset(cleared_cells):
                lines += 1

    return lines


def legal_moves(state: GameState) -> list[Move]:
    moves: list[Move] = []
    for piece_idx, piece in enumerate(state.hand):
        if piece is None:
            continue
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if state.board.can_place(piece, row, col):
                    moves.append((piece_idx, row, col))
    return moves


def _refill_hand_if_needed(state: GameState) -> GameState:
    if any(piece is not None for piece in state.hand):
        return state
    return replace(state, hand=deal_hand(state.rng))


def step(state: GameState, action: Move) -> StepResult:
    legal = legal_moves(state)
    if action not in legal:
        raise ValueError(f"Illegal action: {action}")

    piece_idx, row, col = action
    piece = state.hand[piece_idx]
    if piece is None:
        raise ValueError(f"No piece in hand slot {piece_idx}")

    new_state = state.copy()
    new_state.board.place(piece, row, col)

    cleared_cells = new_state.board.find_clears()
    cleared_count = new_state.board.clear_cells(cleared_cells)
    num_lines = _count_cleared_lines(new_state.board, cleared_cells)
    combo_tier = count_combo_tier(cleared_count, num_lines)

    if cleared_count > 0:
        new_state.streak += 1
        new_state.combo = max(new_state.combo, combo_tier)
    else:
        new_state.streak = 0

    move_score = compute_move_score(cleared_count, combo_tier, new_state.streak)
    new_state.score += move_score

    new_hand = list(new_state.hand)
    new_hand[piece_idx] = None
    new_state.hand = new_hand
    new_state = _refill_hand_if_needed(new_state)

    game_over = len(legal_moves(new_state)) == 0
    return StepResult(
        state=new_state,
        cleared_count=cleared_count,
        num_lines=num_lines,
        move_score=move_score,
        game_over=game_over,
    )
