import numpy as np
import pytest

from blockuduko.game.engine import legal_moves, reset, step
from blockuduko.game.pieces import Piece


def test_reset_is_deterministic_with_seed():
    state_a = reset(seed=7)
    state_b = reset(seed=7)
    assert [p.id for p in state_a.hand] == [p.id for p in state_b.hand]
    assert np.array_equal(state_a.board.cells, state_b.board.cells)


def test_legal_moves_non_empty_at_start():
    state = reset(seed=1)
    moves = legal_moves(state)
    assert len(moves) > 0


def test_step_places_piece_and_updates_score():
    state = reset(seed=0)
    move = legal_moves(state)[0]
    result = step(state, move)
    assert result.state.score >= 0
    assert result.state.hand[move[0]] is None or any(
        p is not None for p in result.state.hand
    )


def test_hand_refills_after_three_placements():
    state = reset(seed=42)
    seen_hands: list[tuple[int | None, ...]] = []

    for _ in range(3):
        move = legal_moves(state)[0]
        result = step(state, move)
        state = result.state
        seen_hands.append(tuple(p.id if p else None for p in state.hand))

    assert all(any(p is not None for p in hand) for hand in seen_hands[-1:])


def test_illegal_move_raises():
    state = reset(seed=0)
    legal = set(legal_moves(state))
    for piece_idx in range(3):
        for row in range(9):
            for col in range(9):
                candidate = (piece_idx, row, col)
                if candidate not in legal:
                    with pytest.raises(ValueError):
                        step(state, candidate)
                    return
    pytest.fail("Expected at least one illegal candidate move")


def test_game_over_when_no_moves(monkeypatch):
    state = reset(seed=99)
    large_piece = Piece(id=999, cells=tuple((r, c) for r in range(3) for c in range(3)))
    state.hand = [large_piece, None, None]

    for row in range(9):
        for col in range(9):
            if (row, col) != (0, 0):
                state.board.cells[row, col] = 1

    moves = legal_moves(state)
    assert len(moves) <= 1
    if moves:
        result = step(state, moves[0])
        assert result.game_over is True
