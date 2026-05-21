import numpy as np
import pytest

from blockuduko.game.board import Board
from blockuduko.game.pieces import Piece
from blockuduko.game.scoring import compute_move_reward, count_combo_tier


MONOMINO = Piece(id=99, cells=((0, 0),))
DOMINO_H = Piece(id=100, cells=((0, 0), (0, 1)))
L_TRIPLET = Piece(id=101, cells=((0, 0), (0, 1), (1, 0)))


def test_can_place_rejects_out_of_bounds():
    board = Board.empty()
    assert board.can_place(MONOMINO, 0, 0) is True
    assert board.can_place(MONOMINO, 8, 8) is True
    assert board.can_place(MONOMINO, 9, 0) is False
    assert board.can_place(DOMINO_H, 0, 8) is False


def test_can_place_rejects_overlap():
    board = Board.empty()
    board.place(MONOMINO, 0, 0)
    assert board.can_place(MONOMINO, 0, 0) is False
    assert board.can_place(DOMINO_H, 0, 0) is False


def test_row_clear():
    board = Board.empty()
    for col in range(8):
        board.place(MONOMINO, 0, col)
    board.place(MONOMINO, 0, 8)

    cleared = board.find_clears()
    assert len(cleared) == 9
    assert board.clear_cells(cleared) == 9
    assert np.all(board.cells[0, :] == 0)


def test_column_clear():
    board = Board.empty()
    for row in range(9):
        board.place(MONOMINO, row, 0)

    cleared = board.find_clears()
    assert len(cleared) == 9
    board.clear_cells(cleared)
    assert np.all(board.cells[:, 0] == 0)


def test_box_clear():
    board = Board.empty()
    for dr in range(3):
        for dc in range(3):
            board.place(MONOMINO, dr, dc)

    cleared = board.find_clears()
    assert len(cleared) == 9
    board.clear_cells(cleared)
    assert np.all(board.cells[0:3, 0:3] == 0)


def test_overlapping_row_and_column_clear():
    board = Board.empty()
    for col in range(9):
        board.place(MONOMINO, 0, col)
    for row in range(1, 9):
        board.place(MONOMINO, row, 0)

    cleared = board.find_clears()
    assert len(cleared) == 17  # 9 + 9 - 1 overlap
    board.clear_cells(cleared)
    assert board.cells[0, 0] == 0


def test_scoring_combo_tier():
    assert count_combo_tier(9, 1) == 1
    assert count_combo_tier(18, 2) == 2
    assert count_combo_tier(27, 3) == 3
    assert compute_move_reward(0, 0, 0) == 0.0
    assert compute_move_reward(9, 2, 1) > 9.0
