from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from blockuduko.game.pieces import Piece

Cell = tuple[int, int]
BOARD_SIZE = 9


@dataclass
class Board:
    """9x9 occupancy grid: 0 = empty, 1 = occupied."""

    cells: np.ndarray

    @classmethod
    def empty(cls) -> Board:
        return cls(cells=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8))

    def copy(self) -> Board:
        return Board(cells=self.cells.copy())

    def can_place(self, piece: Piece, anchor_row: int, anchor_col: int) -> bool:
        for dr, dc in piece.cells:
            row, col = anchor_row + dr, anchor_col + dc
            if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
                return False
            if self.cells[row, col] != 0:
                return False
        return True

    def place(self, piece: Piece, anchor_row: int, anchor_col: int) -> None:
        if not self.can_place(piece, anchor_row, anchor_col):
            raise ValueError(
                f"Cannot place piece {piece.id} at ({anchor_row}, {anchor_col})"
            )
        for dr, dc in piece.cells:
            self.cells[anchor_row + dr, anchor_col + dc] = 1

    def find_clears(self) -> set[Cell]:
        cleared: set[Cell] = set()

        for row in range(BOARD_SIZE):
            if np.all(self.cells[row, :] == 1):
                cleared.update((row, col) for col in range(BOARD_SIZE))

        for col in range(BOARD_SIZE):
            if np.all(self.cells[:, col] == 1):
                cleared.update((row, col) for row in range(BOARD_SIZE))

        for box_row in range(0, BOARD_SIZE, 3):
            for box_col in range(0, BOARD_SIZE, 3):
                box = self.cells[box_row : box_row + 3, box_col : box_col + 3]
                if np.all(box == 1):
                    for dr in range(3):
                        for dc in range(3):
                            cleared.add((box_row + dr, box_col + dc))

        return cleared

    def clear_cells(self, cells: Iterable[Cell]) -> int:
        count = 0
        for row, col in cells:
            if self.cells[row, col] != 0:
                self.cells[row, col] = 0
                count += 1
        return count

    def empty_cell_ratio(self) -> float:
        empty = np.count_nonzero(self.cells == 0)
        return empty / (BOARD_SIZE * BOARD_SIZE)
