from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Piece:
    id: int
    cells: tuple[tuple[int, int], ...]


def _normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
    return normalized


def _piece(cells: list[tuple[int, int]], piece_id: int) -> Piece:
    return Piece(id=piece_id, cells=_normalize(cells))


# Catalog of common Blockudoku polyomino shapes (no rotation variants).
PIECE_CATALOG: tuple[Piece, ...] = (
    _piece([(0, 0)], 0),
    _piece([(0, 0), (0, 1)], 1),
    _piece([(0, 0), (1, 0)], 2),
    _piece([(0, 0), (0, 1), (0, 2)], 3),
    _piece([(0, 0), (1, 0), (2, 0)], 4),
    _piece([(0, 0), (0, 1), (1, 0)], 5),
    _piece([(0, 1), (1, 0), (1, 1)], 6),
    _piece([(0, 0), (0, 1), (1, 1)], 7),
    _piece([(0, 0), (1, 0), (1, 1)], 8),
    _piece([(0, 0), (0, 1), (0, 2), (0, 3)], 9),
    _piece([(0, 0), (1, 0), (2, 0), (3, 0)], 10),
    _piece([(0, 0), (0, 1), (1, 0), (1, 1)], 11),
    _piece([(0, 0), (0, 1), (0, 2), (1, 0)], 12),
    _piece([(0, 0), (0, 1), (0, 2), (1, 1)], 13),
    _piece([(0, 1), (1, 0), (1, 1), (1, 2)], 14),
    _piece([(0, 0), (1, 0), (1, 1), (2, 1)], 15),
    _piece([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], 16),
    _piece([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], 17),
    _piece([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)], 18),
    _piece([(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)], 19),
    _piece([(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)], 20),
    _piece([(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)], 21),
)


def deal_hand(rng: np.random.Generator) -> list[Piece]:
    indices = rng.integers(0, len(PIECE_CATALOG), size=3)
    return [PIECE_CATALOG[i] for i in indices]


def encode_hand_piece(piece: Piece | None, canvas_size: int = 5) -> np.ndarray:
    """Render a piece into a padded local mask centered in the canvas."""
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    if piece is None:
        return canvas

    max_r = max(dr for dr, _ in piece.cells)
    max_c = max(dc for _, dc in piece.cells)
    offset_r = (canvas_size - (max_r + 1)) // 2
    offset_c = (canvas_size - (max_c + 1)) // 2

    for dr, dc in piece.cells:
        canvas[offset_r + dr, offset_c + dc] = 1.0
    return canvas
