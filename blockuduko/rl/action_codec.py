from __future__ import annotations

HAND_SIZE = 3
BOARD_SIZE = 9
MAX_ACTIONS = HAND_SIZE * BOARD_SIZE * BOARD_SIZE  # 243

Move = tuple[int, int, int]  # (piece_idx, row, col)


def encode_action(piece_idx: int, row: int, col: int) -> int:
    if not (0 <= piece_idx < HAND_SIZE):
        raise ValueError(f"piece_idx must be in [0, {HAND_SIZE}), got {piece_idx}")
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"row/col must be in [0, {BOARD_SIZE}), got ({row}, {col})")
    return piece_idx * BOARD_SIZE * BOARD_SIZE + row * BOARD_SIZE + col


def decode_action(action_id: int) -> Move:
    if not (0 <= action_id < MAX_ACTIONS):
        raise ValueError(f"action_id must be in [0, {MAX_ACTIONS}), got {action_id}")
    piece_idx = action_id // (BOARD_SIZE * BOARD_SIZE)
    remainder = action_id % (BOARD_SIZE * BOARD_SIZE)
    row = remainder // BOARD_SIZE
    col = remainder % BOARD_SIZE
    return piece_idx, row, col


def moves_to_mask(legal_moves: list[Move]) -> list[bool]:
    mask = [False] * MAX_ACTIONS
    for move in legal_moves:
        mask[encode_action(*move)] = True
    return mask
