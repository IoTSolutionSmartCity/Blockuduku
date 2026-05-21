from __future__ import annotations


def count_combo_tier(cleared_cells: int, num_lines: int) -> int:
    """Return combo tier based on how many distinct lines/boxes cleared."""
    if cleared_cells == 0:
        return 0
    if num_lines >= 3:
        return 3
    if num_lines == 2:
        return 2
    return 1


def compute_move_score(
    cleared_count: int,
    combo_tier: int,
    streak: int,
) -> int:
    """Approximate Blockudoku scoring for a single move."""
    if cleared_count == 0:
        return 0

    base = cleared_count * 9
    combo_multiplier = 1 + 0.5 * max(combo_tier - 1, 0)
    streak_bonus = streak * 3
    return int(base * combo_multiplier + streak_bonus)


def compute_move_reward(
    cleared_count: int,
    combo_tier: int,
    streak: int,
) -> float:
    """RL reward signal derived from move scoring."""
    if cleared_count == 0:
        return 0.0

    reward = float(cleared_count)
    reward += 0.5 * max(combo_tier - 1, 0) * cleared_count
    reward += 0.1 * streak
    return reward
