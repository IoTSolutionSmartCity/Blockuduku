from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from blockuduko.game.engine import GameState, legal_moves, reset, step
from blockuduko.game.pieces import encode_hand_piece
from blockuduko.game.scoring import compute_move_reward, count_combo_tier
from blockuduko.rl.action_codec import MAX_ACTIONS, decode_action, encode_action, moves_to_mask


@dataclass
class EnvConfig:
    survival_reward: float = 0.01
    illegal_action_penalty: float = -0.1
    terminal_score_scale: float = 0.001
    low_score_penalty: float = -1.0
    low_score_threshold: int = 100


class BlockudukoEnv(gym.Env):
    """Gymnasium environment for Blockudoku with action masking support."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.state: GameState | None = None

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(0, 1, (9, 9), dtype=np.float32),
                "hand": spaces.Box(0, 1, (3, 5, 5), dtype=np.float32),
                "scalars": spaces.Box(0, np.inf, (4,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(MAX_ACTIONS)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        self.state = reset(seed=seed)
        return self._obs(), self._info(cleared_count=0)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before step()")

        mask = self.action_masks()
        if not mask[action]:
            reward = self.config.illegal_action_penalty
            terminated = len(legal_moves(self.state)) == 0
            return self._obs(), reward, terminated, False, self._info(cleared_count=0, illegal=True)

        move = decode_action(int(action))
        result = step(self.state, move)
        self.state = result.state

        combo_tier = count_combo_tier(result.cleared_count, result.num_lines)
        reward = compute_move_reward(
            result.cleared_count,
            combo_tier,
            self.state.streak,
        )
        reward += self.config.survival_reward

        terminated = result.game_over
        if terminated:
            reward += self.state.score * self.config.terminal_score_scale
            if self.state.score < self.config.low_score_threshold:
                reward += self.config.low_score_penalty

        return (
            self._obs(),
            reward,
            terminated,
            False,
            self._info(cleared_count=result.cleared_count),
        )

    def action_masks(self) -> np.ndarray:
        if self.state is None:
            return np.zeros(MAX_ACTIONS, dtype=bool)
        return np.array(moves_to_mask(legal_moves(self.state)), dtype=bool)

    def render(self) -> str | None:
        if self.state is None:
            return None

        lines: list[str] = []
        board = self.state.board.cells
        for row in range(9):
            row_chars = []
            for col in range(9):
                row_chars.append("#" if board[row, col] else ".")
                if col in (2, 5):
                    row_chars.append("|")
            lines.append(" ".join(row_chars))
            if row in (2, 5):
                lines.append("- " * 17)

        hand_repr = [
            f"slot{i}:{piece.id if piece else '-'}" for i, piece in enumerate(self.state.hand)
        ]
        lines.append(
            f"score={self.state.score} streak={self.state.streak} hand=[{', '.join(hand_repr)}]"
        )
        return "\n".join(lines)

    def _obs(self) -> dict[str, np.ndarray]:
        assert self.state is not None
        board = self.state.board.cells.astype(np.float32)
        hand = np.stack(
            [encode_hand_piece(piece) for piece in self.state.hand],
            axis=0,
        )
        scalars = np.array(
            [
                self.state.score / 1000.0,
                float(self.state.combo),
                float(self.state.streak),
                self.state.board.empty_cell_ratio(),
            ],
            dtype=np.float32,
        )
        return {"board": board, "hand": hand, "scalars": scalars}

    def _info(self, cleared_count: int, illegal: bool = False) -> dict:
        assert self.state is not None
        legal = legal_moves(self.state)
        return {
            "score": self.state.score,
            "cleared_cells": cleared_count,
            "legal_action_count": len(legal),
            "illegal_action": illegal,
        }
