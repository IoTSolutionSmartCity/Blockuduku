import gym as classic_gym
import gym_blocksudoku  # Registers "blocksudoku-v0" with classic Gym.
import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.distributions import MaskableCategorical
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs


ENV_ID = "blocksudoku-v0"
TOTAL_TIMESTEPS = 3_000_000
TENSORBOARD_LOG_DIR = "./blocksudoku_tensorboard/"
MODEL_OUTPUT = "ppo_blockudoku_survival"
VEC_NORMALIZE_OUTPUT = "vecnormalize_blockudoku_survival.pkl"

# Reward shaping: survival first. More safe steps means more raw score opportunities.
RAW_SCORE_REWARD = 1.0
SURVIVAL_REWARD = 2.0
BRICK_REWARD = 0.0
LINE_CLEAR_REWARD = 35.0
COMBO_BASE_REWARD = 25.0
MULTI_CLEAR_BONUS = 20.0
LINE_PROGRESS_REWARD = 0.75
NEAR_COMPLETE_BONUS = 0.0
NO_PROGRESS_PENALTY = 0.2
OPEN_RATIO_REWARD = 2.0
VALID_MOVE_COUNT_REWARD = 0.03
VALID_MOVE_REWARD = 0.03
VALID_MOVE_LOSS_PENALTY = 0.06
LOW_VALID_ACTION_THRESHOLD = 20
LOW_VALID_ACTION_PENALTY = 2.0
DANGEROUS_NEAR_COMPLETE_PENALTY = 0.1
SET_DEPTH_REWARD = 8.0
SET_BLOCKED_PENALTY = 12.0
FIVE_LANE_REWARD = 4.0
MISSING_FIVE_LANE_PENALTY = 10.0
OPEN_SQUARE_REWARD = 3.0
NO_OPEN_SQUARE_PENALTY = 8.0
ISOLATED_HOLE_PENALTY = 1.5
BOARD_FULLNESS_THRESHOLD = 0.65
BOARD_FULLNESS_PENALTY = 0.15
GAME_OVER_PENALTY = 150.0
MAX_COMBO_MULTIPLIER = 8.0
NEAR_COMPLETE_FILL = 7
INITIAL_LEARNING_RATE = 3e-4
FINAL_LEARNING_RATE = 3e-5
ENTROPY_COEF = 0.015
LOGIT_CLAMP = 50.0
CHECKPOINT_DIR = "./checkpoints/"
CHECKPOINT_FREQ = 100_000


def _patch_maskable_categorical():
    """Work around PyTorch Simplex validate_args failures on masked float32 logits."""
    if getattr(MaskableCategorical, "_blockudoku_patched", False):
        return

    def apply_masking(self, masks):
        if masks is not None:
            device = self.logits.device
            self.masks = th.as_tensor(masks, dtype=th.bool, device=device).reshape(self.logits.shape)
            row_has_valid = self.masks.any(dim=-1, keepdim=True)
            if not bool(row_has_valid.all()):
                fallback = th.zeros_like(self.masks)
                fallback[..., 0] = True
                self.masks = th.where(row_has_valid.expand_as(self.masks), self.masks, fallback)
            huge_neg = th.tensor(-1e8, dtype=self.logits.dtype, device=device)
            logits = th.where(self.masks, self._original_logits, huge_neg)
        else:
            self.masks = None
            logits = self._original_logits

        Categorical.__init__(self, logits=logits, validate_args=False)
        self.probs = logits_to_probs(self.logits)

    def init_with_relaxed_validation(
        self,
        probs=None,
        logits=None,
        validate_args=None,
        masks=None,
    ):
        self.masks = None
        Categorical.__init__(self, probs, logits, validate_args=False)
        self._original_logits = self.logits
        apply_masking(self, masks)

    MaskableCategorical.apply_masking = apply_masking
    MaskableCategorical.__init__ = init_with_relaxed_validation
    MaskableCategorical._blockudoku_patched = True


_patch_maskable_categorical()


class StableMaskableMultiInputActorCriticPolicy(MaskableMultiInputActorCriticPolicy):
    """Clamp policy logits to avoid NaN/Inf softmax failures during PPO updates."""

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        action_logits = self.action_net(latent_pi)
        action_logits = th.clamp(action_logits, -LOGIT_CLAMP, LOGIT_CLAMP)
        action_logits = th.nan_to_num(action_logits, nan=0.0, posinf=LOGIT_CLAMP, neginf=-LOGIT_CLAMP)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    @staticmethod
    def _sanitize_action_masks(action_masks: th.Tensor) -> th.Tensor:
        masks = action_masks > 0.5 if action_masks.dtype != th.bool else action_masks
        masks = masks.reshape(-1, masks.shape[-1])
        row_has_valid = masks.any(dim=-1, keepdim=True)
        if not bool(row_has_valid.all()):
            fallback = th.zeros_like(masks)
            fallback[:, 0] = True
            masks = th.where(row_has_valid.expand_as(masks), masks, fallback)
        return masks

    def evaluate_actions(self, obs, actions, action_masks=None):
        if action_masks is not None:
            action_masks = self._sanitize_action_masks(action_masks)
        return super().evaluate_actions(obs, actions, action_masks=action_masks)

    def get_distribution(self, obs, action_masks=None):
        if action_masks is not None:
            action_masks = self._sanitize_action_masks(th.as_tensor(action_masks))
        return super().get_distribution(obs, action_masks=action_masks)


class StrategicBlockudokuEnv(gym.Env):
    """Gymnasium adapter with clear-focused rewards, line-progress shaping, and action masks."""

    metadata = {"render_modes": ["human"]}
    _LINE_COUNT = 27

    def __init__(self):
        super().__init__()
        self.env = classic_gym.make(ENV_ID, disable_env_checker=True)
        self.base_env = self.env.unwrapped
        self.combo_streak = 0
        self.episode_steps = 0

        self.action_space = spaces.Discrete(3 * 9 * 9)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0.0, high=1.0, shape=(9, 9), dtype=np.float32),
                "shapes": spaces.Box(low=0.0, high=1.0, shape=(75,), dtype=np.float32),
                "open_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "line_fills": spaces.Box(
                    low=0.0, high=1.0, shape=(self._LINE_COUNT,), dtype=np.float32
                ),
                "near_complete": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "valid_action_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "survival_features": spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.combo_streak = 0
        self.episode_steps = 0
        self.env.reset()
        return self._build_observation(), {}

    def step(self, action):
        action_index = int(np.asarray(action).item())
        board_before = self._board().copy()
        valid_actions_before = self._valid_action_count()
        queue_pos, x_pos, y_pos = self.decode_action(action_index)
        selected_block = self._selected_block(queue_pos)
        placed_board = self._preview_board_after_placement(board_before, selected_block, x_pos, y_pos)

        _obs, original_reward, done, info = self.env.step(action_index)
        self.episode_steps += 1
        reward = self._strategic_reward(
            original_reward=original_reward,
            done=done,
            board_before=board_before,
            placed_board=placed_board,
            selected_block=selected_block,
            valid_actions_before=valid_actions_before,
        )
        info = dict(info)
        info["score"] = float(self.base_env.total_score)
        info["episode_steps"] = self.episode_steps

        return self._build_observation(), reward, bool(done), False, info

    def action_masks(self):
        mask = self.base_env.get_valid_action_space().astype(bool)
        if not mask.any():
            # Game-over boards have no legal moves; MaskablePPO requires >=1 valid action.
            mask = np.zeros_like(mask, dtype=bool)
            mask[0] = True
        return mask

    def render(self):
        return self.env.render(mode="human")

    def close(self):
        self.env.close()

    @staticmethod
    def decode_action(action_index):
        queue_pos = action_index // 81
        remainder = action_index % 81
        x_pos = remainder // 9
        y_pos = remainder % 9
        return queue_pos, x_pos, y_pos

    def _board(self):
        return np.asarray(self.base_env.main_board[:9, :9], dtype=np.float32)

    def _selected_block(self, queue_pos):
        if queue_pos >= len(self.base_env.block_queue):
            return None
        return np.asarray(self.base_env.block_queue[queue_pos], dtype=np.float32)

    def _valid_action_count(self):
        return int(np.sum(self.base_env.get_valid_action_space()))

    def _preview_board_after_placement(self, board, selected_block, x_pos, y_pos):
        if selected_block is None:
            return None

        placed_board = board.copy()
        occupied_cells = np.argwhere(selected_block > 0)

        for block_row, block_col in occupied_cells:
            board_row = x_pos + int(block_row)
            board_col = y_pos + int(block_col)
            if board_row < 0 or board_col < 0 or board_row >= 9 or board_col >= 9:
                return None
            if placed_board[board_row, board_col] > 0:
                return None
            placed_board[board_row, board_col] = 1.0

        return placed_board

    def _build_observation(self):
        board = self._board()
        filled_cells = float(board.sum())
        open_ratio = np.array([(81.0 - filled_cells) / 81.0], dtype=np.float32)
        line_fills = self._line_fill_counts(board) / 9.0
        near_complete = np.array(
            [float(np.sum(line_fills >= (NEAR_COMPLETE_FILL / 9.0)) / self._LINE_COUNT)],
            dtype=np.float32,
        )
        valid_action_ratio = np.array([self._valid_action_count() / float(self.action_space.n)], dtype=np.float32)
        survival_features = self._survival_features(board)

        shapes = np.zeros((3, 5, 5), dtype=np.float32)
        for index, block in enumerate(self.base_env.block_queue[:3]):
            block_array = np.asarray(block, dtype=np.float32)
            rows, cols = block_array.shape
            shapes[index, :rows, :cols] = block_array

        return {
            "board": board.astype(np.float32),
            "shapes": shapes.reshape(-1),
            "open_ratio": open_ratio,
            "line_fills": line_fills.astype(np.float32),
            "near_complete": near_complete,
            "valid_action_ratio": valid_action_ratio,
            "survival_features": survival_features,
        }

    def _strategic_reward(
        self,
        original_reward,
        done,
        board_before,
        placed_board,
        selected_block,
        valid_actions_before,
    ):
        if selected_block is None or placed_board is None:
            self.combo_streak = 0
            return float(original_reward)

        lines_cleared = self._count_cleared_lines(placed_board)
        fills_before = self._line_fill_counts(board_before)
        fills_after = self._line_fill_counts(placed_board)
        potential_before = self._clear_potential(fills_before)
        potential_after = self._clear_potential(fills_after)

        board_after = self._board()
        filled_cells = float(board_after.sum())
        open_ratio = (81.0 - filled_cells) / 81.0
        excess_filled_cells = max(0.0, filled_cells - (81.0 * BOARD_FULLNESS_THRESHOLD))
        valid_actions_after = self._valid_action_count()
        valid_action_delta = valid_actions_after - valid_actions_before

        reward = 0.0
        reward += RAW_SCORE_REWARD * float(original_reward)
        reward += SURVIVAL_REWARD if not done else 0.0
        reward += BRICK_REWARD * float(np.sum(selected_block > 0))
        reward += OPEN_RATIO_REWARD * open_ratio
        reward -= BOARD_FULLNESS_PENALTY * excess_filled_cells
        reward += self._setup_reward(fills_before, fills_after)
        reward += VALID_MOVE_COUNT_REWARD * valid_actions_after
        if valid_action_delta >= 0:
            reward += VALID_MOVE_REWARD * valid_action_delta
        else:
            reward += VALID_MOVE_LOSS_PENALTY * valid_action_delta
        if valid_actions_after < LOW_VALID_ACTION_THRESHOLD:
            reward -= LOW_VALID_ACTION_PENALTY * (LOW_VALID_ACTION_THRESHOLD - valid_actions_after)
        reward += self._survival_strategy_reward(board_after, valid_actions_after)

        if lines_cleared > 0:
            self.combo_streak += 1
            combo_multiplier = min(2 ** (self.combo_streak - 1), MAX_COMBO_MULTIPLIER)
            reward += LINE_CLEAR_REWARD * (lines_cleared**2)
            reward += COMBO_BASE_REWARD * combo_multiplier * lines_cleared
            reward += MULTI_CLEAR_BONUS * max(0, lines_cleared - 1)
        else:
            self.combo_streak = 0
            if potential_after <= potential_before:
                reward -= NO_PROGRESS_PENALTY
            reward -= DANGEROUS_NEAR_COMPLETE_PENALTY * float(np.sum(fills_after == 8))

        if done:
            reward -= GAME_OVER_PENALTY

        return float(reward)

    def _setup_reward(self, fills_before, fills_after):
        reward = 0.0
        for before, after in zip(fills_before, fills_after):
            if 6 <= before < 8 and after == 8:
                reward += LINE_PROGRESS_REWARD
            if after == 8:
                reward += NEAR_COMPLETE_BONUS
        return reward

    def _survival_strategy_reward(self, board_after, valid_actions_after):
        horizontal_lane, vertical_lane = self._longest_empty_lanes(board_after)
        open_squares = self._open_square_count(board_after)
        isolated_holes = self._isolated_hole_count(board_after)
        remaining_blocks = [np.asarray(block, dtype=np.float32) for block in self.base_env.block_queue]
        set_depth = self._max_placeable_depth(board_after, remaining_blocks)

        reward = 0.0
        reward += SET_DEPTH_REWARD * set_depth
        if remaining_blocks and set_depth < len(remaining_blocks):
            reward -= SET_BLOCKED_PENALTY * (len(remaining_blocks) - set_depth)

        reward += FIVE_LANE_REWARD * (min(horizontal_lane, 5) / 5.0)
        reward += FIVE_LANE_REWARD * (min(vertical_lane, 5) / 5.0)
        if horizontal_lane < 5:
            reward -= MISSING_FIVE_LANE_PENALTY
        if vertical_lane < 5:
            reward -= MISSING_FIVE_LANE_PENALTY

        reward += OPEN_SQUARE_REWARD * min(open_squares, 2)
        if open_squares == 0:
            reward -= NO_OPEN_SQUARE_PENALTY

        reward -= ISOLATED_HOLE_PENALTY * isolated_holes
        if valid_actions_after == 0:
            reward -= GAME_OVER_PENALTY

        return reward

    def _survival_features(self, board):
        horizontal_lane, vertical_lane = self._longest_empty_lanes(board)
        open_squares = self._open_square_count(board)
        empty_squares = self._empty_square_count(board)
        isolated_holes = self._isolated_hole_count(board)
        remaining_blocks = [np.asarray(block, dtype=np.float32) for block in self.base_env.block_queue]
        set_depth = self._max_placeable_depth(board, remaining_blocks)

        return np.array(
            [
                horizontal_lane / 9.0,
                vertical_lane / 9.0,
                min(open_squares, 2) / 2.0,
                min(empty_squares, 2) / 2.0,
                min(isolated_holes, 12) / 12.0,
                set_depth / 3.0,
            ],
            dtype=np.float32,
        )

    def _max_placeable_depth(self, board, blocks):
        if not blocks:
            return 0

        best_depth = 0
        for block_index, block in enumerate(blocks):
            for x_pos in range(9):
                for y_pos in range(9):
                    placed_board = self._preview_board_after_placement(board, block, x_pos, y_pos)
                    if placed_board is None:
                        continue
                    next_board = self._clear_completed_lines(placed_board)
                    remaining_blocks = blocks[:block_index] + blocks[block_index + 1 :]
                    best_depth = max(best_depth, 1 + self._max_placeable_depth(next_board, remaining_blocks))
                    if best_depth == len(blocks):
                        return best_depth
        return best_depth

    @staticmethod
    def _clear_completed_lines(board):
        next_board = board.copy()
        clear_mask = np.zeros((9, 9), dtype=bool)

        for row in range(9):
            if next_board[row, :].sum() == 9:
                clear_mask[row, :] = True

        for col in range(9):
            if next_board[:, col].sum() == 9:
                clear_mask[:, col] = True

        for row_block in range(3):
            for col_block in range(3):
                row_start = row_block * 3
                col_start = col_block * 3
                square = next_board[row_start : row_start + 3, col_start : col_start + 3]
                if square.sum() == 9:
                    clear_mask[row_start : row_start + 3, col_start : col_start + 3] = True

        next_board[clear_mask] = 0.0
        return next_board

    @staticmethod
    def _longest_empty_lanes(board):
        horizontal_lane = max(StrategicBlockudokuEnv._longest_zero_run(row) for row in board)
        vertical_lane = max(StrategicBlockudokuEnv._longest_zero_run(col) for col in board.T)
        return int(horizontal_lane), int(vertical_lane)

    @staticmethod
    def _longest_zero_run(values):
        longest = 0
        current = 0
        for value in values:
            if value == 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    @staticmethod
    def _open_square_count(board):
        open_squares = 0
        for row_block in range(3):
            for col_block in range(3):
                row_start = row_block * 3
                col_start = col_block * 3
                square = board[row_start : row_start + 3, col_start : col_start + 3]
                if square.sum() <= 2:
                    open_squares += 1
        return open_squares

    @staticmethod
    def _empty_square_count(board):
        empty_squares = 0
        for row_block in range(3):
            for col_block in range(3):
                row_start = row_block * 3
                col_start = col_block * 3
                square = board[row_start : row_start + 3, col_start : col_start + 3]
                if square.sum() == 0:
                    empty_squares += 1
        return empty_squares

    @staticmethod
    def _isolated_hole_count(board):
        holes = 0
        for row in range(9):
            for col in range(9):
                if board[row, col] != 0:
                    continue
                neighbors = []
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbor_row = row + delta_row
                    neighbor_col = col + delta_col
                    if 0 <= neighbor_row < 9 and 0 <= neighbor_col < 9:
                        neighbors.append(board[neighbor_row, neighbor_col])
                    else:
                        neighbors.append(1)
                if all(value > 0 for value in neighbors):
                    holes += 1
        return holes

    @staticmethod
    def _line_fill_counts(board):
        rows = board.sum(axis=1)
        cols = board.sum(axis=0)
        squares = []
        for row_block in range(3):
            for col_block in range(3):
                row_start = row_block * 3
                col_start = col_block * 3
                square = board[row_start : row_start + 3, col_start : col_start + 3]
                squares.append(square.sum())
        return np.concatenate([rows, cols, np.asarray(squares, dtype=np.float32)])

    @staticmethod
    def _clear_potential(fills):
        potential = 0.0
        for fill in fills:
            if fill >= 8:
                potential += 3.0
            elif fill >= 7:
                potential += 2.0
            elif fill >= 6:
                potential += 1.0
        return potential

    @staticmethod
    def _count_cleared_lines(board):
        lines_cleared = 0
        lines_cleared += int(np.sum(board.sum(axis=1) == 9))
        lines_cleared += int(np.sum(board.sum(axis=0) == 9))

        for row_block in range(3):
            for col_block in range(3):
                row_start = row_block * 3
                col_start = col_block * 3
                square = board[row_start : row_start + 3, col_start : col_start + 3]
                if square.sum() == 9:
                    lines_cleared += 1

        return lines_cleared


def make_env():
    return StrategicBlockudokuEnv()


def make_training_env():
    env = DummyVecEnv([make_env])
    return VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=25.0)


def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


class AverageScoreCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollout_scores = []
        self.rollout_lengths = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done and "score" in info:
                self.rollout_scores.append(float(info["score"]))
                self.rollout_lengths.append(int(info.get("episode_steps", 0)))

        return True

    def _on_rollout_end(self):
        if not self.rollout_scores:
            return

        avg_score = float(np.mean(self.rollout_scores))
        max_score = float(np.max(self.rollout_scores))
        avg_length = float(np.mean(self.rollout_lengths))
        max_length = int(np.max(self.rollout_lengths))
        episodes = len(self.rollout_scores)

        self.logger.record("score/avg_game_score", avg_score)
        self.logger.record("score/max_game_score", max_score)
        self.logger.record("score/avg_episode_steps", avg_length)
        self.logger.record("score/max_episode_steps", max_length)
        print(
            f"[score] rollout_avg={avg_score:.2f} "
            f"rollout_max={max_score:.2f} "
            f"steps_avg={avg_length:.1f} steps_max={max_length} episodes={episodes}"
        )

        self.rollout_scores.clear()
        self.rollout_lengths.clear()


def main():
    print("Starting survival-focused MaskablePPO training from scratch...")
    print("Rewards prioritize more steps, future valid moves, and avoiding game over...")

    env = make_training_env()
    valid_actions = int(env.envs[0].action_masks().sum())
    print(f"Action mask active. Legal actions on reset: {valid_actions}")

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = MaskablePPO(
        StableMaskableMultiInputActorCriticPolicy,
        env,
        gamma=0.995,
        vf_coef=1.0,
        ent_coef=ENTROPY_COEF,
        learning_rate=linear_schedule(INITIAL_LEARNING_RATE, FINAL_LEARNING_RATE),
        clip_range=0.2,
        max_grad_norm=0.5,
        n_steps=4096,
        batch_size=256,
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )

    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps with strategic rewards...")
    callbacks = [
        AverageScoreCallback(),
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=CHECKPOINT_DIR,
            name_prefix="ppo_survival",
        ),
    ]
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(MODEL_OUTPUT)
    env.save(VEC_NORMALIZE_OUTPUT)
    print(f"Training complete. Model saved as {MODEL_OUTPUT}.zip")
    print(f"VecNormalize stats saved as {VEC_NORMALIZE_OUTPUT}")
    print(f"TensorBoard logs are in: {TENSORBOARD_LOG_DIR}")
    print(f"Open TensorBoard with: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    env.close()


if __name__ == "__main__":
    main()
