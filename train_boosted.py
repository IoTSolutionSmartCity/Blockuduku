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
TOTAL_TIMESTEPS = 2_000_000          # much longer training to learn long survival
TENSORBOARD_LOG_DIR = "./blocksudoku_tensorboard/"
MODEL_OUTPUT = "ppo_blockudoku_survival_v2"
VEC_NORMALIZE_OUTPUT = "vecnormalize_blockudoku_survival_v2.pkl"

# ========== REWARD CONSTANTS (designed for long survival) ==========
SURVIVAL_REWARD = 2.0                 # per step alive
# Instead of a small positive openness reward, use a continuous penalty for fullness
FULLNESS_PENALTY_SCALE = 5.0          # penalty when board >50% full (quadratic)
LINE_CLEAR_REWARD = 500.0             # per cleared line (increased from 100)
NEAR_COMPLETE_REWARD = 100.0          # per line with 7 or 8 filled cells (increased from 20)
POST_CLEAR_OPEN_BONUS = 20.0          # bonus * open_ratio after clearing lines
OPEN_SQUARE_REWARD = 5.0              # per 3x3 square with ≤2 filled cells
LONG_LANE_REWARD = 5.0                # per lane (row/col) of length ≥5 (normalised)
VALID_ACTION_REWARD = 0.1             # per valid move after placement (scaled)
ISOLATED_HOLE_PENALTY = 10.0          # per isolated hole (cell surrounded by filled cells)
USELESS_PLACEMENT_PENALTY = 0.5       # per brick placed without progress
GAME_OVER_PENALTY = 200.0
BRICK_PLACEMENT_REWARD = 0.2          # positive reward for placing bricks that help progress

# ========== EXPLORATION & TRAINING ==========
LEARNING_RATE = 1e-4                  # lower, more stable
ENTROPY_COEF = 0.01                   # lower exploration now (policy is getting deterministic)
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


class LongSurvivalBlockudokuEnv(gym.Env):
    """
    Gymnasium environment with rich observations and reward shaping designed
    to encourage very long episodes (2000+ steps) by rewarding board openness,
    line progress, and future flexibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Try to set max_steps to a large number if the environment supports it
        try:
            self.env = classic_gym.make(ENV_ID, disable_env_checker=True, max_steps=5000)
        except TypeError:
            self.env = classic_gym.make(ENV_ID, disable_env_checker=True)
        self.base_env = self.env.unwrapped
        self.episode_steps = 0

        self.action_space = spaces.Discrete(3 * 9 * 9)
        # Observation includes openness metrics to help the agent
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0.0, high=1.0, shape=(9, 9), dtype=np.float32),
                "shapes": spaces.Box(low=0.0, high=1.0, shape=(75,), dtype=np.float32),
                "open_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "longest_lane": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "valid_action_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        self.env.reset()
        return self._build_observation(), {}

    def step(self, action):
        action_index = int(np.asarray(action).item())
        # Preview for reward calculation
        board_before = self._board().copy()
        queue_pos, x_pos, y_pos = self.decode_action(action_index)
        selected_block = self._selected_block(queue_pos)
        placed_board = self._preview_board_after_placement(board_before, selected_block, x_pos, y_pos)

        _obs, original_reward, done, info = self.env.step(action_index)
        self.episode_steps += 1

        # Compute the dense survival reward
        reward = self._dense_survival_reward(
            done=done,
            placed_board=placed_board,
            selected_block=selected_block,
            board_before=board_before,
        )

        info = dict(info)
        info["score"] = float(self.base_env.total_score)
        info["episode_steps"] = self.episode_steps
        # True open ratio at death (before reset)
        board_at_death = self._board()
        final_open_ratio = (81.0 - float(board_at_death.sum())) / 81.0
        info["final_open_ratio"] = final_open_ratio

        return self._build_observation(), reward, bool(done), False, info

    def action_masks(self):
        mask = self.base_env.get_valid_action_space().astype(bool)
        if not mask.any():
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
        open_ratio = (81.0 - filled_cells) / 81.0
        longest_lane = self._longest_empty_lane(board) / 9.0
        valid_actions = self._valid_action_count()
        valid_action_ratio = valid_actions / float(self.action_space.n)

        shapes = np.zeros((3, 5, 5), dtype=np.float32)
        for index, block in enumerate(self.base_env.block_queue[:3]):
            block_array = np.asarray(block, dtype=np.float32)
            rows, cols = block_array.shape
            shapes[index, :rows, :cols] = block_array

        return {
            "board": board.astype(np.float32),
            "shapes": shapes.reshape(-1),
            "open_ratio": np.array([open_ratio], dtype=np.float32),
            "longest_lane": np.array([longest_lane], dtype=np.float32),
            "valid_action_ratio": np.array([valid_action_ratio], dtype=np.float32),
        }

    def _valid_action_count(self):
        return int(np.sum(self.base_env.get_valid_action_space()))

    def _longest_empty_lane(self, board):
        # Longest consecutive empty cells in any row or column
        max_run = 0
        for row in range(9):
            run = 0
            for col in range(9):
                if board[row, col] == 0:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
        for col in range(9):
            run = 0
            for row in range(9):
                if board[row, col] == 0:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
        return max_run

    def _line_fill_counts(self, board):
        rows = board.sum(axis=1)
        cols = board.sum(axis=0)
        squares = []
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                squares.append(board[r:r+3, c:c+3].sum())
        return np.concatenate([rows, cols, squares])

    def _open_square_count(self, board):
        # Number of 3x3 squares with ≤2 filled cells
        count = 0
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                if board[r:r+3, c:c+3].sum() <= 2:
                    count += 1
        return count

    def _isolated_hole_count(self, board):
        # Empty cell completely surrounded by filled cells (or edges)
        holes = 0
        for row in range(9):
            for col in range(9):
                if board[row, col] != 0:
                    continue
                # Check four orthogonal neighbors
                surrounded = True
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = row+dr, col+dc
                    if 0 <= nr < 9 and 0 <= nc < 9:
                        if board[nr, nc] == 0:
                            surrounded = False
                            break
                    # edge counts as filled (blocked)
                if surrounded:
                    holes += 1
        return holes

    def _dense_survival_reward(self, done, placed_board, selected_block, board_before):
        reward = SURVIVAL_REWARD

        if placed_board is None or selected_block is None:
            # Invalid placement should not happen, but if it does, just survival
            return float(reward)

        # ---- Board openness after placement (board_after) ----
        board_after = self._board()
        empty_cells = 81.0 - float(board_after.sum())
        open_ratio = empty_cells / 81.0
        filled_ratio = 1.0 - open_ratio

        # Continuous penalty when board is more than half full
        if filled_ratio > 0.5:
            fullness_penalty = FULLNESS_PENALTY_SCALE * (filled_ratio - 0.5) ** 2
            reward -= fullness_penalty

        # ---- Line progress and clears ----
        fills = self._line_fill_counts(placed_board)  # before clearing lines
        near_complete = np.sum((fills >= 7) & (fills < 9))
        reward += NEAR_COMPLETE_REWARD * near_complete

        lines_cleared = self._count_cleared_lines(placed_board)
        if lines_cleared > 0:
            reward += LINE_CLEAR_REWARD * lines_cleared
            # Bonus for how open the board becomes after clears
            reward += POST_CLEAR_OPEN_BONUS * open_ratio

        # ---- Open squares (3x3 with ≤2 cells) ----
        open_squares = self._open_square_count(board_after)
        reward += OPEN_SQUARE_REWARD * open_squares

        # ---- Longest empty lane (keeps board open) ----
        longest_lane = self._longest_empty_lane(board_after)
        reward += LONG_LANE_REWARD * (longest_lane / 9.0)

        # ---- Future flexibility: number of valid moves ----
        valid_actions = self._valid_action_count()
        reward += VALID_ACTION_REWARD * (valid_actions / 10.0)   # scaled

        # ---- Penalties ----
        isolated_holes = self._isolated_hole_count(board_after)
        reward -= ISOLATED_HOLE_PENALTY * isolated_holes

        # Penalise placing bricks that do not contribute to progress
        bricks_placed = int(np.sum(selected_block > 0))
        if lines_cleared == 0 and near_complete == 0:
            reward -= USELESS_PLACEMENT_PENALTY * bricks_placed
        else:
            # Slight positive for placing bricks that help
            reward += BRICK_PLACEMENT_REWARD * bricks_placed

        # ---- Death penalty that depends on how full the board is ----
        if done:
            board_after = self._board()
            empty_cells = 81.0 - float(board_after.sum())
            open_ratio = empty_cells / 81.0

            # Define mapping: open_ratio -> penalty (positive value to subtract)
            open_ratios = [0.1, 0.34, 0.6, 0.8, 0.9]
            penalties = [1000.0, 800.0, 400.0, 200.0, 50.0]

            death_penalty = np.interp(open_ratio, open_ratios, penalties)
            reward -= death_penalty

        return float(reward)

    @staticmethod
    def _count_cleared_lines(board):
        lines_cleared = 0
        lines_cleared += int(np.sum(board.sum(axis=1) == 9))
        lines_cleared += int(np.sum(board.sum(axis=0) == 9))
        for row_block in range(3):
            for col_block in range(3):
                row_start = row_block * 3
                col_start = col_block * 3
                square = board[row_start:row_start+3, col_start:col_start+3]
                if square.sum() == 9:
                    lines_cleared += 1
        return lines_cleared


def make_env():
    return LongSurvivalBlockudokuEnv()


def make_training_env():
    env = DummyVecEnv([make_env])
    return VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=25.0)


class DetailedMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollout_scores = []
        self.rollout_lengths = []
        self.rollout_open_ratios = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done and "score" in info:
                self.rollout_scores.append(float(info["score"]))
                self.rollout_lengths.append(int(info.get("episode_steps", 0)))
                # Correct open ratio recorded at the end of the episode
                if "final_open_ratio" in info:
                    self.rollout_open_ratios.append(float(info["final_open_ratio"]))
        return True

    def _on_rollout_end(self):
        if not self.rollout_scores:
            return
        avg_score = float(np.mean(self.rollout_scores))
        max_score = float(np.max(self.rollout_scores))
        avg_length = float(np.mean(self.rollout_lengths))
        max_length = int(np.max(self.rollout_lengths))
        avg_open = float(np.mean(self.rollout_open_ratios)) if self.rollout_open_ratios else 0.0
        episodes = len(self.rollout_scores)

        self.logger.record("score/avg_game_score", avg_score)
        self.logger.record("score/max_game_score", max_score)
        self.logger.record("score/avg_episode_steps", avg_length)
        self.logger.record("score/max_episode_steps", max_length)
        self.logger.record("score/avg_final_open_ratio", avg_open)
        print(f"[metrics] steps_avg={avg_length:.1f} steps_max={max_length} "
              f"score_avg={avg_score:.1f} open_ratio_at_death={avg_open:.2f} episodes={episodes}")

        self.rollout_scores.clear()
        self.rollout_lengths.clear()
        self.rollout_open_ratios.clear()


def main():
    print("=== Long survival training for Blockudoku ===")
    print("Reward components:")
    print(f"  +{SURVIVAL_REWARD} per step")
    print(f"  Continuous fullness penalty (quadratic when >50% full, scale={FULLNESS_PENALTY_SCALE})")
    print(f"  +{LINE_CLEAR_REWARD} per cleared line")
    print(f"  +{NEAR_COMPLETE_REWARD} per line with 7-8 filled")
    print(f"  +{POST_CLEAR_OPEN_BONUS} * open_ratio after line clear")
    print(f"  +{OPEN_SQUARE_REWARD} per open 3x3 square")
    print(f"  +{LONG_LANE_REWARD} * (longest_lane/9)")
    print(f"  +{VALID_ACTION_REWARD} * (valid_moves/10)")
    print(f"  -{ISOLATED_HOLE_PENALTY} per isolated hole")
    print(f"  -{USELESS_PLACEMENT_PENALTY} per brick placed without progress")
    print(f"  Death penalty depends on final open ratio (0.1→1000, 0.9→50)")
    print(f"Observation includes board, shapes, open_ratio, longest_lane, valid_action_ratio.")
    print(f"Learning rate: {LEARNING_RATE} (constant), entropy coef: {ENTROPY_COEF}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}\n")

    env = make_training_env()
    valid_actions = int(env.envs[0].action_masks().sum())
    print(f"Legal actions on reset: {valid_actions}")

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = MaskablePPO(
        StableMaskableMultiInputActorCriticPolicy,
        env,
        gamma=0.99,                       # lower to focus on immediate survival
        vf_coef=1.0,
        ent_coef=ENTROPY_COEF,
        learning_rate=LEARNING_RATE,
        clip_range=0.2,
        max_grad_norm=0.5,
        n_steps=32768,                    # even longer rollouts for sparse signals
        batch_size=2048,
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )

    callbacks = [
        DetailedMetricsCallback(),
        CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR, name_prefix="ppo_long_survival"),
    ]
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

    model.save(MODEL_OUTPUT)
    env.save(VEC_NORMALIZE_OUTPUT)
    print(f"\nTraining complete. Model saved as {MODEL_OUTPUT}.zip")
    print(f"VecNormalize stats saved as {VEC_NORMALIZE_OUTPUT}")
    print(f"TensorBoard logs: {TENSORBOARD_LOG_DIR}")
    env.close()


if __name__ == "__main__":
    main()