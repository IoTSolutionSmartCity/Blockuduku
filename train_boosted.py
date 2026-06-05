import gym as classic_gym
import gym_blocksudoku  # Registers "blocksudoku-v0" with classic Gym.
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ENV_ID = "blocksudoku-v0"
TOTAL_TIMESTEPS = 1_000_000
TENSORBOARD_LOG_DIR = "./blocksudoku_tensorboard/"
MODEL_OUTPUT = "ppo_blockudoku_strategic"
VEC_NORMALIZE_OUTPUT = "vecnormalize_blockudoku_strategic.pkl"

SURVIVAL_REWARD = 2.0
BRICK_REWARD = 0.5
LINE_CLEAR_REWARD = 25.0
COMBO_BASE_REWARD = 10.0
BOARD_FULLNESS_PENALTY = 0.5
GAME_OVER_PENALTY = 20.0
MAX_COMBO_MULTIPLIER = 8.0


class StrategicBlockudokuEnv(gym.Env):
    """Gymnasium adapter with strategic observations, action masks, and survival-shaped rewards."""

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.env = classic_gym.make(ENV_ID, disable_env_checker=True)
        self.base_env = self.env.unwrapped
        self.combo_streak = 0

        self.action_space = spaces.Discrete(3 * 9 * 9)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0.0, high=1.0, shape=(9, 9), dtype=np.float32),
                "shapes": spaces.Box(low=0.0, high=1.0, shape=(75,), dtype=np.float32),
                "open_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.combo_streak = 0
        self.env.reset()
        return self._build_observation(), {}

    def step(self, action):
        action_index = int(np.asarray(action).item())
        board_before = self._board().copy()
        queue_pos, x_pos, y_pos = self.decode_action(action_index)
        selected_block = self._selected_block(queue_pos)
        placed_board = self._preview_board_after_placement(board_before, selected_block, x_pos, y_pos)

        _obs, original_reward, done, info = self.env.step(action_index)
        reward = self._strategic_reward(
            original_reward=original_reward,
            done=done,
            selected_block=selected_block,
            placed_board=placed_board,
        )

        return self._build_observation(), reward, bool(done), False, info

    def action_masks(self):
        return self.base_env.get_valid_action_space().astype(bool)

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
        block_rows, block_cols = selected_block.shape
        end_row = x_pos + block_rows
        end_col = y_pos + block_cols

        if x_pos < 0 or y_pos < 0 or end_row > 9 or end_col > 9:
            return None

        placed_board[x_pos:end_row, y_pos:end_col] += selected_block
        if placed_board.max() > 1:
            return None

        return placed_board

    def _build_observation(self):
        board = self._board()
        filled_cells = float(board.sum())
        open_ratio = np.array([(81.0 - filled_cells) / 81.0], dtype=np.float32)

        shapes = np.zeros((3, 5, 5), dtype=np.float32)
        for index, block in enumerate(self.base_env.block_queue[:3]):
            block_array = np.asarray(block, dtype=np.float32)
            rows, cols = block_array.shape
            shapes[index, :rows, :cols] = block_array

        return {
            "board": board.astype(np.float32),
            "shapes": shapes.reshape(-1),
            "open_ratio": open_ratio,
        }

    def _strategic_reward(self, original_reward, done, selected_block, placed_board):
        if selected_block is None or placed_board is None:
            self.combo_streak = 0
            return float(original_reward)

        brick_count = float(np.sum(selected_block > 0))
        lines_cleared = self._count_cleared_lines(placed_board)
        board_after = self._board()
        filled_cells = float(board_after.sum())

        reward = 0.0
        reward += SURVIVAL_REWARD if not done else 0.0
        reward += BRICK_REWARD * brick_count
        reward -= BOARD_FULLNESS_PENALTY * filled_cells

        if lines_cleared > 0:
            self.combo_streak += 1
            combo_multiplier = min(2 ** (self.combo_streak - 1), MAX_COMBO_MULTIPLIER)
            reward += LINE_CLEAR_REWARD * lines_cleared
            reward += COMBO_BASE_REWARD * combo_multiplier * lines_cleared
        else:
            self.combo_streak = 0

        if done:
            reward -= GAME_OVER_PENALTY

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
                square = board[row_start : row_start + 3, col_start : col_start + 3]
                if square.sum() == 9:
                    lines_cleared += 1

        return lines_cleared


def make_env():
    return StrategicBlockudokuEnv()


def make_training_env():
    env = DummyVecEnv([make_env])
    return VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)


def main():
    print("Starting strategic MaskablePPO training from scratch...")
    print("Creating dict-observation env with action masking and VecNormalize rewards...")

    env = make_training_env()
    valid_actions = int(env.envs[0].action_masks().sum())
    print(f"Action mask active. Legal actions on reset: {valid_actions}")

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        gamma=0.995,
        vf_coef=1.0,
        ent_coef=0.03,
        learning_rate=0.00025,
        clip_range=0.2,
        n_steps=4096,
        batch_size=256,
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )

    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps with strategic rewards...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    model.save(MODEL_OUTPUT)
    env.save(VEC_NORMALIZE_OUTPUT)
    print(f"Training complete. Model saved as {MODEL_OUTPUT}.zip")
    print(f"VecNormalize stats saved as {VEC_NORMALIZE_OUTPUT}")
    print(f"TensorBoard logs are in: {TENSORBOARD_LOG_DIR}")
    print(f"Open TensorBoard with: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    env.close()


if __name__ == "__main__":
    main()
