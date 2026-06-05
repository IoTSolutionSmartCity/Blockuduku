import gym
import gym_blocksudoku  # Registers "blocksudoku-v0".
import numpy as np
from stable_baselines3 import PPO


ENV_ID = "blocksudoku-v0"
TOTAL_TIMESTEPS = 1_500_000
TENSORBOARD_LOG_DIR = "./blocksudoku_tensorboard/"
MODEL_OUTPUT = "ppo_blockudoku_agent"


class CustomBlockudokuRewardWrapper(gym.Wrapper):
    """Replace the package reward with block-size, clear, and combo bonuses."""

    def __init__(self, env):
        super().__init__(env)
        self.current_streak = 0

    def reset(self, **kwargs):
        self.current_streak = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        action_index = int(np.asarray(action).item())
        base_env = self.unwrapped
        board_before = np.asarray(base_env.main_board[:9, :9], dtype=int).copy()

        queue_pos, x_pos, y_pos = self.decode_action(action_index)
        selected_block = self.get_selected_block(base_env, queue_pos)
        placed_board = self.preview_board_after_placement(board_before, selected_block, x_pos, y_pos)

        obs, original_reward, done, info = self.env.step(action_index)
        reward = self.calculate_custom_reward(
            original_reward=original_reward,
            done=done,
            selected_block=selected_block,
            placed_board=placed_board,
        )

        return obs, reward, done, info

    @staticmethod
    def decode_action(action_index):
        queue_pos = action_index // 81
        remainder = action_index % 81
        x_pos = remainder // 9
        y_pos = remainder % 9
        return queue_pos, x_pos, y_pos

    @staticmethod
    def get_selected_block(base_env, queue_pos):
        if queue_pos >= len(base_env.block_queue):
            return None
        return np.asarray(base_env.block_queue[queue_pos], dtype=int)

    @staticmethod
    def preview_board_after_placement(board, selected_block, x_pos, y_pos):
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

    def calculate_custom_reward(self, original_reward, done, selected_block, placed_board):
        if selected_block is None or placed_board is None:
            self.current_streak = 0
            return float(original_reward)

        block_marks = int(np.sum(selected_block > 0))
        eliminated_cells = self.count_eliminated_cells(placed_board)

        if eliminated_cells > 0:
            self.current_streak += 1
            combo_multiplier = 2**self.current_streak
            clear_bonus = eliminated_cells * combo_multiplier
        else:
            self.current_streak = 0
            clear_bonus = 0

        custom_reward = block_marks + clear_bonus

        if done:
            custom_reward -= 10

        return float(custom_reward)

    @staticmethod
    def count_eliminated_cells(board):
        clear_mask = np.zeros((9, 9), dtype=bool)

        full_rows = np.where(board.sum(axis=1) == 9)[0]
        full_cols = np.where(board.sum(axis=0) == 9)[0]

        clear_mask[full_rows, :] = True
        clear_mask[:, full_cols] = True

        for row_block in range(3):
            for col_block in range(3):
                row_start = row_block * 3
                col_start = col_block * 3
                square = board[row_start : row_start + 3, col_start : col_start + 3]
                if square.sum() == 9:
                    clear_mask[row_start : row_start + 3, col_start : col_start + 3] = True

        return int(np.sum(clear_mask & (board > 0)))


def main():
    print("Starting fresh PPO training with custom Blockudoku rewards...")
    print("Creating Blockudoku environment...")

    env = gym.make(ENV_ID, disable_env_checker=True)
    env = CustomBlockudokuRewardWrapper(env)

    model = PPO(
        "MlpPolicy",
        env,
        ent_coef=0.05,
        learning_rate=0.0001,
        clip_range=0.1,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )

    print(f"Training from scratch for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    model.save(MODEL_OUTPUT)
    print(f"Training complete. Model saved as {MODEL_OUTPUT}.zip")
    print(f"TensorBoard logs are in: {TENSORBOARD_LOG_DIR}")
    env.close()


if __name__ == "__main__":
    main()