import gym
import gym_blocksudoku  # Registers "blocksudoku-v0" with classic Gym.
from shimmy import GymV21CompatibilityV0
from stable_baselines3 import PPO


ENV_ID = "blocksudoku-v0"
SOURCE_MODEL = "ppo_blockudoku_agent"
OUTPUT_MODEL = "ppo_blockudoku_agent_v2"
ADDITIONAL_TIMESTEPS = 1_000_000
TENSORBOARD_LOG_DIR = "./blocksudoku_tensorboard/"


def make_gymnasium_env():
    classic_env = gym.make(ENV_ID, disable_env_checker=True)
    return GymV21CompatibilityV0(env=classic_env)


def main():
    print("Recreating Blockudoku environment with Gymnasium compatibility...")
    env = make_gymnasium_env()

    print(f"Loading existing PPO model from {SOURCE_MODEL}.zip...")
    model = PPO.load(SOURCE_MODEL, env=env)
    model.tensorboard_log = TENSORBOARD_LOG_DIR
    print("Model loaded successfully. Resuming training...")

    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="PPO",
        progress_bar=True,
    )

    model.save(OUTPUT_MODEL)
    print(f"Training complete. Updated model saved as {OUTPUT_MODEL}.zip")
    print(f"To open TensorBoard, run: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    env.close()


if __name__ == "__main__":
    main()
