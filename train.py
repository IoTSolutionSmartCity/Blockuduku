import sys

import gym
import gym_blocksudoku  # Registers "blocksudoku-v0".
from stable_baselines3 import PPO


def main():
    print("train.py is executing")
    print(f"Python executable: {sys.executable}")
    print("Initializing Blockudoku environment...")

    env = gym.make("blocksudoku-v0", disable_env_checker=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        tensorboard_log="./blocksudoku_tensorboard/",
    )

    print("Training the AI... This will take a few minutes.")
    model.learn(total_timesteps=100_000)

    model.save("ppo_blockudoku_agent")
    print("Training complete! Model saved as ppo_blockudoku_agent.zip")


if __name__ == "__main__":
    main()