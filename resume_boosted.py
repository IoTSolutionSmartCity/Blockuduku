from sb3_contrib import MaskablePPO

from train_boosted import TENSORBOARD_LOG_DIR, VEC_NORMALIZE_OUTPUT, make_training_env


SOURCE_MODEL = "ppo_blockudoku_strategic"
OUTPUT_MODEL = "ppo_blockudoku_strategic_v2"
ADDITIONAL_TIMESTEPS = 500_000


def main():
    print("Resuming boosted MaskablePPO training...")
    print("Using the same reward wrapper and action masking as train_boosted.py")

    env = make_training_env()
    valid_actions = int(env.envs[0].action_masks().sum())
    print(f"Action mask active. Legal actions on reset: {valid_actions}")

    print(f"Loading existing model from {SOURCE_MODEL}.zip...")
    model = MaskablePPO.load(SOURCE_MODEL, env=env)
    model.tensorboard_log = TENSORBOARD_LOG_DIR
    print("Model loaded successfully.")

    print(f"Training for {ADDITIONAL_TIMESTEPS:,} more timesteps...")
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="MaskablePPO",
        progress_bar=True,
    )

    model.save(OUTPUT_MODEL)
    env.save(VEC_NORMALIZE_OUTPUT)
    print(f"Training complete. Updated model saved as {OUTPUT_MODEL}.zip")
    print(f"Updated VecNormalize stats saved as {VEC_NORMALIZE_OUTPUT}")
    print(f"TensorBoard logs are in: {TENSORBOARD_LOG_DIR}")
    print(f"Open TensorBoard with: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    env.close()


if __name__ == "__main__":
    main()
