from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

from train_boosted import (
    CHECKPOINT_DIR,
    CHECKPOINT_FREQ,
    TENSORBOARD_LOG_DIR,
    VEC_NORMALIZE_OUTPUT,
    AverageScoreCallback,
    StableMaskableMultiInputActorCriticPolicy,
    make_training_env,
)


SOURCE_MODEL = "ppo_blockudoku_survival"
OUTPUT_MODEL = "ppo_blockudoku_survival_v2"
ADDITIONAL_TIMESTEPS = 3_000_000


def main():
    print("Resuming survival-focused MaskablePPO training...")
    print("Using the same survival reward wrapper and action masking as train_boosted.py")

    env = make_training_env()
    valid_actions = int(env.envs[0].action_masks().sum())
    print(f"Action mask active. Legal actions on reset: {valid_actions}")

    print(f"Loading existing model from {SOURCE_MODEL}.zip...")
    model = MaskablePPO.load(
        SOURCE_MODEL,
        env=env,
        custom_objects={"policy_class": StableMaskableMultiInputActorCriticPolicy},
    )
    model.tensorboard_log = TENSORBOARD_LOG_DIR
    print("Model loaded successfully.")

    print(f"Training for {ADDITIONAL_TIMESTEPS:,} more timesteps...")
    callbacks = [
        AverageScoreCallback(),
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=CHECKPOINT_DIR,
            name_prefix="ppo_survival",
        ),
    ]
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="MaskablePPO",
        callback=callbacks,
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
