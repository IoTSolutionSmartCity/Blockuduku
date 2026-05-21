from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from blockuduko.rl.action_codec import decode_action
from blockuduko.rl.env import BlockudukoEnv


def make_env(seed: int | None = None) -> BlockudukoEnv:
    env = BlockudukoEnv()
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def train(
    timesteps: int = 500_000,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 20,
    log_dir: str = "logs",
    model_dir: str = "models",
    seed: int = 42,
) -> Path:
    log_path = Path(log_dir)
    model_path = Path(model_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(make_env, n_envs=4, seed=seed)
    eval_env = make_vec_env(lambda: make_env(seed=seed + 1), n_envs=1, seed=seed + 1)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(model_path),
        log_path=str(log_path / "eval"),
        eval_freq=max(eval_freq // 4, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(log_path),
        seed=seed,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)

    final_path = model_path / "final_model"
    model.save(final_path)
    train_env.close()
    eval_env.close()
    return final_path


def evaluate(
    model_path: str,
    episodes: int = 100,
    seed: int = 123,
) -> dict[str, float]:
    env = BlockudukoEnv()
    model = MaskablePPO.load(model_path)

    scores: list[int] = []
    lengths: list[int] = []
    clears: list[int] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        ep_clears = 0
        steps = 0

        while not done:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            ep_clears += info.get("cleared_cells", 0)
            steps += 1
            done = terminated or truncated

        scores.append(info["score"])
        lengths.append(steps)
        clears.append(ep_clears)

    env.close()
    return {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_cleared_cells": float(np.mean(clears)),
    }


def watch(
    model_path: str | None = None,
    seed: int = 0,
    delay: float = 0.5,
    random_agent: bool = False,
) -> None:
    env = BlockudukoEnv()
    model = None if random_agent or model_path is None else MaskablePPO.load(model_path)

    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    print(f"\n=== Episode start (seed={seed}) ===")
    print(env.render())

    while not done:
        time.sleep(delay)
        mask = env.action_masks()
        legal = np.flatnonzero(mask)

        if random_agent or model is None:
            action = int(env.np_random.choice(legal))
            agent_label = "random"
        else:
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            agent_label = "model"

        piece_idx, row, col = decode_action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated

        print(f"\n--- Step {step} ({agent_label}) ---")
        print(f"action: piece={piece_idx} at ({row}, {col})")
        print(f"reward: {reward:.2f}  cleared: {info.get('cleared_cells', 0)}  score: {info['score']}")
        print(env.render())

    print(f"\n=== Game over after {step} moves | final score: {info['score']} ===")
    env.close()


def random_baseline(episodes: int = 100, seed: int = 0) -> dict[str, float]:
    env = BlockudukoEnv()
    scores: list[int] = []
    lengths: list[int] = []

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        done = False
        steps = 0
        info: dict = {}

        while not done:
            mask = env.action_masks()
            legal_actions = np.flatnonzero(mask)
            action = int(env.np_random.choice(legal_actions))
            _obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        scores.append(info["score"])
        lengths.append(steps)

    env.close()
    return {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate Blockuduko RL agent")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", type=str, default=None, help="Path to saved model")
    parser.add_argument("--watch", action="store_true", help="Watch one episode in the terminal")
    parser.add_argument("--model", type=str, default=None, help="Model path for --watch")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between moves in --watch")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--random-baseline", action="store_true")
    args = parser.parse_args()

    if args.watch:
        watch(
            model_path=args.model or args.evaluate,
            seed=args.seed,
            delay=args.delay,
            random_agent=args.random_baseline or (args.model is None and args.evaluate is None),
        )
        return

    if args.random_baseline:
        stats = random_baseline(episodes=args.episodes, seed=args.seed)
        print("Random baseline:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        return

    if args.evaluate:
        stats = evaluate(args.evaluate, episodes=args.episodes, seed=args.seed)
        print(f"Evaluation ({args.evaluate}):")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        return

    final_path = train(
        timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        seed=args.seed,
    )
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
