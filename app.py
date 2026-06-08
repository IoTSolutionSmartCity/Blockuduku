from pathlib import Path
import os

import gym
import gym_blocksudoku  # Registers "blocksudoku-v0".
import numpy as np
from flask import Flask, jsonify, render_template
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from train_boosted import LongSurvivalBlockudokuEnv

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    "ppo_blockudoku_survival_v2.zip",
    "ppo_blockudoku_survival.zip",
    "ppo_blockudoku_strategic_v2.zip",
    "ppo_blockudoku_strategic.zip",
    "ppo_blockudoku_agent.zip",
]
ENV_ID = "blocksudoku-v0"

app = Flask(__name__)

env = None
model = None
model_kind = None
current_obs = None
game_over = False
last_reward = 0.0


class CompatibleBlockudokuEnv:
    """
    Wrapper around LongSurvivalBlockudokuEnv that adapts the observation
    to match the keys expected by the loaded model.
    """
    def __init__(self, model_obs_space):
        self._env = LongSurvivalBlockudokuEnv()
        # Determine which keys the model expects
        self._expected_keys = set(model_obs_space.spaces.keys())
        print(f"Model expects observation keys: {self._expected_keys}")

    def reset(self):
        obs, info = self._env.reset()
        return self._filter_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action)
        return self._filter_obs(obs), reward, done, truncated, info

    def action_masks(self):
        return self._env.action_masks()

    def render(self, mode="human"):
        return self._env.render(mode)

    def close(self):
        self._env.close()

    def _filter_obs(self, obs_dict):
        # Return only the keys that the model expects
        filtered = {k: obs_dict[k] for k in self._expected_keys if k in obs_dict}
        # If the model expects keys that are not in the environment's observation,
        # you may need to handle that case. Here we assume the model's keys are a subset.
        return filtered


def resolve_model_path():
    configured = os.environ.get("BLOCKUDOKU_MODEL")
    if configured:
        configured_path = Path(configured)
        if not configured_path.is_absolute():
            configured_path = BASE_DIR / configured
        return configured_path

    for candidate in MODEL_CANDIDATES:
        candidate_path = BASE_DIR / candidate
        if candidate_path.exists():
            return candidate_path

    return BASE_DIR / MODEL_CANDIDATES[-1]


def get_model():
    global model, model_kind

    if model is None:
        model_path = resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find trained model: {model_path}")

        try:
            model = MaskablePPO.load(str(model_path))
            model_kind = "maskable"
        except Exception:
            model = PPO.load(str(model_path))
            model_kind = "ppo"

        print(f"Using model: {model_path.name} ({model_kind})")
    return model


def get_env():
    global env

    if env is None:
        if model_kind is None:
            get_model()
        if model_kind == "maskable":
            # Create a compatible environment that matches the model's observation space
            env = CompatibleBlockudokuEnv(model.observation_space)
        else:
            env = gym.make(ENV_ID, disable_env_checker=True)
    return env


def unwrap_env(wrapped_env):
    # For compatibility with existing code
    if hasattr(wrapped_env, '_env'):
        return wrapped_env._env.base_env
    return getattr(wrapped_env, "unwrapped", wrapped_env)


def reset_env():
    result = get_env().reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(action):
    result = get_env().step(action)

    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info

    obs, reward, done, info = result
    return obs, reward, done, info


def matrix_to_list(value):
    return np.asarray(value, dtype=int).tolist()


def trim_piece(piece):
    piece_array = np.asarray(piece, dtype=int)
    filled = np.argwhere(piece_array > 0)

    if filled.size == 0:
        return piece_array.tolist()

    min_row, min_col = filled.min(axis=0)
    max_row, max_col = filled.max(axis=0)
    return piece_array[min_row : max_row + 1, min_col : max_col + 1].tolist()


def get_valid_actions():
    game_env = get_env()
    if hasattr(game_env, "action_masks"):
        valid_mask = np.asarray(game_env.action_masks()).reshape(-1)
        return np.flatnonzero(valid_mask > 0)

    base_env = unwrap_env(game_env)
    valid_action_space = np.asarray(base_env.get_valid_action_space()).reshape(-1)
    return np.flatnonzero(valid_action_space > 0)


def get_board_env():
    game_env = get_env()
    # For the wrapper, we need to access the underlying base_env
    if hasattr(game_env, '_env'):
        return game_env._env.base_env
    if hasattr(game_env, "base_env"):
        return game_env.base_env
    base_env = unwrap_env(game_env)
    if hasattr(base_env, "base_env"):
        return base_env.base_env
    return base_env


def choose_playable_action(predicted_action):
    valid_actions = get_valid_actions()

    if valid_actions.size == 0:
        return None, False

    if predicted_action in valid_actions:
        return predicted_action, True

    return int(np.random.choice(valid_actions)), False


def serialize_state(reward=None, action=None, predicted_action=None, action_was_valid=None, info=None):
    board_env = get_board_env()
    board = matrix_to_list(board_env.main_board[:9, :9])
    pieces = [trim_piece(piece) for piece in board_env.block_queue]

    return {
        "board": board,
        "pieces": pieces,
        "reward": float(last_reward if reward is None else reward),
        "score": int(board_env.total_score),
        "game_over": bool(game_over),
        "action": None if action is None else int(action),
        "predicted_action": None if predicted_action is None else int(predicted_action),
        "action_was_valid": action_was_valid,
        "info": info or {},
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/reset")
def api_reset():
    global current_obs, game_over, last_reward

    current_obs = reset_env()
    game_over = False
    last_reward = 0.0
    return jsonify(serialize_state(reward=0.0))


@app.post("/api/ai_move")
def api_ai_move():
    global current_obs, game_over, last_reward

    if current_obs is None:
        current_obs = reset_env()
        game_over = False
        last_reward = 0.0

    if game_over:
        return jsonify(serialize_state(reward=last_reward, info={"message": "Game is over. Reset to play again."}))

    if model_kind == "maskable":
        action_masks = np.asarray(get_env().action_masks(), dtype=bool)
        action, _state = get_model().predict(current_obs, deterministic=True, action_masks=action_masks)
    else:
        action, _state = get_model().predict(current_obs, deterministic=True)
    predicted_action = int(np.asarray(action).item())
    action_index, action_was_valid = choose_playable_action(predicted_action)

    if action_index is None:
        game_over = True
        last_reward = 0.0
        return jsonify(
            serialize_state(
                reward=0.0,
                predicted_action=predicted_action,
                action_was_valid=False,
                info={"message": "No valid moves remain."},
            )
        )

    current_obs, reward, game_over, info = step_env(action_index)
    last_reward = float(reward)

    return jsonify(
        serialize_state(
            reward=reward,
            action=action_index,
            predicted_action=predicted_action,
            action_was_valid=action_was_valid,
            info=info,
        )
    )


if __name__ == "__main__":
    get_model()
    current_obs = reset_env()
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)