from pathlib import Path

import gym
import gym_blocksudoku  # Registers "blocksudoku-v0".
import numpy as np
from flask import Flask, jsonify, render_template
from stable_baselines3 import PPO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ppo_blockudoku_agent.zip"
ENV_ID = "blocksudoku-v0"

app = Flask(__name__)

env = None
model = None
current_obs = None
game_over = False
last_reward = 0.0


def get_model():
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Could not find trained model: {MODEL_PATH}")
        model = PPO.load(str(MODEL_PATH))
    return model


def get_env():
    global env

    if env is None:
        env = gym.make(ENV_ID, disable_env_checker=True)
    return env


def unwrap_env(wrapped_env):
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
    base_env = unwrap_env(get_env())
    valid_action_space = np.asarray(base_env.get_valid_action_space()).reshape(-1)
    return np.flatnonzero(valid_action_space > 0)


def choose_playable_action(predicted_action):
    valid_actions = get_valid_actions()

    if valid_actions.size == 0:
        return None, False

    if predicted_action in valid_actions:
        return predicted_action, True

    return int(np.random.choice(valid_actions)), False


def serialize_state(reward=None, action=None, predicted_action=None, action_was_valid=None, info=None):
    base_env = unwrap_env(get_env())
    board = matrix_to_list(base_env.main_board[:9, :9])
    pieces = [trim_piece(piece) for piece in base_env.block_queue]

    return {
        "board": board,
        "pieces": pieces,
        "reward": float(last_reward if reward is None else reward),
        "score": int(base_env.total_score),
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
