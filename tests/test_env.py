import numpy as np

from blockuduko.rl.action_codec import MAX_ACTIONS, decode_action, encode_action, moves_to_mask
from blockuduko.rl.env import BlockudukoEnv


def test_action_codec_roundtrip():
    for piece_idx in range(3):
        for row in range(9):
            for col in range(9):
                action = encode_action(piece_idx, row, col)
                assert decode_action(action) == (piece_idx, row, col)


def test_env_reset_observation_shapes():
    env = BlockudukoEnv()
    obs, info = env.reset(seed=0)
    assert obs["board"].shape == (9, 9)
    assert obs["hand"].shape == (3, 5, 5)
    assert obs["scalars"].shape == (4,)
    assert info["legal_action_count"] > 0


def test_env_reset_reproducible():
    env_a = BlockudukoEnv()
    env_b = BlockudukoEnv()
    obs_a, _ = env_a.reset(seed=5)
    obs_b, _ = env_b.reset(seed=5)
    assert np.allclose(obs_a["board"], obs_b["board"])
    assert np.allclose(obs_a["hand"], obs_b["hand"])


def test_action_mask_matches_legal_moves():
    env = BlockudukoEnv()
    env.reset(seed=3)
    mask = env.action_masks()
    assert mask.shape == (MAX_ACTIONS,)
    assert mask.dtype == bool
    assert mask.sum() == env.unwrapped._info(0)["legal_action_count"]


def test_masked_action_succeeds():
    env = BlockudukoEnv()
    env.reset(seed=1)
    mask = env.action_masks()
    action = int(np.argmax(mask))
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["board"].shape == (9, 9)
    assert info["illegal_action"] is False
    assert not truncated


def test_illegal_action_penalized():
    env = BlockudukoEnv()
    env.reset(seed=1)
    mask = env.action_masks()
    illegal = int(np.flatnonzero(~mask)[0])
    assert not mask[illegal]
    _obs, reward, _terminated, _truncated, info = env.step(illegal)
    assert reward < 0
    assert info["illegal_action"] is True


def test_random_valid_rollout():
    env = BlockudukoEnv()
    env.reset(seed=10)
    done = False
    steps = 0

    while not done and steps < 200:
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        action = int(np.random.default_rng(steps).choice(legal))
        _obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        steps += 1

    assert steps > 0
