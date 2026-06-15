import numpy as np

from src.env import make_mario_env


def test_raw_env_builds_and_steps():
    env = make_mario_env(world_stage="SuperMarioBros-1-1-v0", preprocess=False)
    obs, info = env.reset(seed=0)
    assert obs.shape == (240, 256, 3)  # raw NES frame
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(float(reward), float)
    assert "flag_get" in info  # Mario env exposes this
    env.close()


def test_random_episode_reports_reward_and_done():
    env = make_mario_env(preprocess=False)
    env.reset(seed=0)
    total = 0.0
    done = False
    steps = 0
    while not done and steps < 500:
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += float(reward)
        done = terminated or truncated
        steps += 1
    assert steps > 0
    assert isinstance(total, float)
    env.close()
