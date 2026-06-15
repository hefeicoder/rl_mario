"""Watch or record a trained Mario agent.

    python src/play.py --checkpoint checkpoints/mario.zip            # live window
    python src/play.py --checkpoint checkpoints/mario.zip --record   # save gif
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio
import numpy as np
from stable_baselines3 import DQN, PPO

from src.env import make_mario_env


def _load(checkpoint):
    if checkpoint is None:
        return None
    # A PPO and a DQN zip are not interchangeable; try PPO first, fall back.
    try:
        return PPO.load(checkpoint)
    except Exception:
        return DQN.load(checkpoint)


def _select_action(model, obs, env):
    if model is None:
        return env.action_space.sample()
    action, _ = model.predict(np.asarray(obs), deterministic=True)
    return int(action)


def record_episode(checkpoint, out_path, max_steps=4000):
    model = _load(checkpoint)
    env = make_mario_env(preprocess=True)
    obs, info = env.reset(seed=0)
    frames = []
    for _ in range(max_steps):
        action = _select_action(model, obs, env)
        obs, _, terminated, truncated, info = env.step(action)
        frames.append(np.asarray(env.render()))  # raw NES screen
        if terminated or truncated:
            break
    env.close()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=30)
    return out_path


def play_live(checkpoint, episodes=3, max_steps=4000, fps=30):
    """Open a real-time NES window and watch the agent play.

    Uses render_mode="human" so nes-py pops a game window. We play a few
    episodes back-to-back (an early agent dies fast, so one episode is a
    blink) with a small delay so it's watchable.
    """
    import time

    model = _load(checkpoint)
    env = make_mario_env(preprocess=True, render_mode="human")
    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        env.render()
        for _ in range(max_steps):
            action = _select_action(model, obs, env)
            obs, _, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(1.0 / fps)
            if terminated or truncated:
                break
        print(f"episode {ep}: x_pos={info.get('x_pos')} flag={info.get('flag_get')}")
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--record", action="store_true")
    p.add_argument("--out", default="videos/mario_1-1.gif")
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()
    if args.record:
        path = record_episode(args.checkpoint, args.out)
        print(f"recorded -> {path}")
    else:
        play_live(args.checkpoint, episodes=args.episodes)


if __name__ == "__main__":
    main()
