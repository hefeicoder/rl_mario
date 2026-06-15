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


def _select_action(model, obs, env, deterministic=True):
    if model is None:
        return env.action_space.sample()
    action, _ = model.predict(np.asarray(obs), deterministic=deterministic)
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


# Native NES screen size; the live window is this scaled by `scale`.
NES_HEIGHT, NES_WIDTH = 240, 256


def play_live(checkpoint, episodes=3, max_steps=4000, fps=30,
              deterministic=False, scale=3):
    """Open a real-time, upscaled NES window and watch the agent play.

    nes-py's built-in "human" window is hardcoded to the native 240x256
    (tiny on a Retina display), so instead we render frames ourselves and
    blit them into our own viewer sized ``scale`` x larger — nes-py's viewer
    scales the image to fill the window, giving a crisp NxN view.

    deterministic=False (default) samples from the policy like training does,
    so each episode varies and an early agent explores further than its brittle
    greedy action would. deterministic=True always picks the argmax action —
    best for a well-trained agent, but on a deterministic emulator it produces
    the identical run every episode.
    """
    import time

    from nes_py._image_viewer import ImageViewer

    model = _load(checkpoint)
    env = make_mario_env(preprocess=True, render_mode="rgb_array")
    viewer = ImageViewer(
        caption=f"RL Mario ({scale}x)",
        height=NES_HEIGHT * scale,
        width=NES_WIDTH * scale,
    )
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=ep)
            for _ in range(max_steps):
                action = _select_action(model, obs, env, deterministic=deterministic)
                obs, _, terminated, truncated, info = env.step(action)
                viewer.show(np.asarray(env.render()))
                time.sleep(1.0 / fps)
                if terminated or truncated:
                    break
            print(f"episode {ep}: x_pos={info.get('x_pos')} flag={info.get('flag_get')}")
    finally:
        viewer.close()
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--record", action="store_true")
    p.add_argument("--out", default="videos/mario_1-1.gif")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--deterministic", action="store_true",
                   help="always pick the argmax action (best for a trained agent)")
    p.add_argument("--scale", type=int, default=3,
                   help="live window size multiplier over the native 240x256")
    args = p.parse_args()
    if args.record:
        path = record_episode(args.checkpoint, args.out)
        print(f"recorded -> {path}")
    else:
        play_live(args.checkpoint, episodes=args.episodes,
                  deterministic=args.deterministic, scale=args.scale)


if __name__ == "__main__":
    main()
