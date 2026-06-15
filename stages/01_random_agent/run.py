"""Random agent baseline: take random actions, print the reward signal.

Purpose: build intuition for the RL loop — observation in, action out,
reward + done back — before any learning happens. Run from the repo root:

    python stages/01_random_agent/run.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env import make_mario_env


def main(episodes=3, max_steps=2000):
    env = make_mario_env(preprocess=False)
    for ep in range(episodes):
        env.reset(seed=ep)
        total, done, steps = 0.0, False, 0
        max_x = 0
        info = {}
        while not done and steps < max_steps:
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += float(reward)
            max_x = max(max_x, info.get("x_pos", 0))
            done = terminated or truncated
            steps += 1
        print(f"episode {ep}: steps={steps} total_reward={total:.1f} "
              f"max_x_pos={max_x} flag={info.get('flag_get', False)}")
    env.close()


if __name__ == "__main__":
    main()
