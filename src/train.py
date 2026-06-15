"""Unified training entrypoint for DQN and PPO.

Usage:
    python src/train.py --algo ppo --timesteps 4000000
    python src/train.py --algo dqn --timesteps 1000000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from src.config import DEVICE, DQN_CONFIG, N_ENVS, PPO_CONFIG
from src.env import make_mario_env

ALGOS = {"ppo": (PPO, PPO_CONFIG), "dqn": (DQN, DQN_CONFIG)}


def _env_thunk():
    return make_mario_env(preprocess=True)


def build_model(algo, n_envs=1, tb_log="tb_logs"):
    algo = algo.lower()
    cls, cfg = ALGOS[algo]
    if algo == "ppo" and n_envs > 1:
        venv = SubprocVecEnv([_env_thunk for _ in range(n_envs)])
    else:
        venv = DummyVecEnv([_env_thunk])
    venv = VecMonitor(venv)
    model = cls(env=venv, tensorboard_log=tb_log, verbose=1, device=DEVICE, **cfg)
    return model, venv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=ALGOS, default="ppo")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--out", default="checkpoints/mario")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_envs = N_ENVS if args.algo == "ppo" else 1
    model, venv = build_model(args.algo, n_envs=n_envs)
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.out)
    venv.close()
    print(f"saved -> {args.out}.zip")


if __name__ == "__main__":
    main()
