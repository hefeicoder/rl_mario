"""Unified training entrypoint for DQN and PPO.

Usage:
    python src/train.py --algo ppo --timesteps 4000000
    python src/train.py --algo dqn --timesteps 1000000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from src.config import DEVICE, DQN_CONFIG, N_ENVS, PPO_CONFIG
from src.env import make_mario_env

ALGOS = {"ppo": (PPO, PPO_CONFIG), "dqn": (DQN, DQN_CONFIG)}


class MarioStatsCallback(BaseCallback):
    """Log Mario-specific progress to TensorBoard so you can watch the agent
    learn to beat the level, not just watch reward go up.

    Adds two curves under ``mario/``:
      - ``mario/max_x_pos``: rolling mean of how far right the agent got each
        episode (the 1-1 flagpole is near x_pos ~3160).
      - ``mario/flag_rate``: rolling fraction of recent episodes that reached
        the flag (this is the win rate — the number you want to reach 1.0).
    """

    def __init__(self, window=100):
        super().__init__()
        self.window = window
        self._ep_max_x = {}
        self._x_hist = []
        self._flag_hist = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        for i, info in enumerate(infos):
            self._ep_max_x[i] = max(self._ep_max_x.get(i, 0), info.get("x_pos", 0))
            if dones[i]:
                self._x_hist.append(self._ep_max_x.get(i, 0))
                self._flag_hist.append(1.0 if info.get("flag_get") else 0.0)
                self._ep_max_x[i] = 0
                self._x_hist = self._x_hist[-self.window:]
                self._flag_hist = self._flag_hist[-self.window:]
                self.logger.record("mario/max_x_pos", float(np.mean(self._x_hist)))
                self.logger.record("mario/flag_rate", float(np.mean(self._flag_hist)))
        return True


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
    p.add_argument("--run-name", default=None,
                   help="TensorBoard run folder name (default: PPO_N / DQN_N)")
    p.add_argument("--save-freq", type=int, default=100_000,
                   help="save an intermediate checkpoint every N total timesteps "
                        "(so you can play snapshots while training continues)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_envs = N_ENVS if args.algo == "ppo" else 1
    model, venv = build_model(args.algo, n_envs=n_envs)

    # CheckpointCallback counts its own _on_step calls (one per n_envs steps),
    # so divide by n_envs to save roughly every `save_freq` *total* timesteps.
    prefix = os.path.basename(args.out)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_freq // n_envs, 1),
        save_path=os.path.dirname(args.out) or ".",
        name_prefix=prefix,
    )
    callbacks = CallbackList([MarioStatsCallback(), checkpoint_cb])

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        callback=callbacks,
        tb_log_name=args.run_name or args.algo.upper(),
    )
    model.save(args.out)
    venv.close()
    print(f"saved -> {args.out}.zip")


if __name__ == "__main__":
    main()
