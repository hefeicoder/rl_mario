# RL Mario (Stages 00–03) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up an open-source, documentation-first RL project that trains a Stable-Baselines3 agent to clear Super Mario Bros World 1-1, built as a staged learning curriculum.

**Architecture:** A single Mario environment factory applies a fixed preprocessing wrapper stack (grayscale → resize → frame-skip → frame-stack → reduced actions), bridged from the legacy Gym API to Gymnasium. A unified `train.py` drives either DQN or PPO via `--algo`; `play.py` loads a checkpoint to render live or record a video. Each curriculum stage is a folder with its own explainer that links into the top-level README tutorial.

**Tech Stack:** Python 3.12, `gym-super-mario-bros` + `nes-py`, `shimmy` (Gym→Gymnasium bridge), Stable-Baselines3 (PyTorch, MPS where useful), TensorBoard, pytest.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Pinned dependencies (locked in Task 1) |
| `.gitignore` | Exclude venv, checkpoints, videos, tb logs |
| `LICENSE` | MIT |
| `README.md` | Source-of-truth RL tutorial (grown across stages) |
| `src/__init__.py` | Marks `src` as a package |
| `src/env.py` | Env factory + preprocessing wrappers (the one place env config lives) |
| `src/config.py` | Per-algorithm hyperparameters |
| `src/train.py` | Unified training entrypoint (`--algo dqn\|ppo`) |
| `src/play.py` | Load checkpoint → live render or `--record` video |
| `tests/test_env.py` | Smoke + shape tests for the env stack |
| `stages/NN_*/README.md` | Per-stage explainer linking to README sections |

---

## Task 1: Project scaffold, license, and pinned environment

**Files:**
- Create: `.gitignore`, `LICENSE`, `requirements.txt`, `README.md` (stub), `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create the virtualenv**

Run:
```bash
cd /Users/xishu/work/rl/rl_mario
python3 -m venv venv
source venv/bin/activate
python --version   # expect Python 3.12.x
```

- [ ] **Step 2: Attempt the install and LOCK working versions**

This is the project's main risk (legacy `nes-py`/`gym` on Py3.12). Install, and if it builds, freeze exact versions. Run:
```bash
pip install --upgrade pip wheel setuptools
pip install "stable-baselines3[extra]" gym-super-mario-bros nes-py shimmy pytest tensorboard imageio imageio-ffmpeg
```

If `nes-py` fails to build on 3.12, fall back in this order and record what worked in `stages/00_setup/README.md`:
1. `pip install nes-py --no-build-isolation` after `pip install setuptools wheel`.
2. Recreate the venv on Python 3.10 via pyenv (`pyenv install 3.10.14 && pyenv local 3.10.14`) — most reliable for this stack.
3. Pin older known-good combo: `gym==0.23.1 nes-py==8.2.1 gym-super-mario-bros==7.4.0`.

- [ ] **Step 3: Freeze the working set**

Run:
```bash
pip freeze > requirements.txt
```
Expected: `requirements.txt` lists exact versions that actually import.

- [ ] **Step 4: Write `.gitignore`**

```gitignore
venv/
__pycache__/
*.pyc
checkpoints/
videos/
runs/
tb_logs/
.DS_Store
.pytest_cache/
# keep the final winning model and demo gif (force-added explicitly)
!checkpoints/mario_1-1_best.zip
!videos/mario_1-1.gif
```

- [ ] **Step 5: Write `LICENSE` (MIT)**

```text
MIT License

Copyright (c) 2026 hefeicoder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 6: Create package markers and README stub**

`src/__init__.py`:
```python
```
`tests/__init__.py`:
```python
```
`README.md`:
```markdown
# RL Mario

Self-teaching reinforcement learning by training an agent to beat
Super Mario Bros (NES) World 1-1. Built with Stable-Baselines3.

> Tutorial sections are filled in as the curriculum progresses. See `stages/`.
```

- [ ] **Step 7: Commit**

```bash
git add .gitignore LICENSE requirements.txt README.md src/__init__.py tests/__init__.py
git commit -m "chore: scaffold project, MIT license, pinned deps"
git push -u origin main
```

---

## Task 2: Stage 00 — env factory + render verification

**Files:**
- Create: `src/env.py`
- Create: `stages/00_setup/README.md`
- Test: `tests/test_env.py`

- [ ] **Step 1: Write the failing test**

`tests/test_env.py`:
```python
import numpy as np
from src.env import make_mario_env


def test_raw_env_builds_and_steps():
    env = make_mario_env(world_stage="SuperMarioBros-1-1-v0", preprocess=False)
    obs, info = env.reset(seed=0)
    assert obs.shape == (240, 256, 3)          # raw NES frame
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(float(reward), float)
    assert "flag_get" in info                   # Mario env exposes this
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env.py::test_raw_env_builds_and_steps -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.env'`.

- [ ] **Step 3: Implement `src/env.py` (factory, raw path only for now)**

```python
"""Super Mario Bros environment factory.

One place that knows how to construct the Mario env and apply the
preprocessing wrapper stack. Bridges the legacy Gym API used by
gym-super-mario-bros to the modern Gymnasium API expected by SB3.
"""
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy import GymV21CompatibilityV0


def make_mario_env(world_stage="SuperMarioBros-1-1-v0", preprocess=True):
    legacy = gym_super_mario_bros.make(world_stage)
    legacy = JoypadSpace(legacy, SIMPLE_MOVEMENT)   # reduce action space
    env = GymV21CompatibilityV0(env=legacy)         # -> Gymnasium API
    if preprocess:
        from src.env import apply_preprocessing      # defined in Task 4
        env = apply_preprocessing(env)
    return env
```

Note: the exact shimmy class (`GymV21CompatibilityV0` vs `GymV26CompatibilityV0`) depends on the gym version locked in Task 1. If `reset()`/`step()` arity is wrong, switch the class and record it in the stage README.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env.py::test_raw_env_builds_and_steps -v`
Expected: PASS. If it fails on the shimmy bridge, swap the compatibility class and re-run.

- [ ] **Step 5: Manual render check (the Stage 00 milestone)**

Create a throwaway check and run it:
```bash
python -c "
from src.env import make_mario_env
env = make_mario_env(preprocess=False)
env.reset(seed=0)
for _ in range(300):
    env.step(env.action_space.sample())
    env.render()
env.close()
"
```
Expected: a window shows Mario; random inputs visibly move him. If the SDL window misbehaves on macOS, note it — `play.py` will provide record-mode as the reliable path.

- [ ] **Step 6: Write `stages/00_setup/README.md`**

```markdown
# Stage 00 — Setup & Render Check

**Goal:** prove the environment stack works on this machine before any learning.

## What worked
- Python version: <fill in>
- shimmy compatibility class used: <GymV21CompatibilityV0 / other>
- nes-py install path: <pip / no-build-isolation / pyenv 3.10>

## What you learned
The Mario env is the *environment* in the RL loop: it takes an action each
frame and returns an observation (the screen), a reward, and done flags.
See the README section "Turning Mario into an RL problem".

## Run it
\`\`\`bash
pytest tests/test_env.py -v
\`\`\`
```

- [ ] **Step 7: Commit**

```bash
git add src/env.py tests/test_env.py stages/00_setup/README.md
git commit -m "feat: Stage 00 - Mario env factory + render verification"
git push
```

---

## Task 3: Stage 01 — random agent baseline

**Files:**
- Create: `stages/01_random_agent/run.py`
- Create: `stages/01_random_agent/README.md`
- Test: add to `tests/test_env.py`

- [ ] **Step 1: Write the failing test (reward/episode signal is readable)**

Add to `tests/test_env.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env.py::test_random_episode_reports_reward_and_done -v`
Expected: FAIL only if the helper isn't importable; otherwise it should drive development of the run script. (If it already passes, that's fine — it locks the contract.)

- [ ] **Step 3: Implement `stages/01_random_agent/run.py`**

```python
"""Random agent baseline: take random actions, print the reward signal.

Purpose: build intuition for the RL loop — observation in, action out,
reward + done back — before any learning happens.
"""
from src.env import make_mario_env


def main(episodes=3, max_steps=2000):
    env = make_mario_env(preprocess=False)
    for ep in range(episodes):
        env.reset(seed=ep)
        total, done, steps = 0.0, False, 0
        max_x = 0
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
```

- [ ] **Step 4: Run it**

Run: `python stages/01_random_agent/run.py`
Expected: three lines printing reward and max x position; a random agent rarely gets far (low `max_x_pos`, `flag=False`) — that's the baseline that motivates learning.

- [ ] **Step 5: Run the test suite**

Run: `pytest tests/test_env.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Write `stages/01_random_agent/README.md`**

```markdown
# Stage 01 — Random Agent Baseline

**Goal:** understand the RL loop and the reward signal with zero learning.

A random policy almost never reaches the flag. The Mario reward combines
rightward progress, a time penalty, and a death penalty — so "good" behavior
means moving right and staying alive. This baseline is the bar every trained
agent must beat. See README section "RL in 5 minutes".

## Run it
\`\`\`bash
python stages/01_random_agent/run.py
\`\`\`
```

- [ ] **Step 7: Commit**

```bash
git add stages/01_random_agent tests/test_env.py
git commit -m "feat: Stage 01 - random agent baseline"
git push
```

---

## Task 4: Preprocessing wrapper stack

**Files:**
- Modify: `src/env.py` (add `apply_preprocessing`)
- Test: add to `tests/test_env.py`

- [ ] **Step 1: Write the failing test (preprocessed observation shape)**

Add to `tests/test_env.py`:
```python
def test_preprocessed_observation_shape():
    env = make_mario_env(preprocess=True)
    obs, info = env.reset(seed=0)
    # 4 stacked grayscale 84x84 frames, channels-first for SB3 CNN
    assert obs.shape == (4, 84, 84)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (4, 84, 84)
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env.py::test_preprocessed_observation_shape -v`
Expected: FAIL with `ImportError: cannot import name 'apply_preprocessing'`.

- [ ] **Step 3: Implement `apply_preprocessing` in `src/env.py`**

Add the imports at the top and the function:
```python
from gymnasium.wrappers import (
    GrayScaleObservation,
    ResizeObservation,
    FrameStack,
)
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv


def apply_preprocessing(env, skip=4, size=84, stack=4):
    """Make raw NES frames tractable for a CNN policy.

    grayscale -> resize 84x84 -> act every `skip` frames -> stack `stack`
    frames so motion is observable. Output is (stack, size, size).
    """
    env = MaxAndSkipEnv(env, skip=skip)            # frame-skip + max-pool
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (size, size))
    env = FrameStack(env, stack)                    # -> (stack, size, size)
    return env
```

Note: `FrameStack` yields a `LazyFrames` whose `np.asarray` is `(stack, H, W)`. If a wrapper version mismatches arity with the shimmy-bridged env, record the working wrapper versions in `stages/02_dqn/README.md`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env.py::test_preprocessed_observation_shape -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/test_env.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/env.py tests/test_env.py
git commit -m "feat: preprocessing wrapper stack (grayscale/resize/skip/stack)"
git push
```

---

## Task 5: Config + unified training entrypoint

**Files:**
- Create: `src/config.py`
- Create: `src/train.py`
- Test: add `tests/test_train.py`

- [ ] **Step 1: Write `src/config.py`**

```python
"""Per-algorithm hyperparameters, kept out of train.py so experiments
are easy to read and diff."""

DQN_CONFIG = dict(
    policy="CnnPolicy",
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=32,
    gamma=0.9,
    train_freq=4,
    target_update_interval=10_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.02,
)

PPO_CONFIG = dict(
    policy="CnnPolicy",
    learning_rate=2.5e-4,
    n_steps=512,
    batch_size=256,
    n_epochs=4,
    gamma=0.9,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
)

# Number of parallel envs for PPO (M1 Max has 10 cores; leave headroom).
N_ENVS = 8
```

- [ ] **Step 2: Write the failing test (train builds a model, runs N steps)**

`tests/test_train.py`:
```python
from src.train import build_model


def test_build_ppo_model_smoke():
    model, env = build_model(algo="ppo", n_envs=1, tb_log=None)
    model.learn(total_timesteps=64)   # tiny smoke run
    env.close()


def test_build_dqn_model_smoke():
    model, env = build_model(algo="dqn", n_envs=1, tb_log=None)
    model.learn(total_timesteps=64)
    env.close()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_model'`.

- [ ] **Step 4: Implement `src/train.py`**

```python
"""Unified training entrypoint for DQN and PPO.

Usage:
    python src/train.py --algo ppo --timesteps 2000000
    python src/train.py --algo dqn --timesteps 1000000
"""
import argparse

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from src.config import DQN_CONFIG, PPO_CONFIG, N_ENVS
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
    model = cls(env=venv, tensorboard_log=tb_log, verbose=1, **cfg)
    return model, venv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=ALGOS, default="ppo")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--out", default="checkpoints/mario")
    args = p.parse_args()

    n_envs = N_ENVS if args.algo == "ppo" else 1
    model, venv = build_model(args.algo, n_envs=n_envs)
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.out)
    venv.close()
    print(f"saved -> {args.out}.zip")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (two tiny smoke runs complete without error). This may take a minute.

- [ ] **Step 6: Commit**

```bash
git add src/config.py src/train.py tests/test_train.py
git commit -m "feat: config + unified DQN/PPO training entrypoint"
git push
```

---

## Task 6: Play / record script

**Files:**
- Create: `src/play.py`
- Test: add `tests/test_play.py`

- [ ] **Step 1: Write the failing test (record produces a file)**

`tests/test_play.py`:
```python
import os
from src.play import record_episode


def test_record_episode_writes_gif(tmp_path):
    out = tmp_path / "demo.gif"
    # random policy (checkpoint=None) keeps the test fast and dependency-free
    record_episode(checkpoint=None, out_path=str(out), max_steps=50)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_play.py -v`
Expected: FAIL with `ImportError: cannot import name 'record_episode'`.

- [ ] **Step 3: Implement `src/play.py`**

```python
"""Watch or record a trained Mario agent.

    python src/play.py --checkpoint checkpoints/mario.zip            # live window
    python src/play.py --checkpoint checkpoints/mario.zip --record   # save mp4+gif
"""
import argparse

import imageio
import numpy as np
from stable_baselines3 import PPO, DQN

from src.env import make_mario_env


def _load(checkpoint):
    if checkpoint is None:
        return None
    # PPO/DQN zips are interchangeable to load via either loader header; try PPO first.
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
        frames.append(env.render())          # rgb array of the NES screen
        if terminated or truncated:
            break
    env.close()
    imageio.mimsave(out_path, frames, fps=30)
    return out_path


def play_live(checkpoint, max_steps=4000):
    model = _load(checkpoint)
    env = make_mario_env(preprocess=True)
    obs, info = env.reset(seed=0)
    for _ in range(max_steps):
        action = _select_action(model, obs, env)
        obs, _, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--record", action="store_true")
    p.add_argument("--out", default="videos/mario_1-1.gif")
    args = p.parse_args()
    if args.record:
        path = record_episode(args.checkpoint, args.out)
        print(f"recorded -> {path}")
    else:
        play_live(args.checkpoint)


if __name__ == "__main__":
    main()
```

Note: `env.render()` must return an rgb array. If the bridged env defaults to a window, construct it with render mode rgb (record path) — record uses the returned frames; live uses the window. If render returns `None`, set the underlying env's `render_mode="rgb_array"` in `make_mario_env` and record from that; document the working approach in `stages/03_ppo/README.md`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_play.py -v`
Expected: PASS; a small gif is written to a temp dir.

- [ ] **Step 5: Commit**

```bash
git add src/play.py tests/test_play.py
git commit -m "feat: play.py - live render + record gif/mp4"
git push
```

---

## Task 7: Stage 02 — DQN training run

**Files:**
- Create: `stages/02_dqn/README.md`

- [ ] **Step 1: Short verification run (does it learn anything?)**

Run:
```bash
python src/train.py --algo dqn --timesteps 50000 --out checkpoints/dqn_smoke
```
Expected: completes; `checkpoints/dqn_smoke.zip` exists. Watch TensorBoard:
```bash
tensorboard --logdir tb_logs
```
Expected: `rollout/ep_rew_mean` trends upward (even slightly) over the run — confirms the pipeline learns before a long run.

- [ ] **Step 2: Longer DQN run (the Stage 02 milestone)**

Run:
```bash
python src/train.py --algo dqn --timesteps 1000000 --out checkpoints/dqn_mario
```
Expected (milestone): the agent consistently moves right and makes progress (rising `x_pos` / episode reward). DQN may not reliably clear 1-1 — that motivates PPO in Stage 03.

- [ ] **Step 3: Watch it play**

Run:
```bash
python src/play.py --checkpoint checkpoints/dqn_mario.zip
```
Expected: Mario advances meaningfully further than the random baseline.

- [ ] **Step 4: Write `stages/02_dqn/README.md`**

```markdown
# Stage 02 — DQN (value-based learning)

**Goal:** train a value-based agent and learn what a Q-function, replay
buffer, target network, and epsilon-greedy exploration do.

DQN learns Q(s,a): the expected return of taking action a in state s. It
stores transitions in a replay buffer and trains a CNN to predict Q-values,
using a slowly-updated target network for stability and epsilon-greedy
exploration to keep trying new actions. See README section "Value-based
learning: DQN".

## What to expect
The agent learns to move right and survive, but value-based methods are
sample-inefficient here and may not reliably reach the flag. That is the
motivation for PPO in Stage 03.

## Run it
\`\`\`bash
python src/train.py --algo dqn --timesteps 1000000 --out checkpoints/dqn_mario
python src/play.py --checkpoint checkpoints/dqn_mario.zip
\`\`\`
```

- [ ] **Step 5: Commit**

```bash
git add stages/02_dqn/README.md
git commit -m "docs: Stage 02 - DQN training run + explainer"
git push
```

---

## Task 8: Stage 03 — PPO agent that beats 1-1 (primary success)

**Files:**
- Create: `stages/03_ppo/README.md`
- Modify: `README.md` (embed the win GIF + quickstart)

- [ ] **Step 1: Short PPO verification run**

Run:
```bash
python src/train.py --algo ppo --timesteps 100000 --out checkpoints/ppo_smoke
```
Expected: completes using parallel envs; `rollout/ep_rew_mean` in TensorBoard trends up.

- [ ] **Step 2: Full PPO training run (THE milestone)**

Run:
```bash
python src/train.py --algo ppo --timesteps 4000000 --out checkpoints/ppo_mario
```
Expected: over training, episode reward and `x_pos` climb until episodes end with `flag_get=True`. On M1 Max with 8 parallel envs this is typically a few hours. If it plateaus before the flag, that is expected pre-tuning and is addressed in the (separate) Stage 04 plan — but PPO with these defaults commonly clears 1-1.

- [ ] **Step 3: Confirm the win and record the artifact**

Run:
```bash
python src/play.py --checkpoint checkpoints/ppo_mario.zip --record --out videos/mario_1-1.gif
```
Expected: the recorded gif shows Mario reaching the flagpole. **This is the primary success criterion.** Save the winning model:
```bash
cp checkpoints/ppo_mario.zip checkpoints/mario_1-1_best.zip
git add -f checkpoints/mario_1-1_best.zip videos/mario_1-1.gif
```

- [ ] **Step 4: Write `stages/03_ppo/README.md`**

```markdown
# Stage 03 — PPO (policy-gradient learning) 🎯

**Goal:** train a policy-gradient agent that reaches the flagpole on 1-1.

PPO directly optimizes the policy (a CNN mapping screen → action
probabilities) using the advantage estimate (how much better an action was
than expected), with a clipped objective that prevents destructively large
updates. It runs many emulators in parallel for fast, stable learning. See
README section "Policy-gradient learning: PPO".

## Result
The agent reaches the flagpole on World 1-1. See `videos/mario_1-1.gif`.

## Run it
\`\`\`bash
python src/train.py --algo ppo --timesteps 4000000 --out checkpoints/ppo_mario
python src/play.py --checkpoint checkpoints/ppo_mario.zip --record
\`\`\`
```

- [ ] **Step 5: Fill in the README tutorial body**

Expand `README.md` to the full outline from the spec (sections 1–11): what the project is + embedded `videos/mario_1-1.gif`, RL in 5 minutes, why Mario is hard, turning Mario into an RL problem (the preprocessing pipeline), DQN, PPO, reading TensorBoard results, reward shaping teaser (links to future Stage 04), watching your agent play (live vs record), reproduce-it commands, and references (Sutton & Barto; SB3 docs; gym-super-mario-bros). Each `stages/NN_*/README.md` links into the matching section.

- [ ] **Step 6: Commit**

```bash
git add README.md stages/03_ppo/README.md
git commit -m "feat: Stage 03 - PPO agent beats 1-1 + full README tutorial"
git push
```

---

## Self-Review Notes

- **Spec coverage:** stack (Task 1–2), preprocessing pipeline (Task 4), repo structure (Tasks 1–8), README tutorial outline (Task 8 Step 5 + per-stage READMEs), curriculum Stages 00–03 (Tasks 2,3,7,8), watch-the-agent live+record (Task 6), verification/smoke tests (Tasks 2,4,5,6,7,8), MIT license (Task 1). Stage 04 tuning is intentionally a separate future plan (noted in spec).
- **Determinism caveat** from the spec is honored via `seed=` in tests/play; full reproducibility is not asserted (emulator + parallelism), matching the spec.
- **Risk handling:** the env-stack install risk is front-loaded in Task 1 with explicit fallbacks and recorded in Stage 00.
```
