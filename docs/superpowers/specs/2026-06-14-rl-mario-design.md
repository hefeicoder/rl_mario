# RL Mario — Design Spec

**Date:** 2026-06-14
**Repo:** https://github.com/hefeicoder/rl_mario (open source, MIT)
**Status:** Approved design, ready for implementation planning

## Goal

Self-teach how reinforcement learning works by training an agent to play
Super Mario Bros (NES). The project is **library-first** (learn by using and
tuning a maintained RL library) and **documentation-first** (the README is a
source-of-truth tutorial on RL methodology for Mario).

**Primary success:** an agent that reaches the flagpole on **World 1-1**.
**Polish goal:** beats 1-1 reliably (>80% of runs).
**Stretch goal (documented, not required):** beat the full game (all 32 levels).

## Audience & Constraints

- **Learner profile:** library-first; wants to understand *how RL works*, not
  reimplement algorithms from scratch.
- **Hardware:** Apple M1 Max, 10 cores, 64 GB RAM. Emulator is CPU-bound; the
  CNN is small. Favor **vectorized CPU envs** for throughput; use PyTorch
  **MPS** (Apple GPU) where it helps.
- **Python:** 3.12.12 in a dedicated virtualenv.

## Stack

- **Environment:** `gym-super-mario-bros` + `nes-py` (NES emulator), wrapped to
  the modern **Gymnasium** API via `shimmy` if needed.
- **RL library:** **Stable-Baselines3** (provides both DQN and PPO).
- **Backend:** PyTorch (MPS where beneficial) + vectorized CPU envs.
- **Logging:** TensorBoard learning curves; model checkpoints; recorded
  gameplay videos/GIFs.

### Risk: the Mario env stack

`gym-super-mario-bros` + `nes-py` are lightly maintained, compile C++, and
historically target the old Gym API. Installing on Python 3.12 and bridging to
Stable-Baselines3/Gymnasium may need a compatibility shim. **De-risk this in
Stage 00 before any learning work.** If the stack proves unworkable on 3.12,
fallbacks (in priority order) are: pin a compatible Python via pyenv; use
`shimmy`'s Gym→Gymnasium conversion; as a last resort, an alternative NES Mario
env. Decision recorded in Stage 00's README.

### Preprocessing pipeline (each piece explained in the README)

grayscale → resize to 84×84 → frame-skip (act every 4 frames) → frame-stack
(4 frames, so motion is perceivable) → reduced action set (movement/jump
oriented, e.g. SIMPLE/RIGHT-only movements rather than all button combos).

## Repository Structure

```
rl_mario/
├── README.md              # THE tutorial / source of truth
├── LICENSE                # MIT
├── requirements.txt       # pinned deps for reproducibility
├── .gitignore             # venv, checkpoints, videos, tensorboard logs
├── src/
│   ├── env.py             # Mario env factory + preprocessing wrappers
│   ├── train.py           # unified entrypoint (--algo dqn|ppo)
│   ├── play.py            # load checkpoint -> watch live or --record video
│   └── config.py          # hyperparameters per algorithm
├── stages/                # staged curriculum, one folder per milestone
│   ├── 00_setup/          # install + render check
│   ├── 01_random_agent/   # baseline; understand env & rewards
│   ├── 02_dqn/            # value-based agent
│   ├── 03_ppo/            # policy-gradient agent that beats 1-1
│   └── 04_tuning/         # reward shaping & hyperparameter experiments
│       └── README.md      # each stage has its own short explainer
├── checkpoints/           # gitignored, except a final "beats 1-1" model
└── videos/                # gitignored
```

Top-level `README.md` is the full tutorial; each `stages/NN_*/README.md` is a
focused explainer that links back to the relevant README section.

## README Tutorial Outline (source of truth)

1. **What this project is** — goal, results (GIF of agent beating 1-1), quickstart.
2. **RL in 5 minutes** — agent/environment/state/action/reward/policy, the loop, intuition (light on math).
3. **Why Mario is hard for RL** — sparse/delayed rewards, huge pixel state space, credit assignment.
4. **Turning Mario into an RL problem** — the preprocessing pipeline piece by piece (grayscale, resize, frame-skip, frame-stack, action space, reward function).
5. **Value-based learning: DQN** — Q-values, replay buffer, target networks, ε-greedy exploration; what to expect when you run it.
6. **Policy-gradient learning: PPO** — policy vs. value methods, advantages, why PPO is stable, parallel envs.
7. **Training & reading the results** — TensorBoard curves, what "learning" looks like, common failure modes.
8. **Reward shaping & tuning** — how we got from "moves right sometimes" to "beats 1-1".
9. **Watching your agent play** — live render vs. `--record` video/GIF; watching mid-training checkpoints improve.
10. **Reproduce it yourself** — exact commands, expected wall-clock time on Apple Silicon.
11. **References & further reading** — papers, Sutton & Barto, tutorials we learned from.

## Staged Curriculum (milestones & success criteria)

| Stage | Build | Learn | Done when |
|-------|-------|-------|-----------|
| **00 Setup** | Install stack, render Mario | env stack; de-risk nes-py on Py3.12 | A window shows Mario; random inputs move him |
| **01 Random agent** | Random-action loop, print rewards | state/action/reward shape, the env loop | Can read reward signal & episode end |
| **02 DQN** | SB3 DQN + preprocessing wrappers | value functions, replay buffer, target nets, ε-greedy | Agent consistently moves right, makes progress |
| **03 PPO** | SB3 PPO + vectorized envs | policy gradients, advantages, parallel rollout | **Agent reaches the flagpole on 1-1** |
| **04 Tuning** | Reward shaping & hyperparameter sweeps | reward design, stability, sample efficiency | Agent beats 1-1 reliably (>80% of runs) |

## Watching the Agent Play

`play.py` supports two modes:

1. **Live render** — load a checkpoint and watch Mario play in a real-time
   `nes-py` SDL window. Best for the live "aha" moment. May be finicky on
   Apple Silicon windowing; the record mode is the reliable fallback.
2. **Record** — `play.py --record` runs an episode and saves an MP4 (+ GIF for
   the README). Rock-solid regardless of windowing. Produces the "beats 1-1"
   artifact embedded in the README.

Mid-training checkpoints can be played to watch the agent improve over time.

## Verification & Reproducibility

- **Smoke test:** build env, run ~100 random steps, assert observation/reward
  shapes — catches install/wrapper breakage early.
- **Short training run** (a few thousand steps) to confirm the pipeline learns
  *anything* before committing to a long run.
- **Seeding:** seed envs/torch where possible; document that emulator +
  parallelism limit exact reproducibility, but learning curves are comparable.
- **"It works" artifact:** committed final checkpoint + recorded video/GIF of
  the agent clearing 1-1, embedded in the README.

## Out of Scope (for now)

- Implementing RL algorithms from scratch (using Stable-Baselines3).
- Beating levels beyond 1-1 as a hard requirement (documented stretch goal).
- Multi-machine / cloud training (single M1 Max is sufficient for 1-1).
