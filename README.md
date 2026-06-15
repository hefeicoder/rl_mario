# RL Mario — Teaching an Agent to Beat Super Mario Bros

A self-teaching reinforcement-learning project: train an agent to clear
**Super Mario Bros (NES) World 1-1**, built as a documented, stage-by-stage
curriculum. Library-first (Stable-Baselines3 on Gymnasium), open source, and
written to be read as a tutorial on *how RL actually works* — not just code.

> **Status:** full training/inference pipeline complete and tested. Training is
> reproducible end-to-end; drop your winning run's GIF into the badge below
> once you've cleared 1-1.

<!-- After your first win: python src/play.py --checkpoint checkpoints/ppo_mario.zip --deterministic --record -->
<!-- ![Mario clears World 1-1](videos/mario_1-1.gif) -->

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [RL in 5 minutes](#2-rl-in-5-minutes)
3. [Why Mario is hard for RL](#3-why-mario-is-hard-for-rl)
4. [Turning Mario into an RL problem](#4-turning-mario-into-an-rl-problem)
5. [Value-based learning: DQN](#5-value-based-learning-dqn)
6. [Policy-gradient learning: PPO](#6-policy-gradient-learning-ppo)
7. [Training & reading the results](#7-training--reading-the-results)
8. [Reward shaping & tuning](#8-reward-shaping--tuning)
9. [Watching your agent play](#9-watching-your-agent-play)
10. [Reproduce it yourself](#10-reproduce-it-yourself)
11. [References & further reading](#11-references--further-reading)

---

## 1. What this project is

The goal is to **learn reinforcement learning by doing the full loop**: take a
hard, real task (beat NES Mario), turn it into something an RL algorithm can
chew on, train an agent, and watch it go from random flailing to clearing the
level. We use a maintained RL library (Stable-Baselines3) so we spend our
effort on *understanding and tuning* rather than reimplementing algorithms.

The project is organized as a curriculum under `stages/`, each with its own
short explainer:

| Stage | What it teaches | Folder |
|-------|-----------------|--------|
| 00 Setup | Getting the NES env onto a modern RL stack | `stages/00_setup/` |
| 01 Random agent | The RL loop and the reward signal | `stages/01_random_agent/` |
| 02 DQN | Value-based learning | `stages/02_dqn/` |
| 03 PPO | Policy-gradient learning (clears 1-1) 🎯 | `stages/03_ppo/` |

**Quickstart:**

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # 6 tests, all green
python stages/01_random_agent/run.py        # see the baseline
python src/train.py --algo ppo --timesteps 4000000 --out checkpoints/ppo_mario
```

---

## 2. RL in 5 minutes

Reinforcement learning is learning by trial and error. An **agent** interacts
with an **environment** in a loop:

```
        ┌───────────── action a_t ─────────────┐
        │                                       ▼
   ┌─────────┐                            ┌─────────────┐
   │  Agent  │                            │ Environment │
   └─────────┘                            └─────────────┘
        ▲                                       │
        └──── observation s_t+1, reward r_t ────┘
```

- **State / observation** `s` — what the agent sees (here: the screen).
- **Action** `a` — what it can do (here: presses like *right*, *right+jump*).
- **Reward** `r` — a scalar score for the last action (here: mostly "did you
  move right and stay alive?").
- **Policy** `π(a|s)` — the agent's strategy: given a state, which action.

The agent's objective is to maximize **cumulative discounted reward**
`R = Σ γᵗ rₜ`, where the discount `γ` (we use 0.9) makes near-term reward worth
more than far-future reward. Learning = adjusting the policy so the actions it
picks lead to more total reward. Two big families of methods do this:

- **Value-based** (DQN): learn how good each action is, then act greedily.
- **Policy-gradient** (PPO): adjust the policy directly toward higher reward.

We try both — Stages 02 and 03.

---

## 3. Why Mario is hard for RL

Mario looks simple to a human but is genuinely hard for an RL agent:

- **Huge observation space.** The raw frame is 240×256×3 ≈ 184k numbers, and it
  changes every frame. The agent has to learn to *see* from pixels.
- **Sparse, delayed reward.** Clearing the level is hundreds of correct actions
  in a row. The big payoff (the flag) comes only at the very end, long after the
  decisions that earned it.
- **Credit assignment.** When Mario dies, *which* of the last 200 actions was
  the mistake? The agent has to figure out which actions deserve blame or credit.
- **Exploration.** Random play almost never reaches the flag (see Stage 01: it
  gets ~20% of the way by luck), so the agent rarely sees the winning signal to
  learn from.

Most of the engineering below exists to make these problems tractable.

---

## 4. Turning Mario into an RL problem

The raw NES env is too big and too raw to learn from directly. We wrap it in a
pipeline (all in `src/env.py`) that is standard for pixel-based RL:

| Step | Why |
|------|-----|
| **Reduced action set** (`SIMPLE_MOVEMENT`, 7 actions) | The NES has 256 button combos; Mario only needs a handful (right, right+jump, …). Smaller action space = far easier to learn. |
| **Grayscale** | Color doesn't help Mario; dropping it cuts the input 3×. |
| **Resize → 84×84** | Smaller image = smaller, faster CNN. 84×84 is the classic Atari-DQN size. |
| **Frame-skip = 4** (max-pool) | Act once every 4 frames. The agent reacts at a sensible rate and trains ~4× faster; max-pooling over the skipped frames avoids NES sprite flicker. |
| **Frame-stack = 4** | A single frame can't show motion (which way is Mario moving? how fast?). Stacking the last 4 frames gives the network velocity. |

The result is a compact `(4, 84, 84)` stack of grayscale frames — the input to
our CNN policy.

**The reward** comes from the Mario environment itself and combines three
things: `+` for moving right (progress), `−` a small penalty each step (finish
faster), and `−` a penalty for dying. So "good" behavior is precisely *move
right, stay alive, reach the flag*.

**One stack note worth knowing:** `gym-super-mario-bros` and `nes-py` only speak
the legacy `gym` API and break on numpy 2.x. We bridge them to modern Gymnasium
with `shimmy` and pin `numpy<2`. The gory details (and why each line exists) are
in [`stages/00_setup/README.md`](stages/00_setup/README.md) — a good read if you
ever have to revive an unmaintained env.

---

## 5. Value-based learning: DQN

**Deep Q-Network** learns a function **Q(s, a)** = "expected total future reward
if I take action `a` in state `s`, then play well after." A CNN outputs one
Q-value per action; the policy is just "pick the action with the highest Q."

Three tricks make training a neural net on this stable:

- **Replay buffer.** Store transitions `(s, a, r, s′)` and train on random
  samples. Without it, consecutive frames are nearly identical and the net
  overfits to whatever it just saw.
- **Target network.** Compute the learning target with a *frozen copy* of the
  network that updates slowly. Otherwise the target moves every step and
  training oscillates.
- **ε-greedy exploration.** Act randomly with probability ε, decaying from 100%
  to ~2%, so the agent keeps discovering new states early on.

DQN is foundational and great for intuition, but **sample-inefficient** on a
long, sparse task like Mario — it learns to move right and survive but often
won't reliably reach the flag in a reasonable budget. That's the motivation for
PPO. Full runbook: [`stages/02_dqn/README.md`](stages/02_dqn/README.md).

---

## 6. Policy-gradient learning: PPO

**Proximal Policy Optimization** skips value-then-greedy and optimizes the
**policy directly**: the CNN outputs a probability over the 7 actions, and we
nudge those probabilities toward actions that did better than expected.

- **Advantage** `A = (actual return) − (value estimate)`: did this action beat
  the baseline? Push its probability up if yes, down if no.
- **Clipped objective.** PPO limits how far the policy can move in one update.
  This single idea is why PPO is so stable — it avoids the catastrophic, too-big
  updates that wreck vanilla policy gradients.
- **Parallel envs.** We run 8 emulators at once (`N_ENVS=8`) for diverse,
  decorrelated experience and fast wall-clock training.

PPO is the workhorse that actually clears 1-1: more stable and sample-efficient
here than DQN. Full runbook: [`stages/03_ppo/README.md`](stages/03_ppo/README.md).

---

## 7. Training & reading the results

Train, then watch the curves live:

```bash
python src/train.py --algo ppo --timesteps 4000000 --out checkpoints/ppo_mario
tensorboard --logdir tb_logs    # http://localhost:6006
```

This project adds two Mario-specific TensorBoard curves (via
`MarioStatsCallback` in `src/train.py`) so progress toward *winning* is legible,
not just reward:

- **`mario/flag_rate`** — fraction of recent episodes that reached the flag.
  **This is the win rate.** It's 0 for a long time, then climbs. Target: 1.0.
- **`mario/max_x_pos`** — how far right the agent gets, averaged over recent
  episodes. Climbs from the ~700 random baseline toward **3160** (the flagpole).
  This usually moves well before `flag_rate` does.
- **`rollout/ep_rew_mean`** — overall reward; should trend up.

**What "learning" looks like:** flat curves for the first several hundred
thousand steps (Mario is a hard exploration problem), then `max_x_pos` creeps
right, then `flag_rate` lifts off 0. **Common failure modes:** stuck behind the
first pipe/gap (needs more exploration — raise `ent_coef`), or reward rising
while `max_x_pos` stalls (agent farming a local reward instead of progressing).

**Performance note:** measured ~207 env-steps/s with 8 envs on an Apple M1 Max,
so 4M steps ≈ 5.4 hours. Training is **emulator-bound, not compute-bound** —
the NES emulator on CPU is the bottleneck, so a GPU/MPS doesn't help here. CPU
is the right choice.

---

## 8. Reward shaping & tuning

The built-in Mario reward (progress − time − death) is enough to learn 1-1, but
getting from "moves right sometimes" to "beats it reliably" is a tuning
exercise — and a great way to *feel* how RL responds to its knobs. Levers, in
rough order of impact:

- **Train longer.** The single biggest lever; many "failures" just needed more
  steps.
- **`ent_coef`** (exploration). Raise it if the agent gets stuck early; lower it
  once it's reliably progressing, to sharpen the policy.
- **`learning_rate` / `n_steps` / `clip_range`.** Stability vs. speed trade-offs.
- **Reward shaping.** Optionally add your own bonuses (e.g., for grabbing the
  flag) by wrapping the env — a clean place to experiment.

Systematic tuning and reward-shaping experiments are a planned **Stage 04**.

---

## 9. Watching your agent play

Two ways, both in `src/play.py`:

```bash
# Live: open a real-time NES window (plays 3 episodes by default)
python src/play.py --checkpoint checkpoints/ppo_mario.zip

# Record: save a GIF (rock-solid regardless of windowing)
python src/play.py --checkpoint checkpoints/ppo_mario.zip --record --out videos/mario_1-1.gif
```

**Stochastic vs. deterministic actions.** By default `play.py` *samples* from
the policy, exactly like training does — so each episode varies and an early,
weak agent explores further than its brittle greedy action would. Pass
`--deterministic` to always take the argmax action; that's best for a
*well-trained* agent (crisp, optimal play), but on the deterministic NES
emulator a weak greedy policy produces the **identical run every episode** and
often gets stuck in a death-loop. Rule of thumb: stochastic while learning,
`--deterministic` for the final winner.

```bash
python src/play.py --checkpoint checkpoints/ppo_mario.zip --deterministic   # final agent
python src/play.py --checkpoint checkpoints/ppo_mario.zip --episodes 5       # watch more runs
```

You can point `--checkpoint` at **mid-training checkpoints** too, to watch the
agent improve over time — random flailing → moves right → clears the level. The
recorded GIF is what goes at the top of this README once you've won.

---

## 10. Reproduce it yourself

```bash
# 1. Environment (Apple Silicon / Python 3.12)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Verify the stack (should print "6 passed")
pytest -q

# 3. See the random baseline (gets ~20% of the level)
python stages/01_random_agent/run.py

# 4. Train PPO to beat 1-1 (~5.4 h on M1 Max); watch tb_logs in TensorBoard
python src/train.py --algo ppo --timesteps 4000000 --out checkpoints/ppo_mario

# 5. Watch / record the win (--deterministic = the trained agent's best play)
python src/play.py --checkpoint checkpoints/ppo_mario.zip --deterministic --record
```

Exact pinned versions are in `requirements.txt`. Note: the NES emulator plus
parallel envs mean runs aren't bit-for-bit reproducible, but learning curves are
comparable across runs.

---

## 11. References & further reading

- **Sutton & Barto, *Reinforcement Learning: An Introduction*** — the canonical
  textbook (free online): http://incompleteideas.net/book/the-book.html
- **DQN** — Mnih et al., *Human-level control through deep reinforcement
  learning*, Nature 2015.
- **PPO** — Schulman et al., *Proximal Policy Optimization Algorithms*, 2017:
  https://arxiv.org/abs/1707.06347
- **Stable-Baselines3 docs** — https://stable-baselines3.readthedocs.io
- **Gymnasium** — https://gymnasium.farama.org
- **gym-super-mario-bros** — https://github.com/Kautenja/gym-super-mario-bros

---

*Open source under the [MIT License](LICENSE). Built as a self-teaching project —
issues and improvements welcome.*
