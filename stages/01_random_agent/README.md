# Stage 01 — Random Agent Baseline

**Goal:** understand the RL loop and the reward signal with zero learning.

A random policy almost never reaches the flag. On our runs it gets to
`x_pos ≈ 600–720` out of the ~3160 needed for the World 1-1 flagpole — about
20% of the level, purely by luck — and `flag_get` stays `False`.

The Mario reward combines rightward progress, a time penalty, and a death
penalty, so "good" behavior means *moving right and staying alive*. This
baseline is the bar every trained agent must beat. See the top-level README
section "RL in 5 minutes".

## What you learned

The loop is: `obs, info = env.reset()` then repeatedly
`obs, reward, terminated, truncated, info = env.step(action)`. A *policy* is
just the rule that picks `action` from `obs`. Here the policy is random;
learning (Stages 02–03) replaces it with a neural network trained to
maximize cumulative reward.

## Run it

```bash
source venv/bin/activate
python stages/01_random_agent/run.py
```
