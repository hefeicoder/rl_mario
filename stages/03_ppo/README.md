# Stage 03 — PPO (policy-gradient learning) 🎯

**Goal:** train a policy-gradient agent that reaches the flagpole on World 1-1.

## The idea

Instead of learning action *values* and acting greedily, PPO learns the
**policy directly**: a CNN maps the stacked frames to a probability
distribution over the 7 actions. It improves the policy using the
**advantage** — how much better an action turned out than the value function
expected — and a **clipped objective** that prevents any single update from
changing the policy too much (the usual cause of RL blowups). It runs many
emulators in parallel (`N_ENVS=8`) to gather diverse experience quickly and
stably.

Why PPO beats DQN here: it is more sample-efficient and far more stable on
sparse, long-horizon tasks like running to the end of a level. Hyperparameters
live in `src/config.py` (`PPO_CONFIG`).

## Run it

```bash
source venv/bin/activate

# Short sanity run (~10 min): confirm the curves move.
python src/train.py --algo ppo --timesteps 100000 --out checkpoints/ppo_smoke

# THE run. ~5.4 hours on M1 Max (measured ~207 env-steps/s, 8 envs, CPU).
python src/train.py --algo ppo --timesteps 4000000 --out checkpoints/ppo_mario
```

Watch it learn in another terminal:

```bash
tensorboard --logdir tb_logs   # http://localhost:6006
```

## What to watch (the win is legible here)

- `mario/flag_rate` — **this is the win rate.** It sits at 0 for a long while,
  then starts ticking up as the agent first stumbles onto the flag. Reaching a
  consistently non-zero (then high) flag_rate is the goal.
- `mario/max_x_pos` — climbs from the ~700 random baseline toward **3160**
  (the flagpole). This usually moves well before flag_rate does.
- `rollout/ep_rew_mean` — overall upward trend.

Expect little visible progress for the first several hundred thousand steps —
that is normal. Mario is a hard exploration problem; the curves often stay flat
and then climb.

## Confirm the win and save the artifact

Once `mario/flag_rate` is reliably above 0:

```bash
# Watch it live
python src/play.py --checkpoint checkpoints/ppo_mario.zip

# Record the winning run for the README
python src/play.py --checkpoint checkpoints/ppo_mario.zip --record --out videos/mario_1-1.gif

# Preserve the winning model (force-add past .gitignore)
cp checkpoints/ppo_mario.zip checkpoints/mario_1-1_best.zip
git add -f checkpoints/mario_1-1_best.zip videos/mario_1-1.gif
git commit -m "result: PPO agent beats World 1-1"
```

## If it plateaus before winning

That is expected pre-tuning and is the subject of Stage 04 (reward shaping &
hyperparameter sweeps — a separate plan). First levers to try: train longer,
raise `ent_coef` for more exploration, or adjust `n_steps`/`learning_rate`.

See the top-level README section "Policy-gradient learning: PPO".
