# Stage 02 — DQN (value-based learning)

**Goal:** train a value-based agent and learn what a Q-function, replay
buffer, target network, and epsilon-greedy exploration do.

## The idea

DQN learns **Q(s, a)** — the expected total future reward of taking action
`a` in state `s`. A CNN reads the stacked 84×84 frames and outputs one
Q-value per action; the agent acts greedily on the highest Q-value (with some
random exploration). Three ingredients make it stable:

- **Replay buffer** — past transitions `(s, a, r, s')` are stored and sampled
  randomly, so the network doesn't overfit to the most recent, highly
  correlated frames.
- **Target network** — a slowly-updated copy used to compute the learning
  target, so we aren't chasing a moving goalpost every step.
- **ε-greedy exploration** — act randomly a fraction of the time (decaying
  from 100% toward 2%) so the agent keeps discovering new states.

Hyperparameters live in `src/config.py` (`DQN_CONFIG`).

## Run it

```bash
source venv/bin/activate

# Short sanity run first (~a few minutes): confirms it learns *anything*.
python src/train.py --algo dqn --timesteps 50000 --out checkpoints/dqn_smoke

# The Stage 02 run (long — about an hour-plus on M1 Max).
python src/train.py --algo dqn --timesteps 1000000 --out checkpoints/dqn_mario
```

In another terminal, watch it learn:

```bash
tensorboard --logdir tb_logs
# open http://localhost:6006
```

## What to watch on TensorBoard

- `rollout/ep_rew_mean` — should trend upward above the random baseline (~600).
- `mario/max_x_pos` — how far right it gets (1-1 flag is near **3160**).
- `mario/flag_rate` — fraction of episodes that reached the flag (the win rate).
- Note: DQN only *starts* learning after `learning_starts=10000` steps, so the
  first curves are flat — that's expected.

## What to expect

The agent reliably learns to move right and survive, but value-based methods
are sample-inefficient on Mario and may not reliably reach the flag within
1M steps. That gap is exactly why we move to PPO in Stage 03.

## Watch a checkpoint play

```bash
python src/play.py --checkpoint checkpoints/dqn_mario.zip            # live window
python src/play.py --checkpoint checkpoints/dqn_mario.zip --record   # save a gif
```

See the top-level README section "Value-based learning: DQN".
