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

# Torch device. "auto" -> CPU on this machine (SB3 only auto-selects CUDA).
# Set to "mps" to experiment with the Apple GPU for the CNN.
DEVICE = "auto"
