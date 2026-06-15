# Stage 00 — Setup & Render Check

**Goal:** prove the environment stack works on this machine before any learning.

## What worked (Apple M1 Max, macOS, Python 3.12.12)

The Mario packages are unmaintained and only speak the legacy `gym` API, so
getting them onto a modern Stable-Baselines3 (Gymnasium) stack took three
fixes. All are encoded in `src/env.py`:

| Problem | Fix |
|---------|-----|
| `nes-py` does `uint8 * 1024`, which **overflows on numpy 2.x** (numpy 2.0 tightened scalar promotion) | Pin **`numpy<2`** (we use 1.26.4) |
| `opencv-python` 4.13 (pulled by `sb3[extra]`) requires numpy≥2 | Pin **`opencv-python<4.11`** (4.10.0.84) |
| gym 0.26 wraps the env with new-API wrappers, but the base env returns the old 4-tuple | `gym_super_mario_bros.make(..., apply_api_compatibility=True)` |
| `render()` must return frames for recording | `render_mode="rgb_array"` |
| `shimmy` forwards `reset(seed=…, options=…)`; nes-py's `reset()` rejects them | `ResetCompat` wrapper swallows them (the NES emulator is deterministic) |
| SB3 needs a Gymnasium env; the base is legacy gym | `shimmy.GymV26CompatibilityV0` |

**Key versions** (full list in `requirements.txt`): nes-py 8.2.1,
gym-super-mario-bros 7.4.0, gym 0.26.2, gymnasium 1.2.3, shimmy 2.0.1,
stable-baselines3 2.8.0, torch 2.12.0 (MPS available), numpy 1.26.4,
opencv-python 4.10.0.84.

## What you learned

The Mario env is the *environment* in the RL loop: each step it takes an
action and returns an observation (the screen), a reward, and done flags
(`terminated` / `truncated`). The `info` dict exposes `x_pos`, `flag_get`,
`time`, etc. — these drive the reward and tell us when Mario wins. See the
top-level README section "Turning Mario into an RL problem".

## Run it

```bash
source venv/bin/activate
pytest tests/test_env.py -v
```
