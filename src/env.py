"""Super Mario Bros environment factory.

One place that knows how to construct the Mario env and apply the
preprocessing wrapper stack. The Mario packages (gym-super-mario-bros,
nes-py) only speak the legacy ``gym`` API, so this module bridges them to
the modern Gymnasium API that Stable-Baselines3 expects:

    nes-py emulator (legacy gym, old 4-tuple API)
        -> apply_api_compatibility=True   (gym 0.26 new 5-tuple API)
        -> ResetCompat                    (swallow seed/options for nes-py)
        -> shimmy GymV26CompatibilityV0   (a real Gymnasium env)
        -> [optional] preprocessing wrappers
        -> Stable-Baselines3

See ``stages/00_setup/README.md`` for why each piece is needed.
"""
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    MaxAndSkipObservation,
    ResizeObservation,
)
from nes_py.wrappers import JoypadSpace
from shimmy import GymV26CompatibilityV0


class ResetCompat(gym.Wrapper):
    """nes-py's ``reset()`` predates the ``seed``/``options`` kwargs that
    shimmy forwards. The NES emulator is deterministic and has no seedable
    RNG, so we accept and drop those kwargs."""

    def reset(self, *, seed=None, options=None):
        return self.env.reset()


def make_mario_env(world_stage="SuperMarioBros-1-1-v0", preprocess=True,
                   render_mode="rgb_array"):
    """Build a Gymnasium-compatible Super Mario Bros environment.

    Args:
        world_stage: registered Mario env id (e.g. ``SuperMarioBros-1-1-v0``).
        preprocess: if True, apply the CNN preprocessing wrapper stack
            (grayscale / resize / frame-skip / frame-stack). If False,
            return raw 240x256x3 RGB frames (useful for rendering/inspection).
        render_mode: ``"rgb_array"`` (default) returns frames from
            ``render()`` — used for training and for recording videos.
            ``"human"`` opens a real-time NES window — used to watch live.
    """
    legacy = gym_super_mario_bros.make(
        world_stage,
        apply_api_compatibility=True,  # old 4-tuple env -> gym 0.26 new API
        render_mode=render_mode,
    )
    legacy = JoypadSpace(legacy, SIMPLE_MOVEMENT)  # reduce action space to 7
    legacy = ResetCompat(legacy)
    env = GymV26CompatibilityV0(env=legacy)  # -> Gymnasium API
    if preprocess:
        env = apply_preprocessing(env)
    return env


def apply_preprocessing(env, skip=4, size=84, stack=4):
    """Make raw NES frames tractable for a CNN policy.

    skip+max-pool frames -> grayscale -> resize to ``size`` x ``size`` ->
    stack ``stack`` frames so motion is observable. Output is a
    channels-first ``(stack, size, size)`` uint8 tensor, the layout SB3's
    ``CnnPolicy`` expects.
    """
    env = MaxAndSkipObservation(env, skip=skip)  # act every `skip` frames
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (size, size))
    env = FrameStackObservation(env, stack_size=stack)  # -> (stack, size, size)
    return env
