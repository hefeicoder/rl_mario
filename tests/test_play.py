import os

from src.play import record_episode


def test_record_episode_writes_gif(tmp_path):
    out = tmp_path / "demo.gif"
    # random policy (checkpoint=None) keeps the test fast and dependency-free
    record_episode(checkpoint=None, out_path=str(out), max_steps=50)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
