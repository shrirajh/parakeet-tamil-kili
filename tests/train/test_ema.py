"""kili.train.ema: the average survives a checkpoint round trip (CPU; needs torch and lightning).  pytest tests/train"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from kili.train import ema  # noqa: E402


class _Trainer:  # minimal trainer stand-in
    def __init__(self, step):
        self.global_step, self.max_steps, self.accumulate_grad_batches, self.is_global_zero = step, 100, 1, True


def _run(cb, m, steps):
    for s in steps:
        with torch.no_grad():
            m.weight.add_(1.0)
        cb.on_train_batch_end(_Trainer(s), m, None, None, 0)


def test_resume_continues_average():
    torch.manual_seed(0)
    a, b = torch.nn.Linear(4, 3), torch.nn.Linear(4, 3)
    b.load_state_dict(a.state_dict())
    full = ema.callback(start=0.5, every=1)
    _run(full, a, range(50, 80))
    part = ema.callback(start=0.5, every=1)
    _run(part, b, range(50, 65))
    st = {k: ([t.clone() for t in v] if k == "ema" else v) for k, v in part.state_dict().items()}
    resumed = ema.callback(start=0.5, every=1)
    resumed.load_state_dict(st)
    _run(resumed, b, range(65, 80))
    assert resumed.n == full.n
    for x, y in zip(full.ema, resumed.ema):
        assert torch.allclose(x, y)


def test_earlier_window_discarded():
    m = torch.nn.Linear(4, 3)
    old = ema.callback(start_step=10, every=1)
    _run(old, m, range(10, 20))
    st = old.state_dict()
    new = ema.callback(start_step=60, every=1)  # a later decay window resuming a checkpoint with the old average
    new.load_state_dict(st)
    _run(new, m, range(60, 61))
    assert new.n == 0  # restarted from the current weights


def test_empty_state_before_start():
    cb = ema.callback(start=0.5)
    assert cb.state_dict() == {}
    cb.load_state_dict({})
    assert cb.ema is None
    assert cb.apply(torch.nn.Linear(2, 2)) is False
