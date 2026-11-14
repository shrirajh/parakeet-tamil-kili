"""kili.train.wsd: run 10's plan, segment bounds and the lr curve (CPU; needs torch).  pytest tests/train"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kili.train import wsd  # noqa: E402


def test_run10_plan():
    p = wsd.plan(43238, 6, 0.1)  # run 10 extended to 6 epochs
    assert p["decay_steps"] == 721
    assert p["forks"] == [6485, 13692, 20898, 28104, 35311]
    assert p["ends"] == [7206, 14413, 21619, 28825, 36032, 43238]
    b5 = wsd.segment_bounds(p, "b5")  # the released model
    assert b5 == {"max_steps": 36032, "decay_start": 35311, "decay_end": 36032}
    assert wsd.segment_bounds(p, "main", stop=5)["max_steps"] == 35311
    assert wsd.plan(21619, 3, 0.1)["forks"] == [6485, 13692]  # the original 3-epoch run


def test_lr_curve():
    peak, mn = 1.49e-4, 1e-6
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=peak)
    s = wsd.WSD(opt, warmup_steps=100, decay_start=35311, decay_end=36032, min_lr=mn, rewarm_start=21619,
                rewarm_steps=300)
    assert s.factor(0, peak) == pytest.approx(peak / 101)
    assert s.factor(100, peak) == pytest.approx(peak)
    assert s.factor(5000, peak) == pytest.approx(peak)
    assert s.factor(21619, peak) == pytest.approx(mn + (peak - mn) / 301)  # re-warm from min_lr
    assert s.factor(21619 + 300, peak) == pytest.approx(peak)
    mid = 35311 + (36032 - 35311) // 4
    x = (mid - 35311) / (36032 - 35311)
    assert s.factor(mid, peak) == pytest.approx(mn + (peak - mn) * (1 - x ** 0.5))  # 1 - sqrt decay
    assert s.factor(36032, peak) == pytest.approx(mn)


def test_state_is_only_the_step():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    main = wsd.WSD(opt, decay_start=900, decay_end=1000)
    for _ in range(500):
        opt.step()
        main.step()
    st = main.state_dict()
    assert set(st) == {"last_epoch", "_step_count"}
    branch = wsd.WSD(opt, decay_start=400, decay_end=600)  # a branch resuming the main run's checkpoint
    branch.load_state_dict(st)
    assert opt.param_groups[0]["lr"] == pytest.approx(branch.factor(500, 1.0))
