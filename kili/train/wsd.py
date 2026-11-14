"""Warm-up / stable / decay (WSD) learning-rate schedule with decay forks: 1-, 2-, ... N-epoch models from one run.

The lr warms up, stays flat, and decays only at the end, with the 1 - sqrt cool-down (Haegele et al. 2024, "Scaling
Laws and Compute-Optimal Training beyond Fixed Training Durations"). Shortly before the end of epoch k (k < N) the run
forks: a branch takes the main run's checkpoint at the fork and decays over the last `decay_frac` of epoch k, ending
exactly at k epochs with a finished (decayed + EMA-averaged) model; the main run continues from the fork at the flat lr.
Every model gets the same decay length; only the flat part differs. (Under a cosine schedule a mid-run checkpoint,
taken at a high lr and without an average, understates what a k-epoch run gives.)

  epoch E = total_steps / epochs, decay D = round(decay_frac x E)
  fork k  F_k = round(k E) - D                                  (k = 1 .. epochs - 1)
  segment "main" (stop k): steps .. F_k, flat lr               segment "main": .. total, decay from total - D
  segment "b<k>": resumes the main run's checkpoint at F_k, decays F_k -> round(k E)

  lr(s) = peak (s + 1) / (W + 1) for s <= W (warm-up), peak until the decay starts, then
          min_lr + (peak - min_lr) (1 - sqrt(x)), x = progress through the decay.

Extending a finished run: only the decayed checkpoint exists, so the continuation re-warms: from step rewarm_start the
lr ramps from min_lr back to the peak over rewarm_steps (then flat, forks and decay as planned for the new total).

Run 10 (release model = the 5-epoch fork): 3 epochs of 7,206 steps (21,619 total), warm-up 100, peak 1.49e-4, min_lr
1e-6, decay 721 steps; then extended to 6 epochs (43,238 steps) with a 300-step re-warm at step 21,619; the 5-epoch
branch forks at step 35,311 and decays to 36,032. EMA over each model's decay window.

The scheduler's state is only its step counter, so a branch resuming the main run's checkpoint does not inherit the
main run's decay start.

  python -c "from kili.train.wsd import plan; print(plan(43238, 6, 0.1))"
"""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import _LRScheduler


class WSD(_LRScheduler):
    def __init__(self, optimizer, *, warmup_steps: int = 0, decay_start: int, decay_end: int, min_lr: float = 0.0,
                 rewarm_start: int = -1, rewarm_steps: int = 0, last_epoch: int = -1, **_ignored):
        assert decay_end > decay_start >= 0, (decay_start, decay_end)
        self.warmup_steps, self.decay_start, self.decay_end = int(warmup_steps), int(decay_start), int(decay_end)
        self.min_lr = float(min_lr)
        self.rewarm_start, self.rewarm_steps = int(rewarm_start), int(rewarm_steps)
        super().__init__(optimizer, last_epoch)

    def factor(self, step: int, peak: float) -> float:
        """The lr at optimizer step `step` for a peak lr `peak`."""
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return peak * (step + 1) / (self.warmup_steps + 1)
        if self.rewarm_steps > 0 and self.rewarm_start <= step < min(self.rewarm_start + self.rewarm_steps,
                                                                      self.decay_start):
            return self.min_lr + (peak - self.min_lr) * (step - self.rewarm_start + 1) / (self.rewarm_steps + 1)
        if step < self.decay_start:
            return peak
        x = min(1.0, (step - self.decay_start) / (self.decay_end - self.decay_start))
        return self.min_lr + (peak - self.min_lr) * (1.0 - math.sqrt(x))

    def get_lr(self):
        return [self.factor(self.last_epoch, b) for b in self.base_lrs]

    def state_dict(self):
        return {"last_epoch": self.last_epoch, "_step_count": self._step_count}

    def load_state_dict(self, state):
        self.last_epoch = int(state["last_epoch"])
        self._step_count = int(state.get("_step_count", self.last_epoch + 1))
        # the optimizer's lr was restored from the checkpoint (the saving run's schedule): set this schedule's value
        for g, lr in zip(self.optimizer.param_groups, self.get_lr()):
            g["lr"] = lr
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]


def register() -> None:
    """Make "WSD" available to NeMo's optim.sched config."""
    from dataclasses import dataclass

    from nemo.core.config.schedulers import SchedulerParams
    from nemo.core.optim.lr_scheduler import AVAILABLE_SCHEDULERS, register_scheduler

    @dataclass
    class WSDParams(SchedulerParams):
        warmup_steps: int = 0
        decay_start: int = 0
        decay_end: int = 1
        min_lr: float = 0.0
        rewarm_start: int = -1
        rewarm_steps: int = 0

    if "WSD" not in AVAILABLE_SCHEDULERS:
        register_scheduler("WSD", WSD, WSDParams)


def plan(total: int, epochs: int, decay_frac: float) -> dict:
    """Epoch ends, decay length and fork steps for a run of `total` optimizer steps."""
    E = total / epochs
    D = max(1, round(decay_frac * E))
    ends = [round(k * E) for k in range(1, epochs + 1)]
    ends[-1] = total
    forks = [e - D for e in ends[:-1]]
    return {"total": total, "epoch_steps": E, "decay_steps": D, "ends": ends, "forks": forks}


def segment_bounds(p: dict, segment: str, stop=None) -> dict:
    """-> {max_steps, decay_start, decay_end} for one segment ("main", optionally stopping at fork `stop`, or "b<k>")."""
    if segment == "main":
        ms = p["forks"][int(stop) - 1] if stop else p["total"]
        return {"max_steps": ms, "decay_start": p["total"] - p["decay_steps"], "decay_end": p["total"]}
    assert segment.startswith("b"), f"segment must be main or b<k>, got {segment}"
    k = int(segment[1:])
    assert 1 <= k <= len(p["forks"]), f"no fork {k} (epochs {len(p['forks']) + 1})"
    return {"max_steps": p["ends"][k - 1], "decay_start": p["forks"][k - 1], "decay_end": p["ends"][k - 1]}


def apply(cfg) -> dict:
    """Rewrite an OmegaConf training config in place for one segment: trainer.max_steps, optim.sched (WSD) and, when
    an ema section is enabled, ema.start_step = the decay start. Reads
      wsd: {epochs, decay_frac, segment: main | b<k>, stop: k | null, total_steps: N | null,
            rewarm_start: -1, rewarm_steps: 0}
    (total_steps defaults to trainer.max_steps), optim.sched.warmup_steps and optim.sched.min_lr."""
    from omegaconf import open_dict

    w = cfg.wsd
    total = int(w.get("total_steps") or cfg.trainer.max_steps)
    p = plan(total, int(w.get("epochs", 1)), float(w.get("decay_frac", 0.1)))
    seg = str(w.get("segment", "main"))
    b = segment_bounds(p, seg, w.get("stop"))
    with open_dict(cfg):
        cfg.trainer.max_steps = b["max_steps"]
        sched = cfg.optim.get("sched") or {}
        cfg.optim.sched = {"name": "WSD", "warmup_steps": int(sched.get("warmup_steps", 0) or 0),
                           "decay_start": b["decay_start"], "decay_end": b["decay_end"],
                           "min_lr": float(sched.get("min_lr", 0.0) or 0.0),
                           "rewarm_start": int(w.get("rewarm_start", -1)), "rewarm_steps": int(w.get("rewarm_steps", 0))}
        if (cfg.get("ema", {}) or {}).get("enabled", False):
            cfg.ema.start_step = b["decay_start"]
    register()
    return {**p, **b, "segment": seg}
