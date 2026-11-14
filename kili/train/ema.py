"""Exponential moving average of the trainable weights over the end of training (Lightning callback).

The average becomes the saved model: a cheap, reliable gain over the last raw weights, like averaging the last few
checkpoints. Run 10 averaged each model's learning-rate decay window only (start_step = the decay start, see
kili.train.wsd), updating every 4 optimizer steps.

- Starts at `start_step` (or start x max_steps); decay chosen so the average spans about half the remaining steps.
- Updated every `every` optimizer steps with the decay compounded (same window as a per-step update): updating a
  2.4 GB fp32 copy every step costs ~40% speed on Parakeet 0.6B.
- fp32 copy of the trainable parameters (device="cpu" keeps it in host memory: same maths).
- Checkpointed with the trainer (callback state), so a resume continues the same average. A restored average from an
  earlier window (e.g. a decay branch or extension resuming a checkpoint that carries a previous decay's average) is
  discarded rather than mixed in.

  ema_cb = callback(start_step=decay_start, every=4); trainer = Trainer(callbacks=[ema_cb, ...]); trainer.fit(model)
  ema_cb.apply(model); model.save_to("final.nemo")
"""
from __future__ import annotations

import torch


def callback(start: float = 0.66, every: int = 4, device=None, start_step: int | None = None):
    """start: fraction of trainer.max_steps where the average starts; start_step (absolute optimizer step) overrides
    it. device: where the average lives (default: each parameter's device)."""
    import lightning.pytorch as pl

    every = max(1, int(every))

    class EMA(pl.Callback):
        def __init__(self):
            self.ema, self.params, self.decay, self.n = None, None, None, 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if (batch_idx + 1) % max(1, trainer.accumulate_grad_batches):
                return  # not an optimizer step
            ms = trainer.max_steps if trainer.max_steps and trainer.max_steps > 0 else 0
            first = int(start_step) if start_step is not None else int(start * ms)
            if not ms or trainer.global_step < first:
                return
            with torch.no_grad():
                if (self.ema is not None and self.params is None
                        and trainer.global_step - self.n * every < first - 2 * every - 2):
                    # restored average from an earlier window: start fresh (a genuine resume implies a start ~first)
                    self.ema, self.n = None, 0
                if self.ema is not None and self.params is None:  # restored from a checkpoint: bind to the model
                    self.params = [p for p in pl_module.parameters() if p.requires_grad]
                    if len(self.params) != len(self.ema) or any(p.shape != e.shape for p, e in zip(self.params, self.ema)):
                        raise RuntimeError("EMA: checkpointed average does not match the model's trainable params")
                    self.ema = [e.to(device or p.device, torch.float32) for p, e in zip(self.params, self.ema)]
                if self.ema is None:
                    self.params = [p for p in pl_module.parameters() if p.requires_grad]
                    self.ema = [p.detach().float().to(device or p.device).clone() for p in self.params]
                    window = max(10, (ms - trainer.global_step) // 2)
                    self.decay = 1.0 - 1.0 / window
                    return
                if trainer.global_step % every:
                    return
                d = self.decay ** every
                torch._foreach_mul_(self.ema, d)
                torch._foreach_add_(self.ema, [p.detach().float().to(e.device) for p, e in zip(self.params, self.ema)],
                                    alpha=1.0 - d)
                self.n += 1

        def state_dict(self):
            if self.ema is None:
                return {}
            return {"ema": self.ema, "decay": self.decay, "n": self.n}

        def load_state_dict(self, state):
            if state and state.get("ema") is not None:
                self.ema, self.decay, self.n, self.params = list(state["ema"]), float(state["decay"]), int(state["n"]), None

        def apply(self, model) -> bool:
            """Copy the averaged weights into the model (before saving). False if there is no average yet."""
            if self.ema is None or self.n == 0:
                return False
            with torch.no_grad():
                for p, e in zip(self.params, self.ema):
                    p.copy_(e.to(p.dtype))
            return True

    return EMA()
