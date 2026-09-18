from authgenforge import *


class CosineLRScheduler:
    """Learning-rate schedule.

    - warmup_epochs (if > 0): one linear ramp from warmup_lr up to base_lr,
      applied once at the very start of training (not repeated per cycle).
    - scheduler_type="cyclic": once warmup ends, LR repeats a full
      top -> low -> top cosine oscillation every T_0_epochs. The first
      cyclic step picks up exactly where warmup left off (base_lr, the
      top), so there's no jump at the warmup/cyclic handoff; every later
      cycle is self-consistent for the same reason (each one both starts
      and ends at base_lr).
    - scheduler_type="annealing": after warmup, a single cosine decay down
      to min_lr across the rest of training (no repeat).
    """

    def __init__(
        self,
        optimizer,
        num_epochs: int,
        steps_per_epoch: int,
        base_lr: float,
        min_lr: float,
        warmup_epochs: float,
        warmup_lr: float,
        scheduler_type: str = "cyclic",
        T_0_epochs: int = 10,
    ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_lr = warmup_lr
        # per-group base lr (layer_decay gives each group its own value) —
        # captured now, at construction time, same as torch's schedulers do.
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

        num_training_steps = max(int(round(num_epochs * steps_per_epoch)), 1)
        self.num_warmup_steps = max(int(round(warmup_epochs * steps_per_epoch)), 0)

        if scheduler_type == "cyclic":
            self.cycle_len = max(int(round(T_0_epochs * steps_per_epoch)), 1)
        elif scheduler_type == "annealing":
            self.cycle_len = max(num_training_steps - self.num_warmup_steps, 1)
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type!r}")

        self._step_count = 0
        self._apply_lr()

    def _lr_at(self, step: int, base_lr_i: float) -> float:
        if self.num_warmup_steps > 0 and step < self.num_warmup_steps:
            start_frac = self.warmup_lr / self.base_lr if self.base_lr > 0 else 0.0
            t = step / self.num_warmup_steps
            return base_lr_i * (start_frac + (1.0 - start_frac) * t)

        post_warmup_step = step - self.num_warmup_steps

        if self.scheduler_type == "cyclic":
            cycle_pos = post_warmup_step % self.cycle_len
            progress = cycle_pos / self.cycle_len
            cos_term = np.cos(2 * np.pi * progress)
            if self.num_warmup_steps == 0:
                # no warmup precedes the first cycle, so start low -> high
                # -> low instead of jumping straight to base_lr at step 0.
                cos_term = -cos_term
            # (with warmup) top -> low -> top: continuous with warmup's
            # endpoint (base_lr) and with every following cycle's start/end.
            return self.min_lr + (base_lr_i - self.min_lr) * (1 + cos_term) / 2

        # annealing: single cosine decay to min_lr, no repeat.
        progress = min(max(post_warmup_step / self.cycle_len, 0.0), 1.0)
        return self.min_lr + (base_lr_i - self.min_lr) * (1 + np.cos(np.pi * progress)) / 2

    def _apply_lr(self) -> None:
        for group, base_lr_i in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._lr_at(self._step_count, base_lr_i)

    def step(self):
        self._step_count += 1
        self._apply_lr()

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {"step_count": self._step_count}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict.get("step_count", 0)
        self._apply_lr()