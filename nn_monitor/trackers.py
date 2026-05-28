"""
Optional experiment-tracker sinks (TensorBoard, Weights & Biases).

TrainingMonitor writes JSON/PNG to disk regardless; trackers let the same
scalar metrics stream live to a dashboard. Each tracker exposes ``log(metrics,
step)`` and ``close()``. Imports are lazy so neither library is a hard dep.
"""

import logging
from numbers import Number
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


def flatten_scalars(d: Dict[str, Any], prefix: str = '', sep: str = '/') -> Dict[str, float]:
    """Recursively pull numeric (non-bool) leaves out of a nested dict.

    Lists/strings/bools are skipped — trackers want scalar time series.
    """
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f'{prefix}{sep}{k}' if prefix else str(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, Number):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(flatten_scalars(v, key, sep))
    return out


class TensorBoardTracker:
    """Thin wrapper over torch.utils.tensorboard.SummaryWriter."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter  # lazy
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, metrics: Dict[str, Any], step: int):
        for k, v in flatten_scalars(metrics).items():
            self.writer.add_scalar(k, v, step)

    def close(self):
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass


class WandbTracker:
    """Thin wrapper over an existing wandb run (does not init/finish it)."""

    def __init__(self, run=None):
        import wandb  # lazy
        self.wandb = wandb
        self.run = run or wandb.run
        if self.run is None:
            raise RuntimeError("No active wandb run; call wandb.init() first or pass run=")

    def log(self, metrics: Dict[str, Any], step: int):
        self.run.log(flatten_scalars(metrics), step=step)

    def close(self):
        pass  # leave run lifecycle to the caller


def _log_all(trackers: Iterable, metrics: Dict[str, Any], step: int):
    for t in trackers:
        try:
            t.log(metrics, step)
        except Exception as e:  # a broken tracker must never kill training
            logger.warning(f"Tracker {type(t).__name__} log failed: {e}")
