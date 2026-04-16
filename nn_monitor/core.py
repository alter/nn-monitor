"""
TrainingMonitor — main orchestrator class.
Attach to your training loop for comprehensive monitoring.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn

from .metrics import (
    collect_weight_stats, compute_weight_update_ratios, snapshot_weights,
    collect_gradient_stats, compute_ece, prediction_entropy, confidence_gap,
    temporal_stability, track_effective_ranks,
)
from .sanity import check_loss_at_init, check_overfit_one_batch
from .plots import (
    plot_reliability_diagram, plot_gradient_flow,
    plot_weight_update_ratios, plot_training_curves,
)

logger = logging.getLogger(__name__)


class ActivationMonitor:
    """Attach forward hooks to track activation statistics and dead neurons.

    Usage:
        monitor = ActivationMonitor(model)
        model(some_input)  # triggers hooks
        stats = monitor.summary()
        monitor.remove()  # always remove when done
    """

    def __init__(self, model: nn.Module, layer_types=(nn.Linear, nn.Conv1d)):
        self.stats: Dict[str, Dict[str, float]] = {}
        self._hooks = []
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, inp, output):
            with torch.no_grad():
                act = output.detach().float()
                if torch.isnan(act).any():
                    logger.critical(f"NaN DETECTED in layer: {name} activations!")
                if torch.isinf(act).any():
                    logger.critical(f"Inf DETECTED in layer: {name} activations!")

                m, s = act.mean().item(), act.std().item()
                dead = (act.abs() < 1e-8).float().mean().item() * 100
                mx = act.abs().max().item()

                if name not in self.stats:
                    self.stats[name] = {'mean': m, 'std': s, 'dead_pct': dead, 'max_abs': mx, 'n': 1}
                else:
                    st = self.stats[name]
                    n = st['n']
                    st['mean'] = (st['mean'] * n + m) / (n + 1)
                    st['std'] = (st['std'] * n + s) / (n + 1)
                    st['dead_pct'] = (st['dead_pct'] * n + dead) / (n + 1)
                    st['max_abs'] = max(st['max_abs'], mx)
                    st['n'] = n + 1
        return hook_fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self) -> Dict[str, float]:
        if not self.stats:
            return {}
        dead_pcts = [s['dead_pct'] for s in self.stats.values()]
        stds = [s['std'] for s in self.stats.values()]
        return {
            'dead_neuron_pct_mean': float(np.mean(dead_pcts)),
            'dead_neuron_pct_max': float(np.max(dead_pcts)),
            'activation_std_mean': float(np.mean(stds)),
            'activation_std_min': float(np.min(stds)),
            'worst_layer': max(self.stats, key=lambda k: self.stats[k]['dead_pct']),
            'n_monitored_layers': len(self.stats),
        }


class OverfitDetector:
    """Track train/val divergence and emit alerts.

    Usage:
        detector = OverfitDetector()
        alerts = detector.update(train_loss, val_loss, train_acc, val_acc, epoch)
        for alert in alerts:
            logger.warning(alert)
    """

    def __init__(self, patience: int = 5, acc_gap_threshold: float = 10.0):
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
        }
        self.patience = patience
        self.acc_gap_threshold = acc_gap_threshold

    def update(self, train_loss, val_loss, train_acc, val_acc, epoch) -> List[str]:
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)

        alerts = []

        if len(self.history['val_loss']) >= self.patience:
            recent_val = self.history['val_loss'][-self.patience:]
            recent_train = self.history['train_loss'][-self.patience:]
            val_rising = all(recent_val[i] < recent_val[i + 1] for i in range(len(recent_val) - 1))
            train_falling = all(recent_train[i] > recent_train[i + 1] for i in range(len(recent_train) - 1))
            if val_rising and train_falling:
                alerts.append(f"OVERFIT: val_loss rising for {self.patience} epochs while train_loss falling")

        acc_gap = train_acc - val_acc
        if acc_gap > self.acc_gap_threshold:
            alerts.append(f"GAP: train_acc={train_acc:.1f}% val_acc={val_acc:.1f}% gap={acc_gap:.1f}pp")

        if train_loss > 0 and val_loss / train_loss > 1.5:
            alerts.append(f"LOSS_RATIO: val/train = {val_loss/train_loss:.2f}x")

        return alerts


class TrainingMonitor:
    """Main orchestrator for training monitoring.

    Minimal integration:
        monitor = TrainingMonitor('./output')
        monitor.run_sanity_checks(model, loader, criterion, optimizer, device)

        for epoch in range(epochs):
            monitor.before_optimizer_step(model)
            train(...)
            optimizer.step()
            monitor.after_optimizer_step(model)

            val_probs, val_targets = validate(...)
            monitor.log_epoch(epoch, model, val_probs, val_targets,
                            train_loss, val_loss, train_acc, val_acc, lr)

        monitor.save_summary()
    """

    def __init__(self, output_dir: str, detect_anomalies: bool = False):
        self.diag_dir = Path(output_dir) / 'diagnostics'
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        self.overfit_detector = OverfitDetector()
        self._weight_snapshot = None
        self._update_ratios = {}
        self._all_epoch_data: List[Dict] = []
        self.detect_anomalies = detect_anomalies
        if self.detect_anomalies:
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection ENABLED. Training will be slower.")

    def run_sanity_checks(self, model, loader, criterion, optimizer, device, n_classes=None):
        """Run pre-training sanity checks. Call before epoch 0."""
        logger.info("=" * 50)
        logger.info("Pre-training sanity checks")
        logger.info("=" * 50)

        loss_check = check_loss_at_init(model, loader, criterion, device, n_classes)
        overfit_check = check_overfit_one_batch(model, loader, criterion, optimizer, device)

        checks = {'loss_at_init': loss_check, 'overfit_one_batch': overfit_check}
        with open(self.diag_dir / 'sanity_checks.json', 'w') as f:
            json.dump(checks, f, indent=2, default=str)

        return checks

    def before_optimizer_step(self, model):
        """Call BEFORE optimizer.step() to snapshot weights."""
        self._weight_snapshot = snapshot_weights(model)

    def after_optimizer_step(self, model):
        """Call AFTER optimizer.step() to compute update ratios."""
        if self._weight_snapshot:
            self._update_ratios = compute_weight_update_ratios(self._weight_snapshot, model)
            self._weight_snapshot = None

    def log_epoch(
        self,
        epoch: int,
        model: nn.Module,
        val_probs: np.ndarray,
        val_targets: np.ndarray,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        learning_rate: Union[float, Dict[str, float]],
        data_time: float = 0.0,
        compute_time: float = 0.0,
        class_names: Optional[Dict[int, str]] = None,
        full_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Log all diagnostics for one epoch.

        Args:
            val_probs: (N, C) softmax probabilities from validation set
            val_targets: (N,) integer labels
            full_diagnostics: if True, run expensive checks (spectral analysis)
        """
        n_classes = val_probs.shape[1] if val_probs.ndim == 2 else 2
        val_preds = val_probs.argmax(axis=1)
        val_targets = np.asarray(val_targets).flatten()

        if class_names is None:
            class_names = {i: f'class_{i}' for i in range(n_classes)}

        # Calibration
        ece_val, ece_bins = compute_ece(val_probs, val_targets)
        entropy = prediction_entropy(val_probs)
        conf = confidence_gap(val_probs, val_targets)

        # Class distribution
        pred_dist = {class_names.get(c, f'c{c}'): int((val_preds == c).sum()) for c in range(n_classes)}
        target_dist = {class_names.get(c, f'c{c}'): int((val_targets == c).sum()) for c in range(n_classes)}

        # Per-class accuracy
        per_class_acc = {}
        for c in range(n_classes):
            mask = val_targets == c
            name = class_names.get(c, f'class_{c}')
            per_class_acc[name] = round(float((val_preds[mask] == c).mean() * 100), 1) if mask.any() else 0.0

        # Weight stats
        weight_stats_raw = collect_weight_stats(model)
        weight_summary = {}
        if weight_stats_raw:
            norms = [s['frobenius_norm'] for s in weight_stats_raw.values()]
            near_zeros = [s['near_zero_pct'] for s in weight_stats_raw.values()]
            weight_summary = {
                'frobenius_norm_mean': round(float(np.mean(norms)), 4),
                'near_zero_pct_max': round(float(np.max(near_zeros)), 2),
            }

        # Update ratios
        ur_summary = {}
        if self._update_ratios:
            ur_vals = list(self._update_ratios.values())
            ur_summary = {
                'mean': round(float(np.mean(ur_vals)), 6),
                'min': round(float(np.min(ur_vals)), 6),
                'max': round(float(np.max(ur_vals)), 6),
                'frozen_layers': sum(1 for v in ur_vals if v < 1e-6),
            }

        # Gradient stats
        grad_stats = collect_gradient_stats(model)

        # Temporal stability
        temporal = temporal_stability(val_preds)

        # Overfit detection
        overfit_alerts = self.overfit_detector.update(train_loss, val_loss, train_acc, val_acc, epoch)

        # Spectral (expensive)
        spectral = {}
        if full_diagnostics:
            ranks = track_effective_ranks(model)
            if ranks:
                ratios = [r['ratio'] for r in ranks.values()]
                spectral = {
                    'effective_rank_ratio_mean': round(float(np.mean(ratios)), 4),
                    'effective_rank_ratio_min': round(float(np.min(ratios)), 4),
                }

        # Performance/System (Memory and Time)
        gpu_mem_alloc_mb = float(torch.cuda.memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        gpu_max_mem_alloc_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

        # Build diagnostics dict
        diag = {
            'epoch': epoch,
            'performance': {
                'data_time': round(data_time, 4),
                'compute_time': round(compute_time, 4),
                'gpu_mem_alloc_mb': round(gpu_mem_alloc_mb, 2),
                'gpu_max_mem_alloc_mb': round(gpu_max_mem_alloc_mb, 2),
            },
            'losses': {
                'train_loss': round(train_loss, 6),
                'val_loss': round(val_loss, 6),
                'gap_pct': round((val_loss - train_loss) / (train_loss + 1e-8) * 100, 1),
            },
            'accuracy': {
                'train_acc': round(train_acc, 2),
                'val_acc': round(val_acc, 2),
                'gap': round(train_acc - val_acc, 2),
                'per_class': per_class_acc,
            },
            'calibration': {
                'ece': round(ece_val, 4),
                **conf,
                'prediction_entropy_mean': round(entropy['mean'], 4),
                'entropy_normalized': round(entropy['normalized_mean'], 4),
            },
            'class_distribution': {'predictions': pred_dist, 'targets': target_dist},
            'gradients': grad_stats,
            'weights': weight_summary,
            'update_ratios': ur_summary,
            'stability': temporal,
            'spectral': spectral,
            'learning_rates': learning_rate,
            'alerts': overfit_alerts,
        }

        # Save JSON
        with open(self.diag_dir / f'epoch_{epoch:03d}.json', 'w') as f:
            json.dump(diag, f, indent=2, default=str)

        self._all_epoch_data.append(diag)

        # Plots
        try:
            plot_reliability_diagram(val_probs, val_targets,
                                    self.diag_dir / f'reliability_{epoch:03d}.png')
            if self._update_ratios:
                plot_weight_update_ratios(self._update_ratios,
                                         self.diag_dir / f'update_ratios_{epoch:03d}.png')
        except Exception as e:
            logger.warning(f"Plot failed: {e}")

        # Log
        logger.info(
            f"  Monitor: ECE={ece_val:.4f} | entropy={entropy['normalized_mean']:.3f} | "
            f"conf_gap={conf['gap']:.3f} | ur={ur_summary.get('mean', 0):.2e}"
        )
        for alert in overfit_alerts:
            logger.warning(f"  ALERT: {alert}")

        return diag

    def save_summary(self):
        """Save summary of all epochs. Call after training completes."""
        if not self._all_epoch_data:
            return

        def _get(d, *keys, default=0):
            for k in keys:
                if isinstance(d, dict):
                    d = d.get(k, default)
                else:
                    return default
            return d

        summary = {
            'epochs': [d['epoch'] for d in self._all_epoch_data],
            'train_loss': [_get(d, 'losses', 'train_loss') for d in self._all_epoch_data],
            'val_loss': [_get(d, 'losses', 'val_loss') for d in self._all_epoch_data],
            'val_acc': [_get(d, 'accuracy', 'val_acc') for d in self._all_epoch_data],
            'ece': [_get(d, 'calibration', 'ece') for d in self._all_epoch_data],
            'entropy_mean': [_get(d, 'calibration', 'prediction_entropy_mean') for d in self._all_epoch_data],
            'confidence_gap': [_get(d, 'calibration', 'gap') for d in self._all_epoch_data],
            'update_ratio_mean': [_get(d, 'update_ratios', 'mean') for d in self._all_epoch_data],
            'gpu_max_mem_mb_mean': float(np.mean([_get(d, 'performance', 'gpu_max_mem_alloc_mb') for d in self._all_epoch_data])) if self._all_epoch_data and 'performance' in self._all_epoch_data[0] else 0.0,
            'compute_time_mean': float(np.mean([_get(d, 'performance', 'compute_time') for d in self._all_epoch_data])) if self._all_epoch_data and 'performance' in self._all_epoch_data[0] else 0.0,
        }

        with open(self.diag_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        plot_training_curves(summary, self.diag_dir / 'training_curves.png')
        logger.info(f"Training summary saved to {self.diag_dir}")
