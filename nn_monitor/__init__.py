"""
nn-monitor: Neural Network Training Monitoring Framework

Standalone framework for monitoring, debugging, and diagnosing
neural network training. Works with any PyTorch model.

Usage:
    from nn_monitor import TrainingMonitor

    monitor = TrainingMonitor(output_dir='./diagnostics')
    monitor.run_sanity_checks(model, train_loader, criterion, optimizer, device)

    for epoch in range(n_epochs):
        # Before optimizer.step():
        monitor.before_optimizer_step(model)

        train_one_epoch(...)

        # After optimizer.step():
        monitor.after_optimizer_step(model)

        # After validation:
        monitor.log_epoch(
            epoch=epoch,
            model=model,
            val_probs=val_probs,        # (N, C) softmax probabilities
            val_targets=val_targets,    # (N,) integer labels
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            learning_rate=lr,
        )

    monitor.save_summary()
"""

from .core import (
    TrainingMonitor,
    ActivationMonitor,
    OverfitDetector,
)

from .metrics import (
    compute_ece,
    prediction_entropy,
    temporal_stability,
    prediction_stability_noise,
    collect_weight_stats,
    compute_weight_update_ratios,
    snapshot_weights,
    effective_rank,
    track_effective_ranks,
)

from .sanity import (
    check_loss_at_init,
    check_overfit_one_batch,
)

from .plots import (
    plot_reliability_diagram,
    plot_gradient_flow,
    plot_weight_update_ratios,
    plot_training_curves,
)

from .lgbm import run_lgbm_diagnostics

__version__ = '1.0.0'
__all__ = [
    'TrainingMonitor',
    'ActivationMonitor',
    'OverfitDetector',
    'compute_ece',
    'prediction_entropy',
    'temporal_stability',
    'prediction_stability_noise',
    'collect_weight_stats',
    'compute_weight_update_ratios',
    'snapshot_weights',
    'effective_rank',
    'track_effective_ranks',
    'check_loss_at_init',
    'check_overfit_one_batch',
    'plot_reliability_diagram',
    'plot_gradient_flow',
    'plot_weight_update_ratios',
    'plot_training_curves',
    'run_lgbm_diagnostics',
]
