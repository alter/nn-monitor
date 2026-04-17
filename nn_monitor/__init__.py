"""
nn-monitor: Neural Network Training Monitoring Framework

Standalone framework for monitoring, debugging, and diagnosing
neural network training. Works with any PyTorch model.

Usage:
    from nn_monitor import TrainingMonitor

    with TrainingMonitor(output_dir='./diagnostics') as monitor:
        monitor.run_sanity_checks(model, train_loader, criterion, optimizer, device)

        for epoch in range(n_epochs):
            monitor.before_optimizer_step(model)
            train_one_epoch(...)
            monitor.after_optimizer_step(model)

            val_probs, val_targets = validate(...)
            monitor.log_epoch(
                epoch=epoch, model=model,
                val_probs=val_probs, val_targets=val_targets,
                train_loss=train_loss, val_loss=val_loss,
                train_acc=train_acc, val_acc=val_acc,
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
    collect_gradient_stats,
    compute_weight_update_ratios,
    snapshot_weights,
    effective_rank,
    track_effective_ranks,
    compute_psi,
    compute_attention_entropy,
    confidence_gap,
    GradientClipTracker,
    detect_loss_spike,
)

from .sanity import (
    check_loss_at_init,
    check_overfit_one_batch,
    verify_gradient_flow,
    check_receptive_field_gradients,
    check_causal_leakage,
    check_time_split,
)

from .plots import (
    plot_reliability_diagram,
    plot_gradient_flow,
    plot_weight_update_ratios,
    plot_training_curves,
    plot_grad_profile,
    plot_attention_heatmap,
)

from .lgbm import (
    run_lgbm_diagnostics,
    training_curve_stats,
    tree_structure_stats,
    importance_disagreement,
    feature_concentration,
    feature_drift,
)

from .hmm import (
    run_hmm_diagnostics,
    check_transition_matrix,
    state_occupancy,
    dwell_times,
    emission_entropy,
    gaussian_emission_separability,
    check_ll_convergence,
    viterbi_stability,
    check_forward_backward_stability,
)

from .transformer import (
    AttentionMonitor,
    ResidualStreamMonitor,
    attention_collapse_stats,
    head_redundancy,
    positional_encoding_drift,
    tcn_receptive_field,
    check_layer_causal_leakage,
)

__version__ = '1.1.0'
__all__ = [
    # core
    'TrainingMonitor', 'ActivationMonitor', 'OverfitDetector',
    # metrics
    'compute_ece', 'prediction_entropy', 'temporal_stability',
    'prediction_stability_noise', 'collect_weight_stats', 'collect_gradient_stats',
    'compute_weight_update_ratios', 'snapshot_weights', 'effective_rank',
    'track_effective_ranks', 'compute_psi', 'compute_attention_entropy',
    'confidence_gap', 'GradientClipTracker', 'detect_loss_spike',
    # sanity
    'check_loss_at_init', 'check_overfit_one_batch', 'verify_gradient_flow',
    'check_receptive_field_gradients', 'check_causal_leakage', 'check_time_split',
    # plots
    'plot_reliability_diagram', 'plot_gradient_flow', 'plot_weight_update_ratios',
    'plot_training_curves', 'plot_grad_profile', 'plot_attention_heatmap',
    # lgbm
    'run_lgbm_diagnostics', 'training_curve_stats', 'tree_structure_stats',
    'importance_disagreement', 'feature_concentration', 'feature_drift',
    # hmm
    'run_hmm_diagnostics', 'check_transition_matrix', 'state_occupancy',
    'dwell_times', 'emission_entropy', 'gaussian_emission_separability',
    'check_ll_convergence', 'viterbi_stability', 'check_forward_backward_stability',
    # transformer
    'AttentionMonitor', 'ResidualStreamMonitor', 'attention_collapse_stats',
    'head_redundancy', 'positional_encoding_drift', 'tcn_receptive_field',
    'check_layer_causal_leakage',
]
