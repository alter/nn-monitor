"""
Visualization functions. All save to file, no interactive display.
"""

from typing import Dict, List, Optional
import numpy as np

from .metrics import compute_ece


def _get_plt():
    """Lazy matplotlib import with Agg backend."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_reliability_diagram(probs, targets, save_path, n_bins=10):
    """Reliability diagram (calibration plot) with confidence histogram.

    Top panel: accuracy vs confidence (diagonal = perfect calibration).
    Bottom panel: histogram of confidence values.
    """
    plt = _get_plt()
    ece_val, bin_stats = compute_ece(probs, targets, n_bins)
    if not bin_stats:
        return

    confs = [b['confidence'] for b in bin_stats]
    accs = [b['accuracy'] for b in bin_stats]
    counts = [b['count'] for b in bin_stats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax1.bar(confs, accs, width=0.08, alpha=0.7, color='#42A5F5',
            edgecolor='black', linewidth=0.5, label=f'ECE={ece_val:.4f}')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Reliability Diagram')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()

    ax2.bar(confs, counts, width=0.08, color='#66BB6A', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_gradient_flow(model, save_path):
    """Bar chart of gradient norms per layer.

    Red = vanishing (<1e-6), Green = healthy, Orange = large (>10).
    """
    plt = _get_plt()
    layers = []
    norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            layers.append(name.replace('.weight', '').replace('.bias', '(b)'))
            norms.append(param.grad.data.norm(2).item())

    if not layers:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.3), 5))
    colors = ['#ef5350' if n < 1e-6 else '#66BB6A' if n < 10 else '#FFA726' for n in norms]
    ax.barh(range(len(layers)), norms, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=6)
    ax.set_xlabel('Gradient Norm')
    ax.set_title('Gradient Flow')
    ax.set_xscale('log')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_weight_update_ratios(ratios: Dict[str, float], save_path):
    """Horizontal bar chart of |update|/|weight| per layer.

    Blue dashed line = target 1e-3 (Karpathy).
    Red = too small (<1e-5), Green = healthy, Orange = too large (>0.01).
    """
    plt = _get_plt()
    names = [n.replace('.weight', '') for n in ratios.keys()]
    vals = list(ratios.values())

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    colors = ['#ef5350' if v < 1e-5 else '#66BB6A' if v < 0.01 else '#FFA726' for v in vals]
    ax.barh(range(len(names)), vals, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('|update| / |weight|')
    ax.set_title('Weight Update Ratios (healthy: ~1e-3)')
    ax.axvline(x=1e-3, color='blue', linestyle='--', alpha=0.5, label='1e-3 target')
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_training_curves(summary: dict, save_path):
    """6-panel training summary: loss, accuracy, ECE, confidence gap, entropy, update ratio."""
    plt = _get_plt()
    epochs = summary.get('epochs', [])
    if len(epochs) < 2:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Training Summary', fontsize=14)

    panels = [
        (axes[0, 0], 'Loss', [
            ('train_loss', 'train', '#42A5F5'),
            ('val_loss', 'val', '#EF5350'),
        ]),
        (axes[0, 1], 'Validation Accuracy', [('val_acc', None, '#66BB6A')]),
        (axes[1, 0], 'Expected Calibration Error', [('ece', None, '#AB47BC')]),
        (axes[1, 1], 'Confidence Gap (correct - wrong)', [('confidence_gap', None, '#FF7043')]),
        (axes[2, 0], 'Prediction Entropy (mean)', [('entropy_mean', None, '#26A69A')]),
        (axes[2, 1], 'Weight Update Ratio', [('update_ratio_mean', None, '#5C6BC0')]),
    ]

    for ax, title, series_list in panels:
        for key, label, color in series_list:
            data = summary.get(key, [])
            if data and any(v > 0 for v in data if v is not None):
                ax.plot(epochs[:len(data)], data, label=label, color=color)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if any(label for _, label, _ in series_list if label):
            ax.legend()

    # Add target line for update ratio
    ax = axes[2, 1]
    if summary.get('update_ratio_mean'):
        ax.set_yscale('log')
        ax.axhline(y=1e-3, color='blue', linestyle='--', alpha=0.5, label='target 1e-3')
        ax.legend()

    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
