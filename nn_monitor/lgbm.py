"""
Diagnostics for LightGBM / tree-based classifiers.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .metrics import compute_ece, prediction_entropy, temporal_stability, confidence_gap
from .plots import plot_reliability_diagram

logger = logging.getLogger(__name__)


def run_lgbm_diagnostics(
    model, X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str], output_dir: str,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Full diagnostics for a trained LightGBM/XGBoost/sklearn classifier.

    Generates: ECE, reliability diagram, prediction entropy, feature importance,
    calibration stats, class distribution, per-class accuracy, temporal stability.

    Args:
        model: fitted classifier with predict_proba() and feature_importances_
        X_val: validation features (N, D)
        y_val: validation labels (N,)
        feature_names: list of feature names
        output_dir: directory for output files
        class_names: optional list of class names

    Returns:
        diagnostics dict
    """
    diag_dir = Path(output_dir) / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        n_classes = len(np.unique(y_val))
        class_names = [f'class_{i}' for i in range(n_classes)]
    n_classes = len(class_names)

    probs = model.predict_proba(X_val)
    preds = probs.argmax(axis=1)
    y_val = np.asarray(y_val).flatten()

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    acc = accuracy_score(y_val, preds)

    # ECE
    ece_val, ece_bins = compute_ece(probs, y_val)

    # Entropy
    entropy = prediction_entropy(probs)

    # Confidence
    conf = confidence_gap(probs, y_val)

    # Per-class accuracy
    per_class_acc = {}
    for c in range(n_classes):
        mask = y_val == c
        name = class_names[c]
        per_class_acc[name] = round(float((preds[mask] == c).mean() * 100), 1) if mask.any() else 0.0

    # Class distribution
    pred_dist = {class_names[c]: int((preds == c).sum()) for c in range(n_classes)}
    target_dist = {class_names[c]: int((y_val == c).sum()) for c in range(n_classes)}

    # Feature importance
    importances = model.feature_importances_
    top_n = min(20, len(feature_names))
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    feat_imp = [
        {'name': feature_names[i], 'importance': int(importances[i]), 'rank': rank + 1}
        for rank, i in enumerate(sorted_idx)
    ]

    # Temporal stability
    temporal = temporal_stability(preds)

    # Build diagnostics
    diag = {
        'accuracy': round(acc * 100, 2),
        'per_class_accuracy': per_class_acc,
        'calibration': {
            'ece': round(ece_val, 4),
            'ece_bins': ece_bins,
            **conf,
            'prediction_entropy_mean': round(entropy['mean'], 4),
            'prediction_entropy_std': round(entropy['std'], 4),
            'entropy_normalized': round(entropy['normalized_mean'], 4),
        },
        'class_distribution': {'predictions': pred_dist, 'targets': target_dist},
        'feature_importance_top20': feat_imp,
        'stability': temporal,
        'confusion_matrix': confusion_matrix(y_val, preds).tolist(),
        'classification_report': classification_report(y_val, preds, target_names=class_names),
    }

    # Save
    with open(diag_dir / 'lgbm_diagnostics.json', 'w') as f:
        json.dump(diag, f, indent=2, default=str)

    plot_reliability_diagram(probs, y_val, diag_dir / 'reliability_lgbm.png')

    # Confidence histogram
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        max_probs = probs.max(axis=1)
        correct = preds == y_val
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(max_probs[correct], bins=30, alpha=0.6, color='#66BB6A', label='Correct')
        ax.hist(max_probs[~correct], bins=30, alpha=0.6, color='#EF5350', label='Wrong')
        ax.set_xlabel('Max Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'Confidence Distribution (ECE={ece_val:.4f})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(diag_dir / 'confidence_histogram.png', dpi=100)
        plt.close()
    except Exception:
        pass

    logger.info(f"LightGBM diagnostics: acc={acc*100:.1f}% ECE={ece_val:.4f} "
                f"conf_gap={conf['gap']:.3f} entropy={entropy['mean']:.3f}")

    return diag
