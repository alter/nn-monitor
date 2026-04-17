"""
Diagnostics for LightGBM / tree-based classifiers.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence

import numpy as np

from .metrics import compute_ece, prediction_entropy, temporal_stability, confidence_gap, compute_psi
from .plots import plot_reliability_diagram

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Per-iteration training curve
# ─────────────────────────────────────────────

def training_curve_stats(evals_result: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    """Extract overfit / early-stop signals from lightgbm's evals_result_.

    evals_result: {'train': {'binary_logloss': [...]}, 'valid': {...}}

    Returns best iter, overfit onset iter (val starts rising while train falling),
    and absolute train-val divergence.
    """
    if not evals_result:
        return {}
    out: Dict[str, Any] = {}
    # Pick first metric
    splits = list(evals_result.keys())
    if not splits:
        return {}
    first_metric = next(iter(evals_result[splits[0]].keys()))

    train_curve = np.asarray(evals_result.get('training', {}).get(first_metric, [])
                             or evals_result.get('train', {}).get(first_metric, []), dtype=float)
    val_curve = np.asarray(
        evals_result.get('valid_1', {}).get(first_metric, [])
        or evals_result.get('valid', {}).get(first_metric, [])
        or evals_result.get('val', {}).get(first_metric, []),
        dtype=float,
    )

    if val_curve.size == 0:
        return {'metric': first_metric, 'train_iters': int(train_curve.size)}

    best_iter = int(np.argmin(val_curve))
    best_val = float(val_curve[best_iter])

    # Overfit onset: earliest i where val[i:] monotonically rises over 5+ iters
    onset = None
    for i in range(best_iter + 5, val_curve.size):
        win = val_curve[best_iter:i + 1]
        if win[-1] > win[0] * 1.02:
            onset = int(best_iter)
            break

    divergence = None
    if train_curve.size == val_curve.size and train_curve.size > 0:
        divergence = float(val_curve[-1] - train_curve[-1])

    return {
        'metric': first_metric,
        'best_iter': best_iter,
        'best_val_score': round(best_val, 6),
        'overfit_onset_iter': onset,
        'final_train_val_gap': round(divergence, 6) if divergence is not None else None,
        'n_iters': int(val_curve.size),
    }


# ─────────────────────────────────────────────
#  Tree structure sanity
# ─────────────────────────────────────────────

def tree_structure_stats(model) -> Dict[str, Any]:
    """Leaf count, depth, and split-gain distribution per tree.

    Uses `model.booster_.dump_model()` which works for lightgbm.sklearn wrappers.
    """
    try:
        booster = getattr(model, 'booster_', None) or model
        dump = booster.dump_model()
    except Exception as e:
        return {'ok': False, 'msg': f"dump_model failed: {e}"}

    trees = dump.get('tree_info', [])
    if not trees:
        return {'ok': False, 'msg': 'no trees'}

    def walk(node, depth=0):
        if 'leaf_index' in node or 'leaf_value' in node:
            return 1, depth, []
        left_leaves, left_depth, left_gains = walk(node.get('left_child', {}), depth + 1)
        right_leaves, right_depth, right_gains = walk(node.get('right_child', {}), depth + 1)
        gains = left_gains + right_gains
        gains.append(float(node.get('split_gain', 0.0)))
        return left_leaves + right_leaves, max(left_depth, right_depth), gains

    leaf_counts = []
    depths = []
    all_gains: List[float] = []
    for t in trees:
        n_leaves, d, gains = walk(t.get('tree_structure', {}))
        leaf_counts.append(n_leaves)
        depths.append(d)
        all_gains.extend(gains)

    # Average split-gain trajectory across trees (quartiles)
    gains_arr = np.asarray(all_gains) if all_gains else np.array([0.0])
    leaves = np.asarray(leaf_counts)
    depths_arr = np.asarray(depths)

    return {
        'ok': True,
        'n_trees': int(len(trees)),
        'leaf_count_mean': float(leaves.mean()),
        'leaf_count_max': int(leaves.max()),
        'leaf_count_growth_slope': float(np.polyfit(np.arange(len(leaves)), leaves, 1)[0]) if len(leaves) > 2 else 0.0,
        'depth_mean': float(depths_arr.mean()),
        'depth_max': int(depths_arr.max()),
        'gain_q25': round(float(np.percentile(gains_arr, 25)), 4),
        'gain_q50': round(float(np.percentile(gains_arr, 50)), 4),
        'gain_q75': round(float(np.percentile(gains_arr, 75)), 4),
        'gain_tail_fraction': round(float((gains_arr < 1e-6).mean()), 4),
    }


# ─────────────────────────────────────────────
#  Feature importance stability
# ─────────────────────────────────────────────

def importance_disagreement(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 5,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Compare gain importance vs permutation importance.

    Large disagreement (Spearman ρ < 0.5) = model relies on leakage or low-gain
    but high-usage features, a red flag for brittle generalization.
    """
    try:
        from sklearn.inspection import permutation_importance  # type: ignore
    except ImportError:
        return {'ok': False, 'msg': 'sklearn not installed'}

    gain = np.asarray(model.feature_importances_, dtype=float)
    if gain.size != len(feature_names):
        return {'ok': False, 'msg': 'feature_importances_ size mismatch'}

    r = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )
    perm = np.asarray(r.importances_mean, dtype=float)

    # Spearman
    def _rank(x):
        order = np.argsort(x)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(x))
        return ranks
    rg, rp = _rank(gain), _rank(perm)
    if gain.size > 1:
        rho = float(np.corrcoef(rg, rp)[0, 1])
    else:
        rho = 1.0

    # Top-5 disagreement
    top_gain = set(np.argsort(gain)[::-1][:5].tolist())
    top_perm = set(np.argsort(perm)[::-1][:5].tolist())
    jaccard = len(top_gain & top_perm) / max(len(top_gain | top_perm), 1)

    return {
        'ok': bool(rho > 0.5 and jaccard > 0.4),
        'spearman_rho': round(rho, 4),
        'top5_jaccard': round(jaccard, 4),
        'gain_importance_top5': [feature_names[i] for i in np.argsort(gain)[::-1][:5]],
        'perm_importance_top5': [feature_names[i] for i in np.argsort(perm)[::-1][:5]],
    }


def feature_concentration(
    model,
    feature_names: Sequence[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Share of total gain captured by top-k features.

    top_k_share > 0.8 → model is brittle, depends on few features.
    """
    gain = np.asarray(model.feature_importances_, dtype=float)
    total = float(gain.sum())
    if total <= 0:
        return {'ok': False, 'msg': 'zero total importance'}
    sorted_gain = np.sort(gain)[::-1]
    share_top_k = float(sorted_gain[:top_k].sum() / total)
    top_names = [feature_names[i] for i in np.argsort(gain)[::-1][:top_k]]
    return {
        'ok': bool(share_top_k < 0.8),
        f'top{top_k}_share': round(share_top_k, 4),
        f'top{top_k}_features': top_names,
        'n_features': int(len(feature_names)),
        'zero_importance_count': int((gain == 0).sum()),
    }


# ─────────────────────────────────────────────
#  Drift
# ─────────────────────────────────────────────

def feature_drift(
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: Sequence[str],
    psi_warn: float = 0.1,
) -> Dict[str, Any]:
    """Per-feature PSI between train and val distributions."""
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    per_feature = {}
    drifted = []
    for i, name in enumerate(feature_names):
        if i >= X_train.shape[1]:
            break
        psi = compute_psi(X_train[:, i], X_val[:, i])
        per_feature[name] = psi['psi_total']
        if psi['psi_total'] >= psi_warn:
            drifted.append((name, psi['psi_total']))
    drifted.sort(key=lambda x: -x[1])
    return {
        'ok': bool(not drifted),
        'drifted_features': drifted[:20],
        'mean_psi': round(float(np.mean(list(per_feature.values()))) if per_feature else 0.0, 4),
        'n_features': int(len(per_feature)),
    }


def run_lgbm_diagnostics(
    model, X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str], output_dir: str,
    class_names: Optional[List[str]] = None,
    X_train: Optional[np.ndarray] = None,
    evals_result: Optional[Dict[str, Dict[str, List[float]]]] = None,
    run_permutation: bool = False,
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

    # Extended diagnostics
    concentration = feature_concentration(model, feature_names)
    tree_stats = tree_structure_stats(model)
    curve_stats = training_curve_stats(evals_result or {})

    drift_stats = {}
    if X_train is not None:
        drift_stats = feature_drift(X_train, X_val, feature_names)

    perm_stats = {}
    if run_permutation:
        perm_stats = importance_disagreement(model, X_val, y_val, feature_names)

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
        'feature_concentration': concentration,
        'tree_structure': tree_stats,
        'training_curve': curve_stats,
        'feature_drift': drift_stats,
        'permutation_vs_gain': perm_stats,
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
