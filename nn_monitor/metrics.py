"""
Pure metric computation functions. No side effects, no I/O.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  Weight & Gradient Statistics
# ─────────────────────────────────────────────

def collect_weight_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Per-layer weight statistics: mean, std, frobenius norm, near-zero %.

    Example:
        stats = collect_weight_stats(model)
        for layer, s in stats.items():
            if s['near_zero_pct'] > 50:
                print(f"WARNING: {layer} has {s['near_zero_pct']:.0f}% dead weights")
    """
    stats = {}
    for name, param in model.named_parameters():
        if 'weight' not in name or param.dim() < 2:
            continue
        w = param.data.float()
        if torch.isnan(w).any() or torch.isinf(w).any():
            w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        stats[name] = {
            'mean': w.mean().item(),
            'std': w.std().item(),
            'min': w.min().item(),
            'max': w.max().item(),
            'frobenius_norm': w.norm().item(),
            'near_zero_pct': (w.abs() < 1e-6).float().mean().item() * 100,
        }
    return stats


def snapshot_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone current weights for update ratio computation.

    Usage:
        snap = snapshot_weights(model)
        optimizer.step()
        ratios = compute_weight_update_ratios(snap, model)
    """
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if 'weight' in name and param.dim() >= 2
    }


def compute_weight_update_ratios(
    prev_weights: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, float]:
    """Ratio |update| / |weight| per layer.

    Healthy range: ~1e-3 (Karpathy's golden metric).
    < 1e-5 = layer barely learning.
    > 0.1 = unstable, LR too high.
    """
    ratios = {}
    for name, param in model.named_parameters():
        if name in prev_weights and param.dim() >= 2:
            update = (param.data - prev_weights[name]).norm().item()
            weight = param.data.norm().item()
            ratios[name] = update / (weight + 1e-8)
    return ratios


def collect_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """Gradient norm statistics across all layers.

    Returns dict with grad_min, grad_max, grad_mean,
    grad_zero_layers (vanishing), grad_exploding_layers.
    """
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.data
            if torch.isnan(g).any() or torch.isinf(g).any():
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norms.append(g.norm(2).item())

    if not grad_norms:
        return {}

    return {
        'grad_min': min(grad_norms),
        'grad_max': max(grad_norms),
        'grad_mean': sum(grad_norms) / len(grad_norms),
        'grad_zero_layers': sum(1 for n in grad_norms if n < 1e-8),
        'grad_exploding_layers': sum(1 for n in grad_norms if n > 100),
        'grad_total_layers': len(grad_norms),
    }


# ─────────────────────────────────────────────
#  Calibration & Uncertainty
# ─────────────────────────────────────────────

def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10):
    """Expected Calibration Error.

    Args:
        probs: (N, C) softmax probabilities
        targets: (N,) integer class labels

    Returns:
        (ece_value, bin_stats_list)

    Interpretation:
        ECE < 0.05 = well calibrated
        ECE > 0.15 = poorly calibrated, confidence thresholds unreliable
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_size = mask.sum() / len(targets)
        ece += bin_size * abs(bin_acc - bin_conf)
        bin_stats.append({
            'range': f'{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}',
            'accuracy': round(float(bin_acc), 4),
            'confidence': round(float(bin_conf), 4),
            'count': int(mask.sum()),
        })
    return float(ece), bin_stats


def prediction_entropy(probs: np.ndarray) -> Dict[str, float]:
    """Shannon entropy of softmax predictions.

    Interpretation:
        entropy_normalized ~1.0 = uniform predictions (model knows nothing)
        entropy_normalized ~0.0 = very confident (check if overconfident)
        std < 0.05 = all predictions identical ("red line" problem)
    """
    eps = 1e-8
    entropy = -(probs * np.log(probs + eps)).sum(axis=1)
    max_entropy = float(np.log(probs.shape[1]))
    return {
        'mean': float(entropy.mean()),
        'std': float(entropy.std()),
        'min': float(entropy.min()),
        'max': float(entropy.max()),
        'max_possible': max_entropy,
        'normalized_mean': float(entropy.mean() / max_entropy) if max_entropy > 0 else 0,
    }


def confidence_gap(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Confidence gap between correct and incorrect predictions.

    gap ~0 = model equally confident when right and wrong (useless confidence).
    gap > 0.05 = model knows when it's right (confidence is meaningful).
    """
    preds = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)
    correct = preds == targets
    conf_correct = float(max_probs[correct].mean()) if correct.any() else 0
    conf_wrong = float(max_probs[~correct].mean()) if (~correct).any() else 0
    return {
        'confidence_correct': round(conf_correct, 4),
        'confidence_wrong': round(conf_wrong, 4),
        'gap': round(conf_correct - conf_wrong, 4),
    }


# ─────────────────────────────────────────────
#  Spectral Analysis
# ─────────────────────────────────────────────

def effective_rank(weight: torch.Tensor) -> float:
    """Effective rank via entropy of normalized singular values.

    Roy & Vetterli (2007). Measures how many dimensions
    the weight matrix actually uses.

    ratio = effective_rank / max_rank:
        ~1.0 = full rank (no compression, like random init)
        0.3-0.8 = healthy (model learned structured patterns)
        < 0.1 = rank collapse (model lost capacity, early overfit sign)
    """
    if weight.dim() < 2:
        return float(weight.numel())
    W = weight.detach().float().cpu()
    if W.dim() > 2:
        W = W.view(W.shape[0], -1)
    try:
        S = torch.linalg.svdvals(W)
    except Exception:
        return float(min(W.shape))
    S = S / (S.sum() + 1e-12)
    entropy = -(S * torch.log(S + 1e-12)).sum()
    return torch.exp(entropy).item()


def track_effective_ranks(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Effective rank ratio per weight matrix."""
    ranks = {}
    for name, param in model.named_parameters():
        if 'weight' not in name or param.dim() < 2:
            continue
        max_rank = min(param.shape)
        eff = effective_rank(param.data)
        ranks[name] = {
            'effective_rank': round(eff, 2),
            'max_rank': max_rank,
            'ratio': round(eff / max_rank, 4),
        }
    return ranks


# ─────────────────────────────────────────────
#  Prediction Stability
# ─────────────────────────────────────────────

def temporal_stability(predictions: np.ndarray) -> Dict[str, float]:
    """How often predictions change between consecutive samples.

    For time series: high change_rate = noisy, unstable predictions.
    """
    if len(predictions) < 2:
        return {'total_changes': 0, 'change_rate': 0, 'avg_hold_bars': 0}
    changes = int((predictions[1:] != predictions[:-1]).sum())
    change_rate = changes / len(predictions)
    return {
        'total_changes': changes,
        'change_rate': round(float(change_rate), 4),
        'avg_hold_bars': round(1.0 / (change_rate + 1e-8), 1),
    }


def prediction_stability_noise(
    model: nn.Module, features: torch.Tensor, device: torch.device,
    noise_std: float = 0.01, n_trials: int = 3,
) -> Dict[str, float]:
    """Flip rate when small Gaussian noise is added to inputs.

    flip_rate < 5% = stable predictions.
    flip_rate > 20% = on decision boundary, unreliable.
    """
    model.eval()
    with torch.no_grad():
        base = model(features.to(device)).argmax(dim=1).cpu()
        flip_rates = []
        for _ in range(n_trials):
            noisy = features + torch.randn_like(features) * noise_std
            noisy_preds = model(noisy.to(device)).argmax(dim=1).cpu()
            flip_rates.append((base != noisy_preds).float().mean().item())
    return {
        'mean_flip_rate': round(float(np.mean(flip_rates)), 4),
        'noise_std': noise_std,
    }


# ─────────────────────────────────────────────
#  Time-Series & Structural Specific
# ─────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Population Stability Index (PSI) to detect feature drift (Covariate Shift).
    
    Useful for LightGBM, HMM, or any model handling structured/timeseries data 
    where the distribution might shift between training (expected) and inference (actual).
    
    Interpretation:
        PSI < 0.1: No significant drift (safe).
        PSI 0.1 - 0.2: Moderate drift (monitor closely).
        PSI > 0.2: Significant shift (retraining advised).
    """
    expected_flat = np.asarray(expected).flatten()
    actual_flat = np.asarray(actual).flatten()
    
    if len(expected_flat) == 0 or len(actual_flat) == 0:
        return {'psi_total': 0.0, 'drift_detected': False}

    breakpoints = np.percentile(expected_flat, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return {'psi_total': 0.0, 'drift_detected': False}
        
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected_flat, bins=breakpoints)
    actual_counts, _ = np.histogram(actual_flat, bins=breakpoints)

    expected_perc = np.maximum(expected_counts / len(expected_flat), 1e-4)
    actual_perc = np.maximum(actual_counts / len(actual_flat), 1e-4)

    psi_values = (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)
    total_psi = float(np.sum(psi_values))

    return {
        'psi_total': round(total_psi, 4),
        'drift_detected': total_psi > 0.1,
        'bins_used': len(breakpoints) - 1,
    }


def compute_attention_entropy(attention_weights: torch.Tensor) -> Dict[str, float]:
    """Shannon entropy of attention head distributions.
    
    Detects if self-attention mechanism collapses (focusing solely on 1 token) 
    or becomes uniform (focusing on all tokens equally).
    
    Args:
        attention_weights: (B, num_heads, seq_len, seq_len) or (seq_len, seq_len)
    
    Interpretation:
        normalized_entropy ~ 0.0: Collasped attention (attending to a single token)
        normalized_entropy ~ 1.0: Uniform attention (uninformative mapping) 
    """
    if attention_weights.dim() < 2:
        return {}
    
    seq_len = attention_weights.shape[-1]
    if seq_len <= 1:
        return {'normalized_mean': 1.0}
        
    eps = 1e-8
    A = attention_weights.detach().float().cpu()
    
    entropy = -(A * torch.log(A + eps)).sum(dim=-1)
    max_entropy = float(np.log(seq_len))
    avg_entropy = float(entropy.mean())
    
    return {
        'entropy_mean': round(avg_entropy, 4),
        'max_possible': round(max_entropy, 4),
        'normalized_mean': round(avg_entropy / max_entropy, 4) if max_entropy > 0 else 0.0,
    }
