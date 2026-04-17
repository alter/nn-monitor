"""
Pure metric computation functions. No side effects, no I/O.
"""

from typing import Dict, List, Optional, Any
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
        q = torch.quantile(
            w.flatten().cpu(),
            torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]),
        ).tolist()
        stats[name] = {
            'mean': w.mean().item(),
            'std': w.std().item(),
            'min': w.min().item(),
            'max': w.max().item(),
            'frobenius_norm': w.norm().item(),
            'near_zero_pct': (w.abs() < 1e-6).float().mean().item() * 100,
            'q05': q[0], 'q25': q[1], 'q50': q[2], 'q75': q[3], 'q95': q[4],
            'kurtosis_excess': _excess_kurtosis(w),
        }
    return stats


def _excess_kurtosis(t: torch.Tensor) -> float:
    """Pearson excess kurtosis. Heavy tail (>3) often signals pathological weights."""
    x = t.flatten().float()
    if x.numel() < 4:
        return 0.0
    m = x.mean()
    s = x.std(unbiased=False)
    if float(s) < 1e-12:
        return 0.0
    z = (x - m) / s
    return float((z ** 4).mean().item() - 3.0)


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


def collect_gradient_stats(model: nn.Module) -> Dict[str, Any]:
    """Gradient norm statistics across all layers.

    Returns dict with grad_min/max/mean, total grad norm, vanishing/exploding
    layer counts, NaN/Inf layer lists.
    """
    grad_norms: List[float] = []
    total_sq = 0.0
    nan_layers: List[str] = []
    inf_layers: List[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.data
        if torch.isnan(g).any():
            nan_layers.append(name)
        if torch.isinf(g).any():
            inf_layers.append(name)
        if nan_layers or inf_layers:
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        n = float(g.norm(2).item())
        grad_norms.append(n)
        total_sq += n * n

    if not grad_norms:
        return {}

    return {
        'grad_min': float(min(grad_norms)),
        'grad_max': float(max(grad_norms)),
        'grad_mean': float(sum(grad_norms) / len(grad_norms)),
        'grad_total_norm': float(np.sqrt(total_sq)),
        'grad_zero_layers': int(sum(1 for n in grad_norms if n < 1e-8)),
        'grad_exploding_layers': int(sum(1 for n in grad_norms if n > 100)),
        'grad_total_layers': int(len(grad_norms)),
        'grad_nan_layers': nan_layers,
        'grad_inf_layers': inf_layers,
    }


class GradientClipTracker:
    """Count how often gradient clipping was triggered, and by how much.

    Integrate next to `torch.nn.utils.clip_grad_norm_`:

        tracker = GradientClipTracker()
        for batch in loader:
            loss.backward()
            total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            tracker.update(total_norm=total.item(), clip_value=1.0)
            optimizer.step()
        stats = tracker.summary()

    Interpretation:
        clip_rate > 0.5 → LR likely too high (clip as crutch)
        clip_rate < 0.01 → clip_value probably set too loose
    """

    def __init__(self):
        self.clipped = 0
        self.total = 0
        self.ratios: List[float] = []

    def update(self, total_norm: float, clip_value: float):
        self.total += 1
        if total_norm > clip_value:
            self.clipped += 1
            self.ratios.append(total_norm / max(clip_value, 1e-12))

    def summary(self) -> Dict[str, float]:
        if self.total == 0:
            return {}
        return {
            'clip_rate': self.clipped / self.total,
            'clip_triggered': int(self.clipped),
            'total_steps': int(self.total),
            'avg_excess_ratio': float(np.mean(self.ratios)) if self.ratios else 0.0,
            'max_excess_ratio': float(np.max(self.ratios)) if self.ratios else 0.0,
        }

    def reset(self):
        self.clipped = 0
        self.total = 0
        self.ratios.clear()


def detect_loss_spike(
    losses: List[float],
    window: int = 20,
    factor: float = 3.0,
) -> Dict[str, Any]:
    """Check whether the most recent loss is a spike vs rolling median.

    Returns ok=False with detail when loss[-1] > factor * median(loss[-window-1:-1]).
    """
    if len(losses) < window + 1:
        return {'ok': True, 'reason': 'insufficient history'}
    recent = losses[-(window + 1):-1]
    current = float(losses[-1])
    med = float(np.median(recent))
    if med <= 0:
        return {'ok': True, 'reason': 'non-positive median'}
    ratio = current / med
    return {
        'ok': ratio <= factor,
        'current': current,
        'median_window': med,
        'ratio': ratio,
        'factor': factor,
        'window': window,
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

    Note: for Conv2d (out_c, in_c, kH, kW) and higher-dim tensors, we
    flatten to (out_c, in_c*kH*kW). This gives a coarse estimate —
    the true convolutional rank requires unfold + Toeplitz expansion,
    which is expensive.
    """
    if weight.dim() < 2:
        return float(weight.numel())
    W = weight.detach().float().cpu()
    if W.dim() > 2:
        W = W.reshape(W.shape[0], -1)
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

    # Relative smoothing floor — absolute 1e-4 inflates PSI on small samples
    floor_e = 1.0 / max(len(expected_flat), 1)
    floor_a = 1.0 / max(len(actual_flat), 1)
    expected_perc = np.maximum(expected_counts / len(expected_flat), floor_e)
    actual_perc = np.maximum(actual_counts / len(actual_flat), floor_a)

    psi_values = (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)
    total_psi = float(np.sum(psi_values))

    return {
        'psi_total': round(total_psi, 4),
        'drift_detected': bool(total_psi > 0.1),
        'severity': 'none' if total_psi < 0.1 else ('moderate' if total_psi < 0.2 else 'significant'),
        'bins_used': int(len(breakpoints) - 1),
        'n_expected': int(len(expected_flat)),
        'n_actual': int(len(actual_flat)),
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
