"""
Diagnostics specific to Transformers and TCNs on time-series.

Addresses the failure modes that generic activation/weight monitoring misses:
- attention collapse (one token or diagonal)
- head redundancy (heads learning the same map)
- residual stream norm blow-up across depth
- position-encoding drift
- TCN dilation coverage vs required receptive field
- autoregressive causal leakage regression check (per-layer)
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Attention
# ─────────────────────────────────────────────

def attention_collapse_stats(attn: torch.Tensor) -> Dict[str, Any]:
    """Attention collapse / diagonal / uniform metrics.

    attn shape: (B, H, L, L) or (H, L, L) or (L, L).

    Indicators:
        max_prob_mean      > 0.95 → attention collapsed on single key
        diag_mass_mean    ~ 1.0   → identity attention (no info mixing)
        normalized_entropy ~ 0.0  → spiked (1-hot per query)
        normalized_entropy ~ 1.0  → uniform (uninformative)
    """
    if not torch.is_tensor(attn):
        attn = torch.as_tensor(attn)
    while attn.dim() < 4:
        attn = attn.unsqueeze(0)
    # Now (B, H, L, L)

    B, H, L, L2 = attn.shape
    if L != L2:
        return {'ok': False, 'msg': f"non-square attention: {attn.shape}"}
    if L < 2:
        return {'ok': True, 'msg': 'L<2'}

    eps = 1e-8
    A = attn.detach().float().cpu()
    # Ensure rows sum to ~1 (softmax); if not, renormalize for entropy
    row_sums = A.sum(dim=-1, keepdim=True)
    A = A / (row_sums + eps)

    entropy = -(A * torch.log(A + eps)).sum(dim=-1)  # (B, H, L)
    max_ent = float(np.log(L))
    norm_ent = entropy / max_ent

    # Max prob per query
    max_prob = A.max(dim=-1).values  # (B, H, L)

    # Diagonal mass
    eye = torch.eye(L)
    diag_mass = (A * eye).sum(dim=(-1, -2)) / L  # (B, H)

    # Per-head averaged over batch and queries
    per_head_entropy = norm_ent.mean(dim=(0, 2)).tolist()  # (H,)
    per_head_max = max_prob.mean(dim=(0, 2)).tolist()
    per_head_diag = diag_mass.mean(dim=0).tolist()

    return {
        'ok': bool(
            float(max_prob.mean()) < 0.95
            and float(norm_ent.mean()) > 0.05
            and float(diag_mass.mean()) < 0.95
        ),
        'normalized_entropy_mean': round(float(norm_ent.mean()), 4),
        'max_prob_mean': round(float(max_prob.mean()), 4),
        'diag_mass_mean': round(float(diag_mass.mean()), 4),
        'per_head_entropy': [round(float(x), 4) for x in per_head_entropy],
        'per_head_max_prob': [round(float(x), 4) for x in per_head_max],
        'per_head_diag_mass': [round(float(x), 4) for x in per_head_diag],
        'n_heads': int(H),
        'seq_len': int(L),
    }


def head_redundancy(attn: torch.Tensor, corr_threshold: float = 0.95) -> Dict[str, Any]:
    """Pairwise correlation between attention heads.

    Pairs with corr > threshold are redundant — one can be pruned.
    """
    if not torch.is_tensor(attn):
        attn = torch.as_tensor(attn)
    while attn.dim() < 4:
        attn = attn.unsqueeze(0)
    B, H, L, _ = attn.shape
    if H < 2:
        return {'ok': True, 'msg': 'H<2'}

    A = attn.detach().float().cpu().mean(dim=0).reshape(H, -1)  # (H, L*L)
    A = A - A.mean(dim=1, keepdim=True)
    norm = A.norm(dim=1, keepdim=True) + 1e-12
    A = A / norm
    corr = (A @ A.T).numpy()

    redundant = []
    for i in range(H):
        for j in range(i + 1, H):
            if corr[i, j] > corr_threshold:
                redundant.append((int(i), int(j), round(float(corr[i, j]), 4)))

    # Mean off-diagonal correlation
    off_diag = corr[np.triu_indices(H, k=1)]
    return {
        'ok': bool(len(redundant) == 0),
        'mean_off_diag_corr': round(float(off_diag.mean()), 4),
        'max_off_diag_corr': round(float(off_diag.max()), 4),
        'redundant_pairs': redundant,
        'n_heads': int(H),
    }


class AttentionMonitor:
    """Capture attention weights from nn.MultiheadAttention layers via forward hooks.

    Requires layer(..., need_weights=True, average_attn_weights=False) to be the
    model's behaviour, otherwise torch returns None / averaged. Use the manual
    register helper `.attach_to(module, name)` if your model uses a custom attn.
    """

    def __init__(self, model: Optional[nn.Module] = None):
        self.attentions: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        if model is not None:
            for name, module in model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    self.attach_to(module, name)

    def attach_to(self, module: nn.Module, name: str):
        def hook_fn(mod, inp, output):
            # output is (attn_output, attn_weights) for MHA
            if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
                self.attentions[name] = output[1].detach()
        self._hooks.append(module.register_forward_hook(hook_fn))

    def reset(self):
        self.attentions.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False

    def summary(self) -> Dict[str, Any]:
        out = {}
        for name, A in self.attentions.items():
            out[name] = {
                'collapse': attention_collapse_stats(A),
                'redundancy': head_redundancy(A),
            }
        return out


# ─────────────────────────────────────────────
#  Residual stream norm tracking
# ─────────────────────────────────────────────

class ResidualStreamMonitor:
    """Track L2 norm of activations at specific hook points (e.g. each block output).

    In a healthy Transformer the residual-stream norm grows sublinearly with depth.
    Exponential growth = instability. Collapse to ~0 = dead path.
    """

    def __init__(self):
        self.norms: Dict[str, List[float]] = {}
        self._hooks: List = []

    def attach_to(self, module: nn.Module, name: str):
        def hook_fn(mod, inp, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(t):
                # Per-token mean L2 norm, averaged over batch
                with torch.no_grad():
                    # Expect (B, L, D) or (B, D, L) or (B, D)
                    if t.dim() >= 2:
                        norms = t.float().flatten(0, -2).norm(dim=-1)
                        self.norms.setdefault(name, []).append(float(norms.mean().item()))
                    else:
                        self.norms.setdefault(name, []).append(float(t.float().norm().item()))
        self._hooks.append(module.register_forward_hook(hook_fn))

    def attach_to_blocks(self, model: nn.Module, block_types: Tuple[type, ...]):
        for name, module in model.named_modules():
            if isinstance(module, block_types):
                self.attach_to(module, name)

    def reset(self):
        self.norms.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False

    def summary(self) -> Dict[str, Any]:
        if not self.norms:
            return {}
        names = list(self.norms.keys())
        means = [float(np.mean(v)) for v in self.norms.values()]

        # Detect exponential growth: fit log(norm) vs depth, check slope
        slope = None
        if len(means) >= 3 and all(m > 0 for m in means):
            x = np.arange(len(means))
            y = np.log(means)
            slope = float(np.polyfit(x, y, 1)[0])

        ok = True
        alerts = []
        if slope is not None and slope > 0.5:
            ok = False
            alerts.append(f"Exponential residual norm growth: slope(log)={slope:.2f}")
        if any(m < 1e-6 for m in means):
            ok = False
            alerts.append("Residual stream collapse to ~0 in at least one layer")

        return {
            'ok': bool(ok),
            'layers': names,
            'norm_means': [round(m, 4) for m in means],
            'log_slope': round(slope, 4) if slope is not None else None,
            'alerts': alerts,
        }


# ─────────────────────────────────────────────
#  Position encoding drift
# ─────────────────────────────────────────────

def positional_encoding_drift(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> Dict[str, Any]:
    """Cosine similarity between current and reference positional embeddings.

    Useful to track whether learnable positional embeddings have drifted far
    from init (or from a prior epoch).

    Shapes: (max_len, D).
    """
    a = current.detach().float().cpu().flatten()
    b = reference.detach().float().cpu().flatten()
    if a.numel() != b.numel() or a.numel() == 0:
        return {'ok': False, 'msg': f"shape mismatch: {current.shape} vs {reference.shape}"}
    cos = float((a @ b) / (a.norm() * b.norm() + 1e-12))
    l2 = float((a - b).norm())
    return {
        'ok': bool(cos > 0.3),
        'cosine_similarity': round(cos, 4),
        'l2_distance': round(l2, 4),
    }


# ─────────────────────────────────────────────
#  TCN dilation coverage
# ─────────────────────────────────────────────

def tcn_receptive_field(
    kernel_sizes: Sequence[int],
    dilations: Sequence[int],
    required_length: int,
) -> Dict[str, Any]:
    """Compute theoretical receptive field of a stacked dilated conv network.

    RF = 1 + sum_i (k_i - 1) * d_i

    For TCN with residual blocks that stack two convs per block, pass the full
    flat lists of (k, d) in execution order.
    """
    if len(kernel_sizes) != len(dilations):
        return {'ok': False, 'msg': 'length mismatch'}
    rf = 1 + sum((k - 1) * d for k, d in zip(kernel_sizes, dilations))
    return {
        'ok': bool(rf >= required_length),
        'receptive_field': int(rf),
        'required_length': int(required_length),
        'coverage_ratio': round(rf / max(required_length, 1), 3),
        'n_layers': int(len(kernel_sizes)),
    }


# ─────────────────────────────────────────────
#  Per-layer causal leakage (for stacked blocks)
# ─────────────────────────────────────────────

def check_layer_causal_leakage(
    model: nn.Module,
    loader,
    device,
    block_types: Tuple[type, ...],
    time_dim: int = 1,
    t_probe: Optional[int] = None,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    """Per-block causal leakage test: hooks collect each block's output,
    compares originals vs inputs perturbed after t_probe.

    Returns per-block max_diff with ok flag.
    """
    captured_orig: Dict[str, torch.Tensor] = {}
    captured_pert: Dict[str, torch.Tensor] = {}
    hooks: List = []

    def make_hook(store):
        def hook_fn(mod, inp, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(t):
                store[str(id(mod))] = t.detach().cpu()
        return hook_fn

    try:
        for name, mod in model.named_modules():
            if isinstance(mod, block_types):
                hooks.append(mod.register_forward_hook(make_hook(captured_orig)))

        batch = next(iter(loader))
        if not isinstance(batch, (list, tuple)):
            return {'ok': False, 'msg': 'loader must yield tuples'}
        features = batch[0].to(device).clone().detach()
        extras = batch[1].to(device) if len(batch) == 3 else None

        L = int(features.shape[time_dim])
        if t_probe is None:
            t_probe = L // 2

        was_training = model.training
        model.eval()

        with torch.no_grad():
            _ = model(features) if extras is None else _safe_forward(model, features, extras)

        # Swap stores
        for h in hooks:
            h.remove()
        hooks = []
        for name, mod in model.named_modules():
            if isinstance(mod, block_types):
                hooks.append(mod.register_forward_hook(make_hook(captured_pert)))

        perturbed = features.clone()
        sl = [slice(None)] * features.dim()
        sl[time_dim] = slice(t_probe + 1, L)
        perturbed[tuple(sl)] = torch.randn_like(perturbed[tuple(sl)]) * 10.0

        with torch.no_grad():
            _ = model(perturbed) if extras is None else _safe_forward(model, perturbed, extras)

        if was_training:
            model.train()

    finally:
        for h in hooks:
            h.remove()

    diffs = {}
    any_leak = False
    for k, t_orig in captured_orig.items():
        t_pert = captured_pert.get(k)
        if t_pert is None or t_pert.shape != t_orig.shape:
            continue
        # Compare past slice along time_dim (t_orig may be (B,L,D) or (B,D,L))
        if t_orig.dim() >= 3 and t_orig.shape[-2] >= L:
            past_o = t_orig[..., :t_probe + 1, :]
            past_p = t_pert[..., :t_probe + 1, :]
        elif t_orig.dim() >= 3 and t_orig.shape[-1] >= L:
            past_o = t_orig[..., :t_probe + 1]
            past_p = t_pert[..., :t_probe + 1]
        else:
            past_o = t_orig
            past_p = t_pert
        d = float((past_o - past_p).abs().max().item())
        leak = d > tol
        if leak:
            any_leak = True
        diffs[k] = {'max_abs_diff': d, 'ok': bool(not leak)}

    return {
        'ok': bool(not any_leak),
        'per_block': diffs,
        't_probe': int(t_probe),
        'tolerance': float(tol),
    }


def _safe_forward(model, features, extras):
    try:
        return model(features, global_features=extras)
    except TypeError:
        return model(features)
