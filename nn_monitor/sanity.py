"""
Pre-training sanity checks (inspired by Karpathy's "Recipe for Training Neural Networks").
"""

import copy
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _unpack_batch(batch, device):
    """Unpack a loader batch regardless of its arity.

    Supported layouts:
      2-tuple: (features, targets)
      3-tuple: (features, global_features, targets)
      5-tuple: (features, global_features, targets, sample_weights, swing_amp)
               — emitted by CudaBatchLoader when swing_amplitude_pct is present

    ``targets`` is always the 3rd element (index 2) for 3-and-5-tuples, or the
    last element for 2-tuples.  Using ``batch[-1]`` broke when the 5-tuple was
    introduced because it picked up swing_amplitude_pct instead of the class
    labels, causing out-of-bounds scatter inside SwingLoss.
    """
    if not isinstance(batch, (list, tuple)):
        raise ValueError("Loader must yield tuple/list")
    n = len(batch)
    features = batch[0].to(device)
    if n == 2:
        # (features, targets)
        targets = batch[1].to(device)
        extras = None
    elif n == 3:
        # (features, global_features, targets)
        extras = batch[1].to(device) if batch[1] is not None else None
        targets = batch[2].to(device)
    else:
        # 4- or 5-tuple: (features, global_features, targets, ...)
        # targets is always at index 2; ignore sample_weights / swing_amp
        extras = batch[1].to(device) if batch[1] is not None else None
        targets = batch[2].to(device)
    return features, extras, targets


def _forward(model, features, extras):
    """Call model with or without `global_features=` kwarg."""
    if extras is None:
        return model(features)
    try:
        return model(features, global_features=extras)
    except TypeError:
        return model(features)


def check_loss_at_init(model, loader, criterion, device, n_classes=None) -> Dict[str, float]:
    """Verify loss at init ≈ -log(1/n_classes).

    If loss deviates >30% from expected, something is wrong
    with the model, loss function, or data pipeline.

    Args:
        model: untrained model
        loader: DataLoader (train or val)
        criterion: loss function
        device: torch device
        n_classes: number of classes (auto-detected from first batch if None)
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        features, extras, targets = _unpack_batch(batch, device)
        outputs = _forward(model, features, extras)
        raw_loss = criterion(outputs, targets)
        # SwingLoss returns a dict {'loss': tensor, ...}; plain losses return a scalar tensor.
        if isinstance(raw_loss, dict):
            loss = raw_loss['loss'].item()
        else:
            loss = raw_loss.item()

        if n_classes is None:
            if outputs.dim() == 2 and outputs.shape[1] > 1:
                n_classes = outputs.shape[1]
            elif outputs.dim() >= 2 and outputs.shape[1] == 1:
                n_classes = 2  # sigmoid logit
            elif outputs.dim() == 1:
                n_classes = 2  # BCE-style
            else:
                n_classes = int(outputs.shape[-1])

    expected = float(-np.log(1.0 / n_classes))
    deviation = abs(loss - expected) / expected if expected > 0 else 0.0
    ok = deviation < 0.3

    result = {
        'actual_loss': round(loss, 4),
        'expected_loss': round(expected, 4),
        'deviation_pct': round(deviation * 100, 1),
        'n_classes': n_classes,
        'ok': bool(ok),
    }
    if not ok:
        logger.warning(f"Loss at init ({loss:.4f}) deviates {deviation*100:.0f}% from expected ({expected:.4f})")
    else:
        logger.info(f"Loss at init OK: {loss:.4f} (expected ~{expected:.4f})")
    return result


def check_overfit_one_batch(
    model, loader, criterion, optimizer, device, n_steps=100, target_acc=90.0,
) -> Dict[str, Any]:
    """Try to overfit a single batch to near-100% accuracy.

    If model can't memorize one batch in 100 steps, something
    is fundamentally broken (architecture, loss, data pipeline).

    Restores model and optimizer state after the test.
    """
    model.train()

    batch = next(iter(loader))
    features, extras, targets = _unpack_batch(batch, device)

    # Save original state (deepcopy — optimizer state tensors are mutated in-place by step())
    original_state = copy.deepcopy(model.state_dict())
    original_opt_state = copy.deepcopy(optimizer.state_dict())

    losses = []
    accs = []
    for step in range(n_steps):
        optimizer.zero_grad()
        outputs = _forward(model, features, extras)
        raw_loss = criterion(outputs, targets)
        # SwingLoss returns a dict {'loss': tensor, ...}; plain losses return a scalar tensor.
        loss = raw_loss['loss'] if isinstance(raw_loss, dict) else raw_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if outputs.dim() >= 2 and outputs.shape[-1] > 1:
                preds = outputs.argmax(dim=-1)
                acc = (preds == targets).float().mean().item() * 100
            else:
                # Binary sigmoid case
                preds = (outputs.view(-1) > 0).long()
                t = targets.view(-1).long()
                acc = (preds == t).float().mean().item() * 100
        losses.append(loss.item())
        accs.append(acc)

    # Restore original state
    model.load_state_dict(original_state)
    optimizer.load_state_dict(original_opt_state)

    final_acc = accs[-1]
    ok = final_acc > target_acc

    result = {
        'final_accuracy': round(final_acc, 1),
        'final_loss': round(losses[-1], 4),
        'initial_loss': round(losses[0], 4),
        'loss_at_step_10': round(losses[min(9, len(losses) - 1)], 4),
        'n_steps': n_steps,
        'target_acc': target_acc,
        'ok': ok,
    }
    if ok:
        logger.info(f"Overfit-one-batch OK: {final_acc:.1f}% in {n_steps} steps")
    else:
        logger.warning(f"Overfit-one-batch FAILED: {final_acc:.1f}% after {n_steps} steps")
    return result


def verify_gradient_flow(model) -> Dict[str, Any]:
    """Check that gradients reach ALL layers after one backward pass.

    Call after loss.backward() but before optimizer.step().
    """
    no_grad = []
    zero_grad = []
    healthy = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            no_grad.append(name)
        elif param.grad.norm() == 0:
            zero_grad.append(name)
        else:
            healthy.append(name)

    ok = len(no_grad) == 0 and len(zero_grad) == 0
    result = {
        'ok': ok,
        'healthy_layers': len(healthy),
        'no_grad_layers': no_grad,
        'zero_grad_layers': zero_grad,
    }
    if not ok:
        logger.warning(f"Gradient flow broken: {len(no_grad)} layers with no grad, {len(zero_grad)} with zero grad")
    else:
        logger.info(f"Gradient flow OK: all {len(healthy)} layers receiving gradients")
    return result


def check_receptive_field_gradients(
    model, loader, criterion, device,
    time_dim: Optional[int] = None,
    vanish_threshold: float = 1e-8,
) -> Dict[str, Any]:
    """Check if gradients reach the entire historical receptive window.

    Designed for TCN and Transformers on time-series.
    Verifies that dL/dx does not vanish for old time steps.

    Args:
        time_dim: explicit sequence-axis index on the input tensor.
                  If None — heuristic (argmax of non-batch axes).
        vanish_threshold: |grad| below this → oldest token considered dead.

    Side-effect safety:
        - uses model.eval() to avoid perturbing BatchNorm running stats
        - clears any accumulated grads on model parameters on exit
    """
    was_training = model.training
    model.eval()

    batch = next(iter(loader))
    features, extras, targets = _unpack_batch(batch, device)
    features = features.clone().detach().requires_grad_(True)

    # Clear any stale grads on model params before our backward
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    try:
        outputs = _forward(model, features, extras)
        loss = criterion(outputs, targets)
        loss.backward()

        if features.grad is None:
            logger.warning("Features have no gradient. Receptive field test failed.")
            return {'ok': False, 'msg': "Features received no gradients"}

        if features.dim() < 3:
            return {'ok': True, 'msg': "Input is not 3D, skipping receptive field test."}

        # Determine time axis:
        # - explicit param wins
        # - else pick the longest non-batch axis (axis 0 is batch by convention)
        if time_dim is None:
            non_batch_axes = list(range(1, features.dim()))
            sizes = [features.shape[i] for i in non_batch_axes]
            time_dim = non_batch_axes[int(np.argmax(sizes))]
        elif time_dim < 0:
            time_dim = features.dim() + time_dim

        other_dims = [i for i in range(features.dim()) if i != time_dim]
        grad = features.grad.abs().mean(dim=other_dims)
        L = int(grad.shape[0])

        if L < 2:
            return {'ok': True, 'msg': "Sequence length < 2, test not applicable"}

        oldest_grad = float(grad[0].item())
        newest_grad = float(grad[-1].item())
        mid_grad = float(grad[L // 2].item())

        vanished_history = oldest_grad < vanish_threshold

        result = {
            'ok': bool(not vanished_history),
            'oldest_token_grad': oldest_grad,
            'middle_token_grad': mid_grad,
            'newest_token_grad': newest_grad,
            'oldest_over_newest_ratio': oldest_grad / (newest_grad + 1e-12),
            'sequence_length': L,
            'time_dim': int(time_dim),
            'grad_profile': [float(x) for x in grad.detach().cpu().tolist()],
        }

        if vanished_history:
            logger.warning(
                f"Receptive Field Collapse: grad at t=0 is {oldest_grad:.2e} "
                f"(ratio to t=L-1: {result['oldest_over_newest_ratio']:.2e})"
            )
        else:
            logger.info(
                f"Receptive Field OK: grad at t=0 is {oldest_grad:.2e} "
                f"(newest={newest_grad:.2e})"
            )
        return result
    finally:
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        if was_training:
            model.train()


def check_causal_leakage(
    model, loader, device,
    time_dim: Optional[int] = None,
    t_probe: Optional[int] = None,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    """Detect future-information leakage in causal/autoregressive models.

    Perturbs inputs at positions strictly AFTER `t_probe` and checks whether
    the output at `t_probe` (or the sequence output up to `t_probe`) changes.
    A correctly-causal model must produce identical outputs.

    This is the #1 silent bug in TCN/Transformer on time-series.

    Args:
        time_dim: sequence axis (None → longest non-batch axis)
        t_probe: position to check (default: L // 2)
        tol: max permissible absolute difference
    """
    was_training = model.training
    model.eval()

    batch = next(iter(loader))
    features, extras, _ = _unpack_batch(batch, device)
    features = features.clone().detach()

    if features.dim() < 3:
        return {'ok': True, 'msg': "Input not 3D, causal test N/A"}

    if time_dim is None:
        non_batch_axes = list(range(1, features.dim()))
        sizes = [features.shape[i] for i in non_batch_axes]
        time_dim = non_batch_axes[int(np.argmax(sizes))]
    elif time_dim < 0:
        time_dim = features.dim() + time_dim

    L = int(features.shape[time_dim])
    if L < 4:
        return {'ok': True, 'msg': f"Sequence too short ({L}) for causal probe"}

    if t_probe is None:
        t_probe = L // 2

    try:
        with torch.no_grad():
            out_orig = _forward(model, features, extras)
            if not torch.is_tensor(out_orig):
                return {'ok': True, 'msg': "Non-tensor output, cannot compare"}

            perturbed = features.clone()
            future_slice = [slice(None)] * features.dim()
            future_slice[time_dim] = slice(t_probe + 1, L)
            perturbed[tuple(future_slice)] = torch.randn_like(perturbed[tuple(future_slice)]) * 10.0

            out_pert = _forward(model, perturbed, extras)

            # If model returns sequence output, compare up to t_probe
            if out_orig.dim() >= 3 and out_orig.shape[-2] == L:
                # Assume (B, L, C) or similar with seq on dim -2
                past = out_orig[..., :t_probe + 1, :]
                past_pert = out_pert[..., :t_probe + 1, :]
                max_diff = float((past - past_pert).abs().max().item())
            elif out_orig.dim() >= 3 and out_orig.shape[-1] == L:
                past = out_orig[..., :t_probe + 1]
                past_pert = out_pert[..., :t_probe + 1]
                max_diff = float((past - past_pert).abs().max().item())
            else:
                # Scalar/aggregated output — compare whole
                max_diff = float((out_orig - out_pert).abs().max().item())
    finally:
        if was_training:
            model.train()

    leakage = max_diff > tol
    result = {
        'ok': bool(not leakage),
        'max_abs_diff': max_diff,
        'tolerance': tol,
        't_probe': int(t_probe),
        'sequence_length': L,
    }
    if leakage:
        logger.warning(
            f"CAUSAL LEAKAGE: output at t<={t_probe} changed by {max_diff:.2e} "
            f"after perturbing t>{t_probe}. Model is NOT causal."
        )
    else:
        logger.info(f"Causal OK: output invariant up to t={t_probe} (diff={max_diff:.2e})")
    return result


def check_time_split(
    train_times: np.ndarray,
    val_times: np.ndarray,
    min_gap: float = 0.0,
) -> Dict[str, Any]:
    """Verify that validation timestamps are strictly after training timestamps.

    Accepts numeric timestamps (unix seconds / ordinal / any comparable scalar).
    Detects the #1 silent bug in time-series ML: random shuffle splits that
    leak future info into training.
    """
    train_times = np.asarray(train_times).flatten()
    val_times = np.asarray(val_times).flatten()

    if len(train_times) == 0 or len(val_times) == 0:
        return {'ok': False, 'msg': 'empty train or val'}

    train_max = float(np.max(train_times))
    val_min = float(np.min(val_times))
    gap = val_min - train_max

    # Overlap: any val sample timestamp <= any train sample timestamp
    overlap_count = int(np.sum(val_times <= train_max))
    ok = overlap_count == 0 and gap >= min_gap

    result = {
        'ok': bool(ok),
        'train_max': train_max,
        'val_min': val_min,
        'gap': gap,
        'required_min_gap': min_gap,
        'overlap_count': overlap_count,
    }
    if not ok:
        logger.warning(
            f"TIME LEAKAGE: {overlap_count} val samples <= train_max; "
            f"gap={gap:.3f} (required {min_gap})"
        )
    else:
        logger.info(f"Time-split OK: gap={gap:.3f}")
    return result
