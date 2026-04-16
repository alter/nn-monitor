"""
Pre-training sanity checks (inspired by Karpathy's "Recipe for Training Neural Networks").
"""

import logging
from typing import Dict, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
        if isinstance(batch, (list, tuple)):
            features = batch[0].to(device)
            targets = batch[-1].to(device)
            # Pass extra inputs if model accepts them
            if len(batch) == 3:
                try:
                    outputs = model(features, global_features=batch[1].to(device))
                except TypeError:
                    outputs = model(features)
            else:
                outputs = model(features)
        else:
            raise ValueError("Loader must yield (features, targets) or (features, extras, targets)")

        loss = criterion(outputs, targets).item()

        if n_classes is None:
            if outputs.dim() >= 2:
                n_classes = outputs.shape[1]
            else:
                n_classes = 2 # Best guess for 1D output (e.g. BCELoss)

    expected = -np.log(1.0 / n_classes)
    deviation = abs(loss - expected) / expected
    ok = deviation < 0.3

    result = {
        'actual_loss': round(loss, 4),
        'expected_loss': round(expected, 4),
        'deviation_pct': round(deviation * 100, 1),
        'n_classes': n_classes,
        'ok': ok,
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
    if isinstance(batch, (list, tuple)):
        features = batch[0].to(device)
        targets = batch[-1].to(device)
        extras = batch[1].to(device) if len(batch) == 3 else None
    else:
        raise ValueError("Loader must yield tuples")

    # Save original state
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    original_opt_state = optimizer.state_dict()

    losses = []
    accs = []
    for step in range(n_steps):
        optimizer.zero_grad()
        try:
            outputs = model(features, global_features=extras) if extras is not None else model(features)
        except TypeError:
            outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item() * 100
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


def check_receptive_field_gradients(model, loader, criterion, device) -> Dict[str, Any]:
    """Check if gradients reach the entire historical receptive window.
    
    Specially designed for TCN and Transformers on Time Series. 
    Verifies that the loss derivatives wrt the inputs do not vanish for old time steps.
    """
    model.train()
    
    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)):
        # Assume input format where L is sequence length
        features = batch[0].clone().detach().to(device)
        targets = batch[-1].to(device)
        extras = batch[1].to(device) if len(batch) == 3 else None
    else:
        raise ValueError("Loader must yield tuples")
        
    features.requires_grad_(True)
    model.zero_grad()
    
    try:
        outputs = model(features, global_features=extras) if extras is not None else model(features)
    except TypeError:
        outputs = model(features)
        
    loss = criterion(outputs, targets)
    loss.backward()
    
    if features.grad is None:
        logger.warning("Features have no gradient. Receptive field test failed.")
        return {'ok': False, 'msg': "Features received no gradients"}
        
    # Assume time axis is the last or second to last. Let's flatten all but the last dimension.
    # TCN/Transformer usually have (N, C, L) or (N, L, C). We'll assume the time axis is the one with length L > features.shape[1] or similar.
    # To be safe and architecture agnostic, we'll just take the gradient norm over the batch and channel features.
    
    # If the format is (N, C, L), dim=2 is L
    # If the format is (N, L, C), dim=1 is L
    if features.dim() < 3:
        return {'ok': True, 'msg': "Input is not 3D, skipping receptive field test."}
        
    time_dim = 2 if features.shape[2] > features.shape[1] else 1 
    other_dims = [i for i in range(features.dim()) if i != time_dim]
    
    grad = features.grad.abs().mean(dim=other_dims)
    L = grad.shape[0]
    
    if L < 2:
        return {'ok': True, 'msg': "Sequence length < 2, test not applicable"}
        
    oldest_grad = grad[0].item()
    newest_grad = grad[-1].item()
    
    vanished_history = oldest_grad < 1e-8
    
    result = {
        'ok': not vanished_history,
        'oldest_token_grad': oldest_grad,
        'newest_token_grad': newest_grad,
        'decay_ratio': oldest_grad / (newest_grad + 1e-8), 
        'sequence_length': L
    }
    
    if vanished_history:
        logger.warning(f"Receptive Field Collapse: gradient for oldest token is {oldest_grad:.2e}")
    else:
        logger.info(f"Receptive Field OK: gradient for oldest token is {oldest_grad:.2e}")
        
    return result
