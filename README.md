# nn-monitor

Training monitoring and diagnostics framework for neural networks (PyTorch) and classical models (HMM, LightGBM). Architecture-agnostic — works with any model.

## What it does

During training, many things go wrong silently: vanishing gradients, rank collapse, attention collapse, miscalibrated confidence, future-information leakage in time-series models, Baum-Welch non-convergence, layers that stop learning. These are usually discovered after deployment — when the model already fails in production.

nn-monitor checks model health every epoch and warns about problems before they become critical:

- **Before training** — sanity checks (loss-at-init, overfit one batch, gradient flow, receptive field, causal leakage, time-split)
- **During training** — 30+ health metrics per epoch
- **After training** — summary report and diagnostic plots
- **Model-specific** — dedicated diagnostics for HMM, LightGBM, and TCN/Transformer

## Modules

| Module | What it covers |
|--------|----------------|
| `core` | `TrainingMonitor`, `ActivationMonitor`, `OverfitDetector` — generic NN training loop |
| `metrics` | Calibration (ECE), entropy, weight/gradient stats, PSI drift, spectral rank, clip tracking, spike detection |
| `sanity` | Pre-training checks + time-series specific: receptive field, causal leakage, time split |
| `plots` | Reliability diagrams, gradient flow, training curves (8 panels), grad profile, attention heatmap |
| `hmm` | Transition matrix, state occupancy, dwell times, emission entropy, LL convergence, Viterbi stability |
| `transformer` | `AttentionMonitor`, `ResidualStreamMonitor`, head redundancy, attention collapse, TCN receptive field, per-block causal leakage |
| `lgbm` | LightGBM: per-iteration curves, tree structure, feature concentration, permutation vs gain disagreement, feature drift (PSI) |

## Installation

```bash
pip install -e nn-monitor/
# or copy nn_monitor/ into your project
```

Core deps: `numpy`, `torch`, `matplotlib`. Optional: `lightgbm`, `scikit-learn` (for LightGBM extras), `scipy` (for Hungarian-matched Viterbi stability), `hmmlearn` (for HMM orchestrator).

## TrainingMonitor — full example

```python
import torch, torch.nn as nn
from nn_monitor import TrainingMonitor

# Use as a context manager — automatically restores global state
# (e.g. torch.autograd.set_detect_anomaly) on exit.
with TrainingMonitor('./my_experiment', detect_anomalies=False) as monitor:
    monitor.run_sanity_checks(model, train_loader, criterion, optimizer, device)

    for epoch in range(50):
        monitor.before_optimizer_step(model)      # snapshot weights

        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.cuda()), y.cuda())
            loss.backward()
            optimizer.step()

        monitor.after_optimizer_step(model)       # compute |ΔW|/|W|

        val_probs, val_targets = validate(model, val_loader)  # (N, C), (N,)

        monitor.log_epoch(
            epoch=epoch, model=model,
            val_probs=val_probs, val_targets=val_targets,
            train_loss=train_loss, val_loss=val_loss,
            train_acc=train_acc, val_acc=val_acc,
            learning_rate=optimizer.param_groups[0]['lr'],
            data_time=data_t, compute_time=compute_t,   # optional
            full_diagnostics=(epoch % 10 == 0),         # SVD every 10 epochs
        )

    monitor.save_summary()
```

### Inputs

| What you pass | Type | Purpose |
|---------------|------|---------|
| `model` | `nn.Module` | Framework reads weights, gradients, activations |
| `val_probs` | `np.ndarray (N, C)` / `(N,)` / `(N, 1)` | Softmax probs or sigmoid outputs — auto-expanded for BCE |
| `val_targets` | `np.ndarray (N,)` | Ground truth labels (int) |
| `train_loss`, `val_loss`, `train_acc`, `val_acc` | `float` | Loss / accuracy curves + overfit detection |
| `learning_rate` | `float` or `dict` | Per-group LR supported |
| `data_time`, `compute_time` | `float` | Optional — tracked for performance plots |

### Outputs per epoch

`output/diagnostics/epoch_000.json` — 30+ metrics per epoch, including new `performance` panel (GPU memory, compute/data time), `learning_rate` + `learning_rates` keys for backward compatibility.

### Final outputs (after `save_summary()`)

- `training_summary.json` — all epochs aggregated
- `training_curves.png` — **8-panel** plot: loss, val_acc, ECE, confidence gap, entropy, update ratio, GPU max memory, epoch time (data vs compute)
- `sanity_checks.json` — pre-training check results
- `reliability_*.png`, `update_ratios_*.png` — per epoch plots

## Sanity checks — generic + time-series

```python
from nn_monitor import (
    check_loss_at_init, check_overfit_one_batch, verify_gradient_flow,
    check_receptive_field_gradients, check_causal_leakage, check_time_split,
)

# Generic
check_loss_at_init(model, loader, criterion, device)
check_overfit_one_batch(model, loader, criterion, optimizer, device, n_steps=100)
verify_gradient_flow(model)   # after loss.backward(), before optimizer.step()

# Time-series specific — critical for TCN / Transformer on sequential data
check_receptive_field_gradients(model, loader, criterion, device, time_dim=2)
# Verifies dL/dx does not vanish for old time steps. Returns grad_profile
# per position (oldest → newest).

check_causal_leakage(model, loader, device, time_dim=2, t_probe=None, tol=1e-5)
# Perturbs inputs at positions strictly AFTER t_probe and verifies the
# output at t<=t_probe does NOT change. Catches the #1 silent bug in
# causal models: forgotten causal mask, symmetric padding in TCN, etc.

check_time_split(train_times, val_times, min_gap=0)
# Verifies val timestamps are strictly after train timestamps. Catches
# random-shuffle splits that leak future into training.
```

## ActivationMonitor — Welford-corrected

```python
from nn_monitor import ActivationMonitor

# Tracks all common layers by default: Linear, Conv1d/2d/3d, LayerNorm,
# BatchNorm1d/2d, ReLU, GELU, SiLU, Tanh.
with ActivationMonitor(model) as mon:
    for batch in loader:
        model(batch)
    stats = mon.summary()
# Hooks auto-removed on __exit__.
# stats contains: dead_neuron_pct_mean/max, activation_std_mean/min,
# worst_layer, nan_layers, inf_layers.
# Call mon.reset() between epochs to avoid stale averaging.
```

Online statistics are computed via **Chan/Welford parallel variance** — simple per-batch std averaging is biased (Jensen's inequality) and was a bug in earlier versions.

## Transformer / TCN diagnostics

```python
from nn_monitor import (
    AttentionMonitor, ResidualStreamMonitor,
    attention_collapse_stats, head_redundancy,
    positional_encoding_drift, tcn_receptive_field,
    check_layer_causal_leakage,
)

# Attention health — auto-attaches to all nn.MultiheadAttention modules
with AttentionMonitor(model) as am:
    model(x)                        # forward
    report = am.summary()
# For each MHA layer: normalized_entropy_mean, max_prob_mean,
# diag_mass_mean, per_head_*, redundant head pairs (corr > 0.95).

# Custom attention? Attach manually:
with AttentionMonitor() as am:
    am.attach_to(my_custom_attn_module, 'block_0')
    model(x)

# Residual-stream norm across depth — detect blow-up / collapse
with ResidualStreamMonitor() as rs:
    rs.attach_to_blocks(model, block_types=(MyTransformerBlock,))
    model(x)
    report = rs.summary()  # log_slope > 0.5 → exponential growth

# Positional-encoding drift from reference (e.g. init snapshot)
positional_encoding_drift(model.pos_embed.weight, reference_snapshot)

# TCN theoretical receptive field
tcn_receptive_field(kernel_sizes=[3]*6, dilations=[1,2,4,8,16,32],
                    required_length=128)  # returns coverage_ratio

# Per-block causal leakage (finds which block leaks)
check_layer_causal_leakage(model, loader, device,
                           block_types=(MyTCNBlock, MyTransformerBlock),
                           time_dim=1)
```

## HMM diagnostics

```python
from nn_monitor import (
    run_hmm_diagnostics,
    check_transition_matrix, state_occupancy, dwell_times,
    emission_entropy, gaussian_emission_separability,
    check_ll_convergence, viterbi_stability,
    check_forward_backward_stability,
)

# One-shot diagnostics for a fitted hmmlearn model
diag = run_hmm_diagnostics(
    model=fitted_hmm,
    X=observations,             # (N, D)
    output_dir='./hmm_results',
    ll_history=ll_per_iter,     # list of per-iteration log-likelihoods
    extra_paths=[path_seed1, path_seed2],  # optional: Viterbi paths from refits
)

# Or use individual checks
check_transition_matrix(model.transmat_)
# row-stochastic? absorbing states? spectral gap (mixing speed)?

state_occupancy(viterbi_path, n_states=model.n_components)
# dominant_share > 0.8 → degenerate HMM, one state eats everything

emission_entropy(model.emissionprob_)            # categorical HMM
gaussian_emission_separability(model.means_, model.covars_)  # Gaussian HMM
# min_pairwise_tv_distance < 0.1 → indistinguishable states

check_ll_convergence(ll_history)
# Baum-Welch LL must be monotonically non-decreasing — any drop = bug

viterbi_stability([path_seed0, path_seed1, path_seed2])
# Permutation-invariant agreement between Viterbi paths from different
# seeds. Uses Hungarian matching (label-switching aware).

check_forward_backward_stability(log_likelihoods_per_sample)
# -inf / nan in per-sample score → underflow without log-space
```

## LightGBM diagnostics

```python
from nn_monitor import run_lgbm_diagnostics

diag = run_lgbm_diagnostics(
    model=trained_lgbm,             # fitted LGBMClassifier
    X_val=X_val, y_val=y_val,
    feature_names=feature_names,
    output_dir='./lgbm_results',
    class_names=['SELL', 'HOLD', 'BUY'],
    X_train=X_train,                # enables feature_drift (per-feature PSI)
    evals_result=booster.evals_result_,  # enables training_curve_stats
    run_permutation=True,           # enables permutation vs gain disagreement
)
```

New in v1.1:
- `training_curve_stats` — best_iter, overfit_onset_iter, final train/val gap
- `tree_structure_stats` — leaf count / depth distribution, split-gain quartiles, tail-fraction of no-gain splits
- `feature_concentration` — top-K share of total gain (brittleness signal)
- `importance_disagreement` — Spearman ρ between gain vs permutation importance (low ρ = brittle / leakage-prone)
- `feature_drift` — per-feature PSI train→val

## Standalone metrics

```python
from nn_monitor import (
    compute_ece, prediction_entropy, confidence_gap,
    compute_psi, compute_attention_entropy,
    collect_weight_stats, collect_gradient_stats,
    effective_rank, track_effective_ranks,
    temporal_stability, prediction_stability_noise,
    GradientClipTracker, detect_loss_spike,
)

# Drift (Population Stability Index)
psi = compute_psi(train_feature, val_feature)
# psi_total < 0.1 none, 0.1–0.2 moderate, > 0.2 significant

# Weight health (now includes quantiles q05/q25/q50/q75/q95 and excess kurtosis)
stats = collect_weight_stats(model)

# Gradient health (now reports NaN/Inf per-layer lists and total grad norm)
grad = collect_gradient_stats(model)

# Gradient clipping statistics
tracker = GradientClipTracker()
for batch in loader:
    loss.backward()
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    tracker.update(total_norm=total.item(), clip_value=1.0)
    optimizer.step()
# tracker.summary(): clip_rate > 0.5 → LR likely too high; < 0.01 → clip too loose

# Loss spike detection (rolling-median)
spike = detect_loss_spike(loss_history, window=20, factor=3.0)
```

## Red flags — what to worry about

| Signal | Problem | Action |
|--------|---------|--------|
| ECE > 0.15 | Confidence is useless | Don't threshold on confidence |
| Normalized entropy > 0.95 | Model can't differentiate | Check data, architecture, loss |
| Confidence gap ≈ 0 | Confidence is meaningless | Add calibration loss |
| Update ratio < 1e-5 | Layer frozen | Increase LR / check init |
| Update ratio > 0.1 | Unstable | Decrease LR |
| Dead neurons > 30% | Capacity loss | Check init, activation, LR |
| Effective rank < 0.1 | Rank collapse | Regularization / reduce capacity |
| Grad NaN/Inf layers non-empty | Numerical blow-up | Add grad clipping, reduce LR, use fp32 |
| Clip rate > 0.5 | LR too high (clip as crutch) | Decrease LR |
| `check_causal_leakage` fails | Future info leaks to past | Fix causal mask / conv padding |
| `check_time_split` fails | Val overlaps with train time | Use time-based split |
| Receptive field t=0 grad → 0 | Vanishing through depth | Add dilation / residuals / more RF |
| Overfit alert (5 epochs) | train↓ val↑ | Early stop / dropout / aug |
| Loss spike alert | 3× rolling median | Save checkpoint, reduce LR |
| HMM absorbing states | Trained stuck | Reinit / prior on transition |
| HMM LL drop during fit | Numerical bug | Check log-space, scaling |
| HMM dominant_share > 0.8 | Degenerate (1 state) | Reinit / regularize emissions |
| Viterbi agreement < 0.5 (across seeds) | Multi-modal / bad init | More restarts / sticky prior |
| Attention max_prob > 0.95 | Head collapsed | Regularize / prune head |
| Residual stream log_slope > 0.5 | Exponential blow-up | LayerNorm / lower LR / deepnorm |
| LightGBM top-5 share > 0.8 | Brittle model | More features / regularize |
| LightGBM gain-vs-permutation ρ < 0.5 | Leakage or brittle | Investigate top gain features |

## Version

Current: **1.1.0**

See commit history for breaking-change notes (`learning_rate` key is preserved alongside new `learning_rates` for backward compatibility).
