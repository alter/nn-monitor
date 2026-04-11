# nn-monitor

Training monitoring and diagnostics framework for neural networks (PyTorch). Architecture-agnostic — works with any model.

## What it does

During training, many things can go wrong silently: vanishing gradients, overfitting, miscalibrated confidence, layers that stop learning. These are usually discovered after deployment — when the model already fails in production.

nn-monitor automatically checks model health every epoch and warns about problems before they become critical:

- **Before training**: verifies model, loss, and data are correct (sanity checks)
- **During training**: tracks 15+ health metrics every epoch
- **After training**: generates summary report and diagnostic plots

## How it works

The framework **does not know** which model to monitor. You pass your model into every call — the framework analyzes it through standard PyTorch API (`named_parameters`, `register_forward_hook`, `state_dict`).

### Inputs

| What you pass | Type | Purpose |
|---------------|------|---------|
| `model` | `nn.Module` | Any PyTorch model — framework reads weights, gradients, activations |
| `val_probs` | `np.ndarray (N, C)` | Softmax probabilities on validation set — for calibration and entropy |
| `val_targets` | `np.ndarray (N,)` | Ground truth labels (int) — for accuracy and ECE |
| `train_loss`, `val_loss` | `float` | Losses — for overfit detection |
| `train_acc`, `val_acc` | `float` | Accuracy — for gap tracking |
| `learning_rate` | `float` | Current LR — for plots |
| `loader` | `DataLoader` | Only for sanity checks (one batch) |
| `criterion` | loss function | Only for sanity checks |
| `optimizer` | optimizer | Only for sanity checks (overfit-one-batch) |

### Outputs

Each epoch generates:
- **JSON file** with 30+ metrics (`epoch_000.json`)
- **Reliability diagram** — calibration plot (`reliability_000.png`)
- **Weight update ratios** — per-layer training health (`update_ratios_000.png`)

After training:
- **Training summary** — all metrics across all epochs in one JSON (`training_summary.json`)
- **Training curves** — 6-panel plot: loss, accuracy, ECE, confidence gap, entropy, update ratio (`training_curves.png`)

```
output/diagnostics/
  sanity_checks.json           # pre-training check results
  epoch_000.json               # epoch 0 metrics
  epoch_005.json               # epoch 5 metrics
  ...
  reliability_000.png          # calibration diagram
  reliability_005.png
  update_ratios_000.png        # per-layer weight health
  training_summary.json        # all epochs compiled
  training_curves.png          # 6-panel summary plot
```

## Installation

```bash
pip install -e nn-monitor/
# or just copy nn_monitor/ into your project
```

Dependencies: `numpy`, `torch`, `matplotlib`. Optional: `lightgbm`, `scikit-learn`.

## Full Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from nn_monitor import TrainingMonitor

# --- Your model (any architecture) ---
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 classes
        )
    def forward(self, x):
        return self.net(x)

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- Data ---
X_train = torch.randn(1000, 100)
y_train = torch.randint(0, 3, (1000,))
X_val = torch.randn(200, 100)
y_val = torch.randint(0, 3, (200,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# --- Monitoring ---
monitor = TrainingMonitor('./my_experiment')

# 1. Sanity checks (before training)
monitor.run_sanity_checks(model, train_loader, criterion, optimizer, device='cuda')
# Checks:
#   - loss at random weights ≈ -log(1/3) = 1.099?
#   - can model memorize 1 batch in 100 steps?

# 2. Training loop
for epoch in range(50):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

        # BEFORE optimizer.step() — snapshot weights
        if total == 0:  # once per epoch
            monitor.before_optimizer_step(model)

        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        # AFTER optimizer.step() — compute update ratios
        if total == 0:
            monitor.after_optimizer_step(model)

        train_loss += loss.item()
        correct += (out.argmax(1) == y_batch).sum().item()
        total += len(y_batch)

    train_loss /= len(train_loader)
    train_acc = correct / total * 100

    # Validation — need softmax probabilities
    model.eval()
    with torch.no_grad():
        val_out = model(X_val.cuda())
        val_probs = torch.softmax(val_out, dim=1).cpu().numpy()  # (N, 3) — MUST be softmax
        val_preds = val_out.argmax(1).cpu()
        val_acc = (val_preds == y_val).float().mean().item() * 100
        val_loss = criterion(val_out, y_val.cuda()).item()

    # 3. Log metrics (every 5 epochs or always)
    if epoch % 5 == 0:
        diag = monitor.log_epoch(
            epoch=epoch,
            model=model,
            val_probs=val_probs,             # (N, C) numpy — softmax probabilities
            val_targets=y_val.numpy(),       # (N,) numpy — ground truth labels
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            learning_rate=optimizer.param_groups[0]['lr'],
            class_names={0: 'cat', 1: 'dog', 2: 'bird'},  # optional
            full_diagnostics=(epoch % 10 == 0),  # SVD every 10 epochs (expensive)
        )
        # diag is a dict with all metrics — use in your code if needed

# 4. Summary after training
monitor.save_summary()
# Generates training_summary.json + training_curves.png
```

## What it checks and which metrics

### Sanity Checks (before training)

**Loss at init** — loss with random weights should be ≈ -log(1/C) where C = number of classes. For 3 classes = 1.099. Large deviation means a bug in the loss function, data, or architecture.

**Overfit one batch** — the model should memorize one batch to 90%+ accuracy in 100 steps. If it can't — architecture is too weak, loss is wrong, or data is broken. Model is restored to original weights after the test.

**Gradient flow** — checks that gradients reach ALL layers. If any layer has grad=None or grad=0 — that layer isn't learning.

### Calibration

**ECE (Expected Calibration Error)** — main calibration metric. When the model says "80% confident" — is it actually correct 80% of the time?

| ECE | Meaning |
|-----|---------|
| < 0.05 | Well calibrated |
| 0.05 - 0.15 | Average |
| > 0.15 | Poor — confidence threshold is useless |

**Prediction entropy** — Shannon entropy of softmax output. Normalized entropy > 0.95 means the model predicts nearly uniformly (can't distinguish classes). < 0.2 means overconfident (possibly overfit).

**Confidence gap** — difference in average confidence between correct and incorrect predictions. Gap ≈ 0 means the model is equally "confident" when right and when wrong — confidence is meaningless.

### Weight Health

**Weight update ratio** — ratio of |weight change| / |current weight| per optimizer step. Andrej Karpathy's golden metric.

| Ratio | Meaning |
|-------|---------|
| ~1e-3 | Healthy training |
| < 1e-5 | Layer barely learning (frozen) |
| > 0.1 | Unstable, LR too high |

**Dead neurons** — percentage of neurons with zero activations. > 30% = model is losing capacity. Checked via forward hooks.

**Effective rank** — how many dimensions the weight matrix actually uses (via SVD). Ratio 0.3-0.8 = healthy. < 0.1 = rank collapse (early overfit signal).

### Overfit Detection

**Train/val gap** — if train_loss falls while val_loss rises for N consecutive epochs — automatic alert "OVERFIT".

**Accuracy gap** — if train_acc - val_acc > 10pp — alert "GAP".

**Loss ratio** — if val_loss / train_loss > 1.5 — alert.

### Prediction Stability

**Temporal stability** — how often predictions change between consecutive samples. For time series: high change_rate = model is jittery.

**Noise stability** — add small noise to input, check how many predictions flip. flip_rate > 20% = model is on the decision boundary, unreliable.

## Standalone Functions (without orchestrator)

```python
from nn_monitor import compute_ece, prediction_entropy, effective_rank, collect_weight_stats

# Calibration
ece, bins = compute_ece(probs, targets)
print(f"ECE: {ece:.4f}")
for b in bins:
    print(f"  conf {b['range']}: acc={b['accuracy']:.2f}, count={b['count']}")

# Entropy
ent = prediction_entropy(probs)
print(f"Entropy: {ent['normalized_mean']:.3f} (1.0 = random, 0.0 = certain)")

# Weight health
stats = collect_weight_stats(model)
for layer, s in stats.items():
    print(f"{layer}: norm={s['frobenius_norm']:.2f}, dead={s['near_zero_pct']:.1f}%")

# Spectral analysis (per layer)
for name, param in model.named_parameters():
    if param.dim() >= 2:
        rank = effective_rank(param.data)
        max_rank = min(param.shape)
        print(f"{name}: rank={rank:.0f}/{max_rank} ({rank/max_rank:.1%})")
```

## LightGBM / XGBoost / Sklearn

Separate function for tree-based models:

```python
from nn_monitor import run_lgbm_diagnostics

diag = run_lgbm_diagnostics(
    model=trained_lgbm_model,             # fitted LGBMClassifier
    X_val=X_val,                          # (N, D) numpy
    y_val=y_val,                          # (N,) numpy
    feature_names=['feat_0', 'feat_1'],   # feature name list
    output_dir='./lgbm_results',
    class_names=['SELL', 'HOLD', 'BUY'],  # optional
)
# Generates:
#   lgbm_diagnostics.json    — ECE, accuracy, confidence, feature importance
#   reliability_lgbm.png     — calibration plot
#   confidence_histogram.png — confidence distribution (correct vs wrong)
```

## ActivationMonitor

Activation monitoring via forward hooks. Shows dead neurons:

```python
from nn_monitor import ActivationMonitor

act_mon = ActivationMonitor(model, layer_types=(nn.Linear, nn.Conv1d, nn.Conv2d))
model(sample_batch.cuda())  # one forward pass — hooks collect stats
stats = act_mon.summary()
print(f"Dead neurons: {stats['dead_neuron_pct_mean']:.1f}% (worst: {stats['worst_layer']})")
act_mon.remove()  # ALWAYS remove hooks when done
```

## OverfitDetector

Automatic alerts when train/val diverge:

```python
from nn_monitor import OverfitDetector

detector = OverfitDetector(patience=5, acc_gap_threshold=10.0)

for epoch in range(100):
    # ... train, validate ...
    alerts = detector.update(train_loss, val_loss, train_acc, val_acc, epoch)
    for alert in alerts:
        print(f"[Epoch {epoch}] {alert}")
        # "OVERFIT: val_loss rising for 5 epochs while train_loss falling"
        # "GAP: train_acc=85.0% val_acc=72.0% gap=13.0pp"
```

## Red Flags — when to worry

| What you see | Problem | What to do |
|-------------|---------|------------|
| ECE > 0.15 | Confidence is useless | Don't use confidence threshold |
| Entropy normalized > 0.95 | Model can't differentiate | Check data, architecture, loss |
| Confidence gap ≈ 0 | Model doesn't know when it's right | Add calibration loss |
| Update ratio < 1e-5 (all layers) | LR too small | Increase LR |
| Update ratio > 0.1 | LR too large | Decrease LR |
| Dead neurons > 30% | Capacity loss | Check init, LR, activation function |
| Effective rank < 0.1 | Rank collapse | Add regularization |
| Overfit alert | train↓ val↑ | Early stopping, dropout, data augmentation |
| Loss at init ≠ -log(1/C) | Bug in loss or data | Check loss function, labels, data pipeline |
| Can't overfit 1 batch | Architecture broken | Check forward pass, loss, gradients |
