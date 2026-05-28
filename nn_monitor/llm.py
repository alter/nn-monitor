"""
Metrics specific to large language model / autoregressive training.

The generic ``TrainingMonitor`` is classification-oriented (softmax probs,
argmax accuracy, ECE). LM pre-training/fine-tuning is step-based and judged by
token-level loss, perplexity, throughput, and hardware utilization. These are
pure functions plus two small stateful trackers; wire them into ``log_step``.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────
#  Loss-derived language-modeling metrics
# ─────────────────────────────────────────────

def perplexity_from_loss(mean_ce_nats: float) -> float:
    """Perplexity = exp(mean token cross-entropy in nats).

    Sanity anchor: at init a model over a vocab of V should sit near
    perplexity ≈ V (uniform guessing). A value far below that at step 0 hints
    at a label/shift bug; a value that plateaus near V means no learning.
    """
    if mean_ce_nats is None or not math.isfinite(mean_ce_nats):
        return float('nan')
    # clamp to avoid overflow on a diverged loss
    return float(math.exp(min(mean_ce_nats, 50.0)))


def bits_per_token(mean_ce_nats: float) -> float:
    """Cross-entropy expressed in bits/token (nats / ln 2)."""
    if mean_ce_nats is None or not math.isfinite(mean_ce_nats):
        return float('nan')
    return float(mean_ce_nats / math.log(2.0))


def bits_per_byte(mean_ce_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Bits-per-byte — tokenizer-independent compression metric.

    bpb = (total nats / ln 2) / n_bytes = bits_per_token * n_tokens / n_bytes.
    Use this to compare models with different tokenizers/vocabularies fairly.
    """
    if n_bytes <= 0 or not math.isfinite(mean_ce_nats):
        return float('nan')
    total_bits = (mean_ce_nats / math.log(2.0)) * n_tokens
    return float(total_bits / n_bytes)


# ─────────────────────────────────────────────
#  Throughput & hardware utilization
# ─────────────────────────────────────────────

def token_throughput(n_tokens: int, seconds: float) -> float:
    """Tokens processed per second."""
    if seconds <= 0:
        return 0.0
    return float(n_tokens / seconds)


def estimate_mfu(
    n_params: int,
    tokens_per_sec: float,
    peak_flops: float,
    *,
    seq_len: Optional[int] = None,
    n_layers: Optional[int] = None,
    d_model: Optional[int] = None,
) -> Dict[str, float]:
    """Model FLOPs Utilization (PaLM, Chowdhery et al. 2022).

    Uses the dense-transformer approximation of ~6N FLOPs per token for a full
    forward+backward step (2N forward, 4N backward). If seq_len/n_layers/d_model
    are given, adds the attention term 12 * n_layers * d_model * seq_len per
    token, which matters at long context.

    Args:
        n_params: total trainable parameters N.
        tokens_per_sec: measured throughput.
        peak_flops: hardware peak (e.g. A100 bf16 ≈ 312e12, H100 bf16 ≈ 989e12).

    Returns dict with achieved_flops_per_token, achieved_flops, and mfu in [0, 1].
    """
    flops_per_token = 6.0 * n_params
    if seq_len and n_layers and d_model:
        flops_per_token += 12.0 * n_layers * d_model * seq_len
    achieved = flops_per_token * tokens_per_sec
    mfu = achieved / peak_flops if peak_flops > 0 else float('nan')
    return {
        'flops_per_token': float(flops_per_token),
        'achieved_flops': float(achieved),
        'mfu': round(float(mfu), 4),
        'tokens_per_sec': float(tokens_per_sec),
    }


# ─────────────────────────────────────────────
#  Gradient noise scale (McCandlish et al. 2018)
# ─────────────────────────────────────────────

def gradient_noise_scale(
    g_small_sq: float, b_small: int,
    g_big_sq: float, b_big: int,
) -> Dict[str, float]:
    """Simple noise scale B_simple = tr(Σ) / |G|^2 from two batch sizes.

    The squared gradient norm is a biased estimator of the true |G|^2:
        E[|G_B|^2] = |G|^2 + tr(Σ)/B.
    Given estimates at two batch sizes b_small < b_big we can solve for both:
        |G|^2  ≈ (b_big * g_big_sq - b_small * g_small_sq) / (b_big - b_small)
        tr(Σ)  ≈ (g_small_sq - g_big_sq) / (1/b_small - 1/b_big)

    B_simple is the batch size at which gradient signal ≈ noise. Training with
    batch ≫ B_simple wastes compute; ≪ B_simple is noisy. Track its trend, not
    a single noisy estimate — see GradientNoiseTracker.

    g_*_sq are squared L2 norms of gradient estimates at the two batch sizes.
    """
    if b_big == b_small:
        return {'ok': False, 'msg': 'b_big must differ from b_small'}
    g2 = (b_big * g_big_sq - b_small * g_small_sq) / (b_big - b_small)
    inv = (1.0 / b_small) - (1.0 / b_big)
    tr_sigma = (g_small_sq - g_big_sq) / inv if inv != 0 else float('nan')
    b_simple = tr_sigma / g2 if g2 > 0 else float('nan')
    return {
        'grad_norm_sq': float(g2),
        'trace_sigma': float(tr_sigma),
        'b_simple': float(b_simple),
    }


class GradientNoiseTracker:
    """EMA-smoothed gradient noise scale.

    Per-step estimates of tr(Σ) and |G|^2 are extremely noisy; the paper
    recommends averaging numerator and denominator separately. Feed it the
    squared gradient norm from a small batch (e.g. one micro-batch / one rank)
    and from a big batch (the full accumulated / all-reduced gradient).

        tracker = GradientNoiseTracker(beta=0.98)
        tracker.update(g_small_sq, b_small, g_big_sq, b_big)
        stats = tracker.summary()   # {'b_simple', 'grad_norm_sq', 'trace_sigma'}
    """

    def __init__(self, beta: float = 0.98):
        self.beta = beta
        self._g2_ema: Optional[float] = None
        self._tr_ema: Optional[float] = None
        self.n = 0

    def update(self, g_small_sq: float, b_small: int, g_big_sq: float, b_big: int) -> Dict[str, float]:
        est = gradient_noise_scale(g_small_sq, b_small, g_big_sq, b_big)
        if 'grad_norm_sq' not in est:
            return est
        g2, tr = est['grad_norm_sq'], est['trace_sigma']
        if not (math.isfinite(g2) and math.isfinite(tr)):
            return self.summary()
        if self._g2_ema is None:
            self._g2_ema, self._tr_ema = g2, tr
        else:
            self._g2_ema = self.beta * self._g2_ema + (1 - self.beta) * g2
            self._tr_ema = self.beta * self._tr_ema + (1 - self.beta) * tr
        self.n += 1
        return self.summary()

    def summary(self) -> Dict[str, float]:
        if self._g2_ema is None or self._tr_ema is None:
            return {}
        b_simple = self._tr_ema / self._g2_ema if self._g2_ema > 0 else float('nan')
        return {
            'b_simple': round(float(b_simple), 2),
            'grad_norm_sq': float(self._g2_ema),
            'trace_sigma': float(self._tr_ema),
            'n_updates': int(self.n),
        }

    def reset(self):
        self._g2_ema = self._tr_ema = None
        self.n = 0
