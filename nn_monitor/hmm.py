"""
Diagnostics for Hidden Markov Models (hmmlearn-compatible).

Focus on the failure modes unique to HMMs:
- transition/emission malformation
- state occupancy collapse (one state eats everything)
- Baum-Welch log-likelihood non-monotonicity (numerical bugs)
- Viterbi path instability across seeds (label switching)
- emission entropy per state (indistinguishable states)
- numerical underflow in forward-backward
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Transition matrix
# ─────────────────────────────────────────────

def check_transition_matrix(transmat: np.ndarray, absorbing_tol: float = 0.99) -> Dict[str, Any]:
    """Validate that transmat is row-stochastic and not absorbing.

    Returns ok=False if any row does not sum to ~1 or diagonal dominates.
    """
    A = np.asarray(transmat, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return {'ok': False, 'msg': f"transmat must be square 2D, got {A.shape}"}

    row_sums = A.sum(axis=1)
    row_ok = np.allclose(row_sums, 1.0, atol=1e-4)
    neg = bool(np.any(A < -1e-9))
    diag = np.diag(A)
    absorbing_states = [i for i, d in enumerate(diag) if d >= absorbing_tol]
    sparsity = float((A < 1e-6).mean())

    # Spectral gap: 1 - |lambda_2|. Small gap → slow mixing.
    try:
        eigs = np.abs(np.linalg.eigvals(A))
        eigs = np.sort(eigs)[::-1]
        spectral_gap = float(1.0 - eigs[1]) if len(eigs) > 1 else 1.0
    except np.linalg.LinAlgError:
        spectral_gap = float('nan')

    ok = row_ok and not neg and not absorbing_states
    result = {
        'ok': bool(ok),
        'row_sums_ok': bool(row_ok),
        'has_negative': neg,
        'absorbing_states': absorbing_states,
        'sparsity_pct': round(sparsity * 100, 2),
        'spectral_gap': spectral_gap,
        'max_row_sum_dev': float(np.max(np.abs(row_sums - 1.0))),
        'n_states': int(A.shape[0]),
    }
    if not ok:
        logger.warning(f"Transition matrix issues: {result}")
    return result


# ─────────────────────────────────────────────
#  State occupancy
# ─────────────────────────────────────────────

def state_occupancy(state_seq: np.ndarray, n_states: Optional[int] = None) -> Dict[str, Any]:
    """Fraction of time spent in each state + concentration index.

    dominant_state_share > 0.8 → degenerate HMM (one state eats everything).
    """
    s = np.asarray(state_seq).flatten().astype(int)
    if s.size == 0:
        return {'ok': False, 'msg': 'empty state sequence'}
    if n_states is None:
        n_states = int(s.max()) + 1

    counts = np.bincount(s, minlength=n_states).astype(float)
    shares = counts / counts.sum()
    dominant = float(shares.max())

    # Shannon entropy of state distribution — normalized
    p = np.maximum(shares, 1e-12)
    ent = float(-(p * np.log(p)).sum())
    max_ent = float(np.log(n_states)) if n_states > 1 else 1.0
    norm_ent = ent / max_ent if max_ent > 0 else 0.0

    # Unused states
    unused = [int(i) for i, c in enumerate(counts) if c == 0]

    return {
        'ok': bool(dominant < 0.8 and not unused),
        'shares': [round(float(x), 4) for x in shares],
        'dominant_state': int(np.argmax(shares)),
        'dominant_share': round(dominant, 4),
        'unused_states': unused,
        'entropy_normalized': round(norm_ent, 4),
        'n_states': int(n_states),
    }


# ─────────────────────────────────────────────
#  Dwell times
# ─────────────────────────────────────────────

def dwell_times(state_seq: np.ndarray) -> Dict[str, Any]:
    """Mean dwell time per state — for comparison with geometric expectation 1/(1-A_ii)."""
    s = np.asarray(state_seq).flatten().astype(int)
    if s.size < 2:
        return {}
    n_states = int(s.max()) + 1
    runs: Dict[int, List[int]] = {i: [] for i in range(n_states)}
    cur_state = int(s[0])
    cur_len = 1
    for x in s[1:]:
        x = int(x)
        if x == cur_state:
            cur_len += 1
        else:
            runs[cur_state].append(cur_len)
            cur_state = x
            cur_len = 1
    runs[cur_state].append(cur_len)

    out = {}
    for st, lengths in runs.items():
        if lengths:
            out[f'state_{st}'] = {
                'mean_dwell': round(float(np.mean(lengths)), 3),
                'median_dwell': int(np.median(lengths)),
                'max_dwell': int(np.max(lengths)),
                'n_visits': int(len(lengths)),
            }
    return out


# ─────────────────────────────────────────────
#  Emission entropy / separability
# ─────────────────────────────────────────────

def emission_entropy(emission_matrix: np.ndarray) -> Dict[str, Any]:
    """Per-state entropy of categorical emissions.

    High entropy per state = state does not commit to specific observations
    (indistinguishable states, training probably failed).
    """
    E = np.asarray(emission_matrix, dtype=float)
    if E.ndim != 2:
        return {'ok': False, 'msg': "emission_matrix must be 2D (n_states, n_symbols)"}

    E = np.maximum(E, 1e-12)
    E = E / E.sum(axis=1, keepdims=True)
    ent = -(E * np.log(E)).sum(axis=1)
    max_ent = float(np.log(E.shape[1])) if E.shape[1] > 1 else 1.0
    norm = ent / max_ent

    # Pairwise total-variation distance between states
    n = E.shape[0]
    tv = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            tv[i, j] = tv[j, i] = 0.5 * float(np.abs(E[i] - E[j]).sum())
    # Minimum separation: if < 0.1 — indistinguishable states
    upper = tv[np.triu_indices(n, k=1)] if n > 1 else np.array([0.0])
    min_tv = float(upper.min()) if upper.size > 0 else 1.0

    return {
        'ok': bool(min_tv >= 0.1 and float(norm.mean()) < 0.9),
        'per_state_entropy_normalized': [round(float(x), 4) for x in norm],
        'mean_normalized_entropy': round(float(norm.mean()), 4),
        'min_pairwise_tv_distance': round(min_tv, 4),
        'n_states': int(n),
    }


def gaussian_emission_separability(
    means: np.ndarray,
    covars: np.ndarray,
) -> Dict[str, Any]:
    """For GaussianHMM: Mahalanobis distance between state centroids.

    Identifies states whose emission distributions overlap too much to be
    distinguishable. covars can be full (S, D, D) or diag (S, D).
    """
    means = np.asarray(means, dtype=float)
    covars = np.asarray(covars, dtype=float)
    n = means.shape[0]
    if n < 2:
        return {'ok': True, 'msg': 'single state'}

    # Symmetrize cov to full
    if covars.ndim == 2:
        cov_full = np.stack([np.diag(c) for c in covars], axis=0)
    else:
        cov_full = covars

    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            avg_cov = 0.5 * (cov_full[i] + cov_full[j])
            try:
                inv = np.linalg.pinv(avg_cov)
                diff = means[i] - means[j]
                d = float(np.sqrt(max(diff @ inv @ diff, 0.0)))
            except Exception:
                d = float('nan')
            dists[i, j] = dists[j, i] = d

    upper = dists[np.triu_indices(n, k=1)]
    min_d = float(upper.min()) if upper.size > 0 else float('inf')
    return {
        'ok': bool(min_d > 1.0),  # <1σ separation = overlapping states
        'min_mahalanobis': round(min_d, 4),
        'mean_mahalanobis': round(float(upper.mean()), 4),
        'n_states': int(n),
    }


# ─────────────────────────────────────────────
#  Log-likelihood convergence
# ─────────────────────────────────────────────

def check_ll_convergence(ll_history: Sequence[float], tol: float = 1e-3) -> Dict[str, Any]:
    """Baum-Welch log-likelihood must be monotonically non-decreasing.

    A drop = numerical bug or over-regularized M-step.
    """
    ll = np.asarray(list(ll_history), dtype=float)
    if len(ll) < 2:
        return {'ok': True, 'msg': 'insufficient history'}

    diffs = np.diff(ll)
    drops = [
        {'iter': int(i + 1), 'delta': float(d)}
        for i, d in enumerate(diffs) if d < -tol
    ]
    last_delta = float(diffs[-1])
    converged = abs(last_delta) < tol

    return {
        'ok': bool(not drops),
        'n_iters': int(len(ll)),
        'final_ll': float(ll[-1]),
        'total_improvement': float(ll[-1] - ll[0]),
        'll_drops': drops,
        'last_delta': last_delta,
        'converged': bool(converged),
    }


# ─────────────────────────────────────────────
#  Viterbi stability across seeds
# ─────────────────────────────────────────────

def viterbi_stability(paths: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Pairwise permutation-invariant agreement between Viterbi paths from
    different seeds/restarts. Uses Hungarian matching on confusion matrix.

    agreement near 1.0 → HMM converged to same solution modulo label-switching.
    """
    paths = [np.asarray(p).flatten() for p in paths]
    if len(paths) < 2:
        return {'ok': True, 'msg': 'need ≥2 paths'}

    L = min(len(p) for p in paths)
    paths = [p[:L] for p in paths]
    n_states = int(max(int(p.max()) for p in paths) + 1)

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        has_scipy = True
    except ImportError:
        has_scipy = False

    agreements = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            cm = np.zeros((n_states, n_states), dtype=int)
            for a, b in zip(paths[i], paths[j]):
                cm[int(a), int(b)] += 1
            if has_scipy:
                row_ind, col_ind = linear_sum_assignment(-cm)
                matched = int(cm[row_ind, col_ind].sum())
            else:
                # Greedy: pick max each row (upper bound, ok as heuristic)
                matched = int(cm.max(axis=1).sum())
            agreements.append(matched / L)

    return {
        'ok': bool(float(np.mean(agreements)) > 0.9),
        'mean_agreement': round(float(np.mean(agreements)), 4),
        'min_agreement': round(float(np.min(agreements)), 4),
        'n_pairs': int(len(agreements)),
        'hungarian': has_scipy,
    }


# ─────────────────────────────────────────────
#  Forward-backward numerical health
# ─────────────────────────────────────────────

def check_forward_backward_stability(log_likelihoods_per_sample: np.ndarray) -> Dict[str, Any]:
    """Per-sample log-likelihoods from score_samples — detect -inf/nan
    (underflow in forward pass without log-space)."""
    ll = np.asarray(log_likelihoods_per_sample, dtype=float).flatten()
    n_ninf = int(np.sum(np.isneginf(ll)))
    n_nan = int(np.sum(np.isnan(ll)))
    n_inf = int(np.sum(np.isposinf(ll)))
    finite = ll[np.isfinite(ll)]
    return {
        'ok': bool(n_ninf == 0 and n_nan == 0 and n_inf == 0),
        'n_neg_inf': n_ninf,
        'n_nan': n_nan,
        'n_pos_inf': n_inf,
        'finite_min': float(finite.min()) if finite.size else float('nan'),
        'finite_mean': float(finite.mean()) if finite.size else float('nan'),
        'n_total': int(ll.size),
    }


# ─────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────

def run_hmm_diagnostics(
    model,
    X: np.ndarray,
    output_dir: str,
    lengths: Optional[np.ndarray] = None,
    ll_history: Optional[Sequence[float]] = None,
    extra_paths: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Full diagnostic pass for a fitted hmmlearn model.

    Args:
        model: fitted hmmlearn HMM (has transmat_, and optionally emissionprob_ / means_ / covars_)
        X: observations (N, D) or (N,)
        lengths: for concatenated sequences (hmmlearn convention)
        ll_history: sequence of per-iteration log-likelihoods captured during fit
        extra_paths: Viterbi paths from re-fits with different seeds
    """
    diag_dir = Path(output_dir) / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)

    diag: Dict[str, Any] = {}

    transmat = np.asarray(getattr(model, 'transmat_', np.zeros((0, 0))))
    diag['transition'] = check_transition_matrix(transmat) if transmat.size else {'ok': False, 'msg': 'no transmat_'}

    # Viterbi path on X
    try:
        _, path = model.decode(X, lengths=lengths, algorithm='viterbi')
        diag['occupancy'] = state_occupancy(path, n_states=transmat.shape[0] if transmat.size else None)
        diag['dwell_times'] = dwell_times(path)
    except Exception as e:
        diag['occupancy'] = {'ok': False, 'msg': f"decode failed: {e}"}

    # Emissions
    if hasattr(model, 'emissionprob_'):
        diag['emissions'] = emission_entropy(np.asarray(model.emissionprob_))
    elif hasattr(model, 'means_') and hasattr(model, 'covars_'):
        diag['emissions'] = gaussian_emission_separability(
            np.asarray(model.means_), np.asarray(model.covars_)
        )

    # Forward-backward health via per-sample score
    try:
        from hmmlearn.base import _BaseHMM  # type: ignore  # noqa: F401
        try:
            # score_samples returns (logprob, posteriors) in hmmlearn ≥0.3
            logprob = model.score(X, lengths=lengths)
            diag['fb_total_logprob'] = float(logprob)
        except Exception as e:
            diag['fb_error'] = str(e)
    except ImportError:
        pass

    # LL convergence
    if ll_history is not None:
        diag['ll_convergence'] = check_ll_convergence(ll_history)

    # Viterbi stability
    if extra_paths:
        try:
            base_path = diag.get('_path') or model.decode(X, lengths=lengths, algorithm='viterbi')[1]
            paths_all = [base_path] + list(extra_paths)
            diag['viterbi_stability'] = viterbi_stability(paths_all)
        except Exception as e:
            diag['viterbi_stability'] = {'ok': False, 'msg': str(e)}

    with open(diag_dir / 'hmm_diagnostics.json', 'w') as f:
        json.dump(diag, f, indent=2, default=str)

    logger.info(f"HMM diagnostics written to {diag_dir / 'hmm_diagnostics.json'}")
    return diag
