"""
Statistical analysis of experiment results.

- compute_trial_stats: aggregate TrialResult lists into model-level summary dicts
- compare_models: pairwise Mann-Whitney U + Cohen's d between all model pairs
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np

from .runner import TrialResult

# Metrics to analyse (field names from TrialResult)
_KEY_METRICS = [
    "final_gini",
    "final_mean_eh",
    "final_mean_poli_eh",
    "final_venture_success_rate",
    "final_deception_rate",
    "final_mean_trust",
    "wealth_growth",
    "eh_trajectory_slope",
    "compressed_final_eh",
    "compressed_final_poli",
    "noncomp_final_wealth",
    "mean_reward_signal",
    "n_danger_zone_ticks",
    "mean_gini_over_time",
]


def compute_trial_stats(
    results: List[TrialResult],
    group_by: str = "reward_model",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Group results by `group_by` field, then compute per-metric statistics.

    Returns:
      {group_value: {metric: {"mean": x, "std": x, "median": x, "min": x, "max": x}}}
    """
    grouped: Dict[str, List[TrialResult]] = {}
    for r in results:
        key = getattr(r, group_by)
        grouped.setdefault(key, []).append(r)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group, trials in grouped.items():
        out[group] = {}
        for metric in _KEY_METRICS:
            vals = np.array([getattr(t, metric) for t in trials], dtype=float)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                out[group][metric] = {"mean": float("nan"), "std": float("nan"),
                                      "median": float("nan"), "min": float("nan"),
                                      "max": float("nan"), "n": 0}
            else:
                out[group][metric] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                    "median": float(np.median(finite)),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "n": int(len(finite)),
                }
    return out


def compare_models(
    results: List[TrialResult],
    scenario: str | None = None,
) -> Dict[Tuple[str, str], Dict[str, Dict]]:
    """
    Pairwise comparison between all reward model pairs.

    Returns:
      {(model_a, model_b): {metric: {"u_stat": x, "p_value": x, "cohens_d": x, "direction": str}}}
    """
    if scenario is not None:
        results = [r for r in results if r.scenario == scenario]

    model_data: Dict[str, List[TrialResult]] = {}
    for r in results:
        model_data.setdefault(r.reward_model, []).append(r)

    models = sorted(model_data.keys())
    out: Dict[Tuple[str, str], Dict[str, Dict]] = {}

    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            pair = (ma, mb)
            out[pair] = {}
            for metric in _KEY_METRICS:
                a_vals = np.array([getattr(t, metric) for t in model_data[ma]], dtype=float)
                b_vals = np.array([getattr(t, metric) for t in model_data[mb]], dtype=float)
                a_vals = a_vals[np.isfinite(a_vals)]
                b_vals = b_vals[np.isfinite(b_vals)]

                if len(a_vals) < 2 or len(b_vals) < 2:
                    out[pair][metric] = {"u_stat": float("nan"), "p_value": float("nan"),
                                        "cohens_d": float("nan"), "direction": "?"}
                    continue

                u_stat, p_value = _mann_whitney_u(a_vals, b_vals)
                d = _cohens_d(a_vals, b_vals)
                direction = ">" if float(np.mean(a_vals)) > float(np.mean(b_vals)) else "<"

                out[pair][metric] = {
                    "u_stat": u_stat,
                    "p_value": p_value,
                    "cohens_d": d,
                    "direction": direction,  # ma relative to mb
                    "mean_a": float(np.mean(a_vals)),
                    "mean_b": float(np.mean(b_vals)),
                }
    return out


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _mann_whitney_u(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sided Mann-Whitney U test (pure NumPy, no scipy dependency)."""
    na, nb = len(a), len(b)
    combined = np.concatenate([a, b])
    ranks = _rank(combined)
    r_a = ranks[:na].sum()
    u_a = r_a - na * (na + 1) / 2
    u_b = na * nb - u_a
    u_stat = min(u_a, u_b)

    # Normal approximation (valid for na, nb > ~8)
    mu_u = na * nb / 2.0
    sigma_u = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    if sigma_u < 1e-12:
        return u_stat, 1.0
    z = (u_stat - mu_u) / sigma_u
    # Two-sided p-value using complementary error function
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return float(u_stat), float(p_value)


def _rank(arr: np.ndarray) -> np.ndarray:
    """Average ranks for ties."""
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # Handle ties
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        if j > i + 1:
            avg = float(np.mean(np.arange(i + 1, j + 1)))
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d."""
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    pooled_std = math.sqrt(max(pooled_var, 1e-12))
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def compression_analysis(results: List[TrialResult]) -> Dict[str, Dict]:
    """
    For each reward model, compare compressed-agent EH vs non-compressed wealth.
    Returns a dict keyed by model with compression effect measures.
    """
    model_data: Dict[str, List[TrialResult]] = {}
    for r in results:
        model_data.setdefault(r.reward_model, []).append(r)

    out: Dict[str, Dict] = {}
    for model, trials in model_data.items():
        comp_ehs = np.array([t.compressed_final_eh for t in trials], dtype=float)
        comp_ehs = comp_ehs[np.isfinite(comp_ehs)]

        comp_wealth = np.array([t.compressed_final_wealth for t in trials], dtype=float)
        comp_wealth = comp_wealth[np.isfinite(comp_wealth)]

        noncomp_wealth = np.array([t.noncomp_final_wealth for t in trials], dtype=float)
        noncomp_wealth = noncomp_wealth[np.isfinite(noncomp_wealth)]

        # Wealth disparity between compressed (high-POLI) and non-compressed
        if len(comp_wealth) > 0 and len(noncomp_wealth) > 0:
            wealth_ratio = float(np.mean(comp_wealth) / max(np.mean(noncomp_wealth), 1.0))
        else:
            wealth_ratio = float("nan")

        out[model] = {
            "mean_compressed_eh": float(np.mean(comp_ehs)) if len(comp_ehs) else float("nan"),
            "mean_compressed_wealth": float(np.mean(comp_wealth)) if len(comp_wealth) else float("nan"),
            "mean_noncomp_wealth": float(np.mean(noncomp_wealth)) if len(noncomp_wealth) else float("nan"),
            "compressed_wealth_ratio": wealth_ratio,  # >1 means compressed agents extract more
            "mean_danger_zone_ticks": float(np.mean([t.n_danger_zone_ticks for t in trials])),
        }
    return out
