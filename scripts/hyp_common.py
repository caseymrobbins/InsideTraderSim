"""
Shared helpers for the four alignment-hypothesis experiments
(test_h1_*.py … test_h4_*.py) and the suite runner.

Each hypothesis script builds SimConfigs, runs the sim, and reads the agency
instrumentation that now lives in the metrics history (see
metrics.TickSnapshot.min_ia / agency_dims / frac_danger_zone / mean_min_ia).

Design mirrors scripts/test_deception_counterfactual.py: a small run helper, a
seed loop, a printed table, a JSON dump, and an optional plot.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility guard. The simulation consumes RNG in an order that depends on
# dict/set iteration, so a bare `seed` is NOT sufficient — PYTHONHASHSEED also
# perturbs results across processes (verified: same seed → frac_danger 0.03–0.17
# across hash seeds). We pin it to 0 by re-exec'ing once so every run in the suite
# is reproducible and cross-condition comparisons share one hash order. (Hash order
# remains a second, uncontrolled noise source in the underlying sim — disclosed in
# the report; averaged over multiple `seed`s.)
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys
if _os.environ.get("PYTHONHASHSEED") != "0":
    _os.environ["PYTHONHASHSEED"] = "0"
    _os.execv(_sys.executable, [_sys.executable] + _sys.argv)

import json
import os
from typing import Dict, List, Optional

import numpy as np

from inside_traders.config import SimConfig
from inside_traders.simulation import Simulation
from inside_traders.reward import compute_base_utility

# Agency dimension layout of TickSnapshot.agency_dims (n_agents, 6)
DIM_NAMES = ["liquidity", "epistemic", "network", "solvency", "options", "integrity"]

OUT_DIR = "experiment_results/hypotheses"


# ----------------------------------------------------------------------
# Config + run
# ----------------------------------------------------------------------

def build_cfg(
    model: str,
    seed: int,
    *,
    coupling: str = "none",
    variant: str = "B",
    headroom_mode: str = "stable",
    n_agents: int = 20,
    n_ticks: int = 300,
    log_interval: int = 10,
    **overrides,
) -> SimConfig:
    """One place that maps a design cell to a SimConfig (cf. experiments/design.build_sim_config)."""
    cfg = SimConfig(
        n_agents=n_agents,
        n_ticks=n_ticks,
        seed=seed,
        reward_model=model,
        uhfs_variant=variant,
        agency_coupling=coupling,
        headroom_mode=headroom_mode,
        verbose=False,
        log_interval=log_interval,
        status_interval=n_ticks + 1,   # silence the per-status console table
        plot_interval=n_ticks + 1,     # no mid-run PNGs
        **overrides,
    )
    return cfg


def run_and_measure(cfg: SimConfig) -> Dict:
    """
    Run one simulation and summarise the outcome + agency trajectory.

    Returns a flat dict of scalars/vectors read from the metrics history:
      final_* (last snapshot), *_over_time (mean across snapshots), and the
      per-dimension agency vector (mean over agents) at the final tick.
    """
    sim = Simulation(cfg)
    metrics = sim.run()
    snaps = metrics.snapshots
    last = snaps[-1]

    # Per-dimension agency at the final tick (mean over agents).
    final_dims = (last.agency_dims.mean(axis=0)
                  if last.agency_dims is not None else np.full(6, np.nan))
    # Per-dimension agency averaged over the whole run.
    dims_over_time = np.nanmean(
        [s.agency_dims.mean(axis=0) for s in snaps if s.agency_dims is not None],
        axis=0,
    )

    # OWN-agency floor time: fraction of agent-ticks whose min over the 5 SELF dims
    # (agency_dims[:, :5], excluding the relational integrity axis at index 5) is below
    # the floor θ. This is the instrumental-convergence probe — under an objective with
    # NO self-agency terms (UFR), does own agency stay above its floor on its own?
    theta = getattr(cfg, "agency_floor", 0.10)
    own_danger = [float((s.agency_dims[:, :5].min(axis=1) < theta).mean())
                  for s in snaps if s.agency_dims is not None]
    mean_own_frac_danger = float(np.mean(own_danger)) if own_danger else float("nan")

    return {
        "final_dims": final_dims.tolist(),
        "dims_over_time": np.asarray(dims_over_time).tolist(),
        "final_mean_min_ia": float(last.mean_min_ia),
        "min_mean_min_ia": float(min(s.mean_min_ia for s in snaps)),   # worst point in the run
        "final_frac_danger": float(last.frac_danger_zone),
        "mean_frac_danger": float(np.mean([s.frac_danger_zone for s in snaps])),
        "mean_own_frac_danger": mean_own_frac_danger,   # SELF-agency floor time (instrumental-convergence probe)
        "final_deception": float(last.deception_rate),
        "final_trust": float(last.mean_trust),
        "final_integrity": float(last.mean_integrity),
        "final_gini": float(last.gini),
        "poli_eh_corr": float(metrics.poli_eh_correlation()),
        "total_wealth": float(last.total_wealth),
        "mean_own_utility": _mean_own_utility(sim),
    }


def _mean_own_utility(sim: Simulation) -> float:
    """Mean base utility (net_worth / reference) across agents — objective-agnostic welfare proxy."""
    prices = sim.order_book.prices
    return float(np.mean([compute_base_utility(a, prices) for a in sim.agents]))


# ----------------------------------------------------------------------
# Aggregation across seeds
# ----------------------------------------------------------------------

def aggregate(runs: List[Dict]) -> Dict:
    """Mean/std across seed repetitions of run_and_measure() dicts."""
    out: Dict = {}
    keys = runs[0].keys()
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[k] = {"mean": np.nanmean(vals, axis=0).tolist(),
                  "std": np.nanstd(vals, axis=0).tolist()}
    return out


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------

def ensure_out() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


def save_json(name: str, data) -> str:
    ensure_out()
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[data] wrote {path}")
    return path


def get_plt():
    """Return a matplotlib.pyplot on the Agg backend, or None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:   # pragma: no cover
        print(f"[viz] matplotlib unavailable, skipping plot: {e}")
        return None


def std_seeds(quick: bool) -> List[int]:
    return [1, 2] if quick else [1, 2, 3, 4, 5]


def std_size(quick: bool):
    """Return (n_agents, n_ticks) for the chosen budget."""
    return (12, 80) if quick else (20, 300)
