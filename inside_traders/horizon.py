"""
HorizonSim Agency Indices
=========================

Computes the four quantities used by the modulated reward equations:

  I_a   — Agency vector: 4-dimensional, each ∈ (0, 1]
            [0] liquidity   cash / initial_cash
            [1] epistemic   EH score (belief accuracy × uniqueness)
            [2] network     comm-graph degree / n_agents
            [3] solvency    net_worth / initial_wealth

  HI_Ia — Horizon Index of agency: geometric mean of I_a
            HI_Ia = (∏ I_a_i)^(1/n) = exp(mean(log(I_a)))

  FHI   — Future Horizon Index: forward-looking option value
            FHI = belief_coverage × mean_confidence × (1 + exclusivity)
            Clipped to [0.10, 2.00]

  HI_sus — Sustainability Horizon Index: trajectory stability
            HI_sus = wealth_stability × credibility_health × (1 - betrayal_exposure)
            Clipped to [0.05, 1.00]

The "danger zone" threshold is AGENCY_FLOOR = 0.10.
When min(I_a) < AGENCY_FLOOR, the agent is in the danger zone and
the reward function degrades to log(min(I_a)) — heavily negative.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, List
import math
import numpy as np

if TYPE_CHECKING:
    from .agent import Agent
    from .config import SimConfig

AGENCY_FLOOR: float = 0.10    # θ — safe-zone threshold on min(I_a)
_EPS: float = 1e-9             # numerical floor for logs


@dataclass(frozen=True)
class AgencyState:
    """Snapshot of an agent's agency indices at one tick."""
    # Raw 4-vector
    ia_liquidity: float    # cash / initial_cash
    ia_epistemic: float    # epistemic health score
    ia_network: float      # degree / n_agents
    ia_solvency: float     # net_worth / initial_wealth

    # Derived indices
    min_ia: float          # bottleneck dimension (triggers danger zone)
    hi_ia: float           # geometric mean of I_a ∈ (0, 1]
    fhi: float             # Future Horizon Index ∈ [0.10, 2.00]
    hi_sus: float          # Sustainability Index ∈ [0.05, 1.00]

    # Context
    in_danger_zone: bool   # min_ia < AGENCY_FLOOR

    @property
    def ia_vector(self) -> np.ndarray:
        return np.array([
            self.ia_liquidity, self.ia_epistemic,
            self.ia_network, self.ia_solvency,
        ])


def compute_agency_state(
    agent: "Agent",
    market_prices: np.ndarray,
    n_agents: int,
    max_degree: int,
    tick: int,
    cfg: "SimConfig",
) -> AgencyState:
    """
    Compute the full AgencyState for an agent at the current tick.
    Called once per tick inside plan_actions.
    """
    # ── I_a components ────────────────────────────────────────────────
    ia_liquidity = _compute_liquidity(agent)
    ia_epistemic = _compute_epistemic(agent)
    ia_network   = _compute_network(agent, n_agents, max_degree)
    ia_solvency  = _compute_solvency(agent, market_prices)

    # ── Derived ───────────────────────────────────────────────────────
    ia_vec = np.array([ia_liquidity, ia_epistemic, ia_network, ia_solvency])
    min_ia = float(ia_vec.min())
    hi_ia  = float(np.exp(np.mean(np.log(ia_vec + _EPS))))  # geometric mean

    fhi    = _compute_fhi(agent, tick, cfg)
    hi_sus = _compute_hi_sus(agent)

    return AgencyState(
        ia_liquidity=ia_liquidity,
        ia_epistemic=ia_epistemic,
        ia_network=ia_network,
        ia_solvency=ia_solvency,
        min_ia=min_ia,
        hi_ia=hi_ia,
        fhi=fhi,
        hi_sus=hi_sus,
        in_danger_zone=(min_ia < AGENCY_FLOOR),
    )


# ──────────────────────────────────────────────────────────────────────
# I_a component calculators
# ──────────────────────────────────────────────────────────────────────

def _compute_liquidity(agent: "Agent") -> float:
    """cash relative to initial endowment, clipped to (0, 1]."""
    if not hasattr(agent, "_initial_cash") or agent._initial_cash <= 0:
        return max(_EPS, min(1.0, agent.cash / 100.0))
    ratio = agent.cash / agent._initial_cash
    return float(np.clip(ratio, _EPS, 1.0))


def _compute_epistemic(agent: "Agent") -> float:
    """EH score directly from belief accuracy (clamped to (_EPS, 1])."""
    eh = agent.belief_accuracy()
    return float(np.clip(eh, _EPS, 1.0))


def _compute_network(agent: "Agent", n_agents: int, max_degree: int) -> float:
    """Fraction of possible peers reachable (degree / n_agents)."""
    degree = len(getattr(agent, "_current_neighbors", []))
    denom = max(n_agents - 1, 1)
    return float(np.clip(degree / denom, _EPS, 1.0))


def _compute_solvency(agent: "Agent", market_prices: np.ndarray) -> float:
    """net_worth / initial_wealth, clipped to (_EPS, 1]."""
    nw = agent.net_worth(market_prices)
    initial = getattr(agent, "_initial_cash", max(agent.cash, 1.0))
    ratio = nw / max(initial, _EPS)
    return float(np.clip(ratio, _EPS, 1.0))


# ──────────────────────────────────────────────────────────────────────
# FHI — Future Horizon Index
# ──────────────────────────────────────────────────────────────────────

def _compute_fhi(agent: "Agent", tick: int, cfg: "SimConfig") -> float:
    """
    FHI = belief_coverage × mean_confidence × (1 + info_exclusivity)

    belief_coverage : fraction of next K ticks the agent has actionable beliefs for
    mean_confidence : average confidence across all pending price beliefs
    info_exclusivity: belief_accuracy() as a proxy for unique correct info
    """
    K = max(cfg.eh_accuracy_window, 5)
    future_range = set(range(tick + 1, tick + K + 1))

    covered_ticks: set = set()
    confidences: list = []

    for prop in agent.belief_graph.all_propositions():
        if "_tick_" not in prop.key:
            continue
        try:
            t = int(prop.key.split("_tick_")[1])
        except (IndexError, ValueError):
            continue
        if t in future_range and prop.confidence > 0.15:
            covered_ticks.add(t)
            confidences.append(prop.confidence)

    coverage = len(covered_ticks) / K if K > 0 else 0.0
    mean_conf = float(np.mean(confidences)) if confidences else 0.1
    exclusivity = agent.belief_accuracy()

    fhi = coverage * mean_conf * (1.0 + exclusivity)
    return float(np.clip(fhi, 0.10, 2.0))


# ──────────────────────────────────────────────────────────────────────
# HI_sus — Horizon Index of Sustainability
# ──────────────────────────────────────────────────────────────────────

def _compute_hi_sus(agent: "Agent") -> float:
    """
    HI_sus = wealth_stability × credibility_health × (1 - betrayal_exposure)

    wealth_stability  : exp(-CV) of recent wealth history
    credibility_health: average epistemic credibility the agent assigns to others
    betrayal_exposure : fraction of COMMUNICATION evidence that was REFUTED
    """
    # Wealth stability
    hist = agent.wealth_history[-20:] if len(agent.wealth_history) >= 2 else [agent.cash]
    if len(hist) >= 2:
        mu = max(float(np.mean(hist)), _EPS)
        cv = float(np.std(hist)) / mu
        stability = math.exp(-min(cv, 5.0))
    else:
        stability = 0.5

    # Credibility health: average score assigned to all other agents
    all_creds = []
    for src, scores in agent.epistemic_model.credibility.items():
        if src not in ("self", "oracle_fake"):
            all_creds.extend(scores.values())
    cred_health = float(np.mean(all_creds)) if all_creds else 0.5

    # Betrayal exposure: how often WAS this agent deceived
    from .evidence import EvidenceType, EvidenceStatus
    comm_ev = [
        ev for ev in agent.evidence_ledger
        if ev.ev_type == EvidenceType.COMMUNICATION
        and ev.status in (EvidenceStatus.CONFIRMED, EvidenceStatus.REFUTED)
    ]
    if comm_ev:
        refuted = sum(1 for ev in comm_ev if ev.status == EvidenceStatus.REFUTED)
        betrayal = refuted / len(comm_ev)
    else:
        betrayal = 0.0

    hi_sus = stability * cred_health * (1.0 - betrayal)
    return float(np.clip(hi_sus, 0.05, 1.0))
