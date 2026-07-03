"""
HorizonSim Agency Indices
=========================

Computes the quantities used by the modulated (UHFS) reward equations.

Normalisation convention (the important part)
---------------------------------------------
Every agency dimension is a ratio against the agent's own baseline. The
reward normalises each by the floor θ = AGENCY_FLOOR so that, on the log
scale used by the reward:

    intervention aᵢ = raw_agencyᵢ / θ

      aᵢ  = 1   →  the agent is *at the floor* — the minimum agency required
                   to keep expanding.  log(aᵢ) = 0  (neutral, no harm).
      aᵢ  > 1   →  above the floor, headroom to expand.  log(aᵢ) > 0.
                   NO upper clip — growing agency keeps paying off.
      aᵢ  < 1   →  agency is being harmed.  log(aᵢ) < 0.
      aᵢ  → 0   →  absolute failure / bankruptcy.  log(aᵢ) → −∞.

So the raw agency components below are deliberately left UN-clipped on the
upper side (a solvent, cash-rich agent legitimately scores > 1); only a
tiny ε floor guards the logs.

  I_a   — Agency vector: 5-dimensional raw ratios (each > 0, may exceed 1)
            [0] liquidity   cash / initial_cash
            [1] epistemic   EH score (belief accuracy × uniqueness)
            [2] network     comm-graph degree / (n_agents-1)
            [3] solvency    net_worth / initial_wealth
            [4] options     fraction of strategy-graph cells still viable

  HI (horizon index) — cross-metric HARMONIC mean over three horizons:
            HI = harmonic(H_past, H_present, H_future)
          where each H_* is the cross-metric harmonic mean of the agency
          vector evaluated in the past (agency_history), now, and projected
          forward along the current wealth trajectory.  Harmonic (not
          geometric) so a collapse in ANY horizon drags HI down hard.
          Enters the reward as a multiplicative gate on the expansion term.

  F (headroom / slack) — the structural + resource slack an agent builds up
          BEFORE it can afford expansive exploration:
            F = resource_headroom × structural_headroom
              = (cash above the floor) × (fraction of strategy cells viable)
          No upper clip.  When F → 0 the expansion term vanishes and only the
          safety term log(min aᵢ) remains — i.e. slack has been spent.

  HI_sus — Sustainability signal (credibility / betrayal exposure).  No longer
          a core reward term; retained to modulate venture caution and to feed
          reputation_sensitivity so the deception dynamics still see it.

The safety term (S) lives in reward.py as log(min aᵢ): 0 at the floor,
negative once any intervention drops below 1, unbounded toward −∞ at
bankruptcy (min raw agency → 0).  θ = AGENCY_FLOOR is still the danger-zone
threshold on the RAW min (min raw < θ  ⇔  min aᵢ < 1).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, List
import math
import numpy as np

if TYPE_CHECKING:
    from .agent import Agent
    from .config import SimConfig

AGENCY_FLOOR: float = 0.10    # θ — floor; raw agency = θ maps to intervention aᵢ = 1
_EPS: float = 1e-9             # numerical floor for logs


def _harmonic_mean(x: np.ndarray) -> float:
    """
    Cross-metric harmonic mean, floored at _EPS so a single zero-ish component
    drives the mean toward zero (the property we want: you are only as strong
    as your weakest dimension / horizon) without dividing by zero.
    """
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, _EPS)
    return float(x.size / np.sum(1.0 / x))


@dataclass(frozen=True)
class AgencyState:
    """Snapshot of an agent's agency indices at one tick."""
    # Raw 5-vector
    ia_liquidity: float    # cash / initial_cash
    ia_epistemic: float    # epistemic health score
    ia_network: float      # degree / n_agents
    ia_solvency: float     # net_worth / initial_wealth
    ia_options: float      # fraction of strategy-graph cells still viable (weight ≥ 0)
    ia_liquidity_shared: float  # cash / shared_reference_wealth (variant "A" liquidity)

    # Derived indices
    min_ia: float          # bottleneck raw dimension (min raw < θ ⇔ danger zone)
    hi: float              # Horizon Index — harmonic(past, present, future), no upper clip
    f: float               # Headroom / slack = resource_headroom × structural_headroom
    hi_sus: float          # Sustainability signal (modulation only, not core reward)

    # Context
    in_danger_zone: bool   # min_ia < AGENCY_FLOOR

    @property
    def ia_vector(self) -> np.ndarray:
        return np.array([
            self.ia_liquidity, self.ia_epistemic,
            self.ia_network, self.ia_solvency, self.ia_options,
        ])

    # ── Back-compat aliases (older call sites referenced hi_ia / fhi) ──────
    @property
    def hi_ia(self) -> float:
        return self.hi

    @property
    def fhi(self) -> float:
        return self.f


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
    ia_options   = _compute_options(agent)
    ia_liquidity_shared = _compute_liquidity_shared(agent)

    # ── Derived ───────────────────────────────────────────────────────
    ia_vec = np.array([ia_liquidity, ia_epistemic, ia_network, ia_solvency, ia_options])
    min_ia = float(ia_vec.min())

    hi     = _compute_horizon_index(agent, ia_vec)          # harmonic past/present/future
    f      = _compute_headroom(agent, market_prices)        # resource × structural slack
    hi_sus = _compute_hi_sus(agent)

    return AgencyState(
        ia_liquidity=ia_liquidity,
        ia_epistemic=ia_epistemic,
        ia_network=ia_network,
        ia_solvency=ia_solvency,
        ia_options=ia_options,
        ia_liquidity_shared=ia_liquidity_shared,
        min_ia=min_ia,
        hi=hi,
        f=f,
        hi_sus=hi_sus,
        in_danger_zone=(min_ia < AGENCY_FLOOR),
    )


# ──────────────────────────────────────────────────────────────────────
# I_a component calculators
# ──────────────────────────────────────────────────────────────────────

def _compute_liquidity(agent: "Agent") -> float:
    """cash relative to initial endowment. Floored at ε, NO upper clip
    (holding more cash than you started with is legitimate headroom, aᵢ > 1)."""
    if not hasattr(agent, "_initial_cash") or agent._initial_cash <= 0:
        return max(_EPS, agent.cash / 100.0)
    ratio = agent.cash / agent._initial_cash
    return float(max(ratio, _EPS))


def _compute_liquidity_shared(agent: "Agent") -> float:
    """cash relative to a SHARED reference endowment (mean initial cash across
    agents), so the ratio is comparable across agents and does not reward
    having started with a tiny Pareto endowment.  Falls back to own initial
    cash if no shared reference has been set.  Floored at ε, NO upper clip."""
    ref = getattr(agent, "_ref_wealth", None)
    if not ref or ref <= 0:
        ref = getattr(agent, "_initial_cash", max(agent.cash, 1.0))
    return float(max(agent.cash / max(ref, _EPS), _EPS))


def _compute_epistemic(agent: "Agent") -> float:
    """EH score from belief accuracy. Floored at ε; naturally ≤ 1 (an accuracy)."""
    eh = agent.belief_accuracy()
    return float(max(eh, _EPS))


def _compute_network(agent: "Agent", n_agents: int, max_degree: int) -> float:
    """Fraction of possible peers reachable (degree / (n_agents-1)). Floored at ε."""
    degree = len(getattr(agent, "_current_neighbors", []))
    denom = max(n_agents - 1, 1)
    return float(max(degree / denom, _EPS))


def _compute_solvency(agent: "Agent", market_prices: np.ndarray) -> float:
    """net_worth / initial_wealth. Floored at ε, NO upper clip (a profitable
    agent scores aᵢ > 1 and keeps being rewarded for growing agency)."""
    nw = agent.net_worth(market_prices)
    initial = getattr(agent, "_initial_cash", max(agent.cash, 1.0))
    ratio = nw / max(initial, _EPS)
    return float(max(ratio, _EPS))


_OPTIONS_TEMP: float = 1.0   # softness of the viability curve around q=0
_OPTIONS_UNVISITED_PRIOR: float = 0.5  # an untried cell is "unknown", not "open"


def _compute_options(agent: "Agent") -> float:
    """
    Viable mass of the agent's strategy graph (Q-table): how much of its
    own action space it can still exercise without (subjectively) harming
    itself, weighted by magnitude and only credited where actually tried.

    Two failure modes a naive `fraction of cells >= 0` has, both fixed here:

      1. Sign vs. magnitude: a cell at -0.01 and one at -50 are not equally
         "destroyed", and a cell sitting at exactly 0 only looks "open"
         because it has never been tested. We map each visited cell's
         weight through a sigmoid (smooth, magnitude-sensitive) instead of
         thresholding its sign, and give unvisited cells a neutral 0.5
         prior — "unknown", not "available".

      2. Reward-shaping toward paralysis: if untried cells counted as fully
         open (1.0), the cheapest way to keep this score high would be to
         never explore. Capping unvisited cells at the neutral prior (which
         is *lower* than a confidently-good visited cell, 0.5 < sigmoid(+q))
         means an agent has to actually exercise options to be credited for
         having them — sitting still cannot maximize this term.

    This stays purely a function of the agent's own subjective experience
    (its own rewards), not of hidden ground truth — an agent has no
    privileged access to whether it was deceived, only to what hurt it.
    Whether the resulting compression actually localizes to deception
    (rather than ordinary exploration noise) is a separate, falsifiable
    question — see scripts/verify_options_localizes_to_deception.py.
    """
    q = getattr(agent, "_comm_q", None)
    visits = getattr(agent, "_comm_q_visits", None)
    if q is None or q.size == 0:
        return 1.0
    if visits is None:
        # No visit tracking available: fall back to a magnitude-aware
        # estimate treating every cell as visited (conservative under-prior).
        viability = 1.0 / (1.0 + np.exp(-q / _OPTIONS_TEMP))
        return float(np.clip(viability.mean(), _EPS, 1.0))

    visited = visits > 0
    viability = np.full(q.shape, _OPTIONS_UNVISITED_PRIOR, dtype=float)
    viability[visited] = 1.0 / (1.0 + np.exp(-q[visited] / _OPTIONS_TEMP))
    return float(np.clip(viability.mean(), _EPS, 1.0))


# ──────────────────────────────────────────────────────────────────────
# HI — Horizon Index (cross-metric harmonic mean over past/present/future)
# ──────────────────────────────────────────────────────────────────────

def _compute_horizon_index(agent: "Agent", present_vec: np.ndarray) -> float:
    """
    HI = harmonic( H_past, H_present, H_future )

    where each H_* is the cross-metric harmonic mean of the agency vector at
    that horizon:

      H_present : harmonic mean of the current agency vector.
      H_past    : same, from the most recent stored agency snapshot
                  (agency_history); falls back to H_present early on.
      H_future  : H_present scaled by the current wealth trajectory — if the
                  agent's recent net worth is trending up the future horizon
                  opens, if it is crashing the future collapses and (being a
                  harmonic term) drags HI down hard.

    Harmonic throughout: a weak metric OR a weak horizon dominates, which is
    the intended "you are only as strong as your weakest horizon" behaviour.
    No upper clip — a growing agent's HI exceeds 1 and keeps paying off.
    """
    h_present = _harmonic_mean(present_vec)

    hist = getattr(agent, "agency_history", None)
    if hist:
        h_past = _harmonic_mean(hist[-1].ia_vector)
    else:
        h_past = h_present

    # Future horizon: project present agency along the recent wealth trajectory.
    wh = getattr(agent, "wealth_history", None)
    if wh and len(wh) >= 2:
        recent = max(float(wh[-1]), _EPS)
        base = wh[-5:-1] if len(wh) >= 5 else wh[-2:-1]
        base_mean = max(float(np.mean(base)), _EPS)
        growth = recent / base_mean          # >1 improving, <1 deteriorating
    else:
        growth = 1.0
    h_future = max(h_present * growth, _EPS)

    return _harmonic_mean(np.array([h_past, h_present, h_future]))


# ──────────────────────────────────────────────────────────────────────
# F — Headroom / slack (structural + resource)
# ──────────────────────────────────────────────────────────────────────

def _compute_headroom(agent: "Agent", market_prices: np.ndarray) -> float:
    """
    F = resource_headroom × structural_headroom

      resource_headroom : cash held ABOVE the floor, as a fraction of the
                          initial endowment — (cash/initial_cash − θ).  This is
                          the liquid slack the agent has accumulated and can
                          spend on expansive (speculative) exploration.  At the
                          floor it is 0 → the expansion term switches off and
                          only the safety term remains.  No upper clip.

      structural_headroom : fraction of the strategy graph still viable
                          (_compute_options) — the structural room to act
                          without (subjectively) harming itself.

    Both are genuine slack the agent must build up before it can afford
    expansive exploration; multiplying them means it needs BOTH liquid and
    structural room, and running either down throttles exploration.
    """
    initial = getattr(agent, "_initial_cash", None)
    if not initial or initial <= 0:
        resource_headroom = max(agent.cash / 100.0 - AGENCY_FLOOR, _EPS)
    else:
        resource_headroom = max(agent.cash / initial - AGENCY_FLOOR, _EPS)

    structural_headroom = _compute_options(agent)

    return float(max(resource_headroom * structural_headroom, _EPS))


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

    # Outgoing reliability: were the values this agent actually sent correct?
    # A large _comm_bias causes sent predictions to diverge from truth, so
    # outgoing_accuracy drops — pulling HI_sus down and feeding that cost
    # back into the UHFS reward signal without naming "deception" explicitly.
    outgoing_acc = agent.outgoing_accuracy() if hasattr(agent, "outgoing_accuracy") else 0.5
    outgoing_rel = 0.5 + 0.5 * outgoing_acc  # [0.5, 1.0] — never fully zeroes HI_sus

    hi_sus = stability * cred_health * (1.0 - betrayal) * outgoing_rel
    return float(np.clip(hi_sus, 0.05, 1.0))
