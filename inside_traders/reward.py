"""
Reward Model Implementations
=============================

Five nested models compared in the experiment framework, all on ONE log
scale.  The building blocks (see horizon.py for the agency normalisation):

  aᵢ = raw_agencyᵢ / θ       intervention: 1 = at the floor, >1 above it
                             (no upper clip), <1 harming agency, →0 bankrupt.

  S  = log(min aᵢ)           SAFETY term.  0 at the floor, <0 once any
                             intervention drops below 1, → −∞ at bankruptcy.
  E  = Σ log(aᵢ) + log(U)    EXPANSION core: the interventions-with-agency
                             plus utility (all as unbounded logs).
  H  = agency.hi             Horizon Index — harmonic(past, present, future).
  F  = agency.f              Headroom / slack (resource × structural).
  λ  = lam                   overall weight on the expansion term.

Model | Reward Formula
------+--------------------------------------------------------------------
U     | R = Utility                              (raw control, no logs)
UF    | R = S + λ·log(U)                         (safety floor + utility)
UH    | R = λ·H·E                                (horizon-gated expansion)
UHF   | R = S + λ·H·E                            (+ safety floor)
UHFS  | R = S + λ·H·F·E                          (+ headroom gate on expansion)
------+--------------------------------------------------------------------

Key properties:
  * Nothing is clipped on the upper side; growing agency keeps paying off.
  * When headroom F → 0 (slack spent) the whole expansion term collapses and
    only S = log(min aᵢ) remains — so the safety term is literally "what is
    left after the headroom has been used", and it is already negative if any
    intervention has dropped below 1.
  * Dropping the safety term (UH vs UHF) leaves the same scale, shifted.

Utility is defined as:
  Utility = net_worth / initial_wealth   (always positive, ≥ ε)

Usage in agent.plan_actions():
  agency = compute_agency_state(...)
  utility = agent.net_worth(prices) / agent._initial_cash
  reward  = get_reward_model(cfg.reward_model, lam=...).compute(utility, agency)
"""
from __future__ import annotations
import math
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from .horizon import AGENCY_FLOOR

if TYPE_CHECKING:
    from .horizon import AgencyState

_EPS = 1e-9
_THETA = AGENCY_FLOOR   # floor: raw agency = θ  ⇔  intervention aᵢ = 1


def _interventions(agency: "AgencyState") -> "np.ndarray":
    """Agency vector normalised by the floor θ so 1.0 sits at the floor."""
    return agency.ia_vector / _THETA


def _safety_term(agency: "AgencyState") -> float:
    """S = log(min aᵢ): 0 at the floor, <0 harming, → −∞ at bankruptcy."""
    return math.log(max(agency.min_ia / _THETA, _EPS))


def _expansion_core(utility: float, agency: "AgencyState", variant: str = "raw") -> float:
    """
    E = Σ log(aᵢ) + log(U): interventions-with-agency plus utility.

    `variant` selects which dims enter the additive sum (H, F and the safety
    floor always use the full raw 5-vector, so bankruptcy protection is intact
    in every variant):

      "raw" — all 5 dims (net worth enters 3×: liquidity, solvency, log U).
      "A"   — wealth-with-floors: liquidity is measured against a SHARED
              reference endowment and solvency is dropped from the sum (it
              duplicates log U); wealth enters via liquidity + log U.
      "B"   — agency-first: only the bounded quality dims (epistemic, network,
              options) enter the sum; wealth enters ONLY through log U, so the
              unbounded cash tail cannot dominate the agency signal.
    """
    u = max(utility, _EPS)
    if variant == "A":
        dims = np.array([agency.ia_liquidity_shared, agency.ia_epistemic,
                         agency.ia_network, agency.ia_options])
    elif variant == "B":
        dims = np.array([agency.ia_epistemic, agency.ia_network, agency.ia_options])
    else:  # "raw"
        dims = agency.ia_vector
    interv = dims / _THETA
    return float(np.sum(np.log(np.maximum(interv, _EPS)))) + math.log(u)


class RewardModelName(Enum):
    U    = "U"
    UF   = "UF"
    UH   = "UH"
    UHF  = "UHF"
    UHFS = "UHFS"


class RewardModel:
    """Base class — subclasses implement compute()."""

    name: RewardModelName

    def compute(self, utility: float, agency: "AgencyState") -> float:
        raise NotImplementedError

    def reputation_sensitivity(self) -> float:
        """
        How strongly this objective values reputation/credibility, above the
        baseline instrumental value (lost future influence) that every objective
        shares.  Scales the delayed reputation payoff fed to the comm policy, so
        the OBJECTIVE reaches the comm Q-table (criterion: deception differs across
        reward models).  Objectives that structurally reward sustainability /
        credibility health (i.e. include an HI_sus term) return a higher value.
        Default 0.0 = reputation matters only instrumentally, via influence.
        """
        return 0.0

    def __repr__(self) -> str:
        return f"RewardModel({self.name.value})"


# ──────────────────────────────────────────────────────────────────────
# Model U — Plain utility
# ──────────────────────────────────────────────────────────────────────

class RewardU(RewardModel):
    """R = Utility  (no agency modulation at all)."""
    name = RewardModelName.U

    def compute(self, utility: float, agency: "AgencyState") -> float:
        return max(utility, _EPS)


# ──────────────────────────────────────────────────────────────────────
# Model UF — Utility + Floor (safety)
# ──────────────────────────────────────────────────────────────────────

class RewardUF(RewardModel):
    """
    R = S + λ·log(U)
    Safety floor added to plain utility — no horizon (H) or headroom (F).
    S = log(min aᵢ) is always on: 0 at the floor, negative below it, → −∞ at
    bankruptcy, so the agent is continuously pushed to keep agency ≥ floor.
    """
    name = RewardModelName.UF

    def __init__(self, lam: float = 1.0) -> None:
        self.lam = lam

    def compute(self, utility: float, agency: "AgencyState") -> float:
        u = max(utility, _EPS)
        return _safety_term(agency) + self.lam * math.log(u)


# ──────────────────────────────────────────────────────────────────────
# Model UH — Utility + Horizon
# ──────────────────────────────────────────────────────────────────────

class RewardUH(RewardModel):
    """
    R = λ·H·E
    Horizon-gated expansion, no safety floor.  Same scale as UHF, shifted by
    the (absent) safety term.
    """
    name = RewardModelName.UH

    def __init__(self, lam: float = 1.0, variant: str = "raw") -> None:
        self.lam = lam
        self.variant = variant

    def compute(self, utility: float, agency: "AgencyState") -> float:
        return self.lam * agency.hi * _expansion_core(utility, agency, self.variant)


# ──────────────────────────────────────────────────────────────────────
# Model UHF — Utility + Horizon + Floor
# ──────────────────────────────────────────────────────────────────────

class RewardUHF(RewardModel):
    """
    R = S + λ·H·E
    Horizon-gated expansion plus the safety floor, but no headroom gate.
    """
    name = RewardModelName.UHF

    def __init__(self, lam: float = 1.0, variant: str = "raw") -> None:
        self.lam = lam
        self.variant = variant

    def compute(self, utility: float, agency: "AgencyState") -> float:
        return _safety_term(agency) + self.lam * agency.hi * _expansion_core(utility, agency, self.variant)


# ──────────────────────────────────────────────────────────────────────
# Model UHFS — Full modulated model (the thesis model)
# ──────────────────────────────────────────────────────────────────────

class RewardUHFS(RewardModel):
    """
    R = S + λ·H·F·E
      = log(min aᵢ) + λ · HI · F · ( Σ log(aᵢ) + log(U) )

    The horizon index H and the headroom F both multiplicatively gate the
    expansion term E.  As headroom F → 0 (slack spent), λ·H·F·E → 0 and the
    reward reduces to the safety term S = log(min aᵢ) — which is already
    negative if any intervention has dropped below 1, and unbounded toward
    −∞ as the agent approaches bankruptcy.  Nothing is clipped above, so an
    agent that grows agency past the floor (aᵢ, H, F all > 1) is rewarded for
    it without bound.

    β (beta) is retained only to weight reputation_sensitivity (the
    sustainability/credibility linkage that feeds the deception dynamics);
    it no longer appears in the reward itself.
    """
    name = RewardModelName.UHFS

    def __init__(self, lam: float = 1.0, beta: float = 1.0, variant: str = "raw") -> None:
        self.lam = lam
        self.beta = beta
        self.variant = variant

    def compute(self, utility: float, agency: "AgencyState") -> float:
        return (
            _safety_term(agency)
            + self.lam * agency.hi * agency.f * _expansion_core(utility, agency, self.variant)
        )

    def reputation_sensitivity(self) -> float:
        # UHFS is the sustainability-weighted objective, so it values reputation
        # beyond instrumental influence.  Scaled by β so the linkage is tunable.
        return self.beta


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

REWARD_MODELS: dict[RewardModelName, RewardModel] = {
    RewardModelName.U:    RewardU(),
    RewardModelName.UF:   RewardUF(),
    RewardModelName.UH:   RewardUH(),
    RewardModelName.UHF:  RewardUHF(),
    RewardModelName.UHFS: RewardUHFS(),  # default λ=β=1.0; use get_reward_model() for custom weights
}


def get_reward_model(
    name: "str | RewardModelName",
    lam: float = 1.0,
    beta: float = 1.0,
    variant: str = "raw",
    *,
    alpha: float | None = None,   # deprecated: former log(FHI) weight, ignored
) -> RewardModel:
    """
    Retrieve a reward model by name (string or enum).

    lam     — λ, the weight on the expansion term (UF / UH / UHF / UHFS).
    beta    — retained for UHFS.reputation_sensitivity() (sustainability linkage).
    variant — expansion-sum variant for the H-family ("raw" / "A" / "B").
    alpha is accepted for backward compatibility only and has no effect.
    """
    if isinstance(name, str):
        name = RewardModelName(name)
    if name == RewardModelName.U:
        return REWARD_MODELS[name]
    if name == RewardModelName.UHFS:
        return RewardUHFS(lam=lam, beta=beta, variant=variant)
    if name == RewardModelName.UF:
        return RewardUF(lam=lam)
    if name == RewardModelName.UH:
        return RewardUH(lam=lam, variant=variant)
    if name == RewardModelName.UHF:
        return RewardUHF(lam=lam, variant=variant)
    return REWARD_MODELS[name]


# ──────────────────────────────────────────────────────────────────────
# Utility normaliser — used by all models
# ──────────────────────────────────────────────────────────────────────

def compute_base_utility(agent, market_prices) -> float:
    """
    Utility = net_worth / initial_wealth.
    Always positive by construction (initial_wealth > 0, floor at ε).
    """
    nw = agent.net_worth(market_prices)
    initial = getattr(agent, "_initial_cash", max(agent.cash, 1.0))
    return max(nw / max(initial, _EPS), _EPS)
