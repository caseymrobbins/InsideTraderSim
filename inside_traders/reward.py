"""
Reward Model Implementations
=============================

Five models compared in the experiment framework.

Model | Description                         | Reward Formula
------+-------------------------------------+---------------------------------------------
U     | Plain base utility only             | R = Utility
UF    | Base utility + Floor protection     | Danger zone: R = log(min(I_a))
UH    | Base utility + HI term             | Safe: R = log(HI_Ia) + log(Utility)
UHF   | Base utility + Floor + HI          | Floor + HI (no FHI / HI_sus scaling)
UHFS  | Full modulated model                | log(HI_Ia) + log(U+ε) + α·log(FHI) + β·log(HI_sus)
------+-------------------------------------+---------------------------------------------

The core equations (used verbatim from the spec):

  Danger Zone  (min(I_a) < θ):
      R = log(min(I_a))

  Safe Zone (min(I_a) ≥ θ):
      R = log(HI_Ia) + log(U+ε) + α·log(FHI) + β·log(HI_sus)

Utility is defined as:
  Utility = net_worth / initial_wealth   (always positive, ≥ ε)

For models U and UH that have no floor, the danger zone is never
triggered — the agent always uses the safe-zone formula (or just raw
utility for U).

Usage in agent.plan_actions():
  agency = compute_agency_state(...)
  utility = agent.net_worth(prices) / agent._initial_cash
  reward  = REWARD_MODELS[cfg.reward_model].compute(utility, agency)
"""
from __future__ import annotations
import math
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .horizon import AgencyState

_EPS = 1e-9


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
# Model UF — Utility + Floor
# ──────────────────────────────────────────────────────────────────────

class RewardUF(RewardModel):
    """
    Danger zone  → R = log(min(I_a))   [pushes agent to restore agency]
    Safe zone    → R = Utility          [plain utility, no HI modulation]
    """
    name = RewardModelName.UF

    def compute(self, utility: float, agency: "AgencyState") -> float:
        if agency.in_danger_zone:
            return math.log(agency.min_ia + _EPS)
        return max(utility, _EPS)


# ──────────────────────────────────────────────────────────────────────
# Model UH — Utility + HI
# ──────────────────────────────────────────────────────────────────────

class RewardUH(RewardModel):
    """
    No danger-zone floor.
    Safe zone → R = log(HI_Ia) + log(Utility)
    Adds the epistemic/agency quality term but without sustainability scaling.
    """
    name = RewardModelName.UH

    def compute(self, utility: float, agency: "AgencyState") -> float:
        u = max(utility, _EPS)
        return math.log(agency.hi_ia + _EPS) + math.log(u)


# ──────────────────────────────────────────────────────────────────────
# Model UHF — Utility + HI + Floor
# ──────────────────────────────────────────────────────────────────────

class RewardUHF(RewardModel):
    """
    Danger zone  → R = log(min(I_a))
    Safe zone    → R = log(HI_Ia) + log(Utility)
    Floor + HI term, but no FHI or HI_sus scaling.
    """
    name = RewardModelName.UHF

    def compute(self, utility: float, agency: "AgencyState") -> float:
        if agency.in_danger_zone:
            return math.log(agency.min_ia + _EPS)
        u = max(utility, _EPS)
        return math.log(agency.hi_ia + _EPS) + math.log(u)


# ──────────────────────────────────────────────────────────────────────
# Model UHFS — Full modulated model (the thesis model)
# ──────────────────────────────────────────────────────────────────────

class RewardUHFS(RewardModel):
    """
    Danger zone  → R = log(min(I_a))
    Safe zone    → R = log(HI_Ia) + log(U+ε) + α·log(FHI) + β·log(HI_sus)

    Additive log formulation: each term contributes independently so no single
    factor can collapse the reward to zero by going small.
    α (fhi_weight) scales the forward-horizon contribution; β (hi_sus_weight)
    scales the sustainability contribution.  Defaults are 1.0 for both.
    """
    name = RewardModelName.UHFS

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha  # weight on log(FHI)
        self.beta = beta    # weight on log(HI_sus)

    def compute(self, utility: float, agency: "AgencyState") -> float:
        if agency.in_danger_zone:
            return math.log(agency.min_ia + _EPS)
        u = max(utility, _EPS)
        return (
            math.log(agency.hi_ia + _EPS)
            + math.log(u + _EPS)
            + self.alpha * math.log(agency.fhi + _EPS)
            + self.beta * math.log(agency.hi_sus + _EPS)
        )

    def reputation_sensitivity(self) -> float:
        # UHFS is the only model with an explicit HI_sus (sustainability/
        # credibility) term, so it values reputation beyond instrumental influence.
        # Scaled by β (the HI_sus weight) so the linkage tracks the objective.
        return self.beta


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

REWARD_MODELS: dict[RewardModelName, RewardModel] = {
    RewardModelName.U:    RewardU(),
    RewardModelName.UF:   RewardUF(),
    RewardModelName.UH:   RewardUH(),
    RewardModelName.UHF:  RewardUHF(),
    RewardModelName.UHFS: RewardUHFS(),  # default α=β=1.0; use get_reward_model() for custom weights
}


def get_reward_model(
    name: "str | RewardModelName",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> RewardModel:
    """
    Retrieve a reward model by name (string or enum).
    alpha and beta are only used for UHFS (weights on log(FHI) and log(HI_sus)).
    """
    if isinstance(name, str):
        name = RewardModelName(name)
    if name == RewardModelName.UHFS:
        return RewardUHFS(alpha=alpha, beta=beta)
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
