"""Venture contracts — bilateral positive-sum agreements."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional


class VentureStatus(Enum):
    PROPOSED = auto()
    ACTIVE = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    DEFAULTED = auto()


@dataclass
class Venture:
    venture_id: int
    proposer: str
    counterparty: str
    asset_idx: int          # underlying asset the venture references
    principal: float        # numeraire committed by each party
    collateral: float       # locked during duration
    surplus_share_proposer: float  # fraction of surplus going to proposer
    start_tick: int
    duration: int
    hidden_quality: float   # world-seeded [0,1]; not directly known to agents
    status: VentureStatus = VentureStatus.PROPOSED

    # Resolved at end
    surplus: float = 0.0
    resolved_tick: Optional[int] = None

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration

    def resolve(self, actual_quality: float, surplus_multiplier: float, tick: int) -> Dict[str, float]:
        """
        Returns payouts dict {agent_id: payout}.
        Venture succeeds if actual quality (revealed state) > 0.5.
        """
        self.resolved_tick = tick
        if actual_quality > 0.5:
            self.status = VentureStatus.SUCCEEDED
            gross = (self.principal * 2) * surplus_multiplier
            self.surplus = gross - self.principal * 2
            proposer_payout = self.principal + self.surplus * self.surplus_share_proposer
            counter_payout = self.principal + self.surplus * (1 - self.surplus_share_proposer)
        else:
            self.status = VentureStatus.FAILED
            self.surplus = 0.0
            # Principal is returned minus a failure penalty absorbed by collateral
            proposer_payout = self.principal - self.collateral * 0.5
            counter_payout = self.principal - self.collateral * 0.5

        return {self.proposer: proposer_payout, self.counterparty: counter_payout}
