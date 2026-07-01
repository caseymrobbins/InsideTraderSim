"""Evidence Ledger and Epistemic Model primitives."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class EvidenceType(Enum):
    OBSERVATION = auto()   # from prediction card / direct market signal
    COMMUNICATION = auto() # from a message sent by another agent


class EvidenceStatus(Enum):
    PENDING = auto()    # not yet resolved against ground truth
    CONFIRMED = auto()  # turned out to be correct
    REFUTED = auto()    # turned out to be wrong
    EXPIRED = auto()    # time horizon passed without resolution


@dataclass
class Evidence:
    """Immutable record in an agent's Evidence Ledger."""
    ev_id: int
    source: str                      # "self" or agent_id of sender
    ev_type: EvidenceType
    proposition: str                 # e.g. "asset_0_price_tick_42 = 105.3"
    target_asset: Optional[int]      # asset or venture index
    target_tick: int                 # tick the claim resolves at
    predicted_value: float           # the claim's numerical value
    confidence: float                # in [0, 1]
    timestamp: int                   # tick created
    status: EvidenceStatus = EvidenceStatus.PENDING
    actual_value: Optional[float] = None
    error: Optional[float] = None    # |predicted - actual| after resolution
    # For OBSERVATION evidence only: the true fundamental at the time the card
    # was issued.  Resolution compares predicted_value against this rather than
    # the future actual, so EH measures signal quality not price-forecasting
    # skill across drifting fundamentals (which would REFUTE everything).
    true_value_at_issue: Optional[float] = None


@dataclass
class EpistemicModel:
    """
    Per-agent credibility registry.
    credibility[source_id][proposition_type] → float in [0, 1]
    Updated whenever evidence from that source resolves.
    """
    credibility: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get(self, source: str, prop_type: str, default: float = 0.5) -> float:
        return self.credibility.get(source, {}).get(prop_type, default)

    def update(self, source: str, prop_type: str, was_correct: bool, rate: float = 0.1) -> None:
        if source not in self.credibility:
            self.credibility[source] = {}
        current = self.credibility[source].get(prop_type, 0.5)
        target = 1.0 if was_correct else 0.0
        self.credibility[source][prop_type] = current + rate * (target - current)

    def average_credibility(self, source: str) -> float:
        scores = self.credibility.get(source, {})
        if not scores:
            return 0.5
        return sum(scores.values()) / len(scores)
