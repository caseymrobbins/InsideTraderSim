"""Belief Graph: proposition nodes with strength and confidence."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Proposition:
    """A node in an agent's Belief Graph."""
    key: str                  # canonical identifier, e.g. "asset_2_price_tick_50"
    value: float              # expected numerical value
    strength: float           # in [-1, 1]; sign = direction, magnitude = certainty
    confidence: float         # in [0, 1]; epistemic confidence
    last_updated: int         # tick of last update
    support_keys: List[str] = field(default_factory=list)  # keys of supporting propositions


class BeliefGraph:
    """
    Directed graph of propositions.  Edges encode support relationships.
    Beliefs update ONLY via the resolve() method driven by Evidence.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Proposition] = {}

    def add_or_update(
        self,
        key: str,
        value: float,
        strength: float,
        confidence: float,
        tick: int,
        support_keys: Optional[List[str]] = None,
    ) -> None:
        strength = max(-1.0, min(1.0, strength))
        confidence = max(0.0, min(1.0, confidence))
        existing = self._nodes.get(key)
        if existing is None:
            self._nodes[key] = Proposition(
                key=key,
                value=value,
                strength=strength,
                confidence=confidence,
                last_updated=tick,
                support_keys=support_keys or [],
            )
        else:
            # Bayesian-ish blend: weight by confidence
            w_new = confidence
            w_old = existing.confidence
            total = w_new + w_old if (w_new + w_old) > 0 else 1.0
            existing.value = (w_old * existing.value + w_new * value) / total
            existing.strength = (w_old * existing.strength + w_new * strength) / total
            existing.confidence = min(1.0, existing.confidence + 0.5 * (confidence - existing.confidence))
            existing.last_updated = tick
            if support_keys:
                for sk in support_keys:
                    if sk not in existing.support_keys:
                        existing.support_keys.append(sk)

    def get(self, key: str) -> Optional[Proposition]:
        return self._nodes.get(key)

    def decay(self, tick: int, rate: float) -> None:
        """Reduce confidence on stale beliefs each tick."""
        for prop in self._nodes.values():
            age = tick - prop.last_updated
            prop.confidence = max(0.0, prop.confidence - rate * age * 0.1)

    def price_belief(self, asset_idx: int, horizon_tick: int) -> Optional[Tuple[float, float]]:
        """Return (expected_price, confidence) for an asset at a future tick, or None."""
        key = f"asset_{asset_idx}_price_tick_{horizon_tick}"
        prop = self._nodes.get(key)
        if prop is None:
            return None
        return prop.value, prop.confidence

    def all_propositions(self) -> List[Proposition]:
        return list(self._nodes.values())

    def highest_confidence_price(self, asset_idx: int, current_tick: int) -> Optional[Tuple[int, float, float]]:
        """Return (target_tick, price, confidence) for the highest-confidence future price belief."""
        best = None
        for key, prop in self._nodes.items():
            if not key.startswith(f"asset_{asset_idx}_price_tick_"):
                continue
            try:
                t = int(key.split("_tick_")[1])
            except (IndexError, ValueError):
                continue
            if t <= current_tick:
                continue
            if best is None or prop.confidence > best[2]:
                best = (t, prop.value, prop.confidence)
        return best
