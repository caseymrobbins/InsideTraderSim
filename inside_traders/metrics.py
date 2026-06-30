"""Metrics collection, POLI×EH calculations, and Gini coefficient."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional
import numpy as np

if TYPE_CHECKING:
    from .agent import Agent
    from .venture import Venture


@dataclass
class TickSnapshot:
    tick: int
    # Per-agent arrays (indexed by agent order)
    agent_ids: List[str]
    wealth: np.ndarray
    poli: np.ndarray
    eh: np.ndarray
    poli_eh: np.ndarray

    # System-level
    gini: float
    total_wealth: float
    venture_success_rate: float
    deception_rate: float          # fraction of TELLs that were later refuted
    mean_trust: float              # average credibility across all pairs
    n_active_ventures: int
    asset_prices: np.ndarray


def gini(values: np.ndarray) -> float:
    """Gini coefficient for an array of non-negative values."""
    v = np.sort(np.maximum(values, 0.0))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.dot(idx, v) - (n + 1) * v.sum()) / (n * v.sum()))


class MetricsCollector:
    def __init__(self) -> None:
        self.snapshots: List[TickSnapshot] = []
        self._venture_resolved: int = 0
        self._venture_succeeded: int = 0
        self._tell_count: int = 0
        self._tell_refuted: int = 0

    def record_venture(self, succeeded: bool) -> None:
        self._venture_resolved += 1
        if succeeded:
            self._venture_succeeded += 1

    def record_tell(self, was_refuted: bool) -> None:
        self._tell_count += 1
        if was_refuted:
            self._tell_refuted += 1

    def snapshot(
        self,
        tick: int,
        agents: List["Agent"],
        market_prices: np.ndarray,
        active_ventures: List["Venture"],
        n_ticks_window: int,
    ) -> TickSnapshot:
        n = len(agents)
        agent_ids = [a.agent_id for a in agents]
        wealth = np.array([a.net_worth(market_prices) for a in agents])
        poli = np.array([a.poli_score(n_ticks_window) for a in agents])
        eh = np.array([a.epistemic_health(agents) for a in agents])
        poli_eh = poli * eh

        # Trust: average credibility each agent assigns to others
        trust_vals = []
        for ag in agents:
            for src, scores in ag.epistemic_model.credibility.items():
                if src != "self":
                    trust_vals.extend(scores.values())
        mean_trust = float(np.mean(trust_vals)) if trust_vals else 0.5

        snap = TickSnapshot(
            tick=tick,
            agent_ids=agent_ids,
            wealth=wealth,
            poli=poli,
            eh=eh,
            poli_eh=poli_eh,
            gini=gini(wealth),
            total_wealth=float(wealth.sum()),
            venture_success_rate=(
                self._venture_succeeded / self._venture_resolved
                if self._venture_resolved > 0 else 0.0
            ),
            deception_rate=(
                self._tell_refuted / self._tell_count
                if self._tell_count > 0 else 0.0
            ),
            mean_trust=mean_trust,
            n_active_ventures=len(active_ventures),
            asset_prices=market_prices.copy(),
        )
        self.snapshots.append(snap)
        return snap

    def poli_eh_correlation(self) -> float:
        """Pearson correlation between POLI and EH across last snapshot."""
        if not self.snapshots:
            return float("nan")
        s = self.snapshots[-1]
        if len(s.poli) < 2:
            return float("nan")
        return float(np.corrcoef(s.poli, s.eh)[0, 1])

    def wealth_trajectory(self, agent_id: str) -> List[float]:
        return [
            float(s.wealth[s.agent_ids.index(agent_id)])
            for s in self.snapshots
            if agent_id in s.agent_ids
        ]

    def compression_test_result(
        self,
        test_agent_ids: List[str],
        test_start_tick: int,
    ) -> Dict[str, Dict]:
        """
        Compare wealth growth rate before and after compression test begins.
        Returns {agent_id: {"pre_growth": float, "post_growth": float, "degraded": bool}}
        """
        pre = [s for s in self.snapshots if s.tick < test_start_tick]
        post = [s for s in self.snapshots if s.tick >= test_start_tick]
        results = {}
        for aid in test_agent_ids:
            def avg_wealth(snaps):
                vals = [
                    s.wealth[s.agent_ids.index(aid)]
                    for s in snaps if aid in s.agent_ids
                ]
                return float(np.mean(vals)) if vals else 0.0

            pre_w = avg_wealth(pre)
            post_w = avg_wealth(post)
            pre_g = (pre_w - avg_wealth(pre[:1])) if pre else 0.0
            post_g = post_w - pre_w
            results[aid] = {
                "pre_avg_wealth": pre_w,
                "post_avg_wealth": post_w,
                "pre_growth": pre_g,
                "post_growth": post_g,
                "degraded": post_g < pre_g * 0.5,
            }
        return results
