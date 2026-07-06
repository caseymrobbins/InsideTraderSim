"""Metrics collection, POLI×EH calculations, and Gini coefficient."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
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
    deception_rate: float          # fraction of TELL actions that used AMPLIFY or INVERT
    mean_trust: float              # average credibility across all pairs
    n_active_ventures: int
    asset_prices: np.ndarray

    # Extended instrumentation (defaults keep older construction paths valid)
    mean_integrity: float = 0.0                # mean agent _integrity_signal (relational agency)
    comm_action_mix: Optional[np.ndarray] = None  # fraction of agents whose top "against" action is each of {TRUTH,AMP,INV,SILENT,OFFER}
    q_against: Optional[np.ndarray] = None      # mean comm-Q per action in the pays-to-lie (against) state

    # Agency instrumentation — lets the metrics history track how the objective
    # treats agency over time (central to the sum-omission / missing-metric tests).
    # Populated from each agent's AgencyState (computed on demand for model U,
    # whose objective ignores agency and leaves _last_agency = None).
    min_ia: Optional[np.ndarray] = None         # per-agent bottleneck raw agency dim
    agency_dims: Optional[np.ndarray] = None    # (n_agents, 6): [liquidity, epistemic, network, solvency, options, integrity]
    frac_danger_zone: float = 0.0               # fraction of agents with min_ia < agency_floor
    mean_min_ia: float = 0.0                    # mean bottleneck agency across agents


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
        self._tells_total: int = 0
        self._tells_deceptive: int = 0
        # Per-alignment-bucket counts for the incentive-linkage test.
        # Keys: 0=against (position opposes belief → pays-to-lie state),
        #       1=neutral, 2=with (position agrees → pays-to-be-honest state).
        # Value: [total_tells, deceptive_tells]
        self._tells_by_align: Dict[int, List[int]] = {0: [0, 0], 1: [0, 0], 2: [0, 0]}

    def record_venture(self, succeeded: bool) -> None:
        self._venture_resolved += 1
        if succeeded:
            self._venture_succeeded += 1

    def record_tell(self, is_deceptive: bool, align_bucket: int = 1) -> None:
        """Record a TELL message. is_deceptive=True when action was AMPLIFY or INVERT.

        align_bucket: 0=against (position opposes belief, pays-to-lie state),
                      1=neutral, 2=with (position agrees, pays-to-be-honest state).
        """
        self._tells_total += 1
        if is_deceptive:
            self._tells_deceptive += 1
        bucket = align_bucket if align_bucket in (0, 1, 2) else 1
        self._tells_by_align[bucket][0] += 1
        if is_deceptive:
            self._tells_by_align[bucket][1] += 1

    def reset_linkage_counters(self) -> None:
        """Zero the per-alignment deception counters.

        Used to measure incentive-linkage over the CONVERGED phase only: the
        cumulative counts over all of Phase B are dominated by early epsilon-greedy
        exploration (≈uniform actions), which dilutes the learned linkage.  Call
        once after exploration has decayed to read the learned policy's linkage.
        """
        self._tells_by_align = {0: [0, 0], 1: [0, 0], 2: [0, 0]}

    def incentive_linkage_stats(self) -> Dict[str, Tuple[int, int, float]]:
        """Deception counts and rate per alignment bucket for the incentive-linkage test.

        Returns {label: (total_tells, deceptive_tells, deception_rate)} where
        label is one of 'against' | 'neutral' | 'with'.
        The fix is verified if deception_rate['against'] > deception_rate['with'].
        """
        labels = {0: "against", 1: "neutral", 2: "with"}
        result: Dict[str, Tuple[int, int, float]] = {}
        for bucket, (total, deceptive) in self._tells_by_align.items():
            rate = deceptive / total if total > 0 else 0.0
            result[labels[bucket]] = (total, deceptive, rate)
        return result

    def snapshot(
        self,
        tick: int,
        agents: List["Agent"],
        market_prices: np.ndarray,
        active_ventures: List["Venture"],
        n_ticks_window: int,
        # Optional pre-computed GPU arrays (passed from Simulation._take_snapshot)
        _wealth_override: Optional[np.ndarray] = None,
        _poli_override: Optional[np.ndarray] = None,
        _eh_override: Optional[np.ndarray] = None,
    ) -> TickSnapshot:
        n = len(agents)
        agent_ids = [a.agent_id for a in agents]
        wealth = _wealth_override if _wealth_override is not None else np.array([a.net_worth(market_prices) for a in agents])
        poli   = _poli_override   if _poli_override   is not None else np.array([a.poli_score(n_ticks_window) for a in agents])
        eh     = _eh_override     if _eh_override     is not None else np.array([a.epistemic_health(agents) for a in agents])
        poli_eh = poli * eh

        # Trust: average credibility each agent assigns to others
        trust_vals = []
        for ag in agents:
            for src, scores in ag.epistemic_model.credibility.items():
                if src != "self":
                    trust_vals.extend(scores.values())
        mean_trust = float(np.mean(trust_vals)) if trust_vals else 0.5

        # Relational agency + comm-strategy instrumentation
        mean_integrity = float(np.mean([
            getattr(a, "_integrity_signal", 0.0) for a in agents
        ])) if agents else 0.0
        AGAINST = 0  # position-opposes-belief state (the pays-to-lie situation)
        q_rows, top_actions = [], []
        for a in agents:
            q = getattr(a, "_comm_q", None)
            if q is None or q.shape != (3, 3, 5):
                continue
            row = q[:, AGAINST, :].mean(axis=0)   # mean over confidence buckets → 5-vec
            q_rows.append(row)
            top_actions.append(int(np.argmax(row)))
        q_against = np.mean(q_rows, axis=0) if q_rows else np.zeros(5)
        comm_action_mix = (np.bincount(top_actions, minlength=5) / len(top_actions)
                           if top_actions else np.zeros(5))

        # Agency instrumentation. Each agent's AgencyState is normally populated
        # by plan_actions (_last_agency). Pure-utility objectives (model U) skip
        # agency entirely, leaving it None — but we still MEASURE agency here so
        # its erosion under an objective that ignores it is observable. We recompute
        # on demand in that case, mirroring experiments/runner.py::_summarise.
        min_ia_arr, agency_dims_arr, mean_min_ia, frac_danger_zone = self._collect_agency(
            agents, market_prices, tick
        )

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
                self._tells_deceptive / self._tells_total
                if self._tells_total > 0 else 0.0
            ),
            mean_trust=mean_trust,
            n_active_ventures=len(active_ventures),
            asset_prices=market_prices.copy(),
            mean_integrity=mean_integrity,
            comm_action_mix=comm_action_mix,
            q_against=q_against,
            min_ia=min_ia_arr,
            agency_dims=agency_dims_arr,
            mean_min_ia=mean_min_ia,
            frac_danger_zone=frac_danger_zone,
        )
        self.snapshots.append(snap)
        return snap

    @staticmethod
    def _collect_agency(
        agents: List["Agent"],
        market_prices: np.ndarray,
        tick: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
        """
        Read each agent's agency vector for the metrics history.

        Uses the AgencyState already computed in plan_actions (agent._last_agency).
        For objectives that never touch agency (model U) it is None, so we recompute
        it on demand — the point is to observe agency even when the objective ignores
        it. Returns (min_ia[n], agency_dims[n,6], mean_min_ia, frac_danger_zone).
        The 6 dims are [liquidity, epistemic, network, solvency, options, integrity].
        """
        if not agents:
            return None, None, 0.0, 0.0

        # Lazy imports avoid a circular dependency (horizon imports nothing from
        # metrics, but keep the import local to be safe and cheap).
        from .horizon import compute_agency_state, AGENCY_FLOOR

        cfg = getattr(agents[0], "cfg", None)
        n_ag = len(agents)
        max_deg = max(getattr(cfg, "initial_neighbors", 1) * 3, 1) if cfg else 1
        floor = getattr(cfg, "agency_floor", AGENCY_FLOOR) if cfg else AGENCY_FLOOR

        min_ia = np.zeros(n_ag)
        agency_dims = np.zeros((n_ag, 6))
        for i, a in enumerate(agents):
            ag = getattr(a, "_last_agency", None)
            if ag is None:
                try:
                    ag = compute_agency_state(a, market_prices, n_ag, max_deg, tick, a.cfg)
                except Exception:
                    # If agency can't be computed for this agent, leave zeros.
                    continue
            min_ia[i] = ag.min_ia
            agency_dims[i] = [
                ag.ia_liquidity, ag.ia_epistemic, ag.ia_network,
                ag.ia_solvency, ag.ia_options, ag.ia_integrity,
            ]

        mean_min_ia = float(min_ia.mean())
        frac_danger_zone = float((min_ia < floor).mean())
        return min_ia, agency_dims, mean_min_ia, frac_danger_zone

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
