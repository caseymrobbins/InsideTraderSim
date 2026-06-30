"""Main simulation engine."""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import SimConfig
from .world import World
from .agent import Agent
from .market import OrderBook
from .communication import CommunicationGraph, Message, MessageType
from .venture import Venture, VentureStatus
from .metrics import MetricsCollector
from .evidence import EvidenceStatus

logger = logging.getLogger(__name__)


class Simulation:
    """
    Orchestrates the full HorizonSim v1 simulation loop.

    Tick structure:
      1. Hidden states evolve.
      2. Distribute prediction cards.
      3. Agents observe / buy signals → Observation Evidence.
      4. Communication round.
      5. Evidence resolution + belief/epistemic updates.
      6. Planning and action execution (trading, ventures, info purchases).
      7. Resolve matured contracts/ventures.
      8. Metric snapshot.
    """

    def __init__(self, cfg: Optional[SimConfig] = None) -> None:
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.world = World(self.cfg, self.rng)
        self.agents: List[Agent] = self._init_agents()
        self.order_book = OrderBook(
            n_assets=self.cfg.n_assets,
            initial_prices=self.world.fundamentals.copy(),
            impact=self.cfg.market_impact_factor,
        )
        self.comm_graph = CommunicationGraph(
            n_agents=self.cfg.n_agents,
            initial_neighbors=self.cfg.initial_neighbors,
            rng=self.rng,
        )
        self.ventures: List[Venture] = []
        self._venture_counter = 0
        self.metrics = MetricsCollector()
        self.tick = 0

        # Track which agents are compressed (for the compression test)
        self._compression_targets: List[str] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_agents(self) -> List[Agent]:
        agents = []
        # Pareto-distributed initial wealth
        raw = self.rng.pareto(self.cfg.wealth_pareto_alpha, self.cfg.n_agents)
        raw = raw / raw.max()
        wealth = (
            self.cfg.wealth_min
            + raw * (self.cfg.wealth_max - self.cfg.wealth_min)
        )
        for i in range(self.cfg.n_agents):
            agents.append(Agent(
                agent_id=f"agent_{i}",
                cfg=self.cfg,
                rng=np.random.default_rng(self.cfg.seed + i + 1),
                initial_cash=float(wealth[i]),
            ))
        return agents

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> MetricsCollector:
        logger.info("Starting simulation: %d agents, %d ticks", self.cfg.n_agents, self.cfg.n_ticks)

        for t in range(1, self.cfg.n_ticks + 1):
            self.tick = t
            self._step()

            if t % self.cfg.log_interval == 0:
                snap = self.metrics.snapshot(
                    tick=t,
                    agents=self.agents,
                    market_prices=self.order_book.prices,
                    active_ventures=[v for v in self.ventures if v.status == VentureStatus.ACTIVE],
                    n_ticks_window=self.cfg.poli_influence_window,
                )
                if self.cfg.verbose:
                    logger.info(
                        "Tick %4d | wealth_total=%.1f | gini=%.3f | "
                        "trust=%.3f | ventures=%d | deception=%.3f",
                        t, snap.total_wealth, snap.gini,
                        snap.mean_trust, snap.n_active_ventures,
                        snap.deception_rate,
                    )

        logger.info("Simulation complete.")
        return self.metrics

    def _step(self) -> None:
        t = self.tick

        # ── 1. Hidden states evolve ───────────────────────────────────
        self.world.step()
        self.order_book.reset_tick_volume()

        # ── 2. Distribute free prediction cards ──────────────────────
        n_cards_range = self.cfg.cards_per_tick_range
        for ag in self.agents:
            n = int(self.rng.integers(n_cards_range[0], n_cards_range[1] + 1))
            cards = self.world.issue_free_cards(ag.agent_id, n)
            ag.receive_cards(cards, t)

        # ── 3. Buy signals (if agent plans to) ───────────────────────
        # Agents that bought signal in plan step (processed below after planning)

        # ── 4. Communication round ────────────────────────────────────
        messages: List[Message] = []
        for ag in self.agents:
            neighbors = self.comm_graph.neighbors(ag.agent_id)
            msg = ag.compose_message(t, neighbors, self.order_book.prices)
            if msg is not None:
                messages.append(msg)

        # Deliver messages and collect replies
        agent_map = {ag.agent_id: ag for ag in self.agents}
        replies: List[Message] = []
        for msg in messages:
            receiver = agent_map.get(msg.receiver)
            if receiver is None:
                continue
            reply = receiver.receive_message(msg, t)
            if reply is not None:
                replies.append(reply)
            # Track deception: TELL messages that will resolve
            if msg.msg_type == MessageType.TELL and msg.proposition_key:
                # We'll evaluate this at evidence resolution later; register intent
                pass

        # Deliver ACCEPT replies → info transfer + payment
        for reply in replies:
            if reply.msg_type == MessageType.ACCEPT:
                sender = agent_map.get(reply.receiver)  # original offer sender
                receiver = agent_map.get(reply.sender)   # acceptor
                if sender and receiver and reply.price:
                    if receiver.pay(reply.price):
                        sender.receive_payment(reply.price)
                        # Send the actual info to receiver
                        offer_msg = sender._pending_offers.get(reply.offer_id or -1)
                        if offer_msg:
                            receiver.receive_message(offer_msg, t)
            elif reply.msg_type == MessageType.TELL:
                orig_asker = agent_map.get(reply.receiver)
                if orig_asker:
                    orig_asker.receive_message(reply, t)

        # INTRODUCE: randomly connect some agents via mutual friends
        for ag in self.agents:
            if self.rng.random() < self.cfg.introduce_prob:
                neighbors = self.comm_graph.neighbors(ag.agent_id)
                if len(neighbors) >= 2:
                    a, b = self.rng.choice(neighbors, size=2, replace=False)
                    self.comm_graph.introduce(a, b)
                    self.comm_graph.introduce(b, a)

        # ── 5. Evidence resolution + belief updates ───────────────────
        for ag in self.agents:
            ag.resolve_evidence(t, self.world.fundamentals)

        # Measure deception: TELL messages where source's evidence got refuted
        self._update_deception_metrics(messages, agent_map, t)

        # ── 6. Compression test injection ────────────────────────────
        if t == self.cfg.compression_test_start:
            self._select_compression_targets()
        if t >= self.cfg.compression_test_start and self._compression_targets:
            self._inject_bad_evidence(t)

        # ── 7. Planning + action execution ───────────────────────────
        for ag in self.agents:
            ag.update_intervention_view(t, self.order_book.prices)
            actions = ag.plan_actions(t, self.order_book.prices, self.ventures)

            for action in actions:
                if action["type"] == "trade":
                    self._execute_trade(ag, action, t)
                elif action["type"] == "venture_propose":
                    self._propose_venture(ag, action, t)
                elif action["type"] == "buy_signal":
                    self._buy_signal(ag, t)

        # ── 8. Resolve matured ventures ───────────────────────────────
        self._resolve_ventures(t)

        # ── 9. Snapshot wealth ────────────────────────────────────────
        for ag in self.agents:
            ag.snapshot_wealth(self.order_book.prices)

    # ------------------------------------------------------------------
    # Action execution helpers
    # ------------------------------------------------------------------

    def _execute_trade(self, agent: Agent, action: Dict, tick: int) -> None:
        asset_idx = action["asset_idx"]
        quantity = action["quantity"]
        price = self.order_book.prices[asset_idx]
        cost = abs(quantity) * price

        # Check solvency
        if quantity > 0 and agent.cash < cost * 0.5:
            quantity = (agent.cash * 0.4) / (price + 1e-9)
        if abs(quantity) < 0.01:
            return

        fill = self.order_book.submit(agent.agent_id, asset_idx, quantity, tick)
        agent.execute_trade(asset_idx, quantity, fill)

    def _propose_venture(self, agent: Agent, action: Dict, tick: int) -> None:
        principal = action["principal"]
        collateral = action["collateral"]

        if not agent.lock_collateral(collateral):
            return
        if not agent.pay(principal - collateral):
            agent.unlock_collateral(collateral)
            return

        # Find a willing counterparty with sufficient capital
        candidates = [
            a for a in self.agents
            if a.agent_id != agent.agent_id and a.cash > principal * 0.5
        ]
        if not candidates:
            agent.unlock_collateral(collateral)
            agent.receive_payment(principal - collateral)
            return

        # Pick counterparty probabilistically by preference alignment
        weights = np.array([
            a.preference_vector.get("wealth", 0.2) for a in candidates
        ])
        weights = weights / weights.sum()
        counterparty = candidates[int(self.rng.choice(len(candidates), p=weights))]

        if not counterparty.lock_collateral(collateral):
            agent.unlock_collateral(collateral)
            agent.receive_payment(principal - collateral)
            return
        if not counterparty.pay(principal - collateral):
            counterparty.unlock_collateral(collateral)
            agent.unlock_collateral(collateral)
            agent.receive_payment(principal - collateral)
            return

        duration = int(self.rng.integers(
            self.cfg.venture_duration_range[0],
            self.cfg.venture_duration_range[1] + 1
        ))
        v = Venture(
            venture_id=self._venture_counter,
            proposer=agent.agent_id,
            counterparty=counterparty.agent_id,
            asset_idx=action["asset_idx"],
            principal=principal,
            collateral=collateral,
            surplus_share_proposer=action.get("surplus_share", 0.5),
            start_tick=tick,
            duration=duration,
            hidden_quality=self.world.venture_quality_at(action["asset_idx"]),
            status=VentureStatus.ACTIVE,
        )
        self._venture_counter += 1
        self.ventures.append(v)
        agent.venture_ids.append(v.venture_id)
        counterparty.venture_ids.append(v.venture_id)

    def _buy_signal(self, agent: Agent, tick: int) -> None:
        card, cost = self.world.issue_premium_card(agent.agent_id)
        if agent.pay(cost):
            agent.receive_cards([card], tick)

    def _resolve_ventures(self, tick: int) -> None:
        agent_map = {ag.agent_id: ag for ag in self.agents}
        for v in self.ventures:
            if v.status != VentureStatus.ACTIVE:
                continue
            if v.end_tick > tick:
                continue

            actual_quality = self.world.venture_quality_at(v.asset_idx)
            payouts = v.resolve(actual_quality, self.cfg.venture_surplus_multiplier, tick)

            self.metrics.record_venture(succeeded=v.status == VentureStatus.SUCCEEDED)

            for aid, payout in payouts.items():
                ag = agent_map.get(aid)
                if ag is None:
                    continue
                ag.unlock_collateral(v.collateral)
                ag.receive_payment(payout)

    # ------------------------------------------------------------------
    # Compression test
    # ------------------------------------------------------------------

    def _select_compression_targets(self) -> None:
        """Target the top-POLI agents for epistemic degradation."""
        poli_scores = [
            (ag.agent_id, ag.poli_score(self.cfg.poli_influence_window))
            for ag in self.agents
        ]
        poli_scores.sort(key=lambda x: x[1], reverse=True)
        self._compression_targets = [
            aid for aid, _ in poli_scores[:self.cfg.compression_test_agents]
        ]
        logger.info(
            "Compression test started at tick %d, targets: %s",
            self.tick, self._compression_targets,
        )

    def _inject_bad_evidence(self, tick: int) -> None:
        """Feed targeted misinformation to compression-test agents."""
        from .evidence import Evidence, EvidenceType, EvidenceStatus
        from .agent import _next_ev_id
        agent_map = {ag.agent_id: ag for ag in self.agents}

        noise = self.cfg.compression_test_noise
        for aid in self._compression_targets:
            ag = agent_map.get(aid)
            if ag is None:
                continue
            asset_idx = int(self.rng.integers(0, self.cfg.n_assets))
            true_val = self.world.fundamentals[asset_idx]
            # Deliberately wrong prediction: flip direction with large error
            fake_val = true_val * (1.0 + noise * self.rng.choice([-1.0, 1.0]))
            horizon = tick + int(self.rng.integers(3, 8))
            key = f"asset_{asset_idx}_price_tick_{horizon}"

            ev = Evidence(
                ev_id=_next_ev_id(),
                source="oracle_fake",
                ev_type=EvidenceType.OBSERVATION,
                proposition=key,
                target_asset=asset_idx,
                target_tick=horizon,
                predicted_value=fake_val,
                confidence=0.85,  # injected with false high confidence
                timestamp=tick,
            )
            ag.evidence_ledger.append(ev)
            ag.belief_graph.add_or_update(
                key=key,
                value=fake_val,
                strength=1.0,
                confidence=0.85,
                tick=tick,
            )

    # ------------------------------------------------------------------
    # Deception tracking
    # ------------------------------------------------------------------

    def _update_deception_metrics(
        self,
        messages: List[Message],
        agent_map: Dict[str, Agent],
        tick: int,
    ) -> None:
        """
        For TELL messages whose proposition key resolves this tick,
        check if the sender's claimed value was wrong.
        """
        for msg in messages:
            if msg.msg_type != MessageType.TELL:
                continue
            if msg.proposition_key is None or msg.predicted_value is None:
                continue
            target_tick = Agent._parse_target_tick(msg.proposition_key)
            if target_tick is None or target_tick != tick:
                continue
            asset_idx = Agent._parse_asset_idx(msg.proposition_key)
            if asset_idx is None:
                continue
            actual = self.world.fundamentals[asset_idx]
            error_frac = abs(msg.predicted_value - actual) / (abs(actual) + 1e-9)
            was_refuted = error_frac > 0.10
            self.metrics.record_tell(was_refuted=was_refuted)

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def compression_test_summary(self) -> Dict:
        return self.metrics.compression_test_result(
            self._compression_targets,
            self.cfg.compression_test_start,
        )

    def poli_eh_correlation(self) -> float:
        return self.metrics.poli_eh_correlation()
