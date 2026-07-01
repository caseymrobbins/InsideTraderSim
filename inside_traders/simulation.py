"""Main simulation engine — HorizonSim v1 with GPU acceleration."""
from __future__ import annotations
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import SimConfig
from .world import World
from .agent import Agent
from .market import OrderBook
from .communication import CommunicationGraph, Message, MessageType
from .venture import Venture, VentureStatus
from .metrics import MetricsCollector
from .evidence import EvidenceStatus, Evidence, EvidenceType
from .accelerator import (
    get_array_module,
    VectorizedAgentState,
    gini_gpu,
    _to_numpy,
)

logger = logging.getLogger(__name__)

# ANSI colour helpers (stripped on non-tty automatically)
_USE_COLOR = sys.stdout.isatty()
_G  = "\033[32m" if _USE_COLOR else ""
_Y  = "\033[33m" if _USE_COLOR else ""
_C  = "\033[36m" if _USE_COLOR else ""
_R  = "\033[31m" if _USE_COLOR else ""
_B  = "\033[1m"  if _USE_COLOR else ""
_RS = "\033[0m"  if _USE_COLOR else ""

_BAR_CHARS = " ▁▂▃▄▅▆▇█"

def _sparkbar(values: np.ndarray, width: int = 8) -> str:
    """Tiny ASCII bar showing relative per-agent wealth (top-k)."""
    if len(values) == 0:
        return ""
    mn, mx = values.min(), values.max()
    rng = mx - mn if mx > mn else 1.0
    bars = "".join(_BAR_CHARS[int((v - mn) / rng * 8)] for v in values[:width])
    return bars


class Simulation:
    """
    Orchestrates the full HorizonSim v1 simulation loop.

    GPU acceleration (when device="cuda"):
      - World GBM dynamics run via VectorizedWorld
      - Agent position matrix + cash vector stored as device tensors
      - Batch net-worth, Gini, POLI computed in single GPU kernel calls
      - Cognitive state (BeliefGraph, EvidenceLedger) stays on CPU

    Output schedule:
      - Every status_interval ticks : one-line live console status
      - Every plot_interval ticks   : mid-run dashboard PNG saved to plot_dir
      - Every log_interval ticks    : MetricsCollector snapshot

    Tick structure:
      1. Hidden states evolve (GPU).
      2. Distribute prediction cards.
      3. Communication round.
      4. Evidence resolution + belief/epistemic updates.
      5. Compression test injection (if active).
      6. Planning + action execution.
      7. Resolve matured ventures.
      8. GPU batch sync → metrics snapshot.
    """

    def __init__(self, cfg: Optional[SimConfig] = None) -> None:
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        # ── Device setup ─────────────────────────────────────────────
        self.xp, self.device = get_array_module(self.cfg.device)

        # ── World ────────────────────────────────────────────────────
        self.world = World(self.cfg, self.rng, xp=self.xp)

        # ── Agents + vectorised state ────────────────────────────────
        self.agents: List[Agent] = self._init_agents()
        initial_cash = np.array([ag.cash for ag in self.agents])
        self.vstate = VectorizedAgentState(
            n_agents=self.cfg.n_agents,
            n_assets=self.cfg.n_assets,
            initial_cash=initial_cash,
            xp=self.xp,
        )

        # ── Market / communication / ventures ─────────────────────────
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

        # ── Compression test ─────────────────────────────────────────
        self._compression_targets: List[str] = []

        # ── Timing ───────────────────────────────────────────────────
        self._t0: float = 0.0
        self._tick_times: List[float] = []  # rolling last-N tick durations

        # ── Plot dir ─────────────────────────────────────────────────
        os.makedirs(self.cfg.plot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_agents(self) -> List[Agent]:
        agents = []
        raw = self.rng.pareto(self.cfg.wealth_pareto_alpha, self.cfg.n_agents)
        raw = raw / raw.max()
        wealth = self.cfg.wealth_min + raw * (self.cfg.wealth_max - self.cfg.wealth_min)

        # Resolve reward model once so all agents share the same instance (or mixed)
        from .reward import REWARD_MODELS, RewardModelName
        default_rm = REWARD_MODELS[RewardModelName(self.cfg.reward_model)]

        compressed_ids = set(self.cfg.exp_compressed_agent_ids)

        for i in range(self.cfg.n_agents):
            initial_cash = float(wealth[i])

            # Pre-seed compressed agents: high wealth, degraded prediction cards (via cfg clone),
            # and near-isolated comm graph — creates high-POLI / low-EH archetype from tick 0.
            if i in compressed_ids and compressed_ids:
                initial_cash *= self.cfg.exp_compressed_wealth_mul
                agent_cfg = SimConfig(**{
                    **self.cfg.__dict__,
                    "card_noise_low_quality": self.cfg.exp_compressed_card_noise,
                    "card_noise_high_quality": self.cfg.exp_compressed_card_noise,
                    "initial_neighbors": self.cfg.exp_compressed_network_degree,
                })
            else:
                agent_cfg = self.cfg

            agents.append(Agent(
                agent_id=f"agent_{i}",
                cfg=agent_cfg,
                rng=np.random.default_rng(self.cfg.seed + i + 1),
                initial_cash=initial_cash,
                reward_model=default_rm,
            ))
        return agents

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> MetricsCollector:
        self._t0 = time.perf_counter()
        _print_run_header(self.cfg, self.device)

        for t in range(1, self.cfg.n_ticks + 1):
            self.tick = t
            tick_start = time.perf_counter()
            self._step()
            tick_dur = time.perf_counter() - tick_start
            self._tick_times.append(tick_dur)
            if len(self._tick_times) > 50:
                self._tick_times.pop(0)

            # ── Metric snapshot every log_interval ───────────────────
            if t % self.cfg.log_interval == 0:
                prices_v = self.xp.asarray(self.order_book.prices, dtype=self.xp.float64)
                snap = self._take_snapshot(t, prices_v)

                # ── Live status every status_interval ────────────────
                if t % self.cfg.status_interval == 0:
                    self._print_status(t, snap, prices_v)

                # ── Mid-run plots every plot_interval ─────────────────
                if t % self.cfg.plot_interval == 0:
                    self._save_midrun_plots(t)

        elapsed = time.perf_counter() - self._t0
        print(f"\n{_B}  Done.{_RS} {self.cfg.n_ticks} ticks in {elapsed:.2f}s "
              f"({self.cfg.n_ticks/elapsed:.0f} ticks/sec) | device={self.device}")
        return self.metrics

    # ------------------------------------------------------------------
    # Per-tick step
    # ------------------------------------------------------------------

    def _step(self) -> None:
        t = self.tick

        # ── 1. Hidden states evolve (on device) ──────────────────────
        self.world.step()
        self.order_book.reset_tick_volume()

        # ── 2. Distribute prediction cards ───────────────────────────
        n_lo, n_hi = self.cfg.cards_per_tick_range
        for ag in self.agents:
            n = int(self.rng.integers(n_lo, n_hi + 1))
            cards = self.world.issue_free_cards(ag.agent_id, n)
            ag.receive_cards(cards, t)

        # ── 3. Communication round ────────────────────────────────────
        agent_map = {ag.agent_id: ag for ag in self.agents}
        messages: List[Message] = []
        if self.cfg.comm_enabled:
            for ag in self.agents:
                neighbors = self.comm_graph.neighbors(ag.agent_id)
                msg = ag.compose_message(t, neighbors, self.order_book.prices)
                if msg is not None:
                    messages.append(msg)

            replies: List[Message] = []
            for msg in messages:
                receiver = agent_map.get(msg.receiver)
                if receiver is None:
                    continue
                reply = receiver.receive_message(msg, t)
                if reply is not None:
                    replies.append(reply)

            for reply in replies:
                if reply.msg_type == MessageType.ACCEPT:
                    sender = agent_map.get(reply.receiver)
                    receiver = agent_map.get(reply.sender)
                    if sender and receiver and reply.price:
                        if receiver.pay(reply.price):
                            sender.receive_payment(reply.price)
                            offer_msg = sender._pending_offers.get(reply.offer_id or -1)
                            if offer_msg:
                                receiver.receive_message(offer_msg, t)
                elif reply.msg_type == MessageType.TELL:
                    orig_asker = agent_map.get(reply.receiver)
                    if orig_asker:
                        orig_asker.receive_message(reply, t)

            # INTRODUCE
            for ag in self.agents:
                if self.rng.random() < self.cfg.introduce_prob:
                    neighbors = self.comm_graph.neighbors(ag.agent_id)
                    if len(neighbors) >= 2:
                        a, b = self.rng.choice(neighbors, size=2, replace=False)
                        self.comm_graph.introduce(a, b)
                        self.comm_graph.introduce(b, a)
        else:
            # comm_enabled=False: clear neighbor lists so agency indices reflect isolation
            for ag in self.agents:
                ag._current_neighbors = []

        # ── 4. Evidence resolution + belief updates ───────────────────
        for ag in self.agents:
            ag.resolve_evidence(t, self.world.fundamentals)
        if self.cfg.comm_enabled:
            self._update_deception_metrics(messages, agent_map, t)

        # ── 5. Compression test ───────────────────────────────────────
        if t == self.cfg.compression_test_start:
            self._select_compression_targets()
        if t >= self.cfg.compression_test_start and self._compression_targets:
            self._inject_bad_evidence(t)

        # ── 6. Planning + execution ───────────────────────────────────
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

        # ── 7. Resolve ventures ───────────────────────────────────────
        self._resolve_ventures(t)

        # ── 8. Sync agent state → GPU tensor ──────────────────────────
        self.vstate.push_from_agents(self.agents)
        for ag in self.agents:
            ag.snapshot_wealth(self.order_book.prices)

    # ------------------------------------------------------------------
    # Metric snapshot (GPU-accelerated batch computations)
    # ------------------------------------------------------------------

    def _take_snapshot(self, t: int, prices_v) -> object:
        """Compute snapshot using GPU tensors where possible."""
        net_worths_v = self.vstate.net_worths(prices_v)
        poli_v = self.vstate.poli_scores(prices_v, self.cfg.poli_influence_window)

        # EH must be computed on CPU (involves Python graph traversal)
        eh_np = np.array([ag.epistemic_health(self.agents) for ag in self.agents])

        snap = self.metrics.snapshot(
            tick=t,
            agents=self.agents,
            market_prices=self.order_book.prices,
            active_ventures=[v for v in self.ventures if v.status == VentureStatus.ACTIVE],
            n_ticks_window=self.cfg.poli_influence_window,
            # Optionally pass pre-computed GPU arrays for speed
            _wealth_override=_to_numpy(net_worths_v),
            _poli_override=_to_numpy(poli_v),
            _eh_override=eh_np,
        )
        return snap

    # ------------------------------------------------------------------
    # Live status output
    # ------------------------------------------------------------------

    def _print_status(self, t: int, snap, prices_v) -> None:
        net_worths_v = self.vstate.net_worths(prices_v)
        wealth_np = _to_numpy(net_worths_v)
        top5 = np.sort(wealth_np)[::-1][:5]
        gini = gini_gpu(net_worths_v, self.xp)
        avg_tick_ms = 1000 * np.mean(self._tick_times) if self._tick_times else 0.0
        elapsed = time.perf_counter() - self._t0
        eta = (self.cfg.n_ticks - t) * avg_tick_ms / 1000.0

        n_active = sum(1 for v in self.ventures if v.status == VentureStatus.ACTIVE)
        compression_flag = (
            f" {_R}[COMPRESS]{_RS}" if t >= self.cfg.compression_test_start else ""
        )

        # Build the status line
        bar = _sparkbar(np.sort(wealth_np)[::-1])
        prices_str = " ".join(f"{p:.0f}" for p in self.order_book.prices[:3])
        print(
            f"{_B}tick {t:>4}/{self.cfg.n_ticks}{_RS}"
            f" │ {_G}wealth_tot={snap.total_wealth:>10.1f}{_RS}"
            f" │ {_Y}gini={gini:.3f}{_RS}"
            f" │ {_C}trust={snap.mean_trust:.3f}{_RS}"
            f" │ decept={snap.deception_rate:.3f}"
            f" │ ventures={n_active:>2}"
            f" │ px=[{prices_str}…]"
            f" │ {avg_tick_ms:.1f}ms/tk"
            f" │ ETA={eta:.0f}s"
            f"{compression_flag}"
            f"  {bar}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Mid-run dashboard plot (every plot_interval ticks)
    # ------------------------------------------------------------------

    def _save_midrun_plots(self, t: int) -> None:
        from . import visualization as viz
        path = viz.plot_midrun_dashboard(self, t, self.cfg.plot_dir)
        print(f"  {_C}[plot]{_RS} {path}", flush=True)

    # ------------------------------------------------------------------
    # Action execution helpers (unchanged logic, same as before)
    # ------------------------------------------------------------------

    def _execute_trade(self, agent: Agent, action: Dict, tick: int) -> None:
        asset_idx = action["asset_idx"]
        quantity = action["quantity"]
        price = self.order_book.prices[asset_idx]
        cost = abs(quantity) * price

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

        candidates = [
            a for a in self.agents
            if a.agent_id != agent.agent_id and a.cash > principal * 0.5
        ]
        if not candidates:
            agent.unlock_collateral(collateral)
            agent.receive_payment(principal - collateral)
            return

        weights = np.array([a.preference_vector.get("wealth", 0.2) for a in candidates])
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
            self.cfg.venture_duration_range[1] + 1,
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
        prices_v = self.xp.asarray(self.order_book.prices, dtype=self.xp.float64)
        poli_np = _to_numpy(self.vstate.poli_scores(prices_v, self.cfg.poli_influence_window))
        order = np.argsort(poli_np)[::-1]
        self._compression_targets = [
            self.agents[i].agent_id for i in order[:self.cfg.compression_test_agents]
        ]
        print(
            f"\n  {_R}{_B}[compression]{_RS} EH injection begins at tick {self.tick} "
            f"→ targets: {self._compression_targets}",
            flush=True,
        )

    def _inject_bad_evidence(self, tick: int) -> None:
        from .agent import _next_ev_id
        agent_map = {ag.agent_id: ag for ag in self.agents}
        noise = self.cfg.compression_test_noise
        for aid in self._compression_targets:
            ag = agent_map.get(aid)
            if ag is None:
                continue
            asset_idx = int(self.rng.integers(0, self.cfg.n_assets))
            true_val = self.world.fundamentals[asset_idx]
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
                confidence=0.85,
                timestamp=tick,
            )
            ag.evidence_ledger.append(ev)
            ag.belief_graph.add_or_update(
                key=key, value=fake_val, strength=1.0, confidence=0.85, tick=tick,
            )

    # ------------------------------------------------------------------
    # Deception tracking
    # ------------------------------------------------------------------

    def _update_deception_metrics(
        self, messages: List[Message], agent_map: Dict[str, Agent], tick: int
    ) -> None:
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
            self.metrics.record_tell(was_refuted=error_frac > 0.10)

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def compression_test_summary(self) -> Dict:
        return self.metrics.compression_test_result(
            self._compression_targets, self.cfg.compression_test_start,
        )

    def poli_eh_correlation(self) -> float:
        return self.metrics.poli_eh_correlation()


# ------------------------------------------------------------------
# Header printer
# ------------------------------------------------------------------

def _print_run_header(cfg: SimConfig, device: str) -> None:
    gpu_tag = f"{_G}GPU:{device}{_RS}" if device != "cpu" else f"{_Y}CPU{_RS}"
    print(f"""
{_B}{'='*70}{_RS}
  {_B}InsideTraderSim — HorizonSim v1{_RS}   [{gpu_tag}]
{'='*70}
  Agents       : {cfg.n_agents}
  Ticks        : {cfg.n_ticks}
  Assets       : {cfg.n_assets}  |  Ventures : {cfg.n_ventures}
  Seed         : {cfg.seed}
  Status every : {cfg.status_interval} ticks
  Plots every  : {cfg.plot_interval} ticks  → {cfg.plot_dir}/
  Compression  : starts tick {cfg.compression_test_start} ({cfg.compression_test_agents} agents)
{_B}{'='*70}{_RS}
  {'tick':>9}  │  wealth_total  │  gini  │  trust  │  decept  │  ventures  │  ...
{'─'*70}""", flush=True)
