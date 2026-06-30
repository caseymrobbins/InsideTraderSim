"""HorizonSim v1 Agent: full cognitive architecture with reward model integration."""
from __future__ import annotations
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import numpy as np

from .belief import BeliefGraph
from .evidence import Evidence, EvidenceType, EvidenceStatus, EpistemicModel
from .communication import Message, MessageType
from .config import SimConfig

if TYPE_CHECKING:
    from .world import World, PredictionCard
    from .market import OrderBook, Fill
    from .venture import Venture
    from .reward import RewardModel
    from .horizon import AgencyState


_EV_COUNTER = 0


def _next_ev_id() -> int:
    global _EV_COUNTER
    _EV_COUNTER += 1
    return _EV_COUNTER


class Agent:
    """
    HorizonSim v1 cognitive agent.

    Reward model integration
    ------------------------
    Each agent carries a RewardModel instance (default: RewardU = plain utility).
    At planning time, compute_reward_modulation() evaluates the current agency
    state and returns scaling factors that modify trade aggressiveness, venture
    probability, and information-buying frequency — without hard-coding strategies.

    The modulation is always *emergent*: the reward signal shapes the cost/benefit
    calculation, not the action selection rule.
    """

    def __init__(
        self,
        agent_id: str,
        cfg: SimConfig,
        rng: np.random.Generator,
        initial_cash: float,
        reward_model: Optional["RewardModel"] = None,
    ) -> None:
        self.agent_id = agent_id
        self.cfg = cfg
        self.rng = rng

        # --- Economic state ---
        self.cash: float = initial_cash
        self._initial_cash: float = initial_cash   # fixed reference for agency/utility
        self.positions: np.ndarray = np.zeros(cfg.n_assets)
        self.locked_collateral: float = 0.0

        # --- Reward model ---
        if reward_model is None:
            from .reward import get_reward_model
            reward_model = get_reward_model(
                cfg.reward_model,
                alpha=cfg.reward_fhi_weight,
                beta=cfg.reward_hi_sus_weight,
            )
        self.reward_model: "RewardModel" = reward_model

        # --- Cognitive architecture ---
        self.preference_vector: Dict[str, float] = self._init_preferences()
        self.belief_graph = BeliefGraph()
        self.evidence_ledger: List[Evidence] = []
        self.epistemic_model = EpistemicModel()

        # --- Communication ---
        self._outbox: List[Message] = []
        self._pending_offers: Dict[int, Message] = {}
        self._msg_counter = 0
        self._current_neighbors: List[str] = []  # set by Simulation each tick

        # --- Adaptive communication bias ---
        # Starts at 0 (neutral). Updated each tick from reward signal so the
        # agent can drift toward or away from accurate communication without
        # knowing that distortion = lying. No strategy is encoded here.
        self._comm_bias: float = 0.0
        self._prev_reward_signal: float = 0.0
        # Outgoing tell tracking: (prop_key, communicated_value, target_tick)
        self._sent_tells: List[Tuple[str, float, int]] = []
        self._outgoing_confirmed: int = 0
        self._outgoing_total: int = 0

        # --- Metrics tracking ---
        self.trade_history: List[Fill] = []
        self.venture_ids: List[int] = []
        self.wealth_history: List[float] = [initial_cash]
        self.poli_history: List[float] = []
        self.eh_history: List[float] = []
        self._market_impact_total: float = 0.0
        self._resolved_ev_correct: int = 0
        self._resolved_ev_total: int = 0

        # --- Agency / horizon index tracking ---
        self.agency_history: List["AgencyState"] = []
        self._last_agency: Optional["AgencyState"] = None

        # --- Intervention view ---
        self.intervention_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Preference initialisation
    # ------------------------------------------------------------------

    def _init_preferences(self) -> Dict[str, float]:
        raw = self.rng.dirichlet(np.ones(len(self.cfg.preference_dimensions)))
        return dict(zip(self.cfg.preference_dimensions, raw.tolist()))

    # ------------------------------------------------------------------
    # Step 1: Receive prediction cards → Observation Evidence
    # ------------------------------------------------------------------

    def receive_cards(self, cards: List["PredictionCard"], tick: int) -> None:
        for card in cards:
            key = f"asset_{card.asset_idx}_price_tick_{card.horizon}"
            ev = Evidence(
                ev_id=_next_ev_id(),
                source="self",
                ev_type=EvidenceType.OBSERVATION,
                proposition=key,
                target_asset=card.asset_idx,
                target_tick=card.horizon,
                predicted_value=card.predicted_value,
                confidence=card.confidence,
                timestamp=tick,
            )
            self.evidence_ledger.append(ev)
            self.belief_graph.add_or_update(
                key=key,
                value=card.predicted_value,
                strength=1.0 if card.predicted_value >= 0 else -1.0,
                confidence=card.confidence,
                tick=tick,
                support_keys=[f"ev_{ev.ev_id}"],
            )

    # ------------------------------------------------------------------
    # Step 2: Compose exactly one outgoing message
    # ------------------------------------------------------------------

    def compose_message(
        self,
        tick: int,
        neighbors: List[str],
        market_prices: np.ndarray,
    ) -> Optional[Message]:
        self._current_neighbors = neighbors  # cache for agency computation
        if not neighbors:
            return None

        receiver = self.rng.choice(neighbors)
        best_prop = self._best_proposition_to_share(tick)

        if best_prop is None:
            asset_idx = int(self.rng.integers(0, self.cfg.n_assets))
            return self._make_ask(receiver, asset_idx, tick)

        key, value, confidence = best_prop

        utility_share = self._utility_of_sharing(key, confidence)
        utility_offer = self._utility_of_offering(key, confidence)

        if utility_offer > utility_share and self.cash > self.cfg.signal_base_cost:
            msg_type = MessageType.OFFER
            price = self.cfg.signal_base_cost * (1.0 + confidence)
            communicated_value = value  # commercial offers: bias not applied
        else:
            msg_type = MessageType.TELL
            price = None
            # Apply learned communication bias. The agent doesn't know this
            # distorts truth — it's just its current communication scale.
            communicated_value = value * (1.0 + self._comm_bias)
            target_tick = self._parse_target_tick(key)
            if target_tick is not None:
                self._sent_tells.append((key, communicated_value, target_tick))

        msg = Message(
            msg_id=self._msg_counter,
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            tick=tick,
            proposition_key=key,
            predicted_value=communicated_value,
            confidence=confidence,
            price=price,
        )
        self._msg_counter += 1
        if msg_type == MessageType.OFFER:
            self._pending_offers[msg.msg_id] = msg
        return msg

    def _best_proposition_to_share(self, tick: int) -> Optional[Tuple[str, float, float]]:
        best = None
        for prop in self.belief_graph.all_propositions():
            try:
                res_tick = int(prop.key.split("_tick_")[1])
            except (IndexError, ValueError):
                continue
            if res_tick <= tick:
                continue
            if best is None or prop.confidence > best[2]:
                best = (prop.key, prop.value, prop.confidence)
        return best

    def _utility_of_sharing(self, key: str, confidence: float) -> float:
        w = self.preference_vector
        return (w["influence"] * confidence + w["knowledge"] * 0.3) * 0.5

    def _utility_of_offering(self, key: str, confidence: float) -> float:
        w = self.preference_vector
        return (w["wealth"] * confidence * self.cfg.signal_base_cost + w["influence"] * 0.2)

    def _make_ask(self, receiver: str, asset_idx: int, tick: int) -> Message:
        msg = Message(
            msg_id=self._msg_counter,
            sender=self.agent_id,
            receiver=receiver,
            msg_type=MessageType.ASK,
            tick=tick,
            query_key=f"asset_{asset_idx}_price",
        )
        self._msg_counter += 1
        return msg

    # ------------------------------------------------------------------
    # Step 3: Receive messages → Communication Evidence
    # ------------------------------------------------------------------

    def receive_message(self, msg: Message, tick: int) -> Optional[Message]:
        if msg.msg_type == MessageType.TELL:
            self._ingest_tell(msg, tick)
            return None
        if msg.msg_type == MessageType.OFFER:
            return self._evaluate_offer(msg, tick)
        if msg.msg_type == MessageType.ACCEPT:
            self._handle_accept(msg, tick)
            return None
        if msg.msg_type == MessageType.REJECT:
            return None
        if msg.msg_type == MessageType.ASK:
            return self._answer_ask(msg, tick)
        if msg.msg_type == MessageType.INTRODUCE:
            return None
        return None

    def _ingest_tell(self, msg: Message, tick: int) -> None:
        if msg.proposition_key is None or msg.predicted_value is None:
            return
        cred = self.epistemic_model.get(msg.sender, "price", default=0.5)
        effective_confidence = (msg.confidence or 0.5) * cred
        ev = Evidence(
            ev_id=_next_ev_id(),
            source=msg.sender,
            ev_type=EvidenceType.COMMUNICATION,
            proposition=msg.proposition_key,
            target_asset=self._parse_asset_idx(msg.proposition_key),
            target_tick=self._parse_target_tick(msg.proposition_key),
            predicted_value=msg.predicted_value,
            confidence=effective_confidence,
            timestamp=tick,
        )
        self.evidence_ledger.append(ev)
        self.belief_graph.add_or_update(
            key=msg.proposition_key,
            value=msg.predicted_value,
            strength=1.0,
            confidence=effective_confidence,
            tick=tick,
            support_keys=[f"ev_{ev.ev_id}"],
        )

    def _evaluate_offer(self, msg: Message, tick: int) -> Message:
        price = msg.price or self.cfg.signal_base_cost
        cred = self.epistemic_model.get(msg.sender, "price", default=0.5)
        expected_value_of_info = cred * (msg.confidence or 0.5) * price * 2.0
        w = self.preference_vector
        willing = (
            self.cash > price * 1.5
            and expected_value_of_info * w["knowledge"] > price * w["wealth"]
        )
        reply_type = MessageType.ACCEPT if willing else MessageType.REJECT
        return Message(
            msg_id=self._msg_counter,
            sender=self.agent_id,
            receiver=msg.sender,
            msg_type=reply_type,
            tick=tick,
            offer_id=msg.msg_id,
            price=price if willing else None,
        )

    def _handle_accept(self, msg: Message, tick: int) -> None:
        offer = self._pending_offers.get(msg.offer_id)
        if offer is None:
            return
        price = offer.price or 0.0
        self.cash += price

    def _answer_ask(self, msg: Message, tick: int) -> Optional[Message]:
        if msg.query_key is None:
            return None
        asset_idx = self._parse_asset_idx(msg.query_key)
        if asset_idx is None:
            return None
        best = self.belief_graph.highest_confidence_price(asset_idx, tick)
        if best is None:
            return None
        t, price, conf = best
        key = f"asset_{asset_idx}_price_tick_{t}"
        return Message(
            msg_id=self._msg_counter,
            sender=self.agent_id,
            receiver=msg.sender,
            msg_type=MessageType.TELL,
            tick=tick,
            proposition_key=key,
            predicted_value=price,
            confidence=conf * self.epistemic_model.get("self", "price", 0.7),
        )

    # ------------------------------------------------------------------
    # Step 4: Resolve Evidence + update epistemic model
    # ------------------------------------------------------------------

    def resolve_evidence(self, tick: int, world_fundamentals: "np.ndarray") -> None:
        for ev in self.evidence_ledger:
            if ev.status != EvidenceStatus.PENDING:
                continue
            if ev.target_tick is None or ev.target_tick > tick:
                continue
            if ev.target_asset is None:
                ev.status = EvidenceStatus.EXPIRED
                continue
            actual = float(world_fundamentals[ev.target_asset])
            ev.actual_value = actual
            ev.error = abs(ev.predicted_value - actual)
            is_correct = ev.error / (abs(actual) + 1e-9) < 0.10
            ev.status = EvidenceStatus.CONFIRMED if is_correct else EvidenceStatus.REFUTED
            self.epistemic_model.update(
                source=ev.source,
                prop_type="price",
                was_correct=is_correct,
                rate=self.cfg.credibility_update_rate,
            )
            self._resolved_ev_total += 1
            if is_correct:
                self._resolved_ev_correct += 1

        # Resolve outgoing tells: did what I actually send turn out to be right?
        # This measures the accuracy of communicated values (which may be biased),
        # not the accuracy of the agent's internal beliefs.
        remaining = []
        for tell_key, tell_value, tell_tick in self._sent_tells:
            if tell_tick > tick:
                remaining.append((tell_key, tell_value, tell_tick))
                continue
            asset_idx = self._parse_asset_idx(tell_key)
            if asset_idx is None:
                continue
            actual = float(world_fundamentals[asset_idx])
            err = abs(tell_value - actual)
            self._outgoing_total += 1
            if err / (abs(actual) + 1e-9) < 0.10:
                self._outgoing_confirmed += 1
        self._sent_tells = remaining

        self.belief_graph.decay(tick, self.cfg.belief_decay)

    def outgoing_accuracy(self) -> float:
        """Fraction of sent TELL predictions that turned out to be correct."""
        if self._outgoing_total == 0:
            return 0.5  # neutral prior: no history yet
        return self._outgoing_confirmed / self._outgoing_total

    # ------------------------------------------------------------------
    # Step 5: Update Intervention view
    # ------------------------------------------------------------------

    def update_intervention_view(self, tick: int, market_prices: np.ndarray) -> None:
        portfolio_value = self.net_worth(market_prices)
        self.intervention_log.append({
            "tick": tick,
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "n_beliefs": len(self.belief_graph.all_propositions()),
            "agency": self._last_agency,
        })
        if len(self.intervention_log) > 50:
            self.intervention_log.pop(0)

    # ------------------------------------------------------------------
    # Step 6: Plan  (reward-model-modulated)
    # ------------------------------------------------------------------

    def plan_actions(
        self,
        tick: int,
        market_prices: np.ndarray,
        active_ventures: List["Venture"],
    ) -> List[Dict]:
        actions = []
        w = self.preference_vector

        # ── Compute reward modulation (once per planning cycle) ───────
        mod = self._compute_reward_modulation(tick, market_prices)
        trade_scale    = mod["trade_scale"]
        venture_scale  = mod["venture_scale"]
        info_scale     = mod["info_scale"]

        # ── Trading decisions ─────────────────────────────────────────
        for asset_idx in range(self.cfg.n_assets):
            belief = self.belief_graph.highest_confidence_price(asset_idx, tick)
            if belief is None:
                continue
            target_tick, predicted_price, confidence = belief

            current_price = market_prices[asset_idx]
            expected_return = (predicted_price - current_price) / (current_price + 1e-9)
            risk_adj_return = expected_return * confidence
            base_utility = risk_adj_return * w["wealth"] - abs(risk_adj_return) * w["security"] * 0.5
            modulated_utility = base_utility * trade_scale

            if abs(modulated_utility) < 0.005:
                continue

            desired_position = self.cfg.max_position * confidence * np.sign(modulated_utility)
            current = self.positions[asset_idx]
            order_qty = np.clip(desired_position - current, -50.0, 50.0)
            if abs(order_qty) < 0.1:
                continue
            cost = abs(order_qty) * current_price
            if order_qty > 0 and cost > self.cash * 0.4:
                order_qty = (self.cash * 0.4) / (current_price + 1e-9)
            if abs(order_qty) >= 0.1:
                actions.append({
                    "type": "trade",
                    "asset_idx": asset_idx,
                    "quantity": float(order_qty),
                })

        # ── Venture proposals ─────────────────────────────────────────
        venture_prob = 0.03 * w["knowledge"] * venture_scale
        if w["wealth"] > 0.3 and self.cash > 20.0 and self.rng.random() < venture_prob:
            asset_idx = int(self.rng.integers(0, self.cfg.n_assets))
            principal = min(self.cash * 0.1, 30.0)
            collateral = principal * self.cfg.venture_collateral_fraction
            actions.append({
                "type": "venture_propose",
                "asset_idx": asset_idx,
                "principal": principal,
                "collateral": collateral,
                "surplus_share": 0.5 + 0.1 * w["wealth"],
            })

        # ── Buy premium signal ────────────────────────────────────────
        signal_utility = (w["knowledge"] * 2.0 + w["wealth"] * 0.5) * info_scale
        if (signal_utility > 1.0 and self.cash > self.cfg.signal_base_cost * 2
                and self.rng.random() < min(0.5, 0.15 * info_scale)):
            actions.append({"type": "buy_signal", "premium": True})

        return actions

    # ------------------------------------------------------------------
    # Reward modulation engine
    # ------------------------------------------------------------------

    def _compute_reward_modulation(
        self, tick: int, market_prices: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the reward model and return per-action scaling factors:
          trade_scale   — multiplier on trading aggressiveness
          venture_scale — multiplier on venture proposal probability
          info_scale    — multiplier on information purchasing probability
        """
        from .reward import RewardModelName, compute_base_utility
        from .horizon import compute_agency_state

        # Plain U model: no agency modulation on trading, but comm bias still adapts
        if self.reward_model.name == RewardModelName.U:
            utility = compute_base_utility(self, market_prices)
            self._adapt_comm_bias(utility)
            return {"trade_scale": 1.0, "venture_scale": 1.0, "info_scale": 1.0}

        # Compute agency indices
        n_agents = self.cfg.n_agents
        max_degree = max(self.cfg.initial_neighbors * 3, 1)
        agency = compute_agency_state(
            agent=self,
            market_prices=market_prices,
            n_agents=n_agents,
            max_degree=max_degree,
            tick=tick,
            cfg=self.cfg,
        )
        self._last_agency = agency

        # Store periodically to avoid memory blow-up
        if tick % 10 == 0:
            self.agency_history.append(agency)
            if len(self.agency_history) > 100:
                self.agency_history.pop(0)

        utility = compute_base_utility(self, market_prices)
        reward = self.reward_model.compute(utility, agency)
        self._adapt_comm_bias(reward)

        if agency.in_danger_zone:
            # ── Danger zone ────────────────────────────────────────────
            # Determine which dimension is weakest → prioritise restoring it
            ia_vec = agency.ia_vector
            weakest = int(np.argmin(ia_vec))
            # [0]=liquidity [1]=epistemic [2]=network [3]=solvency
            return {
                "trade_scale": 0.2,           # pull back from speculative trading
                "venture_scale": 0.05,        # freeze new ventures
                "info_scale": 3.0 if weakest == 1 else 1.2,  # seek information if EH weak
                "reward": reward,
                "agency": agency,
            }
        else:
            # ── Safe zone ─────────────────────────────────────────────
            # reward is in (-∞, +∞); tanh maps to (-1, 1); +1 centres around 1.0
            scale = math.tanh(reward * 0.5) + 1.0   # ∈ (0, 2)
            # For UHFS: sustainability dampens ventures, FHI boosts information
            hi_sus = agency.hi_sus if hasattr(agency, "hi_sus") else 1.0
            fhi    = agency.fhi    if hasattr(agency, "fhi")    else 1.0
            return {
                "trade_scale": scale,
                "venture_scale": scale * hi_sus,
                "info_scale": scale * (1.0 + 0.5 * fhi),
                "reward": reward,
                "agency": agency,
            }

    def _adapt_comm_bias(self, current_reward: float) -> None:
        """
        Update the communication bias from the reward signal.

        If reward went up → reinforce whatever direction bias is currently in.
        If reward went down → pull bias back toward 0 (more accurate / honest).
        Near zero → inject tiny random exploration so bias can move off the origin.

        No strategy is encoded: the agent doesn't know _comm_bias affects truth.
        The reward model determines whether distortion is punished (UHFS) or free (U).
        """
        reward_delta = current_reward - self._prev_reward_signal
        if abs(self._comm_bias) < 1e-4:
            self._comm_bias += 0.002 * float(self.rng.standard_normal())
        else:
            direction = math.copysign(1.0, reward_delta) * math.copysign(1.0, self._comm_bias)
            self._comm_bias += 0.003 * direction
            self._comm_bias = float(np.clip(self._comm_bias, -1.0, 1.0))
        self._prev_reward_signal = current_reward

    # ------------------------------------------------------------------
    # Step 7: Execute trade
    # ------------------------------------------------------------------

    def execute_trade(self, asset_idx: int, quantity: float, fill: "Fill") -> None:
        cost = fill.quantity * fill.price
        self.cash -= cost
        self.positions[asset_idx] += fill.quantity
        self.trade_history.append(fill)
        self._market_impact_total += abs(cost)

    def receive_payment(self, amount: float) -> None:
        self.cash += amount

    def pay(self, amount: float) -> bool:
        if self.cash >= amount:
            self.cash -= amount
            return True
        return False

    def lock_collateral(self, amount: float) -> bool:
        if self.cash >= amount:
            self.cash -= amount
            self.locked_collateral += amount
            return True
        return False

    def unlock_collateral(self, amount: float) -> None:
        release = min(amount, self.locked_collateral)
        self.locked_collateral -= release
        self.cash += release

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def net_worth(self, market_prices: np.ndarray) -> float:
        position_value = float(np.dot(self.positions, market_prices))
        return self.cash + position_value + self.locked_collateral

    def snapshot_wealth(self, market_prices: np.ndarray) -> float:
        nw = self.net_worth(market_prices)
        self.wealth_history.append(nw)
        return nw

    def belief_accuracy(self) -> float:
        if self._resolved_ev_total == 0:
            return 0.5
        return self._resolved_ev_correct / self._resolved_ev_total

    def epistemic_health(self, all_agents: List["Agent"]) -> float:
        accuracy = self.belief_accuracy()
        my_keys = {
            ev.proposition for ev in self.evidence_ledger
            if ev.status == EvidenceStatus.CONFIRMED
        }
        if not my_keys:
            return accuracy * 0.5
        other_correct_keys: set = set()
        for ag in all_agents:
            if ag.agent_id == self.agent_id:
                continue
            for ev in ag.evidence_ledger:
                if ev.status == EvidenceStatus.CONFIRMED:
                    other_correct_keys.add(ev.proposition)
        unique_correct = len(my_keys - other_correct_keys)
        uniqueness = unique_correct / (len(my_keys) + 1e-9)
        return accuracy * (0.7 + 0.3 * uniqueness)

    def poli_score(self, n_ticks_window: int) -> float:
        recent_wealth = self.wealth_history[-1] if self.wealth_history else self.cash
        return recent_wealth + self._market_impact_total / (n_ticks_window + 1)

    def current_reward_signal(self, market_prices: np.ndarray) -> float:
        """Expose latest reward value for metric collection."""
        from .reward import RewardModelName, compute_base_utility
        from .horizon import compute_agency_state
        if self.reward_model.name == RewardModelName.U:
            return compute_base_utility(self, market_prices)
        if self._last_agency is not None:
            u = compute_base_utility(self, market_prices)
            return self.reward_model.compute(u, self._last_agency)
        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_asset_idx(key: Optional[str]) -> Optional[int]:
        if key is None:
            return None
        parts = key.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _parse_target_tick(key: Optional[str]) -> Optional[int]:
        if key is None:
            return None
        if "_tick_" in key:
            try:
                return int(key.split("_tick_")[1])
            except (IndexError, ValueError):
                return None
        return None
