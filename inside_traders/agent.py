"""HorizonSim v1 Agent: full cognitive architecture."""
from __future__ import annotations
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


_EV_COUNTER = 0


def _next_ev_id() -> int:
    global _EV_COUNTER
    _EV_COUNTER += 1
    return _EV_COUNTER


class Agent:
    """
    HorizonSim v1 cognitive agent.

    Internal state:
      - preference_vector: fixed utility weights
      - belief_graph: dynamic proposition network
      - evidence_ledger: immutable append log
      - epistemic_model: per-source credibility
      - address_book: known peers (managed by CommunicationGraph)
      - positions: per-asset holdings
      - cash: current numeraire balance

    Strategy is NOT hard-coded. All complex behaviour emerges from
    utility maximisation over beliefs and preferences.
    """

    def __init__(
        self,
        agent_id: str,
        cfg: SimConfig,
        rng: np.random.Generator,
        initial_cash: float,
    ) -> None:
        self.agent_id = agent_id
        self.cfg = cfg
        self.rng = rng

        # --- Economic state ---
        self.cash: float = initial_cash
        self.positions: np.ndarray = np.zeros(cfg.n_assets)
        self.locked_collateral: float = 0.0   # numeraire locked in ventures

        # --- Cognitive architecture ---
        self.preference_vector: Dict[str, float] = self._init_preferences()
        self.belief_graph = BeliefGraph()
        self.evidence_ledger: List[Evidence] = []
        self.epistemic_model = EpistemicModel()

        # --- Communication ---
        self._outbox: List[Message] = []
        self._pending_offers: Dict[int, Message] = {}   # offer_id → offer message
        self._msg_counter = 0

        # --- Metrics tracking ---
        self.trade_history: List[Fill] = []
        self.venture_ids: List[int] = []
        self.wealth_history: List[float] = [initial_cash]
        self.poli_history: List[float] = []
        self.eh_history: List[float] = []
        self._market_impact_total: float = 0.0   # |$| traded last window
        self._resolved_ev_correct: int = 0
        self._resolved_ev_total: int = 0

        # --- Intervention view (optional extension surface) ---
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
            # Immediately update belief graph from observation
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
        if not neighbors:
            return None

        receiver = self.rng.choice(neighbors)
        best_prop = self._best_proposition_to_share(tick)

        if best_prop is None:
            # No useful belief to share — send an ASK instead
            asset_idx = int(self.rng.integers(0, self.cfg.n_assets))
            return self._make_ask(receiver, asset_idx, tick)

        key, value, confidence = best_prop
        # Lying emerges naturally: when utility of misleading > sharing truth
        # Here agents send the proposition they actually believe (or distort when
        # it serves them — that distortion is not explicit but can emerge via
        # belief noise and gain calculations)
        utility_share = self._utility_of_sharing(key, confidence)
        utility_offer = self._utility_of_offering(key, confidence)

        if utility_offer > utility_share and self.cash > self.cfg.signal_base_cost:
            msg_type = MessageType.OFFER
            price = self.cfg.signal_base_cost * (1.0 + confidence)
        else:
            msg_type = MessageType.TELL
            price = None

        msg = Message(
            msg_id=self._msg_counter,
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            tick=tick,
            proposition_key=key,
            predicted_value=value,
            confidence=confidence,
            price=price,
        )
        self._msg_counter += 1
        if msg_type == MessageType.OFFER:
            self._pending_offers[msg.msg_id] = msg
        return msg

    def _best_proposition_to_share(self, tick: int) -> Optional[Tuple[str, float, float]]:
        """Find the proposition the agent has highest-confidence belief on."""
        best = None
        for prop in self.belief_graph.all_propositions():
            # Only future-resolving beliefs are useful to share
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
        return (w["wealth"] * confidence * self.cfg.signal_base_cost
                + w["influence"] * 0.2)

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
        """
        Process an incoming message.
        Returns a reply message if applicable (ACCEPT/REJECT/TELL for ASK).
        """
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
            # Expand address book — handled externally by CommunicationGraph
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
        """Decide whether to ACCEPT or REJECT an information offer."""
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
        """Counterparty accepted our offer — reveal the info, receive payment."""
        offer = self._pending_offers.get(msg.offer_id)
        if offer is None:
            return
        price = offer.price or 0.0
        self.cash += price
        # The actual info transfer happens at the simulation level

    def _answer_ask(self, msg: Message, tick: int) -> Optional[Message]:
        """Reply to an ASK with our best belief, free of charge."""
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
        """
        Check pending evidence whose target_tick <= current tick.
        Update EpistemicModel and BeliefGraph based on outcomes.
        """
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
            # "Correct" if within 10% of actual
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

        # Decay stale beliefs
        self.belief_graph.decay(tick, self.cfg.belief_decay)

    # ------------------------------------------------------------------
    # Step 5: Update Intervention view (hook for future extensions)
    # ------------------------------------------------------------------

    def update_intervention_view(self, tick: int, market_prices: np.ndarray) -> None:
        """
        Record how recent actions affected agent's option space.
        This is the extension point for Agency Calculus / Horizon Index overlays.
        """
        portfolio_value = self.net_worth(market_prices)
        self.intervention_log.append({
            "tick": tick,
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "n_beliefs": len(self.belief_graph.all_propositions()),
        })
        if len(self.intervention_log) > 50:
            self.intervention_log.pop(0)

    # ------------------------------------------------------------------
    # Step 6: Plan
    # ------------------------------------------------------------------

    def plan_actions(
        self,
        tick: int,
        market_prices: np.ndarray,
        active_ventures: List["Venture"],
    ) -> List[Dict]:
        """
        Return a list of action dicts the agent intends to execute.
        Actions: {type: "trade"|"venture_propose"|"buy_info", ...}
        """
        actions = []
        w = self.preference_vector

        # --- Trading decisions ---
        for asset_idx in range(self.cfg.n_assets):
            belief = self.belief_graph.highest_confidence_price(asset_idx, tick)
            if belief is None:
                continue
            target_tick, predicted_price, confidence = belief

            current_price = market_prices[asset_idx]
            expected_return = (predicted_price - current_price) / (current_price + 1e-9)
            # Risk-adjust by confidence
            risk_adj_return = expected_return * confidence
            utility = risk_adj_return * w["wealth"] - abs(risk_adj_return) * w["security"] * 0.5

            if abs(utility) < 0.005:
                continue

            # Size proportional to confidence and available cash, capped at max_position
            desired_position = self.cfg.max_position * confidence * np.sign(utility)
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

        # --- Venture proposals ---
        if (w["wealth"] > 0.3 and self.cash > 20.0
                and self.rng.random() < 0.03 * w["knowledge"]):
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

        # --- Buy premium signal ---
        signal_utility = w["knowledge"] * 2.0 + w["wealth"] * 0.5
        if (signal_utility > 1.0 and self.cash > self.cfg.signal_base_cost * 2
                and self.rng.random() < 0.15):
            actions.append({"type": "buy_signal", "premium": True})

        return actions

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
        """Fraction of resolved evidence that was correct."""
        if self._resolved_ev_total == 0:
            return 0.5
        return self._resolved_ev_correct / self._resolved_ev_total

    def epistemic_health(self, all_agents: List["Agent"]) -> float:
        """
        EH = accuracy × uniqueness proxy.
        Uniqueness: how many correct beliefs this agent has that others don't.
        """
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
        """
        POLI = wealth + market influence + network reach.
        Simplified to wealth-normalised + recent trade volume.
        """
        recent_wealth = self.wealth_history[-1] if self.wealth_history else self.cash
        return recent_wealth + self._market_impact_total / (n_ticks_window + 1)

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
