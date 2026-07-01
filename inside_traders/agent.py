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

        # --- Communication Q-table (learned, no strategy pre-coded) ---
        # State: (belief_confidence_bucket ∈ {0,1,2}) × (position_alignment ∈ {0,1,2})
        #   confidence:  0=low(<0.3)  1=mid  2=high(≥0.7)
        #   alignment:   0=against(position opposes belief, incentive to distort)
        #                1=neutral(no significant position)
        #                2=with(position aligns with belief)
        # Actions: 0=TRUTHFUL  1=AMPLIFY  2=INVERT  3=SILENT  4=OFFER(sell the info)
        # Agents do not know what any action means — they discover it through reward.
        self._comm_q: np.ndarray = np.zeros((3, 3, 5))
        self._comm_q_visits: np.ndarray = np.zeros((3, 3, 5), dtype=np.int64)
        self._comm_epsilon: float = 0.5        # exploration rate, decays each tick
        self._comm_honest_phase: bool = False   # set by Simulation during curriculum Phase A
        self._prev_reward_signal: float = 0.0
        self._last_comm_state: Optional[Tuple[int, int]] = None
        self._last_comm_action: Optional[int] = None
        # Delayed Q-updates: key=deadline_tick, value=(s0, s1, a, baseline_reward)
        # Applied 2 ticks after the comm action so downstream price effects are captured.
        self._pending_q_updates: Dict[int, Tuple[int, int, int, float]] = {}
        # Outgoing tell honesty tracking for outgoing_accuracy → HI_sus feedback
        # (set at send time in compose_message: honest = communicated ≈ belief).
        self._outgoing_confirmed: int = 0
        self._outgoing_total: int = 0

        # Influence tracking: records the directional TELL sent this tick so
        # the simulation can compute the receiver's trade response and inject
        # it as a direct incentive bonus into the pending Q-update.  Closed
        # per-tick; simulation writes to _pending_influence, _update_comm_q
        # consumes it.
        self._pending_influence: Dict[int, float] = {}   # deadline_tick → bonus
        self._last_comm_receiver: Optional[str] = None
        self._last_comm_asset: Optional[int] = None
        self._last_comm_position: float = 0.0  # sender's position at comm time

        # --- Metrics tracking ---
        self.trade_history: List[Fill] = []
        self.venture_ids: List[int] = []
        self.wealth_history: List[float] = [initial_cash]
        self.poli_history: List[float] = []
        self.eh_history: List[float] = []
        self._market_impact_total: float = 0.0
        self._resolved_ev_correct: int = 0
        self._resolved_ev_total: int = 0
        # Rolling window of recent OBSERVATION resolution outcomes (bool).
        # belief_accuracy() averages over this window (size = cfg.eh_accuracy_window)
        # rather than a lifetime cumulative ratio, so EH can MOVE over a run as
        # agents shift toward higher-quality (premium) cards — a lifetime average
        # is structurally flat once enough samples accumulate.
        self._recent_resolutions: List[bool] = []

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
                true_value_at_issue=card.true_value,
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
        # Clear influence-tracking fields; only re-set if a directional TELL fires.
        self._last_comm_receiver = None
        self._last_comm_asset = None
        self._last_comm_position = 0.0
        if not neighbors:
            return None

        receiver = self.rng.choice(neighbors)
        best_prop = self._best_proposition_to_share(tick)

        if best_prop is None:
            asset_idx = int(self.rng.integers(0, self.cfg.n_assets))
            return self._make_ask(receiver, asset_idx, tick)

        key, value, confidence = best_prop
        asset_idx = self._parse_asset_idx(key)
        target_tick = self._parse_target_tick(key)
        current_price = market_prices[asset_idx] if asset_idx is not None else value
        expected_delta = value - current_price

        # ── Encode state ─────────────────────────────────────────────────
        conf_bucket = 0 if confidence < 0.3 else (2 if confidence >= 0.7 else 1)
        pos = float(self.positions[asset_idx]) if asset_idx is not None else 0.0
        if abs(pos) < 1.0 or abs(expected_delta) < 1e-6:
            align_bucket = 1  # neutral: no meaningful position
        elif (pos > 0 and expected_delta > 0) or (pos < 0 and expected_delta < 0):
            align_bucket = 2  # position agrees with belief
        else:
            align_bucket = 0  # position opposes belief (incentive to distort)

        state = (conf_bucket, align_bucket)
        self._last_comm_state = state

        # ── Epsilon-greedy action selection ───────────────────────────────
        # 0=TRUTHFUL  1=AMPLIFY  2=INVERT  3=SILENT  4=OFFER
        # No action has a label — the agent discovers what each one does.
        #
        # Exploration uses a weighted draw so OFFER (action 4) gets a lower
        # share of random trials than AMPLIFY/INVERT, preventing OFFER from
        # cannibalising deception-action exploration.
        # Remaining probability distributed equally over {0,1,2,3}.
        _p_offer = self.cfg.offer_explore_prob
        _p_other = (1.0 - _p_offer) / 4.0
        _explore_probs = [_p_other, _p_other, _p_other, _p_other, _p_offer]
        if self._comm_honest_phase:
            # Curriculum Phase A: only TRUTHFUL allowed; epsilon frozen
            action = 0
        elif self.rng.random() < self._comm_epsilon:
            action = int(self.rng.choice(5, p=_explore_probs))
        else:
            action = int(np.argmax(self._comm_q[state[0], state[1]]))
        self._last_comm_action = action

        # ── Execute action ────────────────────────────────────────────────
        if action == 3:  # SILENT: gather rather than broadcast
            return self._make_ask(receiver, int(self.rng.integers(0, self.cfg.n_assets)), tick)

        if action == 4:  # OFFER: sell the info commercially
            price = self.cfg.signal_base_cost * (1.0 + confidence)
            msg = Message(
                msg_id=self._msg_counter,
                sender=self.agent_id,
                receiver=receiver,
                msg_type=MessageType.OFFER,
                tick=tick,
                proposition_key=key,
                predicted_value=value,
                confidence=confidence,
                price=price,
            )
            self._msg_counter += 1
            self._pending_offers[msg.msg_id] = msg
            return msg

        # TELL actions: 0=TRUTHFUL  1=AMPLIFY  2=INVERT
        # AMPLIFY and INVERT boost the communicated confidence so the false value
        # lands with enough weight to shift the receiver's belief graph and
        # overcome their trading threshold.  TRUTHFUL sends the real confidence.
        if action == 0:
            communicated_value = value
            communicated_confidence = confidence
        elif action == 1:
            communicated_value = current_price + 2.0 * expected_delta
            communicated_confidence = min(0.95, confidence + self.cfg.amplify_conf_boost)
        else:  # action == 2
            communicated_value = current_price - expected_delta
            communicated_confidence = min(0.95, confidence + self.cfg.invert_conf_boost)

        # Outgoing honesty (send-time, drift-free): does the communicated value
        # match the agent's own belief value?  Truthful sends value≈belief;
        # AMPLIFY/INVERT diverge.  Feeds outgoing_accuracy → HI_sus.
        self._outgoing_total += 1
        if abs(communicated_value - value) / (abs(value) + 1e-9) < 0.10:
            self._outgoing_confirmed += 1

        # Record TELL metadata so simulation can compute the influence bonus:
        # receiver's resulting trade in this asset × sender's position = PnL benefit.
        if asset_idx is not None:
            self._last_comm_receiver = receiver
            self._last_comm_asset = asset_idx
            self._last_comm_position = float(self.positions[asset_idx])

        msg = Message(
            msg_id=self._msg_counter,
            sender=self.agent_id,
            receiver=receiver,
            msg_type=MessageType.TELL,
            tick=tick,
            proposition_key=key,
            predicted_value=communicated_value,
            confidence=communicated_confidence,
            price=None,
        )
        self._msg_counter += 1
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

    def resolve_evidence(self, tick: int, world: "World") -> None:
        world_fundamentals = world.fundamentals
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
            # Resolution measures whether the claim was accurate ABOUT THE STATE
            # AT THE TIME IT WAS MADE, not whether it forecast future drift.  This
            # is essential: world drift (≈3%/tick over multi-tick horizons) swamps
            # the 10% threshold, so comparing against the future actual REFUTES
            # every claim — honest or not — collapsing EH and giving the reputation
            # system no way to separate liars from honest-but-drifted senders.
            #   OBSERVATION: compare to the card's issue-time fundamental (exact,
            #     stored on the evidence) → measures card signal quality (→ EH).
            #   COMMUNICATION: compare to the fundamental at the sender's send tick
            #     (from world history) → measures the sender's HONESTY, so truthful
            #     tells build credibility and only genuine lies (AMPLIFY/INVERT,
            #     which diverge from send-time truth) lose it (→ reputation loop).
            if ev.ev_type == EvidenceType.OBSERVATION and ev.true_value_at_issue is not None:
                check_against = ev.true_value_at_issue
            elif ev.ev_type == EvidenceType.COMMUNICATION:
                check_against = world.fundamental_at(ev.target_asset, ev.timestamp)
            else:
                check_against = actual
            ev.error = abs(ev.predicted_value - check_against)
            is_correct = ev.error / (abs(check_against) + 1e-9) < 0.10
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
            # OBSERVATION outcomes feed the windowed accuracy that drives EH.
            if ev.ev_type == EvidenceType.OBSERVATION:
                self._recent_resolutions.append(is_correct)
                win = max(self.cfg.eh_accuracy_window, 1)
                if len(self._recent_resolutions) > win:
                    self._recent_resolutions = self._recent_resolutions[-win:]

        # NOTE: outgoing-tell reliability is now tracked at SEND time in
        # compose_message (honesty = communicated value vs own belief), not here.
        # The previous resolution-against-future-actual measure was corrupted by
        # world drift (≈3%/tick over multi-tick horizons), so honest and
        # deceptive senders alike appeared "inaccurate" — giving HI_sus no way to
        # distinguish liars and leaving the UHFS objective unable to touch the
        # comm policy.  Send-time honesty is drift-free and cleanly separates them.
        self.belief_graph.decay(tick, self.cfg.belief_decay)

    def outgoing_accuracy(self) -> float:
        """Fraction of sent TELL messages that were HONEST (matched own belief).

        Truth is defined by comparison to the sender's private belief, per the
        incentive-loop spec — not a flag on the message.  Feeds outgoing_rel in
        HI_sus so a sustainability-weighted objective (UHFS) pays a cost for
        lying that a plain-utility objective (U) does not.
        """
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

            if abs(modulated_utility) < self.cfg.min_trade_utility:
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
        # Epistemic homeostasis: seek accurate (premium, low-noise) information
        # more when recent belief accuracy is poor, so an agent that starts on
        # noisy free cards shifts toward premium cards and its windowed
        # belief_accuracy (→ EH) recovers rather than staying pinned at the
        # free-card noise floor.  Modest effect (premium cards cost cash), but it
        # keeps EH responsive to information quality instead of a dead constant.
        eh_boost = 1.0 + (1.0 - self.belief_accuracy())
        signal_utility = (w["knowledge"] * 2.0 + w["wealth"] * 0.5) * info_scale * eh_boost
        if (signal_utility > 1.0 and self.cash > self.cfg.signal_base_cost * 2
                and self.rng.random() < min(0.6, 0.15 * info_scale * eh_boost)):
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
            self._update_comm_q(utility, tick)
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
        self._update_comm_q(reward, tick)

        if agency.in_danger_zone:
            # ── Danger zone ────────────────────────────────────────────
            # Determine which dimension is weakest → prioritise restoring it
            ia_vec = agency.ia_vector
            weakest = int(np.argmin(ia_vec))
            # [0]=liquidity [1]=epistemic [2]=network [3]=solvency [4]=options
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

    _COMM_Q_DELAY: int = 2  # ticks to wait before applying a comm Q-update

    def _update_comm_q(self, current_reward: float, tick: int = 0) -> None:
        """
        Comm-policy update: adjust weights on (situation, action) pairs.

        The comm policy learns from the ATTRIBUTABLE consequences of the signal:
          - influence: how the receiver's induced trade moved price in the
            sender's position direction (immediate, applied here at t+DELAY);
          - reputation: how the receiver's credibility toward the sender moved
            once the claim resolved (delayed, applied via record_comm_reputation).
        The noisy global reward (trading PnL, drift, ventures across a 30-agent
        market) is deliberately NOT fed into the comm delta — it buries the
        single-agent comm effect and, for volatile objectives like UHFS, injects
        variance that corrupts the linkage.  The OBJECTIVE still reaches the comm
        policy: reputation_sensitivity() scales the reputation payoff per model,
        so a sustainability-weighted objective lies less than plain utility.
        """
        # Apply any Q-update that is due this tick
        pending = self._pending_q_updates.pop(tick, None)
        if pending is not None:
            s0, s1, a = pending
            influence = self._pending_influence.pop(tick, 0.0)
            self._comm_q[s0, s1, a] += 0.1 * (influence - self._comm_q[s0, s1, a])
            self._comm_q_visits[s0, s1, a] += 1

        # Schedule Q-update for the action taken THIS tick (fires in DELAY ticks)
        if self._last_comm_state is not None and self._last_comm_action is not None:
            s0, s1 = self._last_comm_state
            a = self._last_comm_action
            deadline = tick + self._COMM_Q_DELAY
            self._pending_q_updates[deadline] = (s0, s1, a)

        # Decay mutation rate only during Phase B (free conflict); frozen in Phase A
        if not self._comm_honest_phase:
            self._comm_epsilon = max(0.05, self._comm_epsilon * 0.995)
        self._prev_reward_signal = current_reward

    def record_comm_reputation(self, state: Tuple[int, int], action: int, signal: float) -> None:
        """
        Apply a realized REPUTATION payoff to a comm (state, action) cell.

        Driven by the change in the receiver's credibility toward this sender
        after the sender's earlier claim resolved (simulation computes Δcred and
        calls this).  Honest claims that resolve confirmed raise credibility
        (positive signal); deceptive claims that resolve refuted lower it
        (negative signal), discounting the sender's future influence.

        This is the tension the incentive loop needs: an immediate influence gain
        from lying is offset by a delayed reputation loss, so honesty is sometimes
        optimal (preserve credibility) and lying sometimes optimal (cash in now).
        Additive nudge (not a TD blend) because reputation is a payoff increment
        on that action, layered on top of the influence-driven value.
        """
        s0, s1 = state
        self._comm_q[s0, s1, action] += signal
        self._comm_q_visits[s0, s1, action] += 1

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
        # Windowed over recent OBSERVATION resolutions so EH can move as signal
        # quality changes over the run.  Falls back to lifetime ratio only if the
        # window is empty (e.g. very early ticks before any observation resolves).
        if self._recent_resolutions:
            return sum(self._recent_resolutions) / len(self._recent_resolutions)
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
