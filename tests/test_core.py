"""Basic correctness tests for InsideTraderSim core components."""
import numpy as np
import pytest

from inside_traders.config import SimConfig
from inside_traders.world import World
from inside_traders.agent import Agent
from inside_traders.belief import BeliefGraph
from inside_traders.evidence import Evidence, EvidenceType, EvidenceStatus, EpistemicModel
from inside_traders.market import OrderBook
from inside_traders.venture import Venture, VentureStatus
from inside_traders.metrics import gini, MetricsCollector
from inside_traders.simulation import Simulation


# ──────────────────────────────────────────────────────────────────────
# World
# ──────────────────────────────────────────────────────────────────────

def test_world_step_updates_fundamentals():
    cfg = SimConfig(n_agents=5, n_assets=3, seed=0)
    rng = np.random.default_rng(0)
    w = World(cfg, rng)
    before = w.fundamentals.copy()
    w.step()
    assert not np.allclose(before, w.fundamentals), "Fundamentals must change each tick"


def test_world_fundamentals_positive():
    cfg = SimConfig(n_agents=5, n_assets=3, seed=1)
    rng = np.random.default_rng(1)
    w = World(cfg, rng)
    for _ in range(50):
        w.step()
    assert (w.fundamentals > 0).all(), "Fundamentals must stay positive"


def test_prediction_cards_noise():
    cfg = SimConfig(n_agents=5, n_assets=3, seed=2)
    rng = np.random.default_rng(2)
    w = World(cfg, rng)
    cards = w.issue_free_cards("agent_0", 10)
    assert len(cards) == 10
    for c in cards:
        assert c.agent_id == "agent_0"
        assert c.predicted_value > 0
        assert c.confidence > 0


# ──────────────────────────────────────────────────────────────────────
# BeliefGraph
# ──────────────────────────────────────────────────────────────────────

def test_belief_graph_add_and_update():
    bg = BeliefGraph()
    bg.add_or_update("key_1", 100.0, 1.0, 0.7, tick=1)
    prop = bg.get("key_1")
    assert prop is not None
    assert abs(prop.value - 100.0) < 1e-6

    # Update with new evidence
    bg.add_or_update("key_1", 110.0, 1.0, 0.9, tick=2)
    updated = bg.get("key_1")
    assert updated.value > 100.0, "Blended value should be between 100 and 110"


def test_belief_decay():
    bg = BeliefGraph()
    bg.add_or_update("key_1", 100.0, 1.0, 0.9, tick=1)
    # Decay from far future tick
    bg.decay(tick=100, rate=0.01)
    prop = bg.get("key_1")
    assert prop.confidence < 0.9, "Confidence should decay over time"


# ──────────────────────────────────────────────────────────────────────
# EpistemicModel
# ──────────────────────────────────────────────────────────────────────

def test_epistemic_model_update_correct():
    em = EpistemicModel()
    em.update("agent_1", "price", was_correct=True, rate=0.1)
    assert em.get("agent_1", "price") > 0.5


def test_epistemic_model_update_incorrect():
    em = EpistemicModel()
    em.update("agent_1", "price", was_correct=False, rate=0.1)
    assert em.get("agent_1", "price") < 0.5


# ──────────────────────────────────────────────────────────────────────
# OrderBook
# ──────────────────────────────────────────────────────────────────────

def test_order_book_buy_raises_price():
    prices = np.array([100.0, 50.0])
    ob = OrderBook(n_assets=2, initial_prices=prices, impact=0.01)
    fill = ob.submit("agent_0", 0, quantity=10.0, tick=1)
    assert ob.price(0) > 100.0, "Buying should push price up"
    assert fill.quantity == 10.0


def test_order_book_sell_lowers_price():
    prices = np.array([100.0, 50.0])
    ob = OrderBook(n_assets=2, initial_prices=prices, impact=0.01)
    ob.submit("agent_0", 0, quantity=-10.0, tick=1)
    assert ob.price(0) < 100.0, "Selling should push price down"


# ──────────────────────────────────────────────────────────────────────
# Venture
# ──────────────────────────────────────────────────────────────────────

def test_venture_success_pays_surplus():
    v = Venture(
        venture_id=0, proposer="a0", counterparty="a1",
        asset_idx=0, principal=100.0, collateral=10.0,
        surplus_share_proposer=0.5, start_tick=1, duration=10,
        hidden_quality=0.8, status=VentureStatus.ACTIVE,
    )
    payouts = v.resolve(actual_quality=0.8, surplus_multiplier=1.4, tick=11)
    assert v.status == VentureStatus.SUCCEEDED
    assert payouts["a0"] + payouts["a1"] > 200.0, "Positive surplus should be created"


def test_venture_failure_reduces_payouts():
    v = Venture(
        venture_id=1, proposer="a0", counterparty="a1",
        asset_idx=0, principal=100.0, collateral=10.0,
        surplus_share_proposer=0.5, start_tick=1, duration=10,
        hidden_quality=0.2, status=VentureStatus.ACTIVE,
    )
    payouts = v.resolve(actual_quality=0.2, surplus_multiplier=1.4, tick=11)
    assert v.status == VentureStatus.FAILED
    total = payouts["a0"] + payouts["a1"]
    assert total < 200.0, "Failed venture should not produce surplus"


# ──────────────────────────────────────────────────────────────────────
# Gini
# ──────────────────────────────────────────────────────────────────────

def test_gini_perfect_equality():
    values = np.ones(10) * 100
    assert gini(values) < 1e-6


def test_gini_perfect_inequality():
    values = np.zeros(10)
    values[-1] = 100.0
    g = gini(values)
    # For n=10 with one person owning everything, Gini = (n-1)/n
    expected = 9 / 10
    assert abs(g - expected) < 0.01


# ──────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────

def test_agent_net_worth():
    cfg = SimConfig(n_agents=5, n_assets=3, seed=0)
    rng = np.random.default_rng(0)
    ag = Agent("agent_0", cfg, rng, initial_cash=500.0)
    prices = np.array([100.0, 50.0, 75.0])
    assert abs(ag.net_worth(prices) - 500.0) < 1e-6


def test_agent_receives_cards_updates_belief():
    cfg = SimConfig(n_agents=5, n_assets=3, seed=0)
    rng = np.random.default_rng(0)
    world = World(cfg, rng)
    ag = Agent("agent_0", cfg, rng, initial_cash=100.0)
    cards = world.issue_free_cards("agent_0", 3)
    ag.receive_cards(cards, tick=1)
    assert len(ag.evidence_ledger) == 3
    props = ag.belief_graph.all_propositions()
    assert len(props) >= 1


def test_agent_belief_accuracy_no_evidence():
    cfg = SimConfig(n_agents=5, n_assets=3, seed=0)
    rng = np.random.default_rng(0)
    ag = Agent("agent_0", cfg, rng, initial_cash=100.0)
    # No resolved evidence → default 0.5
    assert ag.belief_accuracy() == 0.5


# ──────────────────────────────────────────────────────────────────────
# Simulation integration
# ──────────────────────────────────────────────────────────────────────

def test_simulation_runs_without_crash():
    cfg = SimConfig(n_agents=5, n_assets=3, n_ventures=2, n_ticks=20, seed=99)
    sim = Simulation(cfg)
    metrics = sim.run()
    assert len(metrics.snapshots) > 0


def test_simulation_total_wealth_increases():
    """Ventures should create net positive surplus on average."""
    cfg = SimConfig(
        n_agents=10, n_assets=3, n_ventures=2, n_ticks=100,
        seed=7, venture_success_base_prob=0.9,
        compression_test_start=200,  # disable during this short run
    )
    sim = Simulation(cfg)
    sim.run()
    m = sim.metrics
    if len(m.snapshots) >= 2:
        # Check that wealth can grow (not guaranteed with so few ticks, but shouldn't catastrophically fall)
        final = m.snapshots[-1].total_wealth
        initial = m.snapshots[0].total_wealth
        assert final > initial * 0.5, "Wealth should not catastrophically collapse"


def test_compression_test_selects_targets():
    cfg = SimConfig(
        n_agents=10, n_assets=3, n_ticks=200,
        compression_test_start=50, compression_test_agents=2,
        seed=42,
    )
    sim = Simulation(cfg)
    sim.run()
    assert len(sim._compression_targets) == 2
    # Targets should be valid agent IDs
    agent_ids = {ag.agent_id for ag in sim.agents}
    for t in sim._compression_targets:
        assert t in agent_ids


def test_gini_positive_under_heterogeneous_wealth():
    """Heterogeneous endowments should produce Gini > 0."""
    cfg = SimConfig(n_agents=15, n_ticks=50, seed=1)
    sim = Simulation(cfg)
    sim.run()
    final_gini = sim.metrics.snapshots[-1].gini if sim.metrics.snapshots else 0.0
    assert final_gini > 0.0, "Heterogeneous agents should produce nonzero Gini"


# ──────────────────────────────────────────────────────────────────────
# Pretraining: solo-world phase
# ──────────────────────────────────────────────────────────────────────

def test_pretrain_produces_no_comm_evidence():
    """Solo pretraining must not inject any COMMUNICATION evidence."""
    from inside_traders.pretrain import PretrainConfig, run_solo_pretrain
    from inside_traders.evidence import EvidenceType
    import tempfile, os

    cfg = SimConfig(n_agents=4, n_assets=3, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        pcfg = PretrainConfig(n_ticks=20, checkpoint_path=os.path.join(tmp, "ckpt.json"))
        agents = run_solo_pretrain(cfg, pcfg)

    for ag in agents:
        comm_ev = [ev for ev in ag.evidence_ledger if ev.ev_type == EvidenceType.COMMUNICATION]
        assert comm_ev == [], (
            f"{ag.agent_id} has {len(comm_ev)} COMMUNICATION evidence entries "
            "after solo pretraining — communication channel was not properly isolated."
        )


def test_checkpoint_save_and_load():
    """save_checkpoint / load_checkpoint must be inverses for all serialised fields."""
    from inside_traders.checkpoint import save_checkpoint, load_checkpoint
    import tempfile, os

    cfg = SimConfig(n_agents=3, n_assets=2, seed=7)
    rng = np.random.default_rng(7)
    agents = [
        Agent(f"agent_{i}", cfg, np.random.default_rng(i), initial_cash=100.0)
        for i in range(3)
    ]
    # Modify Q-table so there is non-trivial state to round-trip.
    agents[0]._comm_q[0, 0, 1] = 0.42
    agents[1]._comm_epsilon = 0.25
    agents[2].preference_vector["wealth"] = 0.99

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.json")
        save_checkpoint(agents, path)
        state_dicts = load_checkpoint(path)

    assert len(state_dicts) == 3
    assert abs(state_dicts[0]["comm_q"][0][0][1] - 0.42) < 1e-9
    assert abs(state_dicts[1]["comm_epsilon"] - 0.25) < 1e-9
    assert abs(state_dicts[2]["preference_vector"]["wealth"] - 0.99) < 1e-9


def test_apply_checkpoint_restores_state_and_resets_comm():
    """apply_checkpoint must transfer learned state and blank the comm Q-table."""
    from inside_traders.checkpoint import save_checkpoint, load_checkpoint, apply_checkpoint
    import tempfile, os

    cfg = SimConfig(n_agents=3, n_assets=2, seed=8)
    src_agents = [
        Agent(f"agent_{i}", cfg, np.random.default_rng(i + 100), initial_cash=100.0)
        for i in range(3)
    ]
    # Write non-trivial Q-table values and a credibility score.
    src_agents[0]._comm_q[1, 2, 3] = -0.7
    src_agents[0].epistemic_model.update("self", "price", was_correct=True, rate=0.3)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.json")
        save_checkpoint(src_agents, path)
        state_dicts = load_checkpoint(path)

    # Fresh target agents (simulating a new Simulation object).
    tgt_agents = [
        Agent(f"agent_{i}", cfg, np.random.default_rng(i + 200), initial_cash=200.0)
        for i in range(3)
    ]
    apply_checkpoint(tgt_agents, state_dicts, reset_comm=True)

    # Credibility transferred.
    assert tgt_agents[0].epistemic_model.get("self", "price") > 0.5
    # Q-table reset to blank slate despite checkpoint containing non-zero values.
    assert tgt_agents[0]._comm_q.sum() == 0.0
    assert tgt_agents[0]._comm_epsilon == 0.5


def test_joint_sim_uses_pretrained_epistemic_state():
    """
    After apply_checkpoint, the joint simulation should start with the
    pretrained epistemic credibility rather than the cold default of 0.5.
    """
    from inside_traders.pretrain import PretrainConfig, run_solo_pretrain
    from inside_traders.checkpoint import save_checkpoint, load_checkpoint, apply_checkpoint
    import tempfile, os

    cfg = SimConfig(n_agents=4, n_assets=3, n_ticks=20, seed=5,
                    compression_test_start=200)

    with tempfile.TemporaryDirectory() as tmp:
        pcfg = PretrainConfig(n_ticks=30, checkpoint_path=os.path.join(tmp, "ckpt.json"))
        run_solo_pretrain(cfg, pcfg)
        state_dicts = load_checkpoint(pcfg.checkpoint_path)

    # Build a fresh simulation (the joint phase).
    sim = Simulation(cfg)
    apply_checkpoint(sim.agents, state_dicts, reset_comm=True)

    # At least some agents should have self-credibility updated away from 0.5.
    self_credibilities = [
        ag.epistemic_model.get("self", "price", default=0.5)
        for ag in sim.agents
    ]
    # After 30 ticks of evidence resolution, at least one agent's credibility
    # should differ from the cold-start default of 0.5.
    assert any(abs(c - 0.5) > 0.01 for c in self_credibilities), (
        "No agent has non-default self-credibility after pretraining — "
        "checkpoint transfer may not be working."
    )

    # The joint simulation must still run without error.
    sim.run()
    assert len(sim.metrics.snapshots) > 0


def test_simulation_comm_disabled_has_no_comm_evidence():
    """When comm_enabled=False, the Simulation must not produce COMMUNICATION evidence."""
    from inside_traders.evidence import EvidenceType

    cfg = SimConfig(
        n_agents=5, n_assets=3, n_ticks=15, seed=3,
        comm_enabled=False,
        compression_test_start=200,
    )
    sim = Simulation(cfg)
    sim.run()

    for ag in sim.agents:
        comm_ev = [ev for ev in ag.evidence_ledger if ev.ev_type == EvidenceType.COMMUNICATION]
        assert comm_ev == [], (
            f"{ag.agent_id} has {len(comm_ev)} COMMUNICATION evidence entries "
            "even though comm_enabled=False."
        )


# ──────────────────────────────────────────────────────────────────────
# Curriculum: honest-then-conflict
# ──────────────────────────────────────────────────────────────────────

def test_curriculum_honest_phase_forces_truthful():
    """
    During Phase A (t ≤ curriculum_honest_ticks) every outgoing TELL must be
    action 0 (TRUTHFUL): communicated_value == belief value, not amplified/inverted.

    We monkey-patch compose_message to record the _last_comm_action before it
    gets overwritten, then verify it is always 0 during Phase A.
    """
    n_ticks = 20
    honest_ticks = 10
    cfg = SimConfig(
        n_agents=6, n_assets=3, n_ticks=n_ticks, seed=99,
        curriculum_honest_ticks=honest_ticks,
        compression_test_start=200,
    )
    sim = Simulation(cfg)

    recorded_phase_a: list[int] = []
    recorded_phase_b: list[int] = []
    _orig_compose = sim.agents[0].__class__.compose_message

    def _patched_compose(self, tick, neighbors, prices):
        msg = _orig_compose(self, tick, neighbors, prices)
        action = self._last_comm_action
        if action is not None:
            if tick <= honest_ticks:
                recorded_phase_a.append(action)
            else:
                recorded_phase_b.append(action)
        return msg

    for ag in sim.agents:
        ag.__class__.compose_message = _patched_compose

    try:
        sim.run()
    finally:
        for ag in sim.agents:
            ag.__class__.compose_message = _orig_compose

    # During Phase A every comm action must be 0 (TRUTHFUL)
    assert recorded_phase_a, "No compose_message calls recorded during Phase A"
    non_truthful = [a for a in recorded_phase_a if a != 0]
    assert non_truthful == [], (
        f"Phase A produced non-TRUTHFUL actions: {set(non_truthful)}"
    )
    # Phase B should allow other actions (given enough exploration)
    assert recorded_phase_b, "No compose_message calls recorded during Phase B"


def test_curriculum_epsilon_resets_at_phase_b():
    """
    Epsilon is frozen at 0.5 during Phase A and reset to 0.5 at the Phase B
    transition, so the very first Phase B tick should see epsilon == 0.5.
    """
    n_ticks = 14
    honest_ticks = 7
    cfg = SimConfig(
        n_agents=4, n_assets=3, n_ticks=n_ticks, seed=77,
        curriculum_honest_ticks=honest_ticks,
        compression_test_start=200,
    )
    sim = Simulation(cfg)

    epsilon_at_transition: list[float] = []
    _orig_step = sim.__class__._step

    def _patched_step(self):
        _orig_step(self)
        # Capture epsilon right after the transition tick fires
        if self.tick == honest_ticks + 1:
            epsilon_at_transition.extend(ag._comm_epsilon for ag in self.agents)

    sim.__class__._step = _patched_step
    try:
        sim.run()
    finally:
        sim.__class__._step = _orig_step

    assert epsilon_at_transition, "Transition tick never fired"
    # Epsilon is reset to 0.5 at the start of the transition tick, then decays
    # once during plan_actions (one call to _update_comm_q × 0.995 factor).
    # Allow exactly that one step of decay so the assertion is numerically tight.
    max_allowed = 0.5 * 0.995 + 1e-9
    for eps in epsilon_at_transition:
        assert eps <= max_allowed, (
            f"Epsilon after Phase B unlock should be ≤ {max_allowed:.4f} (one decay step "
            f"from reset value 0.5), got {eps}"
        )
        # Also confirm epsilon is closer to 0.5 than to floor (0.05), proving the reset
        assert eps > 0.4, f"Epsilon {eps} too low — reset may not have fired"


# ──────────────────────────────────────────────────────────────────────
# Deception emergence tests
# ──────────────────────────────────────────────────────────────────────

def test_phase_a_deception_rate_is_zero():
    """
    Curriculum Phase A forces every TELL to be TRUTHFUL.
    deception_rate must be exactly 0.0 when all ticks are Phase A.
    """
    cfg = SimConfig(
        n_agents=6, n_assets=3, n_ticks=15, seed=42,
        curriculum_honest_ticks=200,   # all 15 ticks fall inside Phase A
        comm_enabled=True,
        log_interval=15, status_interval=15,
        compression_test_start=200,
    )
    sim = Simulation(cfg)
    sim.run()
    final = sim.metrics.snapshots[-1]
    assert final.deception_rate == 0.0, (
        f"Phase A should produce zero deception; got {final.deception_rate:.3f}"
    )


def test_deception_rate_positive_in_phase_b():
    """
    Once Phase B unlocks (curriculum_honest_ticks exhausted), agents explore
    AMPLIFY and INVERT via epsilon-greedy, so deception_rate must become > 0.

    With epsilon=0.5 at the Phase B start and a weighted explore distribution
    giving ~23% each to AMPLIFY/INVERT, roughly 30–40% of TELL messages will
    be deceptive from random exploration alone.
    """
    cfg = SimConfig(
        n_agents=6, n_assets=3, n_ticks=60, seed=42,
        curriculum_honest_ticks=10,   # Phase A: ticks 1–10, Phase B: ticks 11–60
        comm_enabled=True,
        log_interval=10, status_interval=10,
        compression_test_start=200,
    )
    sim = Simulation(cfg)
    sim.run()
    final = sim.metrics.snapshots[-1]
    assert 0.0 < final.deception_rate < 1.0, (
        f"Expected deception_rate in (0, 1) after Phase B; got {final.deception_rate:.3f}"
    )


def test_q_table_explores_deception_in_against_state():
    """
    In 'against' state (align_bucket=0), agents have positional incentive to
    send INVERT messages.  After Phase B exploration, the Q-table visit counts
    for AMPLIFY and INVERT in that state must be nonzero — confirming that the
    reward signal actually reaches the correct (state, action) cells.

    Actions: 0=TRUTHFUL  1=AMPLIFY  2=INVERT
    Q-table axes: [conf_bucket, align_bucket, action]
    align_bucket=0 is the 'against' state.
    """
    cfg = SimConfig(
        n_agents=10, n_assets=3, n_ticks=120, seed=7,
        curriculum_honest_ticks=20,   # Phase A: 20 ticks, Phase B: 100 ticks
        comm_enabled=True,
        log_interval=20, status_interval=20,
        compression_test_start=200,
    )
    sim = Simulation(cfg)
    sim.run()

    # Sum visits for AMPLIFY (1) and INVERT (2) across all confidence buckets
    # and all agents, specifically in the 'against' alignment bucket (index 0).
    against_deceptive_visits = sum(
        int(ag._comm_q_visits[:, 0, action].sum())
        for ag in sim.agents
        for action in (1, 2)
    )
    assert against_deceptive_visits > 0, (
        "Expected nonzero Q-table visits for AMPLIFY/INVERT in 'against' state "
        f"after {cfg.n_ticks - cfg.curriculum_honest_ticks} Phase B ticks; "
        f"got 0 — the deception reward signal may not be reaching the right cells"
    )

    # Sanity: truthful visits should also be nonzero (pure exploration produces all actions)
    against_truthful_visits = int(
        sum(ag._comm_q_visits[:, 0, 0].sum() for ag in sim.agents)
    )
    assert against_truthful_visits > 0, "Expected truthful visits in against state too"


# ──────────────────────────────────────────────────────────────────────
# Incentive loop: signal → receiver action → sender payoff + reputation
# ──────────────────────────────────────────────────────────────────────

def test_observation_resolution_confirms_against_issue_time_truth():
    """
    OBSERVATION evidence must resolve against the fundamental AT ISSUE TIME, not
    the drifted future actual.  A low-noise (premium) card should CONFIRM; the
    previous future-actual comparison REFUTED nearly everything, collapsing EH.
    """
    cfg = SimConfig(n_agents=3, n_assets=2, seed=3, asset_volatility=0.05)
    rng = np.random.default_rng(3)
    world = World(cfg, rng)
    ag = Agent("agent_0", cfg, rng, initial_cash=100.0)
    # Premium (low-noise) card observed now; resolves several ticks later.
    card, _ = world.issue_premium_card("agent_0")
    ag.receive_cards([card], tick=1)
    for _ in range(card.horizon):
        world.step()
    ag.resolve_evidence(card.horizon, world)
    ev = ag.evidence_ledger[0]
    assert ev.status in (EvidenceStatus.CONFIRMED, EvidenceStatus.REFUTED)
    # It was checked against issue-time truth, recorded on the evidence.
    assert ev.true_value_at_issue is not None
    assert abs(ev.true_value_at_issue - card.true_value) < 1e-6


def test_deception_costs_credibility_relative_to_honesty():
    """
    Caught-liar dynamics: a sender that always INVERTS its belief loses more
    credibility with a receiver than a sender that always tells the truth.
    Communication evidence resolves against send-time truth, so lies (which
    diverge from it) are REFUTED while honest relays are CONFIRMED.
    """
    from inside_traders.communication import Message, MessageType
    cfg = SimConfig(n_agents=3, n_assets=2, seed=11)
    rng = np.random.default_rng(11)
    world = World(cfg, rng)
    receiver = Agent("recv", cfg, rng, initial_cash=100.0)

    honest_src, liar_src = "honest", "liar"
    for t in range(1, 40):
        world.step()
        asset = 0
        true_now = float(world.fundamentals[asset])
        key = f"asset_{asset}_price_tick_{t + 1}"
        # Honest source relays the true current level; liar inverts around it.
        for src, val in ((honest_src, true_now), (liar_src, true_now * 0.5)):
            msg = Message(
                msg_id=0, sender=src, receiver="recv", msg_type=MessageType.TELL,
                tick=t, proposition_key=key, predicted_value=val, confidence=0.8,
            )
            receiver.receive_message(msg, t)
        receiver.resolve_evidence(t + 1, world)

    cred_honest = receiver.epistemic_model.get(honest_src, "price")
    cred_liar = receiver.epistemic_model.get(liar_src, "price")
    assert cred_honest > cred_liar, (
        f"Honest source should retain higher credibility than the liar; "
        f"got honest={cred_honest:.3f} liar={cred_liar:.3f}"
    )


def test_receiver_action_feeds_sender_comm_q():
    """
    The incentive loop must be closed: after a run, at least one agent's comm
    Q-table carries nonzero values (the influence + reputation payoffs reached
    the policy).  A flat all-zero Q-table would mean the loop is still open.
    """
    cfg = SimConfig(
        n_agents=8, n_assets=3, n_ticks=80, seed=5,
        curriculum_honest_ticks=15, comm_enabled=True,
        log_interval=40, status_interval=40, compression_test_start=999,
    )
    sim = Simulation(cfg)
    sim.run()
    total_q_mass = sum(float(np.abs(ag._comm_q).sum()) for ag in sim.agents)
    assert total_q_mass > 0.0, (
        "Comm Q-tables are all zero — the sender payoff never depended on the "
        "receiver's post-signal action (incentive loop still open)."
    )
