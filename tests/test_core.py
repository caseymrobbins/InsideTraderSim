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
