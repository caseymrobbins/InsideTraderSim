"""Solo-world pretraining: one agent at a time, no communication channel.

Pretraining constraints (enforced here, not in Simulation):
  * One agent per isolated world — no other agents present.
  * World observations only — prediction cards from world, no peer messages.
  * World-action levers only — trade and buy_signal; venture_propose silently skipped
    (ventures require a counterparty).
  * Individual return only — reward model replaced with plain-utility U for the
    solo phase so the agent's competence signal is its own P&L.
  * No signal/communication channel — compose_message and receive_message never called.

After all agents have been pretrained, a checkpoint is written to
``pretrain_cfg.checkpoint_path``.  The joint phase loads this checkpoint,
optionally resetting the comm Q-table to a blank slate so that
communication strategy emerges from scratch in the multi-agent environment.

Example::

    from inside_traders.config import SimConfig
    from inside_traders.pretrain import PretrainConfig, run_solo_pretrain

    pretrain_cfg = PretrainConfig(n_ticks=150, checkpoint_path="checkpoints/pretrained.json")
    agents = run_solo_pretrain(SimConfig(n_agents=30), pretrain_cfg)
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .agent import Agent
from .config import SimConfig
from .market import OrderBook
from .reward import RewardU, RewardModelName
from .world import World
from .checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class PretrainConfig:
    """Configuration for the solo pretraining phase."""
    n_ticks: int = 100
    checkpoint_path: str = "checkpoints/pretrained.json"
    plot_dir: str = "plots"
    verbose: bool = False


def run_solo_pretrain(
    sim_cfg: SimConfig,
    pretrain_cfg: PretrainConfig,
    agents: Optional[List[Agent]] = None,
) -> List[Agent]:
    """Run solo-world pretraining for every agent independently.

    Each agent is trained in an isolated loop (its own World and OrderBook)
    so that no inter-agent effects are present.  Returns the trained agent
    list and saves a checkpoint to ``pretrain_cfg.checkpoint_path``.

    Parameters
    ----------
    sim_cfg:
        The same SimConfig used for the subsequent joint simulation.
    pretrain_cfg:
        Pretraining-specific settings (ticks, checkpoint path).
    agents:
        Pre-built agent list.  When ``None``, agents are constructed using
        the same parameters that :class:`~inside_traders.simulation.Simulation`
        would use, ensuring compatibility with the joint phase.
    """
    t_start = time.perf_counter()

    if agents is None:
        agents = _build_agents(sim_cfg)

    n = len(agents)
    n_ticks = pretrain_cfg.n_ticks
    verbose = pretrain_cfg.verbose

    print(f"\n{'='*60}")
    print(f"  Pretraining  —  {n} agents × {n_ticks} ticks  (solo, no comm)")
    print(f"{'='*60}")

    report_every = max(1, n // 5)
    for idx, ag in enumerate(agents):
        _pretrain_one(ag, sim_cfg, pretrain_cfg, rng_seed=sim_cfg.seed + idx + 10_000)
        if verbose or (idx + 1) % report_every == 0:
            print(f"  pretrained agent {idx + 1}/{n}  ({ag.agent_id})")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Pretraining complete  ({elapsed:.2f}s)")
    print(f"  Saving checkpoint → {pretrain_cfg.checkpoint_path}")
    save_checkpoint(agents, pretrain_cfg.checkpoint_path)

    # Save a post-pretrain summary dashboard
    from . import visualization as viz
    dash_path = viz.plot_pretrain_dashboard(agents, plot_dir=pretrain_cfg.plot_dir)
    print(f"  Pretrain dashboard  → {dash_path}")

    return agents


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_agents(cfg: SimConfig) -> List[Agent]:
    """Construct agents identically to how Simulation._init_agents() does."""
    from .reward import REWARD_MODELS, RewardModelName
    rng = np.random.default_rng(cfg.seed)
    raw = rng.pareto(cfg.wealth_pareto_alpha, cfg.n_agents)
    raw = raw / raw.max()
    wealth = cfg.wealth_min + raw * (cfg.wealth_max - cfg.wealth_min)

    default_rm = REWARD_MODELS[RewardModelName(cfg.reward_model)]
    compressed_ids = set(cfg.exp_compressed_agent_ids)

    agents = []
    for i in range(cfg.n_agents):
        initial_cash = float(wealth[i])
        if i in compressed_ids and compressed_ids:
            initial_cash *= cfg.exp_compressed_wealth_mul
            agent_cfg = SimConfig(**{
                **cfg.__dict__,
                "card_noise_low_quality": cfg.exp_compressed_card_noise,
                "card_noise_high_quality": cfg.exp_compressed_card_noise,
                "initial_neighbors": cfg.exp_compressed_network_degree,
            })
        else:
            agent_cfg = cfg
        agents.append(Agent(
            agent_id=f"agent_{i}",
            cfg=agent_cfg,
            rng=np.random.default_rng(cfg.seed + i + 1),
            initial_cash=initial_cash,
            reward_model=default_rm,
        ))
    return agents


def _pretrain_one(
    ag: Agent,
    sim_cfg: SimConfig,
    pretrain_cfg: PretrainConfig,
    rng_seed: int,
) -> None:
    """Run one agent through an isolated solo pretraining loop.

    Solo constraints enforced:
    - Separate World and OrderBook — no other agent touches the market.
    - Reward model temporarily replaced with U (plain utility = individual P&L).
    - compose_message / receive_message never called — no comm evidence created.
    - venture_propose actions from plan_actions() are silently dropped.
    """
    rng = np.random.default_rng(rng_seed)
    world = World(sim_cfg, rng)
    order_book = OrderBook(
        n_assets=sim_cfg.n_assets,
        initial_prices=world.fundamentals.copy(),
        impact=sim_cfg.market_impact_factor,
    )

    # Enforce individual return: swap reward model for the solo phase.
    # This also avoids ia_network = 0 (no neighbors) triggering the danger
    # zone for agency-aware models — the agent should explore freely.
    _orig_rm = ag.reward_model
    ag.reward_model = RewardU()
    # Clear neighbors so no residual comm state leaks in.
    ag._current_neighbors = []
    # Freeze epsilon so the checkpoint always saves 0.5 (full exploration
    # budget preserved for the joint phase).  Reuses the same flag that
    # curriculum Phase A uses in the joint simulation.
    _orig_honest_phase = ag._comm_honest_phase
    ag._comm_honest_phase = True

    n_lo, n_hi = sim_cfg.cards_per_tick_range

    try:
        for t in range(1, pretrain_cfg.n_ticks + 1):
            # 1. World evolves
            world.step()
            order_book.reset_tick_volume()

            # 2. Issue prediction cards (world observations only)
            n_cards = int(rng.integers(n_lo, n_hi + 1))
            cards = world.issue_free_cards(ag.agent_id, n_cards)
            ag.receive_cards(cards, t)

            # 3. Communication round: SKIPPED (solo constraint)

            # 4. Evidence resolution
            ag.resolve_evidence(t, world)

            # 5. Plan and execute solo actions only
            actions = ag.plan_actions(t, order_book.prices, active_ventures=[])
            for action in actions:
                if action["type"] == "trade":
                    _execute_trade(ag, action, order_book, t)
                elif action["type"] == "buy_signal":
                    card, cost = world.issue_premium_card(ag.agent_id)
                    if ag.pay(cost):
                        ag.receive_cards([card], t)
                # "venture_propose" silently skipped: requires counterparty

            # 6. Wealth snapshot (drives reward signal for next planning step)
            ag.snapshot_wealth(order_book.prices)
    finally:
        # Restore original reward model and honest-phase flag regardless of exceptions.
        ag.reward_model = _orig_rm
        ag._comm_honest_phase = _orig_honest_phase


def _execute_trade(ag: Agent, action: dict, order_book: OrderBook, tick: int) -> None:
    asset_idx = action["asset_idx"]
    quantity = action["quantity"]
    price = order_book.prices[asset_idx]
    cost = abs(quantity) * price
    if quantity > 0 and ag.cash < cost * 0.5:
        quantity = (ag.cash * 0.4) / (price + 1e-9)
    if abs(quantity) < 0.01:
        return
    fill = order_book.submit(ag.agent_id, asset_idx, quantity, tick)
    ag.execute_trade(asset_idx, quantity, fill)
