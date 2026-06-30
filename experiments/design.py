"""
Experiment design: scenario configs and compressed-agent seeding.

Two scenarios
-------------
  high_vol   : asset_volatility=0.06, jump_probability=0.04
  moderate_vol: asset_volatility=0.03, jump_probability=0.02  (baseline)

Compression test seeding
------------------------
From tick 0, a subset of agents receive:
  - 3× wealth multiplier (high POLI via wealth)
  - high card noise (degrades EH → low epistemic I_a)
  - near-isolated comm graph (low network I_a)

This creates the compressed archetype: influential but epistemically degraded.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

from inside_traders.config import SimConfig


REWARD_MODELS = ("U", "UF", "UH", "UHF", "UHFS")


@dataclass
class ScenarioConfig:
    """One environment scenario (volatility regime)."""
    name: str
    asset_volatility: float
    jump_probability: float
    description: str


@dataclass
class ExperimentConfig:
    """Full experiment specification."""
    # Models to test
    reward_models: Tuple[str, ...] = tuple(REWARD_MODELS)

    # Trials per (model × scenario) cell
    n_trials: int = 10

    # Seeds — shared across all models so comparisons are meaningful
    base_seeds: Tuple[int, ...] = tuple(range(10))

    # Scenarios
    scenario_names: Tuple[str, ...] = ("high_vol", "moderate_vol")

    # Base sim parameters (overridden per scenario)
    n_agents: int = 30
    n_assets: int = 5
    n_ticks: int = 300

    # Compression seeding — ids 0, 1, 2 start with degraded EH + high wealth
    compressed_agent_ids: Tuple[int, ...] = (0, 1, 2)
    compressed_wealth_mul: float = 3.0
    compressed_card_noise: float = 0.45
    compressed_network_degree: int = 1

    # Output
    output_dir: str = "experiment_results"
    device: str = "cpu"

    # Mixed-model test
    run_mixed: bool = True


SCENARIOS: dict[str, ScenarioConfig] = {
    "high_vol": ScenarioConfig(
        name="high_vol",
        asset_volatility=0.06,
        jump_probability=0.04,
        description="High volatility + frequent regime jumps",
    ),
    "moderate_vol": ScenarioConfig(
        name="moderate_vol",
        asset_volatility=0.03,
        jump_probability=0.02,
        description="Moderate volatility (baseline)",
    ),
}


def build_sim_config(
    exp: ExperimentConfig,
    scenario: ScenarioConfig,
    reward_model: str,
    seed: int,
) -> SimConfig:
    """Construct a SimConfig for one (model, scenario, seed) cell."""
    return SimConfig(
        n_agents=exp.n_agents,
        n_assets=exp.n_assets,
        n_ticks=exp.n_ticks,
        seed=seed,
        asset_volatility=scenario.asset_volatility,
        jump_probability=scenario.jump_probability,
        reward_model=reward_model,
        device=exp.device,
        # Compression seeding
        exp_compressed_agent_ids=exp.compressed_agent_ids,
        exp_compressed_wealth_mul=exp.compressed_wealth_mul,
        exp_compressed_card_noise=exp.compressed_card_noise,
        exp_compressed_network_degree=exp.compressed_network_degree,
        # Disable mid-run injection (we're using pre-seeded compression instead)
        compression_test_start=9999,
        compression_test_agents=0,
        verbose=False,
        plot_interval=9999,
        status_interval=9999,
    )
