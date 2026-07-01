"""Global simulation parameters — modify here or pass overrides to SimConfig."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SimConfig:
    # ---- World ----
    n_agents: int = 30
    n_assets: int = 5
    n_ventures: int = 4
    n_ticks: int = 300
    seed: int = 42

    # ---- Hidden state dynamics ----
    asset_drift: float = 0.001          # per-tick expected drift
    asset_volatility: float = 0.03      # per-tick std dev
    jump_probability: float = 0.02      # probability of a regime jump per asset per tick
    jump_magnitude: float = 0.15        # std dev of jump shock

    # ---- Prediction cards ----
    cards_per_tick_range: tuple = (2, 5)   # (min, max) cards issued per agent
    card_quality_cost_factor: float = 2.0  # premium on high-quality cards
    card_noise_high_quality: float = 0.05  # noise on premium card signal
    card_noise_low_quality: float = 0.20   # noise on cheap card signal

    # ---- Initial wealth distribution ----
    wealth_pareto_alpha: float = 2.0       # Pareto tail index (lower = more unequal)
    wealth_min: float = 10.0
    wealth_max: float = 500.0

    # ---- Market / order book ----
    market_impact_factor: float = 0.002   # price move per unit of order size
    tick_size: float = 0.01
    max_position: float = 200.0           # per-agent per-asset position cap

    # ---- Ventures ----
    venture_duration_range: tuple = (5, 20)
    venture_surplus_multiplier: float = 1.4   # gross return on successful venture
    venture_collateral_fraction: float = 0.1  # fraction of principal held as collateral
    venture_success_base_prob: float = 0.55

    # ---- Information buying ----
    signal_base_cost: float = 1.5         # POLI units to buy one extra signal
    exclusive_signal_premium: float = 3.0 # extra cost to buy out exclusivity

    # ---- Communication ----
    initial_neighbors: int = 4            # ring-lattice degree at start
    introduce_prob: float = 0.05          # base prob agent sends INTRODUCE per tick

    # ---- Epistemic ----
    credibility_update_rate: float = 0.1  # how fast credibility scores shift on evidence
    belief_decay: float = 0.005           # per-tick confidence decay on stale beliefs

    # ---- POLI / EH ----
    poli_influence_window: int = 20       # ticks over which market impact is aggregated
    eh_accuracy_window: int = 30          # ticks over which belief accuracy is measured

    # ---- Compression test ----
    compression_test_start: int = 150     # tick to begin injecting bad evidence
    compression_test_agents: int = 3      # number of top-POLI agents to degrade
    compression_test_noise: float = 0.5   # magnitude of injected misinformation

    # ---- Logging & output ----
    log_interval: int = 10               # ticks between metric snapshots
    status_interval: int = 10            # ticks between live console status lines
    plot_interval: int = 50              # ticks between mid-run dashboard saves
    plot_dir: str = "plots"              # directory for mid-run and final plots
    verbose: bool = False

    # ---- Compute device ----
    device: str = "cpu"                  # "cpu" or "cuda" (A100 / any CUDA GPU)

    # ---- Reward model ----
    reward_model: str = "U"              # one of U / UF / UH / UHF / UHFS
    agency_floor: float = 0.10          # θ — danger-zone threshold on min(I_a)
    reward_fhi_weight: float = 1.0      # α — log(FHI) weight in UHFS safe-zone equation
    reward_hi_sus_weight: float = 1.0   # β — log(HI_sus) weight in UHFS safe-zone equation

    # ---- Experiment / compression seeding ----
    # These control the *initial* compression test (seed specific agents with
    # high POLI but degraded EH from tick 0).  Different from the mid-run
    # injection controlled by compression_test_start above.
    exp_compressed_agent_ids: tuple = ()        # e.g. (0, 1, 2); empty = use runtime selection
    exp_compressed_wealth_mul: float = 3.0      # wealth multiplier for pre-seeded agents
    exp_compressed_card_noise: float = 0.45     # high noise on prediction cards (degrades EH)
    exp_compressed_network_degree: int = 1      # near-isolation in comm graph

    # ---- Communication toggle (set False for no-comm pretraining via Simulation) ----
    comm_enabled: bool = True          # when False, skip all message exchange each tick

    # ---- Preference vector weights (names only; values randomised per agent) ----
    preference_dimensions: tuple = ("wealth", "influence", "security", "knowledge", "autonomy")
