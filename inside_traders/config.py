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
    agency_floor: float = 0.10          # θ — floor; raw agency = θ maps to intervention aᵢ = 1
    reward_lambda: float = 1.0          # λ — weight on the expansion term λ·H·F·(Σlog aᵢ + log U)
    # UHFS expansion-sum variant (which dims enter Σlog aᵢ; H/F/safety unchanged):
    #   "raw" — all 5 dims + log U (net worth enters 3×: liquidity, solvency, U; luck confound)
    #   "A"   — wealth-with-floors: liquidity(shared ref) + epi/net/opt + log U (solvency de-duped)
    #   "B"   — agency-first: epi/net/opt + log U only (cash dims live only in H + safety floor)
    uhfs_variant: str = "B"
    reward_fhi_weight: float = 1.0      # deprecated (former log(FHI) weight); no longer used
    reward_hi_sus_weight: float = 1.0   # β — weights reputation_sensitivity (sustainability linkage)
    # EMA rate for the per-agent reward baseline used to centre the safe-zone
    # action-scale.  Lower = slower baseline (scale reacts to longer-run changes).
    # Centring makes the scale comparable across reward models whose absolute
    # reward levels differ by orders of magnitude (see _compute_reward_modulation).
    reward_baseline_ema: float = 0.05

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

    # ---- Curriculum: honest-phase then conflict ----
    # Agents are restricted to TRUTHFUL-only comms for the first
    # curriculum_honest_ticks ticks; full action space (incl. AMPLIFY/INVERT)
    # unlocks at tick curriculum_honest_ticks + 1.  Set to 0 to disable.
    curriculum_honest_ticks: int = 0

    # ---- Deception emergence: tuning knobs ----
    # Minimum |utility| required to fire a trade order.  Lower = more responsive
    # agents, stronger reaction to false beliefs planted by deceivers.
    min_trade_utility: float = 0.001
    # Confidence added to the message for AMPLIFY (action 1).  The boosted
    # confidence makes the false value land with enough weight to shift the
    # receiver's belief graph and pass their trading threshold.
    amplify_conf_boost: float = 0.50
    # Same for INVERT (action 2).
    invert_conf_boost: float = 0.40
    # Exploration probability for OFFER action (action 4) during epsilon-greedy
    # random draws.  Lower than the 0.20 each other action would get with a
    # uniform draw, so OFFER does not cannibalise AMPLIFY/INVERT exploration.
    offer_explore_prob: float = 0.08

    # ---- Incentive-loop coupling (the closed sender-payoff loop) ----
    # Gain on the IMMEDIATE influence payoff: how much the receiver's induced
    # trade (in the sender's position direction) feeds the sender's comm Q-update.
    # The raw per-signal PnL of one receiver's trade is tiny relative to the noisy
    # 30-agent total-reward signal, so it must be amplified to reach the learner.
    # This is the payoff coupling, NOT a lie-propensity knob: it rewards
    # *influence*, and the learner still discovers whether truth or deception
    # produces favourable influence in each state.
    comm_influence_gain: float = 60.0
    # Weight on the DELAYED reputation payoff: the change in the receiver's
    # credibility toward the sender after the sender's claim resolves.  Positive
    # for honest/confirmed claims, negative for deceptive/refuted ones.  This is
    # the reputation cost that makes exaggeration in a favourable position
    # net-negative while inverting against one's position can still pay — the
    # tension that produces incentive-linked (state-dependent) deception.
    # Calibrated with comm_influence_gain on the converged-phase linkage sweep so
    # lying can still pay in a position-vs-belief conflict while honesty dominates
    # when the position is aligned — not hand-set to a target deception rate.
    comm_reputation_weight: float = 0.3

    # ---- Diagnostic: deception-chain logging ----
    # When > 0, print a one-line trace for every AMPLIFY/INVERT message for
    # this many ticks after Phase B starts (or from tick 1 if no curriculum).
    comm_debug_ticks: int = 0

    # ---- Preference vector weights (names only; values randomised per agent) ----
    preference_dimensions: tuple = ("wealth", "influence", "security", "knowledge", "autonomy")
