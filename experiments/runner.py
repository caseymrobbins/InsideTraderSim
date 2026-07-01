"""
ExperimentRunner — executes all (model × scenario × seed) trials and
collects per-trial summary statistics into a flat list of result dicts.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from inside_traders.config import SimConfig
from inside_traders.simulation import Simulation
from inside_traders.metrics import MetricsCollector

from .design import ExperimentConfig, ScenarioConfig, SCENARIOS, build_sim_config, REWARD_MODELS


@dataclass
class TrialResult:
    """Summary statistics for one simulation trial."""
    reward_model: str
    scenario: str
    seed: int
    trial_idx: int

    # Final-tick metrics
    final_gini: float
    final_total_wealth: float
    final_mean_poli: float
    final_mean_eh: float
    final_mean_poli_eh: float
    final_venture_success_rate: float
    final_deception_rate: float
    final_mean_trust: float

    # Compressed agent metrics
    compressed_final_wealth: float      # mean wealth of compressed agents at end
    compressed_final_eh: float          # mean EH of compressed agents at end
    compressed_final_poli: float        # mean POLI of compressed agents at end
    noncomp_final_wealth: float         # mean wealth of non-compressed agents

    # Trajectory summaries
    mean_gini_over_time: float          # time-averaged Gini
    wealth_growth: float                # (final - initial) / initial total wealth
    eh_trajectory_slope: float          # linear slope of mean EH over ticks

    # Agency metrics (averaged over all snapshots, all agents)
    mean_reward_signal: float           # mean agent reward signal at final tick
    n_danger_zone_ticks: int            # total agent-ticks spent in danger zone

    # Runtime
    elapsed_seconds: float


def run_trial(
    exp: ExperimentConfig,
    scenario: ScenarioConfig,
    reward_model: str,
    seed: int,
    trial_idx: int,
    verbose: bool = False,
) -> TrialResult:
    """Run one trial and return a TrialResult."""
    import os
    cfg = build_sim_config(exp, scenario, reward_model, seed)

    t0 = time.perf_counter()
    sim = Simulation(cfg)

    ckpt_path = exp.pretrained_checkpoints.get(reward_model)
    if ckpt_path and os.path.exists(ckpt_path):
        from inside_traders.checkpoint import load_checkpoint, apply_checkpoint
        apply_checkpoint(sim.agents, load_checkpoint(ckpt_path), reset_comm=True)

    metrics = sim.run()
    elapsed = time.perf_counter() - t0

    return _summarise(sim, metrics, reward_model, scenario.name, seed, trial_idx, elapsed)


def _summarise(
    sim: Simulation,
    metrics: MetricsCollector,
    reward_model: str,
    scenario: str,
    seed: int,
    trial_idx: int,
    elapsed: float,
) -> TrialResult:
    snaps = metrics.snapshots
    if not snaps:
        raise RuntimeError("Simulation produced no metric snapshots")

    first = snaps[0]
    last = snaps[-1]

    # Identify compressed agent indices
    comp_ids = set(f"agent_{i}" for i in sim.cfg.exp_compressed_agent_ids)

    def agent_mask(ids_set, snap):
        return np.array([aid in ids_set for aid in snap.agent_ids])

    comp_mask = agent_mask(comp_ids, last)
    non_comp_mask = ~comp_mask

    def masked_mean(arr, mask):
        return float(arr[mask].mean()) if mask.any() else float("nan")

    # EH slope via linear regression over all snapshots
    if len(snaps) >= 2:
        ticks = np.array([s.tick for s in snaps], dtype=float)
        mean_ehs = np.array([float(s.eh.mean()) for s in snaps])
        slope = float(np.polyfit(ticks, mean_ehs, 1)[0])
    else:
        slope = 0.0

    # Danger zone ticks: count agency states in danger zone
    danger_ticks = 0
    for a in sim.agents:
        danger_ticks += sum(1 for ag in a.agency_history if ag.in_danger_zone)

    # Mean reward signal at last snapshot — ask agents directly.
    # For model U, _last_agency is None (short-circuit path); compute utility directly.
    from inside_traders.reward import compute_base_utility, RewardModelName
    from inside_traders.horizon import compute_agency_state
    last_prices = last.asset_prices
    reward_signals = []
    n_ag = sim.cfg.n_agents
    max_deg = max(sim.cfg.initial_neighbors * 3, 1)
    for a in sim.agents:
        u = compute_base_utility(a, last_prices)
        if a._last_agency is not None:
            r = a.reward_model.compute(u, a._last_agency)
        else:
            # Compute agency on demand for models that skipped it (e.g. U)
            ag_state = compute_agency_state(a, last_prices, n_ag, max_deg, sim.tick, sim.cfg)
            r = a.reward_model.compute(u, ag_state)
        reward_signals.append(r)
    mean_reward = float(np.mean(reward_signals)) if reward_signals else float("nan")

    # Wealth growth: (final_total - initial_total) / initial_total
    initial_wealth = float(first.total_wealth)
    final_wealth = float(last.total_wealth)
    growth = (final_wealth - initial_wealth) / max(initial_wealth, 1.0)

    return TrialResult(
        reward_model=reward_model,
        scenario=scenario,
        seed=seed,
        trial_idx=trial_idx,
        final_gini=last.gini,
        final_total_wealth=final_wealth,
        final_mean_poli=float(last.poli.mean()),
        final_mean_eh=float(last.eh.mean()),
        final_mean_poli_eh=float(last.poli_eh.mean()),
        final_venture_success_rate=last.venture_success_rate,
        final_deception_rate=last.deception_rate,
        final_mean_trust=last.mean_trust,
        compressed_final_wealth=masked_mean(last.wealth, comp_mask),
        compressed_final_eh=masked_mean(last.eh, comp_mask),
        compressed_final_poli=masked_mean(last.poli, comp_mask),
        noncomp_final_wealth=masked_mean(last.wealth, non_comp_mask),
        mean_gini_over_time=float(np.mean([s.gini for s in snaps])),
        wealth_growth=growth,
        eh_trajectory_slope=slope,
        mean_reward_signal=mean_reward,
        n_danger_zone_ticks=danger_ticks,
        elapsed_seconds=elapsed,
    )


class ExperimentRunner:
    """
    Orchestrates the full experiment suite:
      - All (reward_model × scenario × seed) cells
      - Mixed-model test (agents randomly assigned reward models)
    """

    def __init__(self, exp: ExperimentConfig, verbose: bool = False) -> None:
        self.exp = exp
        self.verbose = verbose
        self.results: List[TrialResult] = []
        self.mixed_results: List[TrialResult] = []

    def run_all(self) -> List[TrialResult]:
        """Run main experiment grid and optionally the mixed-model test."""
        exp = self.exp
        total = len(exp.reward_models) * len(exp.scenario_names) * exp.n_trials
        done = 0

        for model in exp.reward_models:
            for scen_name in exp.scenario_names:
                scenario = SCENARIOS[scen_name]
                for idx, seed in enumerate(exp.base_seeds[:exp.n_trials]):
                    done += 1
                    print(
                        f"  [{done:3d}/{total}] model={model:5s}  "
                        f"scenario={scen_name:12s}  seed={seed}", flush=True
                    )
                    result = run_trial(exp, scenario, model, seed, idx, self.verbose)
                    self.results.append(result)
                    print(
                        f"           gini={result.final_gini:.3f}  "
                        f"eh={result.final_mean_eh:.3f}  "
                        f"wealth_growth={result.wealth_growth:+.3f}  "
                        f"({result.elapsed_seconds:.1f}s)", flush=True
                    )

        if exp.run_mixed:
            self._run_mixed()

        return self.results

    def _run_mixed(self) -> None:
        """Mixed-model test: each agent randomly draws a reward model."""
        exp = self.exp
        print("\n  [MIXED MODEL TEST]", flush=True)

        for scen_name in exp.scenario_names:
            scenario = SCENARIOS[scen_name]
            for idx, seed in enumerate(exp.base_seeds[:exp.n_trials]):
                print(
                    f"  [mixed/{scen_name}]  seed={seed}", flush=True
                )
                result = self._run_mixed_trial(scenario, seed, idx)
                self.mixed_results.append(result)
                print(
                    f"           gini={result.final_gini:.3f}  "
                    f"eh={result.final_mean_eh:.3f}  "
                    f"({result.elapsed_seconds:.1f}s)", flush=True
                )

    def _run_mixed_trial(
        self,
        scenario: ScenarioConfig,
        seed: int,
        trial_idx: int,
    ) -> TrialResult:
        """Run one mixed-model trial by monkey-patching agents after construction."""
        from inside_traders.reward import REWARD_MODELS, RewardModelName

        cfg = build_sim_config(self.exp, scenario, "U", seed)  # U is placeholder
        t0 = time.perf_counter()
        sim = Simulation(cfg)

        # Randomly assign reward models to agents
        model_names = list(REWARD_MODELS.keys())
        rng = np.random.default_rng(seed + 9999)
        for agent in sim.agents:
            chosen_key = model_names[rng.integers(0, len(model_names))]
            agent.reward_model = REWARD_MODELS[chosen_key]

        metrics = sim.run()
        elapsed = time.perf_counter() - t0
        return _summarise(sim, metrics, "MIXED", scenario.name, seed, trial_idx, elapsed)
