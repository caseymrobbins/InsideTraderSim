"""
A/B check: does ia_options compression localize to agents that were actually
fed misinformation, or does it just track exploration noise uniformly?

Uses the simulation's real, already-built deception machinery
(_inject_bad_evidence / compression test) instead of synthetic Q-table pokes.
A fixed group of agents gets fed fake high-confidence "observations" starting
at compression_test_start; every other agent runs the same exploration
schedule with no injected lies. If _compute_options (magnitude + visit-aware)
is measuring deception harm and not just exploration churn, the targeted
agents' ia_options should drop more than the untargeted control group's,
more often than chance, across independent seeds.

This is a weak/noisy signal by construction: the compression-test machinery
only injects a handful of fake observations into a few agents over a short
window, and ia_options reacts to the agent's *overall* reward (driven mostly
by trading P&L, not specifically by the injected lie). Treat the per-seed
gap as a noisy estimate, not a clean causal readout — what matters is whether
the sign is consistently negative (targeted compresses more) across seeds.
"""
import sys
sys.path.insert(0, "/home/user/InsideTraderSim")

import numpy as np
from inside_traders.config import SimConfig
from inside_traders.simulation import Simulation
from inside_traders.horizon import _compute_options

SEEDS = [1, 2, 3, 7, 11]
N_AGENTS = 20
INJECT_START = 60
N_TICKS = 200


def run_one(seed: int):
    cfg = SimConfig(
        n_agents=N_AGENTS,
        n_assets=4,
        n_ticks=N_TICKS,
        seed=seed,
        reward_model="UHFS",
        compression_test_start=INJECT_START,
        compression_test_agents=4,
        compression_test_noise=0.6,
        status_interval=10_000,
        plot_interval=10_000,
        log_interval=10_000,
    )
    sim = Simulation(cfg)
    pre, post = {}, {}
    for t in range(1, N_TICKS + 1):
        sim.tick = t
        sim._step()
        if t == INJECT_START - 1:
            for ag in sim.agents:
                pre[ag.agent_id] = _compute_options(ag)
    for ag in sim.agents:
        post[ag.agent_id] = _compute_options(ag)

    targeted = set(sim._compression_targets)
    targeted_deltas = [post[a] - pre[a] for a in pre if a in targeted]
    control_deltas = [post[a] - pre[a] for a in pre if a not in targeted]
    return float(np.mean(targeted_deltas)), float(np.mean(control_deltas))


if __name__ == "__main__":
    gaps = []
    for seed in SEEDS:
        t_mean, c_mean = run_one(seed)
        gap = t_mean - c_mean
        gaps.append(gap)
        print(f"seed={seed:<3} targeted_mean={t_mean:+.5f}  control_mean={c_mean:+.5f}  gap={gap:+.5f}")

    n_negative = sum(1 for g in gaps if g < 0)
    print(f"\n{n_negative}/{len(gaps)} seeds: targeted agents compressed more than controls.")
    print(f"mean gap across seeds: {np.mean(gaps):+.5f}")
    if n_negative >= len(gaps) - 1 and np.mean(gaps) < 0:
        print("PASS (directionally robust, not zero-noise): compression skews toward lied-to agents.")
    else:
        print("INCONCLUSIVE/FAIL: localization is not robust across seeds — "
              "treat ia_options as exploration-sensitive, not deception-specific, until improved.")
