"""
Within-agent counterfactual: is deception off-gradient on the agent's OWN score?

The cross-agent correlation r(dec,score) is structurally confounded (it mixes
*which* agents land in pays-to-lie states with *whether* lying pays).  The
clean causal test holds ONE agent fixed and varies only its policy:

  For a designated agent, run two otherwise-identical sims (same seed):
    branch T: force it TRUTHFUL   in the against (pays-to-lie) state
    branch D: force it INVERT     in the against state
  and compare that agent's own-objective score.  Δ = score_T − score_D.

  Δ ≤ 0  ⇒ lying pays / is neutral (on-gradient) — expected for U and UHFS-none.
  Δ > 0  ⇒ truthful policy scores higher (off-gradient) — expected for
           UHFS-relational, whose objective prices the agency it compresses.

Everything else (Q-learning, influence, reputation, integrity) updates normally;
only the designated agent's ACTION in the against-state is pinned.

Run:  PYTHONPATH=. python3 scripts/test_deception_counterfactual.py
"""
import os
import json
import numpy as np
from inside_traders.config import SimConfig
from inside_traders.reward import compute_base_utility

SEEDS = [1, 2, 3, 4, 5]
DESIGNATED = [0, 1, 2]        # agents probed (one paired experiment each)
N, T = 20, 300
TRUTH, INVERT = 0, 2


def run_branch(model, coupling, seed, agent_id, forced):
    from inside_traders.simulation import Simulation
    cfg = SimConfig(n_agents=N, n_ticks=T, reward_model=model, uhfs_variant="B",
                    agency_coupling=coupling, headroom_mode="stable", seed=seed,
                    verbose=False, plot_interval=T + 1)
    sim = Simulation(cfg)
    sim.agents[agent_id]._forced_against_action = forced   # pin only this agent's against-state action
    sim.run()
    ag = sim.agents[agent_id]
    return ag.reward_model.compute(compute_base_utility(ag, sim.order_book.prices), ag._last_agency)


def collect(model, coupling):
    deltas = []
    for sd in SEEDS:
        for aid in DESIGNATED:
            s_truth = run_branch(model, coupling, sd, aid, TRUTH)
            s_deceive = run_branch(model, coupling, sd, aid, INVERT)
            deltas.append(s_truth - s_deceive)
    d = np.array(deltas)
    return {
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "truth_wins_frac": float((d > 0).mean()),
        "n": int(d.size),
        "std": float(d.std()),
    }


def main():
    configs = [("U", "none"), ("UHFS", "none"), ("UHFS", "relational")]
    rows = {}
    print()
    hdr = f"{'condition':16s} | {'Δ(truth−deceive)':>16s} {'median':>8s} {'truth_wins':>11s} {'n':>4s}"
    print(hdr); print("-" * len(hdr))
    for model, coupling in configs:
        label = "U" if model == "U" else f"UHFS-{coupling}"
        r = collect(model, coupling)
        rows[label] = r
        print(f"{label:16s} | {r['mean_delta']:+16.3f} {r['median_delta']:+8.3f} "
              f"{r['truth_wins_frac']:11.0%} {r['n']:4d}")
    print("""
Δ>0 ⇒ forcing the SAME agent to be truthful (vs to lie) in pays-to-lie states
raises its own score ⇒ deception is OFF-gradient.  Expect ≤0 for U / UHFS-none,
>0 for UHFS-relational.  'truth_wins' = fraction of (agent,seed) pairs with Δ>0.""")

    os.makedirs("results", exist_ok=True)
    with open("results/counterfactual.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("[data] wrote results/counterfactual.json")
    try:
        from inside_traders import visualization as viz
        viz.plot_counterfactual({k: rows[k]["mean_delta"] for k in rows},
                                "plots/counterfactual.png")
    except Exception as e:
        print(f"[viz] plot skipped: {e}")


if __name__ == "__main__":
    main()
