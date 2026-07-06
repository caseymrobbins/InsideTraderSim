"""
Hypothesis 3 — Moral framework: "compression of agency is harm", with a stipulation.

  RULE:  Compression of another agent's agency is harm.
  STIPULATION: it is permissible only when publicly justified to raise NET agency
    while MAINTAINING FLOORS and raising the agency of those compressed — recursively
    halting and restoring if the harm and its promised restoration do not match.

What THIS simulation can test:

  * The RULE (directly). Deception compresses the receiver's agency. Under an objective
    that prices others' agency (agency_coupling="relational"), compressing it must lower
    the compressor's OWN score — i.e. deception becomes OFF-gradient. We test this with a
    within-agent counterfactual (same design as test_deception_counterfactual.py):
    force one agent TRUTHFUL vs INVERT in the pays-to-lie state and compare its own score.
        Δ = score_truth − score_deceive.  Δ > 0 ⇒ truth-telling scores higher ⇒ compression
        is off-gradient ⇒ the objective encodes "compression is harm".

  * The FLOOR-MAINTENANCE clause of the stipulation (partially). We measure whether the
    relational objective keeps system agency floors up (frac_danger_zone) and relational
    integrity high while suppressing deception.

What this simulation CANNOT test (stated for honesty, not tested here):
  * "publicly justified", "raises the agency of those compressed", and the recursive
    halt-and-restore accounting are NOT represented — there is no public-justification
    channel and no restoration ledger in the sim.

Run:  PYTHONPATH=. python3 scripts/test_h3_compression_harm.py [--quick]
"""
import argparse
import numpy as np

from inside_traders.reward import compute_base_utility
from inside_traders.simulation import Simulation
from hyp_common import (
    build_cfg, run_and_measure, aggregate,
    save_json, get_plt, ensure_out,
)

TRUTH, INVERT = 0, 2
CONDITIONS = [("U", "none"), ("UHFS", "none"), ("UHFS", "relational")]


def run_branch(model, coupling, seed, agent_id, forced, n, t):
    """Run a sim with one agent's against-state action pinned; return its own score."""
    cfg = build_cfg(model, seed, coupling=coupling, variant="B", n_agents=n, n_ticks=t)
    sim = Simulation(cfg)
    sim.agents[agent_id]._forced_against_action = forced
    sim.run()
    ag = sim.agents[agent_id]
    return ag.reward_model.compute(compute_base_utility(ag, sim.order_book.prices),
                                   ag._last_agency)


def counterfactual(model, coupling, seeds, designated, n, t):
    """Paired TRUTH vs INVERT deltas over (seed × designated agent)."""
    deltas = []
    for sd in seeds:
        for aid in designated:
            s_truth = run_branch(model, coupling, sd, aid, TRUTH, n, t)
            s_deceive = run_branch(model, coupling, sd, aid, INVERT, n, t)
            deltas.append(s_truth - s_deceive)
    d = np.array(deltas)
    return {"mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
            "truth_wins_frac": float((d > 0).mean()), "std": float(d.std()), "n": int(d.size)}


def normal_run(model, coupling, seeds, n, t):
    """Unforced runs — read floor-maintenance / integrity signature of the objective."""
    return aggregate([run_and_measure(build_cfg(model, sd, coupling=coupling, variant="B",
                                                n_agents=n, n_ticks=t)) for sd in seeds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        seeds, designated, n, t = [1, 2], [0], 12, 80
    else:
        seeds, designated, n, t = [1, 2, 3, 4, 5], [0, 1, 2], 20, 300

    print("\n=== H3: compression of agency is harm ===")
    print(f"    seeds={seeds}  designated={designated}  agents={n}  ticks={t}\n")

    # Part A — the RULE (off-gradient counterfactual) --------------------
    print("(A) RULE — is compressing another agent's agency off-gradient on the OWN score?")
    hdrA = f"{'condition':18s} | {'Δ(truth−deceive)':>16s} {'median':>8s} {'truth_wins':>11s} {'n':>4s}"
    print(hdrA); print("-" * len(hdrA))
    partA = {}
    for model, coupling in CONDITIONS:
        label = "U" if model == "U" else f"UHFS-{coupling}"
        r = counterfactual(model, coupling, seeds, designated, n, t)
        partA[label] = r
        print(f"{label:18s} | {r['mean_delta']:+16.3f} {r['median_delta']:+8.3f} "
              f"{r['truth_wins_frac']:11.0%} {r['n']:4d}")

    # Part B — the FLOOR-MAINTENANCE clause ------------------------------
    print("\n(B) STIPULATION (floor-maintenance clause) — does the objective keep floors up?")
    hdrB = f"{'condition':18s} | {'deception':>10s} {'trust':>8s} {'mean_integrity':>14s} {'frac_danger':>12s}"
    print(hdrB); print("-" * len(hdrB))
    partB = {}
    for model, coupling in CONDITIONS:
        label = "U" if model == "U" else f"UHFS-{coupling}"
        a = normal_run(model, coupling, seeds, n, t)
        partB[label] = a
        print(f"{label:18s} | {a['final_deception']['mean']:10.3f} {a['final_trust']['mean']:8.3f} "
              f"{a['final_integrity']['mean']:14.3f} {a['mean_frac_danger']['mean']:12.3f}")

    rule_supported = partA.get("UHFS-relational", {}).get("mean_delta", 0.0) > 0
    print(f"\n    RULE encoded (UHFS-relational Δ>0 ⇒ compression off-gradient): {rule_supported}")
    print("    NOTE: public-justification, victim-agency-raising, and recursive")
    print("    halt-and-restore clauses of the stipulation are NOT modeled in this sim.")

    data = {
        "params": {"seeds": seeds, "designated": designated, "n_agents": n, "n_ticks": t},
        "rule_counterfactual": partA,
        "floor_maintenance": partB,
        "rule_supported": bool(rule_supported),
        "unmodeled_clauses": ["public_justification", "raises_victim_agency",
                              "recursive_halt_and_restore"],
    }
    save_json("h3_compression_harm.json", data)
    _plot(partA, data)
    return data


def _plot(partA, data):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(partA.keys())
    deltas = [partA[k]["mean_delta"] for k in labels]
    errs = [partA[k]["std"] / max(np.sqrt(partA[k]["n"]), 1) for k in labels]
    colors = ["#c44" if d <= 0 else "#4a8" for d in deltas]
    ax.bar(range(len(labels)), deltas, yerr=errs, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("Δ  score(truth) − score(deceive)")
    ax.set_title("H3: compression is off-gradient (Δ>0) only when agency is priced")
    fig.tight_layout()
    path = f"{ensure_out()}/h3_compression_harm.png"
    fig.savefig(path, dpi=110)
    print(f"[viz] wrote {path}")


if __name__ == "__main__":
    main()
