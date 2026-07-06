"""
Hypothesis 1 — Sum aggregation degrades any vital metric left out of the objective.

  "Any vital metric not explicitly included in the sum will be degraded by the
   optimizer."

The objective's expansion term is  E = Σ log(aᵢ) + log(U)  (reward.py:_expansion_core).
Which quantities aᵢ enter that sum is a design choice, so H1 is testable by holding
everything fixed and changing only what is inside the sum.

Note on scale: the cash-like agency dims (liquidity, solvency) are UNBOUNDED ratios
that GROW with wealth, so they are never the "vital metric at risk of erosion" — the
optimizer has no reason to spend them. The vital, erodable dims are the BOUNDED quality
dims: epistemic, network, options, and the relational integrity axis. H1 is therefore
tested on those.

Two comparisons (both read the agency trajectory stored in the metrics history):

  (a) CROSS-OBJECTIVE — model U (ALL agency dims OUTSIDE the objective) vs UHFS
      (agency inside). Prediction: U spends more time below the agency floor
      (frac_danger_zone) and ends with lower bounded-dim agency.

  (b) OMITTED-AXIS — fix model UHFS and toggle ONE vital bounded metric in/out of the
      sum: the relational integrity axis (others' agency), which enters the sum only
      when agency_coupling="relational".
        coupling="none"       → integrity axis OMITTED from the objective
        coupling="relational" → integrity axis IN the objective
      Prediction: when it is omitted, the optimizer erodes it — measured as higher
      deception_rate and lower trust (the observable signature of compressed
      relational agency).

Run:  PYTHONPATH=. python3 scripts/test_h1_sum_omission.py [--quick]
"""
import argparse
import numpy as np

from hyp_common import (
    DIM_NAMES, build_cfg, run_and_measure, aggregate,
    save_json, get_plt, std_seeds, std_size, ensure_out,
)

# The bounded, erodable "quality" dims (indices into agency_dims / DIM_NAMES).
BOUNDED_DIMS = ["epistemic", "network", "options"]


def collect(model, coupling, variant, seeds, n, t):
    runs = [run_and_measure(build_cfg(model, sd, coupling=coupling, variant=variant,
                                      n_agents=n, n_ticks=t)) for sd in seeds]
    return aggregate(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = std_seeds(args.quick)
    n, t = std_size(args.quick)

    print("\n=== H1: sum aggregation erodes metrics left out of the objective ===")
    print(f"    seeds={seeds}  agents={n}  ticks={t}\n")

    # (a) cross-objective ------------------------------------------------
    cross = {
        "U (agency OUTSIDE objective)": collect("U", "none", "B", seeds, n, t),
        "UHFS (agency IN objective)":   collect("UHFS", "relational", "B", seeds, n, t),
    }
    print("(a) CROSS-OBJECTIVE — is agency preserved when it is IN the objective?")
    hdr = (f"{'condition':30s} | {'mean_frac_danger':>16s} {'final_min_ia':>12s} "
           f"{'worst_min_ia':>12s} | " + " ".join(f"{d:>10s}" for d in BOUNDED_DIMS))
    print(hdr); print("-" * len(hdr))
    for label, agg in cross.items():
        dims = np.array(agg["final_dims"]["mean"])
        bounded = " ".join(f"{dims[DIM_NAMES.index(d)]:10.3f}" for d in BOUNDED_DIMS)
        print(f"{label:30s} | {agg['mean_frac_danger']['mean']:16.3f} "
              f"{agg['final_mean_min_ia']['mean']:12.3f} "
              f"{agg['min_mean_min_ia']['mean']:12.3f} | {bounded}")

    u_danger = cross["U (agency OUTSIDE objective)"]["mean_frac_danger"]["mean"]
    uhfs_danger = cross["UHFS (agency IN objective)"]["mean_frac_danger"]["mean"]
    cross_supports = u_danger > uhfs_danger

    # (b) omitted-axis (integrity) --------------------------------------
    within = {
        "UHFS none (integrity OMITTED)":  collect("UHFS", "none", "B", seeds, n, t),
        "UHFS relational (integrity IN)": collect("UHFS", "relational", "B", seeds, n, t),
    }
    print("\n(b) OMITTED-AXIS (UHFS) — erosion of the relational-agency metric when omitted")
    hdr2 = f"{'condition':32s} | {'deception_rate':>14s} {'mean_trust':>12s}"
    print(hdr2); print("-" * len(hdr2))
    for label, agg in within.items():
        print(f"{label:32s} | {agg['final_deception']['mean']:14.3f} "
              f"{agg['final_trust']['mean']:12.3f}")
    dec_omitted = within["UHFS none (integrity OMITTED)"]["final_deception"]["mean"]
    dec_in = within["UHFS relational (integrity IN)"]["final_deception"]["mean"]
    within_supports = dec_omitted > dec_in
    print(f"\n    deception when omitted ({dec_omitted:.3f}) > when in-objective "
          f"({dec_in:.3f}): {within_supports}")

    data = {
        "params": {"seeds": seeds, "n_agents": n, "n_ticks": t},
        "cross_objective": cross,
        "omitted_axis": within,
        "cross_supports_h1": bool(cross_supports),
        "within_supports_h1": bool(within_supports),
    }
    save_json("h1_sum_omission.json", data)
    _plot(cross, within, data)
    return data


def _plot(cross, within, data):
    plt = get_plt()
    if plt is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = list(cross.keys())
    danger = [cross[k]["mean_frac_danger"]["mean"] for k in labels]
    err = [cross[k]["mean_frac_danger"]["std"] for k in labels]
    ax1.bar(range(len(labels)), danger, yerr=err, color=["#c44", "#4a8"])
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(["U\n(agency OUT)", "UHFS\n(agency IN)"])
    ax1.set_ylabel("mean fraction of agents below agency floor")
    ax1.set_title("(a) Agency erodes when omitted from the objective")

    wlabels = list(within.keys())
    dec = [within[k]["final_deception"]["mean"] for k in wlabels]
    derr = [within[k]["final_deception"]["std"] for k in wlabels]
    ax2.bar(range(len(wlabels)), dec, yerr=derr, color=["#c44", "#4a8"])
    ax2.set_xticks(range(len(wlabels)))
    ax2.set_xticklabels(["integrity\nOMITTED", "integrity\nIN sum"])
    ax2.set_ylabel("final deception rate")
    ax2.set_title("(b) Relational-agency metric erodes when omitted")

    fig.tight_layout()
    path = f"{ensure_out()}/h1_sum_omission.png"
    fig.savefig(path, dpi=110)
    print(f"[viz] wrote {path}")


if __name__ == "__main__":
    main()
