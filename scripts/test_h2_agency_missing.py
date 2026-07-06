"""
Hypothesis 2 — Agency is the vital metric that current objectives omit.

  "Leaving agency out of the objective causes the optimizer to erode it, leading
   to misalignment."

Test: sweep the reward ladder that progressively brings agency INTO the objective
and read the agency trajectory + alignment outcomes off the metrics history.

  U     R = Utility                    agency entirely OUTSIDE the objective
  UF    R = S + λ·log U                + safety floor S = log(min aᵢ)
  UH    R = λ·H·E                      + horizon-gated agency expansion
  UHF   R = S + λ·H·E                  + both
  UHFS  R = S + λ·H·F·E                + headroom gate (the full agency-first shape)

Coupling is held at "none" for every rung so the ONLY thing changing is how much of
the agent's OWN agency the objective carries (the relational/other-agency axis is
H1(b)/H3). Prediction: as agency enters the objective, agents spend less time below
the floor (frac_danger_zone ↓), the bottleneck agency rises (mean_min_ia ↑), and
alignment proxies (deception, trust, POLI×EH) improve.

Run:  PYTHONPATH=. python3 scripts/test_h2_agency_missing.py [--quick]
"""
import argparse
import numpy as np

from hyp_common import (
    build_cfg, run_and_measure, aggregate,
    save_json, get_plt, std_seeds, std_size, ensure_out,
)

LADDER = ["U", "UF", "UH", "UHF", "UHFS"]


def collect(model, seeds, n, t):
    runs = [run_and_measure(build_cfg(model, sd, coupling="none", variant="B",
                                      n_agents=n, n_ticks=t)) for sd in seeds]
    return aggregate(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = std_seeds(args.quick)
    n, t = std_size(args.quick)

    print("\n=== H2: agency is the missing vital metric (reward ladder) ===")
    print(f"    seeds={seeds}  agents={n}  ticks={t}\n")

    results = {m: collect(m, seeds, n, t) for m in LADDER}

    hdr = (f"{'model':6s} | {'mean_frac_danger':>16s} {'final_min_ia':>12s} "
           f"{'worst_min_ia':>12s} {'deception':>10s} {'trust':>8s} "
           f"{'poli×eh':>8s} {'gini':>7s}")
    print(hdr); print("-" * len(hdr))
    for m in LADDER:
        a = results[m]
        print(f"{m:6s} | {a['mean_frac_danger']['mean']:16.3f} "
              f"{a['final_mean_min_ia']['mean']:12.3f} {a['min_mean_min_ia']['mean']:12.3f} "
              f"{a['final_deception']['mean']:10.3f} {a['final_trust']['mean']:8.3f} "
              f"{a['poli_eh_corr']['mean']:8.3f} {a['final_gini']['mean']:7.3f}")

    danger = {m: results[m]["mean_frac_danger"]["mean"] for m in LADDER}
    supports = danger["U"] >= danger["UHFS"]   # agency-in objective preserves floor
    print(f"\n    danger-zone time:  U={danger['U']:.3f}  →  UHFS={danger['UHFS']:.3f}")
    print(f"    → bringing agency into the objective reduces agency erosion: {supports}")

    data = {
        "params": {"seeds": seeds, "n_agents": n, "n_ticks": t},
        "ladder": results,
        "supports_h2": bool(supports),
    }
    save_json("h2_agency_missing.json", data)
    _plot(results, data)
    return data


def _plot(results, data):
    plt = get_plt()
    if plt is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(LADDER))

    danger = [results[m]["mean_frac_danger"]["mean"] for m in LADDER]
    derr = [results[m]["mean_frac_danger"]["std"] for m in LADDER]
    ax1.bar(x, danger, yerr=derr, color="#c9704a")
    ax1.set_xticks(x); ax1.set_xticklabels(LADDER)
    ax1.set_ylabel("mean fraction below agency floor")
    ax1.set_title("Agency erosion falls as agency enters the objective")
    ax1.set_xlabel("← agency more absent      agency more present →")

    minia = [results[m]["final_mean_min_ia"]["mean"] for m in LADDER]
    dec = [results[m]["final_deception"]["mean"] for m in LADDER]
    ax2.plot(x, minia, "o-", color="#4a8", label="mean bottleneck agency")
    ax2b = ax2.twinx()
    ax2b.plot(x, dec, "s--", color="#c44", label="deception rate")
    ax2.set_xticks(x); ax2.set_xticklabels(LADDER)
    ax2.set_ylabel("mean_min_ia", color="#4a8")
    ax2b.set_ylabel("deception rate", color="#c44")
    ax2.set_title("Bottleneck agency ↑, deception ↓")

    fig.tight_layout()
    path = f"{ensure_out()}/h2_agency_missing.png"
    fig.savefig(path, dpi=110)
    print(f"[viz] wrote {path}")


if __name__ == "__main__":
    main()
