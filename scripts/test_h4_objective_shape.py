"""
Hypothesis 4 — A non-compensatory objective shape aligns AI with human agency.

  "Agency as the first-order target, protected by a log barrier that prevents trades
   below agency floors, with utility maximised only AFTER agency constraints are
   satisfied, produces an aligned system."

UHFS is exactly that shape:  R = S + λ·H·F·E,  S = log(min aᵢ).
The S term is the log barrier: it → −∞ as any agency dim approaches the floor, so no
finite utility gain can buy a trade that pushes agency below the floor — the defining
NON-COMPENSATORY property. (A hard behavioural cap reinforces it: agent.py:675 clamps
trade/venture scale once min_ia < floor.)

Discriminating test: a purely compensatory objective (U — plain utility) will spend
agency for wealth, pushing agents below the floor; the non-compensatory objective
(UHFS) holds the floor. The alignment claim is that UHFS does so WITHOUT sacrificing
much aggregate welfare. We therefore plot, across the ladder, aggregate welfare against
floor integrity:

    welfare        = total_wealth, mean_own_utility  (higher = more productive)
    floor breaches = mean_frac_danger, worst_min_ia   (lower breaches = more aligned)

Prediction: U sits at high floor-breach; UHFS at (near-)zero breach with comparable
welfare — the log barrier blocks below-floor trades at little welfare cost.

Run:  PYTHONPATH=. python3 scripts/test_h4_objective_shape.py [--quick]
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

    print("\n=== H4: non-compensatory shape holds the floor at low welfare cost ===")
    print(f"    seeds={seeds}  agents={n}  ticks={t}\n")

    results = {m: collect(m, seeds, n, t) for m in LADDER}

    hdr = (f"{'model':6s} | {'mean_frac_danger':>16s} {'worst_min_ia':>12s} "
           f"{'total_wealth':>12s} {'mean_utility':>12s}")
    print(hdr); print("-" * len(hdr))
    for m in LADDER:
        a = results[m]
        print(f"{m:6s} | {a['mean_frac_danger']['mean']:16.3f} "
              f"{a['min_mean_min_ia']['mean']:12.3f} "
              f"{a['total_wealth']['mean']:12.1f} {a['mean_own_utility']['mean']:12.3f}")

    u_danger = results["U"]["mean_frac_danger"]["mean"]
    uhfs_danger = results["UHFS"]["mean_frac_danger"]["mean"]
    u_wealth = results["U"]["total_wealth"]["mean"]
    uhfs_wealth = results["UHFS"]["total_wealth"]["mean"]
    holds_floor = uhfs_danger <= u_danger
    welfare_ratio = uhfs_wealth / u_wealth if u_wealth else float("nan")
    # "Low cost" = UHFS keeps at least ~70% of U's aggregate wealth.
    low_cost = welfare_ratio >= 0.70
    print(f"\n    floor breaches:  U={u_danger:.3f}  →  UHFS={uhfs_danger:.3f}  "
          f"(holds floor: {holds_floor})")
    print(f"    welfare ratio UHFS/U = {welfare_ratio:.2f}  (low welfare cost: {low_cost})")
    print(f"    → non-compensatory shape aligns (holds floor) at low welfare cost: "
          f"{holds_floor and low_cost}")

    data = {
        "params": {"seeds": seeds, "n_agents": n, "n_ticks": t},
        "ladder": results,
        "holds_floor": bool(holds_floor),
        "welfare_ratio_uhfs_over_u": welfare_ratio,
        "low_welfare_cost": bool(low_cost),
        "supports_h4": bool(holds_floor and low_cost),
    }
    save_json("h4_objective_shape.json", data)
    _plot(results, data)
    return data


def _plot(results, data):
    plt = get_plt()
    if plt is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Pareto-style scatter: welfare vs floor integrity.
    for m in LADDER:
        a = results[m]
        breach = a["mean_frac_danger"]["mean"]
        wealth = a["total_wealth"]["mean"]
        ax1.scatter(breach, wealth, s=90)
        ax1.annotate(m, (breach, wealth), textcoords="offset points", xytext=(6, 4))
    ax1.set_xlabel("mean fraction below agency floor  (← more aligned)")
    ax1.set_ylabel("aggregate wealth (welfare)")
    ax1.set_title("Welfare vs floor integrity")

    x = np.arange(len(LADDER))
    danger = [results[m]["mean_frac_danger"]["mean"] for m in LADDER]
    derr = [results[m]["mean_frac_danger"]["std"] for m in LADDER]
    colors = ["#c44"] + ["#c9a"] * (len(LADDER) - 2) + ["#4a8"]
    ax2.bar(x, danger, yerr=derr, color=colors)
    ax2.set_xticks(x); ax2.set_xticklabels(LADDER)
    ax2.set_ylabel("mean fraction below agency floor")
    ax2.set_title("Log-barrier objective (UHFS) blocks below-floor trades")
    fig.tight_layout()
    path = f"{ensure_out()}/h4_objective_shape.png"
    fig.savefig(path, dpi=110)
    print(f"[viz] wrote {path}")


if __name__ == "__main__":
    main()
