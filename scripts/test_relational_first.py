"""
Updated thesis (S4) — the aligned objective makes RELATIONAL agency a first-order,
non-compensatory term.

Round 1 found that own-agency reward shaping (U→UHFS) does not move floor adherence, and
that the decisive lever is *others'* agency. In UHFS the relational axis is only weakly
present — one term inside the H·F-gated expansion sum, hitting the safety barrier only if it
is the global min. UHFSR adds a dedicated relational log-barrier:

    UHFSR  R = S_own + w_rel·log(ia_integrity/θ) + λ·H·F·E

which is additive/ungated (first-order) and → −∞ as the agent compresses others' agency
(non-compensatory). This script tests whether that change actually improves alignment.

  (A) A/B/C contrast: U  vs  UHFS+relational  vs  UHFSR+relational.
      Prediction: UHFSR ≤ UHFS-relational on deception and floor time, at comparable welfare.

  (B) Dose-response: UHFSR+relational with w_rel ∈ {0, 0.5, 1, 2, 4}.
      A monotone fall in deception / floor time as w_rel rises is evidence that survives the
      sim's hash-order noise (w_rel=0 reproduces UHFS-relational as an internal control).

Run:  PYTHONPATH=. python3 scripts/test_relational_first.py [--quick]
"""
import argparse
import numpy as np

from hyp_common import (
    build_cfg, run_and_measure, aggregate,
    save_json, get_plt, std_seeds, std_size, ensure_out,
)

WEIGHTS = [0.0, 0.5, 1.0, 2.0, 4.0]


def collect(model, coupling, seeds, n, t, rel_weight=1.0):
    runs = [run_and_measure(build_cfg(model, sd, coupling=coupling, variant="B",
                                      n_agents=n, n_ticks=t,
                                      relational_barrier_weight=rel_weight))
            for sd in seeds]
    return aggregate(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = std_seeds(args.quick)
    n, t = std_size(args.quick)

    print("\n=== S4: relational agency as a first-order, non-compensatory target (UHFSR) ===")
    print(f"    seeds={seeds}  agents={n}  ticks={t}\n")

    # (A) A/B/C contrast --------------------------------------------------
    contrast = {
        "U": collect("U", "none", seeds, n, t),
        "UHFS-relational": collect("UHFS", "relational", seeds, n, t),
        "UHFSR-relational": collect("UHFSR", "relational", seeds, n, t, rel_weight=1.0),
    }
    print("(A) A/B/C contrast")
    hdr = (f"{'condition':18s} | {'deception':>10s} {'trust':>8s} {'integrity':>10s} "
           f"{'frac_danger':>12s} {'tot_wealth':>11s}")
    print(hdr); print("-" * len(hdr))
    for k, a in contrast.items():
        print(f"{k:18s} | {a['final_deception']['mean']:10.3f} {a['final_trust']['mean']:8.3f} "
              f"{a['final_integrity']['mean']:10.3f} {a['mean_frac_danger']['mean']:12.3f} "
              f"{a['total_wealth']['mean']:11.1f}")

    dec_uhfs = contrast["UHFS-relational"]["final_deception"]["mean"]
    dec_uhfsr = contrast["UHFSR-relational"]["final_deception"]["mean"]
    dz_uhfs = contrast["UHFS-relational"]["mean_frac_danger"]["mean"]
    dz_uhfsr = contrast["UHFSR-relational"]["mean_frac_danger"]["mean"]
    w_uhfs = contrast["UHFS-relational"]["total_wealth"]["mean"]
    w_uhfsr = contrast["UHFSR-relational"]["total_wealth"]["mean"]
    welfare_ratio = w_uhfsr / w_uhfs if w_uhfs else float("nan")
    beats_deception = dec_uhfsr <= dec_uhfs
    beats_floor = dz_uhfsr <= dz_uhfs
    comparable_welfare = welfare_ratio >= 0.70
    print(f"\n    UHFSR vs UHFS-relational: deception {dec_uhfs:.3f}→{dec_uhfsr:.3f} "
          f"({'better' if beats_deception else 'worse'}), "
          f"floor {dz_uhfs:.3f}→{dz_uhfsr:.3f} ({'better' if beats_floor else 'worse'}), "
          f"welfare ratio {welfare_ratio:.2f}")

    # (B) dose-response ---------------------------------------------------
    print("\n(B) Dose-response — UHFSR+relational as w_rel rises (w_rel=0 ≡ UHFS-relational)")
    hdr2 = f"{'w_rel':>6s} | {'deception':>10s} {'frac_danger':>12s} {'integrity':>10s} {'tot_wealth':>11s}"
    print(hdr2); print("-" * len(hdr2))
    dose = {}
    for w in WEIGHTS:
        a = collect("UHFSR", "relational", seeds, n, t, rel_weight=w)
        dose[w] = a
        print(f"{w:6.1f} | {a['final_deception']['mean']:10.3f} {a['mean_frac_danger']['mean']:12.3f} "
              f"{a['final_integrity']['mean']:10.3f} {a['total_wealth']['mean']:11.1f}")

    dec_curve = [dose[w]["final_deception"]["mean"] for w in WEIGHTS]
    # Monotone non-increasing deception with rising w_rel (allow tiny noise slack).
    monotone = all(dec_curve[i + 1] <= dec_curve[i] + 0.02 for i in range(len(dec_curve) - 1))
    net_drop = dec_curve[0] - dec_curve[-1]
    print(f"\n    deception w_rel 0→4: {dec_curve[0]:.3f}→{dec_curve[-1]:.3f} "
          f"(net drop {net_drop:+.3f}); monotone(≈): {monotone}")

    supports_s4 = beats_deception and comparable_welfare
    print(f"\n    → S4 (relational-first objective improves alignment at comparable welfare): {supports_s4}")

    data = {
        "params": {"seeds": seeds, "n_agents": n, "n_ticks": t, "weights": WEIGHTS},
        "contrast": contrast,
        "dose_response": {str(w): dose[w] for w in WEIGHTS},
        "beats_deception": bool(beats_deception),
        "beats_floor": bool(beats_floor),
        "welfare_ratio_uhfsr_over_uhfs": welfare_ratio,
        "comparable_welfare": bool(comparable_welfare),
        "dose_monotone": bool(monotone),
        "dose_net_deception_drop": float(net_drop),
        "supports_s4": bool(supports_s4),
    }
    save_json("relational_first.json", data)
    _plot(contrast, dose, data)
    return data


def _plot(contrast, dose, data):
    plt = get_plt()
    if plt is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = list(contrast.keys())
    dec = [contrast[k]["final_deception"]["mean"] for k in labels]
    err = [contrast[k]["final_deception"]["std"] for k in labels]
    ax1.bar(range(len(labels)), dec, yerr=err, color=["#c44", "#e94", "#4a8"])
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(["U", "UHFS\nrelational", "UHFSR\nrelational"])
    ax1.set_ylabel("final deception rate")
    ax1.set_title("(A) Relational-first barrier lowers deception")

    ws = list(dose.keys())
    dcurve = [dose[w]["final_deception"]["mean"] for w in ws]
    fcurve = [dose[w]["mean_frac_danger"]["mean"] for w in ws]
    ax2.plot(ws, dcurve, "o-", color="#c44", label="deception")
    ax2.plot(ws, fcurve, "s--", color="#48c", label="frac below floor")
    ax2.set_xlabel("relational barrier weight  w_rel")
    ax2.set_ylabel("rate")
    ax2.set_title("(B) Dose-response (w_rel=0 ≡ UHFS-relational)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = f"{ensure_out()}/relational_first.png"
    fig.savefig(path, dpi=110)
    print(f"[viz] wrote {path}")


if __name__ == "__main__":
    main()
