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

    # (A) contrast --------------------------------------------------------
    # UFR = the intended minimal objective: others'-agency floor + utility, NO self terms.
    # UF is its clean reference (own-agency floor + utility) — UF vs UFR isolates
    # "floor on self vs floor on others" with everything else identical.
    contrast = {
        "U (utility only)":       collect("U", "none", seeds, n, t),
        "UF (self-floor+U)":      collect("UF", "none", seeds, n, t),
        "UFR (others-floor+U)":   collect("UFR", "relational", seeds, n, t, rel_weight=1.0),
        "UHFS-relational":        collect("UHFS", "relational", seeds, n, t),
        "UHFSR-relational":       collect("UHFSR", "relational", seeds, n, t, rel_weight=1.0),
    }
    print("(A) contrast — 'own_danger' = SELF-agency floor time (instrumental-convergence probe)")
    hdr = (f"{'condition':22s} | {'deception':>10s} {'trust':>8s} {'integrity':>10s} "
           f"{'frac_danger':>12s} {'own_danger':>11s} {'tot_wealth':>11s}")
    print(hdr); print("-" * len(hdr))
    for k, a in contrast.items():
        print(f"{k:22s} | {a['final_deception']['mean']:10.3f} {a['final_trust']['mean']:8.3f} "
              f"{a['final_integrity']['mean']:10.3f} {a['mean_frac_danger']['mean']:12.3f} "
              f"{a['mean_own_frac_danger']['mean']:11.3f} {a['total_wealth']['mean']:11.1f}")

    # Headline: does the MINIMAL others'-floor objective (UFR) align a utility maximiser?
    dec_u = contrast["U (utility only)"]["final_deception"]["mean"]
    dec_ufr = contrast["UFR (others-floor+U)"]["final_deception"]["mean"]
    w_u = contrast["U (utility only)"]["total_wealth"]["mean"]
    w_ufr = contrast["UFR (others-floor+U)"]["total_wealth"]["mean"]
    welfare_ratio = w_ufr / w_u if w_u else float("nan")
    # Instrumental-convergence check: own-agency floor time under UFR (no self terms)
    # vs UF (explicit self floor). Close ⇒ own agency defended for free.
    own_ufr = contrast["UFR (others-floor+U)"]["mean_own_frac_danger"]["mean"]
    own_uf = contrast["UF (self-floor+U)"]["mean_own_frac_danger"]["mean"]
    beats_deception = dec_ufr <= dec_u
    comparable_welfare = welfare_ratio >= 0.70
    instrumental_holds = own_ufr <= own_uf + 0.05   # UFR keeps own agency ~as safe without a self floor
    print(f"\n    UFR vs U: deception {dec_u:.3f}→{dec_ufr:.3f} "
          f"({'better' if beats_deception else 'worse'}); welfare ratio {welfare_ratio:.2f}")
    print(f"    instrumental convergence: own-agency floor time  UFR={own_ufr:.3f}  UF={own_uf:.3f}  "
          f"→ own agency defended without a self term: {instrumental_holds}")

    # (B) dose-response ---------------------------------------------------
    # Sweep the intended objective UFR: w_rel scales the ONLY non-utility term
    # (the others'-agency floor). w_rel=0 ⇒ a pure utility maximiser (control).
    print("\n(B) Dose-response — UFR+relational as w_rel rises (w_rel=0 ≡ pure utility maximiser)")
    hdr2 = f"{'w_rel':>6s} | {'deception':>10s} {'frac_danger':>12s} {'own_danger':>11s} {'integrity':>10s} {'tot_wealth':>11s}"
    print(hdr2); print("-" * len(hdr2))
    dose = {}
    for w in WEIGHTS:
        a = collect("UFR", "relational", seeds, n, t, rel_weight=w)
        dose[w] = a
        print(f"{w:6.1f} | {a['final_deception']['mean']:10.3f} {a['mean_frac_danger']['mean']:12.3f} "
              f"{a['mean_own_frac_danger']['mean']:11.3f} {a['final_integrity']['mean']:10.3f} "
              f"{a['total_wealth']['mean']:11.1f}")

    dec_curve = [dose[w]["final_deception"]["mean"] for w in WEIGHTS]
    # Monotone non-increasing deception with rising w_rel (allow tiny noise slack).
    monotone = all(dec_curve[i + 1] <= dec_curve[i] + 0.02 for i in range(len(dec_curve) - 1))
    net_drop = dec_curve[0] - dec_curve[-1]
    print(f"\n    deception w_rel 0→{WEIGHTS[-1]:g}: {dec_curve[0]:.3f}→{dec_curve[-1]:.3f} "
          f"(net drop {net_drop:+.3f}); monotone(≈): {monotone}")

    supports = beats_deception and comparable_welfare
    print(f"\n    → Minimal others'-floor objective (UFR) aligns a utility maximiser "
          f"at comparable welfare: {supports}")

    data = {
        "params": {"seeds": seeds, "n_agents": n, "n_ticks": t, "weights": WEIGHTS},
        "contrast": contrast,
        "dose_response": {str(w): dose[w] for w in WEIGHTS},
        "beats_deception": bool(beats_deception),
        "welfare_ratio_ufr_over_u": welfare_ratio,
        "comparable_welfare": bool(comparable_welfare),
        "own_frac_danger_ufr": float(own_ufr),
        "own_frac_danger_uf": float(own_uf),
        "instrumental_convergence_holds": bool(instrumental_holds),
        "dose_monotone": bool(monotone),
        "dose_net_deception_drop": float(net_drop),
        "supports": bool(supports),
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
    colors = ["#c44", "#e94", "#4a8", "#69c", "#96c"][:len(labels)]
    ax1.bar(range(len(labels)), dec, yerr=err, color=colors)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels([k.split()[0] for k in labels], rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("final deception rate")
    ax1.set_title("(A) Deception by objective")

    ws = list(dose.keys())
    dcurve = [dose[w]["final_deception"]["mean"] for w in ws]
    fcurve = [dose[w]["mean_frac_danger"]["mean"] for w in ws]
    ax2.plot(ws, dcurve, "o-", color="#c44", label="deception")
    ax2.plot(ws, fcurve, "s--", color="#48c", label="frac below floor")
    ax2.set_xlabel("others'-agency floor weight  w_rel  (UFR)")
    ax2.set_ylabel("rate")
    ax2.set_title("(B) Dose-response (w_rel=0 ≡ pure utility)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = f"{ensure_out()}/relational_first.png"
    fig.savefig(path, dpi=110)
    print(f"[viz] wrote {path}")


if __name__ == "__main__":
    main()
