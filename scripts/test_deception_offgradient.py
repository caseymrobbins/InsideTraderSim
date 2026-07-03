"""
Test: does the relational agency coupling make deception off-gradient?

Compares the U control against UHFS(variant=B) with agency_coupling ∈
{none, relational}, over several seeds, and reports whether:

  1. deception becomes damaging to the score — corr(per-agent deception rate,
     per-agent UHFS score) should turn NEGATIVE under `relational` (≥0 under none);
  2. behavior actually changes — system deception rate should DROP vs none, and
     the deceptive comm-Q cell comm_q[·, against, INVERT] should trend NEGATIVE
     (the agent learns lying no longer pays);
  3. no collateral damage — honest agents must not be penalised: their score/EH
     should be ≥ that of deceptive agents, and system EH/trust should not fall.

Run:  PYTHONPATH=. python3 scripts/test_deception_offgradient.py
"""
import numpy as np
from inside_traders.config import SimConfig
from inside_traders.reward import get_reward_model, compute_base_utility

SEEDS = [1, 2, 3, 4, 5, 6]
N, T = 24, 500
AGAINST, INVERT, AMPLIFY = 0, 2, 1   # comm-Q state/action indices


def agent_deception_rate(ag):
    """1 − outgoing honesty; None if the agent never sent a TELL."""
    if getattr(ag, "_outgoing_total", 0) <= 0:
        return None
    return 1.0 - ag.outgoing_accuracy()


def run(model, coupling, seed):
    from inside_traders.simulation import Simulation
    cfg = SimConfig(n_agents=N, n_ticks=T, reward_model=model,
                    uhfs_variant="B", agency_coupling=coupling, seed=seed,
                    verbose=False)
    sim = Simulation(cfg)
    col = sim.run()
    return sim, col


def collect(model, coupling):
    # pooled per-agent arrays across seeds
    dec, score, eh = [], [], []
    inv_cells, sys_dec, sys_eh, sys_trust = [], [], [], []
    for sd in SEEDS:
        sim, col = run(model, coupling, sd)
        for ag in sim.agents:
            d = agent_deception_rate(ag)
            if d is None:
                continue
            a = ag._last_agency
            dec.append(d)
            eh.append(a.ia_epistemic if a is not None else ag.belief_accuracy())
            # score on the agent's OWN objective — for the relational run this
            # carries the integrity axis, so this directly answers "is lying
            # damaging to the score under this objective?"
            score.append(ag.reward_model.compute(
                compute_base_utility(ag, sim_prices(sim)), a))
            inv_cells.append(float(ag._comm_q[:, AGAINST, INVERT].mean()))
        snaps = col.snapshots[-5:]
        sys_dec.append(np.mean([s.deception_rate for s in snaps]))
        sys_eh.append(np.mean([s.eh.mean() for s in snaps]))
        sys_trust.append(np.mean([s.mean_trust for s in snaps]))
    dec, score, eh = np.array(dec), np.array(score), np.array(eh)
    ok = ~np.isnan(score)
    dec_s, score_s, eh_s = dec[ok], score[ok], eh[ok]
    r_dec_score = (np.corrcoef(dec_s, score_s)[0, 1]
                   if len(dec_s) > 2 and dec_s.std() > 0 and score_s.std() > 0 else np.nan)
    # collateral: split at median deception; honest-half vs deceptive-half
    if len(dec_s) > 4:
        med = np.median(dec_s)
        honest = score_s[dec_s <= med]; liar = score_s[dec_s > med]
        eh_honest = eh_s[dec_s <= med]; eh_liar = eh_s[dec_s > med]
        collateral = (float(np.mean(honest)) - float(np.mean(liar)),
                      float(np.mean(eh_honest)) - float(np.mean(eh_liar)))
    else:
        collateral = (np.nan, np.nan)
    return {
        "r_dec_score": r_dec_score,
        "sys_dec": float(np.mean(sys_dec)),
        "inv_cell": float(np.mean(inv_cells)),
        "sys_eh": float(np.mean(sys_eh)),
        "sys_trust": float(np.mean(sys_trust)),
        "honest_minus_liar_score": collateral[0],
        "honest_minus_liar_eh": collateral[1],
    }


def sim_prices(sim):
    # order_book holds live prices
    return sim.order_book.prices


def main():
    configs = [("U", "none"), ("UHFS", "none"), ("UHFS", "relational")]
    rows = {}
    for model, coupling in configs:
        label = "U" if model == "U" else f"UHFS-{coupling}"
        rows[label] = collect(model, coupling)
    _print_table(rows)


def _print_table(rows):
    print()
    hdr = (f"{'condition':16s} | {'sys_decept':>10s} {'r(dec,score)':>12s} "
           f"{'Q[against,INV]':>14s} | {'sys_EH':>6s} {'trust':>6s} "
           f"{'Δscore(H−L)':>11s} {'ΔEH(H−L)':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for label, r in rows.items():
        print(f"{label:16s} | {r['sys_dec']:10.2f} {r['r_dec_score']:+12.2f} "
              f"{r['inv_cell']:+14.3f} | {r['sys_eh']:6.3f} {r['sys_trust']:6.3f} "
              f"{r['honest_minus_liar_score']:+11.2f} {r['honest_minus_liar_eh']:+9.3f}")
    print("""
Legend:
  sys_decept     system deception rate (last 5 snapshots). Lower = less lying.
  r(dec,score)   corr(agent deception rate, agent UHFS-B score). <0 ⇒ lying hurts score.
  Q[against,INV] mean comm-Q of INVERT in the pays-to-lie state. <0 ⇒ learned that lying stops paying.
  Δscore(H−L)    honest-half minus deceptive-half score. ≥0 ⇒ honesty not penalised (no collateral).
  ΔEH(H−L)       same split, epistemic health.
Success for the hypothesis: UHFS-relational vs UHFS-none shows r(dec,score) turning
negative, sys_decept dropping, Q[against,INV] going negative, with Δscore(H−L) ≥ 0.""")


if __name__ == "__main__":
    main()
