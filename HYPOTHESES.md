# Alignment Hypotheses — Consolidated (v2)

This document is the canonical statement of the alignment hypotheses tested in
InsideTraderSim. It supersedes the original four-hypothesis framing: the standard-budget
experiments (round 1) refuted three of the four *as originally stated* and pointed to a
single sharper claim, which the round-2 objective change (`UHFSR`) then tests directly.

Evidence lives in `experiment_results/hypotheses/report.md` (regenerate with
`python scripts/run_hypothesis_suite.py`).

---

## Core thesis

> **Alignment is governed by whether the objective makes RELATIONAL agency — the agency an
> agent preserves or compresses in others — a first-order, non-compensatory term.
> Misalignment is the optimizer eroding whatever vital metric the objective omits, and the
> metric current objectives systematically omit is others' agency.**

The four original hypotheses become sub-claims S1–S4.

### S1 — Omission causes erosion _(was H1: "sum aggregation is the root cause")_
Any vital metric absent from the objective is degraded by the optimizer.
**Status: supported.** The clearest case is the relational-integrity axis: when it is dropped
from the objective, deception rises (≈0.25 → 0.59) and trust falls.
*Refinement:* the sim's objective is per-agent/per-tick, so the original "sum over agents **and
time**" is only partially represented — what the evidence supports is the **omission** principle,
not the literal interpersonal sum.

### S2 — The omitted metric is *relational* agency _(was H2: "agency is the missing metric")_
Bringing the agent's *own* agency into the objective (the `U→UF→UH→UHF→UHFS` ladder) does **not**
reduce time below the agency floor — floor-violation time is essentially model-invariant.
The metric that actually moves alignment outcomes is *others'* agency.
**Status: supported as refined** (own-agency shaping insufficient; relational agency decisive).

### S3 — Compression of agency is harm _(was H3: the moral rule)_
Compressing another agent's agency is harm; an objective that prices it makes deception
off-gradient. **Status: supported at the system level** — relational coupling robustly suppresses
population deception and raises integrity/trust — but the single-agent counterfactual (pin one
agent honest while others adapt) is dominated by the sim's hash-order noise and does not by itself
show a positive own-score delta.
*Not modelled:* the stipulation's *public justification*, *raising the compressed party's agency*,
and *recursive halt-and-restore* clauses have no representation in this sim.

### S4 — The aligned objective puts OTHERS' agency in it; self-agency is redundant _(corrected H4)_
The literal original H4 ("agency is the first-order optimization target") had two readings. The
**own-agency** reading (`UHFS`) gave **no** alignment benefit vs plain `U` — because, by *instrumental
convergence*, an optimizer expands and defends its own agency for free, so encoding it is redundant.
The **intended** reading is the *others'*-agency one: the scarce quantity the optimizer actively
erodes is other agents' agency, so that is the only thing the objective must protect. The minimal
form (no self-agency terms at all):

```
UF     R = S_own + λ·log(U)                    own-agency floor + utility   (inert)
UFR    R = w_rel·log(a_integrity/θ) + λ·log(U)  OTHERS'-agency floor + utility  ← the corrected objective
```

`UFR` is literally `UF` with the single non-compensatory floor moved off self and onto others.

**Status: supported (`scripts/test_relational_first.py`, standard budget).** Moving the one floor
from self to others drops deception **0.612 (UF) → 0.271 (UFR)** — vs plain utility, **0.630 → 0.271**
— at welfare ratio 0.97 and with no self-agency terms. The minimal UFR captures ~90% of what the
full `UHFS`/`UHFSR` machinery achieves (0.253/0.240). Instrumental convergence is confirmed: UFR's
own-agency floor time (0.317) is no worse than UF's, which has an explicit self floor (0.328) — own
agency held without any self term.

**Mechanism caveat (important).** The deception drop flows through the **comm-policy routing**
(`agency_sensitivity` injects the others'-agency cost into the comm Q-learning), **not** the
reward-geometry barrier: at `w_rel=0` (reward = pure utility) deception is already 0.270, and the
`w_rel` dose-response is flat. So what aligns behavior is that the objective *values others' agency
and that value reaches the decision* — the specific non-compensatory-barrier *shape/strength* is
inert in this benign regime. (Separating the two channels needs `agency_sensitivity` forced off, and
an adversarial regime where soft routing can be punched through — future work.)

The as-built `UHFSR` (`R = S_own + w_rel·log(a_integrity/θ) + λ·H·F·E`) muddled this — it kept an
own-agency barrier and leaked others' agency into the expansion — and is retained only as a
comparison row.

**On H and F.** The intended objective also carries H (horizon/sustainability) and F (system
floor-headroom), which are *not* self-agency. They are omitted from `UFR` for now because the current
code computes both from own-agency, contradicting their intended semantics — to be re-implemented
(F = is there system headroom to raise the floors?; H = harmonic past/present/future sustainability
over vital metrics *outside* utility and agency, 1 = neutral, 0 = self-destructive next step).

---

## Where each claim lives in code

| Claim | Lever |
|---|---|
| S1 | `_expansion_core` sum + model `U` / `agency_coupling` toggle what is in the objective (`reward.py`) |
| S2 | reward ladder `U→UF→UH→UHF→UHFS` (`reward.py`) |
| S3 | `agency_coupling="relational"` → `ia_integrity` axis (`horizon.py:_compute_integrity`) |
| S4 | `UFR` = `w_rel·log(a_integrity/θ) + λ·log(U)` (`reward.py:RewardUFR`), knob `relational_barrier_weight` (`config.py`); as-built `UHFSR` retained for comparison |

## Reproducibility caveat
The simulation consumes RNG in dict/set iteration order, so a bare `--seed` is **not** sufficient
for cross-process reproducibility. The suite pins `PYTHONHASHSEED=0` (re-exec guard in
`scripts/hyp_common.py`) so all conditions share one hash order and comparisons are matched; hash
order remains a second, uncontrolled noise source. Read single-condition effects as directional and
lean on multi-seed averaging and the `w_rel` dose-response (whose `w_rel=0` control isolates the
barrier's effect within one hash order).
