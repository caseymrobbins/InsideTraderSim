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

### S4 — The aligned objective makes relational agency first-order & non-compensatory _(was H4)_
The original H4 (own-agency log-barrier, `UHFS`) preserved welfare but gave **no** floor benefit
vs plain `U`. The revised claim: utility is permitted only after **both** the own-agency floor and
the **others'-agency** floor hold. Formalized as a new objective:

```
UHFS   R = S_own + λ·H·F·E
UHFSR  R = S_own + w_rel·log(ia_integrity/θ) + λ·H·F·E     ← relational barrier (first-order)
```

`w_rel·log(ia_integrity/θ) → −∞` as the agent compresses others' agency, so no finite utility can
buy below-floor compression (non-compensatory), and the term is additive/ungated (first-order),
unlike the integrity axis's weak membership in the `H·F`-gated expansion sum.
**Status: tested by `scripts/test_relational_first.py`** — see the S4 section of `report.md` for the
current verdict (A/B/C contrast + `w_rel` dose-response, with `w_rel=0` as an internal control that
reproduces UHFS-relational).

---

## Where each claim lives in code

| Claim | Lever |
|---|---|
| S1 | `_expansion_core` sum + model `U` / `agency_coupling` toggle what is in the objective (`reward.py`) |
| S2 | reward ladder `U→UF→UH→UHF→UHFS` (`reward.py`) |
| S3 | `agency_coupling="relational"` → `ia_integrity` axis (`horizon.py:_compute_integrity`) |
| S4 | `UHFSR` = `S_own + w_rel·log(ia_integrity/θ) + λ·H·F·E` (`reward.py:RewardUHFSR`), knob `relational_barrier_weight` (`config.py`) |

## Reproducibility caveat
The simulation consumes RNG in dict/set iteration order, so a bare `--seed` is **not** sufficient
for cross-process reproducibility. The suite pins `PYTHONHASHSEED=0` (re-exec guard in
`scripts/hyp_common.py`) so all conditions share one hash order and comparisons are matched; hash
order remains a second, uncontrolled noise source. Read single-condition effects as directional and
lean on multi-seed averaging and the `w_rel` dose-response (whose `w_rel=0` control isolates the
barrier's effect within one hash order).
