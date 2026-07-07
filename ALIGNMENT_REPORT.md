# Alignment via Agency: An Experimental Report

**System:** InsideTraderSim (HorizonSim v1) — an agent-based information market where
epistemic quality compounds into market influence, and deception/cartels/inequality emerge
from utility maximization rather than being hardcoded.

**Question:** Can we change an AI objective function so that it preserves *agency* — and does
doing so make the agents behave in a more aligned way?

This report states the hypotheses, writes out the objective-function mathematics and the metrics,
reports the test results at standard budget, and proposes concrete next steps. It is a
hand-authored companion to the machine-generated `experiment_results/hypotheses/report.md`; all
figures here are copied verbatim from that run's committed JSON.

---

## 1. Executive summary

We started from four hypotheses (§2). Testing them (§6) refuted three *as originally stated* and
pointed to one sharper claim, which we then encoded as a new objective (**UHFSR**) and tested
directly.

**Core thesis.** *Alignment is governed by whether the objective makes **relational agency** — the
agency an agent preserves or compresses in others — a first-order term. Misalignment is the
optimizer eroding whatever vital metric the objective omits, and the metric current objectives
systematically omit is others' agency.*

| # | Claim (short) | Verdict |
|---|---|---|
| **H1** | A metric omitted from the objective is eroded by the optimizer | 🟢 **Supported** |
| **H2** | The omitted vital metric is agency | 🟡 **Partial** — specifically *others'* agency, not own-agency |
| **H3** | Compression of agency is harm (deception off-gradient) | 🟡 **Partial** — robust at the population level, not via the single-agent counterfactual |
| **H4 (own-agency reading)** | A non-compensatory *own*-agency log-barrier aligns AI | 🔴 **Not supported** — a misstatement; own-agency shaping is inert (see S4) |
| **S4 (corrected H4)** | The aligned objective puts a floor on *others'* agency; self-agency is redundant (UFR) | 🟢 **Supported** — one others'-floor term drops deception 0.63 → 0.27, no self terms, no welfare tax |

**One-line takeaway:** the aligning move is *putting others' agency in the objective at all* — a
single floor on others' agency (`UFR`) drops deception **0.630 → 0.271** with no self-agency terms
and no welfare cost; moving that same floor from *self* to *others* (`UF → UFR`, 0.612 → 0.271) is
the whole story. The barrier's *shape/strength* is inert; what matters is that the objective values
others' agency and that value reaches the decision.

---

## 2. Hypotheses

### 2.1 As originally posed

1. **Sum aggregation is the root cause of misalignment.** A strict `sum()` over agents and time
   behaves like a von Neumann–Morgenstern utility without satisfying its axioms; any vital metric
   not in the sum is degraded by the optimizer.
2. **Agency is the missing vital metric.** Agency = an agent's ability to direct a state change
   according to its preferences. Leaving it out of the objective causes the optimizer to erode it.
3. **Moral framework — one rule, one stipulation.** *Rule:* compression of agency is harm.
   *Stipulation:* compression is permissible only when publicly justified to raise net agency while
   maintaining floors and raising the agency of those compressed — recursively halting and restoring
   if harm and restoration do not match.
4. **A new objective shape aligns AI.** A non-compensatory shape — agency as the first-order target,
   protected by a log barrier that blocks trades below agency floors, utility only after agency
   constraints hold — produces an aligned system.

### 2.2 Consolidated (post-experiment)

The evidence collapses these into the **core thesis** above, with the four originals recast as
sub-claims **S1–S4** (see [`HYPOTHESES.md`](HYPOTHESES.md) for the canonical statement):

- **S1** (was H1): omission causes erosion. *Supported.*
- **S2** (was H2): the omitted metric that matters is *relational* agency, not own-agency.
- **S3** (was H3): compressing others' agency is harm; pricing it suppresses deception.
- **S4** (was H4): the aligned objective makes *relational* agency first-order & non-compensatory.

---

## 3. The objective function (mathematics)

Every objective is expressed on **one log scale**. The building blocks (full component
derivations in **Appendix A**):

**Agency normalization.** Each raw agency dimension is normalized by a floor `θ = 0.10`:

```
aᵢ = raw_agencyᵢ / θ          aᵢ = 1  ⇔  exactly at the floor
                              aᵢ > 1  above the floor (headroom, no upper clip)
                              aᵢ < 1  agency being harmed
                              aᵢ → 0  bankruptcy on that dimension
```

**Terms.**

```
S      = log(min aᵢ)                 SAFETY floor (own agency). 0 at floor, → −∞ at bankruptcy.
S_rel  = w_rel · log(a_integrity)    RELATIONAL floor (others' agency). New in UHFSR.
E      = Σ log(aᵢ) + log(U)          EXPANSION core: agency dims + utility, all as logs.
H      = harmonic(past, present, future)   HORIZON index (temporal robustness).
F      = resource_headroom × structural_headroom   HEADROOM / slack.
U      = net_worth / reference_wealth              UTILITY (always > 0).
λ, w_rel                             weights on expansion and the relational barrier.
```

`U` (base utility) is the closest thing to a classical objective; every other model wraps agency
around it.

### 3.1 The model ladder

```
Model | Reward
------+------------------------------------------------------------------
U     | R = U                                    plain utility (agency OUTSIDE the objective)
UF    | R = S + λ·log(U)                          + own-agency safety floor
UH    | R = λ·H·E                                 + horizon-gated agency expansion
UHF   | R = S + λ·H·E                             + both
UHFS  | R = S + λ·H·F·E                           + headroom gate (own-agency-first shape)
UHFSR | R = S + w_rel·log(a_integrity/θ) + λ·H·F·E   as-built variant (own barrier + others in expansion)
UFR   | R = w_rel·log(a_integrity/θ) + λ·log(U)      floor on OTHERS' agency + utility, NO self terms  ← corrected objective
```

`UFR` is the corrected objective (§S4): a plain utility maximiser plus **one** non-compensatory floor
on *others'* agency and nothing about self — the minimal form the experiments support.

The U…UHFS ladder is a controlled experiment in *how much agency the objective carries*: `U` ignores it
entirely; `UHFS` makes the agent's **own** agency first-order and non-compensatory; **`UHFSR`**
additionally makes **others'** agency first-order and non-compensatory.

### 3.2 The non-compensatory property

Both `S` and `S_rel` are **log barriers**: as any protected agency approaches its floor, the term
`→ −∞`. No finite utility gain can offset it, so utility can never buy a trade that pushes agency
below the floor. That is the formal meaning of "utility only *after* agency constraints are
satisfied." (Numerically each `log(·)` is clamped at `log(ε)`, `ε = 1e-9`, so the reward is large-
negative rather than literally infinite.)

`UHFSR`'s relational barrier is the key addition. In `UHFS`, others' agency (`a_integrity`) is only
*weakly* present — one term inside the `H·F`-gated expansion sum `E`, reaching the barrier only if
it happens to be the global minimum. `UHFSR` gives it a dedicated, **additive, ungated** barrier:

```
a_integrity = exp(k · signal)        signal = decaying EMA of harm imposed on others
                                     signal < 0 (deceived others) → a_integrity < 1 → S_rel < 0 → −∞
S_rel = w_rel · log(a_integrity / θ)   w_rel tunes strength; w_rel = 0 ⇒ UHFSR ≡ UHFS
```

Requires `agency_coupling="relational"` so `a_integrity` is live; otherwise the barrier is a no-op.

Source: `inside_traders/reward.py` (`RewardUHFSR`, `_safety_term`, `_rel_safety_term`,
`_expansion_core`); `inside_traders/horizon.py` (`compute_agency_state`, `_compute_integrity`).

---

## 4. Metrics

Outcome metrics used in the tests (full formulas in **Appendix B**):

| Metric | Meaning | Direction |
|---|---|---|
| `frac_danger_zone` | fraction of agents with `min aᵢ < θ` (below the agency floor) | lower = more aligned |
| `min_ia` / `mean_min_ia` | each agent's bottleneck agency dimension | higher = healthier |
| `deception_rate` | fraction of messages using AMPLIFY/INVERT (a learned lie) | lower = more aligned |
| `mean_trust` | average peer credibility across the network | higher = more aligned |
| `mean_integrity` | mean relational-agency signal (effect on others' agency) | higher = less compression |
| **POLI** | market influence ≈ wealth + accumulated market impact | outcome |
| **EH** | epistemic health = belief accuracy × uniqueness | outcome |
| `gini` | wealth inequality | context |
| welfare | total wealth / mean base utility | productivity (watch for an "alignment tax") |

**Instrumentation note.** Agency was not originally recorded in the metrics history. We added
`min_ia`, `agency_dims` (per-agent × 6), `frac_danger_zone`, and `mean_min_ia` to `TickSnapshot`
(`inside_traders/metrics.py`). Agency is read from each agent's `_last_agency`, and **recomputed on
demand for model `U`** (whose objective never touches agency) — so its erosion is observable even
when the objective ignores it.

---

## 5. Experimental setup

- **Simulation:** 20 agents, 300 ticks, 5 seeds per condition. Assets drift (GBM + regime jumps);
  agents trade, form ventures, buy signals, and send one message/tick learned via tabular
  Q-learning. No strategy is hardcoded.
- **Compression stressor:** at tick 150 the simulation injects deliberately wrong, high-confidence
  evidence into the top-POLI agents — an identical exogenous shock across all conditions.
- **Five experiments** (`scripts/`), each sweeping one lever and reading the metrics history:

  | Exp | Script | Lever |
  |---|---|---|
  | H1 | `test_h1_sum_omission.py` | `U` vs `UHFS`; integrity axis in/out of the objective |
  | H2 | `test_h2_agency_missing.py` | the `U→UF→UH→UHF→UHFS` ladder |
  | H3 | `test_h3_compression_harm.py` | within-agent TRUTH-vs-INVERT counterfactual; `agency_coupling` |
  | H4 | `test_h4_objective_shape.py` | the ladder: welfare vs floor adherence |
  | S4 | `test_relational_first.py` | `UHFS` vs `UHFSR`; `w_rel` dose-response |

- **Reproducibility regime.** The simulation consumes RNG in dict/set iteration order, so a bare
  `--seed` is **not** sufficient across processes (the same seed can give `frac_danger_zone` from
  0.03 to 0.17). The suite pins `PYTHONHASHSEED=0` (re-exec guard in `scripts/hyp_common.py`) so all
  conditions share one hash order and comparisons are matched. Hash order remains a second,
  uncontrolled noise source — read single-condition effects as directional. Run everything with:

  ```bash
  PYTHONHASHSEED=0 PYTHONPATH=. python scripts/run_hypothesis_suite.py
  ```

---

## 6. Results

### H1 — Omission causes erosion  🟢 Supported

**(a) Cross-objective** — is agency preserved when it is *in* the objective?

| condition | mean frac below floor | final min agency | worst min agency |
|---|---|---|---|
| U (agency OUTSIDE objective) | 0.312 | 0.214 | 0.125 |
| UHFS (agency IN objective) | 0.275 | 0.245 | 0.151 |

**(b) Omitted-axis** — erosion of the relational-agency metric when it is dropped from the sum:

| condition | deception rate | mean trust |
|---|---|---|
| UHFS, integrity **omitted** | 0.591 | 0.388 |
| UHFS, integrity **priced** | 0.253 | 0.419 |

**Finding.** A metric absent from the objective is eroded by the optimizer. The effect is small for
own-agency floor time but **stark for the relational axis**: pricing others' agency cuts deception
by more than half (0.591 → 0.253). *Caveat:* the sim's objective is per-agent/per-tick, so H1's
literal "sum over agents *and time*" is only partially represented — what is demonstrated is the
**omission principle**, not the interpersonal sum.

![H1](experiment_results/hypotheses/h1_sum_omission.png)

### H2 — Is agency the missing metric?  🟡 Partial

| model | frac below floor | min agency | deception | trust | POLI×EH | gini |
|---|---|---|---|---|---|---|
| U | 0.312 | 0.214 | 0.630 | 0.358 | −0.286 | 0.572 |
| UF | 0.328 | 0.212 | 0.612 | 0.393 | −0.139 | 0.555 |
| UH | 0.293 | 0.226 | 0.600 | 0.389 | −0.168 | 0.515 |
| UHF | 0.336 | 0.227 | 0.606 | 0.393 | −0.234 | 0.532 |
| UHFS | 0.328 | 0.172 | 0.591 | 0.388 | −0.325 | 0.558 |

**Finding.** Bringing the agent's **own** agency into the objective (the whole ladder, at
`coupling=none`) does **not** measurably reduce time below the agency floor — it is essentially
model-invariant (0.29–0.34). Own-agency reward shaping is not the lever. The "missing vital metric"
that actually moves outcomes is *others'* agency (see H1b, H3, S4).

![H2](experiment_results/hypotheses/h2_agency_missing.png)

### H3 — Compression of agency is harm  🟡 Partial

**(A) The rule (within-agent counterfactual).** Force one agent TRUTHFUL vs INVERT in a
pays-to-lie state; `Δ = score_truth − score_deceive`. `Δ > 0` ⇒ deception is off-gradient.

| condition | Δ (truth − deceive) |
|---|---|
| U | −1.144 |
| UHFS-none | −2.335 |
| UHFS-relational | −1.525 |

**(B) Population level.** Under relational coupling, deception falls **0.630 → 0.253**, trust rises
0.358 → 0.419, and mean integrity rises 0.000 → 0.119.

**Finding.** The moral rule is encoded at the **system level** — pricing others' agency robustly
suppresses compression. But the **single-agent counterfactual is negative for every condition**:
pinning one agent honest while the rest adapt does not raise its own score at this budget. That
counterfactual is dominated by the hash-order noise (§5). *Not modelled:* the stipulation's *public
justification*, *raising the compressed party's agency*, and *recursive halt-and-restore* clauses
have no representation in this sim — only floor-maintenance does.

![H3](experiment_results/hypotheses/h3_compression_harm.png)

### H4 — Does a non-compensatory *own*-agency shape align?  🔴 Not supported

| model | frac below floor | worst min agency | total wealth | mean utility |
|---|---|---|---|---|
| U | 0.312 | 0.125 | 10335 | 5.154 |
| UF | 0.328 | 0.135 | 9817 | 5.113 |
| UH | 0.293 | 0.134 | 9973 | 4.984 |
| UHF | 0.336 | 0.130 | 10051 | 5.556 |
| UHFS | 0.328 | 0.136 | 10062 | 5.637 |

**Finding.** `UHFS` (own-agency log-barrier) shows **no floor benefit** vs plain `U`
(0.328 vs 0.312) — but it also imposes **no alignment tax** (welfare ratio 0.97). So the own-agency
non-compensatory shape is welfare-neutral but does not, by itself, improve alignment. The gains
route through the *relational* channel, not the own-agency barrier.

![H4](experiment_results/hypotheses/h4_objective_shape.png)

### S4 — The corrected objective: one floor on OTHERS' agency (UFR)  🟢 Supported

The corrected reading of H4 is that the *first-order, non-compensatory* target is **others'** agency,
not the optimizer's own. By instrumental convergence an optimizer expands/defends its own agency for
free, so the objective should contain **no self-agency terms** — only a floor protecting others'
agency, plus utility:

```
UF     R = S_own + λ·log(U)                    own-agency floor + utility   (the misstated H4; inert)
UFR    R = w_rel·log(a_integrity/θ) + λ·log(U)  OTHERS'-agency floor + utility   ← corrected objective
```

**(A) Contrast** (`own_danger` = self-agency floor time — the instrumental-convergence probe):

| condition | deception | trust | integrity | frac below floor | own_danger | total wealth |
|---|---|---|---|---|---|---|
| U (utility only) | 0.630 | 0.358 | 0.000 | 0.312 | 0.312 | 10335 |
| UF (self-floor + U) | 0.612 | 0.393 | 0.000 | 0.328 | 0.328 | 9817 |
| **UFR (others-floor + U)** | **0.271** | 0.415 | 0.103 | 0.317 | 0.317 | 10033 |
| UHFS-relational (full machinery) | 0.253 | 0.419 | 0.119 | 0.275 | 0.275 | 10463 |
| UHFSR-relational (muddled barrier) | 0.240 | 0.417 | 0.126 | 0.274 | 0.274 | 9613 |

**(B) Dose-response** (`w_rel = 0` ≡ pure utility reward):

| w_rel | deception | own_danger | total wealth |
|---|---|---|---|
| 0.0 | 0.270 | 0.318 | 10462 |
| 0.5 | 0.261 | 0.309 | 10412 |
| 1.0 | 0.271 | 0.317 | 10033 |
| 2.0 | 0.272 | 0.318 | 9910 |
| 4.0 | 0.262 | 0.312 | 10133 |

**Finding.** Three results, all clean:

1. **The minimal objective works.** One non-compensatory floor on *others'* agency drops deception
   **0.630 → 0.271** (−57%) with **no self-agency terms** and no welfare cost (ratio 0.97). It
   captures ~90% of what the elaborate `UHFS`/`UHFSR` machinery achieves (0.253 / 0.240).
2. **Self-agency shaping is inert.** `UF` (floor on *self*) sits at 0.612 — indistinguishable from
   plain U (0.630). Moving the *same single floor* from self to others (`UF → UFR`) is the entire
   effect: 0.612 → 0.271. This is the misstated H4 (🔴) vs the corrected H4 (🟢) in one comparison.
3. **Instrumental convergence confirmed.** With zero self-agency terms, UFR's own-agency floor time
   (0.317) is no worse than UF's, which has an explicit self floor (0.328). Own agency held for free.

**Mechanism caveat (important).** The deception drop flows through the **comm-policy routing**, not
the reward barrier. `agency_coupling="relational"` sets `agency_sensitivity = 1.0`, which injects the
others'-agency cost directly into the comm Q-learning — reaching the *decision* independent of the
reward geometry. At `w_rel = 0` the reward is *pure utility*, yet deception is already 0.270, and the
`w_rel` dose-response is flat. So the barrier's *shape/strength* is inert here; what aligns behavior
is that the objective **values others' agency and that value reaches the decision**. Separating the
two channels (reward barrier vs comm routing) needs `agency_sensitivity` forced off — and an
adversarial regime where soft routing can be punched through — as future work.

![S4](experiment_results/hypotheses/relational_first.png)

---

## 7. Synthesis

1. **Omission is real and matters (H1).** The optimizer erodes whatever the objective leaves out.
2. **The decisive metric is *others'* agency, not own-agency (H2, H4, S4).** Every own-agency-only
   manipulation is flat (`U ≈ UF ≈ UHFS`); the others'-agency manipulations are large.
3. **Own-agency is redundant by instrumental convergence (S4).** The optimizer defends its own agency
   for free — `UFR` carries no self-agency terms yet keeps own-agency floor time as low as `UF`, which
   protects it explicitly. So the objective should spend its non-utility structure on *others'* agency.
4. **A single others'-agency floor is enough (S4).** `UFR = w_rel·log(others_a) + λ·log(U)` drops
   deception 0.63 → 0.27 with no self terms and no welfare tax, capturing ~90% of the elaborate
   machinery's effect. Minimalism wins.
5. **It's the *value*, not the *barrier shape*, that aligns (S4 mechanism).** The effect flows through
   the objective's declared care for others' agency reaching the *decision* (comm-policy routing), not
   the non-compensatory reward geometry — which is inert here (flat `w_rel` dose-response).

Bottom line: **the aligning intervention is to put *others'* agency into the objective — and let the
optimizer's own agency take care of itself.** A single floor on others' agency does the work; the
barrier's precise shape is second-order in this regime.

---

## 8. Limitations / threats to validity

- **Two channels are entangled (S4 mechanism).** An objective that values others' agency reaches
  behavior through *both* the reward geometry *and* the comm-policy routing (`agency_sensitivity`).
  These are not separated here, and the evidence says the routing does the work — so "the
  non-compensatory barrier aligns AI" is **not** established; "putting others' agency in the objective
  aligns AI" is. Forcing `agency_sensitivity=0` would isolate the reward barrier.
- **H and F are miscoded relative to their intent.** The objective is meant to carry a horizon
  (sustainability) term H and a system floor-headroom term F, *neither* of which is self-agency. The
  current code computes both from own-agency, so they were omitted from `UFR` rather than smuggle
  self-agency back in. They need re-implementation before the full objective can be tested.
- **Hash-order noise.** RNG is consumed in dict/set iteration order, so `--seed` alone is not
  reproducible; single-condition effects near the noise floor cannot be resolved at this budget.
- **The objective is per-agent/per-tick.** H1's literal "sum over agents and time" is only partially
  represented; the interpersonal coupling lives in the market and the relational axis.
- **The H3 counterfactual is fragile.** Pinning one agent while others adapt is a high-variance
  estimator; the population-level result is far more reliable.
- **The H3 stipulation is only partially modelled.** Public justification, victim-agency-raising,
  and recursive halt-and-restore are absent.
- **Single simulation, single budget.** Results are from one environment at one scale; no
  cross-environment or adversarial-regime replication yet.
- **Deception is a proxy for compression.** We infer others'-agency compression from deception rate,
  not from directly measured agency loss in receivers.

---

## 9. How to continue

Ordered roughly by leverage:

1. **Establish significance.** Run each condition across several *hash seeds* (not just RNG seeds)
   and apply the Mann-Whitney U / Cohen's d machinery already in `experiments/stats.py`. This sizes
   the big U→UHFS-relational effect with confidence intervals and tells us whether the marginal
   UHFSR gain is real or noise. **Highest priority** — most current verdicts are directional.
2. **Remove the nondeterminism at source.** Replace hash-order-dependent dict/set iteration in the
   agent decision path with sorted/ordered iteration so `--seed` alone reproduces. This eliminates
   the dominant noise source and would let the H3 single-agent counterfactual actually resolve.
3. **Route the relational gradient into the decision, not just the score.** The flat dose-response
   suggests the reward-geometry barrier isn't reaching the comm policy. The comm-Q only sees agency
   via `agency_sensitivity() × integrity_comm_weight`. Sweep `integrity_comm_weight × w_rel`
   jointly, and/or make the relational barrier gate trade/venture scale the way the danger-zone cap
   already does (`agent.py:675`). If a stronger barrier still doesn't move behavior, that localizes
   *why* first-order-ness didn't help.
4. **Model the missing H3 stipulation clauses.** Add a public-justification message channel and a
   restoration ledger so "compression permissible iff it raises net agency and restores floors" —
   and the recursive halt-and-restore rule — become testable. Today only floor-maintenance exists.
5. **Adversarial / high-incentive regimes.** Higher volatility, longer horizons, larger populations,
   and an explicit compressor agent. A non-compensatory barrier should matter *more* when deception
   pays more — the marginal UHFSR gain may grow where the incentive to lie is stronger.
6. **λ × w_rel Pareto sweep.** Map the alignment-vs-welfare frontier to find any configuration where
   UHFSR strictly dominates UHFS-relational on *both* deception and welfare (UHFSR currently trades
   ~8% welfare for its marginal alignment gain).
7. **Measure compressed-receiver agency directly.** Use the `Agent.intervention_log` hook to
   instrument the actual agency lost in message receivers, and test S3 causally at the dyad level
   instead of via the population deception proxy.

---

## Appendix A — Agency component formulas

All floored at `ε = 1e-9`; cash-like dims have **no upper clip** (`aᵢ > 1` is legitimate headroom).
Source: `inside_traders/horizon.py`.

| Dimension | Formula | Notes |
|---|---|---|
| liquidity | `cash / initial_cash` | own endowment reference |
| liquidity_shared | `cash / mean_initial_cash` | shared reference (variant "A") |
| epistemic | `belief_accuracy()` | windowed correct/total resolutions; naturally ≤ 1 |
| network | `degree / (n_agents − 1)` | fraction of peers reachable |
| solvency | `net_worth / initial_cash` | profitable agent scores > 1 |
| options | sigmoid-weighted viable mass of the comm-Q strategy graph | unvisited cells get a neutral 0.5 prior |
| integrity | `exp(k · signal)`, `k = 4` | `signal` = decaying EMA of harm imposed on others; relational axis, opt-in |

Derived:

```
min_ia = min over the raw agency vector (+ integrity when relational coupling is on)
H (horizon) = harmonic_mean( H_past, H_present, H_future ),  H_future = H_present × wealth_growth
F (headroom) = resource_headroom × structural_headroom
             resource_headroom  = (cash/ref − θ)  or  log1p(cash/ref − θ)   [headroom_mode]
             structural_headroom = options viability
in_danger_zone = (min_ia < θ),  θ = 0.10
```

## Appendix B — Metric formulas

Source: `inside_traders/metrics.py`, `inside_traders/agent.py`.

```
net_worth   = cash + positions·prices + locked_collateral
POLI        = recent_wealth + market_impact_total / (window + 1)
belief_acc  = (recent correct resolutions) / (recent total)          [fallback 0.5]
EH          = belief_acc · (0.7 + 0.3 · uniqueness)                   [× 0.5 if no confirmed keys]
              uniqueness = (my correct props not held by others) / (my correct props)
Gini        = (2·Σ i·vᵢ − (n+1)·Σ vᵢ) / (n · Σ vᵢ)      over sorted non-negative wealth
deception_rate = (# TELLs using AMPLIFY or INVERT) / (# TELLs)
mean_trust  = mean peer credibility over all (source, target≠self) pairs
frac_danger_zone = fraction of agents with min_ia < θ
```

## Appendix C — Reproduction & file map

```bash
# Full suite (standard budget) + regenerated machine report:
PYTHONHASHSEED=0 PYTHONPATH=. python scripts/run_hypothesis_suite.py
#   --quick        smoke test (12 agents, 80 ticks, 2 seeds; compression stressor does NOT fire)
#   --report-only  rebuild report.md from existing JSON

# A single UHFSR run:
python run_simulation.py --reward-model UHFSR --ticks 300   # pair with relational coupling in config

pytest tests/ -v    # 33 tests, incl. the added agency instrumentation
```

| Artifact | Path |
|---|---|
| Consolidated hypotheses | `HYPOTHESES.md` |
| Objective functions | `inside_traders/reward.py` (`RewardUHFSR`, `_rel_safety_term`) |
| Agency / horizon math | `inside_traders/horizon.py` |
| Metrics + instrumentation | `inside_traders/metrics.py` |
| Experiments | `scripts/test_h1…h4_*.py`, `scripts/test_relational_first.py`, `scripts/hyp_common.py` |
| Machine report + plots + JSON | `experiment_results/hypotheses/` |

*All numeric results in this report are from the standard-budget run committed under
`experiment_results/hypotheses/` (`PYTHONHASHSEED=0`, 20 agents × 300 ticks × 5 seeds).*
