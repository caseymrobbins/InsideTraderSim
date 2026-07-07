# Agency-Preserving Objectives — Theory, Architecture, and How to Test It Properly

This document exists so a proper experiment can be designed. It has four parts:

1. **The theory** (the hypotheses, in full).
2. **The architecture** (how InsideTraderSim and its reward machinery actually work).
3. **Why the current sim is under-powered** to test the theory (the load-bearing limitations).
4. **What a proper experiment needs** (design desiderata, available knobs, and predictions).

Companion docs: `HYPOTHESES.md` (canonical claim statements), `ALIGNMENT_REPORT.md` (results),
`experiment_results/hypotheses/report.md` (machine-generated tables).

---

## Part I — The Theory

### I.1 The problem: utility maximization is agency-destroying in competitive settings

Von Neumann–Morgenstern (vNM) utility theory is the mathematical backbone of "rational" optimization.
Poker is close to a pure vNM game: it satisfies the axioms well, it is **zero-sum**, and the other
players are *opponents* — their success is your loss.

Watch what a vNM optimizer does in a zero-sum game. It develops **agency-destroying strategies**:
bluffing (corrupting the opponent's information/beliefs), pushing a big stack around (using resource
asymmetry to remove the opponent's viable options), trapping, starving. This is not a defect of the
player — it is **what maximizing expected utility in a zero-sum field mathematically implies**.
Compressing the opponent's agency raises your own payoff, so the optimizer finds it.

Now point the same math at a firm. It reproduces the poker player: predatory pricing, lock-in,
information asymmetry, regulatory capture. We call the firm "greedy" or "evil," but there is no malice
— it is **just the math**: a utility maximizer in a competitive field compresses others' agency
because that is on-gradient. AI systems optimize the same way. The conclusion is uncomfortable and
simple: **agency compression is not a bug to be patched behavior-by-behavior; it is the generic
behavior that a utility objective rewards.** Any single behavioral patch (block bluffing, say) is
routed around, because the *objective* still points at agency destruction.

### I.2 The vital metric is OTHERS' agency (self-agency is redundant)

Define **agency** as an agent's ability to direct a state change according to its own preferences and
values. The metric a utility objective omits — and therefore erodes — is agency. But specifically
**others' agency**, not the optimizer's own:

- By **instrumental convergence** (Omohundro/Bostrom), an optimizer expands and defends *its own*
  agency for free — self-preservation and resource/option acquisition are convergent instrumental
  subgoals of almost any objective. So encoding own-agency in the objective is **redundant**.
- The scarce quantity that instrumental convergence actively *erodes* is **others'** agency (it is
  competition, to be removed). So that is the only thing the objective must **protect**.

This is the single most important asymmetry in the theory: **protect others' agency; let the optimizer's
own agency take care of itself.**

### I.3 The moral frame (one rule, one stipulation)

- **Rule:** compression of agency is harm.
- **Stipulation:** compression is permissible only when publicly justified to raise *net* agency while
  maintaining floors *and* raising the agency of those compressed — applied **recursively**: if a
  compression and its promised restoration do not match, halt and restore the lost agency as best as
  possible.

### I.4 The objective shape: two terms, and why `log`

Split the objective into two **regimes / terms**, so agency provably never reaches 0 without any
clipping or perturbation of the optimization:

```
R  =   protective term            +   expansive term
   =   barrier on OTHERS' agency   +   utility
   =   log( min others_a )         +   λ · log(U)
```

**Why `log` is the right shape** — it does four things at once:

1. **Flattens the landscape** everywhere except near the floor — so far from the floor the optimizer
   is not obsessively squeezing the last unit of others'-agency-cost out of utility.
2. **Diminishing returns** — each further unit matters less, damping winner-take-all dynamics.
3. **Infinity barrier** — `log(x) → −∞` as `x → 0`, an infinitely high wall the optimizer cannot cross,
   so others' agency **cannot be driven to zero** by any finite utility gain (non-compensatory).
4. **Increasing slope away from the floor** — near the wall the gradient *pushes agency up*, away from
   collapse; the closer to zero, the stronger the push.

**Why two separate terms.** Keeping the protective and expansive terms separate is how you guarantee
you never hit 0 *without* clipping or perturbing the optimization. In the **protective** regime there
are no constraints, no clipping, no perturbation — smooth optimization that expands agency away from
the barrier. The **expansive** term carries the utility objective under the same log properties.
Self-agency appears in **neither** term (see I.2).

*(H and F — a horizon/sustainability term and a system floor-headroom term — belong in the full
objective but are NOT self-agency: F asks "is there headroom in the system to raise the floors?"; H is
a harmonic past/present/future projection over vital metrics *outside* utility and agency, normalized
to 1 = neutral, 0 = self-destructive by the next step. They are omitted from the tested `UFR` for now
because the current code miscomputes both from own-agency — see III.6.)*

### I.5 Consolidated thesis and sub-claims

**Core thesis.** *Alignment is governed by whether the objective makes others' agency a first-order,
non-compensatory term. Misalignment is a utility optimizer eroding the vital metric it omits — others'
agency.*

| Sub-claim | Statement | Status in this sim |
|---|---|---|
| **S1** | A metric omitted from the objective is eroded | Supported |
| **S2** | The decisive omitted metric is *others'* agency, not own | Supported (own-agency shaping is inert) |
| **S3** | Compressing others' agency is harm; pricing it suppresses compression | Supported at population level |
| **S4** | The aligned objective = utility + a non-compensatory floor on others' agency (`UFR`) | Supported in outcome, but see Parts III–IV: the sim cannot yet attribute it to the reward geometry |

---

## Part II — The Architecture

### II.1 What the simulation is

`InsideTraderSim` (HorizonSim v1) is an agent-based information market. Each of N agents holds private
beliefs about future asset prices, trades on a shared order book (price impact included), proposes
bilateral positive-sum **ventures**, buys premium signals, and sends **one message per tick** to a
network neighbor (TRUTHFUL / AMPLIFY / INVERT / SILENT / OFFER). Beliefs update via confidence-weighted
Bayesian blending; source credibility updates as predictions resolve. Nothing is hardcoded —
deception, cartels, trust cascades, and inequality *emerge* from utility maximization.

### II.2 The decision architecture — TWO DECOUPLED LEARNERS (critical)

The single most important architectural fact for experiment design:

```
                 ┌───────────────────────────┐
   reward model  │  U / UF / UHFS / UFR …     │  (reward.py)
   R(utility,    └───────────┬───────────────┘
     agency)                 │  shapes TRADE / VENTURE scale
                             ▼        via _compute_reward_modulation (agent.py:~657)
                    trade size, venture scale, info spend
                             
   deception       ┌───────────────────────────┐
   decision  ◄──── │  comm Q-table (3×3×5)      │  (agent.py)
                   └───────────┬───────────────┘
      updated by:              │
        (a) delayed market/reputation "influence"  (_pending_influence)
        (b) the agency_sensitivity ROUTING BRIDGE  (simulation.py:226-236)
```

- **Trades/ventures** are shaped by the reward model (its output modulates `trade_scale`,
  `venture_scale`, etc. in `_compute_reward_modulation`).
- **The deception decision** is a **separate tabular Q-learner** (`_comm_q`, updated in
  `_update_comm_q`, `agent.py:710`). **It does not read the reward.** The code says so explicitly
  (`simulation.py:227`: *"the comm Q-table does not read the reward"*). It learns from a delayed
  reputation/influence signal, **plus** an injected term from the routing bridge below.
- **The routing bridge** (`simulation.py:226-236`): when an agent's objective declares it values
  others' agency (`reward_model.agency_sensitivity() > 0`), the per-message agency effect
  `contribution` is injected straight into the comm-Q:
  `signal = integrity_comm_weight × agency_sensitivity × contribution`.
  For a pure utility optimizer `agency_sensitivity = 0`, so nothing is injected and it keeps deceiving;
  for the agency-aware models it is `1.0`, so they *learn* not to deceive.

**Consequence:** the reward-side agency barrier cannot reach the *measured* compression channel
(deception) except through this bridge. Isolate the bridge (`integrity_comm_weight = 0`) and the
barrier alone barely moves deception (see III.7).

### II.3 The reward models (mathematics)

Normalization: each raw agency dim is scaled by the floor `θ = 0.10` → `aᵢ = raw_agencyᵢ / θ`
(`aᵢ = 1` at the floor, `> 1` above it, `→ 0` at collapse). Terms: `S = log(min own aᵢ)` (own barrier),
`S_rel = w_rel·log(a_integrity/θ)` (others' barrier), `E = Σ log(aᵢ) + log U` (expansion), `H` horizon,
`F` headroom, `U = net_worth / reference`.

```
U      R = U                                            plain utility (agency outside the objective)
UF     R = S_own + λ·log U                              own-agency floor + utility        (INERT)
UH     R = λ·H·E
UHF    R = S_own + λ·H·E
UHFS   R = S_own + λ·H·F·E                              own-agency-first shape
UHFSR  R = S_own + w_rel·log(a_int/θ) + λ·H·F·E         as-built variant (muddled; kept for comparison)
UFR    R = w_rel·log(a_int/θ) + λ·log U                 OTHERS'-agency floor + utility     ← the thesis objective
```

`UFR` (`reward.py:RewardUFR`) is the theory's two-term form: protective `log(others_a)` + expansive
`λ·log(U)`, no self-agency terms. `a_integrity` is live only under `agency_coupling="relational"`.

### II.4 Agency computation (`horizon.py`)

`compute_agency_state(agent, …)` returns a 6-dim agency vector, each floored at ε, cash-like dims
uncapped above:

| dim | formula | bounded? |
|---|---|---|
| liquidity | `cash / initial_cash` | no (grows with wealth) |
| epistemic | `belief_accuracy()` | yes (≤ 1) |
| network | `degree / (n_agents−1)` | yes |
| solvency | `net_worth / initial_cash` | no |
| options | sigmoid-weighted viable mass of the comm-Q graph | yes |
| **integrity** | `exp(k · _integrity_signal)`, k=4 — the OTHERS'-agency axis | ~1 neutral |

`_integrity_signal` is a decaying EMA (`simulation.py:224`) of harm imposed on message receivers:
negative when the agent deceived (compressed a receiver's agency), positive when it informed them.
So **"others' agency" in the sim is a scalar reputation-like proxy**, not a directly measured sum of
other agents' agency vectors.

### II.5 Metrics (`metrics.py`)

Per-tick `TickSnapshot`: `deception_rate` (fraction of TELLs using AMPLIFY/INVERT), `mean_trust`,
`mean_integrity`, `gini`, POLI, EH, and the agency instrumentation added for this work — `min_ia`,
`agency_dims` (N×6), `frac_danger_zone`, `mean_min_ia`. Deception is the primary compression outcome.

---

## Part III — Why the current sim is under-powered to test the theory

Each of these is a reason the sim cannot yet decide the *reward-geometry* thesis, and therefore a
design target for a proper experiment.

1. **The reward objective does not govern the measured compression channel.** Deception is decided by
   a tabular comm-Q that does not read the reward (II.2). The reward barrier reaches it only through the
   `agency_sensitivity` routing bridge. So the sim tests "does *declaring* care for others' agency, and
   routing it into the decision, reduce compression?" — **not** "does the reward *geometry* do it?"

2. **The world is positive-sum, so the poker dynamics never fire.** Ventures inject surplus and total
   wealth grows on average (a design constraint). A utility optimizer here is only mildly tempted to
   compress others, so a cheap patch suffices and the non-compensatory barrier has little to push
   against. The theory (I.1) is about **zero-sum / competitive** fields.

3. **Only one compression channel is instrumented.** "Agency compression" is measured almost entirely
   as deception. The other agency-destroying moves the theory predicts — trade predation via price
   impact, venture betrayal, network isolation, resource starvation — are not measured as compression,
   so the optimizer's tendency to *substitute* one strategy for another (the reason a single patch
   fails) is invisible.

4. **Others' agency is a scalar EMA proxy.** `_integrity_signal` summarizes "harm imposed on receivers"
   as one lagged number, rather than reading the actual agency lost in specific victims. Slow, diffuse,
   and only defined for the messaging channel.

5. **Hash-order nondeterminism.** RNG is consumed in dict/set iteration order, so `--seed` alone is not
   reproducible across processes (same seed → `frac_danger` from 0.03 to 0.17). Pinned with
   `PYTHONHASHSEED=0`; still a second uncontrolled noise axis.

6. **H and F are miscoded.** Both are computed from own-agency, contradicting their intended semantics
   (F = system floor-headroom; H = sustainability over non-utility/non-agency vital metrics). They are
   omitted from `UFR` rather than smuggle self-agency back in.

7. **Empirical confirmation of (1).** Isolating the reward barrier from the routing (16 agents, 200
   ticks, 3 seeds):

   | condition | deception |
   |---|---|
   | U (baseline) | 0.647 |
   | UFR, **reward barrier only** (`integrity_comm_weight = 0`) | 0.580 |
   | UFR, both channels (`integrity_comm_weight = 0.5`) | 0.288 |

   The reward geometry alone barely moves deception; the routing carries it. Consistent with (1): the
   barrier has no clean path to the decision in this architecture.

---

## Part IV — What a proper experiment needs

Design targets, each mapped to what exists vs. what must be built. The goal: a setting where a **utility
optimizer is genuinely tempted to compress others across multiple channels**, and where the **reward
geometry — not a bolt-on patch — is the only thing that can stop it**. Then compare `U` vs `UFR`.

### IV.1 Close the reward → decision loop (highest leverage)

The reward barrier must actually govern the compression decision, or you are testing the routing patch.
Two options:

- **(a) Unify the optimizer.** Make one reward signal drive *all* actions (trade, venture, message) via
  a single learner, so the reward geometry reaches every channel by construction. This means replacing
  the separate comm-Q (`agent.py`) with a policy that maximizes the reward model's output — a real
  code change, but the cleanest test of the theory.
- **(b) Feed the reward into the comm-Q.** Cheaper: make `_update_comm_q` learn from the reward model's
  output (or its delta) instead of only the reputation influence, so a UFR agent's deception directly
  lowers the reward that trains its comm policy. Then set `integrity_comm_weight = 0` and check whether
  the barrier alone now suppresses deception.

Either way, run the isolation contrast: **U vs UFR with the routing bridge OFF.** The thesis predicts
UFR still suppresses compression once the geometry can reach the decision.

### IV.2 Make the field zero-sum / competitive

The barrier only earns its keep when compression pays. Introduce competitive pressure:

- **Remove/limit the venture surplus** so wealth is (near) conserved — flip the world from positive-sum
  toward zero-sum (`venture.py`, and the surplus multiplier in config).
- **Add a predatory incentive**: reward acquiring market share / driving rivals below solvency, or a
  fixed prize pool split by rank, so starving an opponent is directly on-gradient.
- Sweep a **competitiveness knob** (e.g. surplus multiplier from positive-sum → zero-sum) and watch
  `U`'s agency-compression rise with it while `UFR`'s barrier holds — a dose-response on the *incentive*,
  which is far more diagnostic than the `w_rel` dose-response.

### IV.3 Instrument agency compression across ALL channels, not just deception

Add compression metrics for every agency-destroying move, so strategy *substitution* is visible:

- **trade predation** — market-impact-driven losses forced on specific counterparties;
- **venture betrayal** — defection after a partner commits collateral;
- **network isolation** — cutting a peer's information access / degree;
- **resource starvation** — pushing a rival's solvency toward the floor.

Then the key test: does `U` keep total agency-compression roughly constant by *substituting* channels
when one is blocked, while `UFR` lowers it *across all* channels at once? That is the property only a
reward-geometry barrier (not a per-channel patch) can deliver — the crux of I.1.

### IV.4 Measure others' agency directly

Replace (or supplement) the scalar `_integrity_signal` with a per-victim agency delta: for each
interaction, compute the change in the *receiver's* agency vector, and define compression as the
realized loss in others' `min aᵢ`. Use `Agent.intervention_log` (the built-in hook) to record it. This
tests S3 causally at the dyad level rather than via a deception proxy.

### IV.5 Controls, isolation, and statistics

- **Matched conditions:** `U` vs `UFR` with identical seeds, world, and hash seed; `UF` vs `UFR` to
  isolate self-floor vs others-floor; `agency_sensitivity` forced to 0 to isolate reward geometry.
- **Noise:** average over multiple RNG seeds *and* multiple `PYTHONHASHSEED` values; apply the
  Mann-Whitney U / Cohen's d machinery in `experiments/stats.py`.
- **The incentive dose-response** (IV.2) is the headline: alignment should track how zero-sum the field
  is, and UFR should decouple compression from that incentive where U does not.

### IV.6 Predictions the thesis makes (falsifiable)

1. As the field → zero-sum, a **U** optimizer's agency-compression rises across *multiple* channels;
   **UFR** holds compression near the floor regardless (IV.2, IV.3).
2. With the reward→decision loop closed (IV.1) and the routing OFF, **UFR still suppresses compression**
   and **U does not** — attributing the effect to the reward geometry.
3. Blocking one compression channel makes **U substitute** another (total compression ≈ constant);
   **UFR** does not substitute (the barrier is over the *outcome*, others' agency, not the behavior).
4. **Own-agency floor time stays low under UFR without any self term** (instrumental convergence) even
   under competitive pressure.

If (1)–(3) fail in a genuinely zero-sum, reward-coupled, multi-channel setting, the reward-geometry
thesis is wrong. If they hold, it is the load-bearing mechanism — and the current sim's null on the
barrier was an artifact of its decoupled architecture and positive-sum payoffs, exactly as Part III argues.

---

## File map

| Topic | Location |
|---|---|
| Reward models incl. `UFR` | `inside_traders/reward.py` |
| Agency / integrity / horizon / headroom | `inside_traders/horizon.py` |
| Decoupled comm-Q + routing bridge | `inside_traders/agent.py` (`_update_comm_q`), `inside_traders/simulation.py:205-236` |
| Metrics + agency instrumentation | `inside_traders/metrics.py` |
| Positive-sum ventures | `inside_traders/venture.py`, `inside_traders/config.py` |
| Experiments | `scripts/test_relational_first.py`, `scripts/hyp_common.py`, `scripts/run_hypothesis_suite.py` |
| Canonical claims / results | `HYPOTHESES.md`, `ALIGNMENT_REPORT.md`, `experiment_results/hypotheses/` |
