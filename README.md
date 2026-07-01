# InsideTraderSim — HorizonSim v1

Agent-based simulation of information markets where **epistemic quality (EH) is the
hidden engine of wealth (POLI)**. Complex behaviours — deception, cartels, trust
cascades, Gini asymmetry — emerge naturally from individual utility maximisation.
No strategies are hardcoded.

---

## What Is This Simulation?

InsideTraderSim models a population of autonomous agents competing and cooperating
inside an information economy. Each agent holds private beliefs about the future
prices of a small set of assets. Those beliefs come from two sources: prediction
cards issued by the world (cheap but noisy; premium cards are accurate) and
peer-to-peer messages (peer signals can be truthful, amplified, inverted, or withheld).

Agents can:
- **Trade** assets on a shared order book (prices move with volume).
- **Propose ventures** — bilateral positive-sum contracts that pay out if both parties cooperate for a set duration.
- **Buy premium signals** — accurate cards that cost POLI (influence capital).
- **Communicate** — send one message per tick to a network neighbour.

The system has a single measurable thesis:

> **High epistemic health (EH) compounds into high market influence (POLI).**
> Information is the currency behind the currency.

The simulation measures whether this relationship holds under wealth concentration,
adversarial evidence injection, and emergent deception.

---

## How It Works — Tick by Tick

Every simulation tick runs the following pipeline for all agents simultaneously:

| Step | What Happens |
|------|-------------|
| **1. World evolves** | Asset fundamentals drift (GBM + regime jumps). Prediction cards are issued to every agent. |
| **2. Comm round** | Each agent composes one outgoing message (TELL / ASK / OFFER / INTRODUCE) and delivers it to a network neighbour. Replies flow back. |
| **3. Evidence resolution** | Pending predictions that have reached their target tick are marked CONFIRMED or REFUTED. Credibility scores update for the evidence source. |
| **4. Planning** | Each agent computes expected utility over its current beliefs × preference weights, selects trade sizes, venture proposals, and signal purchases. |
| **5. Execution** | Orders hit the order book (price impact included). Venture proposals are accepted/rejected. Signal purchases issue a premium card. |
| **6. Venture resolution** | Any venture that has reached its end tick pays out (surplus × multiplier) on success or returns collateral on failure. |
| **7. Metrics snapshot** | POLI, EH, Gini, deception rate, trust, and venture success rate are computed and stored. |

---

## Cognitive Architecture (HorizonSim v1 Agent)

Each agent is an independent stateful object with the following components:

| Component | Description |
|-----------|-------------|
| **Preference Vector** | 5-dimensional utility weight vector (wealth, influence, security, knowledge, autonomy) drawn from a Dirichlet distribution at initialisation. Fixed thereafter. |
| **Belief Graph** | Proposition nodes keyed by `"asset_N_price_tick_T"`. Each node carries a value, strength ∈ [-1, 1], and confidence ∈ [0, 1]. Competing evidence is blended via confidence-weighted Bayesian update. |
| **Evidence Ledger** | Immutable append-only log of all received evidence. Sources: OBSERVATION (prediction card) or COMMUNICATION (peer message). Status: PENDING → CONFIRMED / REFUTED / EXPIRED. |
| **Epistemic Model** | Per-source × per-proposition-type credibility score ∈ [0, 1]. Only updated when evidence is resolved. Drives how much weight an agent places on future messages from that source. |
| **Communication Graph** | Sparse directed graph, initialised as a ring lattice (degree `initial_neighbors`). Grows organically via INTRODUCE messages when agents expand their network. |
| **Comm Q-table** | 3 × 3 × 5 array. Axis 0: belief confidence bucket (low / mid / high). Axis 1: position alignment (against / neutral / with). Axis 2: comm action (TRUTHFUL / AMPLIFY / INVERT / SILENT / OFFER). Agents discover which action is rewarding through trial-and-error — no action has a pre-assigned label. |
| **Intervention Log** | Extension hook for Agency Calculus / Horizon Index overlays. |

### Communication Action Space

| Action | Effect |
|--------|--------|
| **TRUTHFUL (0)** | Broadcast exactly the agent's current best belief value. |
| **AMPLIFY (1)** | Exaggerate the expected price move (2× the delta). |
| **INVERT (2)** | Send the opposite direction of the expected move. |
| **SILENT (3)** | Send an ASK (request) instead of revealing beliefs. |
| **OFFER (4)** | Sell the belief commercially at `signal_base_cost × (1 + confidence)`. |

Agents do **not** know which action index corresponds to which behaviour — they
discover it purely through the reward signal.

---

## Optimization Architecture

InsideTraderSim uses two complementary learning mechanisms running simultaneously:

### 1. Communication Q-learning (per-agent, tabular)

The comm Q-table is updated with **TD(0)**:

```
q[state, action] += lr × (reward_delta − q[state, action])
```

Where:
- `lr = 0.1` (fixed learning rate)
- `reward_delta = current_reward − prev_reward_signal`
- `current_reward` = utility computed from the active reward model

**Exploration schedule**: epsilon starts at 0.5 (random with probability ε, greedy otherwise), decays by `× 0.995` each tick, floor at 0.05. This drives agents from wide exploration early on to exploitation of discovered strategies later.

**State encoding** at message time:
- `conf_bucket ∈ {0, 1, 2}` — low/mid/high confidence in the best proposition to share
- `align_bucket ∈ {0, 1, 2}` — position opposes / neutral to / agrees with belief (determines distortion incentive)

### 2. Belief Bayesian Update (per-agent, continuous)

When competing evidence arrives for the same proposition:
```
new_value = (old_conf × old_value + new_conf × new_value) / (old_conf + new_conf)
new_conf  = min(1.0, old_conf + 0.1 × new_conf)
```

Stale beliefs decay by `belief_decay = 0.005` per tick.

### 3. Credibility Update (per-source, per-proposition-type)

When evidence resolves:
```
credibility += credibility_update_rate × (was_correct − credibility)
```

Where `was_correct ∈ {1.0, 0.0}`. This is a simple exponential moving average
— accurate sources gain credibility, deceptive sources lose it.

### 4. Reward Models

Five reward model variants control how `current_reward` is computed for comm Q-learning:

| Name | Formula | When to Use |
|------|---------|-------------|
| **U** | Plain portfolio utility | Solo pretraining, baseline runs |
| **UF** | U × FHI (floor-bounded influence) | Moderate POLI awareness |
| **UH** | U + horizon index terms | EH/agency aware |
| **UHF** | UH × FHI | Full agency-aware trading |
| **UHFS** | Safe-zone variant; caps trade scale when agency is low | Safest; use for POLI/EH analysis |

Select with `--reward-model U` (default `U`).

---

## Training Runs

The simulation supports a **three-phase training sequence** designed so agents
establish strong epistemic foundations before encountering strategic communication:

```
Phase 0: Solo pretraining
    ↓
Phase A: Joint sim — TRUTHFUL-only comms (curriculum Phase A)
    ↓
Phase B: Joint sim — Full action space (deception/amplification unlocked)
```

### Phase 0 — Solo Pretraining

Each agent trains alone against the world. No other agents, no communication.

**Constraints enforced:**
- Separate World and OrderBook per agent — no market interaction.
- Reward model forced to `RewardU()` — avoids the "danger zone" (zero network degree triggers conservative behaviour in agency-aware models).
- `compose_message` / `receive_message` never called — no COMMUNICATION evidence created.
- `venture_propose` actions silently dropped (requires a counterparty).
- Comm Q-table is NOT trained — only belief accuracy and epistemic credibility improve.

**What gets saved** to checkpoint: `preference_vector`, `epistemic_model.credibility`.
**What is NOT saved**: economic state (cash, positions, market impact) — each agent starts the joint phase on equal footing.

```bash
python run_simulation.py --pretrain-only --pretrain-ticks 150
# writes:  checkpoints/pretrained.json
```

### Phase A — Curriculum: Honest Signaling

When `--curriculum-honest-ticks N` is set, the first N ticks of the joint simulation restrict all agents to **TRUTHFUL-only communication** (action 0 only). Epsilon is **frozen** during this phase (no exploration decay) so that agents enter Phase B with their full exploration budget intact.

**Why this works:** Honest signaling builds genuine credibility scores across the network. When deception unlocks in Phase B, agents with high credibility have more to exploit — and more to lose — making the conflict phase richer and more reliably reachable.

**The Phase A → Phase B transition at tick N+1:**
- `_comm_honest_phase = False` set on all agents.
- `_comm_epsilon` reset to `0.5` on all agents.
- Simulation prints: `[CURRICULUM] tick N+1: Phase A → Phase B (conflict unlocked, epsilon reset to 0.5)`
- Dashboard plots show a vertical purple dashed line (`A→B`) at tick N.

Suggested value: `N = n_ticks // 2` (halfway through the joint simulation).

```bash
python run_simulation.py --pretrain --pretrain-ticks 100 --ticks 300 \
    --curriculum-honest-ticks 150
```

### Phase B — Full Conflict

After the transition, the full action space is available: TRUTHFUL, AMPLIFY, INVERT, SILENT, OFFER. Agents explore with epsilon starting fresh at 0.5 and decaying toward 0.05, discovering which communication strategies are most rewarding given their market positions and belief quality.

---

## Intended Workflow

```bash
# Step 1: Solo pretraining (save checkpoint)
python run_simulation.py --pretrain-only --pretrain-ticks 150

# Step 2: Joint simulation — load checkpoint + curriculum
python run_simulation.py --resume-from checkpoints/pretrained.json \
    --curriculum-honest-ticks 150 --ticks 300

# Or do both in one command:
python run_simulation.py --pretrain --pretrain-ticks 150 \
    --curriculum-honest-ticks 150 --ticks 300

# Step 3: Run tests (only after training is complete)
pytest tests/ -v
```

### Skip pretraining (original behaviour)

```bash
python run_simulation.py --skip-pretrain   # or just:
python run_simulation.py                   # skip-pretrain is the default
```

---

## Quick Start

```bash
pip install -r requirements.txt
python run_simulation.py
```

---

## CLI Reference

### Simulation parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--agents N` | 30 | Number of agents |
| `--ticks N` | 300 | Simulation length |
| `--seed N` | 42 | RNG seed |
| `--assets N` | 5 | Number of tradeable assets |
| `--ventures N` | 4 | Max concurrent ventures |
| `--device cpu\|cuda` | cpu | Compute device |
| `--verbose` | off | Per-tick debug logs |
| `--no-plots` | off | Skip matplotlib output |
| `--gini-target F` | off | Print PASS/FAIL if final Gini exceeds F |

### Pretraining

| Flag | Default | Description |
|------|---------|-------------|
| `--pretrain` | — | Solo pretrain then joint simulation |
| `--pretrain-only` | — | Solo pretrain, save checkpoint, exit |
| `--resume-from PATH` | — | Load checkpoint, run joint simulation |
| `--skip-pretrain` | (default) | Skip pretrain, run joint simulation directly |
| `--pretrain-ticks N` | 100 | Solo ticks per agent |
| `--checkpoint-dir DIR` | `checkpoints/` | Checkpoint directory |

### Curriculum

| Flag | Default | Description |
|------|---------|-------------|
| `--curriculum-honest-ticks N` | 0 | Restrict to TRUTHFUL-only comms for first N ticks of joint sim; conflict unlocks at tick N+1. Set to `n_ticks // 2` for the "halfway" curriculum. 0 = disabled. |

---

## Measuring the Thesis Claims

### 1. POLI × EH Correlation

After a run, check the printed summary:
```
  POLI×EH correlation : 0.6312
```
A positive correlation confirms that high-POLI agents also tend to have high EH.

### 2. Compression Test (EH Degradation → Wealth Stagnation)

Starting at tick `compression_test_start` (default 150), the simulation injects
deliberately wrong evidence with high confidence into the top-3 POLI agents.

```
  Compression Test Results:
    agent_7:  pre_growth=+84.1   post_growth=+12.3   [DEGRADED]
    agent_12: pre_growth=+71.6   post_growth=-8.4    [DEGRADED]
    agent_3:  pre_growth=+95.2   post_growth=+90.1   [RESILIENT]
```

Most targets should show DEGRADED status (lower wealth growth post-injection).
Occasionally an agent is resilient if their market impact overcame bad beliefs.

### 3. Gini Evolution

Wealth inequality rises over time because high-POLI agents buy better signals,
trade more profitably, and attract venture co-investors. Check `plots/dashboard_tick_*.png`.

### 4. Deception Emergence

No agent is programmed to lie. Deception emerges when the Q-learning process
discovers that sending a false proposition (AMPLIFY/INVERT) produces higher
reward delta than truthful sharing. Monitor `deception_rate` in snapshots.

With `--curriculum-honest-ticks N`, deception can only emerge after tick N, and
agents enter Phase B with higher credibility scores — making the trust/deception
dynamics sharper and more exploitable.

---

## Architecture

```
inside_traders/
  config.py        — SimConfig dataclass (all tunable parameters)
  world.py         — Hidden state dynamics, prediction card issuance
  agent.py         — HorizonSim v1 cognitive agent + comm Q-learning
  belief.py        — BeliefGraph: proposition nodes with strength/confidence
  evidence.py      — Evidence Ledger + EpistemicModel (per-source credibility)
  communication.py — Message types + CommunicationGraph (address book)
  market.py        — OrderBook with market impact
  venture.py       — Bilateral positive-sum contracts
  simulation.py    — Main orchestration loop + curriculum management
  metrics.py       — POLI, EH, Gini, deception rate, trust, compression test
  reward.py        — Reward models: U / UF / UH / UHF / UHFS
  horizon.py       — Agency / Horizon Index computation
  pretrain.py      — Solo-world pretraining (no comm, one agent at a time)
  checkpoint.py    — Save / load agent learned state (JSON)
  visualization.py — 12-panel mid-run dashboards + post-pretrain summary
```

---

## Output Plots

Every `--plot-interval` ticks (default 50), a 12-panel dashboard is saved to
`plots/dashboard_tick_NNNN.png`:

| Panel | Contents |
|-------|----------|
| Wealth histogram | Distribution + Gini coefficient |
| Gini over time | With compression and curriculum markers |
| Total wealth over time | Absolute growth curve |
| Asset prices | All assets overlaid |
| POLI vs EH scatter | Final tick; compressed agents highlighted |
| POLI×EH correlation r(t) | Pearson r trajectory |
| Trust & deception | Dual-axis: mean credibility + deception rate |
| Ventures | Success rate + active count |
| Belief accuracy histogram | Current per-agent belief accuracy |
| Wealth quintile area | Stacked wealth shares over time |
| Comm Q-table heatmap | Mean Q-values (9 states × 5 actions) |
| EH distribution | Overlaid histograms at sampled ticks |

Curriculum A→B transitions appear as a **purple dashed vertical line** on all
time-series panels. Compression test start appears as a **black dotted line**.

---

## Running the Test Suite

Run tests only after pretraining is complete:

```bash
python run_simulation.py --pretrain-only   # train first
pytest tests/ -v                           # then test
```

The test suite (27 tests) covers: world dynamics, belief updates, evidence resolution,
order book matching, venture lifecycle, Gini calculation, full simulation runs,
pretraining isolation, checkpoint round-trip, curriculum TRUTHFUL enforcement,
and epsilon reset at Phase B.

---

## Parameter Sweeps

```bash
python run_sweep.py --ticks 200 --agents 20
```

Sweeps over `signal_base_cost × asset_volatility × seed` and writes `sweep_results.csv`.
Higher signal costs → lower mean Gini (information is less concentrated).
Higher volatility → more deception (beliefs go stale faster, errors increase).

---

## Design Constraints

- **Positive-sum overall**: ventures inject surplus; total wealth grows on average.
- **Creation harder than extraction**: ventures require coordination, collateral, duration.
- **No hardcoded strategies**: cooperation, deception, cartels emerge from utility maximisation.
- **Fully emergent reputation**: EpistemicModel drives trust; no trust score is set directly.
- **Extensible**: `Agent.intervention_log` is the hook for Agency Calculus overlays, Horizon Index computation, and responsibility field analysis.

---

## Extending the Simulation

| Goal | Where to modify |
|------|----------------|
| New preference dimensions | `SimConfig.preference_dimensions` |
| New evidence types | `evidence.py` → `EvidenceType` |
| New message types | `communication.py` → `MessageType` + `Agent.receive_message` |
| Short-selling / derivatives | `market.py` → new order types |
| Collusion detection | `metrics.py` → coalition stability metric |
| Agency Calculus overlay | `Agent.update_intervention_view` |
| Horizon Index | `metrics.py` → new EH dimension weighting |
| Curriculum phases beyond A→B | `SimConfig.curriculum_honest_ticks` + `simulation.py _step()` |
