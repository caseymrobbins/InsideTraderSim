# InsideTraderSim — HorizonSim v1

Agent-based simulation of information markets where **epistemic quality (EH) is the
hidden engine of wealth (POLI)**. Complex behaviours — deception, cartels, trust
cascades, Gini asymmetry — emerge naturally from individual utility maximisation.
No strategies are hardcoded.

## Core Thesis

> Information is the currency behind the currency.

Money (POLI) is a claim; its effective deployment depends on epistemic health (EH).
High-POLI agents can move markets and buy signals, reproducing Gini feedback. Value
creation via ventures is positive-sum but harder than extraction/trading.

## Architecture

```
inside_traders/
  config.py        — SimConfig dataclass (all tunable parameters)
  world.py         — Hidden state dynamics, prediction card issuance
  agent.py         — HorizonSim v1 cognitive agent
  belief.py        — BeliefGraph: proposition nodes with strength/confidence
  evidence.py      — Evidence Ledger + EpistemicModel (per-source credibility)
  communication.py — Message types + CommunicationGraph (address book)
  market.py        — OrderBook with market impact
  venture.py       — Bilateral positive-sum contracts
  simulation.py    — Main orchestration loop
  metrics.py       — POLI, EH, Gini, deception rate, trust, compression test
  visualization.py — Matplotlib plot hooks
  pretrain.py      — Solo-world pretraining (no comm, one agent at a time)
  checkpoint.py    — Save / load agent learned state (JSON)
```

## HorizonSim v1 Cognitive Architecture

Each agent carries:

| Component | Description |
|-----------|-------------|
| **Preference Vector** | Fixed utility weights (wealth, influence, security, knowledge, autonomy) drawn from Dirichlet distribution |
| **Belief Graph** | Proposition nodes keyed by `"asset_N_price_tick_T"`. Strength ∈ [-1,1], confidence ∈ [0,1]. Blended via confidence-weighted Bayesian update |
| **Evidence Ledger** | Immutable append-only log. Sources: Observation (prediction card) or Communication (peer message). Statuses: PENDING → CONFIRMED / REFUTED / EXPIRED |
| **Epistemic Model** | Per-source × per-proposition-type credibility score ∈ [0,1]. Updated only on evidence resolution |
| **Address Book** | Sparse directed graph, starts as ring lattice, grows via INTRODUCE messages |
| **Intervention Log** | Extension point for Agency Calculus / Horizon Index overlays |

**Inference cycle (every tick):**
1. Receive prediction cards → Observation Evidence
2. Compose exactly one outgoing message (TELL/ASK/OFFER/INTRODUCE)
3. Receive incoming messages → Communication Evidence
4. Resolve evidence → update EpistemicModel + BeliefGraph
5. Update Intervention view
6. Plan: expected-utility over preferences × beliefs
7. Execute: trade, venture proposal, signal purchase

Beliefs update **only** through evidence. Deception emerges when agents send propositions
with low-confidence beliefs because the utility of misleading exceeds the utility of truth.

## Intended Workflow (Training Before Testing)

Tests and evaluation must only run **after** a training checkpoint exists.
The sequence is:

```
pretrain → load checkpoint → run joint simulation → then run tests
```

### Step 1 — Solo pretraining (no communication)

Each agent is trained in isolation against the world only:
- one agent at a time, separate world per agent
- world observations (prediction cards) only
- trade and buy-signal actions only (no ventures — require counterparty)
- individual P&L as the only reward signal
- no messages sent or received

```bash
python run_simulation.py --pretrain-only --pretrain-ticks 150
# writes:  checkpoints/pretrained.json
```

### Step 2 — Joint simulation with pretrained agents

Load the checkpoint, reset the comm Q-table to a blank slate (so
communication strategy emerges from scratch), then run the full
multi-agent simulation with the market, ventures, and comm channel:

```bash
python run_simulation.py --resume-from checkpoints/pretrained.json
```

### Step 3 — Run tests / evaluation

```bash
pytest tests/ -v
```

Tests verify correctness of code — they are **not** a training loop.

### One-command shortcut (pretrain + joint in sequence)

```bash
python run_simulation.py --pretrain --pretrain-ticks 150 --ticks 300
```

### Skip pretraining (original behaviour)

```bash
python run_simulation.py --skip-pretrain
python run_simulation.py            # identical — skip-pretrain is the default
```

### All pretraining CLI flags

| Flag | Description |
|------|-------------|
| `--pretrain` | Run solo pretraining, then run joint simulation |
| `--pretrain-only` | Run solo pretraining, save checkpoint, exit |
| `--resume-from PATH` | Load checkpoint, run joint simulation (skip pretrain) |
| `--skip-pretrain` | Skip pretrain, run joint simulation directly |
| `--pretrain-ticks N` | Solo ticks per agent (default 100) |
| `--checkpoint-dir DIR` | Directory for checkpoint files (default `checkpoints/`) |

---

## Quick Start

```bash
pip install -r requirements.txt
python run_simulation.py
```

Options:
```
--agents   N      Number of agents (default 30)
--ticks    N      Simulation length (default 300)
--seed     N      RNG seed
--verbose         Log per-tick metrics
--no-plots        Skip matplotlib output
```

## Running the Test Suite

Run tests only after pretraining is complete:

```bash
python run_simulation.py --pretrain-only   # train first
pytest tests/ -v                           # then test
```

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
Occasionally an agent is resilient if their market impact was enough to
overcome bad beliefs — this is expected and interesting.

### 3. Gini Evolution

Wealth inequality rises over time because high-POLI agents buy better signals,
trade more profitably, and attract venture co-investors. Check `wealth_distribution.png`.

### 4. Deception Emergence

No agent is programmed to lie. Deception emerges when an agent's
best-confidence belief to share happens to be wrong, or when
the utility of sending a false proposition (to mislead competition)
exceeds truthful sharing. Monitor `deception_rate` in snapshots.

## Parameter Sweeps

```bash
python run_sweep.py --ticks 200 --agents 20
```

Sweeps over `signal_base_cost × asset_volatility × seed` and writes `sweep_results.csv`.
Higher signal costs → lower mean Gini (information is less concentrated).
Higher volatility → more deception (beliefs go stale faster, errors increase).

## Design Constraints

- **Positive-sum overall**: ventures inject surplus; total wealth grows on average.
- **Creation harder than extraction**: ventures require coordination, collateral, duration.
- **No hardcoded strategies**: cooperation, deception, cartels emerge from utility maximisation.
- **Fully emergent reputation**: EpistemicModel drives trust; no trust score is set directly.
- **Extensible**: `Agent.intervention_log` is the hook for Agency Calculus overlays, Horizon Index computation, and responsibility field analysis.

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
