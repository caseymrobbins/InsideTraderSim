"""Visualization hooks for InsideTraderSim metrics."""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, List, Optional
import numpy as np

if TYPE_CHECKING:
    from .metrics import MetricsCollector, TickSnapshot
    from .simulation import Simulation
    from .agent import Agent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _poli_eh_corr_series(snaps: List["TickSnapshot"]) -> List[float]:
    """Pearson r between POLI and EH at each snapshot."""
    out = []
    for s in snaps:
        if len(s.poli) >= 2:
            r = float(np.corrcoef(s.poli, s.eh)[0, 1])
        else:
            r = float("nan")
        out.append(r)
    return out


def _wealth_quintile_shares(snaps: List["TickSnapshot"]) -> np.ndarray:
    """
    Return (len(snaps), 5) array of wealth shares for each quintile
    [bottom-20%, Q2, Q3, Q4, top-20%] at each snapshot.
    """
    rows = []
    for s in snaps:
        w = np.sort(np.maximum(s.wealth, 0.0))
        n = len(w)
        total = w.sum()
        if total <= 0 or n < 5:
            rows.append([0.2] * 5)
            continue
        q = max(1, n // 5)
        shares = [
            float(w[i * q: (i + 1) * q].sum() / total) for i in range(4)
        ]
        shares.append(float(w[4 * q:].sum() / total))
        rows.append(shares)
    return np.array(rows)


# ---------------------------------------------------------------------------
# Stand-alone plot functions (used by run_all_plots)
# ---------------------------------------------------------------------------

def plot_wealth_distribution(metrics: "MetricsCollector", title: str = "Wealth Distribution") -> None:
    """Lorenz curve + Gini evolution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping wealth distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    if metrics.snapshots:
        final = metrics.snapshots[-1]
        sorted_w = np.sort(final.wealth)
        cum_w = np.cumsum(sorted_w) / sorted_w.sum()
        n = len(cum_w)
        x = np.arange(1, n + 1) / n
        axes[0].plot([0] + list(x), [0] + list(cum_w), label="Lorenz curve")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect equality")
        axes[0].set_xlabel("Cumulative agents (sorted by wealth)")
        axes[0].set_ylabel("Cumulative wealth share")
        axes[0].set_title(f"Lorenz Curve (Gini={final.gini:.3f})")
        axes[0].legend()

    ticks = [s.tick for s in metrics.snapshots]
    ginis = [s.gini for s in metrics.snapshots]
    axes[1].plot(ticks, ginis, color="crimson")
    axes[1].set_xlabel("Tick")
    axes[1].set_ylabel("Gini Coefficient")
    axes[1].set_title("Wealth Inequality Over Time")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("wealth_distribution.png", dpi=120)
    plt.close(fig)
    print("[viz] Saved: wealth_distribution.png")


def plot_poli_eh(metrics: "MetricsCollector") -> None:
    """POLI × EH product over time + scatter + correlation trajectory."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping POLI×EH plot.")
        return

    snaps = metrics.snapshots
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("POLI × EH (Epistemic Health) Analysis")

    ticks = [s.tick for s in snaps]
    axes[0].plot(ticks, [s.poli_eh.mean() for s in snaps], color="steelblue")
    axes[0].set_xlabel("Tick")
    axes[0].set_ylabel("Mean POLI × EH")
    axes[0].set_title("Mean POLI×EH Over Time")
    axes[0].grid(alpha=0.3)

    if snaps:
        final = snaps[-1]
        axes[1].scatter(final.poli, final.eh, alpha=0.6, color="darkorange",
                        edgecolors="k", linewidths=0.5)
        axes[1].set_xlabel("POLI Score")
        axes[1].set_ylabel("Epistemic Health (EH)")
        axes[1].set_title("POLI vs EH (final tick)")
        if len(final.poli) >= 2:
            corr = float(np.corrcoef(final.poli, final.eh)[0, 1])
            axes[1].annotate(f"r = {corr:.3f}", xy=(0.05, 0.92),
                             xycoords="axes fraction", fontsize=11, color="navy")

    corrs = _poli_eh_corr_series(snaps)
    axes[2].plot(ticks, corrs, color="purple", linewidth=1.5)
    axes[2].axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    axes[2].set_xlabel("Tick")
    axes[2].set_ylabel("Pearson r (POLI, EH)")
    axes[2].set_title("POLI×EH Correlation Trajectory")
    axes[2].set_ylim(-1, 1)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("poli_eh.png", dpi=120)
    plt.close(fig)
    print("[viz] Saved: poli_eh.png")


def plot_trust_evolution(metrics: "MetricsCollector") -> None:
    """Mean trust / credibility over time + deception rate."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping trust plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Trust & Deception Dynamics")

    ticks = [s.tick for s in metrics.snapshots]
    trust = [s.mean_trust for s in metrics.snapshots]
    deception = [s.deception_rate for s in metrics.snapshots]

    axes[0].plot(ticks, trust, color="teal", label="Mean trust")
    axes[0].set_xlabel("Tick")
    axes[0].set_ylabel("Mean credibility score")
    axes[0].set_title("Inter-Agent Trust Over Time")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ticks, deception, color="firebrick", label="Deception rate")
    axes[1].set_xlabel("Tick")
    axes[1].set_ylabel("Fraction of TELLs refuted")
    axes[1].set_title("Deception Rate Over Time")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("trust_deception.png", dpi=120)
    plt.close(fig)
    print("[viz] Saved: trust_deception.png")


def plot_belief_accuracy(agents: List["Agent"], title: str = "Belief Accuracy Distribution") -> None:
    """Histogram of per-agent belief accuracy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping belief accuracy plot.")
        return

    accuracies = [ag.belief_accuracy() for ag in agents]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(accuracies, bins=15, color="mediumslateblue", edgecolor="white")
    ax.set_xlabel("Belief Accuracy (fraction of evidence confirmed)")
    ax.set_ylabel("Number of Agents")
    ax.set_title(title)
    ax.axvline(np.mean(accuracies), color="red", linestyle="--",
               label=f"Mean: {np.mean(accuracies):.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("belief_accuracy.png", dpi=120)
    plt.close(fig)
    print("[viz] Saved: belief_accuracy.png")


def plot_asset_prices(metrics: "MetricsCollector") -> None:
    """Asset price trajectories over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping asset price plot.")
        return

    if not metrics.snapshots:
        return

    n_assets = len(metrics.snapshots[0].asset_prices)
    ticks = [s.tick for s in metrics.snapshots]
    prices = np.array([s.asset_prices for s in metrics.snapshots])

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_assets):
        ax.plot(ticks, prices[:, i], alpha=0.8, label=f"Asset {i}")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Price")
    ax.set_title("Asset Price Trajectories")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("asset_prices.png", dpi=120)
    plt.close(fig)
    print("[viz] Saved: asset_prices.png")


def plot_compression_test(
    metrics: "MetricsCollector",
    compression_targets: List[str],
    test_start_tick: int,
) -> None:
    """Wealth trajectories of compression-test agents vs. system average."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping compression test plot.")
        return

    if not metrics.snapshots:
        return

    ticks = [s.tick for s in metrics.snapshots]
    system_avg = [s.wealth.mean() for s in metrics.snapshots]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ticks, system_avg, "k--", alpha=0.5, linewidth=1.5, label="System average")
    ax.axvline(test_start_tick, color="red", linestyle=":", alpha=0.7, label="Compression start")

    colors = ["crimson", "darkorange", "purple"]
    for i, aid in enumerate(compression_targets):
        traj = metrics.wealth_trajectory(aid)
        snap_ticks = [s.tick for s in metrics.snapshots if aid in s.agent_ids]
        if traj and snap_ticks:
            ax.plot(snap_ticks, traj, color=colors[i % len(colors)],
                    linewidth=2, label=f"{aid} (compressed)")

    ax.set_xlabel("Tick")
    ax.set_ylabel("Wealth")
    ax.set_title("Compression Test: EH Degradation Effect on Wealth")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("compression_test.png", dpi=120)
    plt.close(fig)
    print("[viz] Saved: compression_test.png")


def plot_venture_vs_trading(metrics: "MetricsCollector") -> None:
    """Venture success rate vs active count over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping venture plot.")
        return

    ticks = [s.tick for s in metrics.snapshots]
    success = [s.venture_success_rate for s in metrics.snapshots]
    n_ventures = [s.n_active_ventures for s in metrics.snapshots]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ticks, success, color="seagreen")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Tick")
    axes[0].set_ylabel("Cumulative success rate")
    axes[0].set_title("Venture Success Rate")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ticks, n_ventures, color="navy")
    axes[1].set_xlabel("Tick")
    axes[1].set_ylabel("Active ventures")
    axes[1].set_title("Active Ventures Over Time")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("ventures.png", dpi=120)
    plt.close(fig)
    print("[viz] Saved: ventures.png")


def plot_wealth_quintiles(metrics: "MetricsCollector") -> None:
    """Stacked area of wealth shares by quintile over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping quintile plot.")
        return

    snaps = metrics.snapshots
    if not snaps:
        return

    ticks = [s.tick for s in snaps]
    shares = _wealth_quintile_shares(snaps)

    labels = ["Bottom 20%", "Q2", "Q3", "Q4", "Top 20%"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(ticks, shares.T, labels=labels, colors=colors, alpha=0.78)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Wealth share")
    ax.set_title("Wealth Distribution by Quintile Over Time")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("wealth_quintiles.png", dpi=120)
    plt.close(fig)
    print("[viz] Saved: wealth_quintiles.png")


def plot_comm_strategy_heatmap(agents: List["Agent"]) -> None:
    """
    Mean comm Q-table values aggregated across all agents.

    Rows  = 9 states (3 confidence × 3 alignment, flattened)
    Cols  = 5 actions (TRUTHFUL / AMPLIFY / INVERT / SILENT / OFFER)
    Color = mean Q-value; brighter = more rewarding in that (state, action) pair.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping Q-table heatmap.")
        return

    all_q = np.array([ag._comm_q for ag in agents])   # (N, 3, 3, 5)
    mean_q = all_q.mean(axis=0)                        # (3, 3, 5)
    q_2d = mean_q.reshape(9, 5)                        # (states, actions)

    state_labels = [
        f"C{c} A{a}"
        for c in ("lo", "mid", "hi")
        for a in ("vs", "neu", "with")
    ]
    action_labels = ["TRUTHFUL", "AMPLIFY", "INVERT", "SILENT", "OFFER"]

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(abs(q_2d).max(), 0.01)
    im = ax.imshow(q_2d, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Mean Q-value")
    ax.set_xticks(range(5))
    ax.set_xticklabels(action_labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(9))
    ax.set_yticklabels(state_labels, fontsize=9)
    ax.set_xlabel("Communication action")
    ax.set_ylabel("State (confidence × alignment)")
    ax.set_title(
        "Comm Strategy Q-table  (mean across all agents)\n"
        "Blue = rewarding; Red = penalised"
    )
    for r in range(9):
        for c in range(5):
            ax.text(c, r, f"{q_2d[r, c]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    plt.tight_layout()
    plt.savefig("comm_strategy_heatmap.png", dpi=120)
    plt.close(fig)
    print("[viz] Saved: comm_strategy_heatmap.png")


# ---------------------------------------------------------------------------
# Pretraining progress dashboard
# ---------------------------------------------------------------------------

def plot_pretrain_dashboard(agents: List["Agent"], plot_dir: str = "plots") -> str:
    """
    Save a post-pretraining summary dashboard.

    Panels (2×3):
      [0,0] All agents' solo wealth trajectories
      [0,1] Per-agent wealth at end of pretraining (bar chart)
      [0,2] Distribution of final belief accuracy
      [1,0] Distribution of self-credibility (epistemic model)
      [1,1] Mean wealth trajectory across all agents
      [1,2] Belief accuracy vs initial wealth (scatter)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        return "[viz] matplotlib not installed; skipping pretrain dashboard."

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f"Pretraining Summary  —  {len(agents)} agents  "
        f"({len(agents[0].wealth_history)} solo ticks each)",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    # ── [0,0] Wealth trajectories ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    palette = plt.cm.tab20(np.linspace(0, 1, len(agents)))
    for ag, col in zip(agents, palette):
        ax.plot(ag.wealth_history, color=col, alpha=0.55, linewidth=0.9)
    mean_w = np.mean([ag.wealth_history for ag in agents], axis=0)
    ax.plot(mean_w, color="black", linewidth=2, label="mean", zorder=5)
    ax.set_title("Solo Wealth Trajectories", fontsize=10)
    ax.set_xlabel("Pretrain tick")
    ax.set_ylabel("Wealth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # ── [0,1] Final wealth bar chart ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    final_ws = [ag.wealth_history[-1] for ag in agents]
    ids = [ag.agent_id.replace("agent_", "") for ag in agents]
    bar_cols = ["steelblue" if w >= np.median(final_ws) else "salmon" for w in final_ws]
    ax.bar(ids, final_ws, color=bar_cols, edgecolor="white", linewidth=0.5)
    ax.axhline(np.mean(final_ws), color="k", linestyle="--", linewidth=1,
               label=f"mean={np.mean(final_ws):.0f}")
    ax.set_title("Final Wealth per Agent", fontsize=10)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Wealth")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=6, rotation=60)
    ax.grid(alpha=0.2, axis="y")

    # ── [0,2] Belief accuracy distribution ─────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    accs = [ag.belief_accuracy() for ag in agents]
    ax.hist(accs, bins=12, color="mediumslateblue", edgecolor="white", alpha=0.85)
    ax.axvline(float(np.mean(accs)), color="red", linestyle="--", linewidth=1.2,
               label=f"mean={np.mean(accs):.2f}")
    ax.set_title("Belief Accuracy Distribution", fontsize=10)
    ax.set_xlabel("Fraction of evidence confirmed")
    ax.set_ylabel("Agents")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # ── [1,0] Self-credibility distribution ────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    self_creds = [
        ag.epistemic_model.get("self", "price", default=0.5)
        for ag in agents
    ]
    ax.hist(self_creds, bins=12, color="darkorange", edgecolor="white", alpha=0.85)
    ax.axvline(float(np.mean(self_creds)), color="red", linestyle="--", linewidth=1.2,
               label=f"mean={np.mean(self_creds):.2f}")
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.7, label="prior=0.5")
    ax.set_title("Self-Credibility Distribution\n(epistemic model, 'self' source)", fontsize=10)
    ax.set_xlabel("Credibility score")
    ax.set_ylabel("Agents")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # ── [1,1] Mean wealth trajectory with ±1σ band ─────────────────────
    ax = fig.add_subplot(gs[1, 1])
    all_w = np.array([ag.wealth_history for ag in agents])
    mn = all_w.mean(axis=0)
    sd = all_w.std(axis=0)
    x = np.arange(len(mn))
    ax.fill_between(x, mn - sd, mn + sd, alpha=0.25, color="steelblue", label="±1σ")
    ax.plot(x, mn, color="steelblue", linewidth=2, label="mean wealth")
    ax.set_title("Mean Wealth ± Std Dev", fontsize=10)
    ax.set_xlabel("Pretrain tick")
    ax.set_ylabel("Wealth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # ── [1,2] Belief accuracy vs initial wealth scatter ─────────────────
    ax = fig.add_subplot(gs[1, 2])
    init_ws = [ag._initial_cash for ag in agents]
    ax.scatter(init_ws, accs, color="teal", alpha=0.7, edgecolors="k", linewidths=0.5, s=55)
    if len(init_ws) >= 2:
        corr = float(np.corrcoef(init_ws, accs)[0, 1])
        ax.annotate(f"r = {corr:.3f}", xy=(0.05, 0.90), xycoords="axes fraction",
                    fontsize=10, color="navy")
    ax.set_title("Belief Accuracy vs Initial Wealth", fontsize=10)
    ax.set_xlabel("Initial wealth (endowment)")
    ax.set_ylabel("Belief accuracy")
    ax.grid(alpha=0.2)

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "pretrain_dashboard.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Mid-run dashboard (called every plot_interval ticks by Simulation)
# ---------------------------------------------------------------------------

def plot_midrun_dashboard(sim: "Simulation", tick: int, plot_dir: str = "plots") -> str:
    """
    12-panel mid-run dashboard saved every plot_interval ticks.

    Layout (4 rows × 3 cols):
      Row 0: Wealth histogram    | Gini over time          | Total wealth over time
      Row 1: Asset prices        | POLI vs EH scatter      | POLI×EH correlation r(t)
      Row 2: Trust + deception   | Ventures                | Belief accuracy histogram
      Row 3: Wealth quintile area| Comm Q-table heatmap    | EH distribution over time
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        return "[viz] matplotlib not installed; skipping mid-run dashboard."

    m = sim.metrics
    snaps = m.snapshots
    if not snaps:
        return ""

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(
        f"InsideTraderSim — HorizonSim v1 │ Tick {tick}/{sim.cfg.n_ticks}  "
        f"(device={sim.device})",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)

    ticks_x = [s.tick for s in snaps]
    final = snaps[-1]
    compressed_ids = set(sim._compression_targets)
    cht = sim.cfg.curriculum_honest_ticks  # 0 = disabled

    def _add_curriculum_vline(ax_target, label: bool = True) -> None:
        """Overlay Phase A/B boundary if curriculum is active."""
        if cht > 0:
            ax_target.axvline(
                cht, color="darkorchid", linestyle="--", linewidth=1.2,
                alpha=0.75, label="A→B" if label else None,
            )

    # ── [0,0] Wealth histogram ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(final.wealth, bins=max(8, sim.cfg.n_agents // 4),
            color="steelblue", edgecolor="white", alpha=0.85)
    mean_w = float(np.mean(final.wealth))
    ax.axvline(mean_w, color="red", linestyle="--", linewidth=1.2,
               label=f"mean={mean_w:.0f}")
    ax.set_title(f"Wealth Distribution  (Gini={final.gini:.3f})", fontsize=10)
    ax.set_xlabel("Wealth")
    ax.set_ylabel("Agents")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── [0,1] Gini over time ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ticks_x, [s.gini for s in snaps], color="crimson", linewidth=1.5)
    _add_curriculum_vline(ax)
    if sim.cfg.compression_test_start <= tick:
        ax.axvline(sim.cfg.compression_test_start, color="black",
                   linestyle=":", linewidth=1, alpha=0.6, label="compression")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    ax.set_title("Gini Coefficient Over Time", fontsize=10)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Gini")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

    # ── [0,2] Total wealth over time ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    total_w = [s.total_wealth for s in snaps]
    ax.plot(ticks_x, total_w, color="seagreen", linewidth=1.5)
    _add_curriculum_vline(ax)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    growth = 100 * (total_w[-1] / total_w[0] - 1) if total_w[0] > 0 else 0.0
    ax.set_title(f"Total Wealth  ({growth:+.1f}% from tick {snaps[0].tick})", fontsize=10)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Total wealth")
    ax.grid(alpha=0.25)

    # ── [1,0] Asset prices ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    n_assets = len(snaps[0].asset_prices)
    prices_mat = np.array([s.asset_prices for s in snaps])
    colors_p = plt.cm.tab10(np.linspace(0, 1, n_assets))
    for i in range(n_assets):
        ax.plot(ticks_x, prices_mat[:, i], color=colors_p[i],
                alpha=0.85, linewidth=1.2, label=f"A{i}")
    ax.set_title("Asset Prices", fontsize=10)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Price")
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)

    # ── [1,1] POLI vs EH scatter ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    colors_sc = ["crimson" if aid in compressed_ids else "steelblue"
                 for aid in final.agent_ids]
    ax.scatter(final.poli, final.eh, c=colors_sc, alpha=0.7,
               edgecolors="k", linewidths=0.4, s=55)
    if len(final.poli) >= 2:
        corr = float(np.corrcoef(final.poli, final.eh)[0, 1])
        ax.annotate(f"r = {corr:.3f}", xy=(0.05, 0.90), xycoords="axes fraction",
                    fontsize=10, color="navy")
    if compressed_ids:
        ax.scatter([], [], c="crimson", label="compressed",
                   s=50, edgecolors="k", linewidths=0.4)
        ax.legend(fontsize=8)
    ax.set_title("POLI vs Epistemic Health (EH)", fontsize=10)
    ax.set_xlabel("POLI score")
    ax.set_ylabel("EH")
    ax.grid(alpha=0.25)

    # ── [1,2] POLI×EH correlation trajectory ───────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    corrs = _poli_eh_corr_series(snaps)
    ax.plot(ticks_x, corrs, color="purple", linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    _add_curriculum_vline(ax)
    if sim.cfg.compression_test_start <= tick:
        ax.axvline(sim.cfg.compression_test_start, color="red",
                   linestyle=":", linewidth=1, alpha=0.6, label="compression")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    ax.set_title("POLI×EH Correlation  r(t)", fontsize=10)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.25)

    # ── [2,0] Trust + deception ─────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    trust_line, = ax.plot(ticks_x, [s.mean_trust for s in snaps],
                          color="teal", linewidth=1.5, label="mean trust")
    ax2 = ax.twinx()
    decept_line, = ax2.plot(ticks_x, [s.deception_rate for s in snaps],
                             color="firebrick", linewidth=1.2, linestyle="--",
                             label="deception rate")
    _add_curriculum_vline(ax, label=False)
    curriculum_proxy = (
        [plt.Line2D([0], [0], color="darkorchid", linestyle="--", lw=1.2, label="A→B")]
        if cht > 0 else []
    )
    ax.set_title("Trust & Deception", fontsize=10)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Mean trust", color="teal")
    ax2.set_ylabel("Deception rate", color="firebrick")
    ax.legend([trust_line, decept_line] + curriculum_proxy,
              [trust_line.get_label(), decept_line.get_label()]
              + (["A→B"] if cht > 0 else []),
              fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

    # ── [2,1] Ventures ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(ticks_x, [s.venture_success_rate for s in snaps],
            color="seagreen", linewidth=1.5, label="success rate")
    ax.set_ylim(0, 1)
    ax2b = ax.twinx()
    ax2b.fill_between(ticks_x, [s.n_active_ventures for s in snaps],
                      alpha=0.2, color="navy")
    ax2b.plot(ticks_x, [s.n_active_ventures for s in snaps],
              color="navy", linewidth=1.0, label="active ventures")
    ax.set_title("Ventures", fontsize=10)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Success rate", color="seagreen")
    ax2b.set_ylabel("Active ventures", color="navy")
    ax.legend(
        [plt.Line2D([0], [0], color="seagreen", lw=1.5),
         plt.Line2D([0], [0], color="navy", lw=1.5)],
        ["success rate", "active ventures"], fontsize=8,
    )
    ax.grid(alpha=0.2)

    # ── [2,2] Belief accuracy histogram (current state) ─────────────────
    ax = fig.add_subplot(gs[2, 2])
    accs = [ag.belief_accuracy() for ag in sim.agents]
    ax.hist(accs, bins=max(6, sim.cfg.n_agents // 5),
            color="mediumslateblue", edgecolor="white", alpha=0.85)
    mean_acc = float(np.mean(accs))
    ax.axvline(mean_acc, color="red", linestyle="--", linewidth=1.2,
               label=f"mean={mean_acc:.2f}")
    ax.set_title(f"Belief Accuracy Distribution  (tick {tick})", fontsize=10)
    ax.set_xlabel("Fraction of evidence confirmed")
    ax.set_ylabel("Agents")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── [3,0] Wealth quintile stacked area ──────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    shares = _wealth_quintile_shares(snaps)
    q_labels = ["Bottom 20%", "Q2", "Q3", "Q4", "Top 20%"]
    q_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    ax.stackplot(ticks_x, shares.T, labels=q_labels, colors=q_colors, alpha=0.75)
    _add_curriculum_vline(ax, label=True)
    ax.set_title("Wealth Shares by Quintile", fontsize=10)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Wealth share")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.2)

    # ── [3,1] Comm Q-table heatmap (mean across agents) ─────────────────
    ax = fig.add_subplot(gs[3, 1])
    all_q = np.array([ag._comm_q for ag in sim.agents])   # (N, 3, 3, 5)
    mean_q = all_q.mean(axis=0).reshape(9, 5)              # (9 states, 5 actions)
    vmax = max(float(np.abs(mean_q).max()), 0.01)
    im = ax.imshow(mean_q, aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Mean Q", shrink=0.85)
    state_labels = [
        f"C{c} A{a}"
        for c in ("lo", "mid", "hi")
        for a in ("vs", "neu", "with")
    ]
    ax.set_yticks(range(9))
    ax.set_yticklabels(state_labels, fontsize=7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["TRUTH", "AMP", "INV", "SIL", "OFFER"],
                       rotation=20, ha="right", fontsize=8)
    ax.set_title("Comm Strategy Q-table\n(mean across all agents)", fontsize=10)
    for r in range(9):
        for c in range(5):
            ax.text(c, r, f"{mean_q[r, c]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")

    # ── [3,2] EH distribution over time (sampled snapshots as overlaid histograms)
    ax = fig.add_subplot(gs[3, 2])
    n_snaps = len(snaps)
    sample_idx = sorted(set([0, n_snaps // 4, n_snaps // 2, 3 * n_snaps // 4, n_snaps - 1]))
    sample_idx = [i for i in sample_idx if i < n_snaps]
    cmap_t = plt.cm.viridis(np.linspace(0.15, 0.9, len(sample_idx)))
    for color, si in zip(cmap_t, sample_idx):
        s = snaps[si]
        ax.hist(s.eh, bins=max(5, sim.cfg.n_agents // 5), alpha=0.45,
                color=color, edgecolor="none", label=f"t={s.tick}")
    ax.set_title("EH Distribution at Selected Ticks", fontsize=10)
    ax.set_xlabel("Epistemic Health (EH)")
    ax.set_ylabel("Agents")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"dashboard_tick_{tick:04d}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# run_all_plots — convenience wrapper after a full run
# ---------------------------------------------------------------------------

def run_all_plots(sim: "Simulation") -> None:
    """Generate all standard plots at the end of a run."""
    m = sim.metrics
    os.makedirs(sim.cfg.plot_dir, exist_ok=True)
    plot_wealth_distribution(m)
    plot_poli_eh(m)
    plot_trust_evolution(m)
    plot_belief_accuracy(sim.agents)
    plot_asset_prices(m)
    plot_venture_vs_trading(m)
    plot_wealth_quintiles(m)
    plot_comm_strategy_heatmap(sim.agents)
    if sim._compression_targets:
        plot_compression_test(m, sim._compression_targets, sim.cfg.compression_test_start)
    # Final dashboard
    plot_midrun_dashboard(sim, sim.tick, sim.cfg.plot_dir)
