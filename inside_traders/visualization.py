"""Visualization hooks for InsideTraderSim metrics."""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
import numpy as np

if TYPE_CHECKING:
    from .metrics import MetricsCollector, TickSnapshot
    from .simulation import Simulation


def plot_wealth_distribution(metrics: "MetricsCollector", title: str = "Wealth Distribution") -> None:
    """Lorenz curve + Gini evolution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping wealth distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    # Left: Lorenz curve at final snapshot
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

    # Right: Gini over time
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
    """POLI × EH product over time + scatter."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping POLI×EH plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("POLI × EH (Epistemic Health) Analysis")

    ticks = [s.tick for s in metrics.snapshots]
    mean_poli_eh = [s.poli_eh.mean() for s in metrics.snapshots]
    axes[0].plot(ticks, mean_poli_eh, color="steelblue")
    axes[0].set_xlabel("Tick")
    axes[0].set_ylabel("Mean POLI × EH")
    axes[0].set_title("Mean POLI×EH Over Time")
    axes[0].grid(alpha=0.3)

    # Scatter: final snapshot POLI vs EH
    if metrics.snapshots:
        final = metrics.snapshots[-1]
        axes[1].scatter(final.poli, final.eh, alpha=0.6, color="darkorange", edgecolors="k", linewidths=0.5)
        axes[1].set_xlabel("POLI Score")
        axes[1].set_ylabel("Epistemic Health (EH)")
        axes[1].set_title("POLI vs EH (final tick)")
        # Add correlation annotation
        if len(final.poli) >= 2:
            corr = float(np.corrcoef(final.poli, final.eh)[0, 1])
            axes[1].annotate(
                f"r = {corr:.3f}",
                xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=11, color="navy",
            )

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


def plot_belief_accuracy(agents, title: str = "Belief Accuracy Distribution") -> None:
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
    ax.axvline(np.mean(accuracies), color="red", linestyle="--", label=f"Mean: {np.mean(accuracies):.2f}")
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
    """Venture success rate vs total trade volume proxy."""
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


def run_all_plots(sim: "Simulation") -> None:
    """Convenience function: generate all standard plots after a run."""
    m = sim.metrics
    plot_wealth_distribution(m)
    plot_poli_eh(m)
    plot_trust_evolution(m)
    plot_belief_accuracy(sim.agents)
    plot_asset_prices(m)
    plot_venture_vs_trading(m)
    if sim._compression_targets:
        plot_compression_test(m, sim._compression_targets, sim.cfg.compression_test_start)
