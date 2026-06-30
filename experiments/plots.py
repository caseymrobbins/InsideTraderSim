"""
Experiment-specific visualizations.

All functions accept a list of TrialResult objects and save PNGs to output_dir.
"""
from __future__ import annotations
import math
import os
from typing import Dict, List, Optional
import numpy as np

from .runner import TrialResult
from .stats import compute_trial_stats, compare_models, compression_analysis, _KEY_METRICS

# Lazy matplotlib import so the module loads even without a display
_plt = None

def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


_MODEL_COLORS = {
    "U":    "#4878CF",
    "UF":   "#6ACC65",
    "UH":   "#D65F5F",
    "UHF":  "#B47CC7",
    "UHFS": "#C4AD66",
    "MIXED": "#77BEDB",
}

_MODEL_ORDER = ["U", "UF", "UH", "UHF", "UHFS", "MIXED"]


def plot_model_comparison(
    results: List[TrialResult],
    output_dir: str,
    scenario: Optional[str] = None,
    suffix: str = "",
) -> None:
    """Box-plot grid comparing the 5 reward models on key metrics."""
    plt = _get_plt()
    os.makedirs(output_dir, exist_ok=True)

    subset = [r for r in results if scenario is None or r.scenario == scenario]
    if not subset:
        return

    models = [m for m in _MODEL_ORDER if any(r.reward_model == m for r in subset)]
    metrics = [
        ("final_gini",           "Final Gini"),
        ("final_mean_eh",        "Final Mean EH"),
        ("final_mean_poli_eh",   "Final Mean POLI×EH"),
        ("wealth_growth",        "Wealth Growth"),
        ("final_deception_rate", "Deception Rate"),
        ("final_mean_trust",     "Mean Trust"),
        ("mean_reward_signal",   "Mean Reward Signal"),
        ("n_danger_zone_ticks",  "Danger Zone Ticks"),
    ]

    ncols = 4
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, (field, label) in zip(axes, metrics):
        data = []
        for m in models:
            vals = [getattr(r, field) for r in subset if r.reward_model == m]
            data.append(np.array([v for v in vals if np.isfinite(v)]))

        bplot = ax.boxplot(
            data,
            tick_labels=models,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 2},
        )
        for patch, model in zip(bplot["boxes"], models):
            patch.set_facecolor(_MODEL_COLORS.get(model, "grey"))
            patch.set_alpha(0.85)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Reward Model", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    title_suffix = f" — {scenario}" if scenario else ""
    fig.suptitle(f"Reward Model Comparison{title_suffix}", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = f"model_comparison{'_' + scenario if scenario else ''}{suffix}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, fname)}", flush=True)


def plot_compression_effect(
    results: List[TrialResult],
    output_dir: str,
) -> None:
    """
    Grouped bar chart: compressed-agent EH and wealth ratio by reward model.
    Shows whether the full model (UHFS) protects or exacerbates compression.
    """
    plt = _get_plt()
    os.makedirs(output_dir, exist_ok=True)

    comp = compression_analysis(results)
    models = [m for m in _MODEL_ORDER if m in comp and m != "MIXED"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    def bar(ax, field, title, ylabel):
        vals = [comp[m][field] for m in models]
        colors = [_MODEL_COLORS.get(m, "grey") for m in models]
        bars = ax.bar(models, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Reward Model", fontsize=9)
        for bar_, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar_.get_x() + bar_.get_width() / 2, bar_.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    bar(axes[0], "mean_compressed_eh",
        "Compressed Agent EH\n(lower = more degraded)", "Mean EH")
    bar(axes[1], "compressed_wealth_ratio",
        "Compressed / Non-Compressed\nWealth Ratio (>1 = extraction)", "Ratio")
    bar(axes[2], "mean_danger_zone_ticks",
        "Total Danger Zone Agent-Ticks\n(lower = better agency)", "Agent-Ticks")

    fig.suptitle("Compression Effect by Reward Model", fontsize=13)
    fig.tight_layout()
    fname = "compression_effect.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, fname)}", flush=True)


def plot_scenario_heatmap(
    results: List[TrialResult],
    output_dir: str,
    metric: str = "final_mean_eh",
) -> None:
    """
    Heatmap: reward model (rows) × scenario (columns), coloured by a single metric.
    """
    plt = _get_plt()
    os.makedirs(output_dir, exist_ok=True)

    scenarios = sorted(set(r.scenario for r in results))
    models = [m for m in _MODEL_ORDER if any(r.reward_model == m for r in results) and m != "MIXED"]

    data = np.full((len(models), len(scenarios)), float("nan"))
    for i, m in enumerate(models):
        for j, s in enumerate(scenarios):
            vals = [getattr(r, metric) for r in results
                    if r.reward_model == m and r.scenario == s]
            finite = [v for v in vals if np.isfinite(v)]
            if finite:
                data[i, j] = float(np.mean(finite))

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    plt.colorbar(im, ax=ax, label=metric)

    for i in range(len(models)):
        for j in range(len(scenarios)):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                        color="black")

    ax.set_title(f"Mean {metric}\n(model × scenario)", fontsize=11)
    fig.tight_layout()
    fname = f"heatmap_{metric}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, fname)}", flush=True)


def plot_mixed_vs_pure(
    results: List[TrialResult],
    mixed_results: List[TrialResult],
    output_dir: str,
) -> None:
    """Compare MIXED model trials against pure-model baselines."""
    plt = _get_plt()
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ("final_mean_eh",   "Mean EH"),
        ("final_gini",      "Gini"),
        ("wealth_growth",   "Wealth Growth"),
        ("mean_reward_signal", "Reward Signal"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))

    for ax, (field, label) in zip(axes, metrics):
        pure_vals_by_model = {}
        for m in [m for m in _MODEL_ORDER if m != "MIXED"]:
            vals = [getattr(r, field) for r in results if r.reward_model == m]
            pure_vals_by_model[m] = [v for v in vals if np.isfinite(v)]

        mixed_vals = [getattr(r, field) for r in mixed_results]
        mixed_vals = [v for v in mixed_vals if np.isfinite(v)]

        all_data = list(pure_vals_by_model.values()) + [mixed_vals]
        all_labels = list(pure_vals_by_model.keys()) + ["MIXED"]
        colors = [_MODEL_COLORS.get(m, "grey") for m in all_labels]

        bplot = ax.boxplot(
            all_data,
            tick_labels=all_labels,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 2},
        )
        for patch, c in zip(bplot["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Model", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Pure Models vs Mixed Assignment", fontsize=12)
    fig.tight_layout()
    fname = "mixed_vs_pure.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, fname)}", flush=True)


def plot_all(
    results: List[TrialResult],
    mixed_results: List[TrialResult],
    output_dir: str,
) -> None:
    """Generate the full set of experiment plots."""
    all_results = results + mixed_results

    # Per-scenario comparison
    scenarios = sorted(set(r.scenario for r in results))
    for scen in scenarios:
        plot_model_comparison(results, output_dir, scenario=scen)

    # Combined (all scenarios)
    plot_model_comparison(results, output_dir, scenario=None, suffix="_all")

    plot_compression_effect(results, output_dir)

    for metric in ("final_mean_eh", "final_gini", "wealth_growth"):
        plot_scenario_heatmap(results, output_dir, metric=metric)

    if mixed_results:
        plot_mixed_vs_pure(results, mixed_results, output_dir)

    print(f"  All experiment plots saved to {output_dir}/", flush=True)
