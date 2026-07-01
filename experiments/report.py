"""
Markdown report generator for experiment results.
"""
from __future__ import annotations
import os
from datetime import datetime
from typing import Dict, List

import numpy as np

from .runner import TrialResult
from .stats import compute_trial_stats, compare_models, compression_analysis, _KEY_METRICS
from .design import REWARD_MODELS


_METRIC_LABELS = {
    "final_gini":               "Final Gini Coefficient",
    "final_mean_eh":            "Final Mean EH",
    "final_mean_poli_eh":       "Final Mean POLI×EH",
    "final_venture_success_rate": "Venture Success Rate",
    "final_deception_rate":     "Deception Rate",
    "final_mean_trust":         "Mean Trust",
    "wealth_growth":            "Wealth Growth",
    "eh_trajectory_slope":      "EH Trajectory Slope",
    "compressed_final_eh":      "Compressed Agent Final EH",
    "compressed_final_poli":    "Compressed Agent Final POLI",
    "noncomp_final_wealth":     "Non-Compressed Final Wealth",
    "mean_reward_signal":       "Mean Reward Signal",
    "n_danger_zone_ticks":      "Total Danger-Zone Agent-Ticks",
    "mean_gini_over_time":      "Mean Gini (time avg)",
}


def build_report(
    results: List[TrialResult],
    mixed_results: List[TrialResult],
    output_dir: str,
) -> str:
    """Generate a Markdown report and save it to output_dir/report.md. Returns path."""
    os.makedirs(output_dir, exist_ok=True)
    lines: List[str] = []

    lines.append("# InsideTraderSim — Reward Model Experiment Report")
    lines.append(f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    # Overview
    n_trials = len(results)
    models = sorted(set(r.reward_model for r in results))
    scenarios = sorted(set(r.scenario for r in results))
    lines.append("## Overview\n")
    lines.append(f"- **Trials**: {n_trials}  ({len(results) // max(len(models) * len(scenarios), 1)} per cell)")
    lines.append(f"- **Models**: {', '.join(models)}")
    lines.append(f"- **Scenarios**: {', '.join(scenarios)}")
    lines.append(f"- **Mixed-model trials**: {len(mixed_results)}\n")

    # Per-scenario model comparison
    for scen in scenarios:
        sub = [r for r in results if r.scenario == scen]
        stats = compute_trial_stats(sub)
        lines.append(f"## Scenario: {scen}\n")

        # Summary table
        key_fields = ["final_gini", "final_mean_eh", "wealth_growth", "final_deception_rate",
                      "mean_reward_signal", "n_danger_zone_ticks"]
        header = "| Model | " + " | ".join(_METRIC_LABELS[f] for f in key_fields) + " |"
        sep    = "|-------|" + "|".join(["-------"] * len(key_fields)) + "|"
        lines.append(header)
        lines.append(sep)
        for m in REWARD_MODELS:
            if m not in stats:
                continue
            row = f"| **{m}** |"
            for f in key_fields:
                val = stats[m][f]["mean"]
                std = stats[m][f]["std"]
                row += f" {val:.4f} ± {std:.4f} |"
            lines.append(row)
        lines.append("")

        # Pairwise stats — highlight significant pairs
        comparisons = compare_models(sub, scenario=scen)
        sig_pairs = [
            (pair, metric, data)
            for pair, metrics in comparisons.items()
            for metric, data in metrics.items()
            if np.isfinite(data["p_value"]) and data["p_value"] < 0.05
        ]

        if sig_pairs:
            lines.append(f"### Significant pairwise differences (p < 0.05)\n")
            lines.append("| Pair | Metric | p-value | Cohen's d | Direction |")
            lines.append("|------|--------|---------|-----------|-----------|")
            for (ma, mb), metric, data in sorted(sig_pairs, key=lambda x: x[2]["p_value"]):
                lines.append(
                    f"| {ma} vs {mb} | {_METRIC_LABELS.get(metric, metric)} | "
                    f"{data['p_value']:.4f} | {data['cohens_d']:+.3f} | "
                    f"{ma} {data['direction']} {mb} |"
                )
            lines.append("")
        else:
            lines.append("_No statistically significant pairwise differences at p < 0.05._\n")

    # Compression analysis
    lines.append("## Compression Test Analysis\n")
    comp = compression_analysis(results)
    comp_models = [m for m in REWARD_MODELS if m in comp]
    if comp_models:
        lines.append("| Model | Compressed EH | Wealth Ratio | Danger-Zone Ticks |")
        lines.append("|-------|--------------|--------------|-------------------|")
        for m in comp_models:
            c = comp[m]
            lines.append(
                f"| **{m}** | {c['mean_compressed_eh']:.4f} | "
                f"{c['compressed_wealth_ratio']:.3f} | "
                f"{c['mean_danger_zone_ticks']:.1f} |"
            )
        lines.append("")

        best_eh = max(comp_models, key=lambda m: comp[m]["mean_compressed_eh"])
        worst_ratio = max(comp_models, key=lambda m: comp[m]["compressed_wealth_ratio"])
        lines.append("### Interpretation\n")
        lines.append(
            f"- **{best_eh}** yields the highest epistemic health among compressed agents, "
            f"suggesting it best counters EH degradation."
        )
        lines.append(
            f"- **{worst_ratio}** produces the highest compressed/non-compressed wealth ratio "
            f"({comp[worst_ratio]['compressed_wealth_ratio']:.3f}), "
            f"indicating more extractive dynamics under this model."
        )
        lines.append("")
    else:
        lines.append("_No compression data available for this result set._\n")

    # Mixed-model test
    if mixed_results:
        lines.append("## Mixed-Model Test\n")
        mixed_stats = compute_trial_stats(mixed_results)
        if "MIXED" in mixed_stats:
            ms = mixed_stats["MIXED"]
            for f in ["final_gini", "final_mean_eh", "wealth_growth", "final_deception_rate"]:
                lines.append(f"- **{_METRIC_LABELS[f]}**: {ms[f]['mean']:.4f} ± {ms[f]['std']:.4f}")
        lines.append("")
        lines.append(
            "When agents are assigned heterogeneous reward models, emergent coordination (or conflict) "
            "arises from the interaction of agents optimizing different objective functions."
        )
        lines.append("")

    # Full metric table (appendix)
    if results:
        lines.append("## Appendix: Full Metric Table (all scenarios combined)\n")
        all_stats = compute_trial_stats(results)
        all_models = [m for m in REWARD_MODELS if m in all_stats]
        # Also include any model not in REWARD_MODELS (e.g. single-model reports)
        extra = [m for m in all_stats if m not in REWARD_MODELS]
        all_models = all_models + extra
        header = "| Model | " + " | ".join(_METRIC_LABELS.get(f, f) for f in _KEY_METRICS) + " |"
        sep = "|-------|" + "|".join(["---"] * len(_KEY_METRICS)) + "|"
        lines.append(header)
        lines.append(sep)
        for m in all_models:
            row = f"| **{m}** |"
            for f in _KEY_METRICS:
                v = all_stats[m][f]["mean"]
                row += f" {v:.4f} |"
            lines.append(row)
        lines.append("")

    report_text = "\n".join(lines)
    path = os.path.join(output_dir, "report.md")
    with open(path, "w") as fh:
        fh.write(report_text)
    print(f"  Report saved: {path}", flush=True)
    return path
