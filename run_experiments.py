#!/usr/bin/env python3
"""
run_experiments.py — Full reward-model experiment suite.

Runs all five reward models (U, UF, UH, UHF, UHFS) across two volatility
scenarios with N trials per cell, then generates comparative statistics,
plots, and a Markdown report.

Usage:
    python run_experiments.py [--trials N] [--output-dir DIR] [--device cpu|cuda]
                              [--no-mixed] [--scenarios high_vol moderate_vol]
                              [--models U UF UH UHF UHFS]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import asdict

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from experiments.design import ExperimentConfig, SCENARIOS
from experiments.runner import ExperimentRunner
from experiments.stats import compute_trial_stats, compare_models, compression_analysis
from experiments.plots import plot_all
from experiments.report import build_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="InsideTraderSim reward model experiment suite")
    p.add_argument("--trials",       type=int, default=10,
                   help="Trials per (model × scenario) cell (default: 10)")
    p.add_argument("--output-dir",   default="experiment_results",
                   help="Directory for results, plots, and report (default: experiment_results)")
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda"],
                   help="Compute device (default: cpu)")
    p.add_argument("--no-mixed",     action="store_true",
                   help="Skip the mixed-model test")
    p.add_argument("--scenarios",    nargs="+", default=list(SCENARIOS.keys()),
                   choices=list(SCENARIOS.keys()),
                   help="Scenarios to run (default: all)")
    p.add_argument("--models",       nargs="+",
                   default=["U", "UF", "UH", "UHF", "UHFS"],
                   choices=["U", "UF", "UH", "UHF", "UHFS"],
                   help="Reward models to test (default: all)")
    p.add_argument("--n-agents",     type=int, default=30)
    p.add_argument("--n-ticks",      type=int, default=300)
    p.add_argument("--seed-start",   type=int, default=0,
                   help="Starting seed (seeds will be seed_start, seed_start+1, ...)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seeds = tuple(range(args.seed_start, args.seed_start + args.trials))

    exp = ExperimentConfig(
        reward_models=tuple(args.models),
        n_trials=args.trials,
        base_seeds=seeds,
        scenario_names=tuple(args.scenarios),
        n_agents=args.n_agents,
        n_ticks=args.n_ticks,
        device=args.device,
        output_dir=args.output_dir,
        run_mixed=not args.no_mixed,
    )

    total_cells = len(exp.reward_models) * len(exp.scenario_names) * exp.n_trials
    mixed_cells = len(exp.scenario_names) * exp.n_trials if exp.run_mixed else 0
    print(f"\n{'='*60}")
    print(f"  InsideTraderSim — Reward Model Experiment Suite")
    print(f"{'='*60}")
    print(f"  Models    : {', '.join(exp.reward_models)}")
    print(f"  Scenarios : {', '.join(exp.scenario_names)}")
    print(f"  Trials    : {exp.n_trials} per cell  ({total_cells} + {mixed_cells} mixed)")
    print(f"  Agents    : {exp.n_agents}  |  Ticks: {exp.n_ticks}")
    print(f"  Device    : {exp.device}")
    print(f"  Output    : {args.output_dir}/")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    runner = ExperimentRunner(exp, verbose=False)
    results = runner.run_all()
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"  Completed {len(results)} trials in {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # Save raw results as JSON
    raw_path = os.path.join(args.output_dir, "results.json")
    with open(raw_path, "w") as fh:
        json.dump([asdict(r) for r in results + runner.mixed_results], fh, indent=2)
    print(f"  Raw results: {raw_path}")

    # Print quick summary to console
    all_stats = compute_trial_stats(results)
    print("\n  Quick summary (all scenarios combined):\n")
    print(f"  {'Model':6s}  {'Mean EH':>10s}  {'Gini':>8s}  {'Wealth Growth':>14s}  {'Deception':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*14}  {'-'*10}")
    for m in exp.reward_models:
        if m not in all_stats:
            continue
        s = all_stats[m]
        print(
            f"  {m:6s}  "
            f"{s['final_mean_eh']['mean']:>10.4f}  "
            f"{s['final_gini']['mean']:>8.4f}  "
            f"{s['wealth_growth']['mean']:>+14.4f}  "
            f"{s['final_deception_rate']['mean']:>10.4f}"
        )

    if runner.mixed_results:
        mixed_stats = compute_trial_stats(runner.mixed_results)
        if "MIXED" in mixed_stats:
            ms = mixed_stats["MIXED"]
            print(
                f"  {'MIXED':6s}  "
                f"{ms['final_mean_eh']['mean']:>10.4f}  "
                f"{ms['final_gini']['mean']:>8.4f}  "
                f"{ms['wealth_growth']['mean']:>+14.4f}  "
                f"{ms['final_deception_rate']['mean']:>10.4f}"
            )

    print()

    # Compression analysis
    comp = compression_analysis(results)
    print("  Compression effect (compressed-agent EH vs wealth ratio):\n")
    print(f"  {'Model':6s}  {'Compressed EH':>14s}  {'Wealth Ratio':>13s}  {'Danger Ticks':>13s}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*13}  {'-'*13}")
    for m in exp.reward_models:
        if m not in comp:
            continue
        c = comp[m]
        print(
            f"  {m:6s}  "
            f"{c['mean_compressed_eh']:>14.4f}  "
            f"{c['compressed_wealth_ratio']:>13.3f}  "
            f"{c['mean_danger_zone_ticks']:>13.1f}"
        )
    print()

    # Generate plots
    print("  Generating plots...")
    plot_all(results, runner.mixed_results, args.output_dir)

    # Generate report
    print("  Generating Markdown report...")
    report_path = build_report(results, runner.mixed_results, args.output_dir)

    print(f"\n  Done. Results in {args.output_dir}/")
    print(f"  Report: {report_path}\n")


if __name__ == "__main__":
    main()
