#!/usr/bin/env python3
"""
run_experiments.py — Full reward-model experiment suite.

Trains all five reward models and runs comparative experiments in a single command:

    python run_experiments.py --pretrain

Without --pretrain, skips solo pretraining and runs the joint simulation directly
(original behaviour, useful when checkpoints already exist).

Full usage:
    python run_experiments.py [--pretrain] [--pretrain-ticks N] [--checkpoint-dir DIR]
                              [--trials N] [--output-dir DIR] [--device cpu|cuda]
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

    # Pretraining
    p.add_argument("--pretrain",         action="store_true",
                   help="Solo-pretrain each reward model before running experiments")
    p.add_argument("--pretrain-ticks",   type=int, default=100,
                   help="Solo ticks per agent during pretraining (default: 100)")
    p.add_argument("--checkpoint-dir",   default="checkpoints",
                   help="Base directory for pretrained checkpoints (default: checkpoints)")
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
    print(f"  Pretrain  : {'yes (' + str(args.pretrain_ticks) + ' solo ticks/agent)' if args.pretrain else 'no'}")
    print(f"  Output    : {args.output_dir}/")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Optional pretraining: one solo run per reward model
    # ------------------------------------------------------------------
    pretrained_checkpoints: dict = {}
    if args.pretrain:
        from inside_traders.config import SimConfig
        from inside_traders.pretrain import PretrainConfig, run_solo_pretrain

        print(f"  Pretraining {len(exp.reward_models)} models "
              f"({args.pretrain_ticks} solo ticks each) ...\n")
        for model in exp.reward_models:
            ckpt_path = os.path.join(args.checkpoint_dir, model, "pretrained.json")
            plot_dir  = os.path.join("plots", model)
            pretrain_cfg = PretrainConfig(
                n_ticks=args.pretrain_ticks,
                checkpoint_path=ckpt_path,
                plot_dir=plot_dir,
                verbose=False,
            )
            sim_cfg = SimConfig(
                n_agents=exp.n_agents,
                n_ticks=args.pretrain_ticks,
                seed=args.seed_start,
                device=args.device,
                reward_model=model,
            )
            print(f"  [{model}] solo pretraining → {ckpt_path}")
            run_solo_pretrain(sim_cfg, pretrain_cfg)
            pretrained_checkpoints[model] = ckpt_path
        print()

        # Rebuild exp with checkpoint paths so each trial loads pretrained state
        exp = ExperimentConfig(
            reward_models=exp.reward_models,
            n_trials=exp.n_trials,
            base_seeds=exp.base_seeds,
            scenario_names=exp.scenario_names,
            n_agents=exp.n_agents,
            n_ticks=exp.n_ticks,
            device=exp.device,
            output_dir=exp.output_dir,
            run_mixed=exp.run_mixed,
            pretrained_checkpoints=pretrained_checkpoints,
        )

    t0 = time.perf_counter()
    runner = ExperimentRunner(exp, verbose=False)
    results = runner.run_all()
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"  Completed {len(results)} trials in {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Save data: combined at top level, per-model in subdirectories
    # ------------------------------------------------------------------

    # Combined raw results (all models + mixed)
    combined_path = os.path.join(args.output_dir, "results.json")
    with open(combined_path, "w") as fh:
        json.dump([asdict(r) for r in results + runner.mixed_results], fh, indent=2)
    print(f"  Combined results: {combined_path}")

    # Per-model directories: {output_dir}/{model}/results.json + report + plots
    for model in exp.reward_models:
        model_results = [r for r in results if r.reward_model == model]
        if not model_results:
            continue
        model_dir = os.path.join(args.output_dir, model)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "results.json")
        with open(model_path, "w") as fh:
            json.dump([asdict(r) for r in model_results], fh, indent=2)

        model_report = build_report(model_results, [], model_dir)
        plot_all(model_results, [], model_dir)
        print(f"  {model:5s}  → {model_dir}/  ({len(model_results)} trials)")

    # MIXED data gets its own directory if present
    if runner.mixed_results:
        mixed_dir = os.path.join(args.output_dir, "MIXED")
        os.makedirs(mixed_dir, exist_ok=True)
        with open(os.path.join(mixed_dir, "results.json"), "w") as fh:
            json.dump([asdict(r) for r in runner.mixed_results], fh, indent=2)
        build_report([], runner.mixed_results, mixed_dir)
        print(f"  {'MIXED':5s}  → {mixed_dir}/  ({len(runner.mixed_results)} trials)")

    print()

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

    # Combined comparison plots (require all models together)
    print("  Generating combined comparison plots...")
    plot_all(results, runner.mixed_results, args.output_dir)

    # Combined report
    print("  Generating combined Markdown report...")
    report_path = build_report(results, runner.mixed_results, args.output_dir)

    print(f"\n  Done.")
    print(f"  Per-model data : {args.output_dir}/{{U,UF,UH,UHF,UHFS}}/")
    print(f"  Combined report: {report_path}\n")


if __name__ == "__main__":
    main()
