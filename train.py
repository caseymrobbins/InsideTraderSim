#!/usr/bin/env python3
"""
InsideTraderSim — train all reward models and run comparative experiments.

Each of the five objective functions (U, UF, UH, UHF, UHFS) gets its own
independent training run before evaluation, so Q-tables and epistemic state
are shaped by that model's reward signal before any experiment trial runs.

Usage
-----
  python train.py                   # full run — recommended default
  python train.py --quick           # smoke test (fewer ticks + trials)
  python train.py --skip-training   # re-run experiments on existing checkpoints
  python train.py --device cuda     # GPU acceleration
  python train.py --trials 20 --train-ticks 500

Pipeline
--------
  For each model (U → UF → UH → UHF → UHFS):
    Phase 1  Solo pretrain  — builds epistemic competence, no comms
    Phase 2  Joint training — full multi-agent sim with that model's
                              reward function; trains comm Q-tables

  Then: full experiment grid  (model × scenario × seed)
        per-model reports + plots in  {output}/U/  {output}/UF/  …
        combined comparison plots in  {output}/
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))

from experiments.design import ExperimentConfig, SCENARIOS
from experiments.runner import ExperimentRunner
from experiments.stats import compute_trial_stats, compression_analysis
from experiments.plots import plot_all
from experiments.report import build_report

_MODELS   = ("U", "UF", "UH", "UHF", "UHFS")
_W        = 62   # banner width


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train",
        description="Train all InsideTraderSim reward models and run experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py\n"
            "  python train.py --quick\n"
            "  python train.py --device cuda --trials 20 --train-ticks 500\n"
            "  python train.py --skip-training --output my_results\n"
        ),
    )

    # Core settings
    p.add_argument("--agents",   type=int, default=30,   metavar="N",
                   help="Agents per simulation (default: 30)")
    p.add_argument("--trials",   type=int, default=10,   metavar="N",
                   help="Experiment trials per (model × scenario) cell (default: 10)")
    p.add_argument("--device",   default="cpu", choices=["cpu", "cuda"],
                   help="Compute device (default: cpu)")
    p.add_argument("--output",   default="experiment_results", metavar="DIR",
                   help="Output directory for results, plots, and reports "
                        "(default: experiment_results)")
    p.add_argument("--seed",     type=int, default=0, metavar="N",
                   help="Starting seed for experiment trials (default: 0)")

    # Training depth
    training = p.add_argument_group("training")
    training.add_argument("--pretrain-ticks", type=int, default=100, metavar="N",
                          help="Solo ticks per agent — Phase 1 (default: 100)")
    training.add_argument("--train-ticks",    type=int, default=300, metavar="N",
                          help="Joint training ticks per model — Phase 2 (default: 300)")
    training.add_argument("--eval-ticks",     type=int, default=300, metavar="N",
                          help="Ticks per experiment trial (default: 300)")
    training.add_argument("--curriculum",     type=int, default=0,  metavar="N",
                          dest="curriculum_honest_ticks",
                          help="TRUTHFUL-only comms for first N ticks of joint training "
                               "(default: 0 = disabled)")
    training.add_argument("--checkpoints",    default="checkpoints", metavar="DIR",
                          help="Directory for per-model checkpoints (default: checkpoints)")

    # Shortcuts
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke test: 30 pretrain + 100 train + 100 eval ticks, "
                        "2 trials per cell, no mixed test")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip Phases 1 and 2; load existing posttrain checkpoints "
                        "and go straight to experiments")

    # Experiment scope
    scope = p.add_argument_group("experiment scope")
    scope.add_argument("--models",    nargs="+", default=list(_MODELS),
                       choices=list(_MODELS), metavar="MODEL",
                       help="Subset of models to run (default: all five)")
    scope.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()),
                       choices=list(SCENARIOS.keys()), metavar="SCENARIO",
                       help="Scenarios to include (default: both)")
    scope.add_argument("--no-mixed",  action="store_true",
                       help="Skip the mixed-model test")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Banner helpers
# ──────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'─' * _W}")
    print(f"  {title}")
    print(f"{'─' * _W}")


def _step(label: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  ▸ {label}{suffix}", flush=True)


def _done(label: str, elapsed: float) -> None:
    print(f"  ✓ {label}  ({elapsed:.1f}s)", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_all_models(args: argparse.Namespace) -> dict[str, str]:
    """
    Run Phase 1 (solo pretrain) + Phase 2 (joint training) for every model.
    Returns a mapping  {model_name -> posttrain_checkpoint_path}.
    """
    from inside_traders.config import SimConfig
    from inside_traders.pretrain import PretrainConfig, run_solo_pretrain
    from inside_traders.simulation import Simulation
    from inside_traders.checkpoint import (
        load_checkpoint, apply_checkpoint, save_checkpoint,
    )

    posttrain: dict[str, str] = {}

    for i, model in enumerate(args.models, 1):
        model_dir  = os.path.join(args.checkpoints, model)
        os.makedirs(model_dir, exist_ok=True)
        plot_dir   = os.path.join("plots", model)
        solo_ckpt  = os.path.join(model_dir, "pretrained.json")
        joint_ckpt = os.path.join(model_dir, "posttrain.json")

        _banner(f"Model {i}/{len(args.models)} — {model}")

        # ── Phase 1: solo pretrain ──────────────────────────────────
        _step(f"Phase 1  solo pretrain", f"{args.pretrain_ticks} ticks/agent")
        t0 = time.perf_counter()
        solo_cfg = SimConfig(
            n_agents=args.agents,
            n_ticks=args.pretrain_ticks,
            seed=args.seed,
            device=args.device,
            reward_model=model,
            plot_dir=plot_dir,
            plot_interval=9999,
            status_interval=9999,
        )
        run_solo_pretrain(solo_cfg, PretrainConfig(
            n_ticks=args.pretrain_ticks,
            checkpoint_path=solo_ckpt,
            plot_dir=plot_dir,
            verbose=False,
        ))
        _done(f"pretrained.json saved", time.perf_counter() - t0)

        # ── Phase 2: joint training ─────────────────────────────────
        _step(f"Phase 2  joint training",
              f"{args.train_ticks} ticks  reward={model}"
              + (f"  curriculum={args.curriculum_honest_ticks}" if args.curriculum_honest_ticks else ""))
        t0 = time.perf_counter()
        joint_cfg = SimConfig(
            n_agents=args.agents,
            n_ticks=args.train_ticks,
            seed=args.seed,
            device=args.device,
            reward_model=model,
            curriculum_honest_ticks=args.curriculum_honest_ticks,
            plot_dir=plot_dir,
            plot_interval=9999,
            status_interval=9999,
            verbose=False,
        )
        sim = Simulation(joint_cfg)
        apply_checkpoint(sim.agents, load_checkpoint(solo_ckpt), reset_comm=True)
        sim.run()
        save_checkpoint(sim.agents, joint_ckpt)
        _done(f"posttrain.json saved", time.perf_counter() - t0)

        posttrain[model] = joint_ckpt

    return posttrain


# ──────────────────────────────────────────────────────────────────────
# Experiment
# ──────────────────────────────────────────────────────────────────────

def run_experiment(
    args: argparse.Namespace,
    posttrain_checkpoints: dict[str, str],
) -> None:
    """Run the full experiment grid and save all outputs."""
    seeds = tuple(range(args.seed, args.seed + args.trials))

    exp = ExperimentConfig(
        reward_models=tuple(args.models),
        n_trials=args.trials,
        base_seeds=seeds,
        scenario_names=tuple(args.scenarios),
        n_agents=args.agents,
        n_ticks=args.eval_ticks,
        device=args.device,
        output_dir=args.output,
        run_mixed=not args.no_mixed,
        pretrained_checkpoints=posttrain_checkpoints,
    )

    total = len(exp.reward_models) * len(exp.scenario_names) * exp.n_trials
    mixed = len(exp.scenario_names) * exp.n_trials if exp.run_mixed else 0

    _banner("Experiment")
    print(f"  Models    : {', '.join(exp.reward_models)}")
    print(f"  Scenarios : {', '.join(exp.scenario_names)}")
    print(f"  Trials    : {exp.n_trials} per cell  ({total} pure + {mixed} mixed)")
    print(f"  Ticks     : {exp.n_ticks}  |  Agents: {exp.n_agents}")
    print(f"  Device    : {exp.device}")
    print()

    t0 = time.perf_counter()
    runner = ExperimentRunner(exp, verbose=False)
    results = runner.run_all()
    elapsed = time.perf_counter() - t0

    print(f"\n  {len(results)} trials completed in {elapsed:.1f}s")

    # ── Save outputs ────────────────────────────────────────────────
    _banner("Saving outputs")
    os.makedirs(args.output, exist_ok=True)

    combined_path = os.path.join(args.output, "results.json")
    with open(combined_path, "w") as fh:
        json.dump([asdict(r) for r in results + runner.mixed_results], fh, indent=2)
    _step("Combined results", combined_path)

    for model in exp.reward_models:
        model_results = [r for r in results if r.reward_model == model]
        if not model_results:
            continue
        model_dir = os.path.join(args.output, model)
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "results.json"), "w") as fh:
            json.dump([asdict(r) for r in model_results], fh, indent=2)
        build_report(model_results, [], model_dir)
        plot_all(model_results, [], model_dir)
        _step(f"{model}", f"{model_dir}/")

    if runner.mixed_results:
        mixed_dir = os.path.join(args.output, "MIXED")
        os.makedirs(mixed_dir, exist_ok=True)
        with open(os.path.join(mixed_dir, "results.json"), "w") as fh:
            json.dump([asdict(r) for r in runner.mixed_results], fh, indent=2)
        build_report([], runner.mixed_results, mixed_dir)
        _step("MIXED", f"{mixed_dir}/")

    _step("Combined comparison plots")
    plot_all(results, runner.mixed_results, args.output)

    _step("Combined report")
    report_path = build_report(results, runner.mixed_results, args.output)

    # ── Summary table ───────────────────────────────────────────────
    _banner("Results summary")
    stats = compute_trial_stats(results)
    print(f"  {'Model':6s}  {'EH':>8s}  {'Gini':>8s}  {'Wealth Δ':>10s}  "
          f"{'Deception':>10s}  {'Danger tks':>11s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*11}")
    for m in exp.reward_models:
        if m not in stats:
            continue
        s = stats[m]
        print(
            f"  {m:6s}  "
            f"{s['final_mean_eh']['mean']:>8.4f}  "
            f"{s['final_gini']['mean']:>8.4f}  "
            f"{s['wealth_growth']['mean']:>+10.4f}  "
            f"{s['final_deception_rate']['mean']:>10.4f}  "
            f"{s['n_danger_zone_ticks']['mean']:>11.1f}"
        )

    if runner.mixed_results:
        ms = compute_trial_stats(runner.mixed_results).get("MIXED")
        if ms:
            print(
                f"  {'MIXED':6s}  "
                f"{ms['final_mean_eh']['mean']:>8.4f}  "
                f"{ms['final_gini']['mean']:>8.4f}  "
                f"{ms['wealth_growth']['mean']:>+10.4f}  "
                f"{ms['final_deception_rate']['mean']:>10.4f}  "
                f"{'—':>11s}"
            )

    comp = compression_analysis(results)
    if comp:
        print(f"\n  Compression effect:")
        print(f"  {'Model':6s}  {'Compressed EH':>14s}  {'Wealth ratio':>13s}  "
              f"{'Danger tks':>11s}")
        print(f"  {'─'*6}  {'─'*14}  {'─'*13}  {'─'*11}")
        for m in exp.reward_models:
            if m not in comp:
                continue
            c = comp[m]
            print(
                f"  {m:6s}  "
                f"{c['mean_compressed_eh']:>14.4f}  "
                f"{c['compressed_wealth_ratio']:>13.3f}  "
                f"{c['mean_danger_zone_ticks']:>11.1f}"
            )

    print(f"\n  Report : {report_path}")
    print(f"  Data   : {args.output}/")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.quick:
        args.pretrain_ticks = 30
        args.train_ticks    = 100
        args.eval_ticks     = 100
        args.trials         = 2
        args.no_mixed       = True

    print(f"\n{'═' * _W}")
    print(f"  InsideTraderSim")
    print(f"{'═' * _W}")
    print(f"  Models    : {', '.join(args.models)}")
    print(f"  Agents    : {args.agents}  |  Device: {args.device}")
    if not args.skip_training:
        print(f"  Training  : solo {args.pretrain_ticks} ticks + "
              f"joint {args.train_ticks} ticks per model")
    else:
        print(f"  Training  : skipped (loading existing checkpoints)")
    print(f"  Eval      : {args.eval_ticks} ticks × {args.trials} trials per cell")
    print(f"  Output    : {args.output}/")
    print(f"{'═' * _W}")

    t_total = time.perf_counter()

    # ── Training ────────────────────────────────────────────────────
    if args.skip_training:
        posttrain = {
            m: os.path.join(args.checkpoints, m, "posttrain.json")
            for m in args.models
        }
        missing = [p for p in posttrain.values() if not os.path.exists(p)]
        if missing:
            print("\n  ERROR: --skip-training requires posttrain checkpoints:")
            for p in missing:
                print(f"    missing: {p}")
            print("  Run without --skip-training to train first.\n")
            sys.exit(1)
    else:
        posttrain = train_all_models(args)

    # ── Experiment ──────────────────────────────────────────────────
    run_experiment(args, posttrain)

    print(f"\n{'═' * _W}")
    print(f"  Total time: {time.perf_counter() - t_total:.1f}s")
    print(f"{'═' * _W}\n")


if __name__ == "__main__":
    main()
