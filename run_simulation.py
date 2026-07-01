#!/usr/bin/env python3
"""
InsideTraderSim — sample run demonstrating emergence and the POLI×EH compression effect.

Intended workflow
-----------------
1. Pretrain agents in a solo-world environment (no communication):
       python run_simulation.py --pretrain-only

2. Load pretrained agents and run the joint multi-agent simulation:
       python run_simulation.py --resume-from checkpoints/pretrained.json

3. Or do both steps in one command:
       python run_simulation.py --pretrain

4. Skip pretraining and run the joint sim directly (original behaviour):
       python run_simulation.py --skip-pretrain
       python run_simulation.py            # same, --skip-pretrain is the default

Tests / evaluation should only be run after the training stage has completed
and a checkpoint is available:
       pytest tests/ -v
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
import numpy as np

from inside_traders.config import SimConfig
from inside_traders.simulation import Simulation
from inside_traders import visualization as viz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InsideTraderSim HorizonSim v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Simulation parameters ──────────────────────────────────────────
    p.add_argument("--agents", type=int, default=30)
    p.add_argument("--ticks", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--assets", type=int, default=5)
    p.add_argument("--ventures", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--status-interval", type=int, default=10,
                   help="Print a live status line every N ticks (default 10)")
    p.add_argument("--plot-interval", type=int, default=50,
                   help="Save a mid-run dashboard PNG every N ticks (default 50)")
    p.add_argument("--plot-dir", type=str, default="plots",
                   help="Directory to write plot files into")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                   help="Compute device: 'cpu' (default) or 'cuda' for A100/GPU")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--gini-target", type=float, default=None,
                   help="If set, print PASS/FAIL if final Gini exceeds this value")

    # ── Pretraining workflow ───────────────────────────────────────────
    train_group = p.add_mutually_exclusive_group()
    train_group.add_argument(
        "--pretrain",
        action="store_true",
        help=(
            "Run solo-world pretraining first, then load the pretrained "
            "agents and run the joint multi-agent simulation."
        ),
    )
    train_group.add_argument(
        "--pretrain-only",
        action="store_true",
        help=(
            "Run solo-world pretraining, save the checkpoint, then exit "
            "without running the joint simulation."
        ),
    )
    train_group.add_argument(
        "--resume-from",
        metavar="PATH",
        default=None,
        help=(
            "Load a previously saved checkpoint and run the joint "
            "simulation using pretrained agent state.  Skips pretraining."
        ),
    )
    train_group.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pretraining and run the joint simulation directly (default behaviour).",
    )

    p.add_argument(
        "--pretrain-ticks",
        type=int,
        default=100,
        help="Number of solo ticks per agent during pretraining (default 100).",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint files (default: checkpoints/).",
    )

    return p.parse_args()


def print_summary(sim: Simulation) -> None:
    m = sim.metrics
    if not m.snapshots:
        print("[!] No snapshots recorded.")
        return

    final = m.snapshots[-1]
    first = m.snapshots[0]

    print("\n" + "=" * 60)
    print("  SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Final total wealth  : {final.total_wealth:>10.1f}")
    print(f"  Initial total wealth: {first.total_wealth:>10.1f}")
    print(f"  Wealth growth       : {100*(final.total_wealth/first.total_wealth - 1):>+8.1f}%")
    print(f"  Final Gini          : {final.gini:.4f}")
    print(f"  Venture success rate: {final.venture_success_rate:.3f}")
    print(f"  Deception rate      : {final.deception_rate:.3f}")
    print(f"  Mean trust          : {final.mean_trust:.3f}")
    print(f"  POLI×EH correlation : {m.poli_eh_correlation():.4f}")

    order = np.argsort(final.wealth)[::-1]
    print("\n  Top 5 agents by wealth:")
    print(f"  {'Agent':12s} {'Wealth':>10s} {'POLI':>8s} {'EH':>8s} {'POLI×EH':>10s}")
    print("  " + "-" * 52)
    for i in order[:5]:
        print(f"  {final.agent_ids[i]:12s} {final.wealth[i]:>10.1f} "
              f"{final.poli[i]:>8.1f} {final.eh[i]:>8.3f} {final.poli_eh[i]:>10.1f}")

    if sim._compression_targets:
        print("\n  Compression Test Results:")
        results = sim.compression_test_summary()
        for aid, r in results.items():
            status = "DEGRADED" if r["degraded"] else "RESILIENT"
            print(f"    {aid}: pre_growth={r['pre_growth']:>+8.1f}  "
                  f"post_growth={r['post_growth']:>+8.1f}  [{status}]")

    print("=" * 60)


def _checkpoint_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "pretrained.json")


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    cfg = SimConfig(
        n_agents=args.agents,
        n_assets=args.assets,
        n_ventures=args.ventures,
        n_ticks=args.ticks,
        seed=args.seed,
        log_interval=args.log_interval,
        status_interval=args.status_interval,
        plot_interval=args.plot_interval,
        plot_dir=args.plot_dir,
        device=args.device,
        verbose=args.verbose,
    )

    ckpt_path = _checkpoint_path(args.checkpoint_dir)

    # ------------------------------------------------------------------
    # Step 1: Pretraining (if requested)
    # ------------------------------------------------------------------
    pretrained_state = None  # list[dict] | None

    if args.pretrain or args.pretrain_only:
        from inside_traders.pretrain import PretrainConfig, run_solo_pretrain
        pretrain_cfg = PretrainConfig(
            n_ticks=args.pretrain_ticks,
            checkpoint_path=ckpt_path,
            plot_dir=args.plot_dir,
            verbose=args.verbose,
        )
        run_solo_pretrain(cfg, pretrain_cfg)

        if args.pretrain_only:
            print(f"\n  Checkpoint saved to {ckpt_path}")
            print("  Run the joint simulation with:")
            print(f"    python run_simulation.py --resume-from {ckpt_path}\n")
            return

        # Load the freshly written checkpoint for the joint phase
        from inside_traders.checkpoint import load_checkpoint
        pretrained_state = load_checkpoint(ckpt_path)

    elif args.resume_from:
        from inside_traders.checkpoint import load_checkpoint
        pretrained_state = load_checkpoint(args.resume_from)
        print(f"\n  Loaded pretrained checkpoint from {args.resume_from}")

    # ------------------------------------------------------------------
    # Step 2: Joint multi-agent simulation
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    sim = Simulation(cfg)

    if pretrained_state is not None:
        from inside_traders.checkpoint import apply_checkpoint
        apply_checkpoint(sim.agents, pretrained_state, reset_comm=True)
        print("  Applied pretrained state (comm Q-table reset for joint phase)\n")

    sim.run()
    elapsed = time.perf_counter() - t0

    print_summary(sim)

    if not args.no_plots:
        print("\n  Generating plots...")
        viz.run_all_plots(sim)

    if args.gini_target is not None:
        final_gini = sim.metrics.snapshots[-1].gini if sim.metrics.snapshots else 0.0
        if final_gini > args.gini_target:
            print(f"\n  [PASS] Gini {final_gini:.3f} > threshold {args.gini_target}")
        else:
            print(f"\n  [FAIL] Gini {final_gini:.3f} <= threshold {args.gini_target}")
            sys.exit(1)


if __name__ == "__main__":
    main()
