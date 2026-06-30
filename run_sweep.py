#!/usr/bin/env python3
"""
Parameter sweep runner — vary signal cost and volatility, collect Gini + POLI×EH correlation.

Usage:
    python run_sweep.py
    python run_sweep.py --ticks 200 --agents 20
"""
from __future__ import annotations
import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass
from typing import List

from inside_traders.config import SimConfig
from inside_traders.simulation import Simulation


@dataclass
class SweepResult:
    signal_cost: float
    volatility: float
    seed: int
    final_gini: float
    poli_eh_corr: float
    total_wealth_growth: float
    venture_success: float
    deception_rate: float
    n_ticks: int


def sweep(
    ticks: int = 200,
    agents: int = 20,
    seeds: List[int] = (42, 43, 44),
) -> List[SweepResult]:
    signal_costs = [0.5, 1.5, 4.0]
    volatilities = [0.01, 0.03, 0.07]

    results = []
    combos = list(itertools.product(signal_costs, volatilities, seeds))
    total = len(combos)

    for i, (sc, vol, seed) in enumerate(combos, 1):
        print(f"  [{i}/{total}] signal_cost={sc} volatility={vol} seed={seed} ...", end="", flush=True)
        t0 = time.perf_counter()

        cfg = SimConfig(
            n_agents=agents,
            n_ticks=ticks,
            seed=seed,
            signal_base_cost=sc,
            asset_volatility=vol,
            verbose=False,
        )
        sim = Simulation(cfg)
        sim.run()

        m = sim.metrics
        snap0 = m.snapshots[0] if m.snapshots else None
        snap_f = m.snapshots[-1] if m.snapshots else None

        if snap0 and snap_f:
            growth = (snap_f.total_wealth - snap0.total_wealth) / (snap0.total_wealth + 1e-9)
        else:
            growth = 0.0

        results.append(SweepResult(
            signal_cost=sc,
            volatility=vol,
            seed=seed,
            final_gini=snap_f.gini if snap_f else 0.0,
            poli_eh_corr=m.poli_eh_correlation(),
            total_wealth_growth=growth,
            venture_success=snap_f.venture_success_rate if snap_f else 0.0,
            deception_rate=snap_f.deception_rate if snap_f else 0.0,
            n_ticks=ticks,
        ))
        print(f" done ({time.perf_counter()-t0:.1f}s)  gini={results[-1].final_gini:.3f}")

    return results


def write_csv(results: List[SweepResult], path: str = "sweep_results.csv") -> None:
    if not results:
        return
    fieldnames = list(results[0].__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r.__dict__)
    print(f"\n  Results written to {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="InsideTraderSim parameter sweep")
    p.add_argument("--ticks", type=int, default=200)
    p.add_argument("--agents", type=int, default=20)
    args = p.parse_args()

    print("InsideTraderSim Parameter Sweep")
    print("=" * 50)
    results = sweep(ticks=args.ticks, agents=args.agents)
    write_csv(results)

    # Print aggregate summary by signal_cost
    print("\nAggregate Gini by signal_cost (mean across seeds):")
    from collections import defaultdict
    import statistics
    by_cost = defaultdict(list)
    for r in results:
        by_cost[r.signal_cost].append(r.final_gini)
    for cost, ginis in sorted(by_cost.items()):
        print(f"  signal_cost={cost:.1f} → mean Gini={statistics.mean(ginis):.3f}")


if __name__ == "__main__":
    main()
