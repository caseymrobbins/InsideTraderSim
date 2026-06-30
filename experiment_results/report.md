# InsideTraderSim — Reward Model Experiment Report

_Generated: 2026-06-30 07:34:41_

## Overview

- **Trials**: 50  (5 per cell)
- **Models**: U, UF, UH, UHF, UHFS
- **Scenarios**: high_vol, moderate_vol
- **Mixed-model trials**: 10

## Scenario: high_vol

| Model | Final Gini Coefficient | Final Mean EH | Wealth Growth | Deception Rate | Mean Reward Signal | Total Danger-Zone Agent-Ticks |
|-------|-------|-------|-------|-------|-------|-------|
| **U** | 0.4106 ± 0.0771 | 0.2004 ± 0.0077 | 1.2863 ± 0.5365 | 0.0000 ± 0.0000 | nan ± nan | 0.0000 ± 0.0000 |
| **UF** | 0.4303 ± 0.0699 | 0.2039 ± 0.0029 | 1.1561 ± 0.6285 | 0.0000 ± 0.0000 | 18.3862 ± 9.5488 | 55.0000 ± 30.8950 |
| **UH** | 0.3908 ± 0.0761 | 0.2017 ± 0.0048 | 1.5936 ± 0.6479 | 0.0000 ± 0.0000 | -0.9545 ± 1.7048 | 42.6000 ± 15.8840 |
| **UHF** | 0.3908 ± 0.0761 | 0.2017 ± 0.0048 | 1.5936 ± 0.6479 | 0.0000 ± 0.0000 | -0.5115 ± 1.4985 | 42.6000 ± 15.8840 |
| **UHFS** | 0.4007 ± 0.0887 | 0.2020 ± 0.0064 | 1.1077 ± 0.5513 | 0.0000 ± 0.0000 | -1.9781 ± 1.6019 | 39.0000 ± 26.6927 |

### Significant pairwise differences (p < 0.05)

| Pair | Metric | p-value | Cohen's d | Direction |
|------|--------|---------|-----------|-----------|
| U vs UF | Total Danger-Zone Agent-Ticks | 0.0090 | -2.518 | U < UF |
| U vs UH | Total Danger-Zone Agent-Ticks | 0.0090 | -3.793 | U < UH |
| U vs UHF | Total Danger-Zone Agent-Ticks | 0.0090 | -3.793 | U < UHF |
| U vs UHFS | Total Danger-Zone Agent-Ticks | 0.0090 | -2.066 | U < UHFS |
| UF vs UH | Mean Reward Signal | 0.0090 | +2.820 | UF > UH |
| UF vs UHF | Mean Reward Signal | 0.0090 | +2.765 | UF > UHF |
| UF vs UHFS | Mean Reward Signal | 0.0090 | +2.974 | UF > UHFS |

## Scenario: moderate_vol

| Model | Final Gini Coefficient | Final Mean EH | Wealth Growth | Deception Rate | Mean Reward Signal | Total Danger-Zone Agent-Ticks |
|-------|-------|-------|-------|-------|-------|-------|
| **U** | 0.4680 ± 0.0434 | 0.2473 ± 0.0027 | 0.7012 ± 0.3416 | 0.0000 ± 0.0000 | nan ± nan | 0.0000 ± 0.0000 |
| **UF** | 0.4860 ± 0.1369 | 0.2514 ± 0.0065 | 0.6776 ± 0.4173 | 0.0000 ± 0.0000 | 8.4819 ± 10.7385 | 91.0000 ± 67.2198 |
| **UH** | 0.4802 ± 0.0828 | 0.2505 ± 0.0035 | 0.8058 ± 0.3901 | 0.0000 ± 0.0000 | -4.2298 ± 2.8431 | 86.2000 ± 29.3888 |
| **UHF** | 0.4802 ± 0.0828 | 0.2505 ± 0.0035 | 0.8058 ± 0.3901 | 0.0000 ± 0.0000 | -4.5208 ± 3.8082 | 86.2000 ± 29.3888 |
| **UHFS** | 0.4700 ± 0.0580 | 0.2456 ± 0.0044 | 0.7171 ± 0.5054 | 0.0000 ± 0.0000 | -4.7335 ± 1.8697 | 82.0000 ± 41.6833 |

### Significant pairwise differences (p < 0.05)

| Pair | Metric | p-value | Cohen's d | Direction |
|------|--------|---------|-----------|-----------|
| U vs UF | Total Danger-Zone Agent-Ticks | 0.0090 | -1.915 | U < UF |
| U vs UH | Total Danger-Zone Agent-Ticks | 0.0090 | -4.148 | U < UH |
| U vs UHF | Total Danger-Zone Agent-Ticks | 0.0090 | -4.148 | U < UHF |
| U vs UHFS | Total Danger-Zone Agent-Ticks | 0.0090 | -2.782 | U < UHFS |
| UF vs UH | Mean Reward Signal | 0.0472 | +1.618 | UF > UH |
| UF vs UHF | Mean Reward Signal | 0.0472 | +1.614 | UF > UHF |
| UF vs UHFS | Compressed Agent Final EH | 0.0472 | +1.440 | UF > UHFS |
| UF vs UHFS | Mean Reward Signal | 0.0472 | +1.715 | UF > UHFS |

## Compression Test Analysis

| Model | Compressed EH | Wealth Ratio | Danger-Zone Ticks |
|-------|--------------|--------------|-------------------|
| **U** | 0.2213 | 1.820 | 0.0 |
| **UF** | 0.2330 | 1.769 | 73.0 |
| **UH** | 0.2264 | 2.061 | 64.4 |
| **UHF** | 0.2264 | 2.061 | 64.4 |
| **UHFS** | 0.2209 | 1.754 | 60.5 |

### Interpretation

- **UF** yields the highest epistemic health among compressed agents, suggesting it best counters EH degradation.
- **UH** produces the highest compressed/non-compressed wealth ratio (2.061), indicating more extractive dynamics under this model.

## Mixed-Model Test

- **Final Gini Coefficient**: 0.4302 ± 0.0757
- **Final Mean EH**: 0.2243 ± 0.0267
- **Wealth Growth**: 1.2164 ± 0.5897
- **Deception Rate**: 0.0000 ± 0.0000

When agents are assigned heterogeneous reward models, emergent coordination (or conflict) arises from the interaction of agents optimizing different objective functions.

## Appendix: Full Metric Table (all scenarios combined)

| Model | Final Gini Coefficient | Final Mean EH | Final Mean POLI×EH | Venture Success Rate | Deception Rate | Mean Trust | Wealth Growth | EH Trajectory Slope | Compressed Agent Final EH | Compressed Agent Final POLI | Non-Compressed Final Wealth | Mean Reward Signal | Total Danger-Zone Agent-Ticks | Mean Gini (time avg) |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **U** | 0.4393 | 0.2238 | 5033.2020 | 0.7437 | 0.0000 | 0.4483 | 0.9938 | -0.0001 | 0.2213 | 21352.6157 | 665.3534 | nan | 0.0000 | 0.4179 |
| **UF** | 0.4582 | 0.2276 | 5226.6612 | 0.4750 | 0.0000 | 0.4518 | 0.9168 | -0.0001 | 0.2330 | 23114.7312 | 659.4207 | 13.4340 | 73.0000 | 0.4421 |
| **UH** | 0.4355 | 0.2261 | 4803.9142 | 0.6300 | 0.0000 | 0.4439 | 1.1997 | -0.0001 | 0.2264 | 21588.5591 | 637.8170 | -2.5921 | 64.4000 | 0.4359 |
| **UHF** | 0.4355 | 0.2261 | 4803.9142 | 0.6300 | 0.0000 | 0.4439 | 1.1997 | -0.0001 | 0.2264 | 21588.5591 | 637.8170 | -2.5161 | 64.4000 | 0.4359 |
| **UHFS** | 0.4353 | 0.2238 | 4544.9141 | 0.4500 | 0.0000 | 0.4362 | 0.9124 | -0.0001 | 0.2209 | 21041.7354 | 644.9984 | -3.3558 | 60.5000 | 0.4244 |
