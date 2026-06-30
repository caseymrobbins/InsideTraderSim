"""Experiment framework for 5-model reward comparison."""
from .design import ExperimentConfig, ScenarioConfig, SCENARIOS
from .runner import ExperimentRunner
from .stats import compute_trial_stats, compare_models
from .report import build_report

__all__ = [
    "ExperimentConfig", "ScenarioConfig", "SCENARIOS",
    "ExperimentRunner", "compute_trial_stats", "compare_models",
    "build_report",
]
