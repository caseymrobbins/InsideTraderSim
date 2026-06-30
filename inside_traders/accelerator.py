"""
GPU abstraction layer.

Provides `xp` (CuPy when device="cuda" and CuPy is installed, NumPy otherwise)
and a VectorizedAgentState that keeps positions + cash as a GPU tensor so
batch net-worth, Gini, and POLI calculations happen in a single kernel call.

The cognitive architecture (BeliefGraph, EvidenceLedfger, EpistemicModel) is
inherently graph-structured Python — it stays on CPU. GPU acceleration targets:
  - World GBM dynamics  (n_assets random draws per tick)
  - Batch card generation (n_agents × n_cards draws)
  - Portfolio net-worth  (positions @ prices + cash)
  - Wealth Gini          (sort + cumsum)
  - POLI score vector    (wealth + trade_impact, all agents at once)
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def get_array_module(device: str = "cpu"):
    """Return (xp, device_name) where xp is cupy or numpy."""
    if device == "cuda":
        try:
            import cupy as cp
            # Quick smoke-test — will raise if no GPU is visible
            _ = cp.array([1.0])
            return cp, "cuda"
        except Exception as exc:
            print(f"[accelerator] CuPy unavailable ({exc}); falling back to NumPy/CPU.")
    return np, "cpu"


class VectorizedAgentState:
    """
    Holds per-agent position matrix and cash vector as device tensors.

    Shape:
      positions : (n_agents, n_assets)  — long/short holdings
      cash      : (n_agents,)           — liquid numeraire
      impact    : (n_agents,)           — cumulative |$| traded (for POLI)
    """

    def __init__(
        self,
        n_agents: int,
        n_assets: int,
        initial_cash: np.ndarray,
        xp,
    ) -> None:
        self.xp = xp
        self.n_agents = n_agents
        self.n_assets = n_assets
        self.positions = xp.zeros((n_agents, n_assets), dtype=xp.float64)
        self.cash = xp.asarray(initial_cash, dtype=xp.float64)
        self.impact = xp.zeros(n_agents, dtype=xp.float64)

    # ------------------------------------------------------------------
    # Batch read-outs (stay on device — return device arrays)
    # ------------------------------------------------------------------

    def net_worths(self, prices: "xp.ndarray") -> "xp.ndarray":
        """(n_agents,) net worth vector: positions @ prices + cash."""
        return (self.positions * prices[None, :]).sum(axis=1) + self.cash

    def poli_scores(self, prices: "xp.ndarray", window: int) -> "xp.ndarray":
        """POLI = net_worth + cumulative market impact / window."""
        return self.net_worths(prices) + self.impact / max(window, 1)

    # ------------------------------------------------------------------
    # Sync from/to Python agent objects
    # ------------------------------------------------------------------

    def push_from_agents(self, agents) -> None:
        """Copy positions/cash from Python agent objects → GPU tensors."""
        xp = self.xp
        pos_np = np.stack([ag.positions for ag in agents])   # (n, n_assets)
        cash_np = np.array([ag.cash for ag in agents])
        impact_np = np.array([ag._market_impact_total for ag in agents])
        self.positions = xp.asarray(pos_np, dtype=xp.float64)
        self.cash = xp.asarray(cash_np, dtype=xp.float64)
        self.impact = xp.asarray(impact_np, dtype=xp.float64)

    def pull_to_agents(self, agents) -> None:
        """Copy GPU tensors → Python agent objects (after batch ops)."""
        pos_np = _to_numpy(self.positions)
        cash_np = _to_numpy(self.cash)
        for i, ag in enumerate(agents):
            ag.positions[:] = pos_np[i]
            ag.cash = float(cash_np[i])


# ------------------------------------------------------------------
# Vectorized world dynamics
# ------------------------------------------------------------------

class VectorizedWorld:
    """
    Runs the GBM hidden-state update entirely on the device.
    Also batch-generates all prediction card noise in one call.
    """

    def __init__(
        self,
        n_assets: int,
        n_ventures: int,
        initial_fundamentals: np.ndarray,
        initial_qualities: np.ndarray,
        xp,
        rng_seed: int = 0,
    ) -> None:
        self.xp = xp
        self.n_assets = n_assets
        self.n_ventures = n_ventures
        self.fundamentals = xp.asarray(initial_fundamentals, dtype=xp.float64)
        self.venture_qualities = xp.asarray(initial_qualities, dtype=xp.float64)
        # Use numpy RNG for reproducibility; pull samples to device each tick
        self._rng = np.random.default_rng(rng_seed)

    def step(self, drift: float, volatility: float, jump_prob: float, jump_mag: float) -> np.ndarray:
        """Advance fundamentals by one tick; return CPU copy."""
        xp = self.xp
        n = self.n_assets

        # Generate noise on CPU, push to device
        noise_np = self._rng.standard_normal(n)
        noise = xp.asarray(noise_np, dtype=xp.float64)

        self.fundamentals += drift * self.fundamentals + volatility * self.fundamentals * noise

        # Regime jumps
        jump_mask_np = self._rng.random(n) < jump_prob
        if jump_mask_np.any():
            jump_np = jump_mag * _to_numpy(self.fundamentals) * self._rng.standard_normal(n)
            jump = xp.asarray(jump_np, dtype=xp.float64)
            mask = xp.asarray(jump_mask_np)
            self.fundamentals[mask] += jump[mask]

        self.fundamentals = xp.maximum(self.fundamentals, 1.0)

        # Venture quality drift
        vq_noise = xp.asarray(0.02 * self._rng.standard_normal(self.n_ventures), dtype=xp.float64)
        self.venture_qualities = xp.clip(self.venture_qualities + vq_noise, 0.05, 0.95)

        return _to_numpy(self.fundamentals)

    def batch_card_noise(
        self,
        n_agents: int,
        max_cards: int,
        noise_std: float,
        conf_base: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate noise and confidence for all (agent, card) pairs at once.
        Returns (noise_matrix, conf_matrix) both shape (n_agents, max_cards), on CPU.
        """
        noise = noise_std * self._rng.standard_normal((n_agents, max_cards))
        conf = np.clip(
            conf_base + 0.05 * self._rng.standard_normal((n_agents, max_cards)),
            0.1, 0.99,
        )
        return noise, conf


# ------------------------------------------------------------------
# Gini on device
# ------------------------------------------------------------------

def gini_gpu(wealth_vec: "xp.ndarray", xp) -> float:
    """Gini coefficient computed entirely on device."""
    v = xp.sort(xp.maximum(wealth_vec, 0.0))
    n = len(v)
    s = v.sum()
    if n == 0 or float(s) == 0.0:
        return 0.0
    idx = xp.arange(1, n + 1, dtype=xp.float64)
    return float((2.0 * xp.dot(idx, v) - (n + 1) * s) / (n * s))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_numpy(arr) -> np.ndarray:
    """Convert device array to NumPy (no-op if already NumPy)."""
    if isinstance(arr, np.ndarray):
        return arr
    return arr.get()  # cupy → numpy
