"""World: hidden states, fundamental dynamics, prediction card distribution."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import SimConfig
from .accelerator import VectorizedWorld, _to_numpy


@dataclass
class PredictionCard:
    """Private signal issued to one agent for one tick."""
    card_id: int
    agent_id: str
    asset_idx: int
    predicted_value: float    # noisy observation of fundamental
    true_value: float         # actual fundamental (for metric calculation)
    confidence: float         # card-level confidence hint
    horizon: int              # absolute tick the prediction resolves
    is_premium: bool          # premium (low-noise) card
    cost: float               # how much numeraire was charged (0 for free cards)


class World:
    """
    Hidden state layer.

    fundamentals: shape (n_assets,)  — true asset prices / quality indices
    venture_qualities: shape (n_ventures,) — latent quality of each venture type
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator, xp=None) -> None:
        self.cfg = cfg
        self.rng = rng
        self.tick: int = 0
        self._xp = xp if xp is not None else np

        # Initialise fundamentals on CPU first, then hand off to VectorizedWorld
        init_f = rng.uniform(80.0, 120.0, cfg.n_assets)
        init_q = rng.uniform(0.3, 0.9, cfg.n_ventures)

        self._gpu_world = VectorizedWorld(
            n_assets=cfg.n_assets,
            n_ventures=cfg.n_ventures,
            initial_fundamentals=init_f,
            initial_qualities=init_q,
            xp=self._xp,
            rng_seed=cfg.seed,
        )

        # CPU-side copies (kept in sync after each step)
        self.fundamentals: np.ndarray = init_f.copy()
        self.venture_qualities: np.ndarray = init_q.copy()

        # History for metric evaluation
        self.fundamental_history: List[np.ndarray] = [self.fundamentals.copy()]
        self._card_counter = 0

    # ------------------------------------------------------------------
    # State evolution
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance hidden state by one tick (runs on device via VectorizedWorld)."""
        self.tick += 1
        cfg = self.cfg

        # GPU/CPU update — returns a CPU numpy array
        self.fundamentals = self._gpu_world.step(
            drift=cfg.asset_drift,
            volatility=cfg.asset_volatility,
            jump_prob=cfg.jump_probability,
            jump_mag=cfg.jump_magnitude,
        )
        self.venture_qualities = _to_numpy(self._gpu_world.venture_qualities)
        self.fundamental_history.append(self.fundamentals.copy())

    # ------------------------------------------------------------------
    # Prediction card distribution
    # ------------------------------------------------------------------

    def issue_free_cards(self, agent_id: str, n_cards: int) -> List[PredictionCard]:
        """Issue n_cards low-quality free prediction cards to an agent."""
        return self._make_cards(agent_id, n_cards, premium=False, cost=0.0)

    def issue_premium_card(self, agent_id: str) -> Tuple[PredictionCard, float]:
        """Issue one premium card; return (card, cost)."""
        cost = self.cfg.signal_base_cost
        cards = self._make_cards(agent_id, 1, premium=True, cost=cost)
        return cards[0], cost

    def issue_exclusive_signal(self, agent_id: str) -> Tuple[PredictionCard, float]:
        """Issue an exclusive (very high quality) signal; higher cost."""
        cost = self.cfg.signal_base_cost + self.cfg.exclusive_signal_premium
        cards = self._make_cards(agent_id, 1, premium=True, cost=cost, exclusive=True)
        return cards[0], cost

    def _make_cards(
        self, agent_id: str, n: int, premium: bool, cost: float, exclusive: bool = False
    ) -> List[PredictionCard]:
        cards = []
        noise_std = (
            self.cfg.card_noise_high_quality * 0.5 if exclusive
            else (self.cfg.card_noise_high_quality if premium else self.cfg.card_noise_low_quality)
        )
        conf_base = 0.85 if exclusive else (0.70 if premium else 0.45)

        for _ in range(n):
            asset_idx = int(self.rng.integers(0, self.cfg.n_assets))
            true_val = self.fundamentals[asset_idx]
            noisy_val = true_val * (1 + noise_std * self.rng.standard_normal())
            horizon = self.tick + int(self.rng.integers(3, 15))
            confidence = float(np.clip(conf_base + 0.05 * self.rng.standard_normal(), 0.1, 0.99))

            cards.append(PredictionCard(
                card_id=self._card_counter,
                agent_id=agent_id,
                asset_idx=asset_idx,
                predicted_value=noisy_val,
                true_value=true_val,
                confidence=confidence,
                horizon=horizon,
                is_premium=premium,
                cost=cost / n if n > 0 else cost,
            ))
            self._card_counter += 1
        return cards

    # ------------------------------------------------------------------
    # Venture quality revelation
    # ------------------------------------------------------------------

    def venture_quality_at(self, venture_asset_idx: int) -> float:
        """Return the hidden quality relevant for a given venture's asset."""
        # Map asset index to venture quality via modulo
        vq_idx = venture_asset_idx % self.cfg.n_ventures
        return float(self.venture_qualities[vq_idx])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def fundamental_at(self, asset_idx: int, tick: Optional[int] = None) -> float:
        if tick is None or tick >= len(self.fundamental_history):
            return float(self.fundamentals[asset_idx])
        return float(self.fundamental_history[tick][asset_idx])
