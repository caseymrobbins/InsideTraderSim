"""Order book and public market mechanics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Order:
    order_id: int
    agent_id: str
    asset_idx: int
    quantity: float    # positive = buy, negative = sell
    limit_price: Optional[float]
    tick: int


@dataclass
class Fill:
    order_id: int
    agent_id: str
    asset_idx: int
    quantity: float
    price: float
    tick: int


class OrderBook:
    """
    Simplified continuous double auction with market-impact.
    All orders are market orders that fill immediately at impact-adjusted price.
    """

    def __init__(self, n_assets: int, initial_prices: np.ndarray, impact: float) -> None:
        self.n_assets = n_assets
        self.prices: np.ndarray = initial_prices.copy()
        self.impact = impact          # price move per unit of net order flow
        self._order_counter = 0
        self._volume: np.ndarray = np.zeros(n_assets)  # cumulative per tick

    def reset_tick_volume(self) -> None:
        self._volume[:] = 0.0

    def submit(self, agent_id: str, asset_idx: int, quantity: float, tick: int) -> Fill:
        """Submit a market order; returns the fill."""
        oid = self._order_counter
        self._order_counter += 1

        # Market impact: price moves proportional to signed order size
        impact_move = self.impact * quantity
        exec_price = self.prices[asset_idx] + impact_move * 0.5  # fill at midpoint
        self.prices[asset_idx] += impact_move
        self._volume[asset_idx] += abs(quantity)

        return Fill(
            order_id=oid,
            agent_id=agent_id,
            asset_idx=asset_idx,
            quantity=quantity,
            price=exec_price,
            tick=tick,
        )

    def price(self, asset_idx: int) -> float:
        return float(self.prices[asset_idx])

    def tick_volume(self, asset_idx: int) -> float:
        return float(self._volume[asset_idx])
