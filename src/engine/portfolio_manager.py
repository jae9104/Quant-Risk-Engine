import math
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import numpy.typing as npt

from src.domain.events import FillEvent, MarketEvent, OrderSide
from src.domain.models import PortfolioSnapshot, Position, RiskMetrics


class PortfolioManager:
    """Single source of truth for Mark-to-Market accounting and rolling log-returns."""

    def __init__(
        self,
        initial_capital: float,
        lookback_window: int = 252,
    ) -> None:
        if initial_capital <= 0.0:
            raise ValueError("initial_capital must be strictly positive.")
        if lookback_window < 2:
            raise ValueError("lookback_window must be at least 2.")

        self._initial_capital: float = initial_capital
        self._cash: float = initial_capital
        self._lookback_window: int = lookback_window

        self._positions: dict[str, Position] = {}
        self._latest_prices: dict[str, float] = {}
        self._returns: dict[str, OrderedDict[datetime, float]] = {}

        self._realized_pnl: float = 0.0
        self._total_commissions: float = 0.0
        self._total_slippage_cost: float = 0.0
        self._high_water_mark: float = initial_capital
        self._history: list[PortfolioSnapshot] = []

    def on_market_event(self, event: MarketEvent) -> None:
        """Update latest price, rolling log-return buffer, and open position MTM."""
        symbol = event.symbol
        prev_price = self._latest_prices.get(symbol)
        self._latest_prices[symbol] = event.close

        if prev_price is not None:
            log_return = math.log(event.close / prev_price)
            series = self._returns.setdefault(symbol, OrderedDict())
            series[event.timestamp] = log_return
            if len(series) > self._lookback_window:
                series.popitem(last=False)

        if symbol in self._positions:
            pos = self._positions[symbol]
            self._positions[symbol] = pos.model_copy(update={"market_price": event.close})

    def on_fill_event(self, event: FillEvent) -> None:
        """Apply fill to cash and signed position state (supports long, short, and flips)."""
        self._cash += event.net_cash_impact
        self._total_commissions += event.commission
        self._total_slippage_cost += event.slippage * event.quantity

        symbol = event.symbol
        signed_qty = event.quantity if event.side == OrderSide.BUY else -event.quantity
        mark_price = self._latest_prices.get(symbol, event.fill_price)

        existing = self._positions.get(symbol)
        if existing is None:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=signed_qty,
                avg_entry_price=event.fill_price,
                market_price=mark_price,
            )
            return

        prev_qty = existing.quantity
        prev_avg = existing.avg_entry_price
        new_qty = prev_qty + signed_qty

        if prev_qty * signed_qty > 0.0:
            # Adding to an existing position on the same side
            weighted_cost = abs(prev_qty) * prev_avg + abs(signed_qty) * event.fill_price
            new_avg = weighted_cost / abs(new_qty)
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_entry_price=new_avg,
                market_price=mark_price,
            )
            return

        # Reducing, closing, or flipping side
        closed_qty = min(abs(prev_qty), abs(signed_qty))
        direction = 1.0 if prev_qty > 0.0 else -1.0
        self._realized_pnl += closed_qty * (event.fill_price - prev_avg) * direction

        if math.isclose(new_qty, 0.0, abs_tol=1e-9):
            del self._positions[symbol]
        elif new_qty * prev_qty > 0.0:
            # Partial close: keep original average entry price
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_entry_price=prev_avg,
                market_price=mark_price,
            )
        else:
            # Position flipped side: remaining quantity opens at current fill price
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_entry_price=event.fill_price,
                market_price=mark_price,
            )

    def record_snapshot(
        self,
        timestamp: datetime,
        risk_metrics: RiskMetrics | None = None,
    ) -> PortfolioSnapshot:
        """Capture an immutable portfolio snapshot and update High-Water Mark."""
        current_equity = self.equity
        if current_equity > self._high_water_mark:
            self._high_water_mark = current_equity

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self._cash,
            equity=current_equity,
            gross_exposure=self.gross_exposure,
            net_exposure=self.net_exposure,
            drawdown=self.current_drawdown,
            positions=dict(self._positions),
            risk_metrics=risk_metrics,
        )
        self._history.append(snapshot)
        return snapshot

    def get_returns_matrix(self, symbols: Sequence[str]) -> npt.NDArray[np.float64]:
        """Return a timestamp-aligned (T, N) matrix of log-returns for requested symbols."""
        if not symbols:
            return np.empty((0, 0), dtype=np.float64)

        if any(symbol not in self._returns for symbol in symbols):
            return np.empty((0, len(symbols)), dtype=np.float64)

        common_timestamps = sorted(
            set.intersection(*(set(self._returns[symbol].keys()) for symbol in symbols))
        )
        if not common_timestamps:
            return np.empty((0, len(symbols)), dtype=np.float64)

        return np.array(
            [[self._returns[symbol][ts] for symbol in symbols] for ts in common_timestamps],
            dtype=np.float64,
        )

    def get_exposure_vector(self, symbols: Sequence[str]) -> npt.NDArray[np.float64]:
        """Return signed dollar market values aligned with the given symbol order."""
        return np.array(
            [
                self._positions[s].market_value if s in self._positions else 0.0
                for s in symbols
            ],
            dtype=np.float64,
        )

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def equity(self) -> float:
        return self._cash + sum(pos.market_value for pos in self._positions.values())

    @property
    def gross_exposure(self) -> float:
        return sum(abs(pos.market_value) for pos in self._positions.values())

    @property
    def net_exposure(self) -> float:
        return sum(pos.market_value for pos in self._positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def total_commissions(self) -> float:
        return self._total_commissions

    @property
    def current_drawdown(self) -> float:
        if self._high_water_mark <= 0.0:
            return 0.0
        raw_dd = (self._high_water_mark - self.equity) / self._high_water_mark
        return max(0.0, min(1.0, raw_dd))

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def latest_prices(self) -> dict[str, float]:
        return dict(self._latest_prices)

    @property
    def history(self) -> list[PortfolioSnapshot]:
        return list(self._history)