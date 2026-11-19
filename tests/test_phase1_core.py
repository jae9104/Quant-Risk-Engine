import math
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.domain.events import FillEvent, MarketEvent, OrderSide
from src.engine.data_handler import HistoricalDataHandler
from src.engine.event_bus import InMemoryEventBus
from src.engine.portfolio_manager import PortfolioManager


def _make_ohlcv_frame(dates: list[str], closes: list[float]) -> pd.DataFrame:
    index = pd.to_datetime(dates, utc=True)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "volume": [100_000.0] * len(closes),
        },
        index=index,
    )


def test_data_handler_chronological_streaming_and_zero_lookahead() -> None:
    bus = InMemoryEventBus()
    spy_df = _make_ohlcv_frame(["2025-01-03", "2025-01-01", "2025-01-02"], [105.0, 100.0, 102.0])
    tlt_df = _make_ohlcv_frame(["2025-01-02", "2025-01-03"], [90.0, 91.0])

    handler = HistoricalDataHandler(bus, {"SPY": spy_df, "TLT": tlt_df})

    assert handler.symbols == ("SPY", "TLT")
    assert handler.get_latest_bar("SPY") is None
    assert handler.has_next

    # Step 1: 2025-01-01 (only SPY exists)
    batch_1 = handler.stream_next()
    assert len(batch_1) == 1
    assert batch_1[0].symbol == "SPY"
    assert batch_1[0].close == 100.0
    assert handler.get_latest_bar("TLT") is None
    assert len(handler.get_latest_bars("SPY", n=5)) == 1

    # Step 2: 2025-01-02 (both SPY and TLT exist)
    batch_2 = handler.stream_next()
    assert [e.symbol for e in batch_2] == ["SPY", "TLT"]
    assert handler.get_latest_bar("SPY") is not None
    assert handler.get_latest_bar("TLT") is not None
    assert len(handler.get_latest_bars("SPY", n=5)) == 2

    # Step 3: 2025-01-03
    batch_3 = handler.stream_next()
    assert len(batch_3) == 2
    assert not handler.has_next
    assert bus.pending_count == 5


def test_portfolio_long_short_flip_and_exact_accounting() -> None:
    pm = PortfolioManager(initial_capital=100_000.0)
    ts1 = datetime(2025, 1, 1, tzinfo=UTC)
    ts2 = datetime(2025, 1, 2, tzinfo=UTC)
    ts3 = datetime(2025, 1, 3, tzinfo=UTC)

    # 1. Buy 10 SPY @ 100 (commission = 1)
    pm.on_market_event(
        MarketEvent(
            timestamp=ts1, symbol="SPY", open=100, high=101, low=99, close=100, volume=1000
        )
    )
    pm.on_fill_event(
        FillEvent(
            timestamp=ts1,
            order_id=uuid4(),
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=10.0,
            fill_price=100.0,
            commission=1.0,
            slippage=0.0,
        )
    )
    assert pm.cash == pytest.approx(98_999.0)
    assert pm.equity == pytest.approx(99_999.0)
    assert pm.gross_exposure == pytest.approx(1_000.0)
    assert pm.net_exposure == pytest.approx(1_000.0)

    # 2. Price rises to 110 -> Sell 20 SPY @ 110 (flips from +10 long to -10 short)
    pm.on_market_event(
        MarketEvent(
            timestamp=ts2, symbol="SPY", open=110, high=111, low=109, close=110, volume=1000
        )
    )
    pm.on_fill_event(
        FillEvent(
            timestamp=ts2,
            order_id=uuid4(),
            symbol="SPY",
            side=OrderSide.SELL,
            quantity=20.0,
            fill_price=110.0,
            commission=2.0,
            slippage=0.0,
        )
    )
    assert pm.positions["SPY"].quantity == pytest.approx(-10.0)
    assert pm.positions["SPY"].avg_entry_price == pytest.approx(110.0)
    assert pm.realized_pnl == pytest.approx(100.0)
    assert pm.gross_exposure == pytest.approx(1_100.0)
    assert pm.net_exposure == pytest.approx(-1_100.0)
    assert pm.equity == pytest.approx(100_097.0)

    # 3. Price drops to 100 -> Short profits +100, then Buy 10 @ 100 to flatten
    pm.on_market_event(
        MarketEvent(
            timestamp=ts3, symbol="SPY", open=100, high=101, low=99, close=100, volume=1000
        )
    )
    assert pm.unrealized_pnl == pytest.approx(100.0)
    assert pm.equity == pytest.approx(100_197.0)

    pm.on_fill_event(
        FillEvent(
            timestamp=ts3,
            order_id=uuid4(),
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=10.0,
            fill_price=100.0,
            commission=1.0,
            slippage=0.0,
        )
    )
    assert "SPY" not in pm.positions
    assert pm.realized_pnl == pytest.approx(200.0)
    assert pm.total_commissions == pytest.approx(4.0)
    # Initial 100,000 + 200 realized PnL - 4 total commissions = 100,196
    assert pm.equity == pytest.approx(100_196.0)
    assert pm.cash == pytest.approx(100_196.0)


def test_portfolio_drawdown_and_aligned_log_returns_matrix() -> None:
    pm = PortfolioManager(initial_capital=10_000.0, lookback_window=3)
    dates = [
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
        datetime(2025, 1, 3, tzinfo=UTC),
        datetime(2025, 1, 4, tzinfo=UTC),
    ]
    spy_prices = [100.0, 110.0, 99.0, 108.9]
    tlt_prices = [50.0, 50.0, 55.0, 52.25]

    for i, ts in enumerate(dates):
        pm.on_market_event(
            MarketEvent(
                timestamp=ts,
                symbol="SPY",
                open=spy_prices[i],
                high=spy_prices[i],
                low=spy_prices[i],
                close=spy_prices[i],
                volume=1000.0,
            )
        )
        pm.on_market_event(
            MarketEvent(
                timestamp=ts,
                symbol="TLT",
                open=tlt_prices[i],
                high=tlt_prices[i],
                low=tlt_prices[i],
                close=tlt_prices[i],
                volume=1000.0,
            )
        )
        if i == 0:
            # Buy 100 SPY @ 100 (100% invested, 0 cash)
            pm.on_fill_event(
                FillEvent(
                    timestamp=ts,
                    order_id=uuid4(),
                    symbol="SPY",
                    side=OrderSide.BUY,
                    quantity=100.0,
                    fill_price=100.0,
                    commission=0.0,
                    slippage=0.0,
                )
            )
        pm.record_snapshot(ts)

    # Peak equity on day 2 was 11,000; on day 3 it dropped to 9,900 -> 10% drawdown
    assert pm.history[1].equity == pytest.approx(11_000.0)
    assert pm.history[1].drawdown == pytest.approx(0.0)
    assert pm.history[2].equity == pytest.approx(9_900.0)
    assert pm.history[2].drawdown == pytest.approx(0.10)

    # Aligned log-returns matrix should have shape (3, 2)
    returns_matrix = pm.get_returns_matrix(["SPY", "TLT"])
    assert returns_matrix.shape == (3, 2)
    expected_spy = np.array(
        [math.log(110.0 / 100.0), math.log(99.0 / 110.0), math.log(108.9 / 99.0)]
    )
    expected_tlt = np.array(
        [math.log(50.0 / 50.0), math.log(55.0 / 50.0), math.log(52.25 / 55.0)]
    )
    np.testing.assert_allclose(returns_matrix[:, 0], expected_spy, rtol=1e-12)
    np.testing.assert_allclose(returns_matrix[:, 1], expected_tlt, rtol=1e-12)