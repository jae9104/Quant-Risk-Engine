from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.domain.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalDirection,
    SignalEvent,
)
from src.engine.event_bus import InMemoryEventBus


def _sample_market_event(close: float = 102.0) -> MarketEvent:
    return MarketEvent(
        timestamp=datetime(2025, 1, 15, 14, 30, tzinfo=UTC),
        symbol="SPY",
        open=100.0,
        high=max(105.0, close),
        low=min(99.0, close),
        close=close,
        volume=1_500_000.0,
    )


def test_events_are_immutable() -> None:
    event = _sample_market_event()
    with pytest.raises(ValidationError):
        event.close = 110.0  # type: ignore[misc]


def test_market_event_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        MarketEvent(
            timestamp=datetime(2025, 1, 15, 14, 30),
            symbol="SPY",
            open=100.0,
            high=105.0,
            low=99.0,
            close=102.0,
            volume=1000.0,
        )


def test_market_event_rejects_inconsistent_ohlc() -> None:
    with pytest.raises(ValidationError, match="High price"):
        MarketEvent(
            timestamp=datetime(2025, 1, 15, 14, 30, tzinfo=UTC),
            symbol="SPY",
            open=100.0,
            high=98.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
        )


def test_order_event_validates_limit_price() -> None:
    ts = datetime(2025, 1, 15, 14, 30, tzinfo=UTC)

    with pytest.raises(ValidationError, match="LIMIT orders"):
        OrderEvent(
            timestamp=ts,
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
        )

    with pytest.raises(ValidationError, match="MARKET orders"):
        OrderEvent(
            timestamp=ts,
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            limit_price=100.0,
        )


def test_fill_event_cash_impact_accounting() -> None:
    ts = datetime(2025, 1, 15, 14, 30, tzinfo=UTC)

    buy_fill = FillEvent(
        timestamp=ts,
        order_id=uuid4(),
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=10.0,
        fill_price=100.0,
        commission=2.5,
        slippage=0.1,
    )
    sell_fill = FillEvent(
        timestamp=ts,
        order_id=uuid4(),
        symbol="SPY",
        side=OrderSide.SELL,
        quantity=10.0,
        fill_price=105.0,
        commission=2.5,
        slippage=0.1,
    )

    assert buy_fill.net_cash_impact == -1002.5
    assert sell_fill.net_cash_impact == 1047.5


def test_event_bus_deterministic_fifo_and_chained_events() -> None:
    bus = InMemoryEventBus()
    execution_trace: list[str] = []

    def portfolio_handler(event: MarketEvent) -> None:
        execution_trace.append(f"portfolio:{event.symbol}:{event.close}")

    def strategy_handler(event: MarketEvent) -> None:
        execution_trace.append(f"strategy:{event.symbol}:{event.close}")
        bus.publish(
            SignalEvent(
                timestamp=event.timestamp,
                strategy_id="momentum_v1",
                symbol=event.symbol,
                direction=SignalDirection.LONG,
                strength=0.8,
            )
        )

    def risk_handler(event: SignalEvent) -> None:
        execution_trace.append(f"risk:{event.symbol}:{event.direction}")

    # Subscription order matters: Portfolio must mark-to-market before Strategy runs
    bus.subscribe(MarketEvent, portfolio_handler)
    bus.subscribe(MarketEvent, strategy_handler)
    bus.subscribe(SignalEvent, risk_handler)

    bus.publish(_sample_market_event(close=102.0))
    bus.publish(_sample_market_event(close=104.0))

    assert bus.pending_count == 2
    dispatched = bus.drain()

    assert dispatched == 4
    assert bus.pending_count == 0
    assert bus.processed_count == 4
    # Strict FIFO: initial MarketEvents dispatch before newly enqueued SignalEvents
    assert execution_trace == [
        "portfolio:SPY:102.0",
        "strategy:SPY:102.0",
        "portfolio:SPY:104.0",
        "strategy:SPY:104.0",
        "risk:SPY:LONG",
        "risk:SPY:LONG",
    ]