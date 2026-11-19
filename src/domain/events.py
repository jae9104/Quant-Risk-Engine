from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventType(StrEnum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    RISK_ALERT = "RISK_ALERT"


class SignalDirection(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class RiskAction(StrEnum):
    APPROVED = "APPROVED"
    RESIZED = "RESIZED"
    REJECTED = "REJECTED"
    LIQUIDATE = "LIQUIDATE"


class BaseEvent(BaseModel):
    """Immutable base contract for all engine events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def ensure_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamp must be timezone-aware (UTC).")
        return value.astimezone(UTC)


class MarketEvent(BaseEvent):
    """OHLCV market bar emitted at timestamp t."""

    event_type: Literal[EventType.MARKET] = EventType.MARKET
    symbol: str = Field(min_length=1)
    open: float = Field(gt=0.0, allow_inf_nan=False)
    high: float = Field(gt=0.0, allow_inf_nan=False)
    low: float = Field(gt=0.0, allow_inf_nan=False)
    close: float = Field(gt=0.0, allow_inf_nan=False)
    volume: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_ohlc_consistency(self) -> Self:
        if self.high < max(self.open, self.close):
            raise ValueError("High price cannot be lower than open or close.")
        if self.low > min(self.open, self.close):
            raise ValueError("Low price cannot be higher than open or close.")
        return self


class SignalEvent(BaseEvent):
    """Raw directional trading intent prior to pre-trade risk validation."""

    event_type: Literal[EventType.SIGNAL] = EventType.SIGNAL
    strategy_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    direction: SignalDirection
    strength: float = Field(default=1.0, gt=0.0, le=1.0, allow_inf_nan=False)


class OrderEvent(BaseEvent):
    """Risk-approved and sized order routed to the execution simulator."""

    event_type: Literal[EventType.ORDER] = EventType.ORDER
    symbol: str = Field(min_length=1)
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(gt=0.0, allow_inf_nan=False)
    limit_price: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    is_risk_liquidation: bool = False
    parent_signal_id: UUID | None = None

    @model_validator(mode="after")
    def validate_order_pricing(self) -> Self:
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT orders must specify a limit_price.")
        if self.order_type == OrderType.MARKET and self.limit_price is not None:
            raise ValueError("MARKET orders cannot specify a limit_price.")
        return self


class FillEvent(BaseEvent):
    """Execution confirmation. fill_price already reflects realized slippage."""

    event_type: Literal[EventType.FILL] = EventType.FILL
    order_id: UUID
    symbol: str = Field(min_length=1)
    side: OrderSide
    quantity: float = Field(gt=0.0, allow_inf_nan=False)
    fill_price: float = Field(gt=0.0, allow_inf_nan=False)
    commission: float = Field(ge=0.0, allow_inf_nan=False)
    slippage: float = Field(ge=0.0, allow_inf_nan=False)

    @property
    def notional_value(self) -> float:
        return self.quantity * self.fill_price

    @property
    def net_cash_impact(self) -> float:
        if self.side == OrderSide.BUY:
            return -(self.notional_value + self.commission)
        return self.notional_value - self.commission


class RiskAlertEvent(BaseEvent):
    """Audit trail emitted whenever the pre-trade or real-time risk engine intervenes."""

    event_type: Literal[EventType.RISK_ALERT] = EventType.RISK_ALERT
    action: RiskAction
    reason: str = Field(min_length=1)
    symbol: str | None = None
    current_var: float = Field(ge=0.0, allow_inf_nan=False)
    var_limit: float = Field(gt=0.0, allow_inf_nan=False)
    original_quantity: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    adjusted_quantity: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)


DomainEvent = Annotated[
    MarketEvent | SignalEvent | OrderEvent | FillEvent | RiskAlertEvent,
    Field(discriminator="event_type"),
]