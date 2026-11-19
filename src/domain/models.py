from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Position(BaseModel):
    """Point-in-time open position state (positive quantity = long, negative = short)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    quantity: float = Field(allow_inf_nan=False)
    avg_entry_price: float = Field(ge=0.0, allow_inf_nan=False)
    market_price: float = Field(gt=0.0, allow_inf_nan=False)

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.market_price - self.avg_entry_price)


class RiskMetrics(BaseModel):
    """Quantitative portfolio risk snapshot computed at timestamp t."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    confidence_level: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    horizon_days: int = Field(default=1, ge=1)
    var_parametric: float = Field(ge=0.0, allow_inf_nan=False)
    var_historical: float = Field(ge=0.0, allow_inf_nan=False)
    var_monte_carlo: float = Field(ge=0.0, allow_inf_nan=False)
    cvar_historical: float = Field(ge=0.0, allow_inf_nan=False)
    cvar_monte_carlo: float = Field(ge=0.0, allow_inf_nan=False)
    portfolio_volatility: float = Field(ge=0.0, allow_inf_nan=False)

    @field_validator("timestamp")
    @classmethod
    def ensure_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamp must be timezone-aware (UTC).")
        return value.astimezone(UTC)


class PortfolioSnapshot(BaseModel):
    """Immutable portfolio state snapshot at timestamp t for audit and reporting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    cash: float = Field(allow_inf_nan=False)
    equity: float = Field(allow_inf_nan=False)
    gross_exposure: float = Field(ge=0.0, allow_inf_nan=False)
    net_exposure: float = Field(allow_inf_nan=False)
    drawdown: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    positions: dict[str, Position] = Field(default_factory=dict)
    risk_metrics: RiskMetrics | None = None

    @field_validator("timestamp")
    @classmethod
    def ensure_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamp must be timezone-aware (UTC).")
        return value.astimezone(UTC)