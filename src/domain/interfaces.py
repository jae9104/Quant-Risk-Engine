from abc import ABC, abstractmethod

from src.domain.events import MarketEvent, OrderEvent, SignalEvent


class DataHandler(ABC):
    """Abstract market data feeder guaranteeing strict chronological delivery."""

    @property
    @abstractmethod
    def symbols(self) -> tuple[str, ...]:
        """Universe of tracked symbols."""

    @property
    @abstractmethod
    def has_next(self) -> bool:
        """True if unconsumed chronological timestamps remain."""

    @abstractmethod
    def stream_next(self) -> list[MarketEvent]:
        """Emit all MarketEvents for the next timestamp onto the event bus."""

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> MarketEvent | None:
        """Return the most recently emitted bar for a symbol (never future data)."""

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> list[MarketEvent]:
        """Return up to the last n emitted bars for a symbol in chronological order."""


class Strategy(ABC):
    """Abstract alpha signal generator."""

    @abstractmethod
    def on_market_event(self, event: MarketEvent) -> None:
        """Process a new market bar and optionally publish a SignalEvent."""


class RiskEngine(ABC):
    """Abstract real-time portfolio risk monitor and pre-trade gatekeeper."""

    @abstractmethod
    def on_market_event(self, event: MarketEvent) -> None:
        """Re-evaluate portfolio risk and trigger liquidations if limits are breached."""

    @abstractmethod
    def on_signal_event(self, event: SignalEvent) -> None:
        """Validate, resize, or reject raw signals before converting them into orders."""


class ExecutionHandler(ABC):
    """Abstract order execution simulator (slippage and commissions)."""

    @abstractmethod
    def on_order_event(self, event: OrderEvent) -> None:
        """Simulate execution of an approved order and publish a FillEvent."""