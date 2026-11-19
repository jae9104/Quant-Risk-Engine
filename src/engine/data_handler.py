from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain.events import MarketEvent
from src.domain.interfaces import DataHandler
from src.engine.event_bus import EventBus

REQUIRED_OHLCV_COLUMNS: frozenset[str] = frozenset({"open", "high", "low", "close", "volume"})


class HistoricalDataHandler(DataHandler):
    """Multi-asset historical bar streamer with strict zero look-ahead isolation."""

    def __init__(
        self,
        event_bus: EventBus,
        data: dict[str, pd.DataFrame],
        max_history: int = 512,
    ) -> None:
        if not data:
            raise ValueError("Data dictionary cannot be empty.")

        self._event_bus = event_bus
        self._symbols: tuple[str, ...] = tuple(sorted(data.keys()))
        self._seen_bars: dict[str, deque[MarketEvent]] = {
            symbol: deque(maxlen=max_history) for symbol in self._symbols
        }
        self._timeline: list[tuple[datetime, list[MarketEvent]]] = self._build_timeline(data)
        self._cursor: int = 0

    @classmethod
    def from_csv_files(
        cls,
        event_bus: EventBus,
        csv_paths: dict[str, Path],
        max_history: int = 512,
    ) -> "HistoricalDataHandler":
        """Load OHLCV CSV files indexed by timestamp into a multi-asset handler."""
        frames: dict[str, pd.DataFrame] = {}
        for symbol, path in csv_paths.items():
            df = pd.read_csv(path, parse_dates=True, index_col=0)
            df.columns = [str(col).strip().lower() for col in df.columns]
            frames[symbol] = df
        return cls(event_bus=event_bus, data=frames, max_history=max_history)

    def _build_timeline(
        self,
        data: dict[str, pd.DataFrame],
    ) -> list[tuple[datetime, list[MarketEvent]]]:
        bars_by_timestamp: dict[datetime, list[MarketEvent]] = defaultdict(list)

        for symbol in self._symbols:
            df = data[symbol].copy()
            df.columns = [str(col).strip().lower() for col in df.columns]

            missing = REQUIRED_OHLCV_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"Symbol '{symbol}' is missing columns: {sorted(missing)}")

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)

            df = df.sort_index()
            opens = df["open"].to_numpy(dtype=np.float64)
            highs = df["high"].to_numpy(dtype=np.float64)
            lows = df["low"].to_numpy(dtype=np.float64)
            closes = df["close"].to_numpy(dtype=np.float64)
            volumes = df["volume"].to_numpy(dtype=np.float64)

            for raw_ts, o, h, low, c, vol in zip(
                df.index, opens, highs, lows, closes, volumes, strict=True
            ):
                ts = self._normalize_timestamp(raw_ts)
                event = MarketEvent(
                    timestamp=ts,
                    symbol=symbol,
                    open=float(o),
                    high=float(h),
                    low=float(low),
                    close=float(c),
                    volume=float(vol),
                )
                bars_by_timestamp[ts].append(event)

        return sorted(bars_by_timestamp.items(), key=lambda item: item[0])

    @staticmethod
    def _normalize_timestamp(raw_ts: object) -> datetime:
        ts = pd.Timestamp(raw_ts)
        if ts.tzinfo is None:
            dt: datetime = ts.to_pydatetime().replace(tzinfo=UTC)
            return dt
        dt_utc: datetime = ts.tz_convert(UTC).to_pydatetime()
        return dt_utc

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def has_next(self) -> bool:
        return self._cursor < len(self._timeline)

    def stream_next(self) -> list[MarketEvent]:
        if not self.has_next:
            return []

        _, events = self._timeline[self._cursor]
        self._cursor += 1

        for event in events:
            self._seen_bars[event.symbol].append(event)
            self._event_bus.publish(event)

        return list(events)

    def get_latest_bar(self, symbol: str) -> MarketEvent | None:
        bars = self._seen_bars.get(symbol)
        if not bars:
            return None
        return bars[-1]

    def get_latest_bars(self, symbol: str, n: int = 1) -> list[MarketEvent]:
        if n <= 0:
            return []
        bars = self._seen_bars.get(symbol)
        if not bars:
            return []
        return list(bars)[-n:]