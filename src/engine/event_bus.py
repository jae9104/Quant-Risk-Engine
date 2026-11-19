from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any, TypeVar

from src.domain.events import BaseEvent, DomainEvent

E = TypeVar("E", bound=BaseEvent)


class EventBus(ABC):
    """Abstract pub/sub contract decoupling engine components from event transport."""

    @abstractmethod
    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        """Register a handler for a specific event type."""

    @abstractmethod
    def publish(self, event: DomainEvent) -> None:
        """Enqueue an event for ordered dispatch."""

    @abstractmethod
    def dispatch_next(self) -> DomainEvent | None:
        """Pop the next FIFO event and invoke all registered handlers in order."""

    @abstractmethod
    def drain(self) -> int:
        """Process all queued events until the queue is empty."""

    @property
    @abstractmethod
    def pending_count(self) -> int:
        """Number of events currently waiting in the queue."""

    @property
    @abstractmethod
    def processed_count(self) -> int:
        """Cumulative number of dispatched events."""


class InMemoryEventBus(EventBus):
    """Deterministic, zero-overhead synchronous FIFO bus for high-throughput backtesting."""

    def __init__(self) -> None:
        self._queue: deque[DomainEvent] = deque()
        self._subscribers: dict[type[BaseEvent], list[Callable[[Any], None]]] = defaultdict(list)
        self._processed_count: int = 0

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        handlers = self._subscribers[event_type]
        if handler not in handlers:
            handlers.append(handler)

    def publish(self, event: DomainEvent) -> None:
        self._queue.append(event)

    def dispatch_next(self) -> DomainEvent | None:
        if not self._queue:
            return None

        event = self._queue.popleft()
        for handler in self._subscribers.get(type(event), []):
            handler(event)

        self._processed_count += 1
        return event

    def drain(self) -> int:
        dispatched = 0
        while self._queue:
            self.dispatch_next()
            dispatched += 1
        return dispatched

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def processed_count(self) -> int:
        return self._processed_count