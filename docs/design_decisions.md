# Architecture & Design Decisions (ADR Log)

This document records the core engineering and quantitative trade-offs made in **Quant-Risk-Engine**. Each decision prioritizes deterministic behavior, numerical performance, and realistic market mechanics over unnecessary infrastructure bloat.

---

## ADR-001: Python with Strict Typing & JIT (`Numba`) over Java/C++
* **Status:** Accepted (Phase 0)
* **Context:** Event-driven risk engines sit at the intersection of systems architecture and numerical linear algebra (covariance matrices, Cholesky decomposition, and Monte Carlo paths).
* **Decision:** Build the core engine in Python 3.11+ enforcing strict static typing (`mypy --strict`), runtime contract validation (`Pydantic v2`), and JIT-compiled numerical kernels (`NumPy`/`Numba`).
* **Trade-offs & Rationale:** While C++ or Java dominate ultra-low-latency execution venues, Python is the industry standard for risk prototyping, quantitative analytics, and research-to-production pipelines. Enforcing immutability, abstract interfaces, and `Numba` acceleration delivers clean architecture and sub-millisecond VaR computations without excessive boilerplate.

---

## ADR-002: Deterministic In-Memory FIFO Event Bus over External Brokers
* **Status:** Accepted (Phase 0)
* **Context:** Event-driven trading systems often introduce message brokers (e.g., Redis Streams, Kafka) early in development.
* **Decision:** Implement an abstract `EventBus` interface backed by a synchronous, non-recursive `InMemoryEventBus` (`collections.deque`, $O(1)$ enqueue/dequeue).
* **Trade-offs & Rationale:** External brokers add I/O serialization overhead and operational complexity with zero benefit during offline historical backtesting. An in-memory FIFO queue guarantees **100% deterministic replay**, strict subscriber ordering (Portfolio MTM always executes before Strategy/Risk), and high tick throughput—while keeping the interface decoupled enough to swap in a live broker adapter later.

---

## ADR-003: Validated `float64` over `Decimal` for Numerical State
* **Status:** Accepted (Phase 0)
* **Context:** Financial accounting systems traditionally use fixed-point `Decimal` to prevent floating-point rounding drift, whereas quantitative models operate on IEEE 754 `float64` arrays.
* **Decision:** Use finite `float64` (`allow_inf_nan=False` enforced at event boundaries via Pydantic) across events, portfolio state, and return buffers.
* **Trade-offs & Rationale:** Converting between `Decimal` and `NumPy` `float64` on every tick creates severe latency bottlenecks in rolling covariance and Monte Carlo calculations. Strict boundary validation eliminates silent `NaN`/`Inf` propagation while keeping rounding noise negligible relative to modeled slippage and transaction costs.

---

## ADR-004: Strict Zero Look-Ahead Isolation & UTC-Only Timestamps
* **Status:** Accepted (Phase 0–1)
* **Context:** Vectorized backtesters frequently suffer from accidental look-ahead bias (peeking at future bars) and timezone alignment bugs across multi-asset feeds.
* **Decision:** Encapsulate historical datasets inside `HistoricalDataHandler`, exposing only chronologically streamed bars via a cursor, and reject any timezone-naive timestamps at the domain contract level.
* **Trade-offs & Rationale:** Slower than bulk pandas vectorization, but structurally prevents strategies or risk models from accessing information unavailable at timestamp $t$.

---

## ADR-005: Signed Long/Short Accounting Without Margin or Borrow Modeling
* **Status:** Accepted (Phase 1)
* **Context:** Portfolio risk metrics (VaR/CVaR) and exposure tracking (`gross_exposure` vs. `net_exposure`) are significantly more informative when portfolios can hold hedged long/short positions.
* **Decision:** Support full signed position accounting (longs, shorts, partial closes, and side flips) in `PortfolioManager`, while explicitly excluding broker margin calls, overnight financing, and short-borrow fee curves.
* **Trade-offs & Rationale:** Signed accounting adds minimal code complexity while enabling realistic correlation/hedging tests in the VaR engine. Modeling prime-brokerage margin rules is out of scope and would obscure the core focus on market risk management.