from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from src.domain.models import RiskMetrics

F = TypeVar("F", bound=Callable[..., object])

if TYPE_CHECKING:

    def njit(*args: object, **kwargs: object) -> Callable[[F], F]: ...
else:
    from numba import njit

TRADING_DAYS_PER_YEAR: float = 252.0


@njit(cache=True)
def _monte_carlo_var_cvar_kernel(
    exposures: npt.NDArray[np.float64],
    mean_returns: npt.NDArray[np.float64],
    cholesky_l: npt.NDArray[np.float64],
    confidence_level: float,
    horizon_days: int,
    num_simulations: int,
    seed: int,
) -> tuple[float, float]:
    """JIT-compiled correlated Monte Carlo VaR and CVaR over lower-triangular Cholesky L."""
    np.random.seed(seed)
    n_assets = exposures.shape[0]
    sqrt_h = np.sqrt(float(horizon_days))
    losses = np.empty(num_simulations, dtype=np.float64)
    z = np.empty(n_assets, dtype=np.float64)

    for s in range(num_simulations):
        for i in range(n_assets):
            z[i] = np.random.standard_normal()

        pnl = 0.0
        for i in range(n_assets):
            correlated_z = 0.0
            for j in range(i + 1):
                correlated_z += cholesky_l[i, j] * z[j]
            sim_log_return = horizon_days * mean_returns[i] + sqrt_h * correlated_z
            pnl += exposures[i] * (np.exp(sim_log_return) - 1.0)
        losses[s] = -pnl

    losses.sort()
    pos = confidence_level * (num_simulations - 1)
    lower_idx = int(np.floor(pos))
    upper_idx = int(np.ceil(pos))
    weight = pos - lower_idx
    raw_var = losses[lower_idx] * (1.0 - weight) + losses[upper_idx] * weight
    var_value = raw_var if raw_var > 0.0 else 0.0

    tail_start = upper_idx
    while tail_start > 0 and losses[tail_start - 1] >= raw_var:
        tail_start -= 1

    tail_sum = 0.0
    tail_count = num_simulations - tail_start
    for k in range(tail_start, num_simulations):
        tail_sum += losses[k]

    raw_cvar = tail_sum / tail_count if tail_count > 0 else raw_var
    cvar_value = raw_cvar if raw_cvar > var_value else var_value
    return float(var_value), float(cvar_value)


def _safe_cholesky(cov_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return lower-triangular Cholesky factor, clipping eigenvalues if rank-deficient."""
    symmetric = 0.5 * (cov_matrix + cov_matrix.T)
    if np.allclose(symmetric, 0.0, atol=1e-18):
        return np.zeros_like(symmetric, dtype=np.float64)

    try:
        factor: npt.NDArray[np.float64] = np.linalg.cholesky(symmetric)
        return factor
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        clipped = np.maximum(eigenvalues, 1e-12)
        reconstructed = (eigenvectors * clipped) @ eigenvectors.T
        reconstructed = 0.5 * (reconstructed + reconstructed.T)
        fallback: npt.NDArray[np.float64] = np.linalg.cholesky(reconstructed)
        return fallback


class PortfolioVaRCalculator:
    """Computes Parametric, Historical, and Numba Monte Carlo VaR/CVaR in currency terms."""

    def __init__(
        self,
        confidence_level: float = 0.99,
        horizon_days: int = 1,
        num_simulations: int = 10_000,
        seed: int | None = 42,
    ) -> None:
        if not (0.0 < confidence_level < 1.0):
            raise ValueError("confidence_level must be strictly between 0 and 1.")
        if horizon_days < 1:
            raise ValueError("horizon_days must be at least 1.")
        if num_simulations < 100:
            raise ValueError("num_simulations must be at least 100.")

        self._confidence_level = confidence_level
        self._horizon_days = horizon_days
        self._num_simulations = num_simulations
        self._seed = seed
        self._z_score: float = float(norm.ppf(confidence_level))

    def compute_metrics(
        self,
        timestamp: datetime,
        exposures: npt.NDArray[np.float64],
        returns_matrix: npt.NDArray[np.float64],
        portfolio_equity: float,
    ) -> RiskMetrics:
        """Compute full suite of portfolio VaR/CVaR metrics at timestamp t."""
        self._validate_inputs(exposures, returns_matrix, portfolio_equity)

        if np.allclose(exposures, 0.0, atol=1e-12):
            return RiskMetrics(
                timestamp=timestamp,
                confidence_level=self._confidence_level,
                horizon_days=self._horizon_days,
                var_parametric=0.0,
                var_historical=0.0,
                var_monte_carlo=0.0,
                cvar_historical=0.0,
                cvar_monte_carlo=0.0,
                portfolio_volatility=0.0,
            )

        mean_returns = np.mean(returns_matrix, axis=0, dtype=np.float64)
        cov_matrix = np.atleast_2d(
            np.cov(returns_matrix, rowvar=False, ddof=1)
        ).astype(np.float64, copy=False)

        var_param, daily_dollar_std = self._parametric_var(exposures, mean_returns, cov_matrix)
        var_hist, cvar_hist = self._historical_var_cvar(exposures, returns_matrix)
        var_mc, cvar_mc = self._monte_carlo_var_cvar(exposures, mean_returns, cov_matrix)

        annualized_vol = (daily_dollar_std / portfolio_equity) * float(
            np.sqrt(TRADING_DAYS_PER_YEAR)
        )

        return RiskMetrics(
            timestamp=timestamp,
            confidence_level=self._confidence_level,
            horizon_days=self._horizon_days,
            var_parametric=var_param,
            var_historical=var_hist,
            var_monte_carlo=var_mc,
            cvar_historical=cvar_hist,
            cvar_monte_carlo=cvar_mc,
            portfolio_volatility=max(0.0, annualized_vol),
        )

    def _parametric_var(
        self,
        exposures: npt.NDArray[np.float64],
        mean_returns: npt.NDArray[np.float64],
        cov_matrix: npt.NDArray[np.float64],
    ) -> tuple[float, float]:
        portfolio_mean_pnl = float(exposures @ mean_returns)
        portfolio_variance = float(exposures @ cov_matrix @ exposures)
        daily_dollar_std = float(np.sqrt(max(0.0, portfolio_variance)))

        horizon_mean = self._horizon_days * portfolio_mean_pnl
        horizon_std = float(np.sqrt(self._horizon_days)) * daily_dollar_std
        raw_var = -horizon_mean + self._z_score * horizon_std
        return max(0.0, raw_var), daily_dollar_std

    def _historical_var_cvar(
        self,
        exposures: npt.NDArray[np.float64],
        returns_matrix: npt.NDArray[np.float64],
    ) -> tuple[float, float]:
        simple_returns = np.exp(returns_matrix) - 1.0
        historical_pnl = simple_returns @ exposures
        losses = -historical_pnl * float(np.sqrt(self._horizon_days))

        raw_var = float(np.quantile(losses, self._confidence_level))
        var_value = max(0.0, raw_var)

        tail_losses = losses[losses >= raw_var]
        raw_cvar = float(np.mean(tail_losses)) if tail_losses.size > 0 else raw_var
        cvar_value = max(var_value, raw_cvar)
        return var_value, cvar_value

    def _monte_carlo_var_cvar(
        self,
        exposures: npt.NDArray[np.float64],
        mean_returns: npt.NDArray[np.float64],
        cov_matrix: npt.NDArray[np.float64],
    ) -> tuple[float, float]:
        cholesky_l = _safe_cholesky(cov_matrix)
        seed = (
            self._seed
            if self._seed is not None
            else int(np.random.default_rng().integers(0, 2**31 - 1))
        )
        return _monte_carlo_var_cvar_kernel(
            exposures=np.ascontiguousarray(exposures, dtype=np.float64),
            mean_returns=np.ascontiguousarray(mean_returns, dtype=np.float64),
            cholesky_l=np.ascontiguousarray(cholesky_l, dtype=np.float64),
            confidence_level=self._confidence_level,
            horizon_days=self._horizon_days,
            num_simulations=self._num_simulations,
            seed=seed,
        )

    @staticmethod
    def _validate_inputs(
        exposures: npt.NDArray[np.float64],
        returns_matrix: npt.NDArray[np.float64],
        portfolio_equity: float,
    ) -> None:
        if portfolio_equity <= 0.0:
            raise ValueError("portfolio_equity must be strictly positive.")
        if exposures.ndim != 1:
            raise ValueError("exposures must be a 1D array of shape (N,).")
        if returns_matrix.ndim != 2:
            raise ValueError("returns_matrix must be a 2D array of shape (T, N).")
        if returns_matrix.shape[0] < 2:
            raise ValueError("returns_matrix requires at least 2 observations.")
        if returns_matrix.shape[1] != exposures.shape[0]:
            raise ValueError("Asset count mismatch between exposures and returns_matrix.")