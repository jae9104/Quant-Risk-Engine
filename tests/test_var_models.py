from datetime import UTC, datetime

import numpy as np
import pytest
from scipy.stats import norm

from src.risk.validation import kupiec_pof_test
from src.risk.var_calculators import PortfolioVaRCalculator


def test_parametric_var_matches_closed_form_single_asset() -> None:
    rng = np.random.default_rng(123)
    returns = rng.normal(loc=0.0, scale=0.02, size=(500, 1))
    exposures = np.array([100_000.0], dtype=np.float64)
    ts = datetime(2025, 6, 1, tzinfo=UTC)

    calc = PortfolioVaRCalculator(confidence_level=0.95, horizon_days=1, num_simulations=5_000)
    metrics = calc.compute_metrics(ts, exposures, returns, portfolio_equity=100_000.0)

    sample_mean = float(np.mean(returns[:, 0]))
    sample_std = float(np.std(returns[:, 0], ddof=1))
    normal_quantile = float(norm.ppf(0.95))
    expected_var = max(0.0, -100_000.0 * sample_mean + normal_quantile * 100_000.0 * sample_std)

    assert metrics.var_parametric == pytest.approx(expected_var, rel=1e-9)
    assert metrics.cvar_historical >= metrics.var_historical
    assert metrics.cvar_monte_carlo >= metrics.var_monte_carlo


def test_long_short_hedging_reduces_var_on_correlated_assets() -> None:
    rng = np.random.default_rng(7)
    cov = np.array([[0.0004, 0.00036], [0.00036, 0.0004]], dtype=np.float64)  # rho = 0.90
    returns = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=1_000)
    ts = datetime(2025, 6, 1, tzinfo=UTC)

    calc = PortfolioVaRCalculator(
        confidence_level=0.99,
        horizon_days=1,
        num_simulations=10_000,
        seed=42,
    )
    long_only = calc.compute_metrics(
        ts, np.array([50_000.0, 50_000.0]), returns, portfolio_equity=100_000.0
    )
    hedged = calc.compute_metrics(
        ts, np.array([50_000.0, -50_000.0]), returns, portfolio_equity=100_000.0
    )

    assert hedged.var_parametric < long_only.var_parametric * 0.35
    assert hedged.var_historical < long_only.var_historical * 0.35
    assert hedged.var_monte_carlo < long_only.var_monte_carlo * 0.35
    assert hedged.portfolio_volatility < long_only.portfolio_volatility * 0.35


def test_monte_carlo_converges_to_parametric_and_is_deterministic() -> None:
    rng = np.random.default_rng(99)
    returns = rng.normal(loc=0.0, scale=0.01, size=(2_000, 2))
    exposures = np.array([60_000.0, 40_000.0], dtype=np.float64)
    ts = datetime(2025, 6, 1, tzinfo=UTC)

    calc = PortfolioVaRCalculator(
        confidence_level=0.95,
        horizon_days=1,
        num_simulations=25_000,
        seed=101,
    )
    m1 = calc.compute_metrics(ts, exposures, returns, portfolio_equity=100_000.0)
    m2 = calc.compute_metrics(ts, exposures, returns, portfolio_equity=100_000.0)

    # Bit-for-bit deterministic given the same seed
    assert m1.var_monte_carlo == m2.var_monte_carlo
    assert m1.cvar_monte_carlo == m2.cvar_monte_carlo
    # Monte Carlo should closely track analytical parametric VaR under Gaussian returns
    assert m1.var_monte_carlo == pytest.approx(m1.var_parametric, rel=0.05)


def test_singular_covariance_matrix_does_not_crash_cholesky() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(loc=0.0, scale=0.015, size=(100, 1))
    # Perfectly collinear assets produce a rank-1 singular covariance matrix
    collinear_returns = np.hstack([base, base])
    exposures = np.array([50_000.0, 50_000.0], dtype=np.float64)
    ts = datetime(2025, 6, 1, tzinfo=UTC)

    calc = PortfolioVaRCalculator(confidence_level=0.99, num_simulations=2_000, seed=7)
    metrics = calc.compute_metrics(ts, exposures, collinear_returns, portfolio_equity=100_000.0)

    assert metrics.var_monte_carlo > 0.0
    assert metrics.cvar_monte_carlo >= metrics.var_monte_carlo


def test_zero_exposure_returns_zero_risk_metrics() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.02, size=(50, 2))
    exposures = np.array([0.0, 0.0], dtype=np.float64)
    ts = datetime(2025, 6, 1, tzinfo=UTC)

    calc = PortfolioVaRCalculator()
    metrics = calc.compute_metrics(ts, exposures, returns, portfolio_equity=100_000.0)

    assert metrics.var_parametric == 0.0
    assert metrics.var_historical == 0.0
    assert metrics.var_monte_carlo == 0.0
    assert metrics.cvar_historical == 0.0
    assert metrics.cvar_monte_carlo == 0.0
    assert metrics.portfolio_volatility == 0.0


def test_kupiec_pof_accepts_calibrated_and_rejects_miscalibrated_var() -> None:
    # 1. Calibrated: 5 exceptions out of 100 at 95% VaR (exact expected rate = 5%)
    var_line = np.full(100, 1_000.0, dtype=np.float64)
    calibrated_losses = np.full(100, 500.0, dtype=np.float64)
    calibrated_losses[:5] = 1_500.0

    ok = kupiec_pof_test(calibrated_losses, var_line, confidence_level=0.95)
    assert ok.exceptions == 5
    assert ok.lr_statistic == pytest.approx(0.0, abs=1e-12)
    assert ok.p_value == pytest.approx(1.0, abs=1e-12)
    assert ok.passed is True

    # 2. Underestimated risk: 25 exceptions out of 100 at 95% VaR -> rejected
    bad_losses = np.full(100, 500.0, dtype=np.float64)
    bad_losses[:25] = 1_500.0
    bad = kupiec_pof_test(bad_losses, var_line, confidence_level=0.95)
    assert bad.exceptions == 25
    assert bad.p_value < 0.001
    assert bad.passed is False

    # 3. Boundary: 0 exceptions out of 100 at 95% VaR -> overly conservative, no log(0) crash
    zero_breaches = kupiec_pof_test(
        np.full(100, 100.0, dtype=np.float64), var_line, confidence_level=0.95
    )
    assert zero_breaches.exceptions == 0
    assert zero_breaches.passed is False