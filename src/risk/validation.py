import math

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import chi2


class KupiecTestResult(BaseModel):
    """Statistical outcome of the Kupiec Proportion of Failures (POF) VaR backtest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observations: int = Field(ge=1)
    exceptions: int = Field(ge=0)
    expected_exceptions: float = Field(ge=0.0, allow_inf_nan=False)
    exception_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    lr_statistic: float = Field(ge=0.0, allow_inf_nan=False)
    p_value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    passed: bool


def kupiec_pof_test(
    realized_losses: npt.NDArray[np.float64],
    var_forecasts: npt.NDArray[np.float64],
    confidence_level: float = 0.99,
    significance_level: float = 0.05,
) -> KupiecTestResult:
    """Run Kupiec's unconditional coverage likelihood-ratio test on VaR forecasts.

    H0: True failure rate equals (1 - confidence_level).
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if not (0.0 < significance_level < 1.0):
        raise ValueError("significance_level must be strictly between 0 and 1.")
    if realized_losses.ndim != 1 or var_forecasts.ndim != 1:
        raise ValueError("realized_losses and var_forecasts must be 1D arrays.")
    if realized_losses.shape[0] != var_forecasts.shape[0]:
        raise ValueError("realized_losses and var_forecasts must have equal length.")
    if realized_losses.shape[0] == 0:
        raise ValueError("Input arrays cannot be empty.")

    n = int(realized_losses.shape[0])
    x = int(np.sum(realized_losses > var_forecasts))
    expected_rate = 1.0 - confidence_level
    observed_rate = x / n

    if x == 0:
        log_l0 = n * math.log(1.0 - expected_rate)
        lr_stat = max(0.0, -2.0 * log_l0)
    elif x == n:
        log_l0 = n * math.log(expected_rate)
        lr_stat = max(0.0, -2.0 * log_l0)
    else:
        log_l0 = (n - x) * math.log(1.0 - expected_rate) + x * math.log(expected_rate)
        log_l1 = (n - x) * math.log(1.0 - observed_rate) + x * math.log(observed_rate)
        lr_stat = max(0.0, -2.0 * (log_l0 - log_l1))

    p_value = float(chi2.sf(lr_stat, df=1))

    return KupiecTestResult(
        observations=n,
        exceptions=x,
        expected_exceptions=n * expected_rate,
        exception_rate=observed_rate,
        lr_statistic=lr_stat,
        p_value=p_value,
        passed=p_value >= significance_level,
    )