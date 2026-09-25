"""Interval estimates used by the evaluation of the paper.

Two kinds of uncertainty are reported and they are not interchangeable:

* the *binomial* interval of a BER point, which measures the finite number of
  transmitted symbols. It is a Wilson score interval, which stays inside the unit
  interval and behaves at the small error counts of the high-SNR points, where the
  normal approximation of the plain standard error collapses;
* the *between-realization* dispersion, which measures how much the result depends
  on the jammer or channel realization. It is summarised by the mean, the standard
  error of the mean, the min/max envelope and the quantiles, and it is the quantity
  that shrinks as ``1/n`` with the stratified realizations used by the jammer
  models, so it is worth many realizations rather than many symbols each.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, Optional, Sequence, Tuple

DEFAULT_CONFIDENCE = 0.95
DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


def z_value(confidence: float) -> float:
    """Two-sided normal quantile of a confidence level."""
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    return float(NormalDist().inv_cdf(0.5 + float(confidence) / 2.0))


def wilson_interval(
    errors: int,
    symbols: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Tuple[float, float]:
    """Wilson score interval of a bit error ratio."""
    if isinstance(errors, bool) or not isinstance(errors, (int,)) or errors < 0:
        raise ValueError(f"errors must be a non-negative int, got {errors!r}")
    if isinstance(symbols, bool) or not isinstance(symbols, (int,)) or symbols < 0:
        raise ValueError(f"symbols must be a non-negative int, got {symbols!r}")
    if errors > symbols:
        raise ValueError(f"errors ({errors}) cannot exceed symbols ({symbols})")
    if symbols == 0:
        return (float("nan"), float("nan"))
    z = z_value(confidence)
    n = float(symbols)
    p = float(errors) / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_interval(
    values: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> Dict[str, float]:
    """Mean, standard error, envelope and quantiles of a set of realizations."""
    import numpy as np

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty 1D sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    n = int(array.size)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if n > 1 else 0.0
    half = z_value(confidence) * std / math.sqrt(float(n)) if n > 1 else 0.0
    result: Dict[str, float] = {
        "n": float(n),
        "mean": mean,
        "std": std,
        "sem": float(std / math.sqrt(float(n))) if n > 1 else 0.0,
        "ci_lo": max(0.0, mean - half),
        "ci_hi": mean + half,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    for q in quantiles:
        result[f"q{int(round(100.0 * float(q))):02d}"] = float(np.quantile(array, float(q)))
    return result


def outage_probability(
    values: Sequence[float],
    target: float,
) -> float:
    """Share of realizations whose value exceeds ``target``."""
    import numpy as np

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty 1D sequence")
    return float(np.mean(array > float(target)))


def describe(
    values: Sequence[float],
    target: Optional[float] = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, float]:
    """``mean_interval`` plus the outage probability against ``target``."""
    result = mean_interval(values, confidence=confidence)
    if target is not None:
        result["outage"] = outage_probability(values, float(target))
        result["target"] = float(target)
    return result
