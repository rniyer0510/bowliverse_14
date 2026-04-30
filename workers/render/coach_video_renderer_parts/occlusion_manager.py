from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .shared import FULL_VISIBILITY, MIN_VISIBILITY_HARD
from .analytics import _safe_float


def _visibility_weight(vis: Optional[float]) -> float:
    value = _safe_float(vis)
    if value is None or value <= MIN_VISIBILITY_HARD:
        return 0.0
    if value >= FULL_VISIBILITY:
        return 1.0
    span = max(1e-6, FULL_VISIBILITY - MIN_VISIBILITY_HARD)
    return max(0.0, min(1.0, (float(value) - MIN_VISIBILITY_HARD) / span))


def _smooth_series(
    values: Iterable[Optional[float]],
    sigma: float,
    *,
    weights: Optional[Iterable[float]] = None,
) -> Optional[np.ndarray]:
    value_list = list(values)
    if not value_list:
        return None
    valid = np.array([value is not None for value in value_list], dtype=bool)
    if valid.sum() < 3:
        return None
    idx = np.arange(len(value_list), dtype=float)
    raw = np.full(len(value_list), np.nan, dtype=float)
    raw[valid] = [float(value) for value in value_list if value is not None]
    interp = raw.copy()
    interp[~valid] = np.interp(idx[~valid], idx[valid], raw[valid])

    if weights is not None:
        weight_arr = np.asarray(list(weights), dtype=float)
        if weight_arr.shape[0] != len(value_list):
            raise ValueError("weights length must match values length")
        weight_arr = np.clip(weight_arr, 0.0, 1.0)
        source = interp.copy()
        finite_raw = np.isfinite(raw)
        source[finite_raw] = (
            interp[finite_raw] * (1.0 - weight_arr[finite_raw])
            + raw[finite_raw] * weight_arr[finite_raw]
        )
    else:
        source = interp

    return gaussian_filter1d(source, sigma=max(1.0, sigma))


def _draw_mode_for_quality(
    quality: float,
    raw_weight: float,
    has_raw_point: bool,
) -> Optional[str]:
    if raw_weight >= 0.72 and has_raw_point:
        return "solid"
    if raw_weight > 0.0 and has_raw_point:
        return "dashed"
    if quality > 0.0:
        return "placeholder"
    return None


def _frame_quality_from_joint_qualities(qualities: Iterable[float], joint_count: int) -> float:
    finite = [max(0.0, float(quality)) for quality in qualities if float(quality) > 0.0]
    if not finite:
        return 0.0
    return float(sum(finite) / max(1, joint_count))


__all__ = [name for name in globals() if name != "__builtins__"]
