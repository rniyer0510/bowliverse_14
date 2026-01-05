"""
Elbow legality evaluation — ActionLab V14 (LOCKED)

ANGLES ONLY – camera agnostic

Design goals (NON-NEGOTIABLE):
- Use the full UAH → Release window (do NOT bias late).
- Measure INTERNAL elbow extension relative to elbow angle at UAH.
- Never return INSUFFICIENT_DATA when UAH & Release exist and >=2 valid samples exist.
- Robust to occlusion/jitter: outlier rejection + envelope logic.
- Avoid under-estimating stiff elite actions at low FPS.
- Report confidence explicitly when sample support is weak.
"""

from typing import Any, Dict, List, Optional, Tuple
import math

# ICC thresholds (LOCKED)
THRESH_LEGAL = 18.0
THRESH_BORDERLINE = 22.0

# Minimum valid samples to compute legality
MIN_SAMPLES = 2

# Confidence heuristics (DO NOT affect verdict)
IDEAL_SAMPLES = 5
IDEAL_WINDOW_SEC = 0.10  # 100 ms

# Adaptive sampling (FPS-safe)
MAX_SAMPLES = 7
MIN_STRIDE = 1

# UAH baseline anchoring
UAH_BASELINE_MAX_FRAMES = 2     # hard cap
UAH_BASELINE_RATIO = 0.30       # % of window

# Outlier rejection
MAD_K = 3.5
ANGLE_RANGE = (0.0, 180.0)


# -----------------------------
# Helpers
# -----------------------------

def _build_angle_map(elbow_signal: List[Dict[str, Any]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for it in elbow_signal:
        if not isinstance(it, dict):
            continue
        f = it.get("frame")
        a = it.get("angle_deg")
        if f is None or a is None or not it.get("valid", False):
            continue
        try:
            out[int(f)] = float(a)
        except Exception:
            pass
    return out


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return 0.5 * (float(s[mid - 1]) + float(s[mid]))


def _percentile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    vals = sorted(vals)
    q = max(0.0, min(1.0, q))
    idx = int(round(q * (len(vals) - 1)))
    return vals[max(0, min(idx, len(vals) - 1))]


def _mad(vals: List[float], med: float) -> float:
    dev = [abs(v - med) for v in vals]
    m = _median(dev)
    return float(m) if m is not None else 0.0


def _robust_filter(vals: List[float]) -> Tuple[List[float], Dict[str, Any]]:
    """
    Remove:
    - impossible angles
    - single-frame spikes using MAD
    """
    dbg: Dict[str, Any] = {
        "raw_n": len(vals),
        "range_dropped": 0,
        "mad_dropped": 0,
    }

    lo, hi = ANGLE_RANGE
    in_range: List[float] = []
    for v in vals:
        if lo <= v <= hi and not math.isnan(v):
            in_range.append(v)
        else:
            dbg["range_dropped"] += 1

    if len(in_range) < MIN_SAMPLES:
        return in_range, dbg

    med = _median(in_range)
    if med is None:
        return in_range, dbg

    mad = _mad(in_range, med)
    dbg["median"] = round(float(med), 3)
    dbg["mad"] = round(float(mad), 6)

    # If MAD ~ 0, data is already very stable
    if mad < 1e-6:
        return in_range, dbg

    keep: List[float] = []
    for v in in_range:
        if abs(v - med) <= MAD_K * mad:
            keep.append(v)
        else:
            dbg["mad_dropped"] += 1

    return keep, dbg


def _adaptive_sample(vals: List[float], max_samples: int) -> List[float]:
    """
    Reduce dense signals safely without biasing peak.
    """
    n = len(vals)
    if n <= max_samples:
        return vals

    stride = max(MIN_STRIDE, int(math.floor(n / max_samples)))
    sampled = vals[::stride]

    if sampled[-1] != vals[-1]:
        sampled.append(vals[-1])

    return sampled


# -----------------------------
# Main entry
# -----------------------------

def evaluate_elbow_legality(
    elbow_signal: List[Dict[str, Any]],
    events: Dict[str, Any],
    fps: float,
) -> Dict[str, Any]:

    if not elbow_signal or fps <= 0:
        return {"verdict": "INSUFFICIENT_DATA", "extension_deg": None}

    try:
        uah = int(events["uah"]["frame"])
        rel = int(events["release"]["frame"])
    except Exception:
        return {"verdict": "INSUFFICIENT_DATA", "extension_deg": None}

    if rel <= uah:
        return {"verdict": "INSUFFICIENT_DATA", "extension_deg": None}

    angle_map = _build_angle_map(elbow_signal)
    if not angle_map:
        return {"verdict": "INSUFFICIENT_DATA", "extension_deg": None}

    start = uah
    end = max(uah, rel - 1)

    window_vals = [angle_map[f] for f in range(start, end + 1) if f in angle_map]

    filtered_vals, filt_dbg = _robust_filter(window_vals)
    filtered_vals = _adaptive_sample(filtered_vals, MAX_SAMPLES)

    if len(filtered_vals) < MIN_SAMPLES:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "extension_deg": None,
            "window": {"start_frame": start, "end_frame": end},
            "debug": {"filter": filt_dbg, "valid_samples": len(filtered_vals)},
        }

    # -----------------------------
    # Baseline: anchored tightly at UAH
    # -----------------------------
    win_len = len(filtered_vals)
    early_n = max(
        1,
        min(
            UAH_BASELINE_MAX_FRAMES,
            int(math.ceil(win_len * UAH_BASELINE_RATIO)),
        ),
    )

    early_vals = filtered_vals[:early_n]
    baseline = _median(early_vals)

    if baseline is None:
        return {"verdict": "INSUFFICIENT_DATA", "extension_deg": None}

    # -----------------------------
    # Peak extension logic (KEY FIX)
    # -----------------------------
    # If sample support is sparse (low FPS, elite action),
    # percentile underestimates excursion → use MAX.
    if len(filtered_vals) >= 6:
        peak = _percentile(filtered_vals, 0.90)
        peak_method = "p90"
    else:
        peak = max(filtered_vals)
        peak_method = "max"

    if peak is None:
        return {"verdict": "INSUFFICIENT_DATA", "extension_deg": None}

    extension = max(0.0, float(peak) - float(baseline))

    # -----------------------------
    # Verdict
    # -----------------------------
    if extension < THRESH_LEGAL:
        verdict = "LEGAL"
    elif extension <= THRESH_BORDERLINE:
        verdict = "BORDERLINE"
    else:
        verdict = "ILLEGAL"

    win_sec = max(0.0, (end - start + 1) / fps)
    low_conf = (len(filtered_vals) < IDEAL_SAMPLES) or (win_sec < IDEAL_WINDOW_SEC)

    out: Dict[str, Any] = {
        "verdict": verdict if not low_conf else f"{verdict}_LOW_CONFIDENCE",
        "extension_deg": round(extension, 2),
        "baseline_angle_deg": round(float(baseline), 2),
        "release_angle_deg": round(float(peak), 2),
        "window": {
            "start_frame": start,
            "end_frame": end,
        },
        "samples": {
            "valid": int(len(filtered_vals)),
            "window_sec": round(win_sec, 3),
        },
    }

    out["debug"] = {
        "filter": filt_dbg,
        "raw_window_samples": int(len(window_vals)),
        "uah_baseline_samples": early_n,
        "peak_method": peak_method,
    }

    return out

