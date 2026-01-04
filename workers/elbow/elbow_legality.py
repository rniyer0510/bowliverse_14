"""
Elbow legality evaluation — ActionLab V14 (LOCKED)

ANGLES ONLY – camera agnostic
"""

from typing import Any, Dict, List, Optional

THRESH_LEGAL = 18.0
THRESH_BORDERLINE = 22.0
MIN_SAMPLES = 3
LATE_WINDOW_SEC = 0.18


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


def _percentile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    vals = sorted(vals)
    idx = int(round(q * (len(vals) - 1)))
    return vals[max(0, min(idx, len(vals) - 1))]


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

    angle_map = _build_angle_map(elbow_signal)
    if not angle_map:
        return {"verdict": "INSUFFICIENT_DATA", "extension_deg": None}

    n = max(3, int(LATE_WINDOW_SEC * fps))
    start = max(uah, rel - n)
    end = rel - 1

    window = [angle_map[f] for f in range(start, end + 1) if f in angle_map]
    if len(window) < MIN_SAMPLES:
        return {"verdict": "INSUFFICIENT_DATA", "extension_deg": None}

    baseline = _percentile(window, 0.10)
    release = _percentile(window, 0.90)

    if baseline is None or release is None:
        return {"verdict": "INSUFFICIENT_DATA", "extension_deg": None}

    extension = max(0.0, release - baseline)

    if extension < THRESH_LEGAL:
        verdict = "LEGAL"
    elif extension <= THRESH_BORDERLINE:
        verdict = "BORDERLINE"
    else:
        verdict = "ILLEGAL"

    return {
        "verdict": verdict,
        "extension_deg": round(extension, 2),
        "baseline_angle_deg": round(baseline, 2),
        "release_angle_deg": round(release, 2),
        "window": {
            "start_frame": start,
            "end_frame": end,
        },
    }

