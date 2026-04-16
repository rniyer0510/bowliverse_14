from collections import Counter
from statistics import median
from typing import Any, Dict, List, Optional

BASELINE_WINDOW = 4
MIN_BASELINE_RUNS = 3
MIN_ALLOWED_DELTA_DEG = 6.0
SPREAD_CUSHION_DEG = 3.0
WATCH_BUFFER_DEG = 4.0


def _shape_payload(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result_json, dict):
        return None
    release_shape = result_json.get("release_shape")
    if not isinstance(release_shape, dict) or not release_shape.get("available"):
        return None
    category = release_shape.get("category") or {}
    geometry = release_shape.get("release_geometry") or {}
    key = category.get("key")
    angle = geometry.get("angle_deg")
    if not isinstance(key, str) or not isinstance(angle, (int, float)):
        return None
    return {
        "category_key": key,
        "category_label": category.get("label"),
        "angle_deg": float(angle),
    }


def _drift_payload(current: Dict[str, Any], baseline: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_angles = [float(item["angle_deg"]) for item in baseline]
    baseline_median = float(median(baseline_angles))
    delta_deg = round(float(current["angle_deg"]) - baseline_median, 1)
    abs_delta = abs(delta_deg)
    baseline_spread = max(abs(angle - baseline_median) for angle in baseline_angles)
    allowed_delta = max(MIN_ALLOWED_DELTA_DEG, baseline_spread + SPREAD_CUSHION_DEG)
    watch_limit = allowed_delta + WATCH_BUFFER_DEG
    baseline_mode = Counter(str(item["category_key"]) for item in baseline).most_common(1)[0][0]
    current_key = str(current["category_key"])
    current_label = str(current.get("category_label") or current_key.replace("_", " "))
    baseline_label = baseline_mode.replace("_", " ").title()

    if current_key != baseline_mode and abs_delta > allowed_delta:
        status = "clear_change"
        summary = f"Release shape looks more {current_label.lower()} than the usual {baseline_label.lower()} pattern."
    elif abs_delta > watch_limit:
        status = "clear_change"
        summary = "Release angle is clearly drifting away from the usual pattern."
    elif abs_delta > allowed_delta:
        status = "watch_change"
        summary = "Release angle is starting to move away from the usual pattern."
    else:
        status = "within_range"
        summary = "Release shape is still close to the usual pattern."

    return {
        "available": True,
        "status": status,
        "summary": summary,
        "delta_deg": delta_deg,
        "baseline_angle_deg": round(baseline_median, 1),
        "baseline_category_key": baseline_mode,
        "allowed_delta_deg": round(allowed_delta, 1),
    }


def enrich_release_shape_drift(run_entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    trusted = [
        entry for entry in run_entries
        if entry.get("trusted") and _shape_payload(entry.get("result_json"))
    ]
    result: Dict[str, Dict[str, Any]] = {}
    for index, entry in enumerate(trusted):
        run_id = str(entry.get("run_id") or "")
        current = _shape_payload(entry.get("result_json"))
        baseline_entries = trusted[index + 1:index + 1 + BASELINE_WINDOW]
        baseline = [_shape_payload(item.get("result_json")) for item in baseline_entries]
        baseline = [item for item in baseline if item]
        if current is None:
            continue
        if len(baseline) < MIN_BASELINE_RUNS:
            result[run_id] = {
                "available": False,
                "status": "collecting",
                "summary": "Need a few more trusted runs before release-shape drift can be judged.",
                "delta_deg": None,
            }
            continue
        result[run_id] = _drift_payload(current, baseline)
    return result
