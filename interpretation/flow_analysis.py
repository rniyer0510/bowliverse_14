from typing import Any, Dict, List


def analyze_linear_flow(
    risks: List[Dict[str, Any]],
    kinematics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Linear Action Flow (interpretation-only, KEYED OUTPUT)

    LOCKED RULE (V14):
    - 0 or 1 moderate contributor  -> SMOOTH
    - 2 moderate contributors      -> INTERRUPTED
    - ≥3 moderate contributors     -> FRAGMENTED

    NOTE:
    This returns keys + contributors only (no English strings).
    English explanations come from YAML clinician layer.
    """
    contributors = []
    conf_sum = 0.0

    for r in risks or []:
        if float(r.get("signal_strength", 0.0)) >= 0.4:
            contributors.append(r.get("risk_id"))
            conf_sum += float(r.get("confidence", 0.0))

    if len(contributors) <= 1:
        return {
            "flow_state": "SMOOTH",
            "confidence": 0.85,
            "contributors": contributors,
        }

    if len(contributors) == 2:
        flow_state = "INTERRUPTED"
    else:
        flow_state = "FRAGMENTED"

    base_conf = min(1.0, conf_sum / max(1, len(contributors)))

    return {
        "flow_state": flow_state,
        "confidence": round(base_conf, 2),
        "contributors": contributors,
    }
