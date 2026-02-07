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
    Flow confidence reflects certainty of rhythm assessment,
    NOT injury risk magnitude.
    """
    contributors = []
    conf_sum = 0.0

    for r in risks or []:
        strength = float(r.get("signal_strength", 0.0))
        conf = float(r.get("confidence", 0.0))

        if strength >= 0.4:
            contributors.append(r.get("risk_id"))
            conf_sum += conf * strength

    if len(contributors) <= 1:
        avg_conf = (
            conf_sum / max(1, len(contributors))
            if contributors else 0.9
        )
        return {
            "flow_state": "SMOOTH",
            "confidence": round(avg_conf, 2),
            "contributors": contributors,
        }

    flow_state = "INTERRUPTED" if len(contributors) == 2 else "FRAGMENTED"
    base_conf = min(1.0, conf_sum / max(1, len(contributors)))

    return {
        "flow_state": flow_state,
        "confidence": round(base_conf, 2),
        "contributors": contributors,
    }

