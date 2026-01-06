from typing import Any, Dict, List


def analyze_linear_flow(
    risks: List[Dict[str, Any]],
    kinematics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Linear Action Flow (interpretation-only)

    RULE (V14 LOCKED):
    - 0 or 1 moderate contributor  -> SMOOTH
    - 2 moderate contributors      -> INTERRUPTED
    - ≥3 moderate contributors     -> FRAGMENTED
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
            "coach": {
                "summary": "Forward energy is transferred smoothly through the action.",
                "detail": "No meaningful interruption of kinetic flow detected.",
                "cues": [],
            },
            "user": {
                "summary": "Your bowling action flows smoothly.",
                "detail": "Your body keeps moving forward naturally while you bowl.",
                "try_this": [],
            },
        }

    if len(contributors) == 2:
        flow_state = "INTERRUPTED"
    else:
        flow_state = "FRAGMENTED"

    base_conf = min(1.0, conf_sum / len(contributors))

    coach_detail = (
        "Some interruption of forward flow is present. Energy transfer is not fully continuous "
        "through front-foot contact into release."
        if flow_state == "INTERRUPTED"
        else
        "Multiple risk signals indicate fragmented forward flow, with energy redirected into "
        "rotation or lateral movement instead of releasing smoothly."
    )

    user_detail = (
        "Your movement slows slightly before releasing the ball."
        if flow_state == "INTERRUPTED"
        else
        "Your body slows down suddenly when you bowl, which makes other parts work harder."
    )

    return {
        "flow_state": flow_state,
        "confidence": round(base_conf, 2),
        "contributors": contributors,
        "coach": {
            "summary": f"Linear action flow is {flow_state.lower()}.",
            "detail": coach_detail,
            "cues": [
                "Allow your chest to keep travelling forward",
                "Avoid stopping or collapsing at front-foot contact",
            ],
        },
        "user": {
            "summary": "Your bowling action loses smooth flow.",
            "detail": user_detail,
            "try_this": [
                "Stay tall and keep moving forward as you bowl",
            ],
        },
    }
