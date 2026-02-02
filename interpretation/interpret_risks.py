from app.interpretation.flow_analysis import analyze_linear_flow


def interpret_risks(risks, kinematics=None):
    """
    KEYED interpretation output (no English strings).
    All user/coach language is provided by the YAML clinician layer.

    Returns:
      - coach: structured facts + keys
      - kid: structured keys
      - linear_flow: flow_state/confidence/contributors (keys only)
    """
    if not risks:
        return {
            "coach": {
                "risk_level": "none",
                "primary_risk_region": "none",
                "flow_state": "smooth",
                "confidence": 0.0,
            },
            "kid": {
                "summary_key": "smooth",
            },
            "linear_flow": {
                "flow_state": "SMOOTH",
                "confidence": 0.0,
                "contributors": [],
            },
        }

    # Aggregate
    max_strength = 0.0
    dominant_region = "none"

    region_map = {
        "front_foot_braking_shock": "lower_body",
        "knee_brace_failure": "knee",
        "trunk_rotation_snap": "lower_back",
        "hip_shoulder_mismatch": "spine",
        "lateral_trunk_lean": "side_body",
        "foot_line_deviation": "side_body",  # ✅ NEW RISK MAPPED
    }

    region_loads = {}

    for r in risks:
        strength = float(r.get("signal_strength", 0.0))
        max_strength = max(max_strength, strength)

        region = region_map.get(r.get("risk_id"), "other")
        region_loads[region] = max(region_loads.get(region, 0.0), strength)

    if region_loads:
        dominant_region = max(region_loads, key=region_loads.get)

    if max_strength >= 0.7:
        risk_level = "high"
    elif max_strength >= 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"

    flow = analyze_linear_flow(risks=risks, kinematics=kinematics or {})
    flow_state = (flow.get("flow_state") or "SMOOTH").lower()

    confidence = round(min(1.0, max_strength + 0.2), 2)

    return {
        "coach": {
            "risk_level": risk_level,
            "primary_risk_region": dominant_region,
            "flow_state": flow_state,
            "confidence": confidence,
        },
        "kid": {
            "summary_key": flow_state,
        },
        "linear_flow": flow,
    }
