from app.interpretation.flow_analysis import analyze_linear_flow


def interpret_risks(risks, kinematics=None):
    """
    Produce two interpretations from the full risk set:
    1) Coach-facing (technical but concise)
    2) Kid-facing (8-year-old friendly, supportive)

    Rules:
    - Interpret risks holistically, not individually
    - Add coach guidance to kid interpretation only for moderate/high risk
    - Never mention injury, faults, or stopping play
    """

    # -----------------------------
    # Defensive defaults
    # -----------------------------
    if not risks:
        return {
            "coach": {
                "summary": "No significant biomechanical risk patterns detected.",
                "primary_risk_region": "none",
                "confidence": 0.0,
            },
            "kid": {
                "summary": "You are moving well when you bowl.",
                "what_to_remember": "Keep practicing and enjoying your bowling.",
            },
        }

    # -----------------------------
    # Aggregate risk signals
    # -----------------------------
    max_strength = 0.0
    dominant_region = "none"

    region_map = {
        "front_foot_braking_shock": "lower_body",
        "knee_brace_failure": "knee",
        "trunk_rotation_snap": "trunk",
        "hip_shoulder_mismatch": "trunk",
        "lateral_trunk_lean": "lower_back",
    }

    region_loads = {}

    for r in risks:
        strength = float(r.get("signal_strength", 0.0))
        max_strength = max(max_strength, strength)

        region = region_map.get(r.get("risk_id"), "other")
        region_loads[region] = max(region_loads.get(region, 0.0), strength)

    if region_loads:
        dominant_region = max(region_loads, key=region_loads.get)

    # -----------------------------
    # Risk level classification
    # -----------------------------
    if max_strength >= 0.7:
        risk_level = "high"
    elif max_strength >= 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # -----------------------------
    # Linear flow (interpretive only)
    # -----------------------------
    flow_analysis = analyze_linear_flow(risks=risks, kinematics=kinematics or {})
    flow_state = (flow_analysis.get("flow_state") or "SMOOTH").upper()

    # -----------------------------
    # Coach interpretation
    # -----------------------------
    coach_summary = (
        "Multiple load indicators suggest disrupted forward flow, with energy being "
        "redirected into trunk motion rather than released smoothly."
        if flow_state == "FRAGMENTED"
        else
        "Some interruption of forward flow is present, indicating opportunities for improved sequencing."
        if flow_state == "INTERRUPTED"
        else
        "Overall movement shows good forward continuity with well-managed load transfer."
    )

    coach_interpretation = {
        "summary": coach_summary,
        "load_generation": (
            "Front-foot braking and rotational acceleration contribute most to load generation."
            if risk_level in ("moderate", "high")
            else
            "No excessive load generation detected."
        ),
        "load_absorption": (
            "Lower-body absorption could not be fully assessed due to visual occlusion."
            if any(float(r.get("confidence", 1.0)) == 0.0 for r in risks)
            else
            "Load absorption appears generally controlled."
        ),
        "compensation": (
            "Lateral and trunk compensations are present, suggesting off-axis load handling."
            if dominant_region in ("trunk", "lower_back")
            else
            "No significant compensatory patterns observed."
        ),
        "primary_risk_region": dominant_region,
        "flow_state": flow_state.lower(),
        "confidence": round(min(1.0, max_strength + 0.2), 2),
    }

    # -----------------------------
    # Kid interpretation
    # -----------------------------
    kid_summary = (
        "Your body is working very hard all at once when you bowl."
        if flow_state == "FRAGMENTED"
        else
        "Your body works a bit harder at some moments when you bowl."
        if flow_state == "INTERRUPTED"
        else
        "Your body moves smoothly when you bowl."
    )

    kid_what_is_happening = (
        "Your body slows down suddenly and then twists or bends to help throw the ball."
        if flow_state in ("FRAGMENTED", "INTERRUPTED")
        else
        "Your body keeps moving forward smoothly as you throw the ball."
    )

    kid_what_to_remember = (
        "Try to stay tall and keep moving forward when your front foot lands."
        if flow_state in ("FRAGMENTED", "INTERRUPTED")
        else
        "Keep staying tall and relaxed when you bowl."
    )

    kid_interpretation = {
        "summary": kid_summary,
        "what_is_happening": kid_what_is_happening,
        "what_to_remember": kid_what_to_remember,
    }

    # Add coach guidance ONLY for moderate/high risk
    if risk_level in ("moderate", "high"):
        kid_interpretation["coach_tip"] = (
            "It’s a good idea to show this to your coach so they can help you bowl comfortably."
        )

    # -----------------------------
    # Final combined interpretation
    # -----------------------------
    return {
        "coach": coach_interpretation,
        "kid": kid_interpretation,
        "linear_flow": flow_analysis,
    }
