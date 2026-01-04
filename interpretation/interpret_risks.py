def interpret_risks(risks):
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
    # Coach interpretation
    # -----------------------------
    coach_summary = (
        "High impact and rotation loads are being redirected into the trunk, "
        "with compensatory movement patterns indicating elevated stress."
        if risk_level == "high"
        else
        "Moderate load patterns suggest areas that could benefit from improved control and balance."
        if risk_level == "moderate"
        else
        "Overall load patterns appear well-managed with no dominant risk concentration."
    )

    coach_interpretation = {
        "summary": coach_summary,
        "load_generation": (
            "Front-foot braking and rotational acceleration are the primary load drivers."
            if risk_level in ("moderate", "high")
            else
            "No excessive load generation detected."
        ),
        "load_absorption": (
            "Lower-body absorption could not be fully assessed due to visual occlusion."
            if any(r.get("confidence", 1.0) == 0.0 for r in risks)
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
        "confidence": round(min(1.0, max_strength + 0.2), 2),
    }

    # -----------------------------
    # Kid interpretation
    # -----------------------------
    kid_summary = (
        "Your body is working very hard when your front foot hits the ground."
        if risk_level == "high"
        else
        "Your body is working quite hard when you bowl."
        if risk_level == "moderate"
        else
        "Your body is moving nicely when you bowl."
    )

    kid_what_is_happening = (
        "When you land, your body twists fast and bends to the side to help you throw the ball."
        if risk_level in ("moderate", "high")
        else
        "Your body stays mostly balanced as you throw the ball."
    )

    kid_what_to_remember = (
        "Try to stay tall and balanced when your front foot lands."
        if risk_level in ("moderate", "high")
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
        kid_interpretation[
            "coach_tip"
        ] = "It’s a good idea to show this to your coach so they can help you bowl safely."

    # -----------------------------
    # Final combined interpretation
    # -----------------------------
    return {
        "coach": coach_interpretation,
        "kid": kid_interpretation,
    }

