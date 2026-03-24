"""
app/clinician/comprehensive_why.py

Generates comprehensive "WHY" explanation based on detected risks.
Integrates with ClinicianInterpreter.build() method.

Plain-language biomechanical explanations.
No muscle diagnoses. No gym prescriptions.
"""

from typing import List, Dict, Any, Optional


def generate_comprehensive_why(
    risks: List[Dict[str, Any]],
    action: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate comprehensive WHY explanation based on detected risks.

    Args:
        risks: List of risk dicts from ClinicianInterpreter.build_risks()
               Each risk has: risk_id, severity, confidence, etc.

    Returns:
        Dict with:
            - primary_cause
            - plain_explanation
            - what_to_focus_on
    """

    action_name = ((action or {}).get("action") or "UNKNOWN").upper()

    # Filter to HIGH confidence risks
    high_confidence_risks = [
        r for r in risks
        if r.get("confidence") == "HIGH"
    ]

    detected_risks = [
        r for r in risks
        if (r.get("severity") or "").upper() in {"MODERATE", "HIGH", "VERY_HIGH"}
    ]

    risk_ids = [r.get("risk_id") for r in high_confidence_risks]

    # --------------------------------------------------
    # NO HIGH-CONFIDENCE RISKS
    # --------------------------------------------------
    if not risk_ids:
        if detected_risks:
            if len(detected_risks) >= 2:
                return _explain_multiple_detected_risks()
            return _explain_single_detected_risk(detected_risks[0])
        if action_name == "MIXED":
            return _explain_mixed_action_monitor()
        if action_name == "UNKNOWN":
            return _explain_uncertain_action_profile()
        return {
            "primary_cause": "No strong concern seen in this delivery",
            "plain_explanation": (
                "This delivery does not show a strong confirmed biomechanical concern.\n\n"
                "Every bowler has their own style. What matters most is whether the body "
                "is moving in a way that looks controlled and repeatable.\n\n"
                "On this delivery, your action looks generally well organized."
            ),
            "what_to_focus_on": (
                "Keep building consistency and stay with the rhythm that feels natural to you."
            )
        }

    # --------------------------------------------------
    # ROTATION CLUSTER
    # --------------------------------------------------
    rotation_risks = [
        rid for rid in risk_ids
        if rid in [
            "front_foot_braking_shock",
            "hip_shoulder_mismatch",
            "trunk_rotation_snap"
        ]
    ]

    # Mixed case first (rotation + landing)
    if rotation_risks and "knee_brace_failure" in risk_ids:
        return _explain_mixed_issues()

    if len(rotation_risks) >= 2:
        return _explain_rotation_deficiency(rotation_risks)

    # --------------------------------------------------
    # FRONT LEG / LANDING CONTROL
    # --------------------------------------------------
    if "knee_brace_failure" in risk_ids:
        has_lean = "lateral_trunk_lean" in risk_ids
        return _explain_front_leg_weakness(has_lean)

    # --------------------------------------------------
    # ALIGNMENT ONLY
    # --------------------------------------------------
    alignment_risks = [
        rid for rid in risk_ids
        if rid in ["lateral_trunk_lean", "foot_line_deviation"]
    ]

    if set(risk_ids) == set(alignment_risks):
        return _explain_alignment_issues()

    # --------------------------------------------------
    # HIP-SHOULDER ONLY
    # --------------------------------------------------
    if risk_ids == ["hip_shoulder_mismatch"]:
        return _explain_hip_shoulder_only()

    # --------------------------------------------------
    # DEFAULT
    # --------------------------------------------------
    return _explain_general_pattern()


# ============================================================
# EXPLANATION BLOCKS
# ============================================================

def _explain_rotation_deficiency(rotation_risks: List[str]) -> Dict[str, str]:
    return {
        "primary_cause": "Rotation not fully completing through delivery",

        "plain_explanation": (
            "You build speed as you run in. That speed should keep flowing through "
            "your hips, trunk, and shoulders during delivery.\n\n"
            "In this delivery, some of that speed looks like it is slowing down too "
            "early instead of flowing through the full chain.\n\n"
            "When that happens, the body often tries to finish the action higher up "
            "the chain. That can show up as more load on the front leg, the hips and "
            "shoulders moving too much together, or the trunk working harder than it "
            "should."
        ),

        "what_to_focus_on": (
            "Focus on letting the action keep flowing after release.\n\n"
            "Try to let the body rotate through naturally instead of stopping early "
            "or forcing positions."
        )
    }


def _explain_front_leg_weakness(has_lean: bool) -> Dict[str, str]:
    lean_text = (
        "\n\n"
        "Your body may also lean to the side because it is not staying balanced "
        "over the front leg at landing."
    ) if has_lean else ""

    lean_focus = (
        " As front-leg control improves, that sideways lean often reduces as well."
    ) if has_lean else ""

    return {
        "primary_cause": "Front leg not stabilizing landing smoothly",

        "plain_explanation": (
            "When your front foot lands, that leg becomes the base for the rest of "
            "the action.\n\n"
            "In this delivery, the front side does not look as stable as it could be. "
            "Instead of passing force cleanly up the chain, more of it seems to stay "
            "in the landing phase."
            f"{lean_text}\n\n"
            "That can make the rest of the action work harder to finish the delivery."
        ),

        "what_to_focus_on": (
            "Focus on landing balance and control.\n\n"
            "Think about a smooth front-foot landing and letting the body rotate from "
            "a strong base, rather than forcing the leg to be stiff."

            f"{lean_focus}"
        )
    }


def _explain_mixed_issues() -> Dict[str, str]:
    return {
        "primary_cause": "Rotation timing and landing control both affecting force flow",

        "plain_explanation": (
            "Two linked things may be happening in this delivery.\n\n"
            "First, force is not flowing through rotation as cleanly as it could.\n\n"
            "Second, the landing phase is taking more load than ideal.\n\n"
            "When both happen together, the body can lose flow at the base and then "
            "try to recover it later in the chain."
        ),

        "what_to_focus_on": (
            "Focus on two things:\n\n"
            "1. Better balance and control at landing.\n\n"
            "2. Letting rotation continue smoothly through release.\n\n"
            "Prioritize rhythm and smooth flow over forcing positions."
        )
    }


def _explain_alignment_issues() -> Dict[str, str]:
    return {
        "primary_cause": "Delivery alignment and consistency",

        "plain_explanation": (
            "Your overall action looks functional, but there are small changes in "
            "how you land or line up from delivery to delivery.\n\n"
            "This does not automatically mean there is a major problem. It usually "
            "means the action is not repeating in exactly the same way each time."
        ),

        "what_to_focus_on": (
            "Focus on repetition, rhythm, and a repeatable setup into landing.\n\n"
            "Video feedback can help you notice small changes in alignment over time."
        )
    }


def _explain_single_detected_risk(risk: Dict[str, Any]) -> Dict[str, str]:
    title = risk.get("title") or "a movement pattern"
    return {
        "primary_cause": f"{title} needs monitoring",
        "plain_explanation": (
            f"This delivery showed a risk pattern related to {title.lower()}.\n\n"
            "That does not automatically mean there is a major problem, but it does "
            "mean this delivery should be monitored rather than cleared."
        ),
        "what_to_focus_on": (
            "Keep an eye on this pattern over repeated deliveries and use video "
            "feedback to see if it keeps showing up."
        ),
    }


def _explain_multiple_detected_risks() -> Dict[str, str]:
    return {
        "primary_cause": "More than one risk pattern needs monitoring",
        "plain_explanation": (
            "This delivery showed more than one risk pattern.\n\n"
            "That does not mean there is a crisis, but it does mean this delivery "
            "should not be cleared too quickly from one clip."
        ),
        "what_to_focus_on": (
            "Monitor these patterns across repeated deliveries. If they keep showing "
            "up, work with a coach to review your action in more detail."
        ),
    }


def _explain_mixed_action_monitor() -> Dict[str, str]:
    return {
        "primary_cause": "Mixed action profile worth monitoring",
        "plain_explanation": (
            "This delivery does not show a strong confirmed high-risk pattern, "
            "but your action profile is classified as mixed.\n\n"
            "A mixed action can still work well, but it is worth watching more "
            "carefully because alignment and sequencing can vary from one ball to "
            "the next.\n\n"
            "In simple terms, this is not a strong concern from one delivery, but "
            "it is not something we should fully clear from one delivery either."
        ),
        "what_to_focus_on": (
            "Monitor this action across repeated deliveries, especially trunk "
            "alignment and hip-shoulder sequencing.\n\n"
            "Use a clear bowling-arm-side video angle so the action profile and "
            "risk visuals can be judged more confidently."
        ),
    }


def _explain_uncertain_action_profile() -> Dict[str, str]:
    return {
        "primary_cause": "Action profile could not be confirmed clearly",
        "plain_explanation": (
            "This delivery did not provide enough clear information to give a "
            "fully reassuring action-profile interpretation.\n\n"
            "That does not mean the action is unsafe. It means this clip should be "
            "treated with caution rather than confidently cleared."
        ),
        "what_to_focus_on": (
            "Record another clip with the full body visible from the bowling-arm "
            "side and continue monitoring repeated deliveries rather than "
            "drawing conclusions from one unclear clip."
        ),
    }


def _explain_hip_shoulder_only() -> Dict[str, str]:
    return {
        "primary_cause": "Hips and shoulders rotating together",

        "plain_explanation": (
            "In many bowling actions, the hips start the turn and the shoulders "
            "follow.\n\n"
            "In this delivery, the hips and shoulders look like they are turning "
            "more together.\n\n"
            "That can reduce the build-up of elastic separation, but it does not "
            "automatically make the action unsafe. Many bowlers use this style."
        ),

        "what_to_focus_on": (
            "This is optional to change.\n\n"
            "If you want more separation, try letting the hips start the turn a "
            "little before the shoulders.\n\n"
            "If the action feels natural and effective, you do not need to force it."
        )
    }


def _explain_general_pattern() -> Dict[str, str]:
    return {
        "primary_cause": "Delivery timing and force sequencing",

        "plain_explanation": (
            "Bowling works best when force flows smoothly from the run-up, through "
            "landing, into release, and then through the follow-through.\n\n"
            "In this delivery, some parts of that sequence look slightly out of sync. "
            "When that happens, force can build up in one area instead of flowing "
            "cleanly through the whole action.\n\n"
            "This is more about timing and flow than a major flaw."
        ),

        "what_to_focus_on": (
            "Focus on smooth transitions through the action.\n\n"
            "Video feedback can help you spot where rhythm or timing can improve."
        )
    }
