"""
app/clinician/comprehensive_why.py

Generates comprehensive "WHY" explanation based on detected risks.
Integrates with ClinicianInterpreter.build() method.

Plain-language biomechanical explanations.
No muscle diagnoses. No gym prescriptions.
"""

from typing import List, Dict, Any


def generate_comprehensive_why(risks: List[Dict[str, Any]]) -> Dict[str, str]:
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

    # Filter to HIGH confidence risks
    high_confidence_risks = [
        r for r in risks
        if r.get("confidence") == "HIGH"
    ]

    risk_ids = [r.get("risk_id") for r in high_confidence_risks]

    # --------------------------------------------------
    # NO RISKS
    # --------------------------------------------------
    if not risk_ids:
        return {
            "primary_cause": "Your action is biomechanically sound",
            "plain_explanation": (
                "Your bowling action shows good mechanics with no significant concerns.\n\n"
                "Every bowler has their own style - what matters is that your body moves "
                "efficiently and safely. Your momentum flows smoothly through the delivery "
                "without creating unnecessary stress.\n\n"
                "Keep bowling naturally and stay consistent with your rhythm."
            ),
            "what_to_focus_on": (
                "Maintain your current technique and stay consistent with your preparation."
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
            "Here's what's happening in your action:\n\n"

            "When you run in to bowl, you build forward momentum. At release, "
            "that momentum should transition smoothly into rotation through your "
            "hips, trunk, and shoulders.\n\n"

            "In your action, some of that forward energy is slowing down rather "
            "than flowing through into rotation.\n\n"

            "Think of it like this: Imagine running and then stopping suddenly "
            "versus running and spinning in a circle. Stopping directs force into "
            "one area. Spinning distributes it through the whole body.\n\n"

            "When rotation doesn't complete smoothly:\n"
            "• More load goes into the front leg\n"
            "• Hips and shoulders may move together instead of sequentially\n"
            "• The trunk may compensate to generate speed\n\n"

            "These patterns are closely connected to how rotation is finishing "
            "in your delivery."
        ),

        "what_to_focus_on": (
            "Focus on completing your follow-through.\n\n"

            "After releasing the ball, allow your body to continue rotating "
            "naturally instead of stopping early.\n\n"

            "Think about letting momentum flow rather than forcing positions."
        )
    }


def _explain_front_leg_weakness(has_lean: bool) -> Dict[str, str]:
    lean_text = (
        "\n\n"
        "Your body may shift slightly to the side as a natural way of "
        "redistributing force during landing. This is a compensation pattern, "
        "not a conscious mistake."
    ) if has_lean else ""

    lean_focus = (
        " As landing control improves, that sideways shift usually reduces naturally."
    ) if has_lean else ""

    return {
        "primary_cause": "Front leg not stabilizing landing smoothly",

        "plain_explanation": (
            "Here's what's happening in your action:\n\n"

            "When your front foot lands, that leg acts as the base around which "
            "your body rotates. Ideally, it absorbs force and supports smooth rotation.\n\n"

            "In your action, the landing phase is holding more force than ideal. "
            "Instead of transferring energy upward into rotation, some of that "
            "force remains in the leg."

            f"{lean_text}\n\n"

            "Think of it like rotating around a firm pole versus one that bends "
            "slightly. The more stable the base, the cleaner the movement."
        ),

        "what_to_focus_on": (
            "Focus on improving landing balance and control.\n\n"

            "Work on smooth front-foot contact and controlled rotation rather "
            "than trying to force the leg to stay rigid."

            f"{lean_focus}"
        )
    }


def _explain_mixed_issues() -> Dict[str, str]:
    return {
        "primary_cause": "Rotation timing and landing control both affecting force flow",

        "plain_explanation": (
            "Here's what's happening in your action:\n\n"

            "Two related patterns are influencing your delivery:\n\n"

            "1. Rotation is not fully completing, so forward momentum slows "
            "instead of flowing through.\n\n"

            "2. Your landing phase is absorbing more force than ideal, affecting "
            "how efficiently rotation happens.\n\n"

            "When both occur together, more stress remains in the lower body "
            "instead of being distributed through the full kinetic chain.\n\n"

            "These patterns are adjustable with awareness and focused work."
        ),

        "what_to_focus_on": (
            "Two focus areas:\n\n"

            "1. Allow rotation to continue naturally after release.\n\n"
            "2. Improve landing balance and stability during front-foot contact.\n\n"

            "Prioritize rhythm and smooth transitions over forcing positions."
        )
    }


def _explain_alignment_issues() -> Dict[str, str]:
    return {
        "primary_cause": "Delivery alignment and consistency",

        "plain_explanation": (
            "Here's what's happening in your action:\n\n"

            "Your overall mechanics are solid, but small variations in landing "
            "position or trunk alignment are occurring from delivery to delivery.\n\n"

            "This is not a major biomechanical concern. It reflects consistency "
            "patterns that develop with repetition.\n\n"

            "Like a basketball free throw, repeating the same motion builds "
            "automatic alignment over time."
        ),

        "what_to_focus_on": (
            "Focus on repetition and rhythm.\n\n"

            "Using consistent video feedback can help increase awareness of "
            "small variations in landing and trunk position."
        )
    }


def _explain_hip_shoulder_only() -> Dict[str, str]:
    return {
        "primary_cause": "Hips and shoulders rotating together",

        "plain_explanation": (
            "Here's what's happening in your action:\n\n"

            "In many bowling actions, hips rotate first and shoulders follow, "
            "creating a slight separation that builds elastic energy.\n\n"

            "In your delivery, hips and shoulders are rotating more together. "
            "This reduces the 'coiling' effect but does not automatically make "
            "the action unsafe.\n\n"

            "Many successful bowlers use this style effectively. It reflects "
            "a movement preference rather than a flaw."
        ),

        "what_to_focus_on": (
            "This pattern is optional to change.\n\n"

            "If you want more elastic separation, experiment with allowing "
            "hips to initiate rotation slightly before the shoulders.\n\n"

            "If your action feels natural and effective, you do not need "
            "to force a change."
        )
    }


def _explain_general_pattern() -> Dict[str, str]:
    return {
        "primary_cause": "Delivery timing and force sequencing",

        "plain_explanation": (
            "Bowling is a chain of movements that must flow smoothly:\n"
            "run-up → back foot contact → hip rotation → front foot contact → "
            "release → follow-through.\n\n"

            "In your action, some elements of this sequence are slightly out of "
            "sync. When timing shifts, force may concentrate in certain areas "
            "instead of flowing evenly.\n\n"

            "These are timing patterns rather than fundamental flaws."
        ),

        "what_to_focus_on": (
            "Work on smooth transitions between each phase of your delivery.\n\n"

            "Video feedback can help you identify where rhythm or timing "
            "can be improved."
        )
    }

