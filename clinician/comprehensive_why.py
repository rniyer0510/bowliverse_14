from typing import Any, Dict, List, Optional


def generate_comprehensive_why(
    risks: List[Dict[str, Any]],
    action: Optional[Dict[str, Any]] = None,
    chain: Optional[Dict[str, Any]] = None,
    pillars: Optional[Dict[str, Dict[str, Any]]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    action_name = ((action or {}).get("action") or "UNKNOWN").upper()

    high_confidence_risks = [r for r in risks if (r.get("confidence") or "").upper() == "HIGH"]
    detected_risks = [
        r for r in risks if (r.get("severity") or "").upper() in {"MODERATE", "HIGH", "VERY_HIGH"}
    ]
    risk_ids = [r.get("risk_id") for r in high_confidence_risks]

    if not risk_ids:
        if detected_risks:
            if len(detected_risks) >= 2:
                return _explain_multiple_detected_risks(detected_risks)
            return _explain_single_detected_risk(detected_risks[0])
        if action_name == "MIXED":
            return _explain_mixed_action_monitor()
        if action_name == "UNKNOWN":
            return _explain_uncertain_action_profile()
        return _build_response(
            primary_cause="No strong concern stands out in this one ball",
            plain_explanation=(
                "We did not see a strong high-confidence movement concern in this clip.\n\n"
                "That is good, but one ball is never the full story."
            ),
            what_to_focus_on=(
                "Use repeated clips before changing anything major."
            ),
            what_we_saw="This clip stayed fairly balanced and did not show a strong high-confidence concern.",
            why_flagged="We did not flag a strong high-confidence movement problem in this one ball.",
            why_it_matters="That is a good sign, but a clean clip should still be checked against repeated deliveries.",
            what_to_check_with_coach="Check if the same shape repeats across a few balls before changing anything.",
        )

    rotation_risks = [
        rid
        for rid in risk_ids
        if rid in {"front_foot_braking_shock", "hip_shoulder_mismatch", "trunk_rotation_snap"}
    ]

    if rotation_risks and "knee_brace_failure" in risk_ids:
        return _explain_mixed_issues(high_confidence_risks)
    if len(rotation_risks) >= 2:
        return _explain_rotation_deficiency(high_confidence_risks, chain=chain)
    if "knee_brace_failure" in risk_ids:
        return _explain_front_leg_weakness(
            high_confidence_risks,
            has_lean="lateral_trunk_lean" in risk_ids,
        )

    alignment_risks = [
        rid for rid in risk_ids if rid in {"lateral_trunk_lean", "foot_line_deviation"}
    ]
    if set(risk_ids) == set(alignment_risks):
        return _explain_alignment_issues(high_confidence_risks)

    if risk_ids == ["hip_shoulder_mismatch"]:
        return _explain_hip_shoulder_only(high_confidence_risks[0])

    return _explain_general_pattern(high_confidence_risks, chain=chain, pillars=pillars, summary=summary)


def _build_response(
    *,
    primary_cause: str,
    plain_explanation: str,
    what_to_focus_on: str,
    what_we_saw: str,
    why_flagged: str,
    why_it_matters: str,
    what_to_check_with_coach: str,
) -> Dict[str, str]:
    return {
        "primary_cause": primary_cause,
        "plain_explanation": plain_explanation,
        "what_to_focus_on": what_to_focus_on,
        "what_we_saw": what_we_saw,
        "why_flagged": why_flagged,
        "why_it_matters": why_it_matters,
        "what_to_check_with_coach": what_to_check_with_coach,
    }


def _risk_line(risk: Optional[Dict[str, Any]], key: str, fallback: str) -> str:
    if not risk:
        return fallback
    value = (risk.get(key) or "").strip()
    return value or fallback


def _describe_risks(risks: List[Dict[str, Any]]) -> str:
    names = [(risk.get("title") or "this pattern").strip() for risk in risks if risk.get("title")]
    if not names:
        return "more than one pattern"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]}, {names[1]}, and {names[2]}"


def _pick_main_risk(risks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not risks:
        return None

    def _score(risk: Dict[str, Any]) -> tuple:
        severity_rank = {
            "LOW": 0,
            "MODERATE": 1,
            "MEDIUM": 1,
            "HIGH": 2,
            "VERY_HIGH": 3,
        }.get((risk.get("severity") or "LOW").upper(), 0)
        confidence_rank = {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2,
        }.get((risk.get("confidence") or "LOW").upper(), 0)
        return severity_rank, confidence_rank

    return sorted(risks, key=_score, reverse=True)[0]


def _explain_rotation_deficiency(
    risks: List[Dict[str, Any]],
    chain: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    return _build_response(
        primary_cause="The action is losing some flow before the ball comes out",
        plain_explanation=(
            "This looks more like a whole-action flow problem than one small fault.\n\n"
            "The bowler builds into the crease, but the action does not keep carrying through the ball as cleanly as we would like."
        ),
        what_to_focus_on=(
            "Review how the bowler moves from landing into the ball, and check it across a few balls before changing anything major."
        ),
        what_we_saw=(chain or {}).get(
            "what_we_saw",
            "The action looked broken between landing and ball release in this clip.",
        ),
        why_flagged=(
            f"We flagged this because this clip showed {_describe_risks(risks).lower()}."
        ),
        why_it_matters=(
            "When these parts are out of sync, the ball can come out with less help from the whole action."
        ),
        what_to_check_with_coach=(
            "Check if the bowler keeps moving smoothly from landing into the ball across a few balls."
        ),
    )


def _explain_front_leg_weakness(
    risks: List[Dict[str, Any]],
    has_lean: bool,
) -> Dict[str, str]:
    knee_risk = next((risk for risk in risks if risk.get("risk_id") == "knee_brace_failure"), None)
    extra = (
        " The same clip also showed the bowler moving off line to the side."
        if has_lean
        else ""
    )
    return _build_response(
        primary_cause="The front leg is the main thing to watch in this clip",
        plain_explanation=(
            "The front leg becomes the base for the rest of the action when it lands.\n\n"
            "In this clip, that base does not hold as well as we would like."
            f"{extra}"
        ),
        what_to_focus_on=(
            "Check if the front leg stays firm and the bowler stays balanced across a few balls."
        ),
        what_we_saw=_risk_line(
            knee_risk,
            "what_we_saw",
            "The front leg softened at landing in this clip.",
        ),
        why_flagged=_risk_line(
            knee_risk,
            "why_flagged",
            "The front leg did not hold its shape as well as we wanted.",
        ),
        why_it_matters=(
            "That matters because the rest of the action has less support when the front leg softens."
        ),
        what_to_check_with_coach=(
            "Check if the front leg stays firmer and supports the rest of the action across a few balls."
        ),
    )


def _explain_mixed_issues(risks: List[Dict[str, Any]]) -> Dict[str, str]:
    return _build_response(
        primary_cause="Landing support and whole-action flow both need watching here",
        plain_explanation=(
            "This clip suggests two linked things.\n\n"
            "The landing phase is taking more of the load than we want, and the action is not carrying through the ball as cleanly as it could."
        ),
        what_to_focus_on=(
            "Check landing support first, then check if the action still carries into the ball across a few balls."
        ),
        what_we_saw=(
            f"This clip showed {_describe_risks(risks).lower()}."
        ),
        why_flagged=(
            "We flagged this because the action looked heavy at landing and less connected through the ball."
        ),
        why_it_matters=(
            "When those things happen together, the body can take extra load and the action can lose flow."
        ),
        what_to_check_with_coach=(
            "Check if the bowler lands with support and still keeps moving through the ball across a few balls."
        ),
    )


def _explain_alignment_issues(risks: List[Dict[str, Any]]) -> Dict[str, str]:
    main_risk = _pick_main_risk(risks)
    return _build_response(
        primary_cause="Balance is the main thing to watch in this clip",
        plain_explanation=(
            "This clip points more to balance and line than to one dramatic fault.\n\n"
            "The bowler is moving a little off line through the action."
        ),
        what_to_focus_on=(
            "Check if the bowler stays taller and keeps the same line across a few balls."
        ),
        what_we_saw=_risk_line(
            main_risk,
            "what_we_saw",
            "The bowler moved off line in this clip.",
        ),
        why_flagged=(
            f"We flagged this because this clip showed {_describe_risks(risks).lower()}."
        ),
        why_it_matters=(
            "That matters because moving off line can load the side of the body more than we want."
        ),
        what_to_check_with_coach=(
            "Check if the bowler stays taller and lands on a steadier line across a few balls."
        ),
    )


def _explain_single_detected_risk(risk: Dict[str, Any]) -> Dict[str, str]:
    title = risk.get("title") or "this movement pattern"
    return _build_response(
        primary_cause=f"{title} is the main thing to watch in this clip",
        plain_explanation=(
            f"This delivery showed one clear pattern around {title.lower()}.\n\n"
            "That does not automatically make it a major problem, but it is enough to keep watching."
        ),
        what_to_focus_on=(
            "Check whether the same pattern shows up across repeated deliveries before changing anything major."
        ),
        what_we_saw=_risk_line(
            risk,
            "what_we_saw",
            f"This clip showed {title.lower()}.",
        ),
        why_flagged=_risk_line(
            risk,
            "why_flagged",
            "The model saw enough of this pattern to keep it on the watchlist.",
        ),
        why_it_matters=_risk_line(
            risk,
            "why_it_matters",
            "That matters if it keeps repeating across a few balls.",
        ),
        what_to_check_with_coach=_risk_line(
            risk,
            "what_to_check_with_coach",
            "Check if the same pattern shows up across repeated deliveries.",
        ),
    )


def _explain_multiple_detected_risks(risks: List[Dict[str, Any]]) -> Dict[str, str]:
    return _build_response(
        primary_cause="More than one movement pattern needs watching in this clip",
        plain_explanation=(
            "This clip showed more than one meaningful pattern.\n\n"
            "That does not mean the action needs a rebuild, but it does mean we should not call it fully clear from one ball."
        ),
        what_to_focus_on=(
            "Look for the same patterns across a set of deliveries before deciding what really needs work."
        ),
        what_we_saw=f"This clip showed {_describe_risks(risks).lower()}.",
        why_flagged="We flagged this because more than one meaningful pattern showed up in the same clip.",
        why_it_matters="When more than one pattern shows up together, it is safer to judge the action across repeated deliveries.",
        what_to_check_with_coach="Check which pattern repeats most often across a few balls before changing anything.",
    )


def _explain_mixed_action_monitor() -> Dict[str, str]:
    return _build_response(
        primary_cause="The action looks mixed, so balance is the main thing to watch",
        plain_explanation=(
            "This clip does not show a strong confirmed high-load warning, but the action still looks mixed.\n\n"
            "That makes balance and repeatability more important to monitor."
        ),
        what_to_focus_on=(
            "Track this across repeated deliveries and use a clear bowling-arm-side view."
        ),
        what_we_saw="The action looked mixed in this clip.",
        why_flagged="We flagged this because mixed actions need careful monitoring even when one ball does not show a strong red flag.",
        why_it_matters="That matters because mixed actions can put the body in different positions from ball to ball.",
        what_to_check_with_coach="Check if the same action shape repeats across a few balls before changing anything.",
    )


def _explain_uncertain_action_profile() -> Dict[str, str]:
    return _build_response(
        primary_cause="This clip was too unclear for a trusted movement read",
        plain_explanation=(
            "This clip did not give enough clean information for a trustworthy reading.\n\n"
            "That does not mean the action is unsafe. It means this clip should be treated as unclear."
        ),
        what_to_focus_on=(
            "Record another clip with the full body visible before drawing conclusions."
        ),
        what_we_saw="The full action was not clear enough in this clip.",
        why_flagged="We could not see enough of the movement cleanly enough to trust the reading.",
        why_it_matters="That matters because an unclear clip should not be used to reassure or worry a bowler.",
        what_to_check_with_coach="Record another clip with the full body visible and compare repeated deliveries.",
    )


def _explain_hip_shoulder_only(risk: Dict[str, Any]) -> Dict[str, str]:
    return _build_response(
        primary_cause="Hips and shoulders are the main watchpoint in this clip",
        plain_explanation=(
            "In this clip, the hips and shoulders do not look fully connected.\n\n"
            "That can change how the bowler builds and releases force, but it does not automatically mean the action is wrong."
        ),
        what_to_focus_on=(
            "Treat this as a watchpoint and check if it repeats across a few balls."
        ),
        what_we_saw=_risk_line(
            risk,
            "what_we_saw",
            "The hips and shoulders were out of sync in this clip.",
        ),
        why_flagged=_risk_line(
            risk,
            "why_flagged",
            "The hips and shoulders did not work together as cleanly as we wanted.",
        ),
        why_it_matters=_risk_line(
            risk,
            "why_it_matters",
            "That can change how the action carries force through the middle.",
        ),
        what_to_check_with_coach=_risk_line(
            risk,
            "what_to_check_with_coach",
            "Check if the hips and shoulders work together better across a few balls.",
        ),
    )


def _explain_general_pattern(
    risks: List[Dict[str, Any]],
    chain: Optional[Dict[str, Any]] = None,
    pillars: Optional[Dict[str, Dict[str, Any]]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    primary_pillar = ((summary or {}).get("primary_pillar") or "POSTURE").upper()
    pillar_map = (pillars or {}).get(primary_pillar.lower(), {})
    return _build_response(
        primary_cause="More than one part of the action needs watching here",
        plain_explanation=(
            "This clip looks more like a whole-action watchpoint than one isolated fault.\n\n"
            "Some parts of the movement are not working together as cleanly as we would like."
        ),
        what_to_focus_on=(
            "Check the same pattern across repeated deliveries before changing anything major."
        ),
        what_we_saw=(
            pillar_map.get("what_we_saw")
            or (chain or {}).get("what_we_saw")
            or f"This clip showed {_describe_risks(risks).lower()}."
        ),
        why_flagged=(
            pillar_map.get("why_flagged")
            or (chain or {}).get("why_flagged")
            or "We flagged this because more than one part of the action looked out of sync."
        ),
        why_it_matters=(
            pillar_map.get("why_it_matters")
            or (chain or {}).get("why_it_matters")
            or "That matters because the action can lose flow or take extra load when parts stop working together."
        ),
        what_to_check_with_coach=(
            pillar_map.get("what_to_check_with_coach")
            or (chain or {}).get("what_to_check_with_coach")
            or "Check which part of the action repeats as the main problem across a few balls."
        ),
    )
