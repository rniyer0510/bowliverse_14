from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.clinician.bands import confidence_band, severity_band
from app.clinician.comprehensive_why import generate_comprehensive_why
from app.clinician.loader import load_yaml

ELBOW = load_yaml("elbow.yaml") or {}
RISKS = load_yaml("risks.yaml") or {}

SEVERITY_POINTS = {
    "LOW": 0.0,
    "MODERATE": 1.0,
    "HIGH": 2.0,
    "VERY_HIGH": 3.0,
}

CONFIDENCE_WEIGHTS = {
    "LOW": 0.4,
    "MEDIUM": 0.7,
    "HIGH": 1.0,
}

PILLAR_SPECS = {
    "POSTURE": {
        "display_label": "Balance",
        "slug": "balance",
        "question": "Is the bowler staying balanced and upright?",
        "clear": "The bowler stays fairly balanced and upright in this clip.",
        "watch": "The bowler moves off line a little in this clip, so balance is worth watching.",
        "review": "The bowler moves off line enough here to make balance a main review point.",
        "what_it_measures": "This shows how well the bowler stays upright, balanced, and in line.",
        "what_100_means": "100 means the action stays upright, stable, and in line, with very little visible drift.",
        "what_lowers_score": "The score drops when the body leans sideways, lands off line, or loses balance.",
    },
    "POWER": {
        "display_label": "Ball Carry",
        "slug": "carry",
        "question": "Is the run-up helping the ball come out well?",
        "clear": "The action stays fairly connected from run-up into ball release in this clip.",
        "watch": "The run-up and ball release do not look fully connected in this clip.",
        "review": "The run-up is not getting into the ball well enough here, so ball carry becomes a main review point.",
        "what_it_measures": "This shows how well the run-up and body movement carry into the ball.",
        "what_100_means": "100 means the action carries smoothly from run-up to landing to release, with very little visible pace leak.",
        "what_lowers_score": "The score drops when the action takes too much of the hit at landing or loses flow before release.",
    },
    "PROTECTION": {
        "display_label": "Body Load",
        "slug": "body_load",
        "question": "Is the body handling the bowling well?",
        "clear": "This clip does not show a strong sign of extra body load.",
        "watch": "Some parts of the action may be taking extra load in this clip.",
        "review": "Some parts of the action look like they are taking more load than we would like, so body load becomes a main review point.",
        "what_it_measures": "This shows how safely the body is handling the bowling action.",
        "what_100_means": "100 means the action shows very few visible warning signs for extra stress on the body.",
        "what_lowers_score": "The score drops when the action shows extra side load, sharp trunk load, front-leg load, or elbow concern.",
    },
}

RISK_PILLAR_WEIGHTS = {
    "front_foot_braking_shock": {"POWER": 2.0, "PROTECTION": 1.0},
    "knee_brace_failure": {"POWER": 1.0, "PROTECTION": 2.0},
    "trunk_rotation_snap": {"POWER": 1.0, "PROTECTION": 2.0},
    "hip_shoulder_mismatch": {"POSTURE": 1.0, "POWER": 2.0},
    "lateral_trunk_lean": {"POSTURE": 2.0, "PROTECTION": 1.0},
    "foot_line_deviation": {"POSTURE": 2.0, "PROTECTION": 1.0},
}

OVERALL_SCORE_WEIGHTS = {
    "POSTURE": 0.30,
    "POWER": 0.35,
    "PROTECTION": 0.35,
}

PILLAR_BASE_SCORE = 100.0

PILLAR_PENALTY_SCALE = {
    "POSTURE": 24.0,
    "POWER": 22.0,
    "PROTECTION": 23.0,
}

PILLAR_MAX_BONUS = {
    "POSTURE": 0.0,
    "POWER": 0.0,
    "PROTECTION": 0.0,
}

FEATURE_MAX_DEDUCTIONS = {
    "front_foot_line": 15,
    "front_leg_support": 15,
    "trunk_lean": 15,
    "upper_body_opening": 15,
    "trunk_rotation_load": 10,
    "action_flow": 15,
    "action_type": 10,
    "elbow_review": 5,
}

GROUP_SPECS = {
    "lower_body_alignment": {
        "label": "Lower Body Alignment",
        "what_it_measures": "This shows how well the lower body lands and supports the action.",
        "what_high_score_means": "A high score means the lower body is landing in a cleaner line and supporting the action well.",
    },
    "upper_body_alignment": {
        "label": "Upper Body Alignment",
        "what_it_measures": "This shows how well the upper body stays tall and under control.",
        "what_high_score_means": "A high score means the upper body is staying more upright and under control through release.",
    },
    "whole_body_alignment": {
        "label": "Whole Body Alignment",
        "what_it_measures": "This shows how well the whole action stays together into release.",
        "what_high_score_means": "A high score means the action is staying connected and well organized into release.",
    },
    "momentum_forward": {
        "label": "Momentum Forward",
        "what_it_measures": "This shows how well the action helps momentum carry into the ball.",
        "what_high_score_means": "A high score means the action is carrying momentum smoothly into release.",
    },
    "safety": {
        "label": "Safety",
        "what_it_measures": "This shows how well the body is handling the bowling action.",
        "what_high_score_means": "A high score means the action is not showing strong signs of extra body stress.",
    },
}

GROUP_FEATURE_WEIGHTS = {
    "lower_body_alignment": {
        "front_foot_line": 1.0,
        "front_leg_support": 1.0,
        "action_type": 0.3,
    },
    "upper_body_alignment": {
        "trunk_lean": 1.0,
        "upper_body_opening": 1.0,
        "trunk_rotation_load": 1.0,
        "action_type": 0.6,
    },
    "whole_body_alignment": {
        "front_foot_line": 0.6,
        "front_leg_support": 0.6,
        "trunk_lean": 0.6,
        "upper_body_opening": 1.0,
        "action_flow": 1.0,
        "action_type": 0.6,
    },
    "momentum_forward": {
        "front_foot_line": 0.3,
        "front_leg_support": 1.0,
        "upper_body_opening": 1.0,
        "trunk_rotation_load": 0.6,
        "action_flow": 1.0,
    },
    "safety": {
        "front_foot_line": 0.6,
        "front_leg_support": 0.6,
        "trunk_lean": 1.0,
        "trunk_rotation_load": 1.0,
        "action_type": 1.0,
        "elbow_review": 1.0,
    },
}


def f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d


def _severity_rank(severity: str) -> int:
    return {
        "LOW": 0,
        "MODERATE": 1,
        "MEDIUM": 1,
        "HIGH": 2,
        "VERY_HIGH": 3,
    }.get((severity or "LOW").upper(), 0)


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _pretty_reason(raw: str) -> str:
    return raw.replace("_", " ").strip().title()


def _raw_signal_value(risk: Dict[str, Any]) -> float:
    evidence = risk.get("evidence") or {}
    if evidence.get("signal_strength") is not None:
        return max(0.0, min(1.0, f(evidence.get("signal_strength"), 0.0)))
    return {
        "LOW": 0.20,
        "MODERATE": 0.45,
        "HIGH": 0.75,
        "VERY_HIGH": 0.95,
    }.get((risk.get("severity") or "LOW").upper(), 0.20)


def _raw_confidence_value(risk: Dict[str, Any]) -> float:
    evidence = risk.get("evidence") or {}
    if evidence.get("confidence") is not None:
        return max(0.0, min(1.0, f(evidence.get("confidence"), 0.4)))
    return {
        "LOW": 0.40,
        "MEDIUM": 0.70,
        "HIGH": 1.00,
    }.get((risk.get("confidence") or "LOW").upper(), 0.40)


def _normalized_signal(signal: float) -> float:
    return max(0.0, min(1.0, (signal - 0.20) / 0.80))


def _average(values: List[float], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def _score_band_label(score: int) -> str:
    if score >= 90:
        return "Strong"
    if score >= 75:
        return "Good base"
    if score >= 60:
        return "Needs work"
    return "Review with coach"


def _feature_score_from_deduction(deduction: int, max_deduction: int) -> int:
    if max_deduction <= 0:
        return 100
    scaled_loss = round((max(0, deduction) / max_deduction) * 70.0)
    return _clamp_score(100 - scaled_loss)


def _reporting_lines_for_risk(risk_id: str, severity: str) -> Dict[str, str]:
    sev = (severity or "LOW").upper()

    if risk_id == "front_foot_braking_shock":
        if sev in {"HIGH", "VERY_HIGH"}:
            return {
                "what_we_saw": "The landing step took a very sharp hit in this clip.",
                "why_flagged": "The body slowed hard at landing instead of moving through the action.",
                "why_it_matters": "That can keep too much of the hit at landing instead of letting it travel up the action.",
                "what_to_check": "Check if the bowler lands and still keeps moving smoothly into the ball.",
            }
        if sev in {"MODERATE", "MEDIUM"}:
            return {
                "what_we_saw": "The landing step took a sharp hit in this clip.",
                "why_flagged": "The body slowed more than we wanted at landing.",
                "why_it_matters": "If this keeps repeating, the action can lose some flow at landing.",
                "what_to_check": "Check if the bowler lands and still keeps moving smoothly into the ball.",
            }
        return {
            "what_we_saw": "The landing step looked under control in this clip.",
            "why_flagged": "The landing phase did not stand out as a main concern here.",
            "why_it_matters": "That is a good sign because the action is not taking a heavy hit at landing from this one ball.",
            "what_to_check": "Keep checking if the landing step stays this calm across a few balls.",
        }

    if risk_id == "knee_brace_failure":
        if sev in {"HIGH", "VERY_HIGH"}:
            return {
                "what_we_saw": "The front leg softened a lot at landing in this clip.",
                "why_flagged": "The front leg did not hold its shape well when it landed.",
                "why_it_matters": "That leaves the rest of the action with less support.",
                "what_to_check": "Check if the front leg stays firm when the foot lands.",
            }
        if sev in {"MODERATE", "MEDIUM"}:
            return {
                "what_we_saw": "The front leg softened a little at landing in this clip.",
                "why_flagged": "The front leg did not hold its shape as well as we wanted.",
                "why_it_matters": "If this keeps repeating, the rest of the action has a less steady base.",
                "what_to_check": "Check if the front leg stays firmer across a few balls.",
            }
        return {
            "what_we_saw": "The front leg held its shape fairly well in this clip.",
            "why_flagged": "The front leg did not stand out as a main concern here.",
            "why_it_matters": "That is a good sign because the action keeps a better base at landing.",
            "what_to_check": "Keep checking if the front leg stays firm like this across a few balls.",
        }

    if risk_id == "trunk_rotation_snap":
        if sev in {"HIGH", "VERY_HIGH"}:
            return {
                "what_we_saw": "The upper body turned very sharply in this clip.",
                "why_flagged": "The trunk rotation happened more abruptly than we wanted.",
                "why_it_matters": "That can load the trunk and the side of the body more than we want.",
                "what_to_check": "Check if the body turn looks calmer and smoother across a few balls.",
            }
        if sev in {"MODERATE", "MEDIUM"}:
            return {
                "what_we_saw": "The upper body turned a bit sharply in this clip.",
                "why_flagged": "The trunk rotation looked sharper than ideal.",
                "why_it_matters": "If this keeps repeating, the trunk can take more load than we want.",
                "what_to_check": "Check if the body turn looks calmer across a few balls.",
            }
        return {
            "what_we_saw": "The body turn looked fairly calm in this clip.",
            "why_flagged": "The trunk rotation did not stand out as a main concern here.",
            "why_it_matters": "That is a good sign because the upper body is not taking a sharp extra twist from this one ball.",
            "what_to_check": "Keep checking if the body turn stays this calm across a few balls.",
        }

    if risk_id == "hip_shoulder_mismatch":
        if sev in {"HIGH", "VERY_HIGH"}:
            return {
                "what_we_saw": "The hips and shoulders were too far apart in this clip.",
                "why_flagged": "The hips and shoulders did not work together cleanly.",
                "why_it_matters": "That can split the action and load the trunk harder.",
                "what_to_check": "Check if the hips and shoulders move together better across a few balls.",
            }
        if sev in {"MODERATE", "MEDIUM"}:
            return {
                "what_we_saw": "The hips and shoulders were a little out of sync in this clip.",
                "why_flagged": "The hips and shoulders did not work together as cleanly as we wanted.",
                "why_it_matters": "If this keeps repeating, the action can lose some flow.",
                "what_to_check": "Check if the hips and shoulders work together better across a few balls.",
            }
        return {
            "what_we_saw": "The hips and shoulders worked together fairly well in this clip.",
            "why_flagged": "The hips and shoulders did not stand out as a main concern here.",
            "why_it_matters": "That is a good sign because the action looks less split through the middle.",
            "what_to_check": "Keep checking if the hips and shoulders stay this connected across a few balls.",
        }

    if risk_id == "lateral_trunk_lean":
        if sev in {"HIGH", "VERY_HIGH"}:
            return {
                "what_we_saw": "The bowler leaned a long way to the side in this clip.",
                "why_flagged": "The body moved well off line near release.",
                "why_it_matters": "That can load the side of the body more than we want.",
                "what_to_check": "Check if the bowler can stay taller through release across a few balls.",
            }
        if sev in {"MODERATE", "MEDIUM"}:
            return {
                "what_we_saw": "The bowler leaned to the side in this clip.",
                "why_flagged": "The body moved off line near release.",
                "why_it_matters": "If this keeps repeating, it can load the side of the body more than we want.",
                "what_to_check": "Check if the bowler stays taller across a few balls.",
            }
        return {
            "what_we_saw": "The bowler stayed fairly upright in this clip.",
            "why_flagged": "The body did not move far off line here.",
            "why_it_matters": "That is a good sign because the side of the body did not take obvious extra load from this one ball.",
            "what_to_check": "Keep checking if the bowler stays this tall across a few balls.",
        }

    if risk_id == "foot_line_deviation":
        if sev in {"HIGH", "VERY_HIGH"}:
            return {
                "what_we_saw": "The front foot landed very wide in this clip.",
                "why_flagged": "The landing line moved further out than we wanted.",
                "why_it_matters": "That can push the action off line and load the side of the body more than we want.",
                "what_to_check": "Check if the front foot lands closer to line across a few balls.",
            }
        if sev in {"MODERATE", "MEDIUM"}:
            return {
                "what_we_saw": "The front foot landed a bit wide in this clip.",
                "why_flagged": "The landing line moved wider than we wanted.",
                "why_it_matters": "If this keeps repeating, balance and posture can get harder to hold.",
                "what_to_check": "Check if the front foot lands closer to line across a few balls.",
            }
        return {
            "what_we_saw": "The front foot landed in a fairly good line in this clip.",
            "why_flagged": "The landing line did not stand out as a main concern here.",
            "why_it_matters": "That is a good sign because the action is less likely to be pushed off line from this one ball.",
            "what_to_check": "Keep checking if the front foot keeps landing near the same line.",
        }

    if risk_id == "elbow_review":
        return {
            "what_we_saw": "The elbow action needs another look in this clip.",
            "why_flagged": "The arm path crossed the normal review limit used by the legality check.",
            "why_it_matters": "That matters because elbow legality should not be guessed from one unclear impression.",
            "what_to_check": "Check this carefully with a coach or specialist before changing anything.",
        }

    return {
        "what_we_saw": "This movement pattern stood out in this clip.",
        "why_flagged": "The model saw enough of this pattern to put it on the watchlist.",
        "why_it_matters": "That matters if it keeps repeating across a few balls.",
        "what_to_check": "Check if the same pattern shows up across repeated deliveries.",
    }


class ClinicianInterpreter:
    def _apply_benchmark_summary_guardrail(
        self,
        *,
        summary: Dict[str, Any],
        action: Optional[Dict[str, Any]],
        rating_system_v2: Dict[str, Any],
    ) -> Dict[str, Any]:
        action_name = ((action or {}).get("action") or "UNKNOWN").upper()
        try:
            rating_overall = int((rating_system_v2.get("overall") or {}).get("score") or 0)
        except Exception:
            rating_overall = 0

        if (
            summary.get("overall_severity") == "VERY_HIGH"
            and summary.get("primary_pillar") == "PROTECTION"
            and action_name in {"SEMI_OPEN", "SIDE_ON"}
            and rating_overall >= 60
        ):
            softened = dict(summary)
            softened["overall_assessment"] = (
                "This clip looks sound overall, with no strong concern from this one ball."
            )
            softened["overall_severity"] = "LOW"
            softened["recommendation_level"] = "MAINTAIN"
            return softened

        return summary

    def build_elbow(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ext_raw = raw.get("extension_deg")
        ext = None if ext_raw is None else f(ext_raw)
        verdict = (raw.get("verdict") or "").upper()
        reason = (raw.get("reason") or "").lower()

        if ext is None:
            if verdict == "ILLEGAL":
                band = "ILLEGAL"
                text = "Your bowling-arm flow appears abrupt through delivery and should be reviewed carefully for legality."
            elif verdict == "SUSPECT":
                band = "REVIEW"
                text = "Your bowling-arm motion could not be cleared confidently and should be reviewed for legality."
            elif verdict == "LEGAL":
                band = "OK"
                text = "Your bowling-arm flow appears smooth through delivery."
            else:
                band = "REVIEW"
                text = "Elbow legality could not be confirmed clearly from this clip."
            return {
                "extension_deg": None,
                "band": band,
                "player_text": text,
                "reason": reason or None,
            }

        if verdict == "ILLEGAL":
            band = "ILLEGAL"
            text = "Your elbow action exceeds the legal extension limit."
        elif ext <= 18.0:
            band = "OK"
            text = "Your elbow action is within the legal range."
        elif ext <= 22.0:
            band = "BORDERLINE"
            text = "Your elbow action is close to the legal limit and should be monitored."
        else:
            band = "ILLEGAL"
            text = "Your elbow action exceeds the legal extension limit."

        return {
            "extension_deg": round(ext, 2),
            "band": band,
            "player_text": text,
            "reason": reason or None,
        }

    def build_chain(self, interpretation: Dict[str, Any]) -> Dict[str, Any]:
        flow = (interpretation or {}).get("linear_flow") or {}
        state = (flow.get("flow_state") or "unknown").lower()

        if state == "interrupted":
            msg = "The run-up is not getting into the ball cleanly in this clip."
            what_we_saw = "The action looked broken between landing and ball release in this clip."
            why_flagged = "The body did not keep moving through the action as smoothly as we wanted."
            why_it_matters = "That can make the ball come out with less help from the whole action."
            what_to_check = "Check if the bowler keeps moving through the ball smoothly across a few balls."
        elif state == "smooth":
            msg = "The action stayed fairly connected from run-up into the ball in this clip."
            what_we_saw = "The action looked fairly connected from run-up into ball release."
            why_flagged = "The flow through the action did not stand out as a main concern here."
            why_it_matters = "That is a good sign, though one ball never tells the full story."
            what_to_check = "Keep checking if the same flow repeats across a few balls."
        else:
            msg = "This clip did not show clearly how the run-up carried into the ball."
            what_we_saw = "This clip did not show the full action clearly enough."
            why_flagged = "We could not see enough of the flow from run-up into the ball."
            why_it_matters = "That means this clip should be treated as unclear, not as proof that everything is fine."
            what_to_check = "Record another clip with the full action visible."

        return {
            "state": state.upper(),
            "kid_summary": msg,
            "confidence": f(flow.get("confidence"), 1.0),
            "what_we_saw": what_we_saw,
            "why_flagged": why_flagged,
            "why_it_matters": why_it_matters,
            "what_to_check_with_coach": what_to_check,
        }

    def build_risks(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for r in raw or []:
            rid = r.get("risk_id")
            if not rid:
                continue

            spec = RISKS.get(rid) or {}
            sev = severity_band(f(r.get("signal_strength")))
            conf = confidence_band(f(r.get("confidence")))

            evidence = dict(r)
            evidence["signal_strength"] = round(f(r.get("signal_strength")), 3)
            evidence["confidence"] = round(f(r.get("confidence")), 3)

            kid_text = (
                spec.get("explanations", {}).get("kid", {}).get(sev.lower())
                or spec.get("kid_explanation")
                or "This movement pattern is worth monitoring."
            )
            coach_text = (
                spec.get("explanations", {}).get("coach", {}).get(sev.lower())
                or spec.get("coach_explanation")
                or "Monitor this movement pattern."
            )
            reporting = _reporting_lines_for_risk(rid, sev)

            out.append(
                {
                    "risk_id": rid,
                    "title": spec.get("title", rid.replace("_", " ").title()),
                    "body_region": spec.get("body_region"),
                    "severity": sev,
                    "confidence": conf,
                    "kid_explanation": kid_text,
                    "coach_explanation": coach_text,
                    "what_we_saw": reporting["what_we_saw"],
                    "why_flagged": reporting["why_flagged"],
                    "why_it_matters": reporting["why_it_matters"],
                    "what_to_check_with_coach": reporting["what_to_check"],
                    "cues": spec.get("cues", {}),
                    "evidence": evidence,
                }
            )

        return out

    def _pillar_scores(
        self,
        *,
        chain: Dict[str, Any],
        risks: List[Dict[str, Any]],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        scores = {name: 0.0 for name in PILLAR_SPECS}

        for risk in risks or []:
            rid = risk.get("risk_id")
            severity_points = SEVERITY_POINTS.get((risk.get("severity") or "LOW").upper(), 0.0)
            confidence_weight = CONFIDENCE_WEIGHTS.get((risk.get("confidence") or "LOW").upper(), 0.4)
            for pillar, weight in (RISK_PILLAR_WEIGHTS.get(rid) or {}).items():
                scores[pillar] += severity_points * confidence_weight * weight

        if (chain.get("state") or "").upper() == "INTERRUPTED":
            scores["POWER"] += 1.5

        action_name = ((action or {}).get("action") or "UNKNOWN").upper()
        if action_name == "MIXED":
            scores["POSTURE"] += 1.25
        elif action_name == "UNKNOWN":
            scores["POSTURE"] += 1.5

        elbow_band = (elbow or {}).get("band")
        if elbow_band == "ILLEGAL":
            scores["PROTECTION"] += 3.0
        elif elbow_band == "REVIEW":
            scores["PROTECTION"] += 1.5

        return scores

    def _pillar_status(self, score: float) -> str:
        if score >= 3.0:
            return "REVIEW"
        if score >= 1.0:
            return "WATCH"
        return "CLEAR"

    def _continuous_pillar_penalty(
        self,
        *,
        pillar_name: str,
        risks: List[Dict[str, Any]],
        chain: Dict[str, Any],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> float:
        penalty = 0.0

        for risk in risks or []:
            rid = risk.get("risk_id")
            weights = RISK_PILLAR_WEIGHTS.get(rid) or {}
            if pillar_name not in weights:
                continue
            penalty += (
                _normalized_signal(_raw_signal_value(risk))
                * _raw_confidence_value(risk)
                * weights[pillar_name]
            )

        chain_state = (chain.get("state") or "").upper()
        chain_conf = f(chain.get("confidence"), 0.5)
        if pillar_name == "POWER":
            if chain_state == "INTERRUPTED":
                penalty += 0.90 * chain_conf
            elif chain_state not in {"SMOOTH", "INTERRUPTED"}:
                penalty += 0.35

        action_name = ((action or {}).get("action") or "UNKNOWN").upper()
        action_conf = f((action or {}).get("confidence"), 0.5)
        if pillar_name == "POSTURE":
            if action_name == "MIXED":
                penalty += 0.80 * max(action_conf, 0.6)
            elif action_name == "UNKNOWN":
                penalty += 1.00

        elbow_band = (elbow or {}).get("band")
        if pillar_name == "PROTECTION":
            if elbow_band == "ILLEGAL":
                penalty += 1.60
            elif elbow_band in {"REVIEW", "BORDERLINE"}:
                penalty += 0.90

        return penalty

    def _pillar_positive_bonus(
        self,
        *,
        pillar_name: str,
        risks: List[Dict[str, Any]],
        chain: Dict[str, Any],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> float:
        bonus = 0.0
        related_signals = [
            _raw_signal_value(risk)
            for risk in risks or []
            if pillar_name in (RISK_PILLAR_WEIGHTS.get(risk.get("risk_id")) or {})
        ]
        max_related_signal = max(related_signals, default=0.0)

        chain_state = (chain.get("state") or "").upper()
        chain_conf = f(chain.get("confidence"), 0.5)
        action_name = ((action or {}).get("action") or "UNKNOWN").upper()
        action_conf = f((action or {}).get("confidence"), 0.5)
        elbow_band = (elbow or {}).get("band")

        if pillar_name == "POSTURE":
            if action_name in {"SEMI_OPEN", "SIDE_ON", "FRONT_ON", "OPEN"} and action_conf >= 0.70:
                bonus += 5.0 * action_conf
            if max_related_signal < 0.25:
                bonus += 5.0 * max(action_conf, 0.70)
        elif pillar_name == "POWER":
            if chain_state == "SMOOTH" and chain_conf >= 0.75:
                bonus += 4.5 * chain_conf
            if max_related_signal < 0.25:
                bonus += 4.5 * max(chain_conf, 0.70)
        elif pillar_name == "PROTECTION":
            related_confidences = [
                _raw_confidence_value(risk)
                for risk in risks or []
                if pillar_name in (RISK_PILLAR_WEIGHTS.get(risk.get("risk_id")) or {})
            ]
            protection_conf = _average(related_confidences, 0.75)
            if elbow_band == "OK":
                bonus += 4.0 * protection_conf
            if max_related_signal < 0.25:
                bonus += 4.0 * protection_conf

        return min(bonus, PILLAR_MAX_BONUS.get(pillar_name, 8.0))

    def _pillar_benchmark_score(
        self,
        *,
        pillar_name: str,
        risks: List[Dict[str, Any]],
        chain: Dict[str, Any],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> int:
        base = PILLAR_BASE_SCORE
        bonus = self._pillar_positive_bonus(
            pillar_name=pillar_name,
            risks=risks,
            chain=chain,
            elbow=elbow,
            action=action,
        )
        penalty = self._continuous_pillar_penalty(
            pillar_name=pillar_name,
            risks=risks,
            chain=chain,
            elbow=elbow,
            action=action,
        )
        scale = PILLAR_PENALTY_SCALE.get(pillar_name, 16.0)
        return _clamp_score(base + bonus - (penalty * scale))

    def _overall_score_cap(
        self,
        *,
        chain: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[int] = None,
    ) -> tuple[int, Optional[str]]:
        action_name = ((action or {}).get("action") or "UNKNOWN").upper()
        chain_state = (chain.get("state") or "").upper()

        cap = 100
        reason: Optional[str] = None

        if action_name == "UNKNOWN" and chain_state not in {"SMOOTH", "INTERRUPTED"}:
            cap = 72
            reason = "Score capped because the clip did not show the full action clearly enough."
        elif action_name == "UNKNOWN":
            cap = 80
            reason = "Score capped because the action view was not clear enough to compare against the full benchmark."
        elif chain_state not in {"SMOOTH", "INTERRUPTED"}:
            cap = 80
            reason = "Score capped because the clip did not show clearly how the action carried into the ball."
        elif action_name == "MIXED":
            cap = 86
            reason = "Score capped because a mixed action should not be treated like a fully clear benchmark clip."

        if confidence_score is not None:
            if confidence_score < 55:
                if cap > 84:
                    cap = 84
                    reason = "Score capped because ActionLab could not read the clip clearly enough."
            elif confidence_score < 70:
                if cap > 90:
                    cap = 90
                    reason = "Score capped because the clip was only moderately clear."
            elif confidence_score < 85:
                if cap > 94:
                    cap = 94
                    reason = "Score capped because this was a good read, but not a perfect view."

        return cap, reason

    def _pillar_score_cap(
        self,
        *,
        pillar_name: str,
        chain: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> tuple[int, Optional[str]]:
        action_name = ((action or {}).get("action") or "UNKNOWN").upper()
        chain_state = (chain.get("state") or "").upper()

        if pillar_name == "POSTURE":
            if action_name == "UNKNOWN":
                return 80, "Balance score capped because the action view was not clear enough."
            if action_name == "MIXED":
                return 86, "Balance score capped because a mixed action should not be treated like a full benchmark clip."

        if pillar_name == "POWER" and chain_state not in {"SMOOTH", "INTERRUPTED"}:
            return 80, "Carry score capped because the clip did not show clearly how the action moved into the ball."

        return 100, None

    def _overall_score_label(self, score: int) -> str:
        if score >= 90:
            return "Strong score"
        if score >= 75:
            return "Good score"
        if score >= 60:
            return "Work to do"
        return "Needs review"

    def _overall_score_meaning(self, score: int) -> str:
        if score >= 90:
            return "This is a strong score, but it still sits below the ideal action standard."
        if score >= 75:
            return "This clip has a useful base, but visible deductions are still keeping it below the ideal standard."
        if score >= 60:
            return "This clip shows some useful parts, but the deductions are now large enough to hold the score back clearly."
        return "This clip sits well below the ideal action standard and needs careful review."

    def _confidence_score(self, chain: Dict[str, Any], risks: List[Dict[str, Any]], action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        chain_conf = f(chain.get("confidence"), 0.5)
        risk_conf_values = [
            CONFIDENCE_WEIGHTS.get((risk.get("confidence") or "LOW").upper(), 0.4)
            for risk in risks or []
        ]
        risk_conf = sum(risk_conf_values) / len(risk_conf_values) if risk_conf_values else 0.6
        action_conf = f((action or {}).get("confidence"), 0.5)
        score = _clamp_score((chain_conf * 0.45 + risk_conf * 0.35 + action_conf * 0.20) * 100.0)

        if score >= 85:
            label = "High confidence"
            meaning = "ActionLab had a clear view of the action and the score is easier to trust."
        elif score >= 65:
            label = "Medium confidence"
            meaning = "ActionLab could read the action fairly well, but this clip should still be checked with others."
        else:
            label = "Low confidence"
            meaning = "ActionLab could not read this clip clearly enough to treat the score as a strong conclusion."

        return {
            "score": score,
            "label": label,
            "what_it_measures": "This shows how clearly ActionLab could read the clip.",
            "what_100_means": "100 means ActionLab had a very clear view of the full action.",
            "what_this_score_means": meaning,
        }

    def _score_reason_breakdown(
        self,
        *,
        pillar_name: str,
        risks: List[Dict[str, Any]],
        chain: Dict[str, Any],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        reasons: List[tuple[str, float]] = []

        for risk in risks or []:
            rid = risk.get("risk_id")
            weights = RISK_PILLAR_WEIGHTS.get(rid) or {}
            if pillar_name not in weights:
                continue
            contribution = (
                _normalized_signal(_raw_signal_value(risk))
                * _raw_confidence_value(risk)
                * weights[pillar_name]
            )
            if contribution > 0:
                reasons.append((str(risk.get("title") or rid or "Signal"), contribution))

        if pillar_name == "POWER" and (chain.get("state") or "").upper() == "INTERRUPTED":
            reasons.append(("Action flow", 0.90 * f(chain.get("confidence"), 0.5)))
        elif pillar_name == "POWER" and (chain.get("state") or "").upper() not in {"SMOOTH", "INTERRUPTED"}:
            reasons.append(("Unclear action flow", 0.35))

        action_name = ((action or {}).get("action") or "UNKNOWN").upper()
        if pillar_name == "POSTURE":
            if action_name == "MIXED":
                reasons.append(("Mixed action", 0.80 * max(f((action or {}).get("confidence"), 0.5), 0.6)))
            elif action_name == "UNKNOWN":
                reasons.append(("Unclear action view", 1.00))

        elbow_band = (elbow or {}).get("band")
        if pillar_name == "PROTECTION":
            if elbow_band == "ILLEGAL":
                reasons.append(("Elbow check", 1.60))
            elif elbow_band in {"REVIEW", "BORDERLINE"}:
                reasons.append(("Elbow check", 0.90))

        reasons.sort(key=lambda item: item[1], reverse=True)
        return [_pretty_reason(title) for title, _ in reasons[:3]]

    def _overall_reason_breakdown(self, pillar_scores: Dict[str, Dict[str, Any]]) -> List[str]:
        all_reasons: List[str] = []
        for slug in ("balance", "carry", "body_load"):
            for reason in pillar_scores.get(slug, {}).get("main_reasons_points_were_lost", []):
                if reason not in all_reasons:
                    all_reasons.append(reason)
        return all_reasons[:4]

    def _risk_lookup(self, risks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            str(risk.get("risk_id")): risk
            for risk in (risks or [])
            if risk.get("risk_id")
        }

    def _feature_from_risk(
        self,
        *,
        key: str,
        label: str,
        risk: Optional[Dict[str, Any]],
        max_deduction: int,
        affects: List[str],
        what_it_measures: str,
        why_it_matters: str,
        needs_to_improve: str,
    ) -> Dict[str, Any]:
        if risk:
            deduction = int(
                round(
                    min(
                        max_deduction,
                        _normalized_signal(_raw_signal_value(risk))
                        * _raw_confidence_value(risk)
                        * max_deduction,
                    )
                )
            )
            confidence = _raw_confidence_value(risk)
            score = _feature_score_from_deduction(deduction, max_deduction)
            what_this_means = risk.get("what_we_saw") or "This is worth checking in this clip."
            signal = _raw_signal_value(risk)
            severity_band = (risk.get("severity") or "LOW").lower()
            source_detector = risk.get("risk_id")
        else:
            deduction = 0
            confidence = 0.4
            score = 100
            what_this_means = "This did not stand out as a main issue in this clip."
            signal = 0.0
            severity_band = "low"
            source_detector = None

        return {
            "key": key,
            "label": label,
            "score": score,
            "deduction": deduction,
            "severity_band": severity_band,
            "confidence": round(confidence, 2),
            "what_it_measures": what_it_measures,
            "what_this_means": what_this_means,
            "why_it_matters": why_it_matters,
            "needs_to_improve": needs_to_improve,
            "affects": affects,
            "raw_signal": round(signal, 3),
            "source_detector": source_detector,
        }

    def _build_rating_features(
        self,
        *,
        risks: List[Dict[str, Any]],
        chain: Dict[str, Any],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        lookup = self._risk_lookup(risks)
        features: List[Dict[str, Any]] = []

        features.append(
            self._feature_from_risk(
                key="front_foot_line",
                label="Front Foot Line",
                risk=lookup.get("foot_line_deviation"),
                max_deduction=FEATURE_MAX_DEDUCTIONS["front_foot_line"],
                affects=["lower_body_alignment", "whole_body_alignment", "safety"],
                what_it_measures="This shows where the front foot lands.",
                why_it_matters="A wide landing line can affect balance and can make the body take more load.",
                needs_to_improve="Try to land straighter.",
            )
        )

        knee_feature = self._feature_from_risk(
            key="front_leg_support",
            label="Front-Leg Support",
            risk=lookup.get("knee_brace_failure"),
            max_deduction=FEATURE_MAX_DEDUCTIONS["front_leg_support"],
            affects=["lower_body_alignment", "whole_body_alignment", "momentum_forward", "safety"],
            what_it_measures="This shows how well the front leg supports the action at landing.",
            why_it_matters="If the front leg does not support the action well, the rest of the action has less help.",
            needs_to_improve="Try to land stronger and hold your shape.",
        )
        braking = lookup.get("front_foot_braking_shock")
        if braking:
            braking_deduction = int(
                round(
                    min(
                        4.0,
                        _normalized_signal(_raw_signal_value(braking))
                        * _raw_confidence_value(braking)
                        * 4.0,
                    )
                )
            )
            knee_feature["deduction"] = min(
                FEATURE_MAX_DEDUCTIONS["front_leg_support"],
                knee_feature["deduction"] + braking_deduction,
            )
            knee_feature["score"] = _feature_score_from_deduction(
                knee_feature["deduction"],
                FEATURE_MAX_DEDUCTIONS["front_leg_support"],
            )
            if braking_deduction > 0 and knee_feature["source_detector"] != "knee_brace_failure":
                knee_feature["source_detector"] = "front_foot_braking_shock"
        features.append(knee_feature)

        features.append(
            self._feature_from_risk(
                key="trunk_lean",
                label="Trunk Lean",
                risk=lookup.get("lateral_trunk_lean"),
                max_deduction=FEATURE_MAX_DEDUCTIONS["trunk_lean"],
                affects=["upper_body_alignment", "whole_body_alignment", "safety"],
                what_it_measures="This shows if the upper body is falling away to the side.",
                why_it_matters="Falling away can affect shape and can put more stress on the body.",
                needs_to_improve="Try not to fall away to the side.",
            )
        )

        features.append(
            self._feature_from_risk(
                key="upper_body_opening",
                label="Upper Body Opening",
                risk=lookup.get("hip_shoulder_mismatch"),
                max_deduction=FEATURE_MAX_DEDUCTIONS["upper_body_opening"],
                affects=["upper_body_alignment", "whole_body_alignment", "momentum_forward"],
                what_it_measures="This shows how well the upper body stays in shape into release.",
                why_it_matters="Opening too early can make the action lose some flow before the ball comes out.",
                needs_to_improve="Try not to open up too early.",
            )
        )

        features.append(
            self._feature_from_risk(
                key="trunk_rotation_load",
                label="Trunk Rotation Load",
                risk=lookup.get("trunk_rotation_snap"),
                max_deduction=FEATURE_MAX_DEDUCTIONS["trunk_rotation_load"],
                affects=["upper_body_alignment", "momentum_forward", "safety"],
                what_it_measures="This shows how hard the upper body is turning.",
                why_it_matters="If the upper body turns too sharply, it can add load even when the action still has pace.",
                needs_to_improve="Try not to pull the upper body around too hard.",
            )
        )

        chain_state = (chain.get("state") or "").upper()
        chain_conf = f(chain.get("confidence"), 0.5)
        if chain_state == "SMOOTH":
            flow_deduction = 0
            flow_meaning = chain.get("what_we_saw") or "The action stayed fairly connected into the ball."
        elif chain_state == "INTERRUPTED":
            flow_deduction = int(round(8 + (1.0 - chain_conf) * 4.0))
            flow_meaning = chain.get("what_we_saw") or "The action looked broken between landing and release."
        else:
            flow_deduction = int(round(6 + (1.0 - chain_conf) * 3.0))
            flow_meaning = chain.get("what_we_saw") or "This clip did not show clearly how the action moved into the ball."
        features.append(
            {
                "key": "action_flow",
                "label": "Action Flow",
                "score": _feature_score_from_deduction(
                    flow_deduction,
                    FEATURE_MAX_DEDUCTIONS["action_flow"],
                ),
                "deduction": min(FEATURE_MAX_DEDUCTIONS["action_flow"], flow_deduction),
                "severity_band": "moderate" if flow_deduction >= 6 else "low",
                "confidence": round(chain_conf, 2),
                "what_it_measures": "This shows how well the action stays connected into the ball.",
                "what_this_means": flow_meaning,
                "why_it_matters": "If the action loses flow, the ball can come out with less help from the whole action.",
                "needs_to_improve": "Try to let the action flow all the way into the ball.",
                "affects": ["whole_body_alignment", "momentum_forward"],
                "raw_signal": round(max(0.0, flow_deduction / FEATURE_MAX_DEDUCTIONS["action_flow"]), 3),
                "source_detector": "kinetic_chain",
            }
        )

        action_name = ((action or {}).get("action") or "UNKNOWN").upper()
        action_conf = f((action or {}).get("confidence"), 0.5)
        action_deduction_map = {
            "SIDE_ON": 0,
            "SEMI_CLOSED": 2,
            "SEMI_OPEN": 2,
            "OPEN": 3,
            "FRONT_ON": 4,
            "MIXED": 9,
            "UNKNOWN": 5,
        }
        action_deduction = int(round(action_deduction_map.get(action_name, 3) * max(action_conf, 0.6)))
        features.append(
            {
                "key": "action_type",
                "label": "Action Type",
                "score": _feature_score_from_deduction(
                    action_deduction,
                    FEATURE_MAX_DEDUCTIONS["action_type"],
                ),
                "deduction": min(FEATURE_MAX_DEDUCTIONS["action_type"], action_deduction),
                "severity_band": "high" if action_name == "MIXED" else "moderate" if action_deduction >= 4 else "low",
                "confidence": round(action_conf, 2),
                "what_it_measures": "This shows the overall shape of the action.",
                "what_this_means": f"This clip looks {action_name.replace('_', ' ').lower()}.",
                "why_it_matters": "Some action shapes are easier for the body to handle than others, especially if they start linking with other issues.",
                "needs_to_improve": "Try to keep the action shape more consistent.",
                "affects": ["lower_body_alignment", "upper_body_alignment", "whole_body_alignment", "safety"],
                "raw_signal": round(max(0.0, action_deduction / FEATURE_MAX_DEDUCTIONS["action_type"]), 3),
                "source_detector": "action",
                "classification": action_name.lower(),
            }
        )

        elbow_band = (elbow or {}).get("band") or "OK"
        elbow_deduction_map = {
            "OK": 0,
            "BORDERLINE": 2,
            "REVIEW": 3,
            "ILLEGAL": 5,
        }
        elbow_deduction = elbow_deduction_map.get(elbow_band, 0)
        features.append(
            {
                "key": "elbow_review",
                "label": "Elbow Review",
                "score": _feature_score_from_deduction(
                    elbow_deduction,
                    FEATURE_MAX_DEDUCTIONS["elbow_review"],
                ),
                "deduction": elbow_deduction,
                "severity_band": "high" if elbow_band == "ILLEGAL" else "moderate" if elbow_deduction >= 3 else "low",
                "confidence": 1.0 if elbow_band != "UNKNOWN" else 0.4,
                "what_it_measures": "This checks if the arm action needs a closer look.",
                "what_this_means": (elbow or {}).get("player_text") or "This did not stand out as a main issue in this clip.",
                "why_it_matters": "If the arm path needs review, it should be checked with a coach.",
                "needs_to_improve": "Ask a coach to check the arm path closely.",
                "affects": ["safety"],
                "raw_signal": round(max(0.0, elbow_deduction / FEATURE_MAX_DEDUCTIONS["elbow_review"]), 3),
                "source_detector": "elbow",
            }
        )

        return features

    def _build_linked_penalties(self, features: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        penalties: List[Dict[str, Any]] = []

        def add(key: str, deduction: int, why: str) -> None:
            if deduction <= 0:
                return
            penalties.append(
                {
                    "key": key,
                    "deduction": deduction,
                    "why_it_matters": why,
                }
            )

        foot = int((features.get("front_foot_line") or {}).get("deduction", 0))
        lean = int((features.get("trunk_lean") or {}).get("deduction", 0))
        opening = int((features.get("upper_body_opening") or {}).get("deduction", 0))
        support = int((features.get("front_leg_support") or {}).get("deduction", 0))
        flow = int((features.get("action_flow") or {}).get("deduction", 0))
        trunk = int((features.get("trunk_rotation_load") or {}).get("deduction", 0))
        action_type = features.get("action_type") or {}
        action_class = str(action_type.get("classification") or "")

        add(
            "foot_line_plus_trunk_lean",
            min(4, int(round(min(foot, lean) * 0.25))),
            "The landing line and trunk lean are showing up together. That makes the action harder to control.",
        )
        add(
            "foot_line_plus_upper_body_opening",
            min(3, int(round(min(foot, opening) * 0.20))),
            "Landing off line can make it harder for the upper body to stay in shape.",
        )
        add(
            "front_leg_plus_action_flow",
            min(3, int(round(min(support, flow) * 0.20))),
            "When support and flow both drop, the action loses help into release.",
        )
        add(
            "trunk_lean_plus_trunk_rotation",
            min(3, int(round(min(lean, trunk) * 0.20))),
            "Leaning and sharp upper-body turn together can add more body load.",
        )
        if action_class == "mixed":
            add(
                "mixed_action_plus_trunk_lean",
                min(5, max(3, int(round(lean * 0.35)))),
                "Mixed action and trunk lean together should be treated as a stronger body-load concern.",
            )

        total = 0
        capped: List[Dict[str, Any]] = []
        for penalty in penalties:
            remaining = 10 - total
            if remaining <= 0:
                break
            deduction = min(int(penalty["deduction"]), remaining)
            total += deduction
            capped.append({**penalty, "deduction": deduction})
        return capped

    def _group_score(
        self,
        *,
        group_key: str,
        features: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        weights = GROUP_FEATURE_WEIGHTS[group_key]
        total_weight = sum(weights.values()) or 1.0
        score = round(
            sum((features.get(key) or {}).get("score", 100) * weight for key, weight in weights.items()) / total_weight
        )
        main_feature_key = max(
            weights.keys(),
            key=lambda key: ((features.get(key) or {}).get("deduction", 0) * weights[key]),
        )
        main_feature = features.get(main_feature_key) or {"label": "No main issue", "deduction": 0}
        spec = GROUP_SPECS[group_key]

        if score >= 85:
            meaning = "This part of the action looked solid in this clip."
        elif score >= 70:
            meaning = "This part of the action has a good base, but it can get cleaner."
        elif score >= 55:
            meaning = "This part of the action is losing shape or support in this clip."
        else:
            meaning = "This part of the action needs close work with a coach."

        return {
            "score": score,
            "label": _score_band_label(score),
            "what_it_measures": spec["what_it_measures"],
            "what_high_score_means": spec["what_high_score_means"],
            "what_this_score_means": meaning,
            "main_issue": main_feature.get("label"),
            "main_deductions": [
                (features.get(key) or {}).get("label")
                for key in sorted(
                    weights.keys(),
                    key=lambda item: (features.get(item) or {}).get("deduction", 0),
                    reverse=True,
                )
                if (features.get(key) or {}).get("deduction", 0) > 0
            ][:3],
        }

    def build_rating_system_v2(
        self,
        *,
        chain: Dict[str, Any],
        risks: List[Dict[str, Any]],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
        scorecard: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        features_list = self._build_rating_features(
            risks=risks,
            chain=chain,
            elbow=elbow,
            action=action,
        )
        features = {feature["key"]: feature for feature in features_list}
        linked_penalties = self._build_linked_penalties(features)
        confidence = self._confidence_score(chain, risks, action)

        summary_groups = {
            key: self._group_score(group_key=key, features=features)
            for key in GROUP_SPECS
        }

        weighted_total = (
            summary_groups["lower_body_alignment"]["score"] * 0.22
            + summary_groups["upper_body_alignment"]["score"] * 0.22
            + summary_groups["whole_body_alignment"]["score"] * 0.26
            + summary_groups["momentum_forward"]["score"] * 0.20
            + summary_groups["safety"]["score"] * 0.10
        )
        overall_score = _clamp_score(weighted_total - sum(p["deduction"] for p in linked_penalties))
        overall_cap, cap_reason = self._overall_score_cap(
            chain=chain,
            action=action,
            confidence_score=confidence["score"],
        )
        if overall_score > overall_cap:
            overall_score = overall_cap

        top_deductions = [
            {
                "key": feature["key"],
                "label": feature["label"],
                "deduction": feature["deduction"],
            }
            for feature in sorted(features_list, key=lambda item: item["deduction"], reverse=True)
            if feature["deduction"] > 0
        ][:4]

        needs_to_improve = [
            {
                "key": feature["key"],
                "label": feature["needs_to_improve"],
            }
            for feature in sorted(features_list, key=lambda item: item["deduction"], reverse=True)
            if feature["deduction"] > 0
        ][:3]

        if overall_score >= 85:
            overall_meaning = "This action is working well in this clip, with only a few visible leaks."
        elif overall_score >= 70:
            overall_meaning = "This action has a good base, but a few visible leaks are holding it back."
        elif overall_score >= 55:
            overall_meaning = "This action has useful parts, but clear issues are now pulling it down."
        else:
            overall_meaning = "This action needs close work with a coach before it can work better and safer."

        overall = {
            "score": overall_score,
            "label": _score_band_label(overall_score),
            "what_it_measures": "This score shows how well the bowling action is working in this clip. It also checks how well the body is handling it.",
            "what_high_score_means": "A high score means the action is working well, staying better lined up, and not showing strong signs of extra stress.",
            "what_this_score_means": overall_meaning,
            "what_lowered_the_score": [item["label"] for item in top_deductions[:3]],
            "needs_to_improve": [item["label"] for item in needs_to_improve],
        }
        if cap_reason:
            overall["benchmark_limit_reason"] = cap_reason

        player_view = {
            "metrics": {
                key: summary_groups[key]
                for key in (
                    "upper_body_alignment",
                    "lower_body_alignment",
                    "whole_body_alignment",
                    "momentum_forward",
                )
            },
            "what_to_work_on": [item["label"] for item in top_deductions[:2]],
            "coach_guidance_line": "Talk to a coach to help you understand what this means.",
        }

        coach_view = {
            "metrics": {
                key: summary_groups[key]
                for key in (
                    "upper_body_alignment",
                    "lower_body_alignment",
                    "whole_body_alignment",
                    "momentum_forward",
                    "safety",
                )
            },
            "top_deductions": top_deductions,
            "needs_to_improve": needs_to_improve,
            "linked_penalties": linked_penalties,
            "coach_guidance_line": "Talk to a coach to help you understand what this means.",
        }

        expert_view = {
            "metrics": {**summary_groups, "confidence": confidence},
            "features": features_list,
            "linked_penalties": linked_penalties,
            "action_type": {
                "classification": ((action or {}).get("action") or "UNKNOWN").upper(),
                "confidence": round(f((action or {}).get("confidence"), 0.5), 2),
            },
            "chain": {
                "state": chain.get("state"),
                "confidence": round(f(chain.get("confidence"), 0.5), 2),
            },
            "elbow": {
                "band": elbow.get("band"),
                "extension_deg": elbow.get("extension_deg"),
            },
            "legacy_scorecard": scorecard or {},
        }

        return {
            "version": "rating_system_v2",
            "overall": overall,
            "player_view": player_view,
            "coach_view": coach_view,
            "expert_view": expert_view,
            "features": features_list,
            "linked_penalties": linked_penalties,
            "top_deductions": top_deductions,
            "needs_to_improve": needs_to_improve,
            "coach_guidance_line": "Talk to a coach to help you understand what this means.",
            "confidence": confidence,
        }

    def _primary_pillar_from_scorecard(
        self,
        *,
        scorecard: Dict[str, Any],
        chain: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> str:
        pillar_scores = scorecard.get("pillars") or {}
        action_name = ((action or {}).get("action") or "UNKNOWN").upper()
        chain_state = (chain.get("state") or "").upper()

        candidates = [
            ("POSTURE", pillar_scores.get("balance") or {}),
            ("POWER", pillar_scores.get("carry") or {}),
            ("PROTECTION", pillar_scores.get("body_load") or {}),
        ]

        ranked: List[tuple[float, int, int, str]] = []
        for pillar_name, payload in candidates:
            score = f(payload.get("score"), 0.0)
            reasons = payload.get("main_reasons_points_were_lost") or []
            capped = 1 if payload.get("benchmark_limit_reason") else 0
            loss = 100.0 - score
            ranked.append((loss, len(reasons), capped, pillar_name))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        top_loss, top_reasons, top_capped, primary = ranked[0]

        # If the scores are clustered tightly, lean on the main action story rather
        # than defaulting back to posture by tuple order.
        if len(ranked) > 1 and abs(top_loss - ranked[1][0]) <= 3.0:
            if chain_state not in {"SMOOTH", "INTERRUPTED"}:
                return "POWER"
            if action_name in {"MIXED", "UNKNOWN"}:
                return "POSTURE"
            if top_capped and primary == "POWER":
                return "POWER"

        return primary

    def build_scorecard(
        self,
        *,
        chain: Dict[str, Any],
        risks: List[Dict[str, Any]],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_scores = self._pillar_scores(
            chain=chain,
            risks=risks,
            elbow=elbow,
            action=action,
        )

        pillar_score_map: Dict[str, Dict[str, Any]] = {}
        weighted_total = 0.0
        for pillar_name, raw_score in raw_scores.items():
            spec = PILLAR_SPECS[pillar_name]
            benchmark_score = self._pillar_benchmark_score(
                pillar_name=pillar_name,
                risks=risks,
                chain=chain,
                elbow=elbow,
                action=action,
            )
            pillar_cap, pillar_cap_reason = self._pillar_score_cap(
                pillar_name=pillar_name,
                chain=chain,
                action=action,
            )
            if benchmark_score > pillar_cap:
                benchmark_score = pillar_cap
            weighted_total += benchmark_score * OVERALL_SCORE_WEIGHTS[pillar_name]
            pillar_payload = {
                "label": spec["display_label"],
                "score": benchmark_score,
                "what_it_measures": spec["what_it_measures"],
                "what_100_means": spec["what_100_means"],
                "what_lowers_score": spec["what_lowers_score"],
                "main_reasons_points_were_lost": self._score_reason_breakdown(
                    pillar_name=pillar_name,
                    risks=risks,
                    chain=chain,
                    elbow=elbow,
                    action=action,
                ),
            }
            if pillar_cap_reason:
                pillar_payload["benchmark_limit_reason"] = pillar_cap_reason
            pillar_score_map[spec["slug"]] = pillar_payload

        confidence = self._confidence_score(chain, risks, action)
        overall_score = _clamp_score(weighted_total)
        overall_cap, cap_reason = self._overall_score_cap(
            chain=chain,
            action=action,
            confidence_score=confidence["score"],
        )
        if overall_score > overall_cap:
            overall_score = overall_cap

        overall_payload = {
            "overall": {
                "score": overall_score,
                "label": self._overall_score_label(overall_score),
                "what_it_measures": "This score shows how strong and efficient the bowling action looks in this clip, while also checking if the body is handling the action well.",
                "what_100_means": "100 means the clip is sitting at the ideal action standard for the things ActionLab can measure: balanced, carrying well into release, and showing very few visible stress warnings. Very few clips should reach it.",
                "what_this_score_means": self._overall_score_meaning(overall_score),
                "what_lowers_score": "The score drops when the action loses balance, leaks energy before release, or puts extra stress on the body.",
                "main_reasons_points_were_lost": self._overall_reason_breakdown(pillar_score_map),
            },
            "confidence": confidence,
            "pillars": pillar_score_map,
            "weights": {
                "balance": int(OVERALL_SCORE_WEIGHTS["POSTURE"] * 100),
                "carry": int(OVERALL_SCORE_WEIGHTS["POWER"] * 100),
                "body_load": int(OVERALL_SCORE_WEIGHTS["PROTECTION"] * 100),
            },
            "benchmark": {
                "version": "scorecard_v1",
                "headline": "Overall score is built from Balance, Carry, and Body Load. Confidence is shown separately.",
                "how_to_read": "Higher scores mean the action looks closer to the ideal state for the things ActionLab can measure from video.",
            },
        }
        if cap_reason:
            overall_payload["overall"]["benchmark_limit_reason"] = cap_reason
        return overall_payload

    def build_pillars(
        self,
        *,
        chain: Dict[str, Any],
        risks: List[Dict[str, Any]],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        scores = self._pillar_scores(
            chain=chain,
            risks=risks,
            elbow=elbow,
            action=action,
        )

        pillars: Dict[str, Dict[str, Any]] = {}
        for name, spec in PILLAR_SPECS.items():
            status = self._pillar_status(scores[name])
            tone_key = "clear" if status == "CLEAR" else "watch" if status == "WATCH" else "review"
            pillar_key = name.lower()
            pillar_risks = [
                risk for risk in risks
                if (RISK_PILLAR_WEIGHTS.get(risk.get("risk_id")) or {}).get(name)
            ]
            pillar_risks.sort(
                key=lambda risk: (
                    _severity_rank((risk.get("severity") or "").upper()),
                    CONFIDENCE_WEIGHTS.get((risk.get("confidence") or "LOW").upper(), 0.4),
                    (RISK_PILLAR_WEIGHTS.get(risk.get("risk_id")) or {}).get(name, 0.0),
                ),
                reverse=True,
            )
            anchor = pillar_risks[0] if pillar_risks else None
            if anchor:
                what_we_saw = anchor.get("what_we_saw") or spec[tone_key]
                why_flagged = anchor.get("why_flagged") or spec[tone_key]
                why_it_matters = anchor.get("why_it_matters") or spec[tone_key]
                what_to_check = anchor.get("what_to_check_with_coach") or spec[tone_key]
            elif name == "POWER":
                what_we_saw = chain.get("what_we_saw") or spec[tone_key]
                why_flagged = chain.get("why_flagged") or spec[tone_key]
                why_it_matters = chain.get("why_it_matters") or spec[tone_key]
                what_to_check = chain.get("what_to_check_with_coach") or spec[tone_key]
            else:
                if name == "POSTURE":
                    what_we_saw = "The bowler stayed fairly balanced and upright in this clip."
                    why_flagged = "Balance did not stand out as a main concern here."
                    why_it_matters = "That is a good sign, though one ball never tells the full story."
                    what_to_check = "Keep checking if the bowler repeats the same shape across a few balls."
                else:
                    what_we_saw = "This clip did not show a strong sign of extra body load."
                    why_flagged = "Body load did not stand out as a main concern here."
                    why_it_matters = "That is a good sign, though this still needs repeated clips over time."
                    what_to_check = "Keep checking repeated clips instead of judging from one ball."

            benchmark_score = self._pillar_benchmark_score(
                pillar_name=name,
                risks=risks,
                chain=chain,
                elbow=elbow,
                action=action,
            )
            pillar_cap, pillar_cap_reason = self._pillar_score_cap(
                pillar_name=name,
                chain=chain,
                action=action,
            )
            if benchmark_score > pillar_cap:
                benchmark_score = pillar_cap

            pillar_payload = {
                "label": name.title(),
                "display_label": spec.get("display_label", name.title()),
                "question": spec["question"],
                "status": status,
                "score": round(scores[name], 2),
                "benchmark_score": benchmark_score,
                "what_it_measures": spec["what_it_measures"],
                "what_100_means": spec["what_100_means"],
                "what_lowers_score": spec["what_lowers_score"],
                "summary": spec[tone_key],
                "what_we_saw": what_we_saw,
                "why_flagged": why_flagged,
                "why_it_matters": why_it_matters,
                "what_to_check_with_coach": what_to_check,
            }
            if pillar_cap_reason:
                pillar_payload["benchmark_limit_reason"] = pillar_cap_reason
            pillars[pillar_key] = pillar_payload
        return pillars

    def build_summary(
        self,
        chain: Dict[str, Any],
        risks: List[Dict[str, Any]],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
        scorecard: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        red_hits = sum(
            1
            for r in risks
            if r.get("severity") in ("HIGH", "VERY_HIGH") and r.get("confidence") == "HIGH"
        )
        if scorecard is None:
            scorecard = self.build_scorecard(
                chain=chain,
                risks=risks,
                elbow=elbow,
                action=action,
            )
        pillars = self.build_pillars(
            chain=chain,
            risks=risks,
            elbow=elbow,
            action=action,
        )
        primary_pillar = self._primary_pillar_from_scorecard(
            scorecard=scorecard,
            chain=chain,
            action=action,
        )
        action_name = ((action or {}).get("action") or "UNKNOWN").upper()

        if action_name == "UNKNOWN" and red_hits == 0:
            return {
                "overall_assessment": (
                    "This clip was not clear enough for a trusted reading."
                ),
                "overall_severity": "MODERATE",
                "recommendation_level": "MONITOR",
                "confidence": chain.get("confidence", 1.0),
                "primary_pillar": "POSTURE",
            }
        if primary_pillar == "PROTECTION" and (
            red_hits >= 2 or pillars["protection"]["status"] == "REVIEW"
        ):
            return {
                "overall_assessment": (
                    "This clip needs a closer look because some parts of the body may be taking extra load."
                ),
                "overall_severity": "VERY_HIGH",
                "recommendation_level": "CORRECT",
                "confidence": chain.get("confidence", 1.0),
                "primary_pillar": "PROTECTION",
            }
        if primary_pillar == "POWER" and (
            red_hits >= 1 or pillars["power"]["status"] == "REVIEW"
        ):
            return {
                "overall_assessment": (
                    "This clip needs a closer look because the run-up does not look like it is getting into the ball cleanly."
                ),
                "overall_severity": "MODERATE",
                "recommendation_level": "MONITOR",
                "confidence": chain.get("confidence", 1.0),
                "primary_pillar": "POWER",
            }
        if action_name == "MIXED":
            return {
                "overall_assessment": (
                    "This clip needs monitoring because the action looks mixed."
                ),
                "overall_severity": "MODERATE",
                "recommendation_level": "MONITOR",
                "confidence": chain.get("confidence", 1.0),
                "primary_pillar": "POSTURE",
            }
        if primary_pillar == "POSTURE" and pillars["posture"]["status"] in {"WATCH", "REVIEW"}:
            return {
                "overall_assessment": (
                    "This clip needs monitoring because the bowler does not stay as balanced and upright as we would like."
                ),
                "overall_severity": "MODERATE",
                "recommendation_level": "MONITOR",
                "confidence": chain.get("confidence", 1.0),
                "primary_pillar": "POSTURE",
            }
        return {
            "overall_assessment": (
                "This clip looks sound overall, with no strong concern from this one ball."
            ),
            "overall_severity": "LOW",
            "recommendation_level": "MAINTAIN",
            "confidence": chain.get("confidence", 1.0),
            "primary_pillar": primary_pillar,
        }

    def build(self, elbow, risks, interpretation, basics=None, action=None):
        built_elbow = self.build_elbow(elbow)
        chain = self.build_chain(interpretation)
        built_risks = self.build_risks(risks)
        pillars = self.build_pillars(
            chain=chain,
            risks=built_risks,
            elbow=built_elbow,
            action=action,
        )
        scorecard = self.build_scorecard(
            chain=chain,
            risks=built_risks,
            elbow=built_elbow,
            action=action,
        )
        rating_system_v2 = self.build_rating_system_v2(
            chain=chain,
            risks=built_risks,
            elbow=built_elbow,
            action=action,
            scorecard=scorecard,
        )
        summary = self.build_summary(
            chain=chain,
            risks=built_risks,
            elbow=built_elbow,
            action=action,
            scorecard=scorecard,
        )
        summary = self._apply_benchmark_summary_guardrail(
            summary=summary,
            action=action,
            rating_system_v2=rating_system_v2,
        )
        summary["overall_score"] = scorecard["overall"]["score"]
        summary["confidence_score"] = scorecard["confidence"]["score"]
        summary["rating_overall_score"] = rating_system_v2["overall"]["score"]
        comprehensive_why = generate_comprehensive_why(
            built_risks,
            action=action,
            chain=chain,
            pillars=pillars,
            summary=summary,
        )

        return {
            "summary": summary,
            "elbow": built_elbow,
            "kinetic_chain": chain,
            "risks": built_risks,
            "comprehensive_why": comprehensive_why,
            "pillars": pillars,
            "scorecard": scorecard,
            "rating_system_v2": rating_system_v2,
        }
