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
        "question": "Is the bowler staying balanced and upright?",
        "clear": "The bowler stays fairly balanced and upright in this clip.",
        "watch": "The bowler moves off line a little in this clip, so balance is worth watching.",
        "review": "The bowler moves off line enough here to make balance a main review point.",
    },
    "POWER": {
        "display_label": "Ball Carry",
        "question": "Is the run-up helping the ball come out well?",
        "clear": "The action stays fairly connected from run-up into ball release in this clip.",
        "watch": "The run-up and ball release do not look fully connected in this clip.",
        "review": "The run-up is not getting into the ball well enough here, so ball carry becomes a main review point.",
    },
    "PROTECTION": {
        "display_label": "Body Load",
        "question": "Is the body handling the bowling well?",
        "clear": "This clip does not show a strong sign of extra body load.",
        "watch": "Some parts of the action may be taking extra load in this clip.",
        "review": "Some parts of the action look like they are taking more load than we would like, so body load becomes a main review point.",
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

            pillars[pillar_key] = {
                "label": name.title(),
                "display_label": spec.get("display_label", name.title()),
                "question": spec["question"],
                "status": status,
                "score": round(scores[name], 2),
                "summary": spec[tone_key],
                "what_we_saw": what_we_saw,
                "why_flagged": why_flagged,
                "why_it_matters": why_it_matters,
                "what_to_check_with_coach": what_to_check,
            }
        return pillars

    def build_summary(
        self,
        chain: Dict[str, Any],
        risks: List[Dict[str, Any]],
        elbow: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        red_hits = sum(
            1
            for r in risks
            if r.get("severity") in ("HIGH", "VERY_HIGH") and r.get("confidence") == "HIGH"
        )
        pillars = self.build_pillars(
            chain=chain,
            risks=risks,
            elbow=elbow,
            action=action,
        )
        primary_pillar = max(
            ("POSTURE", pillars["posture"]["score"]),
            ("POWER", pillars["power"]["score"]),
            ("PROTECTION", pillars["protection"]["score"]),
            key=lambda item: item[1],
        )[0]
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
        summary = self.build_summary(
            chain=chain,
            risks=built_risks,
            elbow=built_elbow,
            action=action,
        )
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
        }
