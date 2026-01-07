from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.clinician.loader import load_yaml
from app.clinician.bands import severity_band, confidence_band

ELBOW = load_yaml("elbow.yaml")
RISKS = load_yaml("risks.yaml")
CHAIN = load_yaml("kinetic_chain.yaml")

SEV_RANK = {"VERY_HIGH": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}
CONF_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d


class ClinicianInterpreter:
    """
    ActionLab V14 – FINAL Clinician Interpreter

    Responsibilities:
    - Translate biomech + risk signals into human-safe explanations
    - Resolve contradictions using kinetic-chain context
    - Produce a clinician-style summary for frontend display
    - Output frontend-ready JSON (frontend renders only)
    """

    # ---------- ELBOW ----------
    def build_elbow(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ext = f(raw.get("extension_deg"))
        b = ELBOW["elbow"]["bands"]

        if ext <= b["clear"]["max_deg"]:
            use = b["clear"]
        elif ext <= b["monitor"]["max_deg"]:
            use = b["monitor"]
        elif ext <= b["review"]["max_deg"]:
            use = b["review"]
        else:
            use = b["high_confidence"]

        return {
            "extension_deg": round(ext, 2),
            "band": use["label"],
            "player_text": use["player_text"],
        }

    # ---------- KINETIC CHAIN ----------
    def build_chain(self, interpretation: Dict[str, Any]) -> Dict[str, Any]:
        flow = (interpretation or {}).get("linear_flow") or {}
        contrib = set(flow.get("contributors") or [])

        # Style-consistent protection (e.g., McGrath-type actions)
        if contrib == {"trunk_rotation_snap", "lateral_trunk_lean"}:
            state = "controlled_low_linear"
        else:
            state = (flow.get("flow_state") or "unknown").lower()

        states = CHAIN["kinetic_chain"]["states"]
        if state not in states:
            state = "unknown"

        s = states[state]
        return {
            "state": state.upper(),
            "label": s["label"],
            "kid_summary": s["kid_summary"],
            "coach_summary": s["coach_summary"],
            "confidence": f(flow.get("confidence")),
        }

    # ---------- RISK BUILD ----------
    def build_risks(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for r in raw or []:
            rid = r.get("risk_id")
            if not rid:
                continue

            spec = RISKS["risks"][rid]  # strict contract

            sev = severity_band(f(r.get("signal_strength")))
            conf = confidence_band(f(r.get("confidence")))

            out.append({
                "risk_id": rid,
                "title": spec["title"],
                "body_region": spec["body_region"],
                "severity": sev,
                "confidence": conf,
                "kid_explanation": spec["explanations"]["kid"][sev.lower()],
                "coach_explanation": spec["explanations"]["coach"][sev.lower()],
                "cues": spec["cues"],
                "evidence": {
                    "signal_strength": round(f(r.get("signal_strength")), 3),
                    "confidence": round(f(r.get("confidence")), 3),
                    "note": r.get("note"),
                    "window": r.get("window"),
                },
            })
        return out

    def _risk_sort_key(self, r: Dict[str, Any]) -> Tuple[int, int, float]:
        sev = SEV_RANK.get(r.get("severity", "LOW"), 0)
        conf = CONF_RANK.get(r.get("confidence", "LOW"), 0)
        strength = f((r.get("evidence") or {}).get("signal_strength"), 0.0)
        return (sev, conf, strength)

    def _force_monitor_intent(self, risks: List[Dict[str, Any]]) -> None:
        for r in risks:
            cues = r.get("cues") or {}
            for side in ("kid", "coach"):
                for c in (cues.get(side) or []):
                    c["intent"] = "monitor"

    # ---------- SUMMARY ----------
    def build_summary(
        self,
        chain: Dict[str, Any],
        elbow: Dict[str, Any],
        risks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        max_sev = "LOW"
        for r in risks:
            if SEV_RANK.get(r["severity"], 0) > SEV_RANK.get(max_sev, 0):
                max_sev = r["severity"]

        if chain["state"] == "CONTROLLED_LOW_LINEAR":
            assessment = (
                "Your bowling action is controlled and repeatable with no major injury risks identified."
            )
            recommendation = "MONITOR"
        elif max_sev in ("HIGH", "VERY_HIGH"):
            assessment = (
                "Some parts of your bowling action may place extra load on your body."
            )
            recommendation = "CORRECT"
        else:
            assessment = (
                "Your bowling action is generally sound with a few areas to keep an eye on."
            )
            recommendation = "MONITOR"

        return {
            "overall_assessment": assessment,
            "overall_severity": max_sev,
            "recommendation_level": recommendation,
            "confidence": chain.get("confidence", 1.0),
        }

    # ---------- FINAL ----------
    def build(
        self,
        elbow: Dict[str, Any],
        risks: List[Dict[str, Any]],
        interpretation: Dict[str, Any],
        basics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        chain = self.build_chain(interpretation)
        built_risks = self.build_risks(risks)

        # 🔒 Controlled-flow protection
        if chain["state"] == "CONTROLLED_LOW_LINEAR":
            built_risks = sorted(
                built_risks,
                key=self._risk_sort_key,
                reverse=True
            )[:2]

            self._force_monitor_intent(built_risks)

            for r in built_risks:
                if r["severity"] in ("HIGH", "VERY_HIGH"):
                    r["severity"] = "MODERATE"
                    spec = RISKS["risks"][r["risk_id"]]
                    r["kid_explanation"] = spec["explanations"]["kid"]["moderate"]
                    r["coach_explanation"] = spec["explanations"]["coach"]["moderate"]

        summary = self.build_summary(chain, elbow, built_risks)

        return {
            "summary": summary,                 # 🔹 NEW (top-of-report)
            "elbow": self.build_elbow(elbow),
            "kinetic_chain": chain,
            "risks": built_risks,
        }

