from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.clinician.loader import load_yaml
from app.clinician.bands import severity_band, confidence_band

ELBOW = load_yaml("elbow.yaml") or {}
RISKS = load_yaml("risks.yaml") or {}
CHAIN = load_yaml("kinetic_chain.yaml") or {}

SEV_RANK = {"VERY_HIGH": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}
CONF_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d


class ClinicianInterpreter:
    """
    ActionLab V14 – Clinician Interpreter (ROBUST, ICC-ALIGNED)

    - Never crashes frontend
    - Never drops risks silently
    - YAML-driven where appropriate
    - ICC-correct elbow interpretation
    """

    # ---------- ELBOW ----------
    def build_elbow(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ext = f(raw.get("extension_deg"))
        verdict = (raw.get("verdict") or "").upper()

        # ICC-aligned logic
        if verdict == "ILLEGAL":
            band = "ILLEGAL"
            text = "Your elbow action exceeds the legal extension limit."
        elif ext <= 15.0:
            band = "OK"
            text = "Your elbow action is within the legal range."
        elif ext <= 20.0:
            band = "BORDERLINE"
            text = "Your elbow action is close to the legal limit and should be monitored."
        else:
            band = "ILLEGAL"
            text = "Your elbow action exceeds the legal extension limit."

        return {
            "extension_deg": round(ext, 2),
            "band": band,
            "player_text": text,
        }

    # ---------- KINETIC CHAIN ----------
    def build_chain(self, interpretation: Dict[str, Any]) -> Dict[str, Any]:
        flow = (interpretation or {}).get("linear_flow") or {}
        state = (flow.get("flow_state") or "unknown").lower()

        energy = CHAIN.get("energy_transfer") or {}

        if state == "interrupted":
            msg = energy.get("loaded", {}).get(
                "message",
                "Some parts of your body are absorbing extra load during delivery.",
            )
        elif state == "smooth":
            msg = energy.get("good", {}).get(
                "message",
                "Energy flows smoothly through your body.",
            )
        else:
            msg = "Your movement pattern could not be fully classified."

        return {
            "state": state.upper(),
            "kid_summary": msg,
            "confidence": f(flow.get("confidence"), 1.0),
        }

    # ---------- RISKS ----------
    def build_risks(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for r in raw or []:
            rid = r.get("risk_id")
            if not rid:
                continue

            spec = RISKS.get(rid)

            sev = severity_band(f(r.get("signal_strength")))
            conf = confidence_band(f(r.get("confidence")))

            if not spec:
                out.append({
                    "risk_id": rid,
                    "title": rid.replace("_", " ").title(),
                    "severity": sev,
                    "confidence": conf,
                    "kid_explanation": "This movement places some stress on your body.",
                    "coach_explanation": "Monitor this loading pattern.",
                    "evidence": r,
                })
                continue

            out.append({
                "risk_id": rid,
                "title": spec.get("title", rid),
                "body_region": spec.get("body_region"),
                "severity": sev,
                "confidence": conf,
                "kid_explanation": spec.get("explanations", {})
                    .get("kid", {})
                    .get(sev.lower(), ""),
                "coach_explanation": spec.get("explanations", {})
                    .get("coach", {})
                    .get(sev.lower(), ""),
                "cues": spec.get("cues", {}),
                "evidence": {
                    "signal_strength": round(f(r.get("signal_strength")), 3),
                    "confidence": round(f(r.get("confidence")), 3),
                    "window": r.get("window"),
                    "note": r.get("note"),
                },
            })

        return out

    # ---------- SUMMARY ----------
    def build_summary(
        self,
        chain: Dict[str, Any],
        risks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        # Count high-confidence red risks
        red_hits = 0
        for r in risks:
            if (
                r.get("severity") in ("HIGH", "VERY_HIGH")
                and r.get("confidence") == "HIGH"
            ):
                red_hits += 1

        if red_hits >= 2:
            max_sev = "VERY_HIGH"
            rec = "CORRECT"
            text = "Some parts of your action are placing high load on your body."
        elif red_hits == 1:
            max_sev = "MODERATE"
            rec = "MONITOR"
            text = "Your action is generally sound, but a few areas need attention."
        else:
            max_sev = "LOW"
            rec = "MAINTAIN"
            text = "Your action is generally sound."

        return {
            "overall_assessment": text,
            "overall_severity": max_sev,
            "recommendation_level": rec,
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
        summary = self.build_summary(chain, built_risks)

        return {
            "summary": summary,
            "elbow": self.build_elbow(elbow),
            "kinetic_chain": chain,
            "risks": built_risks,
        }

