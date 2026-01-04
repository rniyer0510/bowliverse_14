# app/action/action_model.py

from dataclasses import dataclass

@dataclass
class ActionResult:
    intent: str                    # side-on / front-on / hybrid
    action_style: str              # descriptive
    compliance_score: float        # 0–1
    conflict_reason: str | None

