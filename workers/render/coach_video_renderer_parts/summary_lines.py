from __future__ import annotations

from .shared import *
from .analytics import _supports_ffc_story, _risk_weight
from .story_logic import _positive_recap_lines, _story_feature_labels


def _summary_issue_lines(
    risk_by_id: Dict[str, Dict[str, Any]],
    *,
    events: Optional[Dict[str, Any]] = None,
    report_story: Optional[Dict[str, Any]] = None,
) -> List[str]:
    positive_lines = _positive_recap_lines(report_story)
    if positive_lines:
        return positive_lines

    story_labels = _story_feature_labels(report_story)
    if story_labels:
        filtered = [
            label
            for label in story_labels
            if label != "Front-Leg Support" or _supports_ffc_story(events)
        ]
        if filtered:
            return filtered[:4]

    ranked: List[Tuple[float, str]] = []
    allow_ffc_stories = _supports_ffc_story(events)
    for risk_id, risk in (risk_by_id or {}).items():
        if risk_id in FFC_DEPENDENT_RISKS and not allow_ffc_stories:
            continue
        weight = _risk_weight(risk)
        signal = float((risk or {}).get("signal_strength") or 0.0)
        if signal < 0.20:
            continue
        title = RISK_TITLE_BY_ID.get(risk_id)
        if not title:
            continue
        ranked.append((weight, title))
    ranked.sort(reverse=True)

    deduped: List[str] = []
    for _, title in ranked:
        if title not in deduped:
            deduped.append(title)
        if len(deduped) >= 4:
            break
    return deduped
