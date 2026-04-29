from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_candidate(
    *,
    frame: Optional[int],
    method: str,
    confidence: float,
    score: Optional[float] = None,
    reason: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if frame is None:
        return None
    item: Dict[str, Any] = {
        "frame": int(frame),
        "method": str(method),
        "confidence": round(float(max(0.0, min(0.99, confidence))), 2),
    }
    if score is not None:
        item["score"] = round(float(score), 4)
    if reason:
        item["reason"] = str(reason)
    return item


def compact_candidates(
    candidates: List[Optional[Dict[str, Any]]],
    *,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    filtered = [c for c in candidates if c is not None]
    filtered.sort(
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )
    return filtered[:limit]


def chain_quality(
    *,
    bfc_frame: Optional[int],
    ffc_frame: Optional[int],
    uah_frame: Optional[int],
    release_frame: Optional[int],
    bfc_confidence: float = 0.0,
    ffc_confidence: float = 0.0,
    uah_confidence: float = 0.0,
    release_confidence: float = 0.0,
) -> Dict[str, Any]:
    ordered = True
    spacing_penalty = 0.0

    if (
        bfc_frame is not None
        and ffc_frame is not None
        and not (bfc_frame <= ffc_frame)
    ):
        ordered = False
        spacing_penalty += 0.35

    if (
        ffc_frame is not None
        and uah_frame is not None
        and not (ffc_frame <= uah_frame)
    ):
        ordered = False
        spacing_penalty += 0.35

    if (
        uah_frame is not None
        and release_frame is not None
        and not (uah_frame <= release_frame)
    ):
        ordered = False
        spacing_penalty += 0.35

    average_confidence = (
        float(bfc_confidence)
        + float(ffc_confidence)
        + float(uah_confidence)
        + float(release_confidence)
    ) / 4.0

    quality = max(0.0, min(0.99, average_confidence - spacing_penalty))

    return {
        "ordered": ordered,
        "quality": round(quality, 2),
    }


def annotate_detection_contract(
    events: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = dict(events or {})
    for key in ("bfc", "ffc", "uah", "release"):
        event = payload.get(key)
        if not isinstance(event, dict):
            continue
        item = dict(event)
        frame = item.get("frame")
        confidence = item.get("confidence")
        if frame is not None:
            item["detected_frame"] = int(frame)
        if confidence is not None:
            item["detection_confidence"] = round(
                float(max(0.0, min(0.99, confidence))),
                2,
            )
        payload[key] = item

    event_chain = payload.get("event_chain")
    if isinstance(event_chain, dict) and "quality" in event_chain:
        event_chain_payload = dict(event_chain)
        event_chain_payload["detection_quality"] = round(
            float(max(0.0, min(0.99, event_chain.get("quality") or 0.0))),
            2,
        )
        payload["event_chain"] = event_chain_payload

    return payload
