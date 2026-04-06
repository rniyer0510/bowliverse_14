"""
ActionLab persistence READ APIs (Phase-I – Auth Locked)

Supports:
- Player listing
- Player creation
- Player profile update
- Deterministic analysis history (run-based)
- Run-id based report fetch

SECURITY:
- All endpoints require authentication
- All reads are scoped to current_account via AccountPlayerLink
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.persistence.session import get_db
from app.persistence.models import (
    Player,
    AccountPlayerLink,
    AnalysisRun,
    AnalysisResultRaw,
)
from app.common.auth import get_current_account

router = APIRouter()


def _extract_scorecard(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result_json, dict):
        return None
    clinician = result_json.get("clinician")
    if not isinstance(clinician, dict):
        return None
    scorecard = clinician.get("scorecard")
    return scorecard if isinstance(scorecard, dict) else None


def _extract_rating_system_v2(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result_json, dict):
        return None
    clinician = result_json.get("clinician")
    if not isinstance(clinician, dict):
        return None
    rating_system = clinician.get("rating_system_v2")
    return rating_system if isinstance(rating_system, dict) else None


def _score_band(score: Optional[int]) -> str:
    if score is None:
        return "unknown"
    if score >= 90:
        return "elite"
    if score >= 75:
        return "strong"
    if score >= 60:
        return "watch"
    if score >= 45:
        return "concern"
    return "review"


def _extract_score_summary(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    scorecard = _extract_scorecard(result_json)
    if not scorecard:
        return None

    overall = scorecard.get("overall") or {}
    pillars = scorecard.get("pillars") or {}
    confidence = scorecard.get("confidence") or {}

    def pillar_score(key: str) -> Optional[int]:
        pillar = pillars.get(key) or {}
        value = pillar.get("score")
        return int(value) if isinstance(value, (int, float)) else None

    overall_score = overall.get("score")
    if not isinstance(overall_score, (int, float)):
        return None

    confidence_score = confidence.get("score")
    return {
        "overall": {
            "score": int(overall_score),
            "band": _score_band(int(overall_score)),
            "label": overall.get("label"),
        },
        "balance": {
            "score": pillar_score("balance"),
            "band": _score_band(pillar_score("balance")),
        },
        "carry": {
            "score": pillar_score("carry"),
            "band": _score_band(pillar_score("carry")),
        },
        "body_load": {
            "score": pillar_score("body_load"),
            "band": _score_band(pillar_score("body_load")),
        },
        "confidence": {
            "score": int(confidence_score) if isinstance(confidence_score, (int, float)) else None,
            "band": _score_band(int(confidence_score)) if isinstance(confidence_score, (int, float)) else "unknown",
        },
    }


def _extract_rating_summary_v2(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rating_system = _extract_rating_system_v2(result_json)
    if not rating_system:
        return None

    overall = rating_system.get("overall") or {}
    player_view = rating_system.get("player_view") or {}
    coach_view = rating_system.get("coach_view") or {}
    confidence = rating_system.get("confidence") or {}
    metrics = player_view.get("metrics") or {}
    coach_metrics = coach_view.get("metrics") or {}

    def metric(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raw = raw or {}
        value = raw.get("score")
        score = int(value) if isinstance(value, (int, float)) else None
        return {
            "score": score,
            "band": _score_band(score),
            "label": raw.get("label"),
        }

    overall_score = overall.get("score")
    if not isinstance(overall_score, (int, float)):
        return None

    return {
        "overall": metric(overall),
        "upper_body_alignment": metric(metrics.get("upper_body_alignment")),
        "lower_body_alignment": metric(metrics.get("lower_body_alignment")),
        "whole_body_alignment": metric(metrics.get("whole_body_alignment")),
        "momentum_forward": metric(metrics.get("momentum_forward")),
        "safety": metric(coach_metrics.get("safety")),
        "confidence": metric(confidence),
    }


def _build_score_heatmap(run_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        ("overall", "Overall"),
        ("balance", "Balance"),
        ("carry", "Carry"),
        ("body_load", "Body Load"),
        ("confidence", "Confidence"),
    ]

    rows = []
    for metric_key, label in metrics:
        cells = []
        for entry in run_entries:
            summary = entry.get("score_summary") or {}
            metric = summary.get(metric_key) or {}
            cells.append(
                {
                    "run_id": entry["run_id"],
                    "created_at": entry["created_at"],
                    "score": metric.get("score"),
                    "band": metric.get("band") or "unknown",
                }
            )
        rows.append(
            {
                "metric": metric_key,
                "label": label,
                "cells": cells,
            }
        )

    return {
        "version": "score_heatmap_v1",
        "headline": "Score trend over recent runs",
        "latest_first": True,
        "rows": rows,
    }


def _build_rating_heatmap_v2(run_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        ("overall", "Overall"),
        ("upper_body_alignment", "Upper Body"),
        ("lower_body_alignment", "Lower Body"),
        ("whole_body_alignment", "Whole Body"),
        ("momentum_forward", "Momentum"),
        ("safety", "Safety"),
        ("confidence", "Confidence"),
    ]

    rows = []
    for metric_key, label in metrics:
        cells = []
        for entry in run_entries:
            summary = entry.get("rating_summary_v2") or {}
            metric = summary.get(metric_key) or {}
            cells.append(
                {
                    "run_id": entry["run_id"],
                    "created_at": entry["created_at"],
                    "score": metric.get("score"),
                    "band": metric.get("band") or "unknown",
                }
            )
        rows.append({"metric": metric_key, "label": label, "cells": cells})

    return {
        "version": "rating_heatmap_v2",
        "headline": "How the action is moving over recent runs",
        "latest_first": True,
        "rows": rows,
    }


# ---------------------------------------------------------------------
# Models (Pydantic v2 compatible)
# ---------------------------------------------------------------------

class PlayerCreateRequest(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=80)
    handedness: str = Field(..., pattern="^(R|L)$")
    age_group: str = Field(..., pattern="^(U10|U14|U16|U19|SENIOR)$")
    season: int = Field(..., ge=2000, le=2100)


class PlayerProfileUpdate(BaseModel):
    age_group: Optional[str] = Field(None, pattern="^(U10|U14|U16|U19|SENIOR)$")
    season: Optional[int] = Field(None, ge=2000, le=2100)


# ---------------------------------------------------------------------
# Players (Auth Scoped)
# ---------------------------------------------------------------------

@router.get("/players")
def list_players(
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    links = (
        db.query(AccountPlayerLink)
        .filter_by(account_id=current_account.account_id)
        .all()
    )

    player_ids = [link.player_id for link in links]
    players_by_id = {}
    if player_ids:
        player_rows = db.query(Player).filter(Player.player_id.in_(player_ids)).all()
        players_by_id = {p.player_id: p for p in player_rows}

    players = []
    for link in links:
        player = players_by_id.get(link.player_id)
        players.append({
            "player_id": str(link.player_id),
            "player_name": link.player_name,
            "link_type": link.link_type,
            "age_group": player.age_group if player else None,
            "season": player.season if player else None,
        })

    return {"players": players}


@router.post("/players", status_code=201)
def create_player(
    payload: PlayerCreateRequest,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    existing = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=current_account.account_id,
            player_name=payload.player_name.strip(),
        )
        .first()
    )

    if existing:
        player = db.query(Player).filter_by(player_id=existing.player_id).first()
        return {
            "player_id": str(existing.player_id),
            "player_name": existing.player_name,
            "age_group": player.age_group,
            "season": player.season,
            "already_exists": True,
        }

    player = Player(
        primary_owner_account_id=current_account.account_id,
        created_by_account_id=current_account.account_id,
        handedness=payload.handedness,
        age_group=payload.age_group,
        season=payload.season,
    )

    db.add(player)
    db.flush()

    db.add(AccountPlayerLink(
        account_id=current_account.account_id,
        player_id=player.player_id,
        link_type="owner",
        player_name=payload.player_name.strip(),
    ))

    db.commit()

    return {
        "player_id": str(player.player_id),
        "player_name": payload.player_name,
        "age_group": player.age_group,
        "season": player.season,
    }


@router.patch("/players/{player_id}/profile")
def update_player_profile(
    player_id: str,
    payload: PlayerProfileUpdate,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    link = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=current_account.account_id,
            player_id=player_id,
        )
        .first()
    )

    if not link:
        raise HTTPException(status_code=403, detail="Access denied")

    player = db.query(Player).filter_by(player_id=player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    if payload.age_group is not None:
        player.age_group = payload.age_group

    if payload.season is not None:
        player.season = payload.season

    db.commit()

    return {
        "player_id": str(player.player_id),
        "age_group": player.age_group,
        "season": player.season,
    }


# ---------------------------------------------------------------------
# Analysis (READ – Auth Scoped)
# ---------------------------------------------------------------------

@router.get("/players/{player_id}/latest")
def latest_analysis(
    player_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """
    Identity-only helper for Home screen.
    NEVER returns report JSON.
    """

    link = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=current_account.account_id,
            player_id=player_id,
        )
        .first()
    )

    if not link:
        raise HTTPException(status_code=403, detail="Access denied")

    run = (
        db.query(AnalysisRun)
        .filter_by(player_id=player_id)
        .order_by(AnalysisRun.created_at.desc())
        .first()
    )

    if not run:
        raise HTTPException(status_code=404, detail="No analysis found")

    return {
        "run_id": str(run.run_id),
        "created_at": run.created_at,
        "season": run.season,
        "age_group": run.age_group,
    }


@router.get("/players/{player_id}/analysis-runs")
def list_analysis_runs(
    player_id: str,
    season: Optional[int] = None,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    link = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=current_account.account_id,
            player_id=player_id,
        )
        .first()
    )

    if not link:
        raise HTTPException(status_code=403, detail="Access denied")

    q = db.query(AnalysisRun).filter_by(player_id=player_id)

    if season is not None:
        q = q.filter_by(season=season)

    runs = q.order_by(AnalysisRun.created_at.desc()).all()
    run_ids = [r.run_id for r in runs]
    raw_by_run_id: Dict[Any, AnalysisResultRaw] = {}
    if run_ids:
        raw_rows = db.query(AnalysisResultRaw).filter(AnalysisResultRaw.run_id.in_(run_ids)).all()
        raw_by_run_id = {row.run_id: row for row in raw_rows}

    items = []
    heatmap_entries = []
    for r in runs:
        raw = raw_by_run_id.get(r.run_id)
        score_summary = _extract_score_summary(raw.result_json if raw else None)
        rating_summary_v2 = _extract_rating_summary_v2(raw.result_json if raw else None)
        item = {
            "run_id": str(r.run_id),
            "created_at": r.created_at,
            "season": r.season,
            "age_group": r.age_group,
            "schema_version": r.schema_version,
            "fps": r.fps,
            "total_frames": r.total_frames,
            "score_summary": score_summary,
            "rating_summary_v2": rating_summary_v2,
        }
        items.append(item)
        heatmap_entries.append(
            {
                "run_id": str(r.run_id),
                "created_at": r.created_at,
                "score_summary": score_summary,
                "rating_summary_v2": rating_summary_v2,
            }
        )

    return {
        "items": items,
        "heatmap": _build_score_heatmap(heatmap_entries),
        "heatmap_v2": _build_rating_heatmap_v2(heatmap_entries),
    }


@router.get("/analysis-runs/{run_id}")
def get_analysis_run(
    run_id: str,
    current_account=Depends(get_current_account),
    db: Session = Depends(get_db),
):
    run = db.query(AnalysisRun).filter_by(run_id=run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    # 🔐 Ownership check
    link = (
        db.query(AccountPlayerLink)
        .filter_by(
            account_id=current_account.account_id,
            player_id=run.player_id,
        )
        .first()
    )

    if not link:
        raise HTTPException(status_code=403, detail="Access denied")

    raw = db.query(AnalysisResultRaw).filter_by(run_id=run_id).first()

    return {
        "run_id": str(run.run_id),
        "player_id": str(run.player_id),
        "season": run.season,
        "age_group": run.age_group,
        "created_at": run.created_at,
        "schema_version": run.schema_version,
        "coach_notes": run.coach_notes,
        "result": raw.result_json if raw else None,
    }
