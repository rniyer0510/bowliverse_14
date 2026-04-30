from __future__ import annotations

from typing import Any, Dict, Optional


stable_cache: Dict[str, Dict[str, Any]] = {}


def _normalize_handedness(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    if text in {"R", "RIGHT"}:
        return "R"
    if text in {"L", "LEFT"}:
        return "L"
    return None


def remember_player_profile(
    *,
    player_id: Any,
    handedness: Any = None,
    age_group: Any = None,
    season: Any = None,
) -> Optional[Dict[str, Any]]:
    cache_key = str(player_id or "").strip()
    if not cache_key:
        return None

    profile = dict(stable_cache.get(cache_key) or {})

    normalized_hand = _normalize_handedness(handedness)
    if normalized_hand is not None:
        profile["handedness"] = normalized_hand

    if age_group is not None:
        text = str(age_group or "").strip().upper()
        if text:
            profile["age_group"] = text

    if season is not None:
        try:
            profile["season"] = int(season)
        except Exception:
            pass

    if not profile:
        return None

    stable_cache[cache_key] = profile
    return dict(profile)


def get_player_profile(player_id: Any) -> Optional[Dict[str, Any]]:
    cache_key = str(player_id or "").strip()
    if not cache_key:
        return None
    profile = stable_cache.get(cache_key)
    return dict(profile) if profile else None


def clear_stable_cache() -> None:
    stable_cache.clear()
