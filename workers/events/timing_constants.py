from __future__ import annotations

from typing import Dict


DEFAULT_FPS = 30.0


def _fps_value(fps: float) -> float:
    return max(1.0, float(fps or DEFAULT_FPS))


def frames_for_ms(
    fps: float,
    milliseconds: float,
    *,
    min_frames: int = 1,
) -> int:
    frame_count = int(round(_fps_value(fps) * (float(milliseconds) / 1000.0)))
    return max(int(min_frames), frame_count)


def sigma_for_ms(
    fps: float,
    milliseconds: float,
    *,
    min_sigma: float = 1.0,
) -> float:
    sigma = _fps_value(fps) * (float(milliseconds) / 1000.0)
    return max(float(min_sigma), sigma)


def signal_cache_timing(fps: float) -> Dict[str, float]:
    return {
        "fps": _fps_value(fps),
        "dt": 1.0 / _fps_value(fps),
        "smooth_sigma": sigma_for_ms(fps, 25.0),
        "max_interp_gap": frames_for_ms(fps, 100.0, min_frames=2),
    }


def release_uah_timing(fps: float, n_frames: int) -> Dict[str, float]:
    fps_value = _fps_value(fps)
    return {
        "fps": fps_value,
        "dt": 1.0 / fps_value,
        "smooth_sigma": sigma_for_ms(fps_value, 25.0),
        "vote_sigma": sigma_for_ms(fps_value, 20.0),
        "search_start": max(2, int(round(max(0, n_frames) * 0.30))),
        "search_end_margin": max(2, int(round(max(0, n_frames) * 0.05))),
        "delivery_slack": frames_for_ms(fps_value, 200.0, min_frames=2),
        "max_interp_gap": frames_for_ms(fps_value, 100.0, min_frames=2),
        "uah_primary_max_before_release": frames_for_ms(fps_value, 240.0, min_frames=4),
        "uah_max_before_release": frames_for_ms(fps_value, 400.0, min_frames=4),
        "uah_min_before_release": frames_for_ms(fps_value, 30.0, min_frames=1),
        "wrist_cross_slack": frames_for_ms(fps_value, 40.0, min_frames=1),
        "post_release_rel": 0.10,
    }


def foot_contact_timing(fps: float) -> Dict[str, int | float]:
    fps_value = _fps_value(fps)
    return {
        "fps": fps_value,
        "dt": 1.0 / fps_value,
        "lookback": frames_for_ms(fps_value, 900.0, min_frames=3),
        "hold": frames_for_ms(fps_value, 50.0, min_frames=3),
        "smooth_k": max(3, frames_for_ms(fps_value, 20.0, min_frames=3)),
        "back_recent": frames_for_ms(fps_value, 30.0, min_frames=2),
        "min_ffc_release_band": frames_for_ms(fps_value, 160.0, min_frames=4),
        "pre_pelvis_lead": frames_for_ms(fps_value, 450.0, min_frames=4),
        "recent_support_band": frames_for_ms(fps_value, 220.0, min_frames=3),
        "edge_override_gap": frames_for_ms(fps_value, 160.0, min_frames=3),
    }


def render_timing(fps: float) -> Dict[str, int | float]:
    fps_value = _fps_value(fps)
    return {
        "fps": fps_value,
        "render_tolerance": frames_for_ms(fps_value, 80.0, min_frames=3),
    }
