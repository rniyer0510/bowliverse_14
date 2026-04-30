from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - renderer falls back to OpenCV text when Pillow is unavailable
    Image = None
    ImageDraw = None
    ImageFont = None

from app.common.logger import get_logger
from ..render_storage import get_render_dir
from ..render_load_watch import (
    _load_hotspot_regions,
    _load_watch_label,
    _preferred_ffc_cue_risk_id,
    _release_hotspot_risk_id,
    _summary_load_watch_title,
    _summary_load_watch_text,
    _summary_symptom_title,
    _summary_symptom_text,
)

logger = get_logger(__name__)

RENDER_DIR = get_render_dir()
DESIGN_BASE_WIDTH = 1080
MIN_VISIBILITY = 0.35
MIN_VISIBILITY_HARD = 0.20
FULL_VISIBILITY = 0.80
SKELETON_COLOR = (255, 240, 88)
SKELETON_SHADOW = (10, 10, 10)
SKELETON_DASHED = (180, 214, 255)
SKELETON_PLACEHOLDER = (118, 126, 140)
JOINT_OUTER = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)
MUTED_TEXT = (236, 240, 246)
PANEL_BG = (10, 12, 16)
PANEL_EDGE = (96, 112, 132)
THEME_PRIMARY = (26, 86, 232)
THEME_SURFACE = (27, 23, 23)
THEME_SURFACE_RAISED = (30, 28, 30)
THEME_STROKE = (49, 42, 42)
THEME_TEXT_PRIMARY = (255, 255, 255)
THEME_TEXT_SECONDARY = (236, 240, 246)
ACTIVE_FILL = (74, 194, 242)
ACTIVE_EDGE = (105, 222, 255)
INACTIVE_FILL = (44, 52, 62)
HOTSPOT_CORE = (0, 140, 255)
HOTSPOT_RING = (0, 122, 255)
HOTSPOT_SOFT = (0, 166, 255)
HOTSPOT_DOT = (0, 0, 170)
FLOW_CARRY = (128, 232, 255)
FLOW_BREAK = (52, 156, 255)
FLOW_GHOST = (108, 114, 126)
FLOW_PAY = (42, 196, 255)
FLOW_PAY_CORE = (40, 118, 255)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
)
TRACKED_JOINTS = tuple(sorted({idx for edge in SKELETON_EDGES for idx in edge}))
PHASES: Tuple[Dict[str, str], ...] = (
    {"key": "approach", "title": "Approach", "short_title": "Approach", "subtitle": "Building into the crease"},
    {"key": "bfc", "title": "Back Foot Contact", "short_title": "Back foot", "subtitle": "Back foot lands"},
    {"key": "ffc", "title": "Front Foot Contact", "short_title": "Front foot", "subtitle": "Front foot lands and supports the action"},
    {"key": "release", "title": "Release", "short_title": "Release", "subtitle": "Ball comes out"},
)
RISK_TITLE_BY_ID: Dict[str, str] = {
    "knee_brace_failure": "Front-Leg Support",
    "lateral_trunk_lean": "Trunk Lean",
    "hip_shoulder_mismatch": "Shoulder Timing",
    "foot_line_deviation": "Foot Line",
    "front_foot_braking_shock": "Landing Load",
    "trunk_rotation_snap": "Trunk Rotation",
}
FEATURE_TO_RENDER_LABEL: Dict[str, str] = {
    "front_leg_support": "Front-Leg Support",
    "trunk_lean": "Trunk Lean",
    "upper_body_opening": "Shoulder Timing",
    "action_flow": "Action Flow",
    "front_foot_line": "Foot Line",
    "trunk_rotation_load": "Trunk Rotation",
}
FFC_DEPENDENT_RISKS = {"knee_brace_failure", "foot_line_deviation", "front_foot_braking_shock"}
WEAK_PHASE_METHODS = {
    "ultimate_fallback",
    "single_foot_fallback",
    "no_foot_data_fallback",
    "degenerate_window",
    "insufficient_landmarks",
}
MIN_FFC_STORY_CONFIDENCE = 0.35
MIN_EVENT_CHAIN_QUALITY = 0.20
POST_RELEASE_SKELETON_TAIL_SECONDS = 0.16
MIN_TRACK_QUALITY = 0.42
MIN_POST_RELEASE_TRACK_QUALITY = 0.58
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
THEME_FONT_DIR_ENV = "ACTIONLAB_THEME_FONT_DIR"
THEME_FRONTEND_REPO_ENV = "ACTIONLAB_FRONTEND_REPO"
DISPLAY_FONT_FILE = "BarlowSemiCondensed-SemiBold.ttf"
LABEL_FONT_FILE = "BarlowSemiCondensed-Medium.ttf"
BODY_FONT_FILE = "Inter-SemiBold.ttf"
BODY_FONT_MEDIUM_FILE = "Inter-Medium.ttf"
LEGEND_DURATION_SECONDS = 2.5
LEGEND_FADE_SECONDS = 0.5


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _safe_point(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    x = _safe_int(value[0])
    y = _safe_int(value[1])
    if x is None or y is None:
        return None
    return (x, y)

__all__ = [name for name in globals() if name != "__builtins__"]
