from __future__ import annotations

import os
from typing import Dict, Tuple

DEFAULT_RENDER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "storage", "renders")
)
RENDER_DIR = os.getenv("ACTIONLAB_RENDER_DIR", DEFAULT_RENDER_DIR)
try:
    os.makedirs(RENDER_DIR, exist_ok=True)
except OSError:
    RENDER_DIR = "/tmp/actionlab_renders"
    os.makedirs(RENDER_DIR, exist_ok=True)

MIN_VISIBILITY = 0.35
RUNNER_MIN_VISIBILITY = 0.45
RUNNER_MIN_CONSECUTIVE_FRAMES = 4
RUNNER_MIN_MOTION_PX = 6.0

SKELETON_COLOR = (240, 225, 70)
SKELETON_SHADOW = (32, 28, 18)
JOINT_OUTER = (255, 255, 255)
TEXT_COLOR = (246, 248, 252)
MUTED_TEXT = (204, 212, 222)
PANEL_BG = (24, 28, 34)
PANEL_EDGE = (62, 73, 88)
ACTIVE_FILL = (74, 194, 242)
ACTIVE_EDGE = (105, 222, 255)
INACTIVE_FILL = (44, 52, 62)

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
    {
        "key": "approach",
        "title": "Approach",
        "subtitle": "Building into the crease",
    },
    {
        "key": "bfc",
        "title": "Back Foot Contact",
        "subtitle": "Back foot lands",
    },
    {
        "key": "ffc",
        "title": "Front Foot Contact",
        "subtitle": "Front foot lands and supports the action",
    },
    {
        "key": "release",
        "title": "Release",
        "subtitle": "Ball comes out",
    },
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

FFC_DEPENDENT_RISKS = {
    "knee_brace_failure",
    "foot_line_deviation",
    "front_foot_braking_shock",
}

MIN_FFC_STORY_CONFIDENCE = 0.35
MIN_EVENT_CHAIN_QUALITY = 0.20


