from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Deviation banding
# ----------------------------
# Band meanings (for UI):
# 1 = better than typical reference
# 2 = within typical reference range
# 3 = high-normal edge
# 4 = outside typical reference range
# 5 = strong outlier
#
# NOTE: We intentionally do NOT call these "ICC limits" for injury risks.
# ICC is reserved for legality (e.g., elbow extension / no-ball laws).


def band_from_percentile(p: Optional[float]) -> Optional[int]:
    """
    Convert percentile (0..100) to a 1..5 deviation band.
    Returns None if p is None.
    """
    if p is None:
        return None
    if p <= 25:
        return 1
    if p <= 75:
        return 2
    if p <= 90:
        return 3
    if p <= 97:
        return 4
    return 5


def percentile_zone_text(band: Optional[int]) -> Optional[str]:
    if band is None:
        return None
    return {
        1: "≤25th",
        2: "25–75th",
        3: "75–90th",
        4: "90–97th",
        5: "≥97th",
    }.get(band)


def interpretation_text(band: Optional[int]) -> Optional[str]:
    if band is None:
        return None
    return {
        1: "Better than typical reference range",
        2: "Within typical reference range",
        3: "High-normal edge",
        4: "Outside typical reference range",
        5: "Strong outlier",
    }.get(band)


def impact_visibility_from_band(band: Optional[int]) -> str:
    """
    Frontend policy:
    - band 1–2: hide or keep very subtle
    - band 3: minimal
    - band 4–5: high
    """
    if band is None:
        return "unknown"
    if band <= 2:
        return "low"
    if band == 3:
        return "medium"
    return "high"


# ----------------------------
# Benchmark definition
# ----------------------------

@dataclass(frozen=True)
class BenchmarkDef:
    risk_id: str
    title: str
    anchor_event: str
    reference_framework: str
    basis: str
    # impact text for the image overlay
    impact_primary: Tuple[str, ...]
    impact_secondary: Tuple[str, ...]
    # optional: what scalar metric is being benchmarked (internal only)
    metric_key: str


# Canonical benchmark wording (injury risks)
_BASIS_COMMON = (
    "Cricket fast-bowling biomechanics literature and established sports biomechanics principles "
    "relating ground reaction forces, loading rates, and injury risk"
)

# Per-risk definitions (ALL 6)
BENCHMARKS: Dict[str, BenchmarkDef] = {
    # 1) Front-Foot Braking Shock
    "front_foot_braking_shock": BenchmarkDef(
        risk_id="front_foot_braking_shock",
        title="Front-Foot Braking Shock",
        anchor_event="FFC",
        reference_framework="Force–time behaviour and rate of force development during front-foot contact",
        basis=_BASIS_COMMON,
        impact_primary=("Front knee", "Ankle"),
        impact_secondary=("Hip", "Lower back"),
        metric_key="ffbs_loading_proxy",
    ),

    # 2) Foot Line Deviation
    "foot_line_deviation": BenchmarkDef(
        risk_id="foot_line_deviation",
        title="Foot Line Deviation",
        anchor_event="FFC",
        reference_framework="Lower-limb alignment and load-path mechanics at front-foot contact",
        basis=_BASIS_COMMON,
        impact_primary=("Front knee", "Groin"),
        impact_secondary=("Hip",),
        metric_key="fld_alignment_proxy",
    ),

    # 3) Knee Brace Failure
    "knee_brace_failure": BenchmarkDef(
        risk_id="knee_brace_failure",
        title="Knee Brace Failure",
        anchor_event="FFC→Release",
        reference_framework="Front-leg stiffness and kinetic chain force transfer near release",
        basis=_BASIS_COMMON,
        impact_primary=("Front knee",),
        impact_secondary=("Hip", "Lower back"),
        metric_key="kbf_brace_proxy",
    ),

    # 4) Lateral Trunk Lean
    "lateral_trunk_lean": BenchmarkDef(
        risk_id="lateral_trunk_lean",
        title="Lateral Trunk Lean",
        anchor_event="Release window",
        reference_framework="Spinal side-flexion control and asymmetric loading around release",
        basis=_BASIS_COMMON,
        impact_primary=("Lower back",),
        impact_secondary=("Opposite hip",),
        metric_key="ltl_lean_proxy",
    ),

    # 5) Hip–Shoulder Mismatch
    "hip_shoulder_mismatch": BenchmarkDef(
        risk_id="hip_shoulder_mismatch",
        title="Hip–Shoulder Mismatch",
        anchor_event="FFC→Release",
        reference_framework="Kinetic chain sequencing and segment timing between hips and shoulders",
        basis=_BASIS_COMMON,
        impact_primary=("Groin", "Lower back"),
        impact_secondary=("Core",),
        metric_key="hsm_timing_proxy",
    ),

    # 6) Trunk Rotation Snap
    "trunk_rotation_snap": BenchmarkDef(
        risk_id="trunk_rotation_snap",
        title="Trunk Rotation Snap",
        anchor_event="Release window",
        reference_framework="Angular acceleration/jerk and torsional load behaviour around release",
        basis=_BASIS_COMMON,
        impact_primary=("Lower back",),
        impact_secondary=("Core", "Hip"),
        metric_key="trs_rotation_proxy",
    ),
}


# ----------------------------
# Benchmark stats storage
# ----------------------------
# You can start with percentile-only benchmarks and replace with real
# distributions later. The important part is the band output.
#
# Expected shape per metric_key:
# {
#   "p25": float,
#   "p50": float,
#   "p75": float,
#   "p90": float,
#   "p97": float
# }
#
# For now, your risk detectors can produce:
# - percentile (0..100) directly, OR
# - a raw metric and we compute percentile via a CDF later (future).


def build_benchmark_block(defn: BenchmarkDef) -> Dict[str, Any]:
    return {
        "type": "biomechanics_reference",
        "label": "Accepted biomechanics reference range",
        "basis": defn.basis,
        "anchor_event": defn.anchor_event,
        "reference_framework": defn.reference_framework,
    }


def attach_deviation_and_impact(
    risk_payload: Dict[str, Any],
    *,
    risk_id: str,
    percentile: Optional[float],
    # You can pass age_group/season/role later to pick different benchmark tables
    age_group: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns a new risk payload with:
      - benchmark block
      - deviation (band + zone + interpretation)
      - impact (primary/secondary + visibility)
    """
    defn = BENCHMARKS.get(risk_id)
    if not defn:
        # Unknown risk — do nothing
        return risk_payload

    band = band_from_percentile(percentile)
    zone = percentile_zone_text(band)
    interp = interpretation_text(band)

    out = dict(risk_payload)

    out["benchmark"] = build_benchmark_block(defn)

    out["deviation"] = {
        "band": band,
        "percentile_zone": zone,            # UI-friendly
        "interpretation": interp,           # UI-friendly
        "percentile": percentile,           # optional; frontend can hide this
        "age_group": age_group,             # optional; useful for future
    }

    out["impact"] = {
        "primary": list(defn.impact_primary),
        "secondary": list(defn.impact_secondary),
        "visibility": impact_visibility_from_band(band),
        # This is the exact text the frontend can write on the image if you want:
        "impact_text_lines": _format_impact_lines(defn, band),
    }

    return out


def _format_impact_lines(defn: BenchmarkDef, band: Optional[int]) -> List[str]:
    """
    Text for writing directly on the risk image.
    Policy:
      - band 1–2: either empty (preferred) or a very quiet line
      - band 3: minimal
      - band 4–5: full primary/secondary
    """
    if band is None:
        return []

    if band <= 2:
        # Keep it quiet to avoid clutter.
        return []

    if band == 3:
        primary = ", ".join(defn.impact_primary[:2])
        return [f"Load focus: {primary}"]

    primary = ", ".join(defn.impact_primary)
    secondary = ", ".join(defn.impact_secondary)
    lines = ["Impact areas:"]
    lines.append(f"Primary – {primary}")
    if secondary:
        lines.append(f"Secondary – {secondary}")
    return lines

