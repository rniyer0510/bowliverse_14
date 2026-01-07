from __future__ import annotations

from typing import Dict, Any

from app.clinician.loader import load_yaml

_GLOBALS = load_yaml("globals.yaml")

def _pick_band(value: float, bands: Dict[str, Dict[str, Any]], default: str) -> str:
    for key, cfg in bands.items():
        vmin = float(cfg.get("min", float("-inf")))
        vmax = float(cfg.get("max", float("inf")))
        if vmin <= value < vmax:
            return key
    return default

def severity_band(signal_strength: float) -> str:
    key = _pick_band(float(signal_strength), _GLOBALS["severity_bands"], "low")
    return key.upper()

def confidence_band(confidence: float) -> str:
    # confidence bands are descending thresholds in globals.yaml (high -> medium -> low)
    conf = float(confidence)
    bands = _GLOBALS["confidence_bands"]
    for key in ("high", "medium", "low"):
        if conf >= float(bands[key]["min"]):
            return key.upper()
    return "LOW"
