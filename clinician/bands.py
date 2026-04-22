from __future__ import annotations

from typing import Dict, Any

from app.clinician.knowledge_pack import load_knowledge_pack
from app.clinician.loader import load_yaml


def _globals() -> Dict[str, Any]:
    data = load_yaml("globals.yaml")
    return data if isinstance(data, dict) else {}


def _band_globals() -> Dict[str, Any]:
    try:
        pack = load_knowledge_pack()
        pack_globals = (pack or {}).get("globals") or {}
        if (
            isinstance(pack_globals, dict)
            and isinstance(pack_globals.get("severity_bands"), dict)
            and isinstance(pack_globals.get("confidence_bands"), dict)
        ):
            return pack_globals
    except Exception:
        pass
    return _globals()

def _pick_band(value: float, bands: Dict[str, Dict[str, Any]], default: str) -> str:
    for key, cfg in bands.items():
        vmin = float(cfg.get("min", float("-inf")))
        vmax = float(cfg.get("max", float("inf")))
        if vmin <= value < vmax:
            return key
    return default

def severity_band(signal_strength: float) -> str:
    key = _pick_band(float(signal_strength), _band_globals()["severity_bands"], "low")
    return key.upper()

def confidence_band(confidence: float) -> str:
    # confidence bands are descending thresholds in globals.yaml (high -> medium -> low)
    conf = float(confidence)
    bands = _band_globals()["confidence_bands"]
    for key in ("high", "medium", "low"):
        if conf >= float(bands[key]["min"]):
            return key.upper()
    return "LOW"
