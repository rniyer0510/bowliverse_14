from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict

_YAML_CACHE: Dict[str, Any] = {}
_YAML_LOCK = threading.Lock()
_KNOWN_YAML_FILES = {
    "globals.yaml",
    "risks.yaml",
    "elbow.yaml",
    "kinetic_chain.yaml",
}


def clear_yaml_cache() -> None:
    with _YAML_LOCK:
        _YAML_CACHE.clear()


def validate_known_yaml_files() -> None:
    for filename in sorted(_KNOWN_YAML_FILES):
        load_yaml(filename)


def _validate_yaml_payload(filename: str, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"Clinician YAML {filename} must load to a mapping")

    if filename == "globals.yaml":
        required = ("confidence_bands", "severity_bands")
        missing = [key for key in required if key not in data or not isinstance(data.get(key), dict)]
        if missing:
            raise ValueError(
                f"Clinician YAML {filename} missing required mapping keys: {', '.join(missing)}"
            )
    elif filename == "risks.yaml":
        if not data:
            raise ValueError(f"Clinician YAML {filename} must define at least one risk")
        for risk_id, risk_cfg in data.items():
            if not isinstance(risk_cfg, dict):
                raise ValueError(f"Clinician YAML {filename} risk {risk_id} must be a mapping")
            explanations = risk_cfg.get("explanations")
            if not isinstance(explanations, dict):
                raise ValueError(
                    f"Clinician YAML {filename} risk {risk_id} missing explanations mapping"
                )
    elif filename == "elbow.yaml":
        if "elbow_extension" not in data or not isinstance(data.get("elbow_extension"), dict):
            raise ValueError(
                "Clinician YAML elbow.yaml must define elbow_extension as a mapping"
            )

    return data


def load_yaml(filename: str) -> Any:
    """
    Loads a YAML file from app/clinician/yaml with simple in-process caching.
    """
    if filename not in _KNOWN_YAML_FILES:
        raise ValueError(f"Unsupported clinician YAML file: {filename}")

    with _YAML_LOCK:
        cached = _YAML_CACHE.get(filename)
    if cached is not None:
        return cached

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required for the clinician layer. "
            "Install it (pip install pyyaml) and restart."
        ) from e

    base = Path(__file__).resolve().parent / "yaml"
    path = base / filename
    if not path.exists():
        raise FileNotFoundError(f"Clinician YAML not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data = _validate_yaml_payload(filename, data)
    with _YAML_LOCK:
        _YAML_CACHE[filename] = data
    return data
