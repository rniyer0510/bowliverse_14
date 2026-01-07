from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

_YAML_CACHE: Dict[str, Any] = {}

def load_yaml(filename: str) -> Any:
    """
    Loads a YAML file from app/clinician/yaml with simple in-process caching.
    """
    if filename in _YAML_CACHE:
        return _YAML_CACHE[filename]

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

    _YAML_CACHE[filename] = data
    return data
