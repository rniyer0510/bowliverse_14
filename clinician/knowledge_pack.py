from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


DEFAULT_KNOWLEDGE_PACK_ID = "actionlab_deterministic_expert"
DEFAULT_KNOWLEDGE_PACK_VERSION = "2026-04-22.v1"
KNOWLEDGE_PACK_ENV_VAR = "ACTIONLAB_KNOWLEDGE_PACK_VERSION"

_KNOWLEDGE_PACK_LOCK = threading.Lock()
_KNOWLEDGE_PACK_CACHE: Dict[str, Dict[str, Any]] = {}

_REQUIRED_INDEX_KEYS = (
    "globals",
    "mechanism_families",
    "symptoms",
    "mechanisms",
    "archetypes",
    "trajectories",
    "prescriptions",
    "followup_checks",
    "render_stories",
    "history_bindings",
    "wording",
)


def clear_knowledge_pack_cache() -> None:
    with _KNOWLEDGE_PACK_LOCK:
        _KNOWLEDGE_PACK_CACHE.clear()


def configured_knowledge_pack_version() -> str:
    raw = str(os.getenv(KNOWLEDGE_PACK_ENV_VAR) or DEFAULT_KNOWLEDGE_PACK_VERSION).strip()
    if not raw:
        raise ValueError("ACTIONLAB_KNOWLEDGE_PACK_VERSION cannot be blank")
    return raw


def validate_default_knowledge_pack() -> None:
    load_knowledge_pack()


def load_knowledge_pack(version: Optional[str] = None) -> Dict[str, Any]:
    pack_version = str(version or configured_knowledge_pack_version()).strip()
    if not pack_version:
        raise ValueError("Knowledge pack version cannot be blank")

    with _KNOWLEDGE_PACK_LOCK:
        cached = _KNOWLEDGE_PACK_CACHE.get(pack_version)
        if cached is not None:
            return copy.deepcopy(cached)

    pack_root = _knowledge_pack_root(pack_version)
    manifest = _load_yaml_document(pack_root / "pack.yaml", "pack manifest")
    index = _validate_manifest(pack_version, manifest)

    documents = {
        "manifest": manifest,
        "globals": _load_yaml_document(pack_root / index["globals"], "knowledge pack globals"),
        "mechanism_families": _load_yaml_document(
            pack_root / index["mechanism_families"],
            "knowledge pack mechanism families",
        ),
        "symptoms": _load_yaml_document(pack_root / index["symptoms"], "knowledge pack symptoms"),
        "mechanisms": _load_yaml_document(
            pack_root / index["mechanisms"],
            "knowledge pack mechanisms",
        ),
        "archetypes": _load_yaml_document(
            pack_root / index["archetypes"],
            "knowledge pack archetypes",
        ),
        "trajectories": _load_yaml_document(
            pack_root / index["trajectories"],
            "knowledge pack trajectories",
        ),
        "prescriptions": _load_yaml_document(
            pack_root / index["prescriptions"],
            "knowledge pack prescriptions",
        ),
        "followup_checks": _load_yaml_document(
            pack_root / index["followup_checks"],
            "knowledge pack follow-up checks",
        ),
        "render_stories": _load_yaml_document(
            pack_root / index["render_stories"],
            "knowledge pack render stories",
        ),
        "history_bindings": _load_yaml_document(
            pack_root / index["history_bindings"],
            "knowledge pack history bindings",
        ),
        "wording": _load_yaml_document(
            pack_root / index["wording"],
            "knowledge pack wording",
        ),
    }

    pack = _validate_knowledge_pack_documents(pack_version, documents)
    with _KNOWLEDGE_PACK_LOCK:
        _KNOWLEDGE_PACK_CACHE[pack_version] = copy.deepcopy(pack)
    return copy.deepcopy(pack)


def _knowledge_pack_root(version: str) -> Path:
    root = (
        Path(__file__).resolve().parent
        / "knowledge_packs"
        / DEFAULT_KNOWLEDGE_PACK_ID
        / version
    )
    if not root.exists():
        raise FileNotFoundError(f"Knowledge pack version not found: {root}")
    return root


def _load_yaml_document(path: Path, label: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for ActionLab knowledge packs. Install it and restart."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"{label} must load to a mapping: {path}")
    return payload


def _validate_manifest(requested_version: str, manifest: Mapping[str, Any]) -> Dict[str, str]:
    pack_id = _require_string(manifest, "pack_id", "pack manifest")
    if pack_id != DEFAULT_KNOWLEDGE_PACK_ID:
        raise ValueError(f"Unexpected knowledge pack id: {pack_id}")

    manifest_version = _require_string(manifest, "pack_version", "pack manifest")
    if manifest_version != requested_version:
        raise ValueError(
            f"Knowledge pack manifest version {manifest_version} does not match requested version {requested_version}"
        )

    runtime = _require_mapping(manifest, "runtime", "pack manifest")
    if runtime.get("static_at_runtime") is not True:
        raise ValueError("Knowledge pack runtime.static_at_runtime must be true")
    if runtime.get("deterministic_only") is not True:
        raise ValueError("Knowledge pack runtime.deterministic_only must be true")

    index = _require_mapping(manifest, "index", "pack manifest")
    missing = [key for key in _REQUIRED_INDEX_KEYS if key not in index]
    if missing:
        raise ValueError(
            "Knowledge pack manifest index missing required sections: "
            + ", ".join(sorted(missing))
        )

    normalized: Dict[str, str] = {}
    for key in _REQUIRED_INDEX_KEYS:
        normalized[key] = _require_string(index, key, "pack manifest index")
    return normalized


def _validate_knowledge_pack_documents(
    pack_version: str,
    documents: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    manifest = documents["manifest"]
    globals_doc = documents["globals"]
    mechanism_families_doc = documents["mechanism_families"]
    symptoms_doc = documents["symptoms"]
    mechanisms_doc = documents["mechanisms"]
    archetypes_doc = documents["archetypes"]
    trajectories_doc = documents["trajectories"]
    prescriptions_doc = documents["prescriptions"]
    followup_checks_doc = documents["followup_checks"]
    render_stories_doc = documents["render_stories"]
    history_bindings_doc = documents["history_bindings"]
    wording_doc = documents["wording"]

    globals_cfg = _validate_globals(globals_doc)
    mechanism_families = _validate_mechanism_families(mechanism_families_doc)
    symptoms = _validate_symptoms(symptoms_doc, globals_cfg["phase_order"])
    followup_checks = _validate_followup_checks(followup_checks_doc)
    trajectories = _validate_trajectories(trajectories_doc, followup_checks.keys())
    prescriptions = _validate_prescriptions(prescriptions_doc)
    render_stories = _validate_render_stories(render_stories_doc)
    history_bindings = _validate_history_bindings(
        history_bindings_doc,
        followup_checks.keys(),
    )
    mechanisms = _validate_mechanisms(
        mechanisms_doc,
        phase_order=globals_cfg["phase_order"],
        mechanism_families=mechanism_families.keys(),
        symptom_ids=symptoms.keys(),
        trajectory_ids=trajectories.keys(),
        prescription_ids=prescriptions.keys(),
        render_story_ids=render_stories.keys(),
    )
    archetypes = _validate_archetypes(
        archetypes_doc,
        mechanism_ids=mechanisms.keys(),
    )
    wording = _validate_wording(wording_doc)

    _validate_symptom_mechanism_links(symptoms, mechanisms.keys())
    _validate_render_story_links(render_stories, mechanisms.keys())
    _validate_history_binding_links(
        history_bindings,
        mechanisms.keys(),
        trajectories.keys(),
    )
    _validate_followup_history_binding_links(followup_checks, history_bindings.keys())

    return {
        "manifest": copy.deepcopy(dict(manifest)),
        "pack_id": manifest["pack_id"],
        "pack_version": pack_version,
        "globals": globals_cfg,
        "mechanism_families": mechanism_families,
        "symptoms": symptoms,
        "mechanisms": mechanisms,
        "archetypes": archetypes,
        "trajectories": trajectories,
        "prescriptions": prescriptions,
        "followup_checks": followup_checks,
        "render_stories": render_stories,
        "history_bindings": history_bindings,
        "wording": wording,
    }


def _validate_globals(document: Mapping[str, Any]) -> Dict[str, Any]:
    _require_string(document, "version", "knowledge pack globals")
    phase_order = _require_str_list(document, "phase_order", "knowledge pack globals")
    if len(set(phase_order)) != len(phase_order):
        raise ValueError("knowledge pack globals phase_order must not contain duplicates")

    thresholds = _require_mapping(document, "match_thresholds", "knowledge pack globals")
    confident = _require_float(thresholds, "confident_match_min", "knowledge pack globals")
    partial = _require_float(thresholds, "partial_match_min", "knowledge pack globals")
    weak = _require_float(thresholds, "weak_match_min", "knowledge pack globals")
    ambiguous = _require_float(thresholds, "ambiguous_match_delta_max", "knowledge pack globals")
    if not (0.0 <= weak <= partial <= confident <= 1.0):
        raise ValueError(
            "knowledge pack globals match thresholds must satisfy weak <= partial <= confident within 0..1"
        )
    if not (0.0 <= ambiguous <= 1.0):
        raise ValueError(
            "knowledge pack globals ambiguous_match_delta_max must be between 0 and 1"
        )

    history_defaults = _require_mapping(
        document,
        "history_window_defaults",
        "knowledge pack globals",
    )
    quick_runs = _require_int(history_defaults, "quick_check_runs", "knowledge pack globals")
    reassessment_runs = _require_int(
        history_defaults,
        "reassessment_runs",
        "knowledge pack globals",
    )
    trend_window_runs = _require_int(
        history_defaults,
        "trend_window_runs",
        "knowledge pack globals",
    )
    if not (0 < quick_runs <= reassessment_runs <= trend_window_runs):
        raise ValueError(
            "knowledge pack globals history windows must satisfy quick <= reassessment <= trend"
        )

    evidence_bands = _require_mapping(document, "evidence_bands", "knowledge pack globals")
    for band_name in ("strong", "supporting", "weak"):
        band_cfg = _require_mapping(evidence_bands, band_name, "knowledge pack globals")
        _require_float(band_cfg, "min", f"knowledge pack globals evidence band {band_name}")
    internal_metrics = _require_str_list(document, "internal_metrics", "knowledge pack globals")
    derived_indices = _require_str_list(document, "derived_indices", "knowledge pack globals")

    return {
        "version": document["version"],
        "phase_order": phase_order,
        "match_thresholds": dict(thresholds),
        "history_window_defaults": dict(history_defaults),
        "evidence_bands": dict(evidence_bands),
        "internal_metrics": internal_metrics,
        "derived_indices": derived_indices,
    }


def _validate_mechanism_families(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    families = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="mechanism_families",
        label="knowledge pack mechanism families",
    )
    for family_id, family in families.items():
        _require_matching_id(family_id, family, "knowledge pack mechanism family")
        _require_string(family, "title", f"knowledge pack mechanism family {family_id}")
        _require_string(family, "summary", f"knowledge pack mechanism family {family_id}")
        _require_str_list(family, "focus_phases", f"knowledge pack mechanism family {family_id}")
    return families


def _validate_symptoms(
    document: Mapping[str, Any],
    phase_order: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    allowed_phases = set(phase_order)
    symptoms = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="symptoms",
        label="knowledge pack symptoms",
    )
    for symptom_id, symptom in symptoms.items():
        _require_matching_id(symptom_id, symptom, "knowledge pack symptom")
        phase = _require_string(symptom, "phase", f"knowledge pack symptom {symptom_id}")
        if phase not in allowed_phases:
            raise ValueError(
                f"knowledge pack symptom {symptom_id} uses unknown phase {phase}"
            )
        _require_string(symptom, "title", f"knowledge pack symptom {symptom_id}")
        _require_string(symptom, "category", f"knowledge pack symptom {symptom_id}")
        _require_string(symptom, "description", f"knowledge pack symptom {symptom_id}")
        _require_str_list(symptom, "evidence_inputs", f"knowledge pack symptom {symptom_id}")
        _require_str_list(symptom, "related_symptoms", f"knowledge pack symptom {symptom_id}")
        _require_str_list(
            symptom,
            "possible_mechanisms",
            f"knowledge pack symptom {symptom_id}",
        )
        _require_str_list(
            symptom,
            "render_focus_regions",
            f"knowledge pack symptom {symptom_id}",
        )
    return symptoms


def _validate_mechanisms(
    document: Mapping[str, Any],
    *,
    phase_order: Iterable[str],
    mechanism_families: Iterable[str],
    symptom_ids: Iterable[str],
    trajectory_ids: Iterable[str],
    prescription_ids: Iterable[str],
    render_story_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    allowed_phases = set(phase_order)
    allowed_families = set(mechanism_families)
    allowed_symptoms = set(symptom_ids)
    allowed_trajectories = set(trajectory_ids)
    allowed_prescriptions = set(prescription_ids)
    allowed_render_stories = set(render_story_ids)

    mechanisms = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="mechanisms",
        label="knowledge pack mechanisms",
    )
    for mechanism_id, mechanism in mechanisms.items():
        label = f"knowledge pack mechanism {mechanism_id}"
        _require_matching_id(mechanism_id, mechanism, "knowledge pack mechanism")
        _require_string(mechanism, "title", label)
        _require_string(mechanism, "summary", label)
        family = _require_string(mechanism, "family", label)
        if family not in allowed_families:
            raise ValueError(f"{label} uses unknown family {family}")

        for phase in _require_str_list(mechanism, "primary_phases", label):
            if phase not in allowed_phases:
                raise ValueError(f"{label} uses unknown phase {phase}")

        _validate_known_ids(
            _require_str_list(mechanism, "required_symptoms", label),
            allowed_symptoms,
            f"{label} required_symptoms",
        )
        _validate_known_ids(
            _require_str_list(mechanism, "supporting_symptoms", label),
            allowed_symptoms,
            f"{label} supporting_symptoms",
        )
        _validate_known_ids(
            _require_str_list(mechanism, "contradictory_symptoms", label),
            allowed_symptoms,
            f"{label} contradictory_symptoms",
        )
        _require_str_list(mechanism, "required_evidence", label)
        _require_str_list(mechanism, "supporting_evidence", label)
        _require_str_list(mechanism, "contradictions", label)
        _require_str_list(mechanism, "performance_effects", label)
        _require_str_list(mechanism, "load_effects", label)
        _validate_known_ids(
            _require_str_list(mechanism, "trajectory_ids", label),
            allowed_trajectories,
            f"{label} trajectory_ids",
        )
        _validate_known_ids(
            _require_str_list(mechanism, "prescription_ids", label),
            allowed_prescriptions,
            f"{label} prescription_ids",
        )
        _validate_known_ids(
            _require_str_list(mechanism, "render_story_ids", label),
            allowed_render_stories,
            f"{label} render_story_ids",
        )
        _require_str_list(mechanism, "history_metrics_to_track", label)
    return mechanisms


def _validate_archetypes(
    document: Mapping[str, Any],
    *,
    mechanism_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    allowed_mechanisms = set(mechanism_ids)
    archetypes = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="archetypes",
        label="knowledge pack archetypes",
    )
    for archetype_id, archetype in archetypes.items():
        label = f"knowledge pack archetype {archetype_id}"
        _require_matching_id(archetype_id, archetype, "knowledge pack archetype")
        _require_string(archetype, "title", label)
        _require_string(archetype, "short_label", label)
        _require_string(archetype, "summary", label)
        _validate_known_ids(
            _require_str_list(archetype, "dominant_mechanisms", label),
            allowed_mechanisms,
            f"{label} dominant_mechanisms",
        )
        _validate_known_ids(
            _require_str_list(archetype, "secondary_mechanisms", label),
            allowed_mechanisms,
            f"{label} secondary_mechanisms",
        )
        _require_str_list(archetype, "expected_strengths", label)
        _require_str_list(archetype, "expected_costs", label)
        _require_string(archetype, "history_story_template", label)
        _require_string(archetype, "coaching_priority_template", label)
    return archetypes


def _validate_trajectories(
    document: Mapping[str, Any],
    followup_check_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    allowed_followup_checks = set(followup_check_ids)
    trajectories = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="trajectories",
        label="knowledge pack trajectories",
    )
    for trajectory_id, trajectory in trajectories.items():
        label = f"knowledge pack trajectory {trajectory_id}"
        _require_matching_id(trajectory_id, trajectory, "knowledge pack trajectory")
        _require_string(trajectory, "title", label)
        _require_string(trajectory, "summary", label)
        _require_string(trajectory, "performance_consequence", label)
        _require_string(trajectory, "repeatability_consequence", label)
        _require_string(trajectory, "load_consequence", label)
        _require_str_list(trajectory, "severity_modifiers", label)
        _validate_known_ids(
            _require_str_list(trajectory, "followup_signals", label),
            allowed_followup_checks,
            f"{label} followup_signals",
        )
    return trajectories


def _validate_prescriptions(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    prescriptions = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="prescriptions",
        label="knowledge pack prescriptions",
    )
    for prescription_id, prescription in prescriptions.items():
        label = f"knowledge pack prescription {prescription_id}"
        _require_matching_id(prescription_id, prescription, "knowledge pack prescription")
        _require_string(prescription, "title", label)
        _require_string(prescription, "goal", label)
        _require_string(prescription, "primary_cue", label)
        _require_string(prescription, "why_this_first", label)
        _require_str_list(prescription, "avoid_for_now", label)
        _require_str_list(prescription, "expected_change", label)
        _require_string(prescription, "coach_check", label)
        _require_string(prescription, "reassess_after", label)
        _require_str_list(prescription, "works_best_when", label)
        _require_str_list(prescription, "contraindicated_when", label)
        _require_str_list(prescription, "followup_metric_targets", label)
    return prescriptions


def _validate_followup_checks(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    followup_checks = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="followup_checks",
        label="knowledge pack follow-up checks",
    )
    for check_id, check in followup_checks.items():
        label = f"knowledge pack follow-up check {check_id}"
        _require_matching_id(check_id, check, "knowledge pack follow-up check")
        _require_string(check, "title", label)
        _require_str_list(check, "success_signals", label)
        _require_str_list(check, "failure_signals", label)
        _require_string(check, "recommended_review_window", label)
        _require_string(check, "history_graph_binding", label)
    return followup_checks


def _validate_render_stories(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    render_stories = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="render_stories",
        label="knowledge pack render stories",
    )
    for story_id, story in render_stories.items():
        label = f"knowledge pack render story {story_id}"
        _require_matching_id(story_id, story, "knowledge pack render story")
        _require_string(story, "title", label)
        _require_str_list(story, "phases", label)
        _require_str_list(story, "trigger_mechanism_ids", label)
        _require_str_list(story, "focus_regions", label)
        storyboard = _require_list(story, "storyboard", label)
        if not storyboard:
            raise ValueError(f"{label} storyboard must contain at least one step")
        for idx, step in enumerate(storyboard):
            if not isinstance(step, dict):
                raise ValueError(f"{label} storyboard step {idx} must be a mapping")
            _require_string(step, "phase", f"{label} storyboard step {idx}")
            _require_string(step, "overlay", f"{label} storyboard step {idx}")
            _require_string(step, "cue", f"{label} storyboard step {idx}")
    return render_stories


def _validate_history_bindings(
    document: Mapping[str, Any],
    followup_check_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    allowed_followup_checks = set(followup_check_ids)
    history_bindings = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="history_bindings",
        label="knowledge pack history bindings",
    )
    for binding_id, binding in history_bindings.items():
        label = f"knowledge pack history binding {binding_id}"
        _require_matching_id(binding_id, binding, "knowledge pack history binding")
        _require_string(binding, "title", label)
        _require_str_list(binding, "mechanism_ids", label)
        _require_str_list(binding, "trajectory_ids", label)
        _validate_known_ids(
            _require_str_list(binding, "followup_check_ids", label),
            allowed_followup_checks,
            f"{label} followup_check_ids",
        )
        metrics = _require_str_list(binding, "metrics", label)
        primary_metric = _require_string(binding, "primary_metric", label)
        if primary_metric not in metrics:
            raise ValueError(f"{label} primary_metric must be listed in metrics")
        _require_string(binding, "chart_summary", label)
    return history_bindings


def _validate_wording(document: Mapping[str, Any]) -> Dict[str, Any]:
    _require_string(document, "version", "knowledge pack wording")
    hedges = _require_mapping(document, "hedges", "knowledge pack wording")
    for key in ("primary_mechanism", "performance", "load", "observed"):
        _require_string(hedges, key, "knowledge pack wording hedges")

    unknown_path = _require_mapping(document, "unknown_path", "knowledge pack wording")
    for key in ("primary_mechanism", "performance_impact", "load_impact", "coach_check"):
        _require_string(unknown_path, key, "knowledge pack wording unknown_path")

    surfaces = _require_mapping(document, "surfaces", "knowledge pack wording")
    normalized_surfaces: Dict[str, Dict[str, str]] = {}
    for surface_name in ("player", "coach"):
        surface_cfg = _require_mapping(surfaces, surface_name, "knowledge pack wording surfaces")
        normalized_surfaces[surface_name] = {
            "primary_mechanism_prefix": _require_string(
                surface_cfg,
                "primary_mechanism_prefix",
                f"knowledge pack wording surface {surface_name}",
            ),
            "performance_prefix": _require_string(
                surface_cfg,
                "performance_prefix",
                f"knowledge pack wording surface {surface_name}",
            ),
            "load_prefix": _require_string(
                surface_cfg,
                "load_prefix",
                f"knowledge pack wording surface {surface_name}",
            ),
        }

    unknown_path_surfaces = _require_mapping(
        document,
        "unknown_path_surfaces",
        "knowledge pack wording",
    )
    normalized_unknown_surfaces: Dict[str, Dict[str, str]] = {}
    for surface_name in ("player", "coach"):
        surface_cfg = _require_mapping(
            unknown_path_surfaces,
            surface_name,
            "knowledge pack wording unknown_path_surfaces",
        )
        normalized_unknown_surfaces[surface_name] = {
            "primary_mechanism": _require_string(
                surface_cfg,
                "primary_mechanism",
                f"knowledge pack wording unknown surface {surface_name}",
            ),
            "performance_impact": _require_string(
                surface_cfg,
                "performance_impact",
                f"knowledge pack wording unknown surface {surface_name}",
            ),
            "load_impact": _require_string(
                surface_cfg,
                "load_impact",
                f"knowledge pack wording unknown surface {surface_name}",
            ),
            "coach_check": _require_string(
                surface_cfg,
                "coach_check",
                f"knowledge pack wording unknown surface {surface_name}",
            ),
        }

    status_leads = _require_mapping(document, "status_leads", "knowledge pack wording")
    normalized_status_leads: Dict[str, Dict[str, str]] = {}
    for status_name in (
        "confident_match",
        "partial_match",
        "weak_match",
        "ambiguous_match",
        "no_match",
    ):
        status_cfg = _require_mapping(status_leads, status_name, "knowledge pack wording status_leads")
        normalized_status_leads[status_name] = {
            "player": _require_string(
                status_cfg,
                "player",
                f"knowledge pack wording status lead {status_name}",
            ),
            "coach": _require_string(
                status_cfg,
                "coach",
                f"knowledge pack wording status lead {status_name}",
            ),
        }

    product_rules = _require_str_list(document, "product_rules", "knowledge pack wording")
    return {
        "version": document["version"],
        "hedges": dict(hedges),
        "unknown_path": dict(unknown_path),
        "surfaces": normalized_surfaces,
        "unknown_path_surfaces": normalized_unknown_surfaces,
        "status_leads": normalized_status_leads,
        "product_rules": product_rules,
    }


def _validate_symptom_mechanism_links(
    symptoms: Mapping[str, Mapping[str, Any]],
    mechanism_ids: Iterable[str],
) -> None:
    allowed_mechanisms = set(mechanism_ids)
    for symptom_id, symptom in symptoms.items():
        _validate_known_ids(
            symptom["possible_mechanisms"],
            allowed_mechanisms,
            f"knowledge pack symptom {symptom_id} possible_mechanisms",
        )


def _validate_render_story_links(
    render_stories: Mapping[str, Mapping[str, Any]],
    mechanism_ids: Iterable[str],
) -> None:
    allowed_mechanisms = set(mechanism_ids)
    for story_id, story in render_stories.items():
        _validate_known_ids(
            story["trigger_mechanism_ids"],
            allowed_mechanisms,
            f"knowledge pack render story {story_id} trigger_mechanism_ids",
        )


def _validate_history_binding_links(
    history_bindings: Mapping[str, Mapping[str, Any]],
    mechanism_ids: Iterable[str],
    trajectory_ids: Iterable[str],
) -> None:
    allowed_mechanisms = set(mechanism_ids)
    allowed_trajectories = set(trajectory_ids)
    for binding_id, binding in history_bindings.items():
        _validate_known_ids(
            binding["mechanism_ids"],
            allowed_mechanisms,
            f"knowledge pack history binding {binding_id} mechanism_ids",
        )
        _validate_known_ids(
            binding["trajectory_ids"],
            allowed_trajectories,
            f"knowledge pack history binding {binding_id} trajectory_ids",
        )


def _validate_followup_history_binding_links(
    followup_checks: Mapping[str, Mapping[str, Any]],
    history_binding_ids: Iterable[str],
) -> None:
    allowed_bindings = set(history_binding_ids)
    for check_id, check in followup_checks.items():
        binding_id = check["history_graph_binding"]
        if binding_id not in allowed_bindings:
            raise ValueError(
                f"knowledge pack follow-up check {check_id} uses unknown history graph binding {binding_id}"
            )


def _require_entity_mapping(
    document: Mapping[str, Any],
    *,
    version_key: str,
    collection_key: str,
    label: str,
) -> Dict[str, Dict[str, Any]]:
    _require_string(document, version_key, label)
    collection = _require_mapping(document, collection_key, label)
    normalized: Dict[str, Dict[str, Any]] = {}
    for entity_id, entity in collection.items():
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError(f"{label} contains a blank entity id")
        if not isinstance(entity, dict):
            raise ValueError(f"{label} entry {entity_id} must be a mapping")
        normalized[entity_id] = dict(entity)
    if not normalized:
        raise ValueError(f"{label} must define at least one entry")
    return normalized


def _require_matching_id(entity_id: str, entity: Mapping[str, Any], label: str) -> None:
    declared = _require_string(entity, "id", label)
    if declared != entity_id:
        raise ValueError(f"{label} id mismatch: key {entity_id} != payload {declared}")


def _require_mapping(payload: Mapping[str, Any], key: str, label: str) -> Dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must define {key} as a mapping")
    return dict(value)


def _require_list(payload: Mapping[str, Any], key: str, label: str) -> List[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{label} must define {key} as a list")
    return list(value)


def _require_str_list(payload: Mapping[str, Any], key: str, label: str) -> List[str]:
    values = _require_list(payload, key, label)
    normalized: List[str] = []
    for idx, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} {key}[{idx}] must be a non-empty string")
        normalized.append(value)
    return normalized


def _require_string(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must define {key} as a non-empty string")
    return value


def _require_float(payload: Mapping[str, Any], key: str, label: str) -> float:
    value = payload.get(key)
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{label} must define {key} as a number") from exc


def _require_int(payload: Mapping[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{label} must define {key} as an integer")
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{label} must define {key} as an integer") from exc


def _validate_known_ids(values: Iterable[str], allowed: Iterable[str], label: str) -> None:
    allowed_set = set(allowed)
    unknown = sorted({value for value in values if value not in allowed_set})
    if unknown:
        raise ValueError(f"{label} contains unknown ids: {', '.join(unknown)}")
