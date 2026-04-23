from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


DEFAULT_KNOWLEDGE_PACK_ID = "actionlab_deterministic_expert"
DEFAULT_KNOWLEDGE_PACK_VERSION = "2026-04-22.v1"
KNOWLEDGE_PACK_ENV_VAR = "ACTIONLAB_KNOWLEDGE_PACK_VERSION"
ACTIONLAB_EXPERT_ENGINE_VERSION = "1.0.0"

_KNOWLEDGE_PACK_LOCK = threading.Lock()
_KNOWLEDGE_PACK_CACHE: Dict[str, Dict[str, Any]] = {}

_REQUIRED_INDEX_KEYS = (
    "globals",
    "mechanism_families",
    "symptoms",
    "mechanisms",
    "contributors",
    "archetypes",
    "trajectories",
    "prescriptions",
    "followup_checks",
    "render_stories",
    "history_bindings",
    "coach_judgments",
    "capture_templates",
    "architecture_principles",
    "research_sources",
    "knowledge_evidence",
    "reconciliation",
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
        "contributors": _load_yaml_document(
            pack_root / index["contributors"],
            "knowledge pack contributors",
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
        "coach_judgments": _load_yaml_document(
            pack_root / index["coach_judgments"],
            "knowledge pack coach judgments",
        ),
        "capture_templates": _load_yaml_document(
            pack_root / index["capture_templates"],
            "knowledge pack capture templates",
        ),
        "architecture_principles": _load_yaml_document(
            pack_root / index["architecture_principles"],
            "knowledge pack architecture principles",
        ),
        "research_sources": _load_yaml_document(
            pack_root / index["research_sources"],
            "knowledge pack research sources",
        ),
        "knowledge_evidence": _load_yaml_document(
            pack_root / index["knowledge_evidence"],
            "knowledge pack evidence catalog",
        ),
        "reconciliation": _load_yaml_document(
            pack_root / index["reconciliation"],
            "knowledge pack reconciliation rules",
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
    _require_string(manifest, "schema_version", "pack manifest")
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
    release_date = str(manifest.get("release_date") or "").strip()
    if not release_date:
        raise ValueError("pack manifest must define release_date")
    min_engine_version = _require_string(manifest, "min_engine_version", "pack manifest")
    if _compare_semver(min_engine_version, ACTIONLAB_EXPERT_ENGINE_VERSION) > 0:
        raise ValueError(
            "Knowledge pack requires engine version "
            f"{min_engine_version}, but this runtime is {ACTIONLAB_EXPERT_ENGINE_VERSION}"
        )
    if not isinstance(manifest.get("breaking_changes"), bool):
        raise ValueError("pack manifest breaking_changes must be a boolean")
    supersedes = manifest.get("supersedes")
    if supersedes is not None and not isinstance(supersedes, str):
        raise ValueError("pack manifest supersedes must be a string or null")
    _require_string(manifest, "changelog_ref", "pack manifest")

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
    normalized["min_engine_version"] = min_engine_version
    normalized["changelog_ref"] = _require_string(manifest, "changelog_ref", "pack manifest")
    return normalized


def _compare_semver(left: str, right: str) -> int:
    left_parts = _parse_semver(left)
    right_parts = _parse_semver(right)
    if left_parts < right_parts:
        return -1
    if left_parts > right_parts:
        return 1
    return 0


def _parse_semver(value: str) -> tuple[int, int, int]:
    text = str(value or "").strip()
    pieces = text.split(".")
    if len(pieces) != 3:
        raise ValueError(f"Invalid semantic version: {text}")
    try:
        return int(pieces[0]), int(pieces[1]), int(pieces[2])
    except Exception as exc:
        raise ValueError(f"Invalid semantic version: {text}") from exc


def _validate_knowledge_pack_documents(
    pack_version: str,
    documents: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    manifest = documents["manifest"]
    globals_doc = documents["globals"]
    mechanism_families_doc = documents["mechanism_families"]
    symptoms_doc = documents["symptoms"]
    mechanisms_doc = documents["mechanisms"]
    contributors_doc = documents["contributors"]
    archetypes_doc = documents["archetypes"]
    trajectories_doc = documents["trajectories"]
    prescriptions_doc = documents["prescriptions"]
    followup_checks_doc = documents["followup_checks"]
    render_stories_doc = documents["render_stories"]
    history_bindings_doc = documents["history_bindings"]
    coach_judgments_doc = documents["coach_judgments"]
    capture_templates_doc = documents["capture_templates"]
    architecture_principles_doc = documents["architecture_principles"]
    research_sources_doc = documents["research_sources"]
    knowledge_evidence_doc = documents["knowledge_evidence"]
    reconciliation_doc = documents["reconciliation"]
    wording_doc = documents["wording"]

    globals_cfg = _validate_globals(globals_doc)
    mechanism_families = _validate_mechanism_families(mechanism_families_doc)
    symptoms = _validate_symptoms(symptoms_doc, globals_cfg["phase_order"])
    contributors = _validate_contributors(
        contributors_doc,
        phase_order=globals_cfg["phase_order"],
    )
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
    coach_judgments = _validate_coach_judgments(
        coach_judgments_doc,
        phase_order=globals_cfg["phase_order"],
    )
    capture_templates = _validate_capture_templates(capture_templates_doc)
    architecture_principles = _validate_architecture_principles(architecture_principles_doc)
    research_sources = _validate_research_sources(research_sources_doc)
    knowledge_evidence = _validate_knowledge_evidence(knowledge_evidence_doc)
    reconciliation = _validate_reconciliation(reconciliation_doc)
    wording = _validate_wording(wording_doc)

    _validate_symptom_mechanism_links(symptoms, mechanisms.keys())
    _validate_contributor_links(
        contributors,
        symptom_ids=symptoms.keys(),
        mechanism_ids=mechanisms.keys(),
    )
    _validate_render_story_links(render_stories, mechanisms.keys())
    _validate_history_binding_links(
        history_bindings,
        mechanisms.keys(),
        trajectories.keys(),
    )
    _validate_followup_history_binding_links(followup_checks, history_bindings.keys())
    _validate_capture_template_links(
        capture_templates,
        coach_judgments=coach_judgments,
        symptom_ids=symptoms.keys(),
        mechanism_ids=mechanisms.keys(),
        contributor_ids=contributors.keys(),
        prescription_ids=prescriptions.keys(),
    )
    _validate_architecture_principle_links(
        architecture_principles,
        source_ids=research_sources.keys(),
        mechanism_ids=mechanisms.keys(),
        contributor_ids=contributors.keys(),
        prescription_ids=prescriptions.keys(),
    )
    _validate_knowledge_evidence_links(
        knowledge_evidence,
        source_ids=research_sources.keys(),
        symptom_ids=symptoms.keys(),
        mechanism_ids=mechanisms.keys(),
        contributor_ids=contributors.keys(),
        prescription_ids=prescriptions.keys(),
        trajectory_ids=trajectories.keys(),
    )
    _validate_reconciliation_links(
        reconciliation,
        symptom_ids=symptoms.keys(),
        mechanism_ids=mechanisms.keys(),
        contributor_ids=contributors.keys(),
        prescription_ids=prescriptions.keys(),
        evidence_ids=knowledge_evidence.keys(),
    )

    return {
        "manifest": copy.deepcopy(dict(manifest)),
        "pack_id": manifest["pack_id"],
        "pack_version": pack_version,
        "globals": globals_cfg,
        "mechanism_families": mechanism_families,
        "symptoms": symptoms,
        "mechanisms": mechanisms,
        "contributors": contributors,
        "archetypes": archetypes,
        "trajectories": trajectories,
        "prescriptions": prescriptions,
        "followup_checks": followup_checks,
        "render_stories": render_stories,
        "history_bindings": history_bindings,
        "coach_judgments": coach_judgments,
        "capture_templates": capture_templates,
        "architecture_principles": architecture_principles,
        "research_sources": research_sources,
        "knowledge_evidence": knowledge_evidence,
        "reconciliation": reconciliation,
        "wording": wording,
    }


def _validate_globals(document: Mapping[str, Any]) -> Dict[str, Any]:
    _require_string(document, "version", "knowledge pack globals")
    phase_order = _require_str_list(document, "phase_order", "knowledge pack globals")
    if len(set(phase_order)) != len(phase_order):
        raise ValueError("knowledge pack globals phase_order must not contain duplicates")
    confidence_bands = _require_mapping(document, "confidence_bands", "knowledge pack globals")
    for key in ("high", "medium", "low"):
        band_cfg = _require_mapping(confidence_bands, key, "knowledge pack globals")
        _require_float(band_cfg, "min", f"knowledge pack globals confidence band {key}")
    if not (
        float(confidence_bands["low"]["min"])
        <= float(confidence_bands["medium"]["min"])
        <= float(confidence_bands["high"]["min"])
        <= 1.0
    ):
        raise ValueError(
            "knowledge pack globals confidence bands must satisfy low <= medium <= high within 0..1"
        )
    severity_bands = _require_mapping(document, "severity_bands", "knowledge pack globals")
    for key in ("low", "moderate", "high", "very_high"):
        band_cfg = _require_mapping(severity_bands, key, "knowledge pack globals")
        _require_float(band_cfg, "min", f"knowledge pack globals severity band {key}")
        _require_float(band_cfg, "max", f"knowledge pack globals severity band {key}")

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
    presentation_rules = _require_mapping(
        document,
        "presentation_downgrade_rules",
        "knowledge pack globals",
    )
    allowed_statuses = {
        "confident_match",
        "partial_match",
        "weak_match",
        "no_match",
        "ambiguous_match",
    }
    for key in (
        "full_causal_story_requires",
        "partial_evidence_requires",
        "event_only_below",
        "prescription_suppressed_below",
    ):
        value = _require_string(presentation_rules, key, "knowledge pack globals")
        if value not in allowed_statuses:
            raise ValueError(
                f"knowledge pack globals presentation rule {key} must be one of {sorted(allowed_statuses)}"
            )
    full_min_evidence = _require_float(
        presentation_rules,
        "full_causal_story_min_evidence_completeness",
        "knowledge pack globals",
    )
    if not (0.0 <= full_min_evidence <= 1.0):
        raise ValueError(
            "knowledge pack globals full_causal_story_min_evidence_completeness must be between 0 and 1"
        )
    cluster_priority_defaults = _require_mapping(
        document,
        "cluster_priority_defaults",
        "knowledge pack globals",
    )
    for key in (
        "no_match_recurring",
        "ambiguous_recurring",
        "prescription_non_response",
        "single_run_low_confidence",
        "renderer_weak_evidence_cluster",
        "coach_feedback_wrong_mechanism",
        "coach_feedback_wrong_prescription",
        "coach_feedback_wording",
        "coach_feedback_renderer",
        "coach_feedback_capture_quality",
    ):
        value = _require_string(cluster_priority_defaults, key, "knowledge pack globals")
        if value not in {"A", "B", "C", "D", "E"}:
            raise ValueError(
                f"knowledge pack globals cluster priority default {key} must be one of A-E"
            )
    followup_defaults = _require_mapping(
        document,
        "followup_defaults",
        "knowledge pack globals",
    )
    insufficient_data_status = _require_string(
        followup_defaults,
        "insufficient_data_status",
        "knowledge pack globals",
    )
    if insufficient_data_status != "INSUFFICIENT_DATA":
        raise ValueError(
            "knowledge pack globals followup insufficient_data_status must be INSUFFICIENT_DATA"
        )
    default_window_type = _require_string(
        followup_defaults,
        "default_window_type",
        "knowledge pack globals",
    )
    if default_window_type not in {"next_3_runs", "next_session", "next_2_weeks"}:
        raise ValueError(
            "knowledge pack globals followup default_window_type must be a supported review window"
        )
    min_valid_runs = _require_mapping(
        followup_defaults,
        "min_valid_runs",
        "knowledge pack globals",
    )
    for key in ("next_3_runs", "next_session", "next_2_weeks"):
        minimum = _require_int(min_valid_runs, key, "knowledge pack globals")
        if minimum <= 0:
            raise ValueError(
                f"knowledge pack globals followup min_valid_runs {key} must be > 0"
            )
    history_uncertainty = _require_mapping(
        document,
        "history_uncertainty",
        "knowledge pack globals",
    )
    unresolved_min_runs = _require_int(
        history_uncertainty,
        "unresolved_min_runs",
        "knowledge pack globals",
    )
    unresolved_window_runs = _require_int(
        history_uncertainty,
        "unresolved_window_runs",
        "knowledge pack globals",
    )
    unresolved_rate_min = _require_float(
        history_uncertainty,
        "unresolved_rate_min",
        "knowledge pack globals",
    )
    if unresolved_min_runs <= 0 or unresolved_window_runs <= 0:
        raise ValueError(
            "knowledge pack globals history uncertainty run thresholds must be > 0"
        )
    if unresolved_min_runs > unresolved_window_runs:
        raise ValueError(
            "knowledge pack globals unresolved_min_runs cannot exceed unresolved_window_runs"
        )
    if not (0.0 <= unresolved_rate_min <= 1.0):
        raise ValueError(
            "knowledge pack globals unresolved_rate_min must be between 0 and 1"
        )
    internal_metrics = _require_str_list(document, "internal_metrics", "knowledge pack globals")
    derived_indices = _require_str_list(document, "derived_indices", "knowledge pack globals")

    return {
        "version": document["version"],
        "phase_order": phase_order,
        "confidence_bands": dict(confidence_bands),
        "severity_bands": dict(severity_bands),
        "match_thresholds": dict(thresholds),
        "history_window_defaults": dict(history_defaults),
        "evidence_bands": dict(evidence_bands),
        "presentation_downgrade_rules": dict(presentation_rules),
        "cluster_priority_defaults": dict(cluster_priority_defaults),
        "followup_defaults": {
            "insufficient_data_status": insufficient_data_status,
            "default_window_type": default_window_type,
            "min_valid_runs": dict(min_valid_runs),
        },
        "history_uncertainty": {
            "unresolved_min_runs": unresolved_min_runs,
            "unresolved_window_runs": unresolved_window_runs,
            "unresolved_rate_min": unresolved_rate_min,
        },
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


def _validate_contributors(
    document: Mapping[str, Any],
    *,
    phase_order: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    allowed_phases = set(phase_order).union(
        {"BFC", "FFC", "UAH", "RELEASE", "FFC_TO_RELEASE", "BFC_TO_FFC"}
    )
    contributors = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="contributors",
        label="knowledge pack contributors",
    )
    for contributor_id, contributor in contributors.items():
        label = f"knowledge pack contributor {contributor_id}"
        _require_matching_id(contributor_id, contributor, "knowledge pack contributor")
        finding_type = _require_string(contributor, "finding_type", label)
        if finding_type not in {
            "risk",
            "metric",
            "symptom",
            "phase_anchor",
            "compensation",
            "positive_anchor",
        }:
            raise ValueError(f"{label} finding_type must be a supported contributor type")
        body_group = _require_string(contributor, "body_group", label)
        if body_group not in {"upper_body", "lower_body", "whole_chain"}:
            raise ValueError(
                f"{label} body_group must be upper_body, lower_body, or whole_chain"
            )
        phase = _require_string(contributor, "phase", label)
        if phase not in allowed_phases:
            raise ValueError(
                f"{label} phase must be one of the configured phases or anchor phases"
            )
        source_type = _require_string(contributor, "source_type", label)
        if source_type not in {"risk", "metric", "event", "manual"}:
            raise ValueError(f"{label} source_type must be risk, metric, event, or manual")
        _require_string(contributor, "source_key", label)
        _require_string(contributor, "title", label)
        _require_string(contributor, "summary", label)
        _require_string(contributor, "definition", label)
        _require_str_list(contributor, "why_it_matters", label)
        _require_str_list(contributor, "evidence_inputs", label)
        _require_str_list(contributor, "renderer_focus_regions", label)
        _require_str_list(contributor, "related_symptom_ids", label)
        _require_str_list(contributor, "possible_mechanism_ids", label)
        _require_str_list(contributor, "common_compensation_ids", label)
        _require_str_list(contributor, "common_coexisting_ids", label)
    return contributors


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
        review_window_type = _require_string(prescription, "review_window_type", label)
        if review_window_type not in {"next_3_runs", "next_session", "next_2_weeks"}:
            raise ValueError(
                f"{label} review_window_type must be one of next_3_runs, next_session, next_2_weeks"
            )
        _require_str_list(prescription, "works_best_when", label)
        _require_str_list(prescription, "contraindicated_when", label)
        _require_str_list(prescription, "followup_metric_targets", label)
        _validate_change_reaction(prescription, label)
    return prescriptions


def _validate_change_reaction(
    prescription: Mapping[str, Any],
    label: str,
) -> Dict[str, Any]:
    reaction = _require_mapping(prescription, "change_reaction", label)
    reaction_label = f"{label} change_reaction"
    _require_str_list(reaction, "near_term_positive", reaction_label)
    _require_str_list(reaction, "near_term_negative", reaction_label)
    _require_str_list(reaction, "medium_term_positive", reaction_label)
    _require_str_list(reaction, "medium_term_negative", reaction_label)
    _require_str_list(reaction, "long_term_positive", reaction_label)
    _require_str_list(reaction, "long_term_negative", reaction_label)
    _require_string(reaction, "selection_window_safety", reaction_label)
    match_pressure_risk = _require_string(reaction, "match_pressure_risk", reaction_label)
    if match_pressure_risk not in {"low", "medium", "high"}:
        raise ValueError(
            f"{reaction_label} match_pressure_risk must be one of low, medium, high"
        )
    _require_string(reaction, "adoption_rationale", reaction_label)
    return reaction


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


def _validate_coach_judgments(
    document: Mapping[str, Any],
    *,
    phase_order: Iterable[str],
) -> Dict[str, Any]:
    _require_string(document, "version", "knowledge pack coach judgments")
    chain_statuses = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="chain_statuses",
        label="knowledge pack chain statuses",
    )
    for status_id, status in chain_statuses.items():
        label = f"knowledge pack chain status {status_id}"
        _require_matching_id(status_id, status, "knowledge pack chain status")
        _require_string(status, "title", label)
        _require_string(status, "summary", label)
        _require_string(status, "coach_prompt", label)

    allowed_break_points = set(phase_order).union({"BFC", "FFC", "UAH", "RELEASE"})
    break_points = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="break_points",
        label="knowledge pack break points",
    )
    for break_point_id, break_point in break_points.items():
        label = f"knowledge pack break point {break_point_id}"
        _require_matching_id(break_point_id, break_point, "knowledge pack break point")
        if break_point_id not in allowed_break_points:
            raise ValueError(f"{label} must match a configured phase or anchor phase")
        _require_string(break_point, "title", label)
        _require_string(break_point, "summary", label)
        _require_string(break_point, "coach_question", label)

    change_size_bands = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="change_size_bands",
        label="knowledge pack change size bands",
    )
    for band_id, band in change_size_bands.items():
        label = f"knowledge pack change size band {band_id}"
        _require_matching_id(band_id, band, "knowledge pack change size band")
        _require_string(band, "title", label)
        _require_string(band, "summary", label)
        _require_string(band, "adoption_hint", label)

    adoption_risk_bands = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="adoption_risk_bands",
        label="knowledge pack adoption risk bands",
    )
    for risk_id, risk in adoption_risk_bands.items():
        label = f"knowledge pack adoption risk band {risk_id}"
        _require_matching_id(risk_id, risk, "knowledge pack adoption risk band")
        _require_string(risk, "title", label)
        _require_string(risk, "summary", label)
        _require_string(risk, "coach_meaning", label)

    reaction_horizons = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="reaction_horizons",
        label="knowledge pack reaction horizons",
    )
    for horizon_id, horizon in reaction_horizons.items():
        label = f"knowledge pack reaction horizon {horizon_id}"
        _require_matching_id(horizon_id, horizon, "knowledge pack reaction horizon")
        _require_string(horizon, "title", label)
        _require_string(horizon, "summary", label)
        _require_string(horizon, "typical_window", label)

    return {
        "version": document["version"],
        "chain_statuses": chain_statuses,
        "break_points": break_points,
        "change_size_bands": change_size_bands,
        "adoption_risk_bands": adoption_risk_bands,
        "reaction_horizons": reaction_horizons,
    }


def _validate_capture_templates(document: Mapping[str, Any]) -> Dict[str, Any]:
    _require_string(document, "version", "knowledge pack capture templates")
    return {
        "version": document["version"],
        "coach_review_questionnaire": _validate_capture_field_collection(
            document,
            collection_key="coach_review_questionnaire",
            label="knowledge pack coach review questionnaire",
        ),
        "clip_annotation_fields": _validate_capture_field_collection(
            document,
            collection_key="clip_annotation_fields",
            label="knowledge pack clip annotation fields",
        ),
        "intervention_outcome_fields": _validate_capture_field_collection(
            document,
            collection_key="intervention_outcome_fields",
            label="knowledge pack intervention outcome fields",
        ),
        "outcome_windows": _validate_capture_window_collection(document),
    }


def _validate_capture_field_collection(
    document: Mapping[str, Any],
    *,
    collection_key: str,
    label: str,
) -> Dict[str, Dict[str, Any]]:
    allowed_response_types = {
        "string",
        "text",
        "single_select",
        "multi_select",
        "number",
        "boolean",
    }
    fields = _require_entity_mapping(
        document,
        version_key="version",
        collection_key=collection_key,
        label=label,
    )
    for field_id, field in fields.items():
        field_label = f"{label} field {field_id}"
        _require_matching_id(field_id, field, field_label)
        _require_string(field, "section", field_label)
        _require_string(field, "prompt", field_label)
        response_type = _require_string(field, "response_type", field_label)
        if response_type not in allowed_response_types:
            raise ValueError(f"{field_label} response_type must be a supported type")
        _require_bool(field, "required", field_label)
        _require_string(field, "maps_to", field_label)
        options = _require_str_list(field, "options", field_label)
        if response_type in {"single_select", "multi_select"} and not options:
            raise ValueError(f"{field_label} select questions must define options")
        if response_type not in {"single_select", "multi_select"} and options:
            raise ValueError(f"{field_label} only select questions may define options")
    return fields


def _validate_capture_window_collection(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    windows = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="outcome_windows",
        label="knowledge pack outcome windows",
    )
    for window_id, window in windows.items():
        label = f"knowledge pack outcome window {window_id}"
        _require_matching_id(window_id, window, "knowledge pack outcome window")
        _require_string(window, "title", label)
        _require_string(window, "summary", label)
        _require_string(window, "typical_use", label)
    return windows


def _validate_architecture_principles(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    principles = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="architecture_principles",
        label="knowledge pack architecture principles",
    )
    for principle_id, principle in principles.items():
        label = f"knowledge pack architecture principle {principle_id}"
        _require_matching_id(principle_id, principle, "knowledge pack architecture principle")
        principle_type = _require_string(principle, "principle_type", label)
        if principle_type not in {
            "classification_phase",
            "validation_phase",
            "human_phase",
            "automation_boundary",
            "trajectory_principle",
            "knowledge_governance",
        }:
            raise ValueError(f"{label} principle_type must be a supported architecture type")
        _require_string(principle, "title", label)
        _require_string(principle, "summary", label)
        _require_string(principle, "why_it_exists", label)
        _require_str_list(principle, "product_implications", label)
        _require_str_list(principle, "do_not_break", label)
        _require_str_list(principle, "source_ids", label)
        _require_str_list(principle, "linked_mechanism_ids", label)
        _require_str_list(principle, "linked_contributor_ids", label)
        _require_str_list(principle, "linked_prescription_ids", label)
    return principles


def _validate_research_sources(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    sources = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="research_sources",
        label="knowledge pack research sources",
    )
    for source_id, source in sources.items():
        label = f"knowledge pack research source {source_id}"
        _require_matching_id(source_id, source, "knowledge pack research source")
        source_type = _require_string(source, "source_type", label)
        if source_type not in {
            "paper",
            "official_body",
            "university_research",
            "expert_practitioner",
            "internal_review",
            "architecture_reference",
        }:
            raise ValueError(f"{label} source_type must be a supported source type")
        evidence_tier = _require_string(source, "evidence_tier", label)
        if evidence_tier not in {"A", "B", "C", "INTERNAL"}:
            raise ValueError(f"{label} evidence_tier must be A, B, C, or INTERNAL")
        _require_string(source, "title", label)
        _require_string(source, "publisher", label)
        _require_string(source, "year", label)
        _require_string(source, "url", label)
        _require_string(source, "scope", label)
        _require_str_list(source, "tags", label)
        _require_string(source, "status", label)
        _require_string(source, "notes", label)
    return sources


def _validate_knowledge_evidence(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    evidence_items = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="evidence_items",
        label="knowledge pack evidence items",
    )
    for evidence_id, evidence in evidence_items.items():
        label = f"knowledge pack evidence item {evidence_id}"
        _require_matching_id(evidence_id, evidence, "knowledge pack evidence item")
        target_type = _require_string(evidence, "target_type", label)
        if target_type not in {
            "symptom",
            "mechanism",
            "contributor",
            "prescription",
            "trajectory",
        }:
            raise ValueError(f"{label} target_type must be a supported knowledge target")
        _require_string(evidence, "target_id", label)
        evidence_kind = _require_string(evidence, "evidence_kind", label)
        if evidence_kind not in {
            "biomechanics_truth",
            "coaching_translation",
            "intervention_heuristic",
            "load_risk",
            "performance_relation",
        }:
            raise ValueError(f"{label} evidence_kind must be a supported evidence kind")
        evidence_tier = _require_string(evidence, "evidence_tier", label)
        if evidence_tier not in {"A", "B", "C", "INTERNAL"}:
            raise ValueError(f"{label} evidence_tier must be A, B, C, or INTERNAL")
        _require_str_list(evidence, "source_ids", label)
        _require_string(evidence, "claim_summary", label)
        _require_str_list(evidence, "extraction_notes", label)
        consensus = _require_string(evidence, "coach_consensus_status", label)
        if consensus not in {"draft", "reviewed", "accepted"}:
            raise ValueError(f"{label} coach_consensus_status must be draft, reviewed, or accepted")
        _require_str_list(evidence, "related_evidence_ids", label)
    return evidence_items


def _validate_reconciliation(document: Mapping[str, Any]) -> Dict[str, Any]:
    _require_string(document, "version", "knowledge pack reconciliation")
    canonical_concepts = _require_entity_mapping(
        document,
        version_key="version",
        collection_key="canonical_concepts",
        label="knowledge pack canonical concepts",
    )
    for concept_id, concept in canonical_concepts.items():
        label = f"knowledge pack canonical concept {concept_id}"
        _require_matching_id(concept_id, concept, "knowledge pack canonical concept")
        concept_type = _require_string(concept, "concept_type", label)
        if concept_type not in {"contributor", "mechanism", "prescription", "symptom"}:
            raise ValueError(f"{label} concept_type must be contributor, mechanism, prescription, or symptom")
        canonical_target_type = _require_string(concept, "canonical_target_type", label)
        if canonical_target_type not in {"contributor", "mechanism", "prescription", "symptom"}:
            raise ValueError(f"{label} canonical_target_type must be a supported target type")
        _require_string(concept, "canonical_target_id", label)
        _require_string(concept, "title", label)
        _require_string(concept, "merge_status", label)
        _require_str_list(concept, "duplicate_targets", label)
        _require_str_list(concept, "similar_targets", label)
        _require_str_list(concept, "related_evidence_ids", label)
        _require_string(concept, "reconciliation_note", label)
    return {
        "version": document["version"],
        "canonical_concepts": canonical_concepts,
    }


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
    required_surfaces = ("player", "coach")
    optional_surfaces = ("reviewer", "clinician")
    for surface_name in required_surfaces:
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
    coach_surface = dict(normalized_surfaces["coach"])
    for surface_name in optional_surfaces:
        raw_cfg = surfaces.get(surface_name)
        if raw_cfg is None:
            normalized_surfaces[surface_name] = dict(coach_surface)
            continue
        if not isinstance(raw_cfg, Mapping):
            raise ValueError(f"knowledge pack wording surface {surface_name} must be a mapping")
        surface_cfg = dict(raw_cfg)
        normalized_surfaces[surface_name] = {
            "primary_mechanism_prefix": _optional_string(
                surface_cfg,
                "primary_mechanism_prefix",
                coach_surface["primary_mechanism_prefix"],
            ),
            "performance_prefix": _optional_string(
                surface_cfg,
                "performance_prefix",
                coach_surface["performance_prefix"],
            ),
            "load_prefix": _optional_string(
                surface_cfg,
                "load_prefix",
                coach_surface["load_prefix"],
            ),
        }

    unknown_path_surfaces = _require_mapping(
        document,
        "unknown_path_surfaces",
        "knowledge pack wording",
    )
    normalized_unknown_surfaces: Dict[str, Dict[str, str]] = {}
    for surface_name in required_surfaces:
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
    coach_unknown_surface = dict(normalized_unknown_surfaces["coach"])
    for surface_name in optional_surfaces:
        raw_cfg = unknown_path_surfaces.get(surface_name)
        if raw_cfg is None:
            normalized_unknown_surfaces[surface_name] = dict(coach_unknown_surface)
            continue
        if not isinstance(raw_cfg, Mapping):
            raise ValueError(
                f"knowledge pack wording unknown surface {surface_name} must be a mapping"
            )
        surface_cfg = dict(raw_cfg)
        normalized_unknown_surfaces[surface_name] = {
            "primary_mechanism": _optional_string(
                surface_cfg,
                "primary_mechanism",
                coach_unknown_surface["primary_mechanism"],
            ),
            "performance_impact": _optional_string(
                surface_cfg,
                "performance_impact",
                coach_unknown_surface["performance_impact"],
            ),
            "load_impact": _optional_string(
                surface_cfg,
                "load_impact",
                coach_unknown_surface["load_impact"],
            ),
            "coach_check": _optional_string(
                surface_cfg,
                "coach_check",
                coach_unknown_surface["coach_check"],
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
        for surface_name in optional_surfaces:
            normalized_status_leads[status_name][surface_name] = _optional_string(
                status_cfg,
                surface_name,
                normalized_status_leads[status_name]["coach"],
            )

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


def _validate_contributor_links(
    contributors: Mapping[str, Mapping[str, Any]],
    *,
    symptom_ids: Iterable[str],
    mechanism_ids: Iterable[str],
) -> None:
    allowed_symptoms = set(symptom_ids)
    allowed_mechanisms = set(mechanism_ids)
    allowed_contributors = set(contributors.keys())
    for contributor_id, contributor in contributors.items():
        _validate_known_ids(
            contributor["related_symptom_ids"],
            allowed_symptoms,
            f"knowledge pack contributor {contributor_id} related_symptom_ids",
        )
        _validate_known_ids(
            contributor["possible_mechanism_ids"],
            allowed_mechanisms,
            f"knowledge pack contributor {contributor_id} possible_mechanism_ids",
        )
        _validate_known_ids(
            contributor["common_compensation_ids"],
            allowed_contributors,
            f"knowledge pack contributor {contributor_id} common_compensation_ids",
        )
        _validate_known_ids(
            contributor["common_coexisting_ids"],
            allowed_contributors,
            f"knowledge pack contributor {contributor_id} common_coexisting_ids",
        )


def _validate_capture_template_links(
    capture_templates: Mapping[str, Any],
    *,
    coach_judgments: Mapping[str, Any],
    symptom_ids: Iterable[str],
    mechanism_ids: Iterable[str],
    contributor_ids: Iterable[str],
    prescription_ids: Iterable[str],
) -> None:
    option_sets = {
        "chain_status": set(coach_judgments["chain_statuses"].keys()),
        "break_point": set(coach_judgments["break_points"].keys()),
        "change_size": set(coach_judgments["change_size_bands"].keys()),
        "adoption_risk": set(coach_judgments["adoption_risk_bands"].keys()),
        "symptom": set(symptom_ids),
        "mechanism": set(mechanism_ids),
        "contributor": set(contributor_ids),
        "prescription": set(prescription_ids),
        "reviewer_role": {"coach", "clinician", "reviewer"},
        "context": {"training", "trial", "match", "rehab", "off_season"},
        "adoption_observed": {"yes", "partial", "no"},
    }
    for collection_name in (
        "coach_review_questionnaire",
        "clip_annotation_fields",
        "intervention_outcome_fields",
    ):
        for field_id, field in capture_templates[collection_name].items():
            options = field["options"]
            if not options:
                continue
            maps_to = str(field["maps_to"]).strip().lower()
            option_group = None
            for key in option_sets:
                if key in maps_to:
                    option_group = key
                    break
            if option_group is None:
                continue
            _validate_known_ids(
                options,
                option_sets[option_group],
                f"knowledge pack capture template field {field_id} options",
            )


def _validate_architecture_principle_links(
    architecture_principles: Mapping[str, Mapping[str, Any]],
    *,
    source_ids: Iterable[str],
    mechanism_ids: Iterable[str],
    contributor_ids: Iterable[str],
    prescription_ids: Iterable[str],
) -> None:
    allowed_sources = set(source_ids)
    allowed_mechanisms = set(mechanism_ids)
    allowed_contributors = set(contributor_ids)
    allowed_prescriptions = set(prescription_ids)
    for principle_id, principle in architecture_principles.items():
        _validate_known_ids(
            principle["source_ids"],
            allowed_sources,
            f"knowledge pack architecture principle {principle_id} source_ids",
        )
        _validate_known_ids(
            principle["linked_mechanism_ids"],
            allowed_mechanisms,
            f"knowledge pack architecture principle {principle_id} linked_mechanism_ids",
        )
        _validate_known_ids(
            principle["linked_contributor_ids"],
            allowed_contributors,
            f"knowledge pack architecture principle {principle_id} linked_contributor_ids",
        )
        _validate_known_ids(
            principle["linked_prescription_ids"],
            allowed_prescriptions,
            f"knowledge pack architecture principle {principle_id} linked_prescription_ids",
        )


def _validate_knowledge_evidence_links(
    knowledge_evidence: Mapping[str, Mapping[str, Any]],
    *,
    source_ids: Iterable[str],
    symptom_ids: Iterable[str],
    mechanism_ids: Iterable[str],
    contributor_ids: Iterable[str],
    prescription_ids: Iterable[str],
    trajectory_ids: Iterable[str],
) -> None:
    allowed_sources = set(source_ids)
    target_sets = {
        "symptom": set(symptom_ids),
        "mechanism": set(mechanism_ids),
        "contributor": set(contributor_ids),
        "prescription": set(prescription_ids),
        "trajectory": set(trajectory_ids),
    }
    allowed_evidence_ids = set(knowledge_evidence.keys())
    for evidence_id, evidence in knowledge_evidence.items():
        _validate_known_ids(
            evidence["source_ids"],
            allowed_sources,
            f"knowledge pack evidence item {evidence_id} source_ids",
        )
        target_type = evidence["target_type"]
        target_id = evidence["target_id"]
        if target_id not in target_sets[target_type]:
            raise ValueError(
                f"knowledge pack evidence item {evidence_id} target_id {target_id} is unknown for type {target_type}"
            )
        _validate_known_ids(
            evidence["related_evidence_ids"],
            allowed_evidence_ids,
            f"knowledge pack evidence item {evidence_id} related_evidence_ids",
        )


def _validate_reconciliation_links(
    reconciliation: Mapping[str, Any],
    *,
    symptom_ids: Iterable[str],
    mechanism_ids: Iterable[str],
    contributor_ids: Iterable[str],
    prescription_ids: Iterable[str],
    evidence_ids: Iterable[str],
) -> None:
    target_sets = {
        "symptom": set(symptom_ids),
        "mechanism": set(mechanism_ids),
        "contributor": set(contributor_ids),
        "prescription": set(prescription_ids),
    }
    allowed_evidence = set(evidence_ids)

    def _validate_target_ref(value: str, label: str) -> None:
        target_type, sep, target_id = value.partition(":")
        if sep != ":" or not target_type or not target_id:
            raise ValueError(f"{label} must use target_type:target_id format")
        if target_type not in target_sets:
            raise ValueError(f"{label} uses unsupported target_type {target_type}")
        if target_id not in target_sets[target_type]:
            raise ValueError(f"{label} references unknown target_id {target_id}")

    for concept_id, concept in reconciliation["canonical_concepts"].items():
        canonical_type = concept["canonical_target_type"]
        canonical_id = concept["canonical_target_id"]
        if canonical_id not in target_sets[canonical_type]:
            raise ValueError(
                f"knowledge pack canonical concept {concept_id} references unknown canonical target {canonical_type}:{canonical_id}"
            )
        for ref in concept["duplicate_targets"]:
            _validate_target_ref(
                ref,
                f"knowledge pack canonical concept {concept_id} duplicate target {ref}",
            )
        for ref in concept["similar_targets"]:
            _validate_target_ref(
                ref,
                f"knowledge pack canonical concept {concept_id} similar target {ref}",
            )
        _validate_known_ids(
            concept["related_evidence_ids"],
            allowed_evidence,
            f"knowledge pack canonical concept {concept_id} related_evidence_ids",
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


def _require_bool(payload: Mapping[str, Any], key: str, label: str) -> bool:
    if key not in payload:
        raise ValueError(f"{label} missing required key: {key}")
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{label} {key} must be a boolean")
    return value


def _optional_string(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key)
    if value is None:
        return str(default)
    if not isinstance(value, str):
        raise ValueError(f"Optional key {key} must be a string when provided")
    text = value.strip()
    return text or str(default)


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
