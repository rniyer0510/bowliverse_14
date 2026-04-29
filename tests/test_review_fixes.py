import importlib
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi import HTTPException


class ResolverFixTests(unittest.TestCase):
    def test_resolve_player_requires_handedness_for_new_player(self):
        from app.persistence.resolver import resolve_player

        db = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = None
        account = SimpleNamespace(role="player", account_id="acc-1")

        with self.assertRaises(ValueError):
            resolve_player(db, account, actor={"player_name": "Test Player"})

    def test_resolve_player_sets_normalized_handedness_and_current_season(self):
        from app.persistence.resolver import resolve_player

        db = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = None
        account = SimpleNamespace(role="player", account_id="acc-1")

        resolve_player(
            db,
            account,
            actor={"player_name": "Test Player", "handedness": "left"},
        )

        created_player = db.add.call_args_list[0].args[0]
        self.assertEqual(created_player.handedness, "L")
        self.assertEqual(created_player.season, datetime.utcnow().year)


class AuthSecretTests(unittest.TestCase):
    def test_auth_module_imports_without_secret_until_token_use(self):
        with patch.dict(os.environ, {}, clear=True):
            module = importlib.import_module("app.common.auth")
            module = importlib.reload(module)

            with self.assertRaises(RuntimeError):
                module.create_access_token(
                    SimpleNamespace(user_id="u1", account_id="a1", role="coach")
                )

        with patch.dict(os.environ, {"ACTIONLAB_SECRET": "test-secret"}, clear=True):
            importlib.reload(module)


class SessionConfigTests(unittest.TestCase):
    def test_session_requires_explicit_db_url_in_production(self):
        module = importlib.import_module("app.persistence.session")

        with patch.dict(os.environ, {"ACTIONLAB_ENV": "production"}, clear=True):
            with self.assertRaises(RuntimeError):
                importlib.reload(module)

        with patch.dict(
            os.environ,
            {
                "ACTIONLAB_ENV": "development",
                "ACTIONLAB_LOCAL_DB_URL": "postgresql+psycopg2://actionlab@localhost/actionlab",
            },
            clear=True,
        ):
            importlib.reload(module)

    def test_session_engine_uses_bounded_pool_defaults(self):
        module = importlib.import_module("app.persistence.session")

        with patch.dict(
            os.environ,
            {"ACTIONLAB_LOCAL_DB_URL": "postgresql+psycopg2://actionlab@localhost/actionlab"},
            clear=True,
        ), patch("sqlalchemy.create_engine") as create_engine_mock:
            importlib.reload(module)

        create_engine_mock.assert_called_once()
        kwargs = create_engine_mock.call_args.kwargs
        self.assertEqual(kwargs["pool_size"], 10)
        self.assertEqual(kwargs["max_overflow"], 20)
        self.assertEqual(kwargs["pool_timeout"], 30)
        self.assertEqual(kwargs["pool_recycle"], 1800)
        with patch.dict(
            os.environ,
            {"ACTIONLAB_LOCAL_DB_URL": "postgresql+psycopg2://actionlab@localhost/actionlab"},
            clear=True,
        ):
            importlib.reload(module)


class ReleaseDebugTests(unittest.TestCase):
    def test_release_uah_debug_defaults_off(self):
        module = importlib.import_module("app.workers.events.release_uah")

        with patch.dict(os.environ, {}, clear=True):
            module = importlib.reload(module)
            self.assertFalse(module.DEBUG)

        with patch.dict(os.environ, {"ACTIONLAB_RELEASE_UAH_DEBUG": "true"}, clear=True):
            module = importlib.reload(module)
            self.assertTrue(module.DEBUG)


class RiskWorkerFixTests(unittest.TestCase):
    def test_synthetic_benchmark_percentiles_disabled_by_default(self):
        from app.workers.risk import risk_worker

        with patch.object(
            risk_worker,
            "ENABLE_SYNTHETIC_BENCHMARK_PERCENTILES",
            False,
        ):
            self.assertIsNone(risk_worker._benchmark_percentile(0.9))


class ClinicianYamlValidationTests(unittest.TestCase):
    def test_globals_yaml_requires_confidence_and_severity_bands(self):
        from app.clinician import loader

        with self.assertRaises(ValueError):
            loader._validate_yaml_payload("globals.yaml", {"confidence_bands": {}})

    def test_risks_yaml_requires_explanations_mapping(self):
        from app.clinician import loader

        with self.assertRaises(ValueError):
            loader._validate_yaml_payload(
                "risks.yaml",
                {"trunk_rotation_snap": {"title": "Body Turn"}},
            )

    def test_clinician_modules_do_not_load_yaml_at_import_time(self):
        from app.clinician import bands, interpreter, loader

        with patch.object(loader, "load_yaml") as load_yaml_mock:
            importlib.reload(bands)
            importlib.reload(interpreter)

        load_yaml_mock.assert_not_called()
        importlib.reload(bands)
        importlib.reload(interpreter)

    def test_default_knowledge_pack_validates(self):
        from app.clinician.knowledge_pack import validate_default_knowledge_pack

        validate_default_knowledge_pack()

    def test_band_helpers_prefer_knowledge_pack_semantic_bands(self):
        from app.clinician import bands

        pack_globals = {
            "globals": {
                "severity_bands": {
                    "low": {"min": 0.0, "max": 0.2},
                    "moderate": {"min": 0.2, "max": 0.5},
                    "high": {"min": 0.5, "max": 0.8},
                    "very_high": {"min": 0.8, "max": 1.01},
                },
                "confidence_bands": {
                    "high": {"min": 0.9},
                    "medium": {"min": 0.4},
                    "low": {"min": 0.0},
                },
            }
        }

        with patch("app.clinician.bands.load_knowledge_pack", return_value=pack_globals), patch(
            "app.clinician.bands.load_yaml"
        ) as load_yaml_mock:
            self.assertEqual(bands.severity_band(0.65), "HIGH")
            self.assertEqual(bands.confidence_band(0.45), "MEDIUM")

        load_yaml_mock.assert_not_called()

    def test_pillar_positive_bonus_caps_are_enabled(self):
        from app.clinician.interpreter import PILLAR_MAX_BONUS

        self.assertGreater(PILLAR_MAX_BONUS["POSTURE"], 0.0)
        self.assertGreater(PILLAR_MAX_BONUS["POWER"], 0.0)
        self.assertGreater(PILLAR_MAX_BONUS["PROTECTION"], 0.0)


class ReviewerRoleTests(unittest.TestCase):
    def test_review_role_helpers_allow_reviewer_and_clinician(self):
        from app.persistence.read_api import _require_release_reviewer
        from app.persistence.write_api import _require_coach_reviewer

        for role in ("coach", "reviewer", "clinician"):
            _require_release_reviewer(SimpleNamespace(role=role))
            _require_coach_reviewer(SimpleNamespace(role=role))

        with self.assertRaises(HTTPException):
            _require_release_reviewer(SimpleNamespace(role="player"))
        with self.assertRaises(HTTPException):
            _require_coach_reviewer(SimpleNamespace(role="parent"))


class AnalyzePersistenceFallbackTests(unittest.TestCase):
    def test_resolve_analysis_hand_prefers_clip_evidence_over_profile_hand(self):
        from app.orchestrator import orchestrator

        pose_frames = [{"frame": idx, "landmarks": []} for idx in range(200)]

        def fake_release(*, hand, **_kwargs):
            if hand == "R":
                return {
                    "release": {"frame": 141, "confidence": 0.77, "method": "multi_signal_consensus", "window": [140, 154]},
                    "uah": {"frame": 139, "confidence": 0.68, "method": "bowling_elbow_height_minimum", "window": [135, 140]},
                    "delivery_window": [140, 154],
                }
            return {
                "release": {"frame": 136, "confidence": 0.60, "method": "multi_signal_consensus", "window": [134, 145]},
                "uah": {"frame": 132, "confidence": 0.85, "method": "bowling_elbow_height_minimum_ffc_reconciled", "window": [130, 135]},
                "delivery_window": [134, 145],
            }

        def fake_feet(*, hand, **_kwargs):
            if hand == "R":
                return {
                    "ffc": {"frame": 137, "confidence": 0.90, "method": "release_backward_chain_grounding", "window": [118, 139]},
                    "bfc": {"frame": 134, "confidence": 0.0, "method": "simple_grounded_bfc"},
                }
            return {
                "ffc": {"frame": 132, "confidence": 0.90, "method": "release_backward_chain_grounding", "window": [113, 134]},
                "bfc": {"frame": 128, "confidence": 0.69, "method": "back_foot_support_edge"},
            }

        def fake_cache(*, hand, **_kwargs):
            raw = np.full(200, 0.95 if hand == "R" else 0.70, dtype=float)
            weight = np.full(200, 0.90 if hand == "R" else 0.60, dtype=float)
            if hand == "R":
                nb_signal = np.linspace(0.8, 0.5, 200)
                nb_signal[141] = 0.2
                shoulder_signal = np.zeros(200, dtype=float)
                shoulder_signal[140] = 5.0
            else:
                nb_signal = np.linspace(0.8, 0.5, 200)
                nb_signal[150] = 0.2
                shoulder_signal = np.zeros(200, dtype=float)
                shoulder_signal[151] = 5.0
            return {
                "wrist_vis_raw": raw,
                "wrist_vis_weight": weight,
                "bowling_elbow_vis_raw": raw,
                "bowling_elbow_vis_weight": weight,
                "nb_elbow_vis_raw": raw,
                "nb_elbow_vis_weight": weight,
                "nb_elbow_y": nb_signal,
                "shoulder_angular_velocity": shoulder_signal,
            }

        with patch.object(orchestrator, "detect_release_uah", side_effect=fake_release), patch.object(
            orchestrator,
            "detect_ffc_bfc",
            side_effect=fake_feet,
        ), patch.object(orchestrator, "build_signal_cache", side_effect=fake_cache):
            resolved_hand, events, debug = orchestrator._resolve_analysis_hand(
                pose_frames=pose_frames,
                fps=25.0,
                delivery_window={"analysis_start": 100, "analysis_end": 160},
                preferred_hand="L",
            )

        self.assertEqual(resolved_hand, "R")
        self.assertEqual((events.get("release") or {}).get("frame"), 141)
        self.assertEqual(debug["resolution_reason"], "clip_evidence_override")
        self.assertIn("uncertain_checks", debug["hypotheses"]["R"])

    def test_hand_resolution_prefers_non_bowling_arm_when_release_hint_matches_rear_view_cluster(self):
        from app.orchestrator import orchestrator

        pose_frames = [{"frame": idx, "landmarks": []} for idx in range(220)]

        def fake_release(*, hand, **_kwargs):
            if hand == "R":
                return {
                    "release": {"frame": 95, "confidence": 0.42, "method": "multi_signal_consensus", "window": [0, 110]},
                    "uah": {"frame": 86, "confidence": 0.75, "method": "kinetic_chain_crossover", "window": [82, 94]},
                    "delivery_window": [0, 110],
                }
            return {
                "release": {"frame": 148, "confidence": 0.55, "method": "multi_signal_consensus", "window": [0, 161]},
                "uah": {"frame": 144, "confidence": 0.75, "method": "kinetic_chain_crossover", "window": [138, 147]},
                "delivery_window": [0, 161],
            }

        def fake_feet(*, hand, **_kwargs):
            if hand == "R":
                return {
                    "ffc": {"frame": 87, "confidence": 0.90, "method": "release_backward_chain_grounding", "window": [41, 93]},
                    "bfc": {"frame": 82, "confidence": 0.50, "method": "back_foot_support_edge"},
                }
            return {
                "ffc": {"frame": 135, "confidence": 0.90, "method": "release_backward_chain_grounding", "window": [94, 146]},
                "bfc": {"frame": 133, "confidence": 0.50, "method": "back_foot_support_edge"},
            }

        def fake_cache(*, hand, **_kwargs):
            raw_r = np.full(220, 0.75, dtype=float)
            raw_l = np.full(220, 0.62, dtype=float)
            nb_raw = raw_r if hand == "R" else raw_l
            nb_weight = np.full(220, 0.82 if hand == "R" else 0.63, dtype=float)
            bowl_raw = np.full(220, 0.61 if hand == "R" else 0.68, dtype=float)
            bowl_weight = np.full(220, 0.67 if hand == "R" else 0.69, dtype=float)
            wrist_raw = np.full(220, 0.75 if hand == "R" else 0.63, dtype=float)
            wrist_weight = np.full(220, 0.82 if hand == "R" else 0.64, dtype=float)
            nb_signal = np.full(220, 0.7, dtype=float)
            shoulder_signal = np.zeros(220, dtype=float)
            if hand == "R":
                nb_signal[93] = 0.2
                shoulder_signal[87] = 5.0
            else:
                nb_signal[147] = 0.2
                shoulder_signal[142] = 5.0
            return {
                "wrist_vis_raw": wrist_raw,
                "wrist_vis_weight": wrist_weight,
                "bowling_elbow_vis_raw": bowl_raw,
                "bowling_elbow_vis_weight": bowl_weight,
                "nb_elbow_vis_raw": nb_raw,
                "nb_elbow_vis_weight": nb_weight,
                "nb_elbow_y": nb_signal,
                "shoulder_angular_velocity": shoulder_signal,
            }

        with patch.object(orchestrator, "detect_release_uah", side_effect=fake_release), patch.object(
            orchestrator,
            "detect_ffc_bfc",
            side_effect=fake_feet,
        ), patch.object(orchestrator, "build_signal_cache", side_effect=fake_cache):
            resolved_hand, events, debug = orchestrator._resolve_analysis_hand(
                pose_frames=pose_frames,
                fps=60.0,
                delivery_window={"release_hint": 88, "analysis_start": 0, "analysis_end": 166},
                preferred_hand="R",
            )

        self.assertEqual(resolved_hand, "R")
        self.assertEqual((events.get("release") or {}).get("frame"), 95)
        self.assertTrue(debug["hypotheses"]["R"]["behind_camera_hint"])
        self.assertGreater(
            debug["hypotheses"]["R"]["behind_camera_non_bowling_arm_score"],
            debug["hypotheses"]["L"]["behind_camera_non_bowling_arm_score"],
        )

    def test_persist_analysis_result_returns_warning_after_retries_fail(self):
        from app.orchestrator import orchestrator

        result = {"run_id": "run-1"}
        first_session = MagicMock()
        second_session = MagicMock()

        with patch.object(
            orchestrator,
            "SessionLocal",
            side_effect=[first_session, second_session],
        ), patch.object(
            orchestrator,
            "write_analysis",
            side_effect=[RuntimeError("db down"), RuntimeError("db still down")],
        ):
            status = orchestrator._persist_analysis_result(
                request_id="req-1",
                run_id="run-1",
                result=result,
                video={"path": "/tmp/video.mp4"},
                bowler_type="pace",
                actor_obj={"account_id": "acc-1"},
                effective_age_group="U16",
                effective_season=2026,
            )

        self.assertFalse(status["persisted"])
        self.assertEqual(status["attempts"], 2)
        self.assertEqual(result["warnings"][0]["code"], "analysis_not_persisted")

    def test_persist_analysis_result_recovers_on_second_attempt(self):
        from app.orchestrator import orchestrator

        result = {"run_id": "run-1"}
        first_session = MagicMock()
        second_session = MagicMock()

        with patch.object(
            orchestrator,
            "SessionLocal",
            side_effect=[first_session, second_session],
        ), patch.object(
            orchestrator,
            "write_analysis",
            side_effect=[RuntimeError("transient"), "run-1"],
        ):
            status = orchestrator._persist_analysis_result(
                request_id="req-1",
                run_id="run-1",
                result=result,
                video={"path": "/tmp/video.mp4"},
                bowler_type="pace",
                actor_obj={"account_id": "acc-1"},
                effective_age_group="U16",
                effective_season=2026,
            )

        self.assertTrue(status["persisted"])
        self.assertEqual(status["attempts"], 2)
        self.assertNotIn("warnings", result)


class WriterSessionOwnershipTests(unittest.TestCase):
    def test_write_analysis_with_external_session_flushes_without_committing(self):
        from app.persistence import writer
        from app.persistence.models import AnalysisExplanationTrace, Player

        db = MagicMock()
        player = SimpleNamespace(season=2026, age_group="U16")

        def get_side_effect(model, key):
            if model is Player:
                return player
            if model is AnalysisExplanationTrace:
                return None
            return None

        db.get.side_effect = get_side_effect
        result = {
            "input": {"player_id": "11111111-1111-1111-1111-111111111111", "hand": "R"},
            "video": {"fps": 30.0, "total_frames": 120},
            "events": {},
            "elbow": {},
            "action": {},
            "risks": [],
            "deterministic_expert_v1": {
                "knowledge_pack_id": "actionlab_deterministic_expert",
                "knowledge_pack_version": "2026-04-22.v1",
                "selection": {
                    "diagnosis_status": "partial_match",
                    "primary_mechanism_id": "soft_block_with_trunk_carry",
                },
                "archetype_v1": {"id": "soft_block_leakage_bowler"},
            },
        }

        run_id = "22222222-2222-2222-2222-222222222222"
        persisted_run_id = writer.write_analysis(
            result=result,
            db=db,
            run_id=run_id,
            season=2026,
            age_group="U16",
        )

        self.assertEqual(persisted_run_id, run_id)
        created_run = db.add.call_args_list[0].args[0]
        self.assertEqual(created_run.knowledge_pack_id, "actionlab_deterministic_expert")
        self.assertEqual(created_run.knowledge_pack_version, "2026-04-22.v1")
        self.assertEqual(created_run.deterministic_diagnosis_status, "partial_match")
        self.assertEqual(
            created_run.deterministic_primary_mechanism_id,
            "soft_block_with_trunk_carry",
        )
        self.assertTrue(
            any(
                isinstance(call.args[0], AnalysisExplanationTrace)
                for call in db.add.call_args_list
            )
        )
        self.assertEqual(
            created_run.deterministic_archetype_id,
            "soft_block_leakage_bowler",
        )
        db.flush.assert_called()
        db.commit.assert_not_called()
        db.rollback.assert_not_called()

    def test_write_analysis_skips_explanation_trace_when_table_is_missing(self):
        from app.persistence import writer
        from app.persistence.models import AnalysisExplanationTrace, Player

        db = MagicMock()
        player = SimpleNamespace(season=2026, age_group="U16")

        def get_side_effect(model, key):
            if model is Player:
                return player
            if model is AnalysisExplanationTrace:
                raise AssertionError("trace lookup should be skipped when table is unavailable")
            return None

        db.get.side_effect = get_side_effect
        result = {
            "input": {"player_id": "11111111-1111-1111-1111-111111111111", "hand": "R"},
            "video": {"fps": 30.0, "total_frames": 120},
            "events": {},
            "elbow": {},
            "action": {},
            "risks": [],
            "deterministic_expert_v1": {
                "knowledge_pack_id": "actionlab_deterministic_expert",
                "knowledge_pack_version": "2026-04-22.v1",
                "selection": {
                    "diagnosis_status": "partial_match",
                    "primary_mechanism_id": "soft_block_with_trunk_carry",
                },
                "archetype_v1": {"id": "soft_block_leakage_bowler"},
            },
        }

        with patch("app.persistence.writer.inspect") as inspect_mock:
            inspect_mock.return_value.has_table.return_value = False
            persisted_run_id = writer.write_analysis(
                result=result,
                db=db,
                run_id="22222222-2222-2222-2222-222222222222",
                season=2026,
                age_group="U16",
            )

        self.assertEqual(persisted_run_id, "22222222-2222-2222-2222-222222222222")
        self.assertFalse(
            any(
                isinstance(call.args[0], AnalysisExplanationTrace)
                for call in db.add.call_args_list
            )
        )
        db.flush.assert_called()

    def test_write_analysis_normalizes_raw_result_json_before_storage(self):
        from app.persistence import writer
        from app.persistence.models import AnalysisExplanationTrace, AnalysisResultRaw, Player

        db = MagicMock()
        player = SimpleNamespace(season=2026, age_group="U16")

        def get_side_effect(model, key):
            if model is Player:
                return player
            if model is AnalysisExplanationTrace:
                return None
            return None

        db.get.side_effect = get_side_effect
        nested_uuid = uuid.UUID("33333333-3333-3333-3333-333333333333")
        now = datetime(2026, 4, 29, 8, 11, 44, tzinfo=timezone.utc)
        result = {
            "input": {"player_id": "11111111-1111-1111-1111-111111111111", "hand": "R"},
            "video": {"fps": 30.0, "total_frames": 120},
            "events": {"release": {"frame": 42, "captured_at": now}},
            "elbow": {},
            "action": {},
            "risks": [],
            "meta": {"generated_at": now, "trace_id": nested_uuid},
            "deterministic_expert_v1": {
                "knowledge_pack_id": "actionlab_deterministic_expert",
                "knowledge_pack_version": "2026-04-22.v1",
                "selection": {
                    "diagnosis_status": "partial_match",
                    "primary_mechanism_id": "soft_block_with_trunk_carry",
                },
                "archetype_v1": {"id": "soft_block_leakage_bowler"},
            },
        }

        writer.write_analysis(
            result=result,
            db=db,
            run_id="22222222-2222-2222-2222-222222222222",
            season=2026,
            age_group="U16",
        )

        raw_row = next(
            call.args[0]
            for call in db.add.call_args_list
            if isinstance(call.args[0], AnalysisResultRaw)
        )
        self.assertEqual(raw_row.result_json["meta"]["generated_at"], now.isoformat())
        self.assertEqual(raw_row.result_json["meta"]["trace_id"], str(nested_uuid))
        self.assertEqual(
            raw_row.result_json["events"]["release"]["captured_at"],
            now.isoformat(),
        )

    def test_write_analysis_skips_existing_explanation_trace_without_failing_persistence(self):
        from app.persistence import writer
        from app.persistence.models import AnalysisExplanationTrace, Player

        db = MagicMock()
        player = SimpleNamespace(season=2026, age_group="U16")
        existing_trace = SimpleNamespace(run_id=uuid.UUID("22222222-2222-2222-2222-222222222222"))

        def get_side_effect(model, key):
            if model is Player:
                return player
            if model is AnalysisExplanationTrace:
                return existing_trace
            return None

        db.get.side_effect = get_side_effect
        result = {
            "input": {"player_id": "11111111-1111-1111-1111-111111111111", "hand": "R"},
            "video": {"fps": 30.0, "total_frames": 120},
            "events": {},
            "elbow": {},
            "action": {},
            "risks": [],
            "deterministic_expert_v1": {
                "knowledge_pack_id": "actionlab_deterministic_expert",
                "knowledge_pack_version": "2026-04-22.v1",
                "selection": {
                    "diagnosis_status": "partial_match",
                    "primary_mechanism_id": "soft_block_with_trunk_carry",
                },
                "archetype_v1": {"id": "soft_block_leakage_bowler"},
            },
        }

        persisted_run_id = writer.write_analysis(
            result=result,
            db=db,
            run_id="22222222-2222-2222-2222-222222222222",
            season=2026,
            age_group="U16",
        )

        self.assertEqual(persisted_run_id, "22222222-2222-2222-2222-222222222222")
        db.flush.assert_called()

    def test_write_analysis_skips_trace_write_failure_without_failing_persistence(self):
        from app.persistence import writer
        from app.persistence.models import AnalysisExplanationTrace, Player

        db = MagicMock()
        player = SimpleNamespace(season=2026, age_group="U16")

        def get_side_effect(model, key):
            if model is Player:
                return player
            if model is AnalysisExplanationTrace:
                return None
            return None

        db.get.side_effect = get_side_effect
        nested_txn = MagicMock()
        nested_txn.__enter__.return_value = nested_txn
        nested_txn.__exit__.return_value = False
        db.begin_nested.return_value = nested_txn
        db.flush.side_effect = [
            None,
            PermissionError("permission denied for table analysis_explanation_trace"),
            None,
        ]
        result = {
            "input": {"player_id": "11111111-1111-1111-1111-111111111111", "hand": "R"},
            "video": {"fps": 30.0, "total_frames": 120},
            "events": {},
            "elbow": {},
            "action": {},
            "risks": [],
            "deterministic_expert_v1": {
                "knowledge_pack_id": "actionlab_deterministic_expert",
                "knowledge_pack_version": "2026-04-22.v1",
                "selection": {
                    "diagnosis_status": "partial_match",
                    "primary_mechanism_id": "soft_block_with_trunk_carry",
                },
                "archetype_v1": {"id": "soft_block_leakage_bowler"},
            },
        }

        persisted_run_id = writer.write_analysis(
            result=result,
            db=db,
            run_id="22222222-2222-2222-2222-222222222222",
            season=2026,
            age_group="U16",
        )

        self.assertEqual(persisted_run_id, "22222222-2222-2222-2222-222222222222")
        db.begin_nested.assert_called_once()


class TempCleanupTests(unittest.TestCase):
    def test_loader_cleans_up_stale_prefixed_temp_uploads(self):
        from app.io import loader

        temp_dir = tempfile.gettempdir()
        stale_path = os.path.join(temp_dir, f"{loader.TEMP_UPLOAD_PREFIX}stale.mp4")
        with open(stale_path, "wb") as handle:
            handle.write(b"video")

        try:
            old_mtime = datetime.utcnow().timestamp() - (loader.STALE_TEMP_UPLOAD_MAX_AGE_SECONDS + 10)
            os.utime(stale_path, (old_mtime, old_mtime))

            result = loader.cleanup_stale_temp_uploads(
                max_age_seconds=loader.STALE_TEMP_UPLOAD_MAX_AGE_SECONDS,
            )

            self.assertGreaterEqual(result["scanned"], 1)
            self.assertGreaterEqual(result["removed"], 1)
            self.assertFalse(os.path.exists(stale_path))
        finally:
            if os.path.exists(stale_path):
                os.remove(stale_path)

    def test_temp_upload_prefix_is_pid_scoped(self):
        from app.io import loader

        self.assertTrue(loader.TEMP_UPLOAD_PREFIX.startswith(loader.TEMP_UPLOAD_ROOT_PREFIX))
        self.assertIn("_", loader.TEMP_UPLOAD_PREFIX[len(loader.TEMP_UPLOAD_ROOT_PREFIX):])


if __name__ == "__main__":
    unittest.main()
