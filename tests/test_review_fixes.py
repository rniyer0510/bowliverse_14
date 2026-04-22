import importlib
import os
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


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


class AnalyzePersistenceFallbackTests(unittest.TestCase):
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

        db = MagicMock()
        db.get.return_value = SimpleNamespace(season=2026, age_group="U16")
        result = {
            "input": {"player_id": "11111111-1111-1111-1111-111111111111", "hand": "R"},
            "video": {"fps": 30.0, "total_frames": 120},
            "events": {},
            "elbow": {},
            "action": {},
            "risks": [],
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
        db.flush.assert_called()
        db.commit.assert_not_called()
        db.rollback.assert_not_called()


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
