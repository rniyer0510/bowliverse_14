import importlib
import os
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


class VisualGeometryCacheTests(unittest.TestCase):
    def test_subject_geometry_cached_per_video_frame(self):
        from app.workers.risk import visual_utils

        visual_utils._SUBJECT_GEOMETRY_CACHE.clear()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        with patch.object(visual_utils, "_read_frame", return_value=frame), patch(
            "app.workers.risk.visual_utils.os.path.exists",
            return_value=True,
        ), patch.object(
            visual_utils,
            "_estimate_subject_geometry",
            return_value=(80, 60),
        ) as estimate_mock, patch(
            "app.workers.risk.visual_utils.cv2.imwrite",
            return_value=True,
        ):
            result_a = visual_utils.draw_and_save_visual(
                video_path="/tmp/example.mp4",
                frame_idx=12,
                risk_id="front_foot_braking_shock",
                run_id="run-1",
            )
            result_b = visual_utils.draw_and_save_visual(
                video_path="/tmp/example.mp4",
                frame_idx=12,
                risk_id="knee_brace_failure",
                run_id="run-1",
            )

        self.assertIsNotNone(result_a)
        self.assertIsNotNone(result_b)
        self.assertEqual(estimate_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
