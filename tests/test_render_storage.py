import importlib
import os
import tempfile
import unittest
from unittest import mock

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from app.workers.render import render_storage


class RenderStorageHelpersTest(unittest.TestCase):
    def test_upload_render_artifact_without_bucket_keeps_local_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.mp4")
            with open(path, "wb") as handle:
                handle.write(b"video")

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ACTIONLAB_RENDER_BUCKET", None)
                result = render_storage.upload_render_artifact(path)

            self.assertFalse(result["uploaded"])
            self.assertEqual(result["storage_backend"], "local")
            self.assertEqual(result["reason"], "render_bucket_not_configured")


class WalkthroughRenderRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("ACTIONLAB_SECRET", "test-secret")
        os.environ["ACTIONLAB_AUTO_CREATE_SCHEMA"] = "false"
        module = importlib.import_module("app.orchestrator.orchestrator")
        cls.module = module

    def test_get_walkthrough_render_serves_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "local.mp4")
            payload = b"local-video"
            with open(target, "wb") as handle:
                handle.write(payload)

            with mock.patch.object(self.module, "RENDERS_DIR", tmpdir), mock.patch.object(
                self.module,
                "download_render_artifact",
                return_value=None,
            ) as download_render:
                response = self.module.get_walkthrough_render("local.mp4")

            self.assertIsInstance(response, FileResponse)
            self.assertEqual(response.path, target)
            self.assertEqual(response.media_type, "video/mp4")
            download_render.assert_called_once_with("local.mp4")

    def test_get_walkthrough_render_prefers_remote_bytes_over_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "remote.mp4")
            with open(target, "wb") as handle:
                handle.write(b"stale-local-video")

            payload = b"fresh-remote-video"
            with mock.patch.object(self.module, "RENDERS_DIR", tmpdir), mock.patch.object(
                self.module,
                "download_render_artifact",
                return_value=payload,
            ) as download_render:
                response = self.module.get_walkthrough_render("remote.mp4")

            self.assertIsInstance(response, Response)
            self.assertEqual(response.body, payload)
            self.assertEqual(response.media_type, "video/mp4")
            download_render.assert_called_once_with("remote.mp4")

    def test_get_walkthrough_render_falls_back_to_remote_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = b"remote-video"
            with mock.patch.object(self.module, "RENDERS_DIR", tmpdir), mock.patch.object(
                self.module,
                "download_render_artifact",
                return_value=payload,
            ) as download_render:
                response = self.module.get_walkthrough_render("remote.mp4")

            self.assertIsInstance(response, Response)
            self.assertEqual(response.body, payload)
            self.assertEqual(response.media_type, "video/mp4")
            download_render.assert_called_once_with("remote.mp4")

    def test_get_walkthrough_render_rejects_invalid_names(self):
        with self.assertRaises(HTTPException) as ctx:
            self.module.get_walkthrough_render("../bad.mp4")

        self.assertEqual(ctx.exception.status_code, 404)

    def test_load_recent_expert_history_queries_only_required_fields(self):
        expected_rows = [
            ("run-1", "2026-04-24T09:00:00Z", {"frontend_surface_v1": {}}),
            ("run-2", "2026-04-23T09:00:00Z", None),
        ]

        class _FakeQuery:
            def __init__(self, rows):
                self._rows = rows

            def join(self, *args, **kwargs):
                return self

            def filter(self, *args, **kwargs):
                return self

            def order_by(self, *args, **kwargs):
                return self

            def limit(self, *args, **kwargs):
                return self

            def all(self):
                return list(self._rows)

        class _FakeDb:
            def __init__(self, rows):
                self.rows = rows
                self.query_args = None

            def query(self, *args):
                self.query_args = args
                return _FakeQuery(self.rows)

        fake_db = _FakeDb(expected_rows)

        history = self.module._load_recent_expert_history(
            db=fake_db,
            player_id="player-1",
            limit=5,
        )

        self.assertEqual(len(fake_db.query_args), 3)
        self.assertIs(fake_db.query_args[0], self.module.AnalysisRun.run_id)
        self.assertIs(fake_db.query_args[1], self.module.AnalysisRun.created_at)
        self.assertIs(fake_db.query_args[2], self.module.AnalysisResultRaw.result_json)
        self.assertEqual(
            history,
            [
                {
                    "run_id": "run-1",
                    "created_at": "2026-04-24T09:00:00Z",
                    "result_json": {"frontend_surface_v1": {}},
                }
            ],
        )

    def test_build_walkthrough_render_keeps_url_shape_and_records_upload_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            with open(video_path, "wb") as handle:
                handle.write(b"source-video")

            def _fake_render(**kwargs):
                output_path = kwargs["output_path"]
                with open(output_path, "wb") as handle:
                    handle.write(b"rendered")
                return {
                    "available": True,
                    "path": output_path,
                    "frames_rendered": 10,
                }

            with mock.patch.object(self.module, "RENDERS_DIR", tmpdir), mock.patch.object(
                self.module,
                "render_skeleton_video",
                side_effect=_fake_render,
            ), mock.patch.object(
                self.module,
                "upload_render_artifact",
                return_value={
                    "uploaded": True,
                    "storage_backend": "gcs",
                    "bucket": "test-bucket",
                    "object_name": "walkthrough-renders/run-1_walkthrough.mp4",
                },
            ) as upload_render, mock.patch.dict(
                os.environ,
                {"ACTIONLAB_PUBLIC_BASE_URL": "https://api.actionlabcricket.in"},
                clear=False,
            ):
                result = self.module._build_walkthrough_render(
                    run_id="run-1",
                    video={"path": video_path, "total_frames": 20},
                    pose_frames=[],
                    events={"release": {"frame": 10}},
                    hand="right",
                    action={},
                    elbow={},
                    risks=[],
                    estimated_release_speed={},
                    report_story=None,
                    root_cause=None,
                )

            self.assertTrue(result["available"])
            self.assertEqual(result["relative_url"], "/renders/run-1_walkthrough.mp4")
            self.assertEqual(
                result["url"],
                "https://api.actionlabcricket.in/renders/run-1_walkthrough.mp4",
            )
            self.assertEqual(
                result["renderer_version"],
                self.module.WALKTHROUGH_RENDERER_VERSION,
            )
            self.assertEqual(result["storage_backend"], "gcs")
            self.assertTrue(result["storage_uploaded"])
            self.assertEqual(result["storage_bucket"], "test-bucket")
            self.assertEqual(
                result["storage_object"],
                "walkthrough-renders/run-1_walkthrough.mp4",
            )
            upload_render.assert_called_once_with(
                os.path.join(tmpdir, "run-1_walkthrough.mp4"),
                artifact_name="run-1_walkthrough.mp4",
            )


if __name__ == "__main__":
    unittest.main()
