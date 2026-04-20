import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from app.workers.render import coach_video_renderer
from app.workers.render.coach_video_renderer import render_skeleton_video, _summary_issue_lines
from app.workers.render.render_load_watch import (
    _load_hotspot_regions,
    _summary_load_watch_text,
    _summary_symptom_text,
)


def _blank_landmarks():
    return [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.0}
        for _ in range(33)
    ]


def _set_point(landmarks, idx, x, y, vis=0.95):
    landmarks[idx] = {
        "x": float(x),
        "y": float(y),
        "z": 0.0,
        "visibility": float(vis),
    }


def _pose_frame(frame_idx: int, shift: float):
    landmarks = _blank_landmarks()
    _set_point(landmarks, 11, 0.42 + shift, 0.28)
    _set_point(landmarks, 12, 0.56 + shift, 0.28)
    _set_point(landmarks, 13, 0.39 + shift, 0.40)
    _set_point(landmarks, 14, 0.59 + shift, 0.40)
    _set_point(landmarks, 15, 0.36 + shift, 0.54)
    _set_point(landmarks, 16, 0.62 + shift, 0.54)
    _set_point(landmarks, 23, 0.45 + shift, 0.50)
    _set_point(landmarks, 24, 0.54 + shift, 0.50)
    _set_point(landmarks, 25, 0.45 + shift, 0.68)
    _set_point(landmarks, 26, 0.56 + shift, 0.68)
    _set_point(landmarks, 27, 0.44 + shift, 0.88)
    _set_point(landmarks, 28, 0.57 + shift, 0.88)
    _set_point(landmarks, 0, 0.49 + shift, 0.16)
    return {"frame": frame_idx, "landmarks": landmarks}


class CoachVideoRendererTest(unittest.TestCase):
    def test_render_skeleton_video_builds_tracks_only_for_render_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")

            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                24.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for _ in range(8):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()

            pose_frames = [_pose_frame(i, shift=0.01 * i) for i in range(8)]
            with mock.patch.object(
                coach_video_renderer,
                "_build_smoothed_tracks",
                wraps=coach_video_renderer._build_smoothed_tracks,
            ) as build_tracks:
                result = render_skeleton_video(
                    video_path=video_path,
                    pose_frames=pose_frames,
                    events={
                        "bfc": {"frame": 2},
                        "ffc": {"frame": 3},
                        "release": {"frame": 5},
                    },
                    output_path=output_path,
                    start_frame=2,
                    end_frame=6,
                    pause_seconds=0.0,
                    end_summary_seconds=0.0,
                )

            self.assertTrue(result["available"])
            self.assertEqual(build_tracks.call_count, 1)
            self.assertEqual(build_tracks.call_args.kwargs["start_frame"], 2)
            self.assertEqual(build_tracks.call_args.kwargs["end_frame"], 6)

    def test_render_skeleton_video_outputs_nonempty_mp4_with_phase_overlay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")

            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                24.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()

            pose_frames = [_pose_frame(i, shift=0.01 * i) for i in range(5)]
            result = render_skeleton_video(
                video_path=video_path,
                pose_frames=pose_frames,
                events={
                    "bfc": {"frame": 1},
                    "ffc": {"frame": 2},
                    "release": {"frame": 4},
                },
                output_path=output_path,
                pause_seconds=0.0,
                end_summary_seconds=0.0,
            )

            self.assertTrue(result["available"])
            self.assertEqual(result["frames_rendered"], 11)
            self.assertEqual(result["style"], "skeleton_phase_v1")
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

            cap = cv2.VideoCapture(output_path)
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertGreater(int(frame.sum()), 0)

    def test_render_skeleton_video_adds_pause_frames_at_landmarks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")

            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()

            pose_frames = [_pose_frame(i, shift=0.01 * i) for i in range(5)]
            result = render_skeleton_video(
                video_path=video_path,
                pose_frames=pose_frames,
                events={
                    "bfc": {"frame": 1},
                    "ffc": {"frame": 2},
                    "release": {"frame": 4},
                },
                output_path=output_path,
                pause_seconds=1.0,
                end_summary_seconds=0.0,
            )

            self.assertTrue(result["available"])
            self.assertEqual(result["pause_seconds"], 1.0)
            self.assertEqual(result["end_frame"], 4)
            self.assertEqual(result["frames_rendered"], 41)

    def test_render_skeleton_video_publishes_stable_output_path_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")

            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                24.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()

            pose_frames = [_pose_frame(i, shift=0.01 * i) for i in range(5)]
            with mock.patch(
                "app.workers.render.coach_video_renderer.shutil.which",
                return_value=None,
            ):
                result = render_skeleton_video(
                    video_path=video_path,
                    pose_frames=pose_frames,
                    events={
                        "bfc": {"frame": 1},
                        "ffc": {"frame": 2},
                        "release": {"frame": 4},
                    },
                    output_path=output_path,
                    pause_seconds=0.0,
                    end_summary_seconds=0.0,
                )

            self.assertTrue(result["available"])
            self.assertEqual(result["encoding"], "mp4v")
            self.assertEqual(result["path"], output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_summary_filters_ffc_risks_when_landing_anchor_is_low_confidence(self):
        lines = _summary_issue_lines(
            {
                "knee_brace_failure": {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 1.0,
                    "confidence": 0.85,
                },
                "foot_line_deviation": {
                    "risk_id": "foot_line_deviation",
                    "signal_strength": 0.9,
                    "confidence": 0.85,
                },
                "lateral_trunk_lean": {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 0.5,
                    "confidence": 0.85,
                },
            },
            events={
                "ffc": {"frame": 10, "confidence": 0.15},
                "event_chain": {"quality": 0.07},
            },
        )

        self.assertNotIn("Front-Leg Support", lines)
        self.assertNotIn("Foot Line", lines)
        self.assertIn("Trunk Lean", lines)

    def test_summary_prefers_story_labels_over_generic_risk_ranking(self):
        lines = _summary_issue_lines(
            {
                "knee_brace_failure": {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 0.9,
                    "confidence": 0.9,
                },
                "lateral_trunk_lean": {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 0.8,
                    "confidence": 0.9,
                },
            },
            events={
                "ffc": {"frame": 10, "confidence": 0.9},
                "event_chain": {"quality": 0.9},
            },
            report_story={
                "theme": "working_pattern",
                "watch_focus": {
                    "key": "upper_body_opening",
                    "label": "Upper Body Opening",
                },
                "key_metrics": [
                    {"key": "upper_body_opening", "label": "Upper Body Opening"},
                    {"key": "trunk_lean", "label": "Trunk Lean"},
                ],
            },
        )

        self.assertEqual(lines[0], "Keep watching Upper Body Opening")
        self.assertIn("Body stays fairly tall", lines)

    def test_summary_load_watch_uses_distinct_body_families(self):
        text = _summary_load_watch_text(
            {
                "knee_brace_failure": {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 0.9,
                    "confidence": 0.9,
                },
                "lateral_trunk_lean": {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 0.8,
                    "confidence": 0.9,
                },
            },
            events={
                "ffc": {"frame": 2, "confidence": 0.9},
                "event_chain": {"quality": 0.9},
            },
        )

        self.assertEqual(text, "Front knee / leg chain\nLower back / side trunk")

    def test_summary_symptom_prefers_release_story_when_present(self):
        text = _summary_symptom_text(
            {
                "lateral_trunk_lean": {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 0.9,
                    "confidence": 0.9,
                }
            },
            report_story={"hero_risk_id": "lateral_trunk_lean"},
            events={"release": {"frame": 4, "confidence": 0.9}},
        )

        self.assertEqual(text, "Body falls away at release")

    def test_load_hotspot_regions_include_leg_and_trunk_targets(self):
        pose_frames = [_pose_frame(i, shift=0.01 * i) for i in range(5)]
        tracks = {
            joint_idx: {"raw": [None] * len(pose_frames)}
            for joint_idx in (11, 12, 23, 24, 25, 26, 27, 28)
        }
        width = 160
        height = 120
        for frame_idx, pose_frame in enumerate(pose_frames):
            for joint_idx in tracks:
                point = pose_frame["landmarks"][joint_idx]
                tracks[joint_idx]["raw"][frame_idx] = (
                    int(round(point["x"] * width)),
                    int(round(point["y"] * height)),
                )

        leg_regions = _load_hotspot_regions(
            tracks=tracks,
            frame_idx=2,
            hand="R",
            risk_id="knee_brace_failure",
        )
        trunk_regions = _load_hotspot_regions(
            tracks=tracks,
            frame_idx=2,
            hand="R",
            risk_id="lateral_trunk_lean",
        )

        self.assertEqual(
            [region["region_key"] for region in leg_regions],
            ["groin", "knee", "shin"],
        )
        self.assertEqual(
            [region["region_key"] for region in trunk_regions],
            ["side_trunk", "lumbar"],
        )


if __name__ == "__main__":
    unittest.main()
