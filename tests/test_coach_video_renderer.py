import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from app.workers.render import coach_video_renderer
from app.workers.render.coach_video_renderer import (
    _format_action_label,
    _draw_load_watch_phase,
    _preferred_hotspot_region_key,
    _select_hotspot_frame_idx,
    render_skeleton_video,
    _render_timeline_events,
    _summary_issue_lines,
)
from app.workers.render.render_load_watch import (
    _load_hotspot_regions,
    _preferred_ffc_cue_risk_id,
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
    def test_format_action_label_prefers_intent_when_present(self):
        self.assertEqual(
            _format_action_label({"action": "FRONT_ON", "intent": "semi_open"}),
            "Semi Open",
        )

    def test_render_timeline_events_falls_back_from_weak_ffc_near_release(self):
        render_events = _render_timeline_events(
            start=400,
            stop=520,
            events={
                "bfc": {"frame": 488, "confidence": 0.15, "method": "ultimate_fallback"},
                "ffc": {"frame": 490, "confidence": 0.15, "method": "ultimate_fallback"},
                "release": {"frame": 492, "confidence": 0.75, "method": "velocity_drop_20pct"},
            },
        )

        self.assertEqual((render_events["bfc"] or {}).get("method"), "render_phase_fallback")
        self.assertEqual((render_events["ffc"] or {}).get("method"), "render_phase_fallback")
        self.assertLess(int((render_events["bfc"] or {}).get("frame")), int((render_events["ffc"] or {}).get("frame")))
        self.assertLess(int((render_events["ffc"] or {}).get("frame")), 492)
        self.assertGreaterEqual(int((render_events["ffc"] or {}).get("frame")), 460)

    def test_preferred_ffc_cue_risk_id_allows_render_phase_fallback_story(self):
        risk_by_id = {
            "knee_brace_failure": {"risk_id": "knee_brace_failure", "signal_strength": 0.8, "confidence": 0.8},
            "foot_line_deviation": {"risk_id": "foot_line_deviation", "signal_strength": 0.3, "confidence": 0.7},
        }
        events = {
            "ffc": {"frame": 468, "confidence": 0.40, "method": "render_phase_fallback"},
            "event_chain": {"quality": 0.05},
        }

        self.assertEqual(
            _preferred_ffc_cue_risk_id(risk_by_id, report_story=None, events=events),
            "knee_brace_failure",
        )

    def test_select_hotspot_frame_idx_prefers_visible_leg_stack_near_anchor(self):
        pose_frames = [_pose_frame(i, shift=0.0) for i in range(8)]
        for idx in (23, 25, 27):
            pose_frames[2]["landmarks"][idx]["visibility"] = 0.0
        tracks = coach_video_renderer._build_smoothed_tracks(
            pose_frames,
            width=160,
            height=120,
            fps=24.0,
        )

        selected = _select_hotspot_frame_idx(
            tracks=tracks,
            hand="R",
            risk_id="knee_brace_failure",
            risk_by_id={
                "knee_brace_failure": {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 0.9,
                    "confidence": 0.9,
                }
            },
            phase_key="ffc",
            anchor_frame=2,
            start=0,
            stop=len(pose_frames),
        )

        self.assertNotEqual(selected, 2)
        self.assertIn(selected, {1, 3, 4})

    def test_draw_load_watch_phase_accepts_all_hotspot_stages(self):
        pose_frames = [_pose_frame(i, shift=0.0) for i in range(5)]
        tracks = coach_video_renderer._build_smoothed_tracks(
            pose_frames,
            width=160,
            height=120,
            fps=24.0,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for stage in ("line", "rings", "label"):
            _draw_load_watch_phase(
                frame.copy(),
                tracks=tracks,
                frame_idx=2,
                hand="R",
                risk_id="knee_brace_failure",
                risk_by_id={
                    "knee_brace_failure": {
                        "risk_id": "knee_brace_failure",
                        "signal_strength": 0.9,
                        "confidence": 0.9,
                    }
                },
                load_watch_text="Front knee / leg chain",
                pulse_phase=0.5,
                stage=stage,
            )

    def test_preferred_hotspot_region_key_uses_curated_region_by_risk(self):
        self.assertEqual(_preferred_hotspot_region_key("knee_brace_failure"), "knee")
        self.assertEqual(_preferred_hotspot_region_key("lateral_trunk_lean"), "side_trunk")

    def test_render_skeleton_video_builds_tracks_for_full_pose_sequence(self):
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
            self.assertNotIn("start_frame", build_tracks.call_args.kwargs)
            self.assertNotIn("end_frame", build_tracks.call_args.kwargs)

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

    def test_render_skeleton_video_uses_clean_raw_frame_for_end_summary(self):
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
            captured_summary_inputs = []

            def fake_draw_skeleton(frame, tracks, frame_idx):
                frame[:, :] = (255, 0, 255)

            def fake_draw_phase_overlay(frame, *, frame_idx, start, stop, events):
                frame[0:10, 0:10] = (0, 255, 255)

            def capture_summary(frame, **kwargs):
                captured_summary_inputs.append(frame.copy())

            with mock.patch.object(
                coach_video_renderer,
                "_draw_skeleton",
                side_effect=fake_draw_skeleton,
            ), mock.patch.object(
                coach_video_renderer,
                "_draw_phase_overlay",
                side_effect=fake_draw_phase_overlay,
            ), mock.patch.object(
                coach_video_renderer,
                "_draw_end_summary",
                side_effect=capture_summary,
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
                    end_summary_seconds=1.0,
                )

            self.assertTrue(result["available"])
            self.assertEqual(len(captured_summary_inputs), 1)
            summary_input = captured_summary_inputs[0]
            self.assertFalse(np.all(summary_input == np.array([255, 0, 255], dtype=np.uint8)))
            self.assertFalse(np.all(summary_input[0:10, 0:10] == np.array([0, 255, 255], dtype=np.uint8)))

    def test_render_skeleton_video_reuses_cached_hotspot_stage_frames(self):
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
            with mock.patch.object(
                coach_video_renderer,
                "_draw_load_watch_phase",
                wraps=coach_video_renderer._draw_load_watch_phase,
            ) as draw_phase:
                result = render_skeleton_video(
                    video_path=video_path,
                    pose_frames=pose_frames,
                    events={
                        "bfc": {"frame": 1, "confidence": 0.9},
                        "ffc": {"frame": 2, "confidence": 0.9},
                        "release": {"frame": 4, "confidence": 0.9},
                        "event_chain": {"quality": 0.9},
                    },
                    risks=[
                        {
                            "risk_id": "knee_brace_failure",
                            "signal_strength": 0.9,
                            "confidence": 0.9,
                        }
                    ],
                    output_path=output_path,
                    pause_seconds=1.0,
                    end_summary_seconds=0.0,
                )

            self.assertTrue(result["available"])
            self.assertGreaterEqual(draw_phase.call_count, 1)
            self.assertLessEqual(draw_phase.call_count, 3)

    def test_render_skeleton_video_stops_drawing_skeleton_late_after_release(self):
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
            for _ in range(10):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()

            pose_frames = [_pose_frame(i, shift=0.01 * i) for i in range(10)]
            with mock.patch.object(
                coach_video_renderer,
                "_draw_skeleton",
                wraps=coach_video_renderer._draw_skeleton,
            ) as draw_skeleton:
                result = render_skeleton_video(
                    video_path=video_path,
                    pose_frames=pose_frames,
                    events={
                        "bfc": {"frame": 0},
                        "ffc": {"frame": 1},
                        "release": {"frame": 2},
                    },
                    output_path=output_path,
                    pause_seconds=0.0,
                    end_summary_seconds=0.0,
                )

            self.assertTrue(result["available"])
            self.assertLess(draw_skeleton.call_count, len(pose_frames))

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

    def test_summary_load_watch_keeps_leg_family_when_ffc_story_is_weak(self):
        text = _summary_load_watch_text(
            {
                "knee_brace_failure": {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 0.95,
                    "confidence": 0.95,
                },
                "lateral_trunk_lean": {
                    "risk_id": "lateral_trunk_lean",
                    "signal_strength": 0.7,
                    "confidence": 0.9,
                },
            },
            events={
                "ffc": {
                    "frame": 2,
                    "confidence": 0.2,
                    "timing_flag": "early_relative_to_release",
                },
                "event_chain": {"quality": 0.1},
                "release": {"frame": 4, "confidence": 0.9},
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
            ["upper_trunk", "side_trunk", "lumbar"],
        )


if __name__ == "__main__":
    unittest.main()
