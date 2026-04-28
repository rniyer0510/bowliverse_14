import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.workers.render import coach_video_renderer
from app.workers.render.coach_video_renderer import (
    _draw_phase_anchor_panel,
    _draw_phase_overlay,
    _draw_body_pay_phase,
    _format_action_label,
    _draw_load_watch_phase,
    _phase_leakage_payload,
    _pause_hold_plan,
    _pause_sequence_plan,
    _reading_hold_frames,
    _preferred_hotspot_region_key,
    _root_cause_proof_step,
    _story_headline_and_support,
    _story_risk_for_phase,
    _summary_telemetry_layout,
    _stacked_hotspot_region_keys,
    _should_render_warning_hotspots,
    _select_hotspot_frame_idx,
    _top_risk_panel_metrics,
    render_skeleton_video,
    _render_timeline_events,
    _summary_issue_lines,
)
from app.workers.render.coach_video_renderer_parts.drawing_base import (
    _draw_skeleton_legend,
)
from app.workers.render.coach_video_renderer_parts.bubble_base import (
    _story_card_layout,
)
from app.workers.render.coach_video_renderer_parts.hotspot_draw import (
    _draw_hotspot_compact_label,
)
from app.workers.render.coach_video_renderer_parts.font_utils import (
    _fit_pil_wrapped_text,
)
from app.workers.render.coach_video_renderer_parts.themed_story import (
    _normalized_story_label,
)
from app.workers.render.coach_video_renderer_parts.render_pause_sequence import (
    _write_stage_frames,
)
from app.workers.render.render_load_watch import (
    _load_hotspot_regions,
    _preferred_ffc_cue_risk_id,
    _release_hotspot_risk_id,
    _summary_load_watch_title,
    _summary_load_watch_text,
    _summary_symptom_title,
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
    def test_phase_overlay_preserves_bottom_rail_surface(self):
        frame = np.zeros((200, 160, 3), dtype=np.uint8)

        _draw_phase_overlay(
            frame,
            frame_idx=2,
            start=0,
            stop=10,
            events={
                "bfc": {"frame": 2},
                "ffc": {"frame": 5},
                "release": {"frame": 8},
            },
        )

        rail_slice = frame[170:198, 8:152]
        self.assertGreater(int(rail_slice.sum()), 0)

    def test_phase_anchor_panel_draws_visible_fallback_copy(self):
        frame = np.zeros((200, 160, 3), dtype=np.uint8)

        _draw_phase_anchor_panel(frame, phase_key="ffc")

        self.assertGreater(int(frame.sum()), 0)

    def test_skeleton_legend_draws_intro_overlay(self):
        frame = np.zeros((200, 160, 3), dtype=np.uint8)

        _draw_skeleton_legend(
            frame,
            fps=30.0,
            frame_idx=0,
            legend_end_frame=75,
        )

        self.assertGreater(int(frame.sum()), 0)

    def test_pause_hold_plan_gives_hotspots_extra_read_time(self):
        normal_cue, normal_hotspot = _pause_hold_plan(
            pause_frames=10,
            has_hotspot=False,
        )
        hotspot_cue, hotspot_hold = _pause_hold_plan(
            pause_frames=10,
            has_hotspot=True,
        )

        self.assertEqual((normal_cue, normal_hotspot), (10, 0))
        self.assertLess(hotspot_cue, normal_cue)
        self.assertGreater(hotspot_hold, 5)
        self.assertGreater(hotspot_cue + hotspot_hold, normal_cue)

    def test_pause_sequence_plan_reserves_body_pay_after_break(self):
        sequence = _pause_sequence_plan(
            pause_frames=10,
            has_hotspot=True,
            has_leakage=True,
        )

        self.assertGreater(sequence["proof"], 0)
        self.assertGreater(sequence["leak"], 0)
        self.assertGreater(sequence["pay"], 0)
        self.assertGreater(sequence["hotspot"], 0)
        self.assertEqual(sum(sequence.values()), sum(_pause_hold_plan(pause_frames=10, has_hotspot=True)))

    def test_hotspot_stage_plan_ends_on_compact_label(self):
        stages = coach_video_renderer._hotspot_stage_plan(8)

        self.assertEqual(stages[-1][0], "label")
        self.assertGreater(stages[-1][1], 0)
        self.assertEqual(sum(count for _, count in stages), 8)

    def test_load_watch_label_stage_draws_only_primary_compact_tag(self):
        pose_frames = [_pose_frame(i, shift=0.0) for i in range(5)]
        tracks = coach_video_renderer._build_smoothed_tracks(
            pose_frames,
            width=160,
            height=120,
            fps=24.0,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        with mock.patch(
            "app.workers.render.coach_video_renderer_parts.hotspot_phase._draw_hotspot_compact_label",
            side_effect=[(1, 1, 10, 10)],
        ) as compact_label:
            _draw_load_watch_phase(
                frame,
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
                pulse_phase=0.85,
                stage="label",
            )
        self.assertEqual(compact_label.call_count, 1)

    def test_write_stage_frames_blends_transition_from_previous_stage(self):
        previous = np.zeros((8, 8, 3), dtype=np.uint8)
        current = np.full((8, 8, 3), 240, dtype=np.uint8)
        writer = mock.Mock()

        written, final_frame = _write_stage_frames(
            writer=writer,
            stage_frames=[current.copy() for _ in range(4)],
            previous_frame=previous,
            fps=24.0,
        )

        self.assertEqual(written, 4)
        self.assertIsNotNone(final_frame)
        first_written = writer.write.call_args_list[0].args[0]
        self.assertGreater(int(first_written.sum()), int(previous.sum()))
        self.assertLess(int(first_written.sum()), int(current.sum()))

    def test_reading_hold_frames_gives_short_bubbles_real_read_time(self):
        hold_frames = _reading_hold_frames(
            text="Energy leaks here.",
            fps=24.0,
            minimum_seconds=1.55,
            max_seconds=2.6,
        )

        self.assertGreaterEqual(hold_frames, 37)

    def test_top_risk_panel_metrics_keep_headline_dominant_and_body_secondary(self):
        metrics = _top_risk_panel_metrics(1080, 1920)

        self.assertEqual(metrics["headline_max_lines"], 2)
        self.assertEqual(metrics["body_max_lines"], 1)
        self.assertGreater(metrics["headline_base_scale"], metrics["body_base_scale"])
        self.assertGreater(metrics["card_w"], 600)
        self.assertLess(metrics["card_h"], 320)

    def test_story_card_layout_uses_opposite_lane_from_left_anchor(self):
        layout = _story_card_layout(width=478, height=850, anchor=(110, 420))

        self.assertEqual(layout["side"], "right")
        self.assertGreater(layout["x0"], 300)
        self.assertLess(layout["line_x"], layout["x1"])
        self.assertGreaterEqual(layout["y1"] - layout["y0"], 24)
        self.assertLess(layout["y1"] - layout["y0"], 80)

    def test_story_card_layout_uses_opposite_lane_from_right_anchor(self):
        layout = _story_card_layout(width=478, height=850, anchor=(380, 420))

        self.assertEqual(layout["side"], "left")
        self.assertLess(layout["x1"], 180)
        self.assertGreater(layout["line_x"], layout["x0"])
        self.assertGreaterEqual(layout["y1"] - layout["y0"], 24)
        self.assertLess(layout["y1"] - layout["y0"], 80)

    def test_story_card_layout_uses_right_lane_for_left_arm_without_anchor(self):
        layout = _story_card_layout(width=478, height=850, bowler_hand="L")

        self.assertEqual(layout["side"], "right")
        self.assertGreater(layout["x0"], 300)

    def test_story_card_layout_prefers_hand_hint_over_anchor_side(self):
        layout = _story_card_layout(width=478, height=850, anchor=(110, 420), bowler_hand="R")

        self.assertEqual(layout["side"], "left")

    def test_hotspot_compact_label_prefers_vertical_lane_for_left_side_anchor(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        rect = _draw_hotspot_compact_label(
            frame,
            center=(100, 120),
            direction=(1.0, 0.0),
            label="Knee",
            scale=240,
            occupied_rects=[],
        )

        self.assertIsNotNone(rect)
        _, y0, _, y1 = rect
        self.assertTrue(y1 <= 120 or y0 >= 120)

    def test_hotspot_compact_label_prefers_vertical_lane_for_right_side_anchor(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        rect = _draw_hotspot_compact_label(
            frame,
            center=(220, 120),
            direction=(-1.0, 0.0),
            label="Knee",
            scale=240,
            occupied_rects=[],
        )

        self.assertIsNotNone(rect)
        _, y0, _, y1 = rect
        self.assertTrue(y1 <= 120 or y0 >= 120)

    def test_fit_pil_wrapped_text_avoids_ellipsis_when_font_can_shrink(self):
        image = Image.new("RGBA", (320, 240), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        font, lines = _fit_pil_wrapped_text(
            draw,
            "Watch how the upper body is sequencing as the ball comes out.",
            font_file="Inter-Medium.ttf",
            base_size=22,
            min_size=14,
            max_width=132,
            max_lines=5,
        )

        self.assertIsNotNone(font)
        self.assertTrue(lines)
        self.assertFalse(any(str(line).endswith("...") for line in lines))

    def test_story_label_normalization_collapses_release_duplicate(self):
        self.assertEqual(_normalized_story_label("Release"), _normalized_story_label("Release."))

    def test_summary_telemetry_layout_stays_compact(self):
        layout = _summary_telemetry_layout(1080, 1920)

        self.assertLess(layout["stat_h"], 200)
        self.assertGreater(layout["stat_w"], 250)
        self.assertLess(layout["gap"], 40)

    def test_story_headline_and_support_split_long_copy(self):
        headline, support = _story_headline_and_support(
            "Front leg doesn't hold strong at landing. The body then falls away."
        )

        self.assertEqual(headline, "Front leg doesn't hold strong at landing")
        self.assertEqual(support, "The body then falls away.")

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

    def test_render_timeline_events_preserves_ordered_bfc_even_if_low_confidence(self):
        render_events = _render_timeline_events(
            start=103,
            stop=168,
            events={
                "bfc": {"frame": 133, "confidence": 0.0, "method": "simple_grounded_bfc"},
                "ffc": {"frame": 134, "confidence": 0.67, "method": "release_backward_chain_grounding"},
                "release": {"frame": 140, "confidence": 0.54, "method": "peak_plus_offset_corrected"},
            },
        )

        self.assertEqual((render_events["bfc"] or {}).get("frame"), 133)
        self.assertEqual((render_events["bfc"] or {}).get("method"), "simple_grounded_bfc")
        self.assertEqual((render_events["ffc"] or {}).get("frame"), 134)

    def test_render_timeline_events_falls_back_when_bfc_is_not_before_ffc(self):
        render_events = _render_timeline_events(
            start=103,
            stop=168,
            events={
                "bfc": {"frame": 136, "confidence": 0.9, "method": "simple_grounded_bfc"},
                "ffc": {"frame": 134, "confidence": 0.67, "method": "release_backward_chain_grounding"},
                "release": {"frame": 140, "confidence": 0.54, "method": "peak_plus_offset_corrected"},
            },
        )

        self.assertEqual((render_events["bfc"] or {}).get("method"), "render_phase_fallback")
        self.assertLess(int((render_events["bfc"] or {}).get("frame")), 134)

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
        self.assertIn(selected, {1, 3})
        self.assertLessEqual(selected, 3)

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

    def test_draw_body_pay_phase_accepts_curated_pay_path(self):
        pose_frames = [_pose_frame(i, shift=0.0) for i in range(5)]
        tracks = coach_video_renderer._build_smoothed_tracks(
            pose_frames,
            width=160,
            height=120,
            fps=24.0,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        _draw_body_pay_phase(
            frame,
            tracks=tracks,
            frame_idx=2,
            hand="R",
            risk_id="knee_brace_failure",
            risk_by_id={
                "knee_brace_failure": {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 0.9,
                    "confidence": 0.9,
                },
                "foot_line_deviation": {
                    "risk_id": "foot_line_deviation",
                    "signal_strength": 0.4,
                    "confidence": 0.7,
                },
                "front_foot_braking_shock": {
                    "risk_id": "front_foot_braking_shock",
                    "signal_strength": 0.5,
                    "confidence": 0.8,
                },
            },
            region_priority=["groin", "shin", "knee"],
            progress=0.75,
        )

    def test_phase_leakage_payload_requires_strong_ffc_evidence(self):
        payload = _phase_leakage_payload(
            kinetic_chain={
                "pace_translation": {
                    "transfer_efficiency": 0.44,
                    "leakage_before_block": 0.40,
                    "leakage_at_block": 0.61,
                    "pace_leakage": [{"stage": "front_foot_block", "severity": 0.61}],
                }
            },
            phase_key="ffc",
            risk_id="knee_brace_failure",
            events={
                "ffc": {"frame": 42, "confidence": 0.20, "method": "ultimate_fallback"},
                "event_chain": {"quality": 0.10},
            },
        )

        self.assertIsNone(payload)

    def test_phase_leakage_payload_builds_ffc_transfer_leak_story(self):
        payload = _phase_leakage_payload(
            kinetic_chain={
                "pace_translation": {
                    "transfer_efficiency": 0.47,
                    "leakage_before_block": 0.38,
                    "leakage_at_block": 0.58,
                    "pace_leakage": [{"stage": "front_foot_block", "severity": 0.58}],
                }
            },
            phase_key="ffc",
            risk_id="knee_brace_failure",
            events={
                "ffc": {"frame": 42, "confidence": 0.61, "method": "release_backward_chain_grounding"},
                "event_chain": {"quality": 0.44},
            },
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["title"], "Transfer leak")
        self.assertIn("landing leg", payload["headline"].lower())

    def test_phase_leakage_payload_builds_release_chain_leak_story(self):
        payload = _phase_leakage_payload(
            kinetic_chain={
                "pace_translation": {
                    "transfer_efficiency": 0.49,
                    "leakage_before_block": 0.42,
                    "leakage_at_block": 0.51,
                    "late_arm_chase": 0.62,
                    "dissipation_burden": 0.55,
                }
            },
            phase_key="release",
            risk_id="hip_shoulder_mismatch",
            events={"release": {"frame": 65, "confidence": 0.72}},
        )

        self.assertIsNotNone(payload)
        self.assertIn("hips and shoulders", payload["headline"].lower())

    def test_preferred_hotspot_region_key_uses_curated_region_by_risk(self):
        self.assertEqual(_preferred_hotspot_region_key("knee_brace_failure"), "knee")
        self.assertEqual(_preferred_hotspot_region_key("lateral_trunk_lean"), "side_trunk")

    def test_leg_hotspot_stack_prioritizes_anchor_that_matches_card(self):
        self.assertEqual(
            _stacked_hotspot_region_keys("knee_brace_failure"),
            ["knee", "shin", "groin"],
        )
        self.assertEqual(
            _stacked_hotspot_region_keys("foot_line_deviation"),
            ["shin", "knee", "groin"],
        )

    def test_connected_story_suppresses_warning_hotspots(self):
        self.assertFalse(
            _should_render_warning_hotspots(
                report_story={"theme": "working_pattern"},
                root_cause={"status": "no_clear_problem"},
            )
        )
        self.assertFalse(
            _should_render_warning_hotspots(
                report_story={"theme": "working_pattern"},
                root_cause={
                    "status": "clear",
                    "renderer_guidance": {"warning_hotspots_allowed": True},
                },
            )
        )
        self.assertTrue(
            _should_render_warning_hotspots(
                report_story={"theme": "problem_pattern"},
                root_cause={"status": "clear"},
            )
        )

    def test_prepare_pause_context_does_not_force_ffc_risk_from_raw_weights(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        pose_frames = [_pose_frame(i, shift=0.0) for i in range(5)]
        tracks = coach_video_renderer._build_smoothed_tracks(
            pose_frames,
            width=160,
            height=120,
            fps=24.0,
        )

        context = coach_video_renderer._prepare_pause_context(
            frame=frame,
            pose_frames=pose_frames,
            tracks=tracks,
            frame_idx=2,
            pause_key="ffc",
            hand="R",
            risk_by_id={
                "knee_brace_failure": {
                    "risk_id": "knee_brace_failure",
                    "signal_strength": 0.9,
                    "confidence": 0.9,
                },
                "foot_line_deviation": {
                    "risk_id": "foot_line_deviation",
                    "signal_strength": 0.8,
                    "confidence": 0.8,
                },
            },
            render_events={
                "ffc": {"frame": 2, "confidence": 0.8, "method": "release_backward_chain_grounding"},
                "event_chain": {"quality": 0.7},
            },
            report_story={"theme": "problem_pattern"},
            root_cause={"status": "holdback", "renderer_guidance": {}},
            kinetic_chain=None,
        )

        self.assertIsNone(context["hotspot_payload"])
        self.assertIsNone(context["leakage_payload"])

    def test_positive_story_summary_text_ignores_issue_fallback_copy(self):
        root_cause = {
            "status": "clear",
            "renderer_guidance": {
                "simple_symptom_text": "Front leg gets soft at landing.",
                "simple_load_watch_text": "This is where the body works harder.",
            },
        }
        report_story = {
            "theme": "working_pattern",
            "watch_focus": {"label": "action shape"},
        }

        self.assertEqual(
            _summary_symptom_text(
                {},
                report_story=report_story,
                root_cause=root_cause,
            ),
            "Keep watching action shape",
        )
        self.assertEqual(
            _summary_load_watch_text(
                {},
                report_story=report_story,
                root_cause=root_cause,
            ),
            "No one area is taking too much load.",
        )
        self.assertEqual(
            _summary_load_watch_title(
                report_story=report_story,
                root_cause=root_cause,
            ),
            "Load Stays Shared",
        )

    def test_uninterpretable_summary_text_stays_evidence_limited(self):
        root_cause = {
            "status": "not_interpretable",
            "renderer_guidance": {
                "simple_symptom_text": "Front leg gets soft at landing.",
                "simple_load_watch_text": "This is where the body works harder.",
            },
        }

        self.assertEqual(
            _summary_symptom_title(root_cause=root_cause),
            "Need Clearer Evidence",
        )
        self.assertEqual(
            _summary_symptom_text(
                {
                    "knee_brace_failure": {
                        "risk_id": "knee_brace_failure",
                        "signal_strength": 0.9,
                        "confidence": 0.9,
                    }
                },
                root_cause=root_cause,
            ),
            "This clip does not show the chain clearly enough to call one issue yet.",
        )
        self.assertEqual(
            _summary_load_watch_title(root_cause=root_cause),
            "Load Not Clear Yet",
        )
        self.assertEqual(
            _summary_load_watch_text(
                {
                    "knee_brace_failure": {
                        "risk_id": "knee_brace_failure",
                        "signal_strength": 0.9,
                        "confidence": 0.9,
                    }
                },
                root_cause=root_cause,
            ),
            "Need a clearer release view before calling where load is building.",
        )

    def test_root_cause_phase_target_blocks_report_story_fallback(self):
        self.assertIsNone(
            _story_risk_for_phase(
                {"hero_risk_id": "hip_shoulder_mismatch"},
                phase_key="release",
                events={"release": {"frame": 10, "confidence": 0.8}},
                root_cause={"status": "no_clear_problem", "renderer_guidance": None},
            )
        )
        self.assertEqual(
            _story_risk_for_phase(
                {"hero_risk_id": "hip_shoulder_mismatch"},
                phase_key="ffc",
                events={"ffc": {"frame": 10, "confidence": 0.8}, "event_chain": {"quality": 0.8}},
                root_cause={
                    "status": "clear",
                    "renderer_guidance": {
                        "phase_targets": {
                            "ffc": {"risk_id": "knee_brace_failure"},
                        }
                    },
                },
            ),
            "knee_brace_failure",
        )

    def test_root_cause_proof_step_reads_backend_phase_step(self):
        self.assertEqual(
            _root_cause_proof_step(
                {
                    "renderer_guidance": {
                        "phase_targets": {
                            "ffc": {
                                "proof_step": {
                                    "title": "Where It Starts",
                                    "headline": "Front leg doesn't hold strong at landing.",
                                }
                            }
                        }
                    }
                },
                phase_key="ffc",
            )["title"],
            "Where It Starts",
        )

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
                coach_video_renderer.render_video,
                "_build_smoothed_tracks",
                wraps=coach_video_renderer.render_video._build_smoothed_tracks,
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
            self.assertEqual(result["slow_motion_factor"], 5.0)
            self.assertEqual(result["frames_rendered"], 17)
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
            self.assertEqual(result["slow_motion_factor"], 5.0)
            self.assertEqual(result["frames_rendered"], 47)

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
                coach_video_renderer.render_video,
                "_draw_skeleton",
                side_effect=fake_draw_skeleton,
            ), mock.patch.object(
                coach_video_renderer.render_video,
                "_draw_phase_overlay",
                side_effect=fake_draw_phase_overlay,
            ), mock.patch.object(
                coach_video_renderer.render_video,
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
                coach_video_renderer.render_pause_sequence,
                "_draw_load_watch_phase",
                wraps=coach_video_renderer.render_pause_sequence._draw_load_watch_phase,
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
            self.assertGreater(result["frames_rendered"], 41)

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
                coach_video_renderer.render_video,
                "_draw_skeleton",
                wraps=coach_video_renderer.render_video._draw_skeleton,
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
                "app.workers.render.coach_video_renderer_parts.render_output.shutil.which",
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

    def test_connected_summary_uses_positive_titles_and_copy(self):
        root_cause = {"status": "no_clear_problem"}

        self.assertEqual(
            _summary_symptom_title(root_cause=root_cause),
            "What Is Working",
        )
        self.assertEqual(
            _summary_symptom_text({}, root_cause=root_cause),
            "Action stays connected through landing and release.",
        )
        self.assertEqual(
            _summary_load_watch_title(root_cause=root_cause),
            "Load Stays Shared",
        )
        self.assertEqual(
            _summary_load_watch_text({}, root_cause=root_cause),
            "No one area is taking too much load.",
        )

    def test_positive_theme_without_watch_focus_does_not_claim_working_title(self):
        report_story = {"theme": "working_pattern"}

        self.assertEqual(
            _summary_symptom_title(report_story=report_story),
            "What To Notice",
        )
        self.assertEqual(
            _summary_symptom_text({}, report_story=report_story),
            "Action has a usable base, but one part still needs watching.",
        )

    def test_summary_load_watch_uses_clearer_fallback_when_no_load_label_exists(self):
        self.assertEqual(
            _summary_load_watch_text({}, report_story={"theme": "alignment"}),
            "Need a clearer release view to read load.",
        )

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

    def test_preferred_ffc_cue_risk_id_prefers_root_cause_anchor(self):
        risk_by_id = {
            "knee_brace_failure": {"risk_id": "knee_brace_failure", "signal_strength": 0.4, "confidence": 0.7},
            "foot_line_deviation": {"risk_id": "foot_line_deviation", "signal_strength": 0.8, "confidence": 0.9},
        }
        events = {
            "ffc": {"frame": 10, "confidence": 0.9},
            "event_chain": {"quality": 0.8},
        }

        preferred = _preferred_ffc_cue_risk_id(
            risk_by_id,
            report_story=None,
            events=events,
            root_cause={
                "renderer_guidance": {
                    "anchor_risk_ids": {"ffc": "knee_brace_failure"},
                }
            },
        )

        self.assertEqual(preferred, "knee_brace_failure")

    def test_release_hotspot_risk_id_prefers_root_cause_anchor(self):
        risk_by_id = {
            "lateral_trunk_lean": {"risk_id": "lateral_trunk_lean", "signal_strength": 0.5, "confidence": 0.8},
            "hip_shoulder_mismatch": {"risk_id": "hip_shoulder_mismatch", "signal_strength": 0.9, "confidence": 0.9},
        }

        preferred = _release_hotspot_risk_id(
            risk_by_id,
            events={"release": {"frame": 14, "confidence": 0.9}},
            report_story=None,
            root_cause={
                "renderer_guidance": {
                    "anchor_risk_ids": {"release": "lateral_trunk_lean"},
                }
            },
        )

        self.assertEqual(preferred, "lateral_trunk_lean")

    def test_summary_text_prefers_root_cause_guidance(self):
        risk_by_id = {
            "knee_brace_failure": {"risk_id": "knee_brace_failure", "signal_strength": 0.9, "confidence": 0.9},
            "lateral_trunk_lean": {"risk_id": "lateral_trunk_lean", "signal_strength": 0.7, "confidence": 0.8},
        }
        root_cause = {
            "renderer_guidance": {
                "simple_symptom_text": "Front leg doesn't hold strong at landing, then the body falls away.",
                "simple_load_watch_text": "Front leg works hard.\nLower back works hard too.",
                "symptom_text": "Front leg softens first and trunk carry appears after it.",
                "load_watch_text": "Front knee / leg chain\nLower back / side trunk",
            }
        }

        symptom_text = _summary_symptom_text(
            risk_by_id,
            root_cause=root_cause,
        )
        load_watch_text = _summary_load_watch_text(
            risk_by_id,
            root_cause=root_cause,
        )

        self.assertEqual(
            symptom_text,
            "Front leg doesn't hold strong at landing, then the body falls away.",
        )
        self.assertEqual(
            load_watch_text,
            "Front leg works hard.\nLower back works hard too.",
        )

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
