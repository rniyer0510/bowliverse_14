"""Coach-style render helpers."""

import shutil

from .coach_video_renderer_parts import render_pause_sequence, render_video
from .coach_video_renderer_parts.font_utils import _theme_font_dirs, _load_theme_font, _pil_text_size, _wrap_pil_text, _fit_pil_wrapped_text, _pil_text_block_height
from .coach_video_renderer_parts.analytics import _safe_float, _safe_int, _risk_lookup, _risk_weight, _event_confidence, _event_chain_quality, _supports_ffc_story, _speed_display_text, _risk_supported_for_phase
from .coach_video_renderer_parts.story_logic import _story_feature_labels, _positive_recap_lines, _story_risk_for_phase, _format_action_label
from .coach_video_renderer_parts.kinetic_story import _kinetic_pace_translation, _pace_leakage_stage, _phase_leakage_payload
from .coach_video_renderer_parts.text_layout import _wrap_text_lines, _fit_wrapped_text
from .coach_video_renderer_parts.joints import _front_leg_joints, _foot_indices
from .coach_video_renderer_parts.render_output import _make_output_path, _intermediate_render_path, _publish_fallback_render, _finalize_render_video
from .coach_video_renderer_parts.tracks import _point_from_landmarks, _smooth_series, _build_smoothed_tracks, _track_point, _frame_point
from .coach_video_renderer_parts.drawing_base import _draw_joint, _draw_skeleton, _overlay_panel, _apply_bottom_scrim
from .coach_video_renderer_parts.timeline_events import _phase_cut_points, _event_method, _tracked_joint_quality, _safe_landmark_value, _should_draw_skeleton_frame, _render_timeline_events
from .coach_video_renderer_parts.phase_rail import _phase_index_for_frame, _draw_phase_rail, _draw_phase_overlay
from .coach_video_renderer_parts.misc_helpers import _top_risk_panel_metrics, _summary_telemetry_layout, _story_headline_and_support
from .coach_video_renderer_parts.pil_context import _bgr_to_rgb, _frame_draw_context, _commit_frame_draw_context
from .coach_video_renderer_parts.bubble_base import _bubble_copy, _reading_hold_frames, _draw_pointer_bubble, _draw_top_risk_panel
from .coach_video_renderer_parts.anchor_panels import _draw_phase_anchor_panel
from .coach_video_renderer_parts.leg_callouts import _front_leg_support_caption, _draw_front_leg_support_callout, _draw_foot_line_overlay
from .coach_video_renderer_parts.trunk_callout import _trunk_lean_caption, _draw_trunk_lean_callout
from .coach_video_renderer_parts.geometry_helpers import _midpoint, _point_between, _partial_polyline, _draw_partial_polyline
from .coach_video_renderer_parts.hip_callout import _hip_shoulder_caption, _draw_hip_shoulder_callout
from .coach_video_renderer_parts.release_callout import _draw_release_callout
from .coach_video_renderer_parts.transfer_core import _draw_transfer_leak_particles, _transfer_leak_geometry, _draw_transfer_break_phase
from .coach_video_renderer_parts.body_pay import _body_pay_region_priority, _select_body_pay_region, _draw_body_pay_phase
from .coach_video_renderer_parts.transfer_phase import _draw_transfer_leak_phase
from .coach_video_renderer_parts.telemetry_chips import _draw_speed_chip, _draw_action_chip, _draw_legality_chip
from .coach_video_renderer_parts.summary_lines import _summary_issue_lines
from .coach_video_renderer_parts.themed_card_shell import _draw_themed_card_shell, _draw_themed_summary_card
from .coach_video_renderer_parts.themed_story import _draw_themed_story_card
from .coach_video_renderer_parts.themed_telemetry import _draw_themed_telemetry_pill, _draw_themed_stat_card
from .coach_video_renderer_parts.story_overlay import _draw_story_overlay_card
from .coach_video_renderer_parts.summary_legacy import _draw_end_summary_legacy, _draw_end_summary
from .coach_video_renderer_parts.hotspot_draw import _draw_hotspot_marker, _draw_hotspot_pointer_line, _draw_hotspot_compact_label
from .coach_video_renderer_parts.hotspot_support import _preferred_hotspot_region_key, _stacked_hotspot_region_keys, _load_watch_support_text, _should_render_warning_hotspots, _root_cause_phase_target, _root_cause_proof_step, _draw_load_watch_card
from .coach_video_renderer_parts.hotspot_phase import _draw_load_watch_phase
from .coach_video_renderer_parts.pause_logic import _hotspot_stage_plan, _hotspot_search_window, _select_hotspot_frame_idx, _pause_anchor_frames, _pause_hold_plan, _pause_sequence_plan, _proof_bubble_text_for_phase
from .coach_video_renderer_parts.render_pause_payloads import _prepare_pause_context
from .coach_video_renderer_parts.render_pause_sequence import _render_pause_sequence
from .coach_video_renderer_parts.render_video import render_skeleton_video
from .coach_video_renderer_parts.shared import RENDER_DIR

__all__ = [
    "RENDER_DIR",
    "_theme_font_dirs",
    "_load_theme_font",
    "_pil_text_size",
    "_wrap_pil_text",
    "_fit_pil_wrapped_text",
    "_pil_text_block_height",
    "_safe_float",
    "_safe_int",
    "_risk_lookup",
    "_risk_weight",
    "_event_confidence",
    "_event_chain_quality",
    "_supports_ffc_story",
    "_speed_display_text",
    "_risk_supported_for_phase",
    "_story_feature_labels",
    "_positive_recap_lines",
    "_story_risk_for_phase",
    "_format_action_label",
    "_kinetic_pace_translation",
    "_pace_leakage_stage",
    "_phase_leakage_payload",
    "_wrap_text_lines",
    "_fit_wrapped_text",
    "_front_leg_joints",
    "_foot_indices",
    "_make_output_path",
    "_intermediate_render_path",
    "_publish_fallback_render",
    "_finalize_render_video",
    "_point_from_landmarks",
    "_smooth_series",
    "_build_smoothed_tracks",
    "_track_point",
    "_frame_point",
    "_draw_joint",
    "_draw_skeleton",
    "_overlay_panel",
    "_apply_bottom_scrim",
    "_phase_cut_points",
    "_event_method",
    "_tracked_joint_quality",
    "_safe_landmark_value",
    "_should_draw_skeleton_frame",
    "_render_timeline_events",
    "_phase_index_for_frame",
    "_draw_phase_rail",
    "_draw_phase_overlay",
    "_top_risk_panel_metrics",
    "_summary_telemetry_layout",
    "_story_headline_and_support",
    "_bgr_to_rgb",
    "_frame_draw_context",
    "_commit_frame_draw_context",
    "_bubble_copy",
    "_reading_hold_frames",
    "_draw_pointer_bubble",
    "_draw_top_risk_panel",
    "_draw_phase_anchor_panel",
    "_front_leg_support_caption",
    "_draw_front_leg_support_callout",
    "_draw_foot_line_overlay",
    "_trunk_lean_caption",
    "_draw_trunk_lean_callout",
    "_midpoint",
    "_point_between",
    "_partial_polyline",
    "_draw_partial_polyline",
    "_hip_shoulder_caption",
    "_draw_hip_shoulder_callout",
    "_draw_release_callout",
    "_draw_transfer_leak_particles",
    "_transfer_leak_geometry",
    "_draw_transfer_break_phase",
    "_body_pay_region_priority",
    "_select_body_pay_region",
    "_draw_body_pay_phase",
    "_draw_transfer_leak_phase",
    "_draw_speed_chip",
    "_draw_action_chip",
    "_draw_legality_chip",
    "_summary_issue_lines",
    "_draw_themed_card_shell",
    "_draw_themed_summary_card",
    "_draw_themed_story_card",
    "_draw_themed_telemetry_pill",
    "_draw_themed_stat_card",
    "_draw_story_overlay_card",
    "_draw_end_summary_legacy",
    "_draw_end_summary",
    "_draw_hotspot_marker",
    "_draw_hotspot_pointer_line",
    "_draw_hotspot_compact_label",
    "_preferred_hotspot_region_key",
    "_stacked_hotspot_region_keys",
    "_load_watch_support_text",
    "_should_render_warning_hotspots",
    "_root_cause_phase_target",
    "_root_cause_proof_step",
    "_draw_load_watch_card",
    "_draw_load_watch_phase",
    "_hotspot_stage_plan",
    "_hotspot_search_window",
    "_select_hotspot_frame_idx",
    "_pause_anchor_frames",
    "_pause_hold_plan",
    "_pause_sequence_plan",
    "_proof_bubble_text_for_phase",
    "_prepare_pause_context",
    "_render_pause_sequence",
    "render_skeleton_video",
]
