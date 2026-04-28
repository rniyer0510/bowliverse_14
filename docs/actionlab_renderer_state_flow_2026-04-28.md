# ActionLab Renderer State Flow

Date: 2026-04-28  
Branch baseline: `codex/renderer_new_design`  
Intent: define the complete walkthrough render flow before visual implementation

## Purpose

This document maps:

1. the current renderer control path
2. the target visual behavior from the constrained renderer spec
3. the exact states we should preserve vs. restyle

This is a renderer-flow reference, not a product-vision doc. It assumes:

- evidence gates stay unchanged
- no new inputs are added to `render_skeleton_video()`
- no player-side interactivity is added
- the constrained spec remains the source of truth for visual changes

## Source Files

- [render_video.py](/Users/rniyer/bowliverse_14/app/workers/render/coach_video_renderer_parts/render_video.py:15)
- [render_pause_payloads.py](/Users/rniyer/bowliverse_14/app/workers/render/coach_video_renderer_parts/render_pause_payloads.py:22)
- [render_pause_sequence.py](/Users/rniyer/bowliverse_14/app/workers/render/coach_video_renderer_parts/render_pause_sequence.py:10)
- [pause_logic.py](/Users/rniyer/bowliverse_14/app/workers/render/coach_video_renderer_parts/pause_logic.py:94)
- [phase_rail.py](/Users/rniyer/bowliverse_14/app/workers/render/coach_video_renderer_parts/phase_rail.py:20)
- [anchor_panels.py](/Users/rniyer/bowliverse_14/app/workers/render/coach_video_renderer_parts/anchor_panels.py:5)
- [timeline_events.py](/Users/rniyer/bowliverse_14/app/workers/render/coach_video_renderer_parts/timeline_events.py:70)

## Non-Negotiables

- `pose` is the observation layer
- `kinematics` is the truth layer
- `heuristics` come last
- proof visuals stay anchored to trustworthy frames or proof windows
- fallback text may orient the viewer but must not fabricate mechanism

## Current Control Path

### 1. Render Window Setup

The renderer:

- opens the source video
- trims to `start_frame -> end_frame`
- builds smoothed skeleton tracks
- derives normalized render events via `_render_timeline_events()`
- computes pause anchors via `_pause_anchor_frames()`
- computes one slow-motion window from `ffc -> release`

Important behavior:

- `BFC`, `FFC`, and `Release` can all become pause anchors
- slow motion is only applied between `FFC` and `Release`
- the phase rail is drawn on every frame

### 2. Per-Frame Base Pass

For each frame in the render window:

- draw skeleton if track quality passes `_should_draw_skeleton_frame()`
- draw phase rail via `_draw_phase_overlay()`
- write the frame once
- if inside the `FFC -> Release` slow-motion window, write repeated copies of the same frame
- if the frame is a pause anchor, render the pause sequence after that

This means the viewer sees:

- normal playback before `FFC`
- slowed playback from `FFC` through `Release`
- hard pause sequences at anchor frames

### 3. Pause Payload Preparation

At an anchor frame, `_prepare_pause_context()` decides what kind of paused state exists.

#### BFC

Always:

- draw anchor fallback panel only
- no leakage payload
- no hotspot payload

#### FFC

If `allow_warning_hotspots` is false:

- draw anchor fallback panel only

If `allow_warning_hotspots` is true but `_supports_ffc_story()` is false:

- draw anchor fallback panel only

If `allow_warning_hotspots` is true and `_supports_ffc_story()` is true:

- choose preferred `FFC` proof risk
- draw proof visualization for that risk
- optionally create `hotspot_payload`
- optionally create `leakage_payload` from `kinetic_chain`

So `FFC` is the main proof anchor, but only when evidence allows it.

#### Release

If `allow_warning_hotspots` is false:

- draw anchor fallback panel only

If `allow_warning_hotspots` is true:

- draw release proof visualization
- optionally create `hotspot_payload`
- optionally create `leakage_payload`
- if no hotspot risk exists but proof step exists, show proof bubble/panel

### 4. Pause Sequence Execution

Once the pause payload exists, `_render_pause_sequence()` renders up to four sequential sub-states:

1. `proof`
2. `leak`
3. `body pay`
4. `hotspot`

If no `leakage_payload` exists:

- only `proof` and optional `hotspot` run

If no `hotspot_payload` exists:

- no `body pay`
- no `hotspot`

If no proof bubble text exists:

- `proof_hold` remains the frame budget from `_pause_sequence_plan()`

### 5. End Summary

After the main render window:

- if `end_summary_seconds > 0`
- draw final summary card on the last clean raw frame
- hold it for the configured duration

This is legacy summary behavior and is out of scope for the current visual spec except where typography overlap is explicitly called out.

## Current State Machine

Use these state IDs for implementation planning.

### S0 Intro Playback

Trigger:

- first rendered frames after `start_frame`

Current output:

- skeleton if visible
- phase rail
- no legend

Target output:

- skeleton if visible
- phase rail
- intro legend in top-left for first `2.5s`, fading over last `0.5s`

### S1 Normal Playback Before BFC

Trigger:

- `start_frame -> BFC - 1`

Current output:

- skeleton
- phase rail
- no callout

Target output:

- same logic
- improved phase rail typography only

### S2 BFC Anchor Pause

Trigger:

- `frame_idx == BFC anchor`

Current output:

- anchor fallback panel
- generic wording

Target output:

- same trigger
- same fallback-only behavior
- updated neutral wording
- darker, more readable bubble treatment if bubble style applies here

### S3 Post-BFC Playback

Trigger:

- frames after `BFC` until `FFC`

Current output:

- normal playback
- phase rail

Target output:

- same logic
- improved phase rail only

### S4 FFC Base Frame

Trigger:

- `frame_idx == FFC anchor`

Current output:

- skeleton
- phase rail
- frame written once

Target output:

- same logic
- plus intro legend if still within legend window

### S5 FFC Slow-Motion Duplication

Trigger:

- any frame from `FFC -> Release`

Current output:

- duplicated frame writes according to `slow_motion_factor`

Target output:

- same slow-motion window
- slower effective pacing after constant tuning

Important:

- no new window logic should be introduced
- only pacing constants change

### S6 FFC Proof Pause

Trigger:

- `pause_key == "ffc"`
- evidence gates allow `FFC` story

Current output:

- risk-specific proof visual
- proof bubble text may increase hold duration

Target output:

- same gating
- same proof-stage semantics
- darker, more readable proof bubble
- better phase rail text still visible below

### S7 FFC Fallback Pause

Trigger:

- `pause_key == "ffc"`
- evidence gates do not allow `FFC` story

Current output:

- anchor fallback panel with generic language

Target output:

- same gating
- updated neutral copy:
  - title/headline remain orienting
  - no mechanism language

### S8 Transfer Leak Stage

Trigger:

- `leakage_payload is not None`

Current output:

- transfer leak animation runs after proof

Target output:

- same gating
- same sequence position
- no logic change
- better surrounding readability only

### S9 Body Pay Stage

Trigger:

- `leakage_payload and hotspot_payload`

Current output:

- body-pay stage runs after leak and before hotspot

Target output:

- same gating
- same sequence position
- no new explanatory state added

### S10 Hotspot Stage

Trigger:

- `hotspot_payload is not None`

Current output:

- stage plan uses `line` then `rings`
- compact label uses tiny OpenCV text

Target output:

- same stage order
- same hotspot selection logic
- compact label restyled with PIL typography

### S11 Release Proof Pause

Trigger:

- `pause_key == "release"`
- warning hotspots allowed

Current output:

- release proof visualization
- optional hotspot payload
- optional leakage payload

Target output:

- same gating
- same sequencing
- better bubble contrast and typography

### S12 Release Fallback Pause

Trigger:

- `pause_key == "release"`
- no release hotspot risk and only fallback proof guidance available

Current output:

- generic release anchor panel

Target output:

- same logic
- revised neutral wording

### S13 Post-Release Tail

Trigger:

- short period after release while post-release track quality remains good

Current output:

- skeleton may persist briefly after release

Target output:

- unchanged

### S14 End Summary

Trigger:

- final frame hold after render window

Current output:

- legacy end summary typography

Target output:

- unchanged for now
- explicitly outside current spec except for telemetry font consistency if later expanded

## Target Visual Changes By State

These are renderer-safe changes from the constrained spec.

### All States

- preserve skeleton draw logic
- preserve event fallback logic
- preserve pause-anchor logic
- preserve proof/leak/pay/hotspot order

### States With Always-Visible UI

- `S0-S13`: phase rail typography and active-label treatment improve
- `S0-S13`: telemetry chips improve when present

### States With Callout Bubbles

- `S6`, `S7`, `S11`, `S12`:
  - dark bubble fill
  - white text
  - no drop shadow
  - accent bar remains

### Fallback-Only States

- `S2`, `S7`, `S12`:
  - wording becomes observational and neutral
  - no mechanism claim is introduced

### Intro State

- `S0` only:
  - legend appears once
  - never reappears later

## Implementation Order

Build in this order:

1. typography surfaces
2. bubble contrast
3. anchor fallback copy
4. intro legend
5. pacing constants

Reason:

- typography and contrast improve every state
- copy changes are low risk once surfaces are legible
- legend is isolated and easy to QA
- pacing should be tuned last so visual timing is judged on final surfaces

## QA Checklist For Flow

- `S0` legend appears once, fades, never returns
- `S2/S7/S12` fallback states remain neutral and non-mechanistic
- `S6 -> S8 -> S9 -> S10` order is preserved exactly
- no new state bypasses evidence gating
- `FFC -> Release` remains the only slow-motion window
- `BFC`, `FFC`, and `Release` remain the only pause anchors
- end summary behavior is unchanged unless explicitly expanded later

## Decision

We will implement the new renderer design by restyling the existing state machine, not by inventing a new one.

That means:

- control flow stays stable
- truth logic stays stable
- visual readability and product polish improve
- the constrained spec remains the build contract
