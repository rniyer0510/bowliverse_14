# ActionLab Pause-State Mobile Sequence

Date: 2026-04-28  
Branch baseline: `codex/renderer_new_design`  
Intent: define the mobile teaching sequence for paused walkthrough states

## Purpose

This document defines how paused walkthrough explanations should behave on mobile.

It is not a gating or analytics spec.

It does not change:

- evidence thresholds
- anchor selection
- `render_skeleton_video()` inputs
- the fact that the walkthrough is a baked video rather than an interactive player

It does define:

- what appears during each paused teaching step
- which overlay is primary in that step
- which overlays must quiet down
- how long each step should remain visible
- how text should behave so a coach can actually read it on a phone

## Core Principle

ActionLab is allowed to explain many things.

It must not explain many things at the same time.

The mobile grammar is:

1. pause the video
2. teach one thing
3. hold long enough to read
4. fade that teaching point away
5. show the next teaching point
6. repeat until the paused story is complete
7. resume motion only after the teaching sequence ends

This means the renderer should optimize for `serial teaching`, not `minimal annotation`.

## Existing Pause Pipeline

The current pause sequence already gives us the right structural slots:

1. `proof`
2. `leak`
3. `body pay`
4. `hotspot`

Source files:

- [pause_logic.py](/Users/rniyer/bowliverse_14/app/workers/render/coach_video_renderer_parts/pause_logic.py:94)
- [render_pause_sequence.py](/Users/rniyer/bowliverse_14/app/workers/render/coach_video_renderer_parts/render_pause_sequence.py:8)

The next design pass should refine these four slots, not invent a new state machine.

## Overlay Hierarchy Rules

These rules apply to every paused teaching step.

### Rule 1: One primary sentence only

At any paused step, only one sentence should be treated as the main read.

Examples:

- `Proof`: “Front foot lands across here.”
- `Leak`: “Energy leaks here.”
- `Body pay`: “Body pays here.”
- `Hotspot`: compact tag only

### Rule 2: Other overlays may remain visible, but must visually step back

Allowed persistent elements:

- phase rail
- telemetry chips
- skeleton
- hotspot marker

But they are never the first read during a paused explanation.

### Rule 3: Pointer text explains, compact labels tag

Use the large pointer bubble for teaching.

Use the compact hotspot label only as a short proof tag.

Do not ask the hotspot tag to carry full teaching copy.

### Rule 4: Each step needs its own hold

Do not rely on one long pause and assume the user will catch all steps.

Each teaching step should have its own readable dwell time.

### Rule 5: Fade between steps, do not hard-cut if avoidable

The ideal behavior is:

- current step fades down
- next step fades in
- the anchor frame remains paused beneath both

Even if we approximate this with frame blocks initially, the target feel is composed, not abrupt.

## Persistent Layers During Pauses

These layers remain visible across paused states unless explicitly suppressed.

### Always visible

- base video frame
- skeleton
- phase rail

### Usually visible

- telemetry chips

### Contextual

- hotspot ring / line
- transfer path
- body-pay emphasis

### Step-specific primary layer

- pointer bubble or anchor panel

## State Definitions

### P0 Intro Legend

Purpose:

- teach the two persistent visual symbols once

Visible:

- video
- skeleton
- phase rail
- legend

Primary read:

- none beyond the legend itself

Legend content:

- `Skeleton`
- `Load / fault point`

Duration:

- visible for the first `2.5s`
- fade during the last `0.5s`

Behavior:

- never reappears later

### P1 BFC Orientation Pause

Purpose:

- orient the viewer to the back-foot anchor when no mechanism claim is being made

Visible:

- paused anchor frame
- skeleton
- phase rail
- telemetry chips
- anchor panel

Primary read:

- neutral orientation sentence only

Allowed copy style:

- observational
- non-mechanistic

Not allowed:

- “leak”
- “transfer”
- “block failure”
- any diagnosis that the evidence gate did not authorize

Hold target:

- `2.2s` minimum readable hold

### P2 FFC Proof

Purpose:

- show where the issue begins on the trusted anchor

Visible:

- paused FFC anchor frame
- skeleton
- phase rail
- telemetry chips
- proof bubble

Primary read:

- one sentence naming the proof point

Examples:

- “Front foot lands across here.”
- “Front leg support softens here.”

Quiet layers:

- telemetry remains visible but visually secondary
- no hotspot label yet

Hold target:

- `2.3s` minimum
- longer if copy length demands it

### P3 Transfer Leak

Purpose:

- show where carry breaks between source and outcome

Visible:

- paused anchor frame
- skeleton
- transfer/leak animation
- phase rail
- telemetry chips
- leak bubble

Primary read:

- one short sentence

Examples:

- “Energy leaks here.”
- “Transfer breaks here.”

Copy rules:

- shorter than proof copy
- never more than one line if avoidable

Quiet layers:

- no hotspot label yet
- hotspot ring only if needed as supporting context

Hold target:

- `1.9s` minimum

### P4 Body Pay

Purpose:

- show where the body compensates after the leak

Visible:

- paused frame
- skeleton
- body-pay emphasis
- phase rail
- telemetry chips
- body-pay bubble

Primary read:

- one short sentence

Examples:

- “Body pays here.”
- “Upper body takes over here.”

Copy rules:

- keep this shorter than proof copy
- this stage is consequence, not analysis

Quiet layers:

- no hotspot compact tag yet

Hold target:

- `1.8s` minimum

### P5 Hotspot Proof Tag

Purpose:

- mark the exact proof area with a compact label after the teaching sentence is done

Visible:

- paused frame
- skeleton
- hotspot line/ring
- compact hotspot label
- phase rail
- telemetry chips

Primary read:

- compact tag only

Examples:

- `Load / fault point`
- `Landing load`
- `Shoulder chase`

Copy rules:

- `2–4 words`
- never sentence-length
- this is a tag, not a caption

Hold target:

- `1.6s` minimum

### P6 Release Proof

Purpose:

- when release is the main truth window, show the release-side proof first

Visible:

- paused release frame
- skeleton
- phase rail
- telemetry chips
- proof bubble or anchor panel

Primary read:

- one release-side proof sentence

Examples:

- “Trunk leans away here.”
- “Shoulders chase through release.”

Hold target:

- `2.3s` minimum

### P7 Release Leak

Purpose:

- if release includes chain-break evidence, show it after release proof

Visible:

- paused release frame
- skeleton
- leak emphasis
- leak bubble

Primary read:

- one short leak sentence

Hold target:

- `1.9s` minimum

### P8 Release Body Pay

Purpose:

- show the release-stage compensation area

Visible:

- paused release frame
- skeleton
- body-pay emphasis
- body-pay bubble

Primary read:

- one short consequence sentence

Hold target:

- `1.8s` minimum

### P9 Release Hotspot Tag

Purpose:

- end the release pause with the exact tagged proof region

Visible:

- paused release frame
- hotspot marker
- compact label

Primary read:

- compact tag only

Hold target:

- `1.6s` minimum

## Text Responsibilities

### Pointer bubble

Use for:

- explanation
- teaching
- “what happened here”

Should be:

- sentence case
- short
- readable in one breath

Target:

- ideally `4–8 words`
- avoid exceeding `12 words`

### Anchor panel

Use for:

- orientation when gates suppress mechanism claims

Should be:

- neutral
- observational
- confidence-respecting

### Compact hotspot label

Use for:

- a named proof tag only

Should be:

- terse
- noun phrase
- never explanatory prose

### Telemetry chips

Use for:

- context only

Should be:

- always readable
- never the primary teaching object during pauses

## Mobile Safe-Zone Guidance

### Top reading zone

Best location for the primary teaching bubble:

- upper third of the frame
- offset away from the anchor body part

Reason:

- easiest place to read on a phone
- avoids collision with the phase rail

### Bottom zone

Reserve for:

- phase rail
- tracker line

Avoid placing primary instructional copy here.

### Near-body zone

Reserve for:

- hotspot tags
- proof markers

Use only for short labels, not full sentences.

## Timing Guidance

These are mobile teaching targets, not hard analytics constants.

### Recommended minimum visible holds

- `Proof`: `2.3s`
- `Leak`: `1.9s`
- `Body pay`: `1.8s`
- `Hotspot tag`: `1.6s`
- `Orientation fallback`: `2.2s`

### Timing principle

If a sentence cannot be read comfortably inside its target hold, shorten the sentence before extending the entire pause budget too far.

Longer is not always better.

The right feel is:

- calm
- clear
- deliberate

Not:

- rushed
- wordy
- static for too long

## Implementation Guidance

### Keep

- `_pause_sequence_plan()` as the structural planner
- `_render_pause_sequence()` as the execution path
- `_reading_hold_frames()` as the readability guardrail

### Refine next

- make each step’s fade behavior more intentional
- ensure proof, leak, body-pay, and hotspot are visually distinct
- keep hotspot text shorter than pointer text
- continue sizing typography for `478x850` as the working mobile render baseline

### Do not do

- do not combine proof, leak, and cost into one large block of text
- do not put long teaching copy inside hotspot labels
- do not let telemetry chips compete with the teaching bubble
- do not fabricate mechanism when fallback mode is active

## Acceptance Standard

On a phone, a first-time coach should be able to say:

1. where the issue starts
2. where transfer breaks
3. where the body pays
4. what exact point the renderer is tagging

And they should be able to do that without pausing manually or squinting at small text.
