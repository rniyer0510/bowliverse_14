# ActionLab Meta Context

Updated: 2026-04-28

## Product Direction

ActionLab is being built as a serious cricket bowling coaching system, not a generic biomechanics viewer.

The core architecture is:

1. Analyze
2. Pattern match
3. Index
4. Interpret

Interpretation must answer real coach questions:

- How is the kinetic chain?
- Where is it breaking?
- Why is it breaking there?
- What is still working?
- What is not working?
- What is the first fix?
- How small should the change be?
- What might improve or worsen near term, medium term, and long term?

## Kinetic Chain Decision

ActionLab uses a two-part kinetic-chain contribution model.

Upper body:
- trunk lean
- hip-shoulder separation
- trunk rotation

Lower body:
- front-foot line alignment
- front-foot braking
- knee brace

Externally the app should tell one whole-chain story.
Internally fused fallback signals are allowed, but they must still roll into this same upper/lower structure.

## Humanization Principle

Most bowling actions are not perfect.
The system must distinguish:

- acceptable
- workable
- problematic

The app should not treat every imperfection as pathology.
This is a key part of making the product humane and coach-like.

## Backend Progress

The deterministic expert now includes:

- `coach_diagnosis_v1`
- `change_strategy`
- `change_reaction`
- `acceptance_summary`
- `key_metrics`
- `kinetic_chain_status`
- `frontend_surface_v1`

The system now supports role-aware surfaces, with coaches seeing more diagnostic detail than players/parents.

## Risk Calibration Progress

These generic risks were reworked without bowler-specific hardcoding:

- front-foot braking
- knee brace
- lateral trunk lean
- trunk rotation snap
- foot-line deviation
- hip-shoulder mismatch

The goal of this pass was to reduce false positives on clean elite actions and improve whole-chain coherence.

## Validation Cases

Amogh is a validation case, not a hardcoded target.
McGrath is the anti-farce reference case.

Current intended reading:

- Amogh:
  - workable but leaking
  - lower-body-led transfer/block issue
  - upper body contributes, but should not hijack the story

- McGrath:
  - connected
  - no clear break point
  - no forced pathology story

## Frontend Contract

Frontend should increasingly bind to `frontend_surface_v1` rather than stitching together older payloads manually.

That surface now contains:

- headline
- summary_lines
- chips
- hero
- body
- guidance
- renderer
- history
- holdback

## Release State

`actionlab_2.0.0` is now the current major milestone across backend and frontend.

This release includes:

- major deterministic-expert and knowledge-base expansion
- kinetic-chain-driven walkthrough truth path
- walkthrough renderer redesign for mobile readability
- better separation of proof, leak, body-pay, and summary states
- frontend report/history work already merged to main

Both repos have been merged to mainline and tagged:

- backend: `/Users/rniyer/bowliverse_14/app`
- frontend: `/Users/rniyer/dev/bowliverse_android_smoke`
- tag: `actionlab_2.0.0`

## Current UX State

The walkthrough renderer has now gone through a full redesign and stabilization pass.

Important current decisions:

- keep the current font family and color decisions
- keep the cleaner, lighter renderer card layout
- prefer the left-accent-rail card treatment over full bordered heavy cards
- keep summary/state cards visually premium, but not busy
- keep the final summary background sharp or only lightly darkened; do not return to heavy blur
- explanation cards should live in a side lane opposite the bowler/action side
- compact proof tags should stay close to the proof ring and not cover the arm/hand inspection path
- card boxes should wrap the text block, not force text into a pre-sized slab
- phase rail should stay readable with a bottom scrim and clear done/active/upcoming distinction

### Renderer Principle

The walkthrough video is the primary explanation surface.

That means:

- the video should do most of the diagnostic work
- post-video report UI should stay light and supportive
- the renderer should prioritize clarity, pause moments, and coach-like wording over decorative UI complexity
- renderer cues must remain anchored to trustworthy proof frames / proof windows
- visuals must not become decorative overlays detached from the actual clip
- pose is the observation layer
- kinematics is the truth layer
- heuristics come last
- correct anchor frames matter because rendering is a hero surface

### Renderer Truth Rule

Renderer interpretation should now be treated as kinetic-chain-first.

What is now true:

- raw renderer risk fallbacks were reduced or removed where they could fabricate pathology
- connected / acceptable chains should not be forced into a negative story
- `not_interpretable` should stay neutral rather than pretending precise pathology
- McGrath is the anti-farce validation case for this rule

What is not fully complete yet:

- some teaching phrases and fallback copy still live in renderer-side helpers instead of the knowledge base / deterministic contract
- interpretation is cleaner than before, but not yet fully knowledge-base-owned end to end

### Report Principle

The report should now be treated as a near-empty canvas around the walkthrough.

Current direction:

- video first
- then only the key post-video action cards
- leaking/problematic actions show:
  - `Immediate win`
  - `Long term`
- connected / no-clear-problem actions show:
  - `What is working`
  - `Keep`

The report should not restate what the walkthrough already shows.

### Current Interaction Direction

The report is now moving toward a stronger link between coaching cards and the actual action moment:

- coaching cards should point back to exact walkthrough moments
- the scrubber should expose time and phase context
- long-term coaching should be rendered as discrete items, not one stitched sentence

## Knowledge-Base Completion Gap

The next important architecture cleanup is to finish the knowledge-base implementation.

Partial completion today:

- deterministic expert is already driving more of the walkthrough truth path
- connected / no-clear-problem / not-interpretable rules now influence renderer behavior
- frontend and backend were both stabilized around the new release milestone

Still pending:

- remove remaining renderer-owned interpretation phrasing where the knowledge base should decide the teaching copy
- move risk-specific coaching language fully into the deterministic / knowledge-pack path

## Speed Meta Context

Release speed should currently be understood as one of the most sensitive outputs to upstream anchor changes.

Important current understanding:

- many recent speed shifts were not caused by the raw speed formula itself
- they were caused by changes in:
  - handedness resolution
  - release frame placement
  - the local release neighborhood reaching the speed estimator
- this means ActionLab must treat speed as downstream of anchor truth, not as an isolated calculator

### Current Speed Rule

The speed estimator is still primarily a local camera-based kinematic estimator.

Its raw terms remain:

- wrist velocity relative to arm scale
- elbow extension speed
- pelvis movement contribution
- shoulder movement penalty

Current direction is:

- keep the raw pace formula mostly intact
- avoid clip-specific patching
- correct upstream frame alignment first
- add only narrow safeguards when the estimate is clearly implausible

### Current Diagnosis

The most important current speed issue is release-window alignment.

What we observed:

- release truth can move earlier when consensus uses stronger kinematic signals
- when the speed metric window stays centered on `release.frame`, it can miss the true wrist-speed peak
- this systematically underestimates pace on some clips
- pathological clips can do the opposite and saturate high when the bowling-arm window is noisy

Key insight:

- `events["peak"]["frame"]` is often a better speed-window anchor than `events["release"]["frame"]`
- release truth and speed-measurement center should not always be assumed to be the same frame

### Current Planned Revisit

When speed is revisited, the next preferred direction is:

- move `pelvis_jerk` out of release locators and treat it as an early-frame suppression gate
- anchor the speed metric window to `peak.frame` with a plausibility guard relative to `release.frame`
- widen the forward side of the metric window so small release shifts do not truncate the true wrist peak
- keep body/arm scale measurement anchored on release rather than blindly moving all windows to wrist peak

### Camera Distance / Occlusion Follow-Up

Another explicit follow-up area is small-subject and occlusion-aware speed handling.

Current belief:

- camera distance itself is not the only problem
- the practical failure mode is loss of distal visibility and stability on far / small-subject clips
- ActionLab already has occlusion-aware rendering and track honesty
- speed does not yet fully consume that occlusion-aware trust model

Future direction:

- integrate occlusion-aware joint trust into speed confidence and compensation
- especially for:
  - bowling wrist
  - bowling elbow
  - bowling shoulder
  - pelvis

### Practical Reminder

Until this revisit is complete:

- do not overfit speed to individual clips
- prefer simpler, defensible behavior over aggressive correction
- treat McGrath as the clean carried-pace reference
- treat `Amogh_new.mp4` as the small-subject / weak-distal-visibility reference
- make renderer-facing story payloads come from one contract instead of a mix of renderer helper phrasing plus deterministic truth
- review all fallback channels so none can recreate pathology outside acceptable kinetic-chain bands
- verify frontend surfaces are consuming the newer deterministic contracts as directly as possible

## Next Thread

The next thread should focus on completing the knowledge-base-driven interpretation path before any major new product surface work.

Primary goal:

- finish the partial implementation so all important interpretation and teaching copy is sourced from the deterministic expert / knowledge base rather than renderer-local phrasing

Secondary goals:

- review remaining renderer helper language and migrate what should be KB-owned
- verify frontend and report bindings against the deterministic contracts
- close any remaining split-brain paths between kinetic-chain truth and presentation wording
- only after that, resume larger frontend/product work such as History refinements
