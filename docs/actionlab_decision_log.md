# ActionLab Decision Log

This log records architecture and product decisions that should remain stable unless explicitly revisited.

## ADR-001: Kinetic Chain Contribution Model

- Date: 2026-04-23
- Status: Accepted

### Decision

ActionLab will interpret bowling mechanics through a two-part kinetic-chain model:

- Upper body
  - trunk lean
  - hip-shoulder separation
  - trunk rotation
- Lower body
  - front-foot line alignment
  - front-foot braking
  - knee brace

These six contributors form the core internal biomechanical structure used for pattern matching, indexing, and interpretation.

### Interpretation Rule

Externally, ActionLab should present one coherent kinetic-chain story rather than exposing detector-by-detector reasoning.

Internally, it is acceptable to fuse multiple supporting signals to infer a contributor when direct visibility is weak, but the interpretation layer should still roll those signals into the same upper-body / lower-body structure above.

### Implementation Guidance

- `Analyze` should compute the six core contributors as robustly as possible.
- `Pattern match` should operate on upper-body and lower-body combinations, not isolated detector spikes.
- `Index` should store canonical contributor truth using this same two-part structure.
- `Interpret` must answer coaching questions from one whole-chain story built from these grouped contributors.
- Fallback signals are allowed internally, but they must not create a separate public story outside this structure.

### Rationale

This keeps the app aligned with the intended coaching model:

- upper-body issues are interpreted from upper-chain organization
- lower-body issues are interpreted from landing and support behavior
- root cause comes from how those two groups interact across the chain

It also prevents the product from drifting into isolated detector storytelling instead of holistic kinetic-chain interpretation.

## ADR-002: Release Speed Should Evolve as a Hybrid Model, Not a Pure Biomechanics Number

- Date: 2026-04-24
- Status: Accepted

### Decision

For now, ActionLab will keep release speed as a camera-based kinematic estimate with confidence gating.

In a future iteration, release speed may evolve into a hybrid model where:

- the raw speed signal still comes from local kinematic estimation
- biomechanics and kinetic-chain truth govern confidence, interpretation, and presentation
- the product explains not only estimated speed, but also how that speed is being made

ActionLab will not attempt to compute absolute release speed from kinetic-chain and biomechanics reasoning alone.

### Future Feature Direction

A later version may add a biomechanics-governed speed layer that:

- distinguishes speed carried through the chain from speed rescued late
- reflects whether pace is being made efficiently or expensively
- downgrades or hides speed more aggressively when chain quality is weak
- uses external ground truth such as radar for validation and calibration where available

### Interpretation Rule

Release speed should eventually behave like an ActionLab explanation object, not a disconnected calculator.

That means the system should be able to express:

- estimated pace output
- confidence in the estimate
- whether the chain is building pace early or rescuing it late
- whether the output appears efficient, expensive, or unstable

### Rationale

Pure biomechanics and kinetic-chain reasoning can explain where pace is coming from, but they are not sufficient by themselves to produce a trustworthy absolute km/h value.

A hybrid approach is the right long-term fit because it:

- preserves the usefulness of a camera-only product
- keeps speed aligned with ActionLab's biomechanics model
- avoids overclaiming precision where the available evidence does not justify it
- creates a clean path for future calibration against radar or other ground-truth systems

### 2026-04-29 Follow-Up Note

This ADR remains correct, but the current implementation needs another deliberate revisit before it should be treated as final.

What we learned from the McGrath and Amogh speed passes:

- speed regressions were driven more by upstream hand / anchor / release-neighborhood changes than by a single bug inside `release_speed.py`
- clean fast-bowler clips such as McGrath need a path back toward the stronger `master` behavior
- far-camera / small-subject clips such as `Amogh_new.mp4` need explicit protection against unstable distal visibility without blindly trusting noisy bowling-arm spikes
- wrong-hand or weak-hand hypotheses can accidentally make a clip look numerically calmer, so speed must stay aligned to profile truth and validated event truth

Current implementation direction at pause point:

- keep the local camera-based release-speed estimate as the primary number source
- keep narrow safety guards for implausible saturated estimates
- allow limited recovery for clean salvage cases
- allow limited compensation for far / small-subject clips when the clip is otherwise orderly

What is intentionally deferred for the next revisit:

- a more rigorous camera-distance / subject-scale treatment that improves small-subject pace estimates without altering raw arm-speed signals directly
- a better mapping between ActionLab release truth and the local speed-estimation window when those two should not be identical
- calibration against a wider real-clip acceptance set, especially:
  - McGrath
  - `Amogh_new.mp4`
  - `Amogh_latest_April-26.mp4`
- eventual validation against external ground truth such as radar where available

Practical rule until revisited:

- do not keep patching speed clip-by-clip
- prefer a simpler `master`-like estimator with narrow guards over a broad governor that suppresses good fast-bowling clips
- treat camera distance and distal-visibility loss as a first-class future calibration topic

## ADR-003: Pose Feeds Kinematics, and Heuristics Must Stay Last

- Date: 2026-04-26
- Status: Accepted

### Decision

ActionLab will treat the analysis stack in this order:

1. pose provides the raw observation layer
2. kinematics provides the primary truth layer
3. heuristics provide only inference, arbitration, and fallback behavior

This means the product should not be architected around brittle one-off detector heuristics or single-frame guesses when richer kinematic evidence is available.

### Interpretation Rule

ActionLab should operate on the following principle:

- pose provides the measurements
- kinematics provides the real signal
- heuristics help choose among plausible interpretations

Heuristics must not invent causal truth. They may:

- rank or score event candidates
- enforce temporal plausibility
- resolve ambiguity between competing anchors
- decide which outputs are trustworthy enough to show

But they must remain downstream of pose quality and kinematic evidence.

### Implementation Guidance

- Event detection should move toward candidate generation plus multi-signal kinematic scoring, not single-anchor heuristics.
- Release, BFC, FFC, and UAH should be inferred from globally plausible temporal chains rather than one detector poisoning everything downstream.
- Capability gating should be per output, not only per clip.
  Examples:
  - a clip may still support structure analysis even when release-dependent stats do not
  - speed and legality should be suppressed when release is weak, invalid, or missing
- Proximal joint signals should be used as robust fallbacks when distal joints are occluded or weak, especially in close-camera or truncated clips.
- Missing release and invalid late release should be treated as different states.
  - missing release may force a full retake path
  - invalid or weak release may still allow partial pre-release analysis

### Product Consequence

ActionLab should aim to extract the maximum trustworthy analysis from imperfect real-world videos rather than rejecting clips unless nothing defensible remains.

The target behavior is:

- full analysis when anchors are confident
- partial analysis when only some anchors are trustworthy
- explicit retake guidance only when the system cannot defend enough useful truth

### Rendering Consequence

Identifying the right frame is paramount.

ActionLab's rendered walkthrough is one of the most important product surfaces, so the system must optimize not only for a durable signal, but also for a durable anchor frame that can be shown and trusted visually.

That means:

- the chosen event frame must be physically plausible, not just numerically convenient
- the renderer should prefer stable, explainable anchor frames over fragile single-frame guesses
- a weak signal paired with the wrong frame is worse than a partial analysis with the right frame
- durable signal plus durable frame is the foundation for trustworthy pause moments, hotspot callouts, and end-card summaries

### Rationale

Users will not consistently capture perfect bowling videos. If the architecture depends too heavily on heuristics or exact distal visibility, the product will fail on too many real clips.

Making kinematics the central truth layer:

- increases robustness to camera angle, occlusion, and truncation
- supports graceful degradation instead of all-or-nothing failure
- improves trust because the system can suppress only the outputs that are not defensible
- creates a cleaner path toward analyzing 70 to 80 percent of real uploaded clips instead of only ideal captures
