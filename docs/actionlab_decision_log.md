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
