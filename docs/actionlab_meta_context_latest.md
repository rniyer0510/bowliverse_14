# ActionLab Meta Context

Updated: 2026-04-24

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

## Next Step

Next work should move back to frontend and use this backend contract to design and refine:

- Report page
- History page

The aim is to make those pages coherent with:

- the new dark sports-performance theme
- the coach-question architecture
- the role-aware detail model
- the humane acceptable/workable/problematic framing
