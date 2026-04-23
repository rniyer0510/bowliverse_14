# ActionLab Knowledge Program First Pass

This first pass turns the coaching-knowledge direction into concrete backend artifacts inside the deterministic expert knowledge pack.

## Goals

The backend should grow toward answering these coach questions:

- How is the kinetic chain overall?
- Where is it breaking?
- Why is it breaking there?
- What is the smallest useful fix?
- What should not be changed yet?
- What positive and negative effects might that change create in the near, medium, and long term?

## New Knowledge-Pack Sections

### `contributors.yaml`

Canonical catalog of coach-relevant biomechanical findings.

Each contributor defines:

- stable id
- source type and source key
- body group
- phase anchor
- summary and deeper definition
- why it matters
- renderer focus regions
- linked symptoms and mechanisms

This is the pack-backed home for details like:

- trunk lean
- hip-shoulder mismatch
- foot-line deviation
- front-foot landing quality
- chest stack over landing
- front-leg support
- neck tilt at BFC

### `coach_judgments.yaml`

Canonical coaching vocabulary for:

- chain-status labels
- break-point labels and summaries
- change-size bands
- adoption-risk bands
- reaction horizons

This is intended to keep report, history, and review tooling aligned on one coaching language.

### `capture_templates.yaml`

Structured data-capture templates for:

- coach review questionnaire
- clip annotation fields
- intervention outcome fields
- standard outcome windows

These templates are the first-pass contract for collecting proprietary ActionLab knowledge.

## Runtime Usage

The deterministic expert now reads contributor descriptions from the knowledge pack rather than hard-coded Python catalogs.

That means future biomechanical additions can increasingly be taught through the pack itself, not only through Python edits.

Break-point titles and summaries also now come from the coach-judgment vocabulary when available.

## Intended Workflow

1. Expand contributor coverage
2. Review real clips against the questionnaire
3. Capture intervention outcomes against the defined windows
4. Promote repeatable expert patterns into mechanisms, prescriptions, and change reactions

## Important Limitation

This first pass creates the structure for richer findings like BFC neck tilt, but it does not automatically detect every new contributor yet.

The next gains come from teaching more real biomechanical signals and linking them into the pack and deterministic reasoning path.
