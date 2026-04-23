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

### `research_sources.yaml`

Canonical source catalog for:

- peer-reviewed biomechanics papers
- official cricket-body or university research pages
- expert-practitioner sites

Each source is tagged with an evidence tier so the system can distinguish primary biomechanics truth from coaching translation and heuristics.

### `knowledge_evidence.yaml`

Structured evidence assertions linking sources to:

- contributors
- mechanisms
- prescriptions
- trajectories
- symptoms

This is the first-pass home for derived claims such as:

- what FFC evidence supports a transfer-break story
- what supports a load-cost interpretation
- what is only a coaching translation versus a direct biomechanics finding

### `reconciliation.yaml`

Canonical duplicate/similar-concept reconciliation catalog.

This is where ActionLab can explicitly say:

- these findings are duplicates
- these findings are similar but not the same
- this is the canonical concept we want report/history to trend
- these are neighboring ideas that need human review rather than automatic merging

### `architecture_principles.yaml`

Explicit backend architecture principles for:

- classification before interpretation
- strict validation gates
- human interpretation boundaries
- momentum-not-fate trajectory framing
- clear automation stopping points

This section is inspired by the Nadi-architecture references you shared and is meant to keep ActionLab honest as the product scales.

## Runtime Usage

The deterministic expert now reads contributor descriptions from the knowledge pack rather than hard-coded Python catalogs.

That means future biomechanical additions can increasingly be taught through the pack itself, not only through Python edits.

Break-point titles and summaries also now come from the coach-judgment vocabulary when available.

The pack now also stores architecture principles so that future implementation does not drift away from:

- pattern classes before narrative
- hard validation before explanation
- human responsibility after classification and validation

## Intended Workflow

1. Expand contributor coverage
2. Review real clips against the questionnaire
3. Capture intervention outcomes against the defined windows
4. Promote repeatable expert patterns into mechanisms, prescriptions, and change reactions
5. Reconcile duplicates and similar concepts before promoting them into runtime truth

## Important Limitation

This first pass creates the structure for richer findings like BFC neck tilt, but it does not automatically detect every new contributor yet.

The next gains come from teaching more real biomechanical signals and linking them into the pack and deterministic reasoning path.

The new source and reconciliation artifacts are intended to keep that growth disciplined rather than letting the knowledge base drift into duplicated or overlapping concepts.
