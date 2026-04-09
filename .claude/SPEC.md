# Auto-Research System v2 — Engineering Specification

## Stack

- Python 3.10+
- OpenAI API for optional generation
- Requests for literature retrieval and PDF download
- YAML state files for registry and configuration
- Local workspace as the only source of truth for artifacts

## Runtime Principles

1. No stage may pass without persisted artifacts.
2. The orchestrator trusts files, not conversational memory.
3. Each stage is resumable from `registry.yaml` plus stage manifests.
4. The experiment stage must not fabricate results. Use simulation mode explicitly when needed.
5. The review stage produces actionable revisions tied to owning agents.

## Registry

`workspace/{project_id}/meta/registry.yaml` stores:

- project metadata
- current stage
- iteration
- per-stage status
- judge retries
- artifact summary
- blocked and failure reasons

## Stage Manifests

Each stage directory contains `stage_manifest.json` with committed outputs, source lineage, hashes, timestamps, and summaries. `references/papers/manifest.json` serves the same role for downloaded PDFs.
