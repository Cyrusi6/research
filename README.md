# Auto-Research System v2

This repository turns the design in `prompts.md` into a runnable Python CLI for a staged idea-to-paper workflow.

## What It Does

- Creates isolated project workspaces with strict artifact management.
- Runs a 5-stage pipeline: literature, planning, experiments, writing, review.
- Downloads paper PDFs into a project-local `references/papers/` directory.
- Tracks every stage output with `stage_manifest.json`.
- Persists pipeline state in `meta/registry.yaml`.
- Supports `init`, `start`, `resume`, `status`, `review`, and `catchup`.

## Quick Start

```bash
uv run auto-research init --topic "multimodal retrieval with hard negatives"
uv run auto-research start --project-id <project_id> --simulate
uv run auto-research status --project-id <project_id>
```

`--simulate` makes the pipeline produce deterministic mock results for validation and tests. Without it, the experiment stage refuses to invent results and will block if no executable experiment plan is available.

## Layout

```text
workspace/<project_id>/
├── references/
│   ├── papers/
│   └── bib/
├── literature/
├── plan/
├── experiment/
├── paper/
├── review/
└── meta/
```

## Environment

- `OPENAI_API_KEY`: required for S1/S2 GPT reasoning when `llm.reasoning_provider: openai`.
- `OPENAI_BASE_URL`: optional, for OpenAI-compatible third-party endpoints. Host-only values such as `https://api-cdn.owlai.tech` are normalized to the SDK API root; full `/v1/responses` values are also accepted.
- `OPENAI_REASONING_EFFORT`: optional reasoning effort, for example `xhigh`.
- `OPENAI_DEFAULT_HEADERS`: optional JSON object or `key=value;key2=value2` string for non-auth proxy headers.
- `OPENAI_ORGANIZATION`: optional, forwarded to the OpenAI client.
- `OPENAI_PROJECT`: optional, forwarded to the OpenAI client.
- `.env` or `.env.local` in the repo root are loaded automatically if present.
- `SEMANTIC_SCHOLAR_API_KEY`: optional, improves literature retrieval rate limits.
- `SERPAPI_API_KEY`: reserved for future use.
- `MM_ROOT`: optional, points to a local multimodal research assets directory when reusing datasets, codebases, or checkpoints.

## Notes

- The system keeps output management strict: stage files are committed atomically and recorded in manifests.
- It is designed to be honest. If a stage lacks the data or execution hooks needed to proceed, the registry is marked blocked or failed instead of fabricating artifacts.
