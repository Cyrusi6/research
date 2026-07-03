# Auto-Research System v2

This repository turns the design in `prompts.md` into a runnable Python CLI for a staged idea-to-paper workflow.

## What It Does

- Creates isolated project workspaces with strict artifact management.
- Runs a 5-stage pipeline: literature, planning, experiments, writing, review.
- Downloads paper PDFs into a project-local `references/papers/` directory.
- Tracks every stage output with `stage_manifest.json`.
- Persists pipeline state in `meta/registry.yaml`.
- Supports `init`, `start`, `resume`, `status`, `report`, `doctor-c2c`, `audit-c2c`, `replay-c2c`, `enrich-s0`, `review`, and `catchup`.

## Quick Start

```bash
uv run auto-research init --topic "multimodal retrieval with hard negatives"
uv run auto-research start --project-id <project_id> --simulate
uv run auto-research status --project-id <project_id>
uv run auto-research report --project-id <project_id>
```

`--simulate` makes the pipeline produce deterministic mock results for validation and tests. Without it, the experiment stage refuses to invent results and will block if no executable experiment plan is available.

Optional S0 semantic enrichment runs during C2C S0 intake after deterministic chunking. The default config uses a small DeepSeek sample first (`limit: 6`) so cost and quality can be inspected before switching to full enrichment. It can also be run manually on an existing S0 bundle:

```bash
uv run auto-research enrich-s0 --project-id <project_id> --limit 6 --dry-run
uv run auto-research enrich-s0 --project-id <project_id> --limit 6
```

The command writes `intake/c2c/semantic_enrichment_sample.json` and `.jsonl`, including token usage and projected full-run cost. During S0 intake, enriched chunks keep their deterministic `chunk_id`, `source_path`, and line/section anchors; DeepSeek only adds semantic fields.

For C2C projects, the E2E guardrail commands can be run without restarting the whole loop:

```bash
uv run auto-research doctor-c2c --project-id <project_id>
uv run auto-research audit-c2c --project-id <project_id>
uv run auto-research replay-c2c --project-id <project_id> --from-stage S3_experiment
```

`doctor-c2c` writes readiness and runtime-health reports under `meta/`, `audit-c2c` checks C2C stage artifacts, schemas, manifest hashes, and stale route-invalidated files, and `replay-c2c` replays deterministic route-policy decisions from frozen artifacts without rerunning S1/S2 LLM calls.

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
- `DEEPSEEK_API_KEY`: optional, used by `auto-research enrich-s0` for DeepSeek S0 semantic enrichment.
- `SEMANTIC_SCHOLAR_API_KEY`: optional, improves literature retrieval rate limits.
- `SERPAPI_API_KEY`: reserved for future use.
- `MM_ROOT`: optional, points to a local multimodal research assets directory when reusing datasets, codebases, or checkpoints.

## Notes

- The system keeps output management strict: stage files are committed atomically and recorded in manifests.
- S2.5 persistent Codex sessions keep metadata/events in `workspace/<project_id>/plan/code_worktrees/`, but new Git worktree repos are stored outside the repo by default via `code_patch.worktree_storage_root` or `AUTO_RESEARCH_WORKTREE_ROOT`.
- It is designed to be honest. If a stage lacks the data or execution hooks needed to proceed, the registry is marked blocked or failed instead of fabricating artifacts.
