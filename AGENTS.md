# Auto-Research System v2 — Pipeline Orchestration Protocol

## Overview

The system is a multi-stage research workflow with five agents and one orchestrator. All agents collaborate through a shared workspace and must register outputs to stage manifests before a stage can complete.

## Agents

| Stage | Agent ID | Responsibility |
| --- | --- | --- |
| S1 | `literature-agent` | paper search, PDF download, survey, idea generation |
| S2 | `plan-agent` | hypotheses, baselines, datasets, task graph, budget |
| S3 | `experiment-agent` | environment checks, experiment execution, self-heal, result packaging |
| S4 | `writing-agent` | outline, sections, bibliography, claim-evidence audit |
| S5 | `review-agent` | reviewer simulation, debate, meta review, revision dispatch |
| S1.5 | `idea-review-agent` | idea challenge scoring, rejection of low-ceiling ideas, prioritization |
| -- | `orchestrator` | state transitions, judge gates, retries, targeted revisions |

## File Management Rules

1. Every agent writes only to its owned stage directory plus shared read-only inputs.
2. Every committed file must appear in the stage `stage_manifest.json`.
3. S1 paper PDFs are stored under `workspace/{project_id}/references/papers/`.
4. Cross-stage reuse happens by copying artifacts and recording the source path in the new manifest entry.
5. Failed or partial outputs stay in `_tmp/` or `failed/`, never in the canonical output path.

## Stage Gates

| Stage | Gate |
| --- | --- |
| S1 | at least 3 ideas, each with novelty and feasibility >= 4, plus a non-empty paper reference manifest |
| S2 | `plan.yaml` contains hypotheses, baselines, datasets, task graph, resource budget |
| S3 | main results exist, hypothesis verification exists, ablation exists |
| S4 | `main.tex` exists, claim audit pass rate >= configured threshold, compile passes if compiler is available |
| S5 | reviewer files exist, meta review exists, revision dispatch is parseable |

## Revision Routing

When S5 returns `REVISE`, the orchestrator reads `review/revision_dispatch.yaml`, groups revisions by assigned agent and dependency order, reruns only the impacted stages, then returns to S5.
