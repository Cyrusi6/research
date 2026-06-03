"""S0 static project evidence intake."""

from __future__ import annotations

import json
from typing import Any

from ..c2c import C2CAdapter, is_c2c_project
from ..utils import read_json
from .base import AgentContext


class IntakeAgent:
    stage_key = "S0_intake"

    def __init__(self, context: AgentContext):
        self.context = context

    def run(self, topic: str) -> dict[str, Any]:
        if is_c2c_project(self.context.config):
            return self._run_c2c_intake(topic)
        note = {
            "schema_version": "static_intake_v1",
            "project_id": self.context.project_root.name,
            "topic": topic,
            "mode": "generic",
            "status": "skipped",
            "reason": "Generic projects keep literature collection in S1.",
        }
        record = self.context.artifacts.write_json(
            self.stage_key,
            "static_intake.json",
            note,
            artifact_type="static_intake",
            summary="Generic static intake placeholder",
        )
        return {"artifacts": [record["path"]], "status": "ok"}

    def _run_c2c_intake(self, topic: str) -> dict[str, Any]:
        cached = self._load_reusable_c2c_static_bundle()
        if cached:
            return {
                "artifacts": _c2c_static_artifact_paths(cached),
                "status": "ok",
                "static_bundle": cached,
                "cache_status": "reused",
            }

        adapter = C2CAdapter(self.context.project_root, self.context.config)
        reference_result = adapter.import_reference_materials()
        if reference_result["status"] == "blocked":
            parse_errors = reference_result.get("parse_errors") or []
            blocked_reason = (
                "C2C reference parsing failed: "
                + "; ".join(str(item.get("error") or item.get("err_msg") or item) for item in parse_errors[:3])
                if parse_errors
                else f"Missing C2C reference path(s): {', '.join(reference_result['missing'])}"
            )
            blocked_record = self.context.artifacts.write_json(
                self.stage_key,
                "c2c/reference_import_blocked.json",
                reference_result,
                artifact_type="c2c_reference_blocked",
                summary="C2C reference import blocked by missing configured paths",
            )
            return {
                "artifacts": [blocked_record["path"]],
                "status": "blocked",
                "blocked_reason": blocked_reason,
            }

        repo_manifest = adapter.build_repo_manifest()
        historical_results = adapter.import_historical_results()
        baseline = adapter.baseline_evidence(historical_results)
        repo_card = adapter.build_repo_card(repo_manifest, historical_results)
        paper_cards = adapter.build_paper_cards(reference_result["cards"])
        paper_chunks = adapter.build_paper_chunks(reference_result["cards"])
        bibliography_cards = adapter.build_bibliography_cards(reference_result["cards"])
        rebuttal_matrix = adapter.build_rebuttal_concern_matrix(reference_result["cards"])
        rebuttal_chunks = adapter.build_rebuttal_chunks(reference_result["cards"])
        code_cards = adapter.build_code_cards(repo_manifest)
        code_intake = adapter.build_code_intake()
        code_chunks = code_intake.chunks
        chunk_index = adapter.build_chunk_index(
            paper_chunks=paper_chunks,
            rebuttal_chunks=rebuttal_chunks,
            code_chunks=code_chunks,
        )
        result_ledger_csv = adapter.build_result_ledger_csv(historical_results, baseline)
        negative_memory = adapter.build_negative_result_memory(historical_results, baseline)
        retrieval_plan = adapter.build_research_retrieval_plan(
            topic=topic,
            repo_card=repo_card,
            paper_cards=paper_cards,
            paper_chunks=paper_chunks,
            rebuttal_matrix=rebuttal_matrix,
            rebuttal_chunks=rebuttal_chunks,
            code_cards=code_cards,
            code_chunks=code_chunks,
            negative_memory=negative_memory,
            baseline=baseline,
        )
        followup_bundle = adapter.build_research_followup_bundle(
            retrieval_plan,
            paper_chunks=paper_chunks,
            rebuttal_chunks=rebuttal_chunks,
            code_chunks=code_chunks,
            negative_memory=negative_memory,
        )
        metadata = [
            {
                "paper_id": card["paper_id"],
                "title": card["title"],
                "source_path": card["source_path"],
                "local_path": card["local_path"],
                "kind": card["kind"],
                "text_snippet": (card.get("text") or "")[:1200],
            }
            for card in reference_result["cards"]
        ]
        evidence_brief = _c2c_evidence_brief(
            topic=topic,
            baseline=baseline,
            repo_card=repo_card,
            paper_cards=paper_cards,
            rebuttal_matrix=rebuttal_matrix,
            code_cards=code_cards,
            negative_memory=negative_memory,
            retrieval_plan=retrieval_plan,
            followup_bundle=followup_bundle,
        )
        cache_summary = _c2c_cache_summary(reference_result.get("paper_full_manifest", []), code_intake.report)
        static_bundle = {
            "schema_version": "c2c_static_intake_bundle_v1",
            "project_id": self.context.project_root.name,
            "topic": topic,
            "metadata": metadata,
            "reference_result": reference_result,
            "paper_full_manifest": reference_result.get("paper_full_manifest", []),
            "repo_manifest": repo_manifest,
            "historical_results": historical_results,
            "baseline": baseline,
            "repo_card": repo_card,
            "paper_cards": paper_cards,
            "paper_chunks": paper_chunks,
            "bibliography_cards": bibliography_cards,
            "rebuttal_matrix": rebuttal_matrix,
            "rebuttal_chunks": rebuttal_chunks,
            "code_cards": code_cards,
            "code_file_manifest": code_intake.file_manifest,
            "code_symbols": code_intake.symbols,
            "code_chunks": code_chunks,
            "code_edges": code_intake.edges,
            "code_repo_map": code_intake.repo_map,
            "code_intake_report": code_intake.report,
            "implementation_surface_map": code_intake.surface_map,
            "code_retrieval_index": code_intake.retrieval_index,
            "cache_summary": cache_summary,
            "chunk_index": chunk_index,
            "result_ledger_csv": result_ledger_csv,
            "negative_memory": negative_memory,
            "retrieval_plan": retrieval_plan,
            "followup_bundle": followup_bundle,
            "evidence_brief": evidence_brief,
        }
        records = []
        records.append(self.context.artifacts.write_json(self.stage_key, "papers/metadata.json", metadata, artifact_type="metadata", summary="C2C configured reference materials"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/paper_full_manifest.json", reference_result.get("paper_full_manifest", []), artifact_type="c2c_paper_full_manifest", summary="MinerU paper_full.md outputs for C2C PDF references"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/static_bundle.json", static_bundle, artifact_type="c2c_static_bundle", summary="C2C static S0 evidence bundle"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/repo_manifest.json", repo_manifest, artifact_type="c2c_repo_manifest", summary="C2C repo intake manifest"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/repo_card.json", repo_card, artifact_type="c2c_repo_card", summary="C2C repo capabilities, constraints, and evidence inventory"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/historical_results.json", historical_results, artifact_type="c2c_historical_results", summary="Imported C2C historical experiment evidence"))
        records.append(self.context.artifacts.write_text(self.stage_key, "c2c/result_ledger.csv", result_ledger_csv, artifact_type="c2c_result_ledger", summary="Normalized C2C historical result ledger"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/baseline_evidence.json", baseline, artifact_type="c2c_baseline_evidence", summary="C2C baseline target to beat"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/paper_cards.json", paper_cards, artifact_type="c2c_paper_cards", summary="Structured cards for configured C2C-area papers"))
        records.append(self.context.artifacts.write_text(self.stage_key, "c2c/paper_chunks.jsonl", _jsonl(paper_chunks), artifact_type="c2c_paper_chunks", summary="Section-aware C2C paper chunks without bibliography text", metadata={"chunk_count": len(paper_chunks)}))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/bibliography.json", bibliography_cards, artifact_type="c2c_bibliography", summary="Reference sections preserved separately for related-work expansion"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/rebuttal_concern_matrix.json", rebuttal_matrix, artifact_type="c2c_rebuttal_concerns", summary="Reviewer concern matrix parsed from rebuttal materials"))
        records.append(self.context.artifacts.write_text(self.stage_key, "c2c/rebuttal_chunks.jsonl", _jsonl(rebuttal_chunks), artifact_type="c2c_rebuttal_chunks", summary="Review and rebuttal chunks with source anchors", metadata={"chunk_count": len(rebuttal_chunks)}))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/code_cards.json", code_cards, artifact_type="c2c_code_cards", summary="Core C2C code symbols, config knobs, and imports"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/code_file_manifest.json", code_intake.file_manifest, artifact_type="c2c_code_file_manifest", summary="Tree-sitter code file manifest and edit surface"))
        records.append(self.context.artifacts.write_text(self.stage_key, "c2c/code_symbols.jsonl", _jsonl(code_intake.symbols), artifact_type="c2c_code_symbols", summary="Tree-sitter code symbols for repository map", metadata={"symbol_count": len(code_intake.symbols)}))
        records.append(self.context.artifacts.write_text(self.stage_key, "c2c/code_chunks.jsonl", _jsonl(code_chunks), artifact_type="c2c_code_chunks", summary="Function-level source chunks for S1/S2 reasoning", metadata={"chunk_count": len(code_chunks)}))
        records.append(self.context.artifacts.write_text(self.stage_key, "c2c/code_edges.jsonl", _jsonl(code_intake.edges), artifact_type="c2c_code_edges", summary="Static code relation graph edges", metadata={"edge_count": len(code_intake.edges)}))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/code_repo_map.json", code_intake.repo_map, artifact_type="c2c_code_repo_map", summary="Compact tree-sitter repository map"))
        records.append(self.context.artifacts.write_text(self.stage_key, "c2c/code_repo_map.md", code_intake.repo_map_md, artifact_type="c2c_code_repo_map_markdown", summary="Readable tree-sitter repository map"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/code_intake_report.json", code_intake.report, artifact_type="c2c_code_intake_report", summary="Code intake coverage and quality diagnostics"))
        records.append(self.context.artifacts.write_text(self.stage_key, "c2c/code_intake_report.md", code_intake.report_md, artifact_type="c2c_code_intake_report_markdown", summary="Readable code intake coverage report"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/implementation_surface_map.json", code_intake.surface_map, artifact_type="c2c_implementation_surface_map", summary="Editable mechanism surfaces for S2.5 patches"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/code_retrieval_index.json", code_intake.retrieval_index, artifact_type="c2c_code_retrieval_index", summary="Precomputed code retrieval entry points for S1/S2"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/cache_summary.json", cache_summary, artifact_type="c2c_static_cache_summary", summary="S0 cache hit/miss summary for MinerU and code intake"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/chunk_index.json", chunk_index, artifact_type="c2c_chunk_index", summary="Full S0 chunk catalog for paper, rebuttal, and code retrieval"))
        records.append(self.context.artifacts.write_text(self.stage_key, "c2c/chunk_index.jsonl", _jsonl(chunk_index.get("entries", [])), artifact_type="c2c_chunk_index_jsonl", summary="Line-delimited S0 chunk catalog for retrieval"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/retrieval_plan.json", retrieval_plan, artifact_type="c2c_retrieval_plan", summary="Chunk retrieval plan for paper, rebuttal, and code evidence"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/retrieval_followup.json", followup_bundle, artifact_type="c2c_retrieval_followup", summary="Follow-up retrieval questions and targets"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/negative_result_memory.json", negative_memory, artifact_type="c2c_negative_result_memory", summary="Static below-baseline local variants and avoid-repeat rules"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/evidence_brief.json", evidence_brief, artifact_type="c2c_evidence_brief", summary="Compact static evidence brief for S1 direction selection"))
        return {"artifacts": [record["path"] for record in records], "status": "ok", "static_bundle": static_bundle}

    def _load_reusable_c2c_static_bundle(self) -> dict[str, Any] | None:
        bundle_path = self.context.project_root / "intake" / "c2c" / "static_bundle.json"
        if not bundle_path.exists():
            return None
        bundle = read_json(bundle_path, default={})
        if not isinstance(bundle, dict):
            return None
        if bundle.get("schema_version") != "c2c_static_intake_bundle_v1":
            return None
        chunk_index = bundle.get("chunk_index")
        if not _valid_chunk_index(chunk_index):
            return None
        for key in [
            "paper_chunks",
            "rebuttal_chunks",
            "code_file_manifest",
            "code_symbols",
            "code_chunks",
            "code_edges",
            "code_repo_map",
            "code_intake_report",
            "implementation_surface_map",
            "code_retrieval_index",
            "cache_summary",
            "paper_full_manifest",
            "evidence_brief",
        ]:
            if key not in bundle:
                return None
        return bundle


def _jsonl(items: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + ("\n" if items else "")


def _valid_chunk_index(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    counts = value.get("counts") or {}
    entries = value.get("entries") or []
    return (
        isinstance(entries, list)
        and bool(entries)
        and int(counts.get("paper") or 0) > 0
        and int(counts.get("rebuttal") or 0) > 0
        and int(counts.get("code") or 0) > 0
    )


def _c2c_static_artifact_paths(bundle: dict[str, Any]) -> list[str]:
    del bundle
    return [
        "intake/papers/metadata.json",
        "intake/c2c/static_bundle.json",
        "intake/c2c/paper_full_manifest.json",
        "intake/c2c/repo_manifest.json",
        "intake/c2c/repo_card.json",
        "intake/c2c/historical_results.json",
        "intake/c2c/result_ledger.csv",
        "intake/c2c/baseline_evidence.json",
        "intake/c2c/paper_cards.json",
        "intake/c2c/paper_chunks.jsonl",
        "intake/c2c/bibliography.json",
        "intake/c2c/rebuttal_concern_matrix.json",
        "intake/c2c/rebuttal_chunks.jsonl",
        "intake/c2c/code_cards.json",
        "intake/c2c/code_file_manifest.json",
        "intake/c2c/code_symbols.jsonl",
        "intake/c2c/code_chunks.jsonl",
        "intake/c2c/code_edges.jsonl",
        "intake/c2c/code_repo_map.json",
        "intake/c2c/code_repo_map.md",
        "intake/c2c/code_intake_report.json",
        "intake/c2c/code_intake_report.md",
        "intake/c2c/implementation_surface_map.json",
        "intake/c2c/code_retrieval_index.json",
        "intake/c2c/cache_summary.json",
        "intake/c2c/chunk_index.json",
        "intake/c2c/chunk_index.jsonl",
        "intake/c2c/retrieval_plan.json",
        "intake/c2c/retrieval_followup.json",
        "intake/c2c/negative_result_memory.json",
        "intake/c2c/evidence_brief.json",
    ]


def _c2c_evidence_brief(
    *,
    topic: str,
    baseline: dict[str, Any],
    repo_card: dict[str, Any],
    paper_cards: list[dict[str, Any]],
    rebuttal_matrix: dict[str, Any],
    code_cards: list[dict[str, Any]],
    negative_memory: dict[str, Any],
    retrieval_plan: dict[str, Any],
    followup_bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "c2c_evidence_brief_v1",
        "topic": topic,
        "baseline_to_beat": baseline,
        "repo_summary": {
            "allowed_surface": repo_card.get("allowed_surface") or repo_card.get("allowed_files") or [],
            "baseline_surface": repo_card.get("baseline_surface") or {},
            "constraints": repo_card.get("constraints") or [],
        },
        "paper_brief": [
            {"paper_id": card.get("paper_id"), "title": card.get("title"), "kind": card.get("kind"), "snippet": (card.get("text") or "")[:900]}
            for card in paper_cards[:8]
        ],
        "rebuttal_concerns": {
            "top_concerns": rebuttal_matrix.get("top_concerns") or [],
            "structured_concerns": (rebuttal_matrix.get("structured_concerns") or [])[:8],
        },
        "code_surface": [
            {"path": card.get("path"), "summary": card.get("summary"), "symbols": (card.get("symbols") or [])[:8]}
            for card in code_cards[:12]
        ],
        "negative_memory": negative_memory,
        "retrieval_targets": {
            "primary_questions": (retrieval_plan.get("primary_questions") or [])[:8],
            "cross_source_targets": (followup_bundle.get("cross_source_targets") or [])[:8],
        },
        "static_rules": [
            "Use S0 evidence as stable context; do not re-import papers, rebuttals, or repo cards in S1.",
            "S1 should choose a mechanism direction; S2 owns concrete experiment planning.",
            "Avoid pure threshold/top-k/fallback tuning and evaluator changes.",
        ],
    }


def _c2c_cache_summary(paper_full_manifest: list[dict[str, Any]], code_intake_report: dict[str, Any]) -> dict[str, Any]:
    mineru_counts: dict[str, int] = {}
    for item in paper_full_manifest:
        if not isinstance(item, dict):
            continue
        status = str(item.get("cache_status") or "disabled")
        mineru_counts[status] = mineru_counts.get(status, 0) + 1
    code_cache = (code_intake_report.get("cache") or {}) if isinstance(code_intake_report, dict) else {}
    return {
        "schema_version": "c2c_static_cache_summary_v1",
        "mineru_pdf": {
            "counts": dict(sorted(mineru_counts.items())),
            "items": [
                {
                    "paper_id": item.get("paper_id"),
                    "sha256": item.get("sha256"),
                    "cache_status": item.get("cache_status", "disabled"),
                    "paper_full_md_path": item.get("paper_full_md_path"),
                }
                for item in paper_full_manifest
                if isinstance(item, dict)
            ],
        },
        "code_intake": {
            "enabled": bool(code_cache.get("enabled")),
            "counts": code_cache.get("counts") or {},
        },
    }
