"""S0 static project evidence intake."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..c2c import C2CAdapter, is_c2c_project
from ..code_intake import rebuild_code_intake_indexes
from ..config import bootstrap_cached_s0_only_enabled
from ..method_memory import load_shared_method_memory, shared_method_memory_query_context
from ..s0_enrichment import DeepSeekS0SemanticEnricher, S0SemanticEnrichmentError, semantic_enrichment_enabled
from ..utils import read_json, sha256_file
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
        force_refresh = bool(
            (self.context.config.get("intake", {}) or {}).get("force_refresh")
            or (self.context.config.get("c2c", {}) or {}).get("s0_force_refresh")
        )
        expected_validity = _c2c_static_bundle_validity(self.context.project_root, self.context.config)
        cached = None if force_refresh else self._load_reusable_c2c_static_bundle(expected_validity)
        if cached:
            shared_memory = load_shared_method_memory(
                self.context.config,
                query_context=shared_method_memory_query_context(
                    self.context.config,
                    project_root=self.context.project_root,
                    topic=topic,
                    negative_memory=cached.get("negative_memory") or {},
                ),
            )
            _merge_shared_method_memory_into_negative_memory(cached.setdefault("negative_memory", {}), shared_memory)
            cached["shared_method_memory"] = shared_memory
            cached["evidence_brief"] = _evidence_brief_with_shared_method_memory(
                _c2c_evidence_brief(
                    topic=topic,
                    baseline=cached.get("baseline") if isinstance(cached.get("baseline"), dict) else {},
                    repo_card=cached.get("repo_card") if isinstance(cached.get("repo_card"), dict) else {},
                    paper_cards=cached.get("paper_cards") if isinstance(cached.get("paper_cards"), list) else [],
                    rebuttal_matrix=cached.get("rebuttal_matrix") if isinstance(cached.get("rebuttal_matrix"), dict) else {},
                    code_cards=cached.get("code_cards") if isinstance(cached.get("code_cards"), list) else [],
                    negative_memory=cached.get("negative_memory") if isinstance(cached.get("negative_memory"), dict) else {},
                    retrieval_plan=cached.get("retrieval_plan") if isinstance(cached.get("retrieval_plan"), dict) else {},
                    followup_bundle=cached.get("followup_bundle") if isinstance(cached.get("followup_bundle"), dict) else {},
                ),
                shared_memory,
            )
            cached["validity"] = expected_validity
            static_record = self.context.artifacts.write_json(
                self.stage_key,
                "c2c/static_bundle.json",
                cached,
                artifact_type="c2c_static_bundle",
                summary="Reused C2C static bundle refreshed with shared method memory",
                source_paths=[shared_memory.get("path")] if shared_memory.get("path") else [],
                metadata={"cache_status": "reused_refreshed", "validity_fingerprint": expected_validity.get("fingerprint")},
            )
            records = [static_record]
            records.extend(
                record
                for record in self._register_or_restore_cached_c2c_artifacts(cached)
                if record.get("path") != static_record["path"]
            )
            records.append(
                self.context.artifacts.write_json(
                    self.stage_key,
                    "c2c/negative_result_memory.json",
                    cached["negative_memory"],
                    artifact_type="c2c_negative_result_memory",
                    summary="Static and shared method-level avoid-repeat memory",
                    source_paths=["intake/c2c/static_bundle.json", shared_memory.get("path")] if shared_memory.get("path") else ["intake/c2c/static_bundle.json"],
                    metadata={"cache_status": "reused_with_shared_method_memory_refresh"},
                )
            )
            records.append(
                self.context.artifacts.write_json(
                    self.stage_key,
                    "c2c/evidence_brief.json",
                    cached["evidence_brief"],
                    artifact_type="c2c_evidence_brief",
                    summary="Compact static evidence brief with shared method memory",
                    source_paths=["intake/c2c/static_bundle.json", shared_memory.get("path")] if shared_memory.get("path") else ["intake/c2c/static_bundle.json"],
                    metadata={"cache_status": "reused_with_shared_method_memory_refresh"},
                )
            )
            records.append(
                self.context.artifacts.write_json(
                    self.stage_key,
                    "shared_method_failure_memory.json",
                    shared_memory,
                    artifact_type="shared_method_failure_memory",
                    summary="Cross-project method-level failure memory reused by S0/S1",
                    source_paths=[shared_memory.get("path")] if shared_memory.get("path") else [],
                )
            )
            return {
                "artifacts": [record["path"] for record in records],
                "status": "ok",
                "static_bundle": cached,
                "cache_status": "reused",
            }

        if bootstrap_cached_s0_only_enabled(self.context.config):
            blocked_reason = "Bootstrap cached-S0-only mode requires a compatible intake/c2c/static_bundle.json; DeepSeek and MinerU fallback are disabled."
            blocked_record = self.context.artifacts.write_json(
                self.stage_key,
                "c2c/cache_required_blocked.json",
                {
                    "status": "blocked",
                    "reason": blocked_reason,
                    "cached_s0_only": True,
                    "force_refresh": force_refresh,
                },
                artifact_type="c2c_cache_required_blocked",
                summary="Bootstrap S0 blocked instead of calling external enrichment or PDF parsing APIs",
            )
            return {
                "artifacts": [blocked_record["path"]],
                "status": "blocked",
                "blocked_reason": blocked_reason,
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
        implementation_surface_map = code_intake.surface_map
        code_retrieval_index = code_intake.retrieval_index
        semantic_enrichment_result: dict[str, Any] | None = None
        if semantic_enrichment_enabled(self.context.config):
            try:
                semantic_enrichment_result = DeepSeekS0SemanticEnricher(self.context.project_root, self.context.config).enrich_c2c_chunks(
                    paper_chunks=paper_chunks,
                    rebuttal_chunks=rebuttal_chunks,
                    code_chunks=code_chunks,
                )
                paper_chunks = semantic_enrichment_result["paper_chunks"]
                rebuttal_chunks = semantic_enrichment_result["rebuttal_chunks"]
                code_chunks = semantic_enrichment_result["code_chunks"]
                implementation_surface_map, code_retrieval_index = rebuild_code_intake_indexes(
                    symbols=code_intake.symbols,
                    chunks=code_chunks,
                    edges=code_intake.edges,
                )
            except S0SemanticEnrichmentError as exc:
                semantic_enrichment_result = {
                    "report": {
                        "enabled": True,
                        "status": "failed_open",
                        "reason": str(exc),
                        "fallback": "raw_chunks_without_semantic_enrichment",
                    },
                    "artifacts": [],
                }
        chunk_index = adapter.build_chunk_index(
            paper_chunks=paper_chunks,
            rebuttal_chunks=rebuttal_chunks,
            code_chunks=code_chunks,
        )
        result_ledger_csv = adapter.build_result_ledger_csv(historical_results, baseline)
        negative_memory = adapter.build_negative_result_memory(historical_results, baseline)
        shared_method_memory = load_shared_method_memory(
            self.context.config,
            query_context=shared_method_memory_query_context(
                self.context.config,
                project_root=self.context.project_root,
                topic=topic,
                negative_memory=negative_memory,
            ),
        )
        _merge_shared_method_memory_into_negative_memory(negative_memory, shared_method_memory)
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
        evidence_brief = _evidence_brief_with_shared_method_memory(evidence_brief, shared_method_memory)
        cache_summary = _c2c_cache_summary(
            reference_result.get("paper_full_manifest", []),
            code_intake.report,
            pdf_ingest_config=adapter.pdf_ingest_config,
        )
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
            "implementation_surface_map": implementation_surface_map,
            "code_retrieval_index": code_retrieval_index,
            "cache_summary": cache_summary,
            "semantic_enrichment": (semantic_enrichment_result or {}).get("report") or {"enabled": False},
            "chunk_index": chunk_index,
            "result_ledger_csv": result_ledger_csv,
            "negative_memory": negative_memory,
            "shared_method_memory": shared_method_memory,
            "retrieval_plan": retrieval_plan,
            "followup_bundle": followup_bundle,
            "evidence_brief": evidence_brief,
            "validity": expected_validity,
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
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/implementation_surface_map.json", implementation_surface_map, artifact_type="c2c_implementation_surface_map", summary="Editable mechanism surfaces for S2.5 patches"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/code_retrieval_index.json", code_retrieval_index, artifact_type="c2c_code_retrieval_index", summary="Precomputed code retrieval entry points for S1/S2"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/cache_summary.json", cache_summary, artifact_type="c2c_static_cache_summary", summary="S0 cache hit/miss summary for MinerU and code intake"))
        if semantic_enrichment_result:
            records.extend({"path": path} for path in semantic_enrichment_result.get("artifacts", []))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/chunk_index.json", chunk_index, artifact_type="c2c_chunk_index", summary="Full S0 chunk catalog for paper, rebuttal, and code retrieval"))
        records.append(self.context.artifacts.write_text(self.stage_key, "c2c/chunk_index.jsonl", _jsonl(chunk_index.get("entries", [])), artifact_type="c2c_chunk_index_jsonl", summary="Line-delimited S0 chunk catalog for retrieval"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/retrieval_plan.json", retrieval_plan, artifact_type="c2c_retrieval_plan", summary="Chunk retrieval plan for paper, rebuttal, and code evidence"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/retrieval_followup.json", followup_bundle, artifact_type="c2c_retrieval_followup", summary="Follow-up retrieval questions and targets"))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/negative_result_memory.json", negative_memory, artifact_type="c2c_negative_result_memory", summary="Static below-baseline local variants and avoid-repeat rules"))
        records.append(self.context.artifacts.write_json(self.stage_key, "shared_method_failure_memory.json", shared_method_memory, artifact_type="shared_method_failure_memory", summary="Cross-project method-level failure memory reused by S0/S1", source_paths=[shared_method_memory.get("path")] if shared_method_memory.get("path") else []))
        records.append(self.context.artifacts.write_json(self.stage_key, "c2c/evidence_brief.json", evidence_brief, artifact_type="c2c_evidence_brief", summary="Compact static evidence brief for S1 direction selection"))
        return {"artifacts": [record["path"] for record in records], "status": "ok", "static_bundle": static_bundle}

    def _load_reusable_c2c_static_bundle(self, expected_validity: dict[str, Any] | None = None) -> dict[str, Any] | None:
        bundle_path = self.context.project_root / "intake" / "c2c" / "static_bundle.json"
        if not bundle_path.exists():
            return None
        bundle = read_json(bundle_path, default={})
        if not isinstance(bundle, dict):
            return None
        if bundle.get("schema_version") != "c2c_static_intake_bundle_v1":
            return None
        validity = bundle.get("validity") if isinstance(bundle.get("validity"), dict) else {}
        if expected_validity and validity.get("fingerprint") != expected_validity.get("fingerprint"):
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
            "baseline",
            "repo_card",
            "paper_cards",
            "rebuttal_matrix",
            "code_cards",
            "negative_memory",
            "retrieval_plan",
            "followup_bundle",
        ]:
            if key not in bundle:
                return None
        return bundle

    def _register_or_restore_cached_c2c_artifacts(self, bundle: dict[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for spec in _c2c_static_artifact_specs():
            path = self.context.project_root / spec["path"]
            if not path.exists():
                payload = _cached_c2c_artifact_payload(bundle, spec)
                if payload is None:
                    continue
                if spec["format"] == "json":
                    records.append(
                        self.context.artifacts.write_json(
                            self.stage_key,
                            spec["relative_path"],
                            payload,
                            artifact_type=spec["artifact_type"],
                            summary=spec["summary"],
                            source_paths=["intake/c2c/static_bundle.json"],
                            metadata={"cache_status": "restored_from_static_bundle"},
                        )
                    )
                else:
                    records.append(
                        self.context.artifacts.write_text(
                            self.stage_key,
                            spec["relative_path"],
                            str(payload),
                            artifact_type=spec["artifact_type"],
                            summary=spec["summary"],
                            source_paths=["intake/c2c/static_bundle.json"],
                            metadata={"cache_status": "restored_from_static_bundle"},
                        )
                    )
                continue
            records.append(
                self.context.artifacts.register_artifact(
                    self.stage_key,
                    path,
                    artifact_type=spec["artifact_type"],
                    summary=spec["summary"],
                    source_paths=["intake/c2c/static_bundle.json"] if spec["path"] != "intake/c2c/static_bundle.json" else [],
                    metadata={"cache_status": "reused_existing_file"},
                )
            )
        return records


def _jsonl(items: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + ("\n" if items else "")


def _merge_shared_method_memory_into_negative_memory(negative_memory: dict[str, Any], shared_memory: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(negative_memory, dict) or not isinstance(shared_memory, dict):
        return negative_memory
    entries = [item for item in shared_memory.get("entries") or [] if isinstance(item, dict)]
    catalog = [item for item in shared_memory.get("memory_catalog") or [] if isinstance(item, dict)]
    if not entries and not catalog:
        negative_memory.setdefault("shared_method_memory", {"enabled": bool(shared_memory.get("enabled")), "entry_count": 0})
        return negative_memory
    blocked = list(negative_memory.get("blocked_idea_patterns") or [])
    for entry in entries[:12]:
        summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
        for rule in summary.get("avoid_repeat_rules") or []:
            if rule and rule not in blocked:
                blocked.append(str(rule))
        for dataset in (summary.get("dragging_datasets") or [])[:4]:
            if isinstance(dataset, dict) and dataset.get("dataset"):
                rule = f"Shared method memory: avoid repeating mechanisms that regress {dataset.get('dataset')} without an explicit repair."
                if rule not in blocked:
                    blocked.append(rule)
    negative_memory["blocked_idea_patterns"] = blocked
    negative_memory["shared_method_memory"] = {
        "enabled": bool(shared_memory.get("enabled")),
        "path": shared_memory.get("path"),
        "entry_count": shared_memory.get("entry_count", 0),
        "ranking_policy": shared_memory.get("ranking_policy") or {},
        "retrieval_policy": shared_memory.get("retrieval_policy") or {},
        "retrieval_context": shared_memory.get("retrieval_context") or {},
        "quality_summary": shared_memory.get("quality_summary") or {},
        "high_quality_memory_ids": (shared_memory.get("quality_summary") or {}).get("high_quality_memory_ids") or [],
        "full_memory_access": shared_memory.get("full_memory_access") or {},
        "memory_catalog": shared_memory.get("memory_catalog") or [],
        "recent_entries": shared_memory.get("memory_catalog") or [],
    }
    return negative_memory


def _evidence_brief_with_shared_method_memory(evidence_brief: dict[str, Any], shared_memory: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence_brief, dict):
        evidence_brief = {}
    updated = dict(evidence_brief)
    updated["shared_method_memory"] = {
        "enabled": bool(shared_memory.get("enabled")) if isinstance(shared_memory, dict) else False,
        "path": shared_memory.get("path") if isinstance(shared_memory, dict) else None,
        "entry_count": shared_memory.get("entry_count", 0) if isinstance(shared_memory, dict) else 0,
        "ranking_policy": shared_memory.get("ranking_policy", {}) if isinstance(shared_memory, dict) else {},
        "retrieval_policy": shared_memory.get("retrieval_policy", {}) if isinstance(shared_memory, dict) else {},
        "retrieval_context": shared_memory.get("retrieval_context", {}) if isinstance(shared_memory, dict) else {},
        "quality_summary": shared_memory.get("quality_summary", {}) if isinstance(shared_memory, dict) else {},
        "high_quality_memory_ids": (shared_memory.get("quality_summary", {}) or {}).get("high_quality_memory_ids", []) if isinstance(shared_memory, dict) else [],
        "full_memory_access": shared_memory.get("full_memory_access", {}) if isinstance(shared_memory, dict) else {},
        "memory_catalog": shared_memory.get("memory_catalog", []) if isinstance(shared_memory, dict) else [],
    }
    return updated


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
    return [spec["path"] for spec in _c2c_static_artifact_specs()]


def _c2c_static_artifact_specs() -> list[dict[str, Any]]:
    return [
        _artifact_spec("intake/papers/metadata.json", "metadata", "C2C configured reference materials", bundle_key="metadata"),
        _artifact_spec("intake/c2c/static_bundle.json", "c2c_static_bundle", "C2C static S0 evidence bundle", bundle_key=None),
        _artifact_spec("intake/c2c/paper_full_manifest.json", "c2c_paper_full_manifest", "MinerU paper_full.md outputs for C2C PDF references", bundle_key="paper_full_manifest"),
        _artifact_spec("intake/c2c/repo_manifest.json", "c2c_repo_manifest", "C2C repo intake manifest", bundle_key="repo_manifest"),
        _artifact_spec("intake/c2c/repo_card.json", "c2c_repo_card", "C2C repo capabilities, constraints, and evidence inventory", bundle_key="repo_card"),
        _artifact_spec("intake/c2c/historical_results.json", "c2c_historical_results", "Imported C2C historical experiment evidence", bundle_key="historical_results"),
        _artifact_spec("intake/c2c/result_ledger.csv", "c2c_result_ledger", "Normalized C2C historical result ledger", bundle_key="result_ledger_csv", file_format="text"),
        _artifact_spec("intake/c2c/baseline_evidence.json", "c2c_baseline_evidence", "C2C baseline target to beat", bundle_key="baseline"),
        _artifact_spec("intake/c2c/paper_cards.json", "c2c_paper_cards", "Structured cards for configured C2C-area papers", bundle_key="paper_cards"),
        _artifact_spec("intake/c2c/paper_chunks.jsonl", "c2c_paper_chunks", "Section-aware C2C paper chunks without bibliography text", bundle_key="paper_chunks", file_format="jsonl"),
        _artifact_spec("intake/c2c/bibliography.json", "c2c_bibliography", "Reference sections preserved separately for related-work expansion", bundle_key="bibliography_cards"),
        _artifact_spec("intake/c2c/rebuttal_concern_matrix.json", "c2c_rebuttal_concerns", "Reviewer concern matrix parsed from rebuttal materials", bundle_key="rebuttal_matrix"),
        _artifact_spec("intake/c2c/rebuttal_chunks.jsonl", "c2c_rebuttal_chunks", "Review and rebuttal chunks with source anchors", bundle_key="rebuttal_chunks", file_format="jsonl"),
        _artifact_spec("intake/c2c/code_cards.json", "c2c_code_cards", "Core C2C code symbols, config knobs, and imports", bundle_key="code_cards"),
        _artifact_spec("intake/c2c/code_file_manifest.json", "c2c_code_file_manifest", "Tree-sitter code file manifest and edit surface", bundle_key="code_file_manifest"),
        _artifact_spec("intake/c2c/code_symbols.jsonl", "c2c_code_symbols", "Tree-sitter code symbols for repository map", bundle_key="code_symbols", file_format="jsonl"),
        _artifact_spec("intake/c2c/code_chunks.jsonl", "c2c_code_chunks", "Function-level source chunks for S1/S2 reasoning", bundle_key="code_chunks", file_format="jsonl"),
        _artifact_spec("intake/c2c/code_edges.jsonl", "c2c_code_edges", "Static code relation graph edges", bundle_key="code_edges", file_format="jsonl"),
        _artifact_spec("intake/c2c/code_repo_map.json", "c2c_code_repo_map", "Compact tree-sitter repository map", bundle_key="code_repo_map"),
        _artifact_spec("intake/c2c/code_repo_map.md", "c2c_code_repo_map_markdown", "Readable tree-sitter repository map", bundle_key="code_repo_map", file_format="repo_map_md"),
        _artifact_spec("intake/c2c/code_intake_report.json", "c2c_code_intake_report", "Code intake coverage and quality diagnostics", bundle_key="code_intake_report"),
        _artifact_spec("intake/c2c/code_intake_report.md", "c2c_code_intake_report_markdown", "Readable code intake coverage report", bundle_key="code_intake_report", file_format="code_intake_report_md"),
        _artifact_spec("intake/c2c/implementation_surface_map.json", "c2c_implementation_surface_map", "Editable mechanism surfaces for S2.5 patches", bundle_key="implementation_surface_map"),
        _artifact_spec("intake/c2c/code_retrieval_index.json", "c2c_code_retrieval_index", "Precomputed code retrieval entry points for S1/S2", bundle_key="code_retrieval_index"),
        _artifact_spec("intake/c2c/cache_summary.json", "c2c_static_cache_summary", "S0 cache hit/miss summary for MinerU and code intake", bundle_key="cache_summary"),
        _artifact_spec("intake/c2c/chunk_index.json", "c2c_chunk_index", "Full S0 chunk catalog for paper, rebuttal, and code retrieval", bundle_key="chunk_index"),
        _artifact_spec("intake/c2c/chunk_index.jsonl", "c2c_chunk_index_jsonl", "Line-delimited S0 chunk catalog for retrieval", bundle_key="chunk_index", file_format="chunk_index_jsonl"),
        _artifact_spec("intake/c2c/retrieval_plan.json", "c2c_retrieval_plan", "Chunk retrieval plan for paper, rebuttal, and code evidence", bundle_key="retrieval_plan"),
        _artifact_spec("intake/c2c/retrieval_followup.json", "c2c_retrieval_followup", "Follow-up retrieval questions and targets", bundle_key="followup_bundle"),
        _artifact_spec("intake/c2c/negative_result_memory.json", "c2c_negative_result_memory", "Static below-baseline local variants and avoid-repeat rules", bundle_key="negative_memory"),
        _artifact_spec("intake/c2c/evidence_brief.json", "c2c_evidence_brief", "Compact static evidence brief for S1 direction selection", bundle_key="evidence_brief"),
    ]


def _artifact_spec(
    path: str,
    artifact_type: str,
    summary: str,
    *,
    bundle_key: str | None,
    file_format: str = "json",
) -> dict[str, Any]:
    return {
        "path": path,
        "relative_path": path.split("/", 1)[1],
        "artifact_type": artifact_type,
        "summary": summary,
        "bundle_key": bundle_key,
        "format": file_format,
    }


def _cached_c2c_artifact_payload(bundle: dict[str, Any], spec: dict[str, Any]) -> Any:
    bundle_key = spec.get("bundle_key")
    if bundle_key is None:
        return bundle
    if bundle_key not in bundle:
        return None
    value = bundle.get(bundle_key)
    file_format = spec.get("format")
    if file_format == "json":
        return value
    if file_format == "text":
        return str(value or "")
    if file_format == "jsonl":
        return _jsonl(value if isinstance(value, list) else [])
    if file_format == "chunk_index_jsonl":
        entries = value.get("entries") if isinstance(value, dict) else []
        return _jsonl(entries if isinstance(entries, list) else [])
    if file_format == "repo_map_md":
        return _repo_map_markdown_from_cached(value)
    if file_format == "code_intake_report_md":
        return _code_intake_report_markdown_from_cached(value)
    return value


def _repo_map_markdown_from_cached(repo_map: Any) -> str:
    if not isinstance(repo_map, dict):
        return "# Code Repo Map\n\nNo cached repo map available.\n"
    counts = repo_map.get("counts") if isinstance(repo_map.get("counts"), dict) else {}
    lines = ["# Code Repo Map", ""]
    if counts:
        lines.append(f"- Files: {counts.get('files', 0)}")
        lines.append(f"- Symbols: {counts.get('symbols', 0)}")
        lines.append(f"- Chunks: {counts.get('chunks', 0)}")
        lines.append(f"- Edges: {counts.get('edges', 0)}")
        lines.append("")
    for item in (repo_map.get("top_editable_symbols") or [])[:40]:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol") or item.get("symbol_id") or ""
        path = item.get("path") or ""
        lines.append(f"- {path}: {symbol}")
    return "\n".join(lines).rstrip() + "\n"


def _code_intake_report_markdown_from_cached(report: Any) -> str:
    if not isinstance(report, dict):
        return "# Code Intake Report\n\nNo cached code intake report available.\n"
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    cache = report.get("cache") if isinstance(report.get("cache"), dict) else {}
    diagnostics = report.get("diagnostics") if isinstance(report.get("diagnostics"), list) else []
    lines = ["# Code Intake Report", ""]
    if counts:
        lines.append("## Counts")
        for key in ["files", "python_files", "symbols", "chunks", "edges", "editable_chunks"]:
            if key in counts:
                lines.append(f"- {key}: {counts.get(key)}")
        lines.append("")
    lines.append(f"- Cache: enabled={bool(cache.get('enabled'))} counts={cache.get('counts') or {}}")
    if diagnostics:
        lines.append("")
        lines.append("## Diagnostics")
        for item in diagnostics[:20]:
            lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def _c2c_static_bundle_validity(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    raw_c2c_cfg = config.get("c2c", {}) if isinstance(config.get("c2c"), dict) else {}
    intake_cfg = config.get("intake", {}) if isinstance(config.get("intake"), dict) else {}
    adapter = C2CAdapter(project_root, {**config, "c2c": raw_c2c_cfg})
    allowed_files = sorted(str(item).strip("/") for item in adapter.allowed_files if item)
    allowed_prefixes = sorted(str(item).strip("/") for item in adapter.allowed_prefixes if item)
    payload = {
        "schema_version": "c2c_static_bundle_validity_v1",
        "ref_paper": _path_fingerprint(_resolve_config_path(project_root, raw_c2c_cfg.get("ref_paper"))),
        "ref_rebuttal": _path_fingerprint(_resolve_config_path(project_root, raw_c2c_cfg.get("ref_rebuttal"))),
        "repo_edit_surface": _c2c_repo_edit_surface_fingerprint(adapter.repo_root, allowed_files, allowed_prefixes),
        "allowed_files": allowed_files,
        "allowed_prefixes": allowed_prefixes,
        "baseline": adapter.baseline,
        "datasets": raw_c2c_cfg.get("datasets") or [],
        "pdf_ingest_hash": _stable_hash(adapter.pdf_ingest_config),
        "semantic_enrichment_hash": _stable_hash((intake_cfg.get("semantic_enrichment") or {}) if isinstance(intake_cfg, dict) else {}),
    }
    payload["fingerprint"] = _stable_hash(payload)
    return payload


def _resolve_config_path(project_root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def _c2c_repo_edit_surface_fingerprint(snapshot: Path, allowed_files: list[str], allowed_prefixes: list[str]) -> dict[str, Any]:
    if not snapshot.exists():
        return {"status": "missing", "path": str(snapshot)}
    paths: list[Path] = []
    for rel in allowed_files:
        if not rel:
            continue
        paths.append(snapshot / str(rel).strip("/"))
    for prefix in allowed_prefixes:
        if not prefix:
            continue
        root = snapshot / str(prefix).strip("/")
        if root.exists():
            paths.extend(child for child in root.rglob("*") if child.is_file())
    if not paths:
        for rel in [
            "rosetta/model/aligner.py",
            "rosetta/model/projector.py",
            "rosetta/model/wrapper.py",
            "script/train/SFT_train.py",
            "script/evaluation/unified_evaluator.py",
            "recipe/train_recipe/C2C_0.6+0.5.json",
            "recipe/eval_recipe/unified_eval.yaml",
        ]:
            paths.append(snapshot / rel)
    return _file_collection_fingerprint(snapshot, paths)


def _path_fingerprint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_configured"}
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    if path.is_file():
        return {
            "status": "file",
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    files = [
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in {".pdf", ".md", ".txt", ".json", ".yaml", ".yml"}
    ]
    return _file_collection_fingerprint(path, files)


def _file_collection_fingerprint(root: Path, files: list[Path]) -> dict[str, Any]:
    seen: dict[str, Path] = {}
    for path in files:
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = str(path)
        if _skip_fingerprint_path(rel):
            continue
        seen[rel] = path
    records = []
    for rel, path in sorted(seen.items()):
        if not path.exists() or not path.is_file():
            records.append({"path": rel, "exists": False})
            continue
        records.append({"path": rel, "exists": True, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "status": "directory" if root.is_dir() else "collection",
        "path": str(root),
        "file_count": len(records),
        "hash": _stable_hash(records),
        "files": records[:200],
    }


def _skip_fingerprint_path(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/").strip("/")
    excluded_prefixes = (
        ".git/",
        "wandb/",
        "__pycache__/",
        ".pytest_cache/",
        "local/checkpoints/",
        "local/snapshots/",
        "local/final_results/",
        "data/",
        "datasets/",
        "models/",
    )
    excluded_suffixes = (".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".parquet", ".arrow")
    return normalized.startswith(excluded_prefixes) or normalized.endswith(excluded_suffixes)


def _stable_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    editable_surface = repo_card.get("editable_surface") or {
        "allowed_files": repo_card.get("allowed_files") or repo_card.get("allowed_surface") or [],
        "allowed_prefixes": repo_card.get("allowed_prefixes") or [],
    }
    protocol_constraints = repo_card.get("protocol_constraints") or repo_card.get("constraints") or []
    retrieval_questions = [
        {
            "question_id": item.get("question_id"),
            "question": item.get("question"),
            "priority_terms": (item.get("priority_terms") or [])[:12],
        }
        for item in (retrieval_plan.get("questions") or retrieval_plan.get("primary_questions") or [])[:8]
        if isinstance(item, dict)
    ]
    cross_source_targets = _followup_cross_source_targets(followup_bundle)
    return {
        "schema_version": "c2c_evidence_brief_v1",
        "topic": topic,
        "baseline_to_beat": baseline,
        "repo_summary": {
            "editable_surface": editable_surface,
            "allowed_surface": editable_surface,
            "baseline_surface": repo_card.get("baseline_surface") or {},
            "protocol_constraints": protocol_constraints,
            "constraints": protocol_constraints,
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
            "questions": retrieval_questions,
            "primary_questions": retrieval_questions,
            "cross_source_targets": cross_source_targets[:12],
        },
        "static_rules": [
            "Use S0 evidence as stable context; do not re-import papers, rebuttals, or repo cards in S1.",
            "S1 should choose a mechanism direction; S2 owns concrete experiment planning.",
            "Avoid pure threshold/top-k/fallback tuning and evaluator changes.",
        ],
    }


def _followup_cross_source_targets(followup_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    direct = followup_bundle.get("cross_source_targets") if isinstance(followup_bundle, dict) else []
    if isinstance(direct, list) and direct:
        return [item for item in direct if isinstance(item, dict)]
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for question in followup_bundle.get("questions") or []:
        if not isinstance(question, dict):
            continue
        for item in question.get("cross_source_targets") or []:
            if not isinstance(item, dict):
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            targets.append(item)
    return targets


def _c2c_cache_summary(paper_full_manifest: list[dict[str, Any]], code_intake_report: dict[str, Any], *, pdf_ingest_config: dict[str, Any] | None = None) -> dict[str, Any]:
    mineru_counts: dict[str, int] = {}
    for item in paper_full_manifest:
        if not isinstance(item, dict):
            continue
        status = str(item.get("cache_status") or "disabled")
        mineru_counts[status] = mineru_counts.get(status, 0) + 1
    code_cache = (code_intake_report.get("cache") or {}) if isinstance(code_intake_report, dict) else {}
    code_fingerprint = code_intake_report.get("parser_fingerprint") if isinstance(code_intake_report, dict) else {}
    mineru_provenance = _mineru_cache_provenance(pdf_ingest_config or {}, paper_full_manifest)
    cache_fingerprint = _cache_summary_fingerprint(
        {
            "mineru": mineru_provenance,
            "code_parser_config_hash": (code_fingerprint or {}).get("parser_config_hash"),
            "code_chunking_config_hash": (code_fingerprint or {}).get("chunking_config_hash"),
            "code_intake_schema_version": (code_fingerprint or {}).get("code_intake_schema_version"),
        }
    )
    return {
        "schema_version": "c2c_static_cache_summary_v1",
        "cache_fingerprint": cache_fingerprint,
        "provenance": {
            "mineru_pdf": mineru_provenance,
            "code_intake": {
                "schema_version": (code_fingerprint or {}).get("code_intake_schema_version"),
                "prompt_schema_version": (code_fingerprint or {}).get("prompt_schema_version"),
                "tree_sitter_version": (code_fingerprint or {}).get("tree_sitter_version"),
                "tree_sitter_python_version": (code_fingerprint or {}).get("tree_sitter_python_version"),
                "parser_config_hash": (code_fingerprint or {}).get("parser_config_hash"),
                "chunking_config_hash": (code_fingerprint or {}).get("chunking_config_hash"),
            },
        },
        "mineru_pdf": {
            "counts": dict(sorted(mineru_counts.items())),
            "items": [
                {
                    "paper_id": item.get("paper_id"),
                    "sha256": item.get("sha256"),
                    "cache_status": item.get("cache_status", "disabled"),
                    "paper_full_md_path": item.get("paper_full_md_path"),
                    "parser": item.get("parser"),
                    "parser_status": item.get("parser_status"),
                    "parser_artifacts": item.get("parser_artifacts"),
                }
                for item in paper_full_manifest
                if isinstance(item, dict)
            ],
        },
        "code_intake": {
            "enabled": bool(code_cache.get("enabled")),
            "counts": code_cache.get("counts") or {},
            "parser_config_hash": code_cache.get("parser_config_hash"),
            "chunking_config_hash": code_cache.get("chunking_config_hash"),
        },
    }


def _mineru_cache_provenance(pdf_ingest_config: dict[str, Any], paper_full_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    provider = str(pdf_ingest_config.get("provider") or "mineru")
    model_versions = sorted({str(item.get("model_version")) for item in paper_full_manifest if isinstance(item, dict) and item.get("model_version")})
    return {
        "provider": provider,
        "schema_version": "mineru_pdf_parse_result_v1",
        "model_version": str(pdf_ingest_config.get("model_version") or (model_versions[0] if len(model_versions) == 1 else "vlm")),
        "language": str(pdf_ingest_config.get("language") or "en"),
        "enable_formula": bool(pdf_ingest_config.get("enable_formula", True)),
        "enable_table": bool(pdf_ingest_config.get("enable_table", True)),
        "is_ocr": bool(pdf_ingest_config.get("is_ocr", False)),
        "prompt_schema_version": "c2c_paper_full_markdown_v1",
        "parser_config_hash": _cache_summary_fingerprint(
            {
                "provider": provider,
                "model_version": str(pdf_ingest_config.get("model_version") or "vlm"),
                "language": str(pdf_ingest_config.get("language") or "en"),
                "enable_formula": bool(pdf_ingest_config.get("enable_formula", True)),
                "enable_table": bool(pdf_ingest_config.get("enable_table", True)),
                "is_ocr": bool(pdf_ingest_config.get("is_ocr", False)),
            }
        ),
    }


def _cache_summary_fingerprint(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
