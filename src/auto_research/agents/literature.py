"""Literature stage."""

from __future__ import annotations

import json
import hashlib
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..c2c import (
    build_c2c_ideas_with_llm,
    is_c2c_project,
)
from ..config import bootstrap_profile_enabled, bootstrap_profile_options
from ..adapters.literature import LiteratureProvider
from ..direction_contracts import (
    build_direction_contract,
    build_direction_scorecard,
    build_s1_direction_fingerprint,
    build_s1_evidence_quality_score,
    build_s1_evidence_retrieval_trace,
    normalize_novelty_audit,
)
from ..research_state import ResearchEventLedger
from ..evidence_refs import (
    direction_bundle_ref_errors_for_repair,
    evidence_ref_errors_for_repair,
    resolve_s1_evidence_refs,
    validate_direction_refs_subset_of_bundle,
)
from ..importers import ConsensusImporter
from ..itr_ideas import build_itr_theme_map, collect_consensus_entries, theme_map_markdown
from ..failure_log import load_c2c_feedback_bundle
from ..method_memory import collect_used_shared_memory_refs, shared_method_memory_for_prompt, shared_method_memory_query_context
from ..llm import codex_subprocess_env
from ..resources import discover_local_mm_resources
from ..shared_cache import shared_cache_root
from ..s0_enrichment import (
    DEFAULT_DEEPSEEK_MODEL,
    S0_CODE_SEMANTIC_ENRICHMENT_PROMPT_VERSION,
    S0_SEMANTIC_ENRICHMENT_PROMPT_VERSION,
    S0_SEMANTIC_ENRICHMENT_SCHEMA_VERSION,
)
from ..s1_retrieval import (
    bundle_ref_set,
    canonical_ref_key,
    default_c2c_evidence_request_plan,
    normalize_c2c_evidence_request_plan,
    retrieve_s1_c2c_requested_evidence,
    validate_c2c_evidence_request_plan,
)
from ..utils import compact_markdown, now_utc, read_json, read_yaml, sanitize_filename, split_sentences, write_yaml
from .base import AgentContext
from .debate import c2c_debate_markdown


class LiteratureAgent:
    stage_key = "S1_literature"

    def __init__(self, context: AgentContext):
        self.context = context
        self.provider = LiteratureProvider(context.config)

    def run(self, topic: str, *, phase: str = "full") -> dict[str, Any]:
        if phase == "related_work_audit":
            return self._run_related_work_audit()
        return self._run_literature(topic)

    def _run_literature(self, topic: str) -> dict[str, Any]:
        if is_c2c_project(self.context.config):
            return self._run_c2c_literature(topic)
        papers = self.provider.search(topic)
        papers.extend(self._search_consensus_imports())
        papers = self.provider._deduplicate(papers)
        if not papers and self.context.config.get("experiment", {}).get("simulate"):
            papers = self._mock_papers(topic)

        ingested = []
        for paper in papers:
            enriched = dict(paper)
            enriched["download_status"] = "not_attempted"
            pdf_url = paper.get("pdf_url")
            if pdf_url and self.context.config.get("literature", {}).get("download_pdfs", True):
                pdf_bytes = self.provider.download_pdf(pdf_url)
                if pdf_bytes:
                    filename = self.provider.pdf_filename(paper)
                    record = self.context.artifacts.write_reference_pdf(filename, pdf_bytes, metadata=enriched)
                    enriched["download_status"] = "downloaded"
                    enriched["local_pdf_path"] = record["local_pdf_path"]
                else:
                    enriched["download_status"] = "failed"
            ingested.append(enriched)

        if self.context.config.get("experiment", {}).get("simulate"):
            ingested = self._ensure_placeholder_pdfs(ingested)

        metadata_record = self.context.artifacts.write_json(
            self.stage_key,
            "papers/metadata.json",
            ingested,
            artifact_type="metadata",
            summary=f"{len(ingested)} literature records",
        )
        survey = self._build_survey(topic, ingested)
        survey_record = self.context.artifacts.write_text(
            self.stage_key,
            "survey.md",
            survey,
            artifact_type="survey",
            summary="Structured survey",
            source_paths=[metadata_record["path"]],
        )
        theme_map_record = None
        imports = ConsensusImporter(self.context.project_root).list_imports()
        imported_entries = collect_consensus_entries(imports)
        evidence_records: list[dict[str, Any]] = []
        if imported_entries:
            theme_map = build_itr_theme_map(imported_entries)
            theme_map_record = self.context.artifacts.write_text(
                self.stage_key,
                "theme_map.md",
                theme_map_markdown(theme_map),
                artifact_type="theme_map",
                summary="Consensus-derived image-text retrieval theme map",
                source_paths=[metadata_record["path"]],
            )
        evidence_result = self._run_generic_evidence_direction(topic=topic, papers=ingested, survey=survey, theme_map_path=theme_map_record["path"] if theme_map_record else None)
        if evidence_result.get("status") == "blocked":
            blocked_record = self.context.artifacts.write_json(
                self.stage_key,
                "evidence_agent_blocked.json",
                evidence_result,
                artifact_type="s1_evidence_agent_blocked",
                summary="S1 Codex evidence agent could not produce valid idea JSON",
                source_paths=[metadata_record["path"], survey_record["path"], *( [theme_map_record["path"]] if theme_map_record else [] )],
            )
            return {
                "papers": ingested,
                "artifacts": [metadata_record["path"], survey_record["path"], *( [theme_map_record["path"]] if theme_map_record else [] ), blocked_record["path"]],
                "status": "blocked",
                "blocked_reason": evidence_result.get("blocked_reason") or evidence_result.get("reason") or "S1 Codex evidence agent blocked",
            }
        ideas = evidence_result["ideas"]
        evidence_requests_record = self.context.artifacts.write_json(
            self.stage_key,
            "evidence_requests.json",
            evidence_result.get("evidence_requests", []),
            artifact_type="s1_evidence_requests",
            summary="S1 requested evidence before idea choice",
            source_paths=[metadata_record["path"], survey_record["path"]],
        )
        evidence_bundle_record = self.context.artifacts.write_json(
            self.stage_key,
            "evidence_bundle.json",
            evidence_result.get("evidence_bundle", {}),
            artifact_type="s1_evidence_bundle",
            summary="S1 evidence bundle for idea choice",
            source_paths=[evidence_requests_record["path"], metadata_record["path"], survey_record["path"]],
        )
        direction_record = self.context.artifacts.write_json(
            self.stage_key,
            "direction.json",
            evidence_result.get("direction", {}),
            artifact_type="s1_direction",
            summary="S1 selected high-level research direction",
            source_paths=[evidence_bundle_record["path"]],
        )
        direction_decision_record = self.context.artifacts.write_json(
            self.stage_key,
            "direction_selection_report.json",
            evidence_result.get("direction_decision", {}),
            artifact_type="s1_direction_decision",
            summary="Compatibility mirror for the S1 selected direction",
            source_paths=[direction_record["path"], evidence_bundle_record["path"]],
        )
        novelty_record = self.context.artifacts.write_json(
            self.stage_key,
            "novelty_audit.json",
            evidence_result.get("novelty_audit", {}),
            artifact_type="s1_novelty_audit",
            summary="Normalized S1 novelty audit for the selected direction",
            source_paths=[direction_record["path"], evidence_bundle_record["path"]],
        )
        scorecard_record = self.context.artifacts.write_json(
            self.stage_key,
            "direction_scorecard.json",
            evidence_result.get("direction_scorecard", {}),
            artifact_type="s1_direction_scorecard",
            summary="S1 direction readiness scorecard",
            source_paths=[direction_record["path"], novelty_record["path"], evidence_bundle_record["path"]],
        )
        evidence_session_record = self.context.artifacts.write_json(
            self.stage_key,
            "evidence_session.json",
            evidence_result.get("evidence_session", {}),
            artifact_type="s1_evidence_session",
            summary="S1 Codex evidence-on-demand session transcript",
            source_paths=[evidence_requests_record["path"], evidence_bundle_record["path"], direction_record["path"]],
        )
        evidence_records = [
            evidence_requests_record,
            evidence_bundle_record,
            direction_record,
            direction_decision_record,
            novelty_record,
            scorecard_record,
            evidence_session_record,
        ]
        evidence_ref_report_record = self.context.artifacts.write_json(
            self.stage_key,
            "evidence_ref_report.json",
            evidence_result.get("evidence_ref_report", {}),
            artifact_type="s1_evidence_ref_report",
            summary="Resolved S1 evidence refs against local artifacts and indexes",
            source_paths=[evidence_bundle_record["path"], direction_record["path"]],
        )
        evidence_records.append(evidence_ref_report_record)
        ideas_record = self.context.artifacts.write_json(
            self.stage_key,
            "candidate_directions.json",
            ideas,
            artifact_type="ideas",
            summary=f"{len(ideas)} candidate ideas",
            source_paths=[
                metadata_record["path"],
                survey_record["path"],
                *( [theme_map_record["path"]] if theme_map_record else [] ),
                *[record["path"] for record in evidence_records],
            ],
        )
        feasibility = self._build_feasibility(ideas)
        feasibility_record = self.context.artifacts.write_text(
            self.stage_key,
            "feasibility_check.md",
            feasibility,
            artifact_type="feasibility",
            summary="Quick feasibility note",
            source_paths=[ideas_record["path"]],
        )
        return {
            "papers": ingested,
            "artifacts": [
                metadata_record["path"],
                survey_record["path"],
                *( [theme_map_record["path"]] if theme_map_record else [] ),
                *[record["path"] for record in evidence_records],
                ideas_record["path"],
                feasibility_record["path"],
            ],
        }

    def _run_c2c_literature(self, topic: str) -> dict[str, Any]:
        static_bundle = self._load_c2c_static_bundle()
        if not static_bundle:
            blocked_record = self.context.artifacts.write_json(
                self.stage_key,
                "c2c_missing_s0_intake.json",
                {
                    "status": "blocked",
                    "reason": "Missing S0 C2C static intake bundle. Run S0_intake before S1_literature.",
                    "required_path": "intake/c2c/static_bundle.json",
                },
                artifact_type="c2c_missing_s0_intake",
                summary="C2C S1 blocked because S0 static intake is missing",
            )
            return {
                "papers": [],
                "artifacts": [blocked_record["path"]],
                "status": "blocked",
                "blocked_reason": "Missing S0 C2C static intake bundle: intake/c2c/static_bundle.json",
            }

        reference_result = static_bundle.get("reference_result") or {}
        reference_cards = reference_result.get("cards") or []
        repo_manifest = static_bundle.get("repo_manifest") or {}
        historical_results = static_bundle.get("historical_results") or {}
        baseline = static_bundle.get("baseline") or {}
        repo_card = static_bundle.get("repo_card") or {}
        paper_cards = static_bundle.get("paper_cards") or []
        paper_chunks = static_bundle.get("paper_chunks") or []
        bibliography_cards = static_bundle.get("bibliography_cards") or []
        rebuttal_matrix = static_bundle.get("rebuttal_matrix") or {}
        rebuttal_chunks = static_bundle.get("rebuttal_chunks") or []
        code_cards = static_bundle.get("code_cards") or []
        code_file_manifest = static_bundle.get("code_file_manifest") or {}
        code_symbols = static_bundle.get("code_symbols") or []
        code_chunks = static_bundle.get("code_chunks") or []
        code_edges = static_bundle.get("code_edges") or []
        code_repo_map = static_bundle.get("code_repo_map") or {}
        code_intake_report = static_bundle.get("code_intake_report") or {}
        implementation_surface_map = static_bundle.get("implementation_surface_map") or {}
        code_retrieval_index = static_bundle.get("code_retrieval_index") or {}
        cache_summary = static_bundle.get("cache_summary") or {}
        chunk_index = static_bundle.get("chunk_index") or {}
        result_ledger_csv = static_bundle.get("result_ledger_csv") or ""
        negative_memory = static_bundle.get("negative_memory") or {}
        retrieval_plan = static_bundle.get("retrieval_plan") or {}
        followup_bundle = static_bundle.get("followup_bundle") or {}
        metadata = static_bundle.get("metadata") or [
            {
                "paper_id": card.get("paper_id"),
                "title": card.get("title"),
                "source_path": card.get("source_path"),
                "local_path": card.get("local_path"),
                "kind": card.get("kind"),
                "text_snippet": (card.get("text") or "")[:1200],
            }
            for card in reference_cards
        ]
        enrichment_merge = _merge_s0_semantic_enrichment_for_s1(
            self.context.project_root,
            paper_chunks=paper_chunks,
            rebuttal_chunks=rebuttal_chunks,
            code_chunks=code_chunks,
            chunk_index=chunk_index,
            config=self.context.config,
        )
        paper_chunks = enrichment_merge["paper_chunks"]
        rebuttal_chunks = enrichment_merge["rebuttal_chunks"]
        code_chunks = enrichment_merge["code_chunks"]
        chunk_index = enrichment_merge["chunk_index"]
        semantic_merge_report = enrichment_merge["report"]
        feedback = self._load_feedback()
        if self.context.config.get("ideation", {}).get("debate", {}).get("enabled", True):
            ideas = []
        else:
            ideas = build_c2c_ideas_with_llm(
                llm=self.context.llm,
                topic=topic,
                repo_manifest=repo_manifest,
                baseline=baseline,
                reference_cards=reference_cards,
                rebuttal_concerns=rebuttal_matrix,
                negative_memory=negative_memory,
            )

        metadata_record = self.context.artifacts.write_json(
            self.stage_key,
            "papers/metadata.json",
            metadata,
            artifact_type="metadata",
            summary="C2C configured reference materials",
            source_paths=["intake/c2c/static_bundle.json"],
        )
        repo_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/repo_manifest.json",
            repo_manifest,
            artifact_type="c2c_repo_manifest",
            summary="C2C repo intake manifest",
            source_paths=["intake/c2c/repo_manifest.json"],
        )
        repo_card_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/repo_card.json",
            repo_card,
            artifact_type="c2c_repo_card",
            summary="C2C repo capabilities, constraints, and evidence inventory",
            source_paths=["intake/c2c/repo_card.json"],
        )
        history_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/historical_results.json",
            historical_results,
            artifact_type="c2c_historical_results",
            summary="Imported C2C historical experiment evidence",
            source_paths=["intake/c2c/historical_results.json"],
        )
        result_ledger_record = self.context.artifacts.write_text(
            self.stage_key,
            "c2c/result_ledger.csv",
            result_ledger_csv,
            artifact_type="c2c_result_ledger",
            summary="Normalized C2C historical result ledger",
            source_paths=["intake/c2c/result_ledger.csv"],
        )
        baseline_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/baseline_evidence.json",
            baseline,
            artifact_type="c2c_baseline_evidence",
            summary="C2C baseline target to beat",
            source_paths=["intake/c2c/baseline_evidence.json"],
        )
        paper_cards_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/paper_cards.json",
            paper_cards,
            artifact_type="c2c_paper_cards",
            summary="Structured cards for configured C2C-area papers",
            source_paths=["intake/c2c/paper_cards.json"],
        )
        paper_chunks_record = self.context.artifacts.write_text(
            self.stage_key,
            "c2c/paper_chunks.jsonl",
            _jsonl(paper_chunks),
            artifact_type="c2c_paper_chunks",
            summary="Section-aware C2C paper chunks without bibliography text",
            source_paths=["intake/c2c/paper_chunks.jsonl"],
            metadata={"chunk_count": len(paper_chunks)},
        )
        bibliography_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/bibliography.json",
            bibliography_cards,
            artifact_type="c2c_bibliography",
            summary="Reference sections preserved separately for related-work expansion",
            source_paths=["intake/c2c/bibliography.json"],
        )
        rebuttal_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/rebuttal_concern_matrix.json",
            rebuttal_matrix,
            artifact_type="c2c_rebuttal_concerns",
            summary="Reviewer concern matrix parsed from rebuttal materials",
            source_paths=["intake/c2c/rebuttal_concern_matrix.json"],
        )
        rebuttal_chunks_record = self.context.artifacts.write_text(
            self.stage_key,
            "c2c/rebuttal_chunks.jsonl",
            _jsonl(rebuttal_chunks),
            artifact_type="c2c_rebuttal_chunks",
            summary="Review and rebuttal chunks with source anchors",
            source_paths=["intake/c2c/rebuttal_chunks.jsonl"],
            metadata={"chunk_count": len(rebuttal_chunks)},
        )
        code_cards_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/code_cards.json",
            code_cards,
            artifact_type="c2c_code_cards",
            summary="Core C2C code symbols, config knobs, and imports",
            source_paths=["intake/c2c/code_cards.json"],
        )
        code_file_manifest_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/code_file_manifest.json",
            code_file_manifest,
            artifact_type="c2c_code_file_manifest",
            summary="Tree-sitter code file manifest and edit surface",
            source_paths=["intake/c2c/code_file_manifest.json"],
        )
        code_symbols_record = self.context.artifacts.write_text(
            self.stage_key,
            "c2c/code_symbols.jsonl",
            _jsonl(code_symbols),
            artifact_type="c2c_code_symbols",
            summary="Tree-sitter code symbols for repository map",
            source_paths=["intake/c2c/code_symbols.jsonl"],
            metadata={"symbol_count": len(code_symbols)},
        )
        code_chunks_record = self.context.artifacts.write_text(
            self.stage_key,
            "c2c/code_chunks.jsonl",
            _jsonl(code_chunks),
            artifact_type="c2c_code_chunks",
            summary="Function-level source chunks for S1 reasoning",
            source_paths=["intake/c2c/code_chunks.jsonl"],
            metadata={"chunk_count": len(code_chunks)},
        )
        code_edges_record = self.context.artifacts.write_text(
            self.stage_key,
            "c2c/code_edges.jsonl",
            _jsonl(code_edges),
            artifact_type="c2c_code_edges",
            summary="Static code relation graph edges",
            source_paths=["intake/c2c/code_edges.jsonl"],
            metadata={"edge_count": len(code_edges)},
        )
        code_repo_map_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/code_repo_map.json",
            code_repo_map,
            artifact_type="c2c_code_repo_map",
            summary="Compact tree-sitter repository map",
            source_paths=["intake/c2c/code_repo_map.json"],
        )
        code_repo_map_md_record = self.context.artifacts.write_text(
            self.stage_key,
            "c2c/code_repo_map.md",
            _code_repo_map_markdown(code_repo_map),
            artifact_type="c2c_code_repo_map_markdown",
            summary="Readable tree-sitter repository map",
            source_paths=["intake/c2c/code_repo_map.md"],
        )
        code_intake_report_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/code_intake_report.json",
            code_intake_report,
            artifact_type="c2c_code_intake_report",
            summary="Code intake coverage and quality diagnostics",
            source_paths=["intake/c2c/code_intake_report.json"],
        )
        implementation_surface_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/implementation_surface_map.json",
            implementation_surface_map,
            artifact_type="c2c_implementation_surface_map",
            summary="Editable mechanism surfaces for S2.5 patches",
            source_paths=["intake/c2c/implementation_surface_map.json"],
        )
        code_retrieval_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/code_retrieval_index.json",
            code_retrieval_index,
            artifact_type="c2c_code_retrieval_index",
            summary="Precomputed code retrieval entry points for S1/S2",
            source_paths=["intake/c2c/code_retrieval_index.json"],
        )
        cache_summary_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/cache_summary.json",
            cache_summary,
            artifact_type="c2c_static_cache_summary",
            summary="S0 cache hit/miss summary for MinerU and code intake",
            source_paths=["intake/c2c/cache_summary.json"],
        )
        semantic_merge_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/semantic_enrichment_merge_report.json",
            semantic_merge_report,
            artifact_type="c2c_s1_semantic_enrichment_merge_report",
            summary="S1 merge report for S0 DeepSeek semantic enrichment records",
            source_paths=semantic_merge_report.get("source_paths") or ["intake/c2c/static_bundle.json"],
        )
        chunk_index_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/chunk_index.json",
            chunk_index,
            artifact_type="c2c_chunk_index",
            summary="Full S0 chunk catalog for paper, rebuttal, and code retrieval",
            source_paths=["intake/c2c/chunk_index.json"],
        )
        chunk_index_jsonl_record = self.context.artifacts.write_text(
            self.stage_key,
            "c2c/chunk_index.jsonl",
            _jsonl(chunk_index.get("entries", []) if isinstance(chunk_index, dict) else []),
            artifact_type="c2c_chunk_index_jsonl",
            summary="Line-delimited S0 chunk catalog for retrieval",
            source_paths=["intake/c2c/chunk_index.jsonl"],
            metadata={"chunk_count": (chunk_index.get("counts", {}) or {}).get("total", 0) if isinstance(chunk_index, dict) else 0},
        )
        retrieval_plan_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/retrieval_plan.json",
            retrieval_plan,
            artifact_type="c2c_retrieval_plan",
            summary="Chunk retrieval plan for paper, rebuttal, and code evidence",
            source_paths=["intake/c2c/retrieval_plan.json"],
        )
        followup_bundle_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/retrieval_followup.json",
            followup_bundle,
            artifact_type="c2c_retrieval_followup",
            summary="Follow-up retrieval questions and targets",
            source_paths=["intake/c2c/retrieval_followup.json"],
        )
        negative_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/negative_result_memory.json",
            negative_memory,
            artifact_type="c2c_negative_result_memory",
            summary="Below-baseline local variants and avoid-repeat rules",
            source_paths=["intake/c2c/negative_result_memory.json"],
        )
        debate_record = None
        debate_md_record = None
        constraints_record = None
        evidence_request_plan_record = None
        evidence_requests_record = None
        evidence_bundle_record = None
        direction_record = None
        evidence_session_record = None
        evidence_ref_report_record = None
        direction_candidate_scorecard_record = None
        novelty_record = None
        evidence_quality_record = None
        evidence_retrieval_trace_record = None
        direction_fingerprint_record = None
        debate: dict[str, Any] = {}
        if self.context.config.get("ideation", {}).get("debate", {}).get("enabled", True):
            debate = self._run_c2c_evidence_on_demand_direction(
                    topic=topic,
                    evidence_brief=static_bundle.get("evidence_brief") or {},
                    chunk_index=chunk_index,
                    paper_chunks=paper_chunks,
                    rebuttal_chunks=rebuttal_chunks,
                    code_chunks=code_chunks,
                    code_edges=code_edges,
                    code_intake_report=code_intake_report,
                    implementation_surface_map=implementation_surface_map,
                    code_retrieval_index=code_retrieval_index,
                    baseline=baseline,
                    negative_memory=negative_memory,
                    rebuttal_matrix=rebuttal_matrix,
                    feedback=feedback,
            )
            if debate.get("status") == "blocked":
                    blocked_record = self.context.artifacts.write_json(
                        self.stage_key,
                        "c2c/evidence_agent_blocked.json",
                        debate,
                        artifact_type="c2c_s1_evidence_agent_blocked",
                        summary="S1 Codex evidence agent could not produce valid direction JSON",
                        source_paths=["intake/c2c/evidence_brief.json", "intake/c2c/chunk_index.json"],
                    )
                    return {
                        "papers": metadata,
                        "artifacts": [blocked_record["path"]],
                        "status": "blocked",
                        "blocked_reason": debate.get("blocked_reason") or debate.get("reason") or "S1 Codex evidence agent blocked",
                    }
            ideas = debate["selected_ideas"]
            if debate.get("strategy") in {"codex_resume_evidence_agent", "codex_two_phase_evidence_direction"}:
                c2c_direction_candidate_scorecard = _s1_direction_candidate_scorecard(
                    {
                        "direction_decision": debate.get("direction_decision") if isinstance(debate.get("direction_decision"), dict) else {},
                        "selected_ideas": ideas[:1] if isinstance(ideas, list) else [],
                    },
                    evidence_bundle=debate.get("evidence_bundle") if isinstance(debate.get("evidence_bundle"), dict) else _evidence_bundle_from_selected_ideas(ideas),
                    raw_scorecard=debate.get("direction_candidate_scorecard") if isinstance(debate.get("direction_candidate_scorecard"), dict) else {},
                )
                debate["direction_candidate_scorecard"] = c2c_direction_candidate_scorecard
                if isinstance(debate.get("evidence_request_plan"), dict):
                    evidence_request_plan_record = self.context.artifacts.write_json(
                        self.stage_key,
                        "c2c/evidence_request_plan.json",
                        debate.get("evidence_request_plan", {}),
                        artifact_type="c2c_s1_evidence_request_plan",
                        summary="S1a requested evidence plan before deterministic C2C retrieval",
                        source_paths=["intake/c2c/evidence_brief.json", "intake/c2c/chunk_index.json"],
                    )
                evidence_requests_record = self.context.artifacts.write_json(
                    self.stage_key,
                    "c2c/evidence_requests.json",
                    debate.get("evidence_requests", []),
                    artifact_type="c2c_s1_evidence_requests",
                    summary="S1 requested evidence before direction choice",
                    source_paths=[evidence_request_plan_record["path"] if evidence_request_plan_record else "intake/c2c/evidence_brief.json"],
                )
                evidence_bundle_record = self.context.artifacts.write_json(
                    self.stage_key,
                    "c2c/evidence_bundle.json",
                    debate.get("evidence_bundle", {}),
                    artifact_type="c2c_s1_evidence_bundle",
                    summary="S1 fetched evidence chunks for direction choice",
                    source_paths=[evidence_requests_record["path"], chunk_index_record["path"], code_retrieval_record["path"]],
                )
                direction_record = self.context.artifacts.write_json(
                    self.stage_key,
                    "c2c/direction_selection_report.json",
                    debate.get("direction_decision", {}),
                    artifact_type="c2c_s1_direction_selection_report",
                    summary="S1 selected mechanism direction after evidence retrieval",
                    source_paths=[evidence_bundle_record["path"], negative_record["path"], baseline_record["path"]],
                )
                direction_candidate_scorecard_record = self.context.artifacts.write_json(
                    self.stage_key,
                    "c2c/direction_candidate_scorecard.json",
                    c2c_direction_candidate_scorecard,
                    artifact_type="c2c_s1_direction_candidate_scorecard",
                    summary="S1c scored candidate high-level directions and recorded why alternatives were not selected",
                    source_paths=[direction_record["path"], evidence_bundle_record["path"], evidence_request_plan_record["path"] if evidence_request_plan_record else evidence_requests_record["path"]],
                )
                evidence_session_record = self.context.artifacts.write_json(
                    self.stage_key,
                    "c2c/evidence_session.json",
                    debate.get("evidence_session", {}),
                    artifact_type="c2c_s1_evidence_session",
                    summary="S1 Codex evidence-on-demand session transcript",
                    source_paths=[evidence_requests_record["path"], evidence_bundle_record["path"], direction_record["path"]],
                )
                evidence_ref_report_record = self.context.artifacts.write_json(
                    self.stage_key,
                    "c2c/evidence_ref_report.json",
                    debate.get("evidence_ref_report", {}),
                    artifact_type="c2c_s1_evidence_ref_report",
                    summary="Resolved C2C S1 evidence refs against S0 chunk/code indexes",
                    source_paths=[evidence_bundle_record["path"], chunk_index_record["path"], code_file_manifest_record["path"], code_symbols_record["path"]],
                )
                if debate.get("novelty_audits"):
                    novelty_record = self.context.artifacts.write_json(
                        self.stage_key,
                        "c2c/novelty_audit.json",
                        debate.get("novelty_audits", []),
                        artifact_type="c2c_s1_novelty_audit",
                        summary="Independent C2C S1 novelty audit against shared and local memory",
                        source_paths=[direction_record["path"], evidence_session_record["path"], negative_record["path"]],
                    )
            debate_record = self.context.artifacts.write_json(
                self.stage_key,
                "direction_analysis.json",
                debate,
                artifact_type="c2c_direction_analysis",
                summary="C2C multi-agent idea debate",
                source_paths=[
                    repo_card_record["path"],
                    paper_cards_record["path"],
                    paper_chunks_record["path"],
                    rebuttal_record["path"],
                    rebuttal_chunks_record["path"],
                    code_cards_record["path"],
                    code_file_manifest_record["path"],
                    code_symbols_record["path"],
                    code_chunks_record["path"],
                    code_edges_record["path"],
                    code_repo_map_record["path"],
                    code_intake_report_record["path"],
                    implementation_surface_record["path"],
                    code_retrieval_record["path"],
                    cache_summary_record["path"],
                    semantic_merge_record["path"],
                    chunk_index_record["path"],
                    retrieval_plan_record["path"],
                    followup_bundle_record["path"],
                    negative_record["path"],
                    *(
                        [
                            evidence_requests_record["path"],
                            *( [evidence_request_plan_record["path"]] if evidence_request_plan_record else [] ),
                            evidence_bundle_record["path"],
                            direction_record["path"],
                            direction_candidate_scorecard_record["path"],
                            evidence_session_record["path"],
                            evidence_ref_report_record["path"],
                            *( [novelty_record["path"]] if novelty_record else [] ),
                        ]
                        if evidence_requests_record and evidence_bundle_record and direction_record and direction_candidate_scorecard_record and evidence_session_record and evidence_ref_report_record
                        else []
                    ),
                ],
            )
            debate_md_record = self.context.artifacts.write_text(
                self.stage_key,
                "direction_analysis.md",
                c2c_debate_markdown(debate),
                artifact_type="c2c_direction_analysis_summary",
                summary="Readable C2C idea debate summary",
                source_paths=[debate_record["path"]],
            )
            constraints_record = self.context.artifacts.write_json(
                self.stage_key,
                "negative_constraints.json",
                debate["negative_constraints"],
                artifact_type="c2c_negative_constraints",
                summary="C2C reviewer and failure constraints",
                source_paths=[debate_record["path"]],
            )
        root_evidence_bundle = (
            debate.get("evidence_bundle")
            if isinstance(debate.get("evidence_bundle"), dict)
            else _evidence_bundle_from_selected_ideas(ideas)
        )
        root_direction_payload = {
            "direction_decision": debate.get("direction_decision") if isinstance(debate.get("direction_decision"), dict) else {},
            "selected_ideas": ideas[:1] if isinstance(ideas, list) else [],
            "evidence_bundle": root_evidence_bundle,
            "negative_constraints": debate.get("negative_constraints") if isinstance(debate.get("negative_constraints"), dict) else {},
            "used_shared_memory_refs": debate.get("used_shared_memory_refs") if isinstance(debate.get("used_shared_memory_refs"), list) else [],
        }
        root_direction = build_direction_contract(
            root_direction_payload,
            mode="c2c",
            used_shared_memory_refs=root_direction_payload.get("used_shared_memory_refs") or None,
        )
        root_novelty_audit = normalize_novelty_audit(
            debate.get("novelty_audits") if isinstance(debate.get("novelty_audits"), list) else [],
            direction_id=str(root_direction.get("direction_id") or ""),
        )
        root_direction_scorecard = build_direction_scorecard(
            root_direction,
            evidence_bundle=root_evidence_bundle,
            novelty_audit=root_novelty_audit,
        )
        root_direction_candidate_scorecard = _s1_direction_candidate_scorecard(
            root_direction_payload,
            evidence_bundle=root_evidence_bundle,
            raw_scorecard=debate.get("direction_candidate_scorecard") if isinstance(debate.get("direction_candidate_scorecard"), dict) else {},
        )
        root_evidence_bundle_record = self.context.artifacts.write_json(
            self.stage_key,
            "evidence_bundle.json",
            root_evidence_bundle,
            artifact_type="s1_evidence_bundle",
            summary="Root S1 evidence bundle for the selected C2C direction",
            source_paths=[evidence_bundle_record["path"] if evidence_bundle_record else "literature/direction.json"],
        )
        root_direction_record = self.context.artifacts.write_json(
            self.stage_key,
            "direction.json",
            root_direction,
            artifact_type="s1_direction",
            summary="Root S1 selected C2C mechanism direction",
            source_paths=[root_evidence_bundle_record["path"]],
        )
        root_novelty_record = self.context.artifacts.write_json(
            self.stage_key,
            "novelty_audit.json",
            root_novelty_audit,
            artifact_type="s1_novelty_audit",
            summary="Root normalized C2C S1 novelty audit",
            source_paths=[root_direction_record["path"]],
        )
        root_scorecard_record = self.context.artifacts.write_json(
            self.stage_key,
            "direction_scorecard.json",
            root_direction_scorecard,
            artifact_type="s1_direction_scorecard",
            summary="Root C2C S1 direction readiness scorecard",
            source_paths=[root_direction_record["path"], root_novelty_record["path"], root_evidence_bundle_record["path"]],
        )
        if direction_candidate_scorecard_record is None:
            direction_candidate_scorecard_record = self.context.artifacts.write_json(
                self.stage_key,
                "c2c/direction_candidate_scorecard.json",
                root_direction_candidate_scorecard,
                artifact_type="c2c_s1_direction_candidate_scorecard",
                summary="S1c scored candidate high-level directions and recorded why alternatives were not selected",
                source_paths=[root_direction_record["path"], root_evidence_bundle_record["path"], root_scorecard_record["path"]],
            )
        root_evidence_ref_report = debate.get("evidence_ref_report") if isinstance(debate.get("evidence_ref_report"), dict) else resolve_s1_evidence_refs(self.context.project_root, root_direction_payload, mode="c2c")
        root_direction_bundle_ref_report = (
            debate.get("direction_bundle_ref_report")
            if isinstance(debate.get("direction_bundle_ref_report"), dict)
            else validate_direction_refs_subset_of_bundle(root_direction_payload, root_evidence_bundle)
        )
        root_deterministic_trace = debate.get("evidence_retrieval_trace") if isinstance(debate.get("evidence_retrieval_trace"), dict) and debate.get("evidence_retrieval_trace", {}).get("deterministic") is True else None
        root_quality_artifacts = _build_c2c_s1_quality_artifacts(
            project_root=self.context.project_root,
            payload=root_direction_payload,
            direction=root_direction,
            evidence_bundle=root_evidence_bundle,
            evidence_ref_report=root_evidence_ref_report,
            novelty_audit=root_novelty_audit,
            shared_memory_checked=bool(debate.get("strategy") in {"codex_resume_evidence_agent", "codex_two_phase_evidence_direction"} or root_direction_payload.get("used_shared_memory_refs")),
            direction_bundle_ref_report=root_direction_bundle_ref_report,
            deterministic_trace=root_deterministic_trace,
        )
        direction_fingerprint_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/direction_fingerprint.json",
            root_quality_artifacts["direction_fingerprint"],
            artifact_type="c2c_s1_direction_fingerprint",
            summary="Deterministic identity and history similarity for the selected C2C S1 direction",
            source_paths=[root_direction_record["path"], negative_record["path"]],
        )
        evidence_quality_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/evidence_quality_score.json",
            root_quality_artifacts["evidence_quality_score"],
            artifact_type="c2c_s1_evidence_quality_score",
            summary="Deterministic C2C S1 evidence quality gate score",
            source_paths=[
                root_direction_record["path"],
                root_evidence_bundle_record["path"],
                root_novelty_record["path"],
                direction_fingerprint_record["path"],
                evidence_ref_report_record["path"] if evidence_ref_report_record else root_scorecard_record["path"],
            ],
        )
        evidence_retrieval_trace_record = self.context.artifacts.write_json(
            self.stage_key,
            "c2c/evidence_retrieval_trace.json",
            root_quality_artifacts["evidence_retrieval_trace"],
            artifact_type="c2c_s1_evidence_retrieval_trace",
            summary="Resolved and unresolved refs used by the C2C S1 evidence quality gate",
            source_paths=[
                root_direction_record["path"],
                root_evidence_bundle_record["path"],
                evidence_quality_record["path"],
                direction_fingerprint_record["path"],
            ],
        )
        survey_record = self.context.artifacts.write_text(
            self.stage_key,
            "survey.md",
            self._build_c2c_survey(topic, baseline, reference_result["cards"], historical_results, rebuttal_matrix, negative_memory),
            artifact_type="survey",
            summary="C2C-focused literature and evidence survey",
            source_paths=["intake/c2c/evidence_brief.json", metadata_record["path"], baseline_record["path"], rebuttal_record["path"], negative_record["path"]],
        )
        ideas_record = self.context.artifacts.write_json(
            self.stage_key,
            "candidate_directions.json",
            ideas,
            artifact_type="ideas",
            summary="C2C candidate research ideas",
            source_paths=[
                repo_card_record["path"],
                baseline_record["path"],
                paper_cards_record["path"],
                paper_chunks_record["path"],
                bibliography_record["path"],
                rebuttal_record["path"],
                rebuttal_chunks_record["path"],
                code_cards_record["path"],
                code_file_manifest_record["path"],
                code_symbols_record["path"],
                code_chunks_record["path"],
                code_edges_record["path"],
                code_repo_map_record["path"],
                code_intake_report_record["path"],
                implementation_surface_record["path"],
                code_retrieval_record["path"],
                cache_summary_record["path"],
                semantic_merge_record["path"],
                chunk_index_record["path"],
                retrieval_plan_record["path"],
                followup_bundle_record["path"],
                negative_record["path"],
                root_evidence_bundle_record["path"],
                root_direction_record["path"],
                root_novelty_record["path"],
                root_scorecard_record["path"],
                direction_candidate_scorecard_record["path"],
                direction_fingerprint_record["path"],
                evidence_quality_record["path"],
                evidence_retrieval_trace_record["path"],
                *(
                    [
                        evidence_requests_record["path"],
                        *( [evidence_request_plan_record["path"]] if evidence_request_plan_record else [] ),
                        evidence_bundle_record["path"],
                        direction_record["path"],
                        direction_candidate_scorecard_record["path"],
                        evidence_session_record["path"],
                        evidence_ref_report_record["path"],
                        debate_record["path"],
                        debate_md_record["path"],
                        constraints_record["path"],
                    ]
                    if debate_record and debate_md_record and constraints_record and evidence_requests_record and evidence_bundle_record and direction_record and direction_candidate_scorecard_record and evidence_session_record and evidence_ref_report_record
                    else []
                ),
            ],
        )
        feasibility_record = self.context.artifacts.write_text(
            self.stage_key,
            "feasibility_check.md",
            self._build_c2c_feasibility(ideas, baseline),
            artifact_type="feasibility",
            summary="C2C feasibility note",
            source_paths=[ideas_record["path"], baseline_record["path"]],
        )
        return {
            "papers": metadata,
            "artifacts": [
                metadata_record["path"],
                repo_record["path"],
                repo_card_record["path"],
                history_record["path"],
                result_ledger_record["path"],
                baseline_record["path"],
                paper_cards_record["path"],
                paper_chunks_record["path"],
                bibliography_record["path"],
                rebuttal_record["path"],
                rebuttal_chunks_record["path"],
                code_cards_record["path"],
                code_file_manifest_record["path"],
                code_symbols_record["path"],
                code_chunks_record["path"],
                code_edges_record["path"],
                code_repo_map_record["path"],
                code_repo_map_md_record["path"],
                code_intake_report_record["path"],
                implementation_surface_record["path"],
                code_retrieval_record["path"],
                cache_summary_record["path"],
                semantic_merge_record["path"],
                chunk_index_record["path"],
                chunk_index_jsonl_record["path"],
                retrieval_plan_record["path"],
                followup_bundle_record["path"],
                negative_record["path"],
                root_evidence_bundle_record["path"],
                root_direction_record["path"],
                root_novelty_record["path"],
                root_scorecard_record["path"],
                direction_candidate_scorecard_record["path"],
                direction_fingerprint_record["path"],
                evidence_quality_record["path"],
                evidence_retrieval_trace_record["path"],
                *(
                    [
                        evidence_requests_record["path"],
                        *( [evidence_request_plan_record["path"]] if evidence_request_plan_record else [] ),
                        evidence_bundle_record["path"],
                        direction_record["path"],
                        direction_candidate_scorecard_record["path"],
                        evidence_session_record["path"],
                        evidence_ref_report_record["path"],
                        debate_record["path"],
                        debate_md_record["path"],
                        constraints_record["path"],
                    ]
                    if debate_record and debate_md_record and constraints_record and evidence_requests_record and evidence_bundle_record and direction_record and direction_candidate_scorecard_record and evidence_session_record and evidence_ref_report_record
                    else []
                ),
                survey_record["path"],
                ideas_record["path"],
                feasibility_record["path"],
            ],
        }

    def _search_consensus_imports(self) -> list[dict[str, Any]]:
        importer = ConsensusImporter(self.context.project_root)
        imported = importer.list_imports()
        papers = []
        for item in imported:
            for query in item.get("queries", [])[:5]:
                papers.extend(self.provider.search(query))
            for title in item.get("paper_title_candidates", [])[:5]:
                papers.extend(self.provider.search(title))
            for arxiv_id in item.get("arxiv_ids", [])[:5]:
                papers.extend(self.provider.search(arxiv_id))
        return papers

    def _load_feedback(self) -> list[dict[str, Any]]:
        bundle = load_c2c_feedback_bundle(self.context.project_root, view="method")
        return [bundle["summary_entry"], *bundle["entries"], *bundle["iteration_traces"]]

    def _load_c2c_static_bundle(self) -> dict[str, Any]:
        path = self.context.project_root / "intake" / "c2c" / "static_bundle.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _run_generic_evidence_direction(self, *, topic: str, papers: list[dict[str, Any]], survey: str, theme_map_path: str | None) -> dict[str, Any]:
        cfg = _s1_codex_agent_config(self.context.config, mode="generic")
        result = _run_s1_codex_evidence_agent(
            project_root=self.context.project_root,
            config=self.context.config,
            prompt=_generic_s1_codex_evidence_prompt(
                topic=topic,
                papers=papers,
                survey=survey,
                theme_map_path=theme_map_path,
                resources=discover_local_mm_resources(self.context.config),
                config=self.context.config,
                excluded_direction_semantic_hashes=ResearchEventLedger(self.context.project_root).state().get("excluded_direction_semantic_hashes") or [],
            ),
            max_repairs=int(cfg.get("max_json_repairs") or 2),
            timeout_seconds=int(cfg.get("timeout_seconds") or (self.context.config.get("llm", {}) or {}).get("timeout_seconds") or 1800),
            mode="generic",
        )
        if result.get("status") != "ok":
            return {
                "status": "blocked",
                "strategy": "codex_resume_evidence_agent",
                "blocked_reason": result.get("reason") or "S1 Codex evidence agent did not return valid JSON.",
                "evidence_session": result,
                "ideas": [],
        }
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        shared_memory = shared_method_memory_for_prompt(
            self.context.config,
            limit=12,
            query_context=shared_method_memory_query_context(
                self.context.config,
                project_root=self.context.project_root,
                topic=topic,
            ),
        )
        used_shared_memory_refs = collect_used_shared_memory_refs(payload, shared_memory)
        ideas = _generic_s1_codex_ideas(payload, used_shared_memory_refs=used_shared_memory_refs)
        _attach_s1_novelty_audit_to_ideas(ideas, result.get("novelty_audits", []))
        direction_decision = dict(payload.get("direction_decision") if isinstance(payload.get("direction_decision"), dict) else {})
        direction_decision["used_shared_memory_refs"] = used_shared_memory_refs
        direction_payload = dict(payload)
        direction_payload["direction_decision"] = direction_decision
        direction_payload["selected_ideas"] = ideas
        direction = build_direction_contract(direction_payload, mode="generic", used_shared_memory_refs=used_shared_memory_refs)
        novelty_audit = normalize_novelty_audit(result.get("novelty_audits", []), direction_id=str(direction.get("direction_id") or ""))
        evidence_bundle = payload.get("evidence_bundle") if isinstance(payload.get("evidence_bundle"), dict) else {"items": []}
        evidence_ref_report = result.get("evidence_ref_report") or resolve_s1_evidence_refs(self.context.project_root, payload, mode="generic")
        return {
            "status": "ok",
            "strategy": "codex_resume_evidence_agent",
            "ideas": ideas,
            "evidence_requests": payload.get("evidence_requests") if isinstance(payload.get("evidence_requests"), list) else [],
            "evidence_bundle": evidence_bundle,
            "direction": direction,
            "direction_scorecard": build_direction_scorecard(direction, evidence_bundle=evidence_bundle, novelty_audit=novelty_audit),
            "novelty_audit": novelty_audit,
            "direction_decision": direction_decision,
            "used_shared_memory_refs": used_shared_memory_refs,
            "evidence_ref_report": evidence_ref_report,
            "evidence_session": {
                "schema_version": "s1_codex_evidence_session_v1",
                "status": "ok",
                "session_key": result.get("session_key"),
                "session_id": result.get("session_id"),
                "used_existing_session": result.get("used_existing_session"),
                "attempts": result.get("attempts", []),
                "repair_count": result.get("repair_count", 0),
                "novelty_audits": result.get("novelty_audits", []),
                "session_reset": result.get("session_reset"),
                "session_reset_reason": result.get("session_reset_reason"),
                "used_shared_memory_refs": used_shared_memory_refs,
                "source": "codex_cli",
            },
        }

    def _run_c2c_evidence_on_demand_direction(
        self,
        *,
        topic: str,
        evidence_brief: dict[str, Any],
        chunk_index: dict[str, Any],
        paper_chunks: list[dict[str, Any]],
        rebuttal_chunks: list[dict[str, Any]],
        code_chunks: list[dict[str, Any]],
        code_edges: list[dict[str, Any]],
        code_intake_report: dict[str, Any],
        implementation_surface_map: dict[str, Any],
        code_retrieval_index: dict[str, Any],
        baseline: dict[str, Any],
        negative_memory: dict[str, Any],
        rebuttal_matrix: dict[str, Any],
        feedback: list[dict[str, Any]],
    ) -> dict[str, Any]:
        two_phase_cfg = _c2c_s1_two_phase_config(self.context.config)
        shared_memory = shared_method_memory_for_prompt(
            self.context.config,
            limit=12,
            query_context=shared_method_memory_query_context(
                self.context.config,
                project_root=self.context.project_root,
                topic=topic,
                feedback=feedback,
                negative_memory=negative_memory,
            ),
        )
        retrieval_feedback: list[str] = []
        request_result: dict[str, Any] = {}
        request_plans: list[dict[str, Any]] = []
        evidence_bundle: dict[str, Any] = {}
        deterministic_trace: dict[str, Any] = {}
        max_request_rounds = int(two_phase_cfg.get("max_request_revision_rounds") or 1)
        for request_round in range(max_request_rounds + 1):
            request_result = _run_s1_c2c_evidence_request_agent(
                project_root=self.context.project_root,
                config=self.context.config,
                topic=topic,
                evidence_brief=evidence_brief,
                chunk_index=chunk_index,
                code_intake_report=code_intake_report,
                implementation_surface_map=implementation_surface_map,
                code_retrieval_index=code_retrieval_index,
                baseline=baseline,
                negative_memory=negative_memory,
                rebuttal_matrix=rebuttal_matrix,
                feedback=feedback,
                shared_memory=shared_memory,
                retrieval_feedback=retrieval_feedback,
            )
            if request_result.get("status") != "ok":
                return _blocked_c2c_two_phase_result(
                    reason=request_result.get("reason") or "S1a evidence request agent did not return valid request plan.",
                    phases=[request_result],
                )
            request_plan = request_result.get("evidence_request_plan") if isinstance(request_result.get("evidence_request_plan"), dict) else default_c2c_evidence_request_plan(topic=topic)
            request_plans = [request_plan]
            evidence_bundle, deterministic_trace = retrieve_s1_c2c_requested_evidence(
                request_plan,
                chunk_index=chunk_index,
                paper_chunks=paper_chunks,
                rebuttal_chunks=rebuttal_chunks,
                code_chunks=code_chunks,
                code_edges=code_edges,
                code_retrieval_index=code_retrieval_index,
                implementation_surface_map=implementation_surface_map,
                negative_memory=negative_memory,
                feedback=feedback,
                shared_memory=shared_memory,
                config=self.context.config,
            )
            retrieval_feedback = _s1b_retrieval_blockers(deterministic_trace)
            if not retrieval_feedback:
                break
            if request_round >= max_request_rounds:
                bootstrap_options = bootstrap_profile_options(self.context.config)
                waivable_retrieval_rules = {"retrieval_coverage.paper<2", "retrieval_coverage.code<2"}
                if (
                    bootstrap_profile_enabled(self.context.config)
                    and bootstrap_options.get("allow_retrieval_warnings", True)
                    and set(retrieval_feedback).issubset(waivable_retrieval_rules)
                ):
                    deterministic_trace = dict(deterministic_trace)
                    deterministic_trace["bootstrap_warnings"] = list(retrieval_feedback)
                    deterministic_trace["bootstrap_degraded_retrieval"] = True
                    break
                return _blocked_c2c_two_phase_result(
                    reason="S1b deterministic retriever could not satisfy required C2C evidence coverage.",
                    phases=[
                        request_result,
                        {
                            "phase": "deterministic_retriever",
                            "status": "blocked",
                            "retrieval_trace": deterministic_trace,
                            "validation_errors": retrieval_feedback,
                        },
                    ],
                )

        direction_phases: list[dict[str, Any]] = []
        max_direction_followups = int(two_phase_cfg.get("max_direction_followup_rounds") or 1)
        direction_result: dict[str, Any] = {}
        for direction_round in range(max_direction_followups + 1):
            direction_result = _run_s1_c2c_direction_agent(
                project_root=self.context.project_root,
                config=self.context.config,
                topic=topic,
                baseline=baseline,
                negative_memory=negative_memory,
                feedback=feedback,
                evidence_bundle=evidence_bundle,
                retrieval_trace=deterministic_trace,
                shared_memory=shared_memory,
            )
            direction_phases.append(direction_result)
            if direction_result.get("status") == "ok":
                break
            if direction_result.get("status") != "needs_more_evidence" or direction_round >= max_direction_followups:
                break
            followup_plan = direction_result.get("followup_evidence_request_plan") if isinstance(direction_result.get("followup_evidence_request_plan"), dict) else {}
            followup_bundle, followup_trace = retrieve_s1_c2c_requested_evidence(
                followup_plan,
                chunk_index=chunk_index,
                paper_chunks=paper_chunks,
                rebuttal_chunks=rebuttal_chunks,
                code_chunks=code_chunks,
                code_edges=code_edges,
                code_retrieval_index=code_retrieval_index,
                implementation_surface_map=implementation_surface_map,
                negative_memory=negative_memory,
                feedback=feedback,
                shared_memory=shared_memory,
                config=self.context.config,
            )
            request_plans.append(followup_plan)
            evidence_bundle = _merge_s1_c2c_evidence_bundles(evidence_bundle, followup_bundle)
            deterministic_trace = _merge_s1_c2c_retrieval_traces(deterministic_trace, followup_trace, followup_round=direction_round + 1)
        if direction_result.get("status") != "ok":
            return _blocked_c2c_two_phase_result(
                reason=direction_result.get("reason") or "S1c direction agent did not return valid direction JSON.",
                phases=[request_result, {"phase": "deterministic_retriever", "status": "ok", "retrieval_trace": deterministic_trace}, *direction_phases],
            )
        payload = direction_result.get("payload") if isinstance(direction_result.get("payload"), dict) else {}
        evidence_requests: list[dict[str, Any]] = []
        for plan in request_plans:
            evidence_requests.extend(item for item in (plan.get("evidence_requests") or []) if isinstance(item, dict))
        payload = dict(payload)
        payload["evidence_requests"] = evidence_requests
        payload["evidence_bundle"] = evidence_bundle
        used_shared_memory_refs = collect_used_shared_memory_refs(payload, shared_memory)
        payload["used_shared_memory_refs"] = used_shared_memory_refs
        payload["s1_agent_source"] = "codex_two_phase_direction_agent"
        direction_decision = dict(payload.get("direction_decision") if isinstance(payload.get("direction_decision"), dict) else {})
        direction_decision["used_shared_memory_refs"] = used_shared_memory_refs
        selected_ideas = _s1_codex_direction_cards(payload, used_shared_memory_refs=used_shared_memory_refs)
        _attach_s1_novelty_audit_to_ideas(selected_ideas, direction_result.get("novelty_audits", []))
        negative_constraints = payload.get("negative_constraints") if isinstance(payload.get("negative_constraints"), dict) else {}
        negative_constraints["used_shared_memory_refs"] = used_shared_memory_refs
        payload["direction_decision"] = direction_decision
        payload["selected_ideas"] = selected_ideas
        payload["negative_constraints"] = negative_constraints
        evidence_ref_report = resolve_s1_evidence_refs(self.context.project_root, payload, mode="c2c")
        direction_bundle_ref_report = validate_direction_refs_subset_of_bundle(payload, evidence_bundle)
        decision_chain = payload.get("decision_chain") if isinstance(payload.get("decision_chain"), dict) else direction_decision.get("decision_chain")
        if not isinstance(decision_chain, dict):
            decision_chain = _s1_codex_decision_chain(payload, selected_ideas)
        direction_payload = dict(payload)
        direction_payload["direction_decision"] = direction_decision
        direction_payload["selected_ideas"] = selected_ideas
        direction_payload["negative_constraints"] = negative_constraints
        direction = build_direction_contract(direction_payload, mode="c2c", used_shared_memory_refs=used_shared_memory_refs)
        direction_candidate_scorecard = _s1_direction_candidate_scorecard(direction_payload, evidence_bundle=evidence_bundle)
        novelty_audit = normalize_novelty_audit(direction_result.get("novelty_audits", []), direction_id=str(direction.get("direction_id") or ""))
        quality_artifacts = _build_c2c_s1_quality_artifacts(
            project_root=self.context.project_root,
            payload=direction_payload,
            direction=direction,
            evidence_bundle=evidence_bundle,
            evidence_ref_report=evidence_ref_report,
            novelty_audit=novelty_audit,
            shared_memory_checked=True,
            direction_bundle_ref_report=direction_bundle_ref_report,
            deterministic_trace=deterministic_trace,
        )
        evidence_session = _c2c_two_phase_session(
            request_result=request_result,
            deterministic_trace=deterministic_trace,
            direction_result=direction_result,
            quality_artifacts=quality_artifacts,
            used_shared_memory_refs=used_shared_memory_refs,
        )
        return {
            "status": "ok",
            "strategy": "codex_two_phase_evidence_direction",
            "roles": ["evidence_request_agent", "deterministic_retriever", "direction_agent"],
            "rounds": [],
            "meta_judge": {
                "role": "direction_agent",
                "status": "ok",
                "session_id": direction_result.get("session_id"),
                "decision_chain": decision_chain,
                "decision_rationale": direction_decision.get("rationale") or direction_decision.get("why_this_direction") or decision_chain.get("conclusion"),
            },
            "decision_chain": decision_chain,
            "evidence_request_plan": request_result.get("evidence_request_plan"),
            "direction_decision": direction_decision,
            "direction": direction,
            "direction_scorecard": build_direction_scorecard(direction, evidence_bundle=evidence_bundle, novelty_audit=novelty_audit),
            "direction_candidate_scorecard": direction_candidate_scorecard,
            "novelty_audits": direction_result.get("novelty_audits", []),
            "novelty_audit": novelty_audit,
            "selected_ideas": selected_ideas,
            "negative_constraints": negative_constraints,
            "used_shared_memory_refs": used_shared_memory_refs,
            "evidence_requests": evidence_requests,
            "evidence_bundle": evidence_bundle,
            "evidence_ref_report": evidence_ref_report,
            "direction_bundle_ref_report": direction_bundle_ref_report,
            "evidence_quality_score": quality_artifacts.get("evidence_quality_score"),
            "evidence_retrieval_trace": quality_artifacts.get("evidence_retrieval_trace"),
            "direction_fingerprint": quality_artifacts.get("direction_fingerprint"),
            "evidence_session": evidence_session,
            "quality_flags": [],
            "run_log": {"progress_path": "literature/c2c/evidence_session.json", "events": evidence_session.get("attempts", [])},
        }

    def _run_related_work_audit(self) -> dict[str, Any]:
        paper_path = self.context.project_root / "paper" / "sections" / "related_work.tex"
        metadata_path = self.context.project_root / "literature" / "papers" / "metadata.json"
        if not paper_path.exists() or not metadata_path.exists():
            audit = {
                "missing_critical": [],
                "missing_recent": [],
                "novelty_conflicts": [],
                "grouping_suggestions": ["Related work or metadata missing."],
            }
        else:
            related = paper_path.read_text(encoding="utf-8")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cited_keys = set()
            for sentence in split_sentences(related):
                if "\\cite{" in sentence:
                    cited_keys.update(part.strip() for chunk in sentence.split("\\cite{")[1:] for part in chunk.split("}", 1)[0].split(","))
            missing = []
            for paper in metadata[:5]:
                key = self._citation_key(paper)
                if key not in cited_keys:
                    missing.append({"citation_key": key, "title": paper.get("title")})
            audit = {
                "missing_critical": missing[:3],
                "missing_recent": missing[3:5],
                "novelty_conflicts": [],
                "grouping_suggestions": ["Preserve grouping by paradigm and discuss nearest competing method explicitly."],
            }
        record = self.context.artifacts.write_text(
            self.stage_key,
            "related_work_audit.md",
            compact_markdown(self._format_audit(audit)),
            artifact_type="audit",
            summary="Related work reverse audit",
        )
        return {"audit": audit, "artifacts": [record["path"]]}

    @staticmethod
    def _citation_key(paper: dict[str, Any]) -> str:
        author = sanitize_filename((paper.get("authors") or ["anon"])[0].split()[-1].lower())
        year = str(paper.get("year") or "xxxx")
        return f"{author}{year}"

    def _build_survey(self, topic: str, papers: list[dict[str, Any]]) -> str:
        lines = [
            f"# Survey: {topic}",
            "",
            "## Coverage",
            f"- Papers collected: {len(papers)}",
            "- Sources: Semantic Scholar and arXiv",
            "",
            "## Key Papers",
        ]
        for paper in papers[:8]:
            lines.append(
                f"- **{paper.get('title', 'Untitled')}** ({paper.get('year', 'n/a')}): {paper.get('abstract', '').split('. ')[0][:220]}"
            )
        lines.extend(
            [
                "",
                "## Emerging Patterns",
                "- Strong methods combine reliable baselines with one clear intervention.",
                "- Related work claims are easiest to defend when the benchmark and evaluation protocol stay fixed.",
                "- Compute-aware ideas have a better path to a complete paper than purely novelty-driven ideas.",
            ]
        )
        return compact_markdown("\n".join(lines))

    def _build_ideas(self, topic: str, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.context.llm.use_real_api:
            default = self._default_ideas(topic, papers)
            prompt = {
                "topic": topic,
                "papers": [
                    {
                        "title": paper.get("title"),
                        "abstract_snippet": (paper.get("abstract") or "")[:220],
                        "year": paper.get("year"),
                    }
                    for paper in papers[:4]
                ],
                "required_fields": [
                    "id",
                    "title",
                    "description",
                    "motivation",
                    "novelty_score",
                    "feasibility_score",
                    "expected_contribution",
                    "key_baselines",
                    "required_compute",
                    "key_references",
                    "selected",
                ],
            }
            try:
                ideas = self.context.llm.generate_json(
                    instructions="You are a research strategist. Return three executable ML research ideas as JSON.",
                    prompt=str(prompt),
                    default=default,
                    agent_name="literature-agent",
                )
                if isinstance(ideas, list) and len(ideas) >= 3:
                    return ideas
            except Exception:
                return default
        return self._default_ideas(topic, papers)

    def _default_ideas(self, topic: str, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        top_titles = [paper.get("title", "Prior work") for paper in papers[:3]]
        references = [paper.get("paper_id") for paper in papers[:5]]
        ideas = []
        for idx in range(3):
            ideas.append(
                {
                    "id": f"idea_{idx + 1}",
                    "title": f"{topic.title()} idea {idx + 1}",
                    "description": f"Focus on a single intervention derived from {top_titles[idx % max(1, len(top_titles))] if top_titles else 'recent literature'}.",
                    "motivation": "A paper-quality result is more likely when novelty is tied to a clear benchmark and a bounded change.",
                    "novelty_score": 7 - idx,
                    "feasibility_score": 8 - idx,
                    "expected_contribution": "Improved effectiveness or efficiency under the same evaluation protocol.",
                    "key_baselines": top_titles[:2] or ["Strong baseline", "Classic baseline"],
                    "required_compute": "1-4 GPU days",
                    "key_references": references,
                    "selected": idx == 0,
                }
            )
        return ideas

    @staticmethod
    def _build_feasibility(ideas: list[dict[str, Any]]) -> str:
        lines = ["# Feasibility Check", ""]
        for idea in ideas[:2]:
            lines.extend(
                [
                    f"## {idea['title']}",
                    f"- Novelty: {idea['novelty_score']}/10",
                    f"- Feasibility: {idea['feasibility_score']}/10",
                    "- Recommendation: start with the selected idea and validate one variable at a time.",
                    "",
                ]
            )
        return compact_markdown("\n".join(lines))

    @staticmethod
    def _build_c2c_survey(
        topic: str,
        baseline: dict[str, Any],
        reference_cards: list[dict[str, Any]],
        historical_results: dict[str, Any],
        rebuttal_matrix: dict[str, Any] | None = None,
        negative_memory: dict[str, Any] | None = None,
    ) -> str:
        lines = [
            f"# C2C Survey: {topic}",
            "",
            "## Baseline Target",
            f"- Method: {baseline.get('name')}",
            f"- Three-dataset mean: {baseline.get('mean')}",
            f"- Dataset scores: {baseline.get('datasets')}",
            "",
            "## Configured References",
        ]
        for card in reference_cards:
            text = (card.get("text") or "").replace("\n", " ")[:260]
            lines.append(f"- **{card.get('title')}** ({card.get('kind')}): {text}")
        lines.extend(["", "## Imported Historical Evidence"])
        counts = historical_results.get("counts", {})
        lines.append(f"- Small-loop rows: {counts.get('small_loop_rows', 0)}")
        lines.append(f"- Summary JSON files: {counts.get('summary_jsons', 0)}")
        if rebuttal_matrix:
            lines.extend(["", "## Reviewer Concern Signals"])
            top_concerns = rebuttal_matrix.get("top_concerns") or []
            lines.append(f"- Top concerns: {top_concerns or ['none detected']}")
            for item in rebuttal_matrix.get("matrix", [])[:4]:
                if item.get("hit_count", 0) > 0:
                    lines.append(f"- {item.get('concern_id')}: priority={item.get('priority')}, hits={item.get('hit_count')}")
        if negative_memory:
            lines.extend(["", "## Negative Result Memory"])
            lines.append(f"- Below-baseline rows: {negative_memory.get('failed_result_count', 0)}")
            for rule in (negative_memory.get("blocked_idea_patterns") or [])[:5]:
                lines.append(f"- Avoid: {rule}")
        lines.extend(
            [
                "",
                "## Working Thesis",
                "- Keep receiver, sharer, datasets, and small2048 protocol fixed.",
                "- Search for bounded changes in cross-tokenizer span alignment, confidence gating, and KV aggregation.",
                "- Treat validation loss as diagnostic only; benchmark mean is the primary signal.",
            ]
        )
        return compact_markdown("\n".join(lines))

    @staticmethod
    def _build_c2c_feasibility(ideas: list[dict[str, Any]], baseline: dict[str, Any]) -> str:
        lines = [
            "# C2C Feasibility Check",
            "",
            f"- Baseline to beat: {baseline.get('name')} mean={baseline.get('mean')}",
            "- Allowed edit scope: core alignment modules plus generated recipe/local configs.",
            "- Verification: py_compile, span-overlap unit test, small2048 train, three dataset eval.",
            "",
        ]
        for idea in ideas:
            lines.extend(
                [
                    f"## {idea['title']}",
                    f"- Novelty: {idea.get('novelty_score')}/10",
                    f"- Feasibility: {idea.get('feasibility_score')}/10",
                    f"- Hypothesis: {idea.get('hypothesis', idea.get('description', ''))}",
                    f"- Risk: {idea.get('risk', 'unknown')}",
                    "",
                ]
            )
        return compact_markdown("\n".join(lines))

    @staticmethod
    def _format_audit(audit: dict[str, Any]) -> str:
        lines = ["# Related Work Audit", ""]
        for key in ["missing_critical", "missing_recent", "novelty_conflicts", "grouping_suggestions"]:
            lines.append(f"## {key}")
            values = audit.get(key) or []
            if not values:
                lines.append("- None")
            elif isinstance(values[0], dict):
                for item in values:
                    lines.append(f"- {item.get('citation_key', '')}: {item.get('title', '')}".strip(": "))
            else:
                for item in values:
                    lines.append(f"- {item}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _mock_papers(topic: str) -> list[dict[str, Any]]:
        papers = []
        for idx in range(5):
            papers.append(
                {
                    "paper_id": f"mock_{idx + 1}",
                    "title": f"{topic.title()} benchmark paper {idx + 1}",
                    "authors": [f"Author {idx + 1}"],
                    "year": 2025 - idx,
                    "abstract": "This mock paper exists to exercise the pipeline when external retrieval is unavailable.",
                    "citation_count": 10 - idx,
                    "source": "mock",
                    "venue": "MockConf",
                    "pdf_url": None,
                    "url": None,
                }
            )
        return papers

    def _ensure_placeholder_pdfs(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated = []
        minimal_pdf = b"%PDF-1.4\n1 0 obj <<>> endobj\ntrailer <<>>\n%%EOF\n"
        for paper in papers:
            current = dict(paper)
            if not current.get("local_pdf_path"):
                filename = self.provider.pdf_filename(current)
                record = self.context.artifacts.write_reference_pdf(filename, minimal_pdf, metadata=current)
                current["download_status"] = "placeholder"
                current["local_pdf_path"] = record["local_pdf_path"]
            updated.append(current)
        return updated


def _jsonl(items: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


def _merge_s0_semantic_enrichment_for_s1(
    project_root: Path,
    *,
    paper_chunks: list[dict[str, Any]],
    rebuttal_chunks: list[dict[str, Any]],
    code_chunks: list[dict[str, Any]],
    chunk_index: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records_by_id, record_source_paths = _load_s0_semantic_records(project_root, config=config or {})
    _alias_s0_semantic_records_by_text(
        records_by_id,
        [*paper_chunks, *rebuttal_chunks, *code_chunks],
        config=config or {},
    )
    paper_chunks = _apply_s1_semantic_records_to_chunks(paper_chunks, records_by_id, source_type="paper")
    rebuttal_chunks = _apply_s1_semantic_records_to_chunks(rebuttal_chunks, records_by_id, source_type="rebuttal")
    code_chunks = _apply_s1_semantic_records_to_chunks(code_chunks, records_by_id, source_type="code")
    chunk_by_id = {
        str(chunk.get("chunk_id") or ""): chunk
        for chunk in [*paper_chunks, *rebuttal_chunks, *code_chunks]
        if isinstance(chunk, dict) and str(chunk.get("chunk_id") or "")
    }
    merged_chunk_index = _merge_s1_semantic_fields_into_chunk_index(chunk_index, chunk_by_id)
    report = _s1_semantic_merge_report(
        records_by_id=records_by_id,
        source_paths=record_source_paths,
        paper_chunks=paper_chunks,
        rebuttal_chunks=rebuttal_chunks,
        code_chunks=code_chunks,
        chunk_index=merged_chunk_index,
    )
    return {
        "paper_chunks": paper_chunks,
        "rebuttal_chunks": rebuttal_chunks,
        "code_chunks": code_chunks,
        "chunk_index": merged_chunk_index,
        "report": report,
    }


def _load_s0_semantic_records(project_root: Path, *, config: dict[str, Any] | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    source_paths: list[str] = []
    for record, source_path in _iter_s0_semantic_artifact_records(project_root):
        _select_best_s0_semantic_record(records, record)
        source_paths.append(source_path)
    for record, source_path in _iter_s0_semantic_cache_records(project_root, config=config or {}):
        _select_best_s0_semantic_record(records, record)
        source_paths.append(source_path)
    return records, _dedupe_texts(source_paths, max_items=2000)


def _iter_s0_semantic_artifact_records(project_root: Path):
    base = project_root / "intake" / "c2c"
    json_path = base / "semantic_enrichment_sample.json"
    payload = read_json(json_path, default={})
    if isinstance(payload, dict):
        for record in payload.get("records") or []:
            if isinstance(record, dict):
                yield record, "intake/c2c/semantic_enrichment_sample.json"
    elif isinstance(payload, list):
        for record in payload:
            if isinstance(record, dict):
                yield record, "intake/c2c/semantic_enrichment_sample.json"
    jsonl_path = base / "semantic_enrichment_sample.jsonl"
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record, "intake/c2c/semantic_enrichment_sample.jsonl"


def _iter_s0_semantic_cache_records(project_root: Path, *, config: dict[str, Any]):
    model = _s0_semantic_model_from_config(config)
    cache_roots = [
        project_root / ".cache" / "auto_research" / "s0_semantic_enrichment" / _s1_cache_safe(model),
        project_root / ".cache" / "auto_research" / "s0_semantic_enrichment" / _s1_cache_safe(DEFAULT_DEEPSEEK_MODEL),
        shared_cache_root(project_root, config) / "s0_semantic_enrichment" / _s1_cache_safe(model),
        shared_cache_root(project_root, config) / "s0_semantic_enrichment" / _s1_cache_safe(DEFAULT_DEEPSEEK_MODEL),
        *_s1_legacy_semantic_cache_roots(project_root, model),
    ]
    for cache_root in cache_roots:
        if not cache_root.exists():
            continue
        for path in sorted(cache_root.glob("*.json")):
            payload = read_json(path, default={})
            if isinstance(payload, dict):
                try:
                    source_path = str(path.relative_to(project_root))
                except ValueError:
                    source_path = str(path)
                yield payload, source_path


def _s0_semantic_model_from_config(config: dict[str, Any]) -> str:
    intake = config.get("intake") if isinstance(config, dict) else {}
    semantic = (intake or {}).get("semantic_enrichment") if isinstance(intake, dict) else {}
    if isinstance(semantic, dict) and semantic.get("model"):
        return str(semantic.get("model"))
    return DEFAULT_DEEPSEEK_MODEL


def _s1_cache_safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "model"


def _s1_legacy_semantic_cache_roots(project_root: Path, model: str) -> list[Path]:
    model_dirs = {_s1_cache_safe(model), _s1_cache_safe(DEFAULT_DEEPSEEK_MODEL)}
    roots: list[Path] = []
    for candidate in sorted(project_root.parent.glob("*/.cache/auto_research/s0_semantic_enrichment/*")):
        if not candidate.is_dir() or candidate.name not in model_dirs:
            continue
        if candidate.parent.parent.parent.parent == project_root:
            continue
        roots.append(candidate)
    return roots


def _select_best_s0_semantic_record(records: dict[str, dict[str, Any]], candidate: dict[str, Any]) -> None:
    chunk = candidate.get("chunk") if isinstance(candidate.get("chunk"), dict) else {}
    chunk_id = str(chunk.get("chunk_id") or candidate.get("chunk_id") or "")
    if not chunk_id:
        return
    current = records.get(chunk_id)
    if current is None or _s0_semantic_record_score(candidate) >= _s0_semantic_record_score(current):
        records[chunk_id] = candidate


def _alias_s0_semantic_records_by_text(records: dict[str, dict[str, Any]], chunks: list[dict[str, Any]], *, config: dict[str, Any]) -> None:
    by_signature: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records.values():
        chunk = record.get("chunk") if isinstance(record.get("chunk"), dict) else {}
        signature = (
            str(chunk.get("source_type") or ""),
            str(chunk.get("text_sha256") or ""),
            str(record.get("prompt_version") or ""),
            str(record.get("model") or ""),
        )
        if not all(signature):
            continue
        current = by_signature.get(signature)
        if current is None or _s0_semantic_record_score(record) >= _s0_semantic_record_score(current):
            by_signature[signature] = record
    model = _s0_semantic_model_from_config(config)
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id or chunk_id in records:
            continue
        source_type = str(chunk.get("source_type") or "")
        signature = (
            source_type,
            _s1_semantic_text_sha(chunk, config=config),
            S0_CODE_SEMANTIC_ENRICHMENT_PROMPT_VERSION if source_type == "code" else S0_SEMANTIC_ENRICHMENT_PROMPT_VERSION,
            model,
        )
        record = by_signature.get(signature)
        if record is not None:
            aliased = dict(record)
            aliased["chunk"] = {**(record.get("chunk") if isinstance(record.get("chunk"), dict) else {}), **chunk, "source_type": source_type, "text_sha256": signature[1]}
            records[chunk_id] = aliased


def _s1_semantic_text_sha(chunk: dict[str, Any], *, config: dict[str, Any]) -> str:
    intake = config.get("intake") if isinstance(config, dict) else {}
    semantic = (intake or {}).get("semantic_enrichment") if isinstance(intake, dict) else {}
    semantic = semantic if isinstance(semantic, dict) else {}
    source_type = str(chunk.get("source_type") or "")
    max_chars = int((semantic.get("code_max_input_chars") if source_type == "code" else semantic.get("max_input_chars")) or (3000 if source_type == "code" else 6000))
    text = str(chunk.get("text") or chunk.get("content") or chunk.get("text_preview") or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _s0_semantic_record_score(record: dict[str, Any]) -> tuple[int, int, int, str]:
    chunk = record.get("chunk") if isinstance(record.get("chunk"), dict) else {}
    source_type = str(chunk.get("source_type") or record.get("source_type") or "")
    prompt_version = str(record.get("prompt_version") or "")
    prompt_score = 0
    if source_type == "code" and prompt_version == S0_CODE_SEMANTIC_ENRICHMENT_PROMPT_VERSION:
        prompt_score = 3
    elif prompt_version == S0_SEMANTIC_ENRICHMENT_PROMPT_VERSION:
        prompt_score = 2
    elif prompt_version:
        prompt_score = 1
    enrichment = record.get("enrichment") if isinstance(record.get("enrichment"), dict) else {}
    quality_score = 0
    if enrichment.get("semantic_summary"):
        quality_score += 1
    if enrichment.get("retrieval_keywords"):
        quality_score += 1
    if not record.get("fallback_used"):
        quality_score += 1
    return (prompt_score, quality_score, 0 if record.get("cache_status") == "dry_run" else 1, str(record.get("generated_at") or ""))


def _apply_s1_semantic_records_to_chunks(chunks: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]], *, source_type: str) -> list[dict[str, Any]]:
    updated = []
    for chunk in chunks:
        item = dict(chunk)
        item.setdefault("source_type", source_type)
        chunk_id = str(item.get("chunk_id") or "")
        record = records_by_id.get(chunk_id)
        enrichment = record.get("enrichment") if isinstance(record, dict) and isinstance(record.get("enrichment"), dict) else None
        if enrichment:
            item["semantic_enrichment"] = {
                "schema_version": S0_SEMANTIC_ENRICHMENT_SCHEMA_VERSION,
                "provider": record.get("provider"),
                "model": record.get("model"),
                "prompt_version": record.get("prompt_version"),
                "cache_status": record.get("cache_status"),
                "fallback_used": bool(record.get("fallback_used")),
                "generated_at": record.get("generated_at"),
                **enrichment,
            }
        enrichment = item.get("semantic_enrichment") if isinstance(item.get("semantic_enrichment"), dict) else enrichment
        if isinstance(enrichment, dict):
            item["semantic_summary"] = str(enrichment.get("semantic_summary") or item.get("semantic_summary") or "")
            item["mechanism_tags"] = _dedupe_texts([*(item.get("mechanism_tags") or []), *(enrichment.get("mechanism_tags") or [])], max_items=20)
            item["failure_modes"] = _dedupe_texts([*(item.get("failure_modes") or []), *(enrichment.get("failure_modes") or [])], max_items=20)
            item["retrieval_keywords"] = _dedupe_texts(
                [*(item.get("keywords") or []), *(item.get("retrieval_keywords") or []), *(enrichment.get("retrieval_keywords") or []), *(enrichment.get("mechanism_tags") or [])],
                max_items=40,
            )
        updated.append(item)
    return updated


def _merge_s1_semantic_fields_into_chunk_index(chunk_index: dict[str, Any], chunk_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(chunk_index, dict):
        return {}
    merged = dict(chunk_index)
    entries = []
    enriched_count = 0
    for entry in chunk_index.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        chunk = chunk_by_id.get(str(item.get("chunk_id") or ""))
        if isinstance(chunk, dict):
            for key in ["semantic_summary", "mechanism_tags", "failure_modes", "retrieval_keywords"]:
                value = chunk.get(key)
                if value:
                    item[key] = value
            if chunk.get("semantic_enrichment"):
                item["semantic_enrichment"] = chunk.get("semantic_enrichment")
                enriched_count += 1
        entries.append(item)
    merged["entries"] = entries
    metadata = dict(merged.get("metadata") or {})
    metadata["s1_semantic_enrichment_entries"] = enriched_count
    merged["metadata"] = metadata
    return merged


def _s1_semantic_merge_report(
    *,
    records_by_id: dict[str, dict[str, Any]],
    source_paths: list[str],
    paper_chunks: list[dict[str, Any]],
    rebuttal_chunks: list[dict[str, Any]],
    code_chunks: list[dict[str, Any]],
    chunk_index: dict[str, Any],
) -> dict[str, Any]:
    def count_enriched(items: list[dict[str, Any]]) -> int:
        return sum(1 for item in items if isinstance(item, dict) and isinstance(item.get("semantic_enrichment"), dict))

    entries = chunk_index.get("entries") if isinstance(chunk_index, dict) and isinstance(chunk_index.get("entries"), list) else []
    records_by_source: dict[str, int] = {}
    fallback_count = 0
    prompt_versions: dict[str, int] = {}
    for record in records_by_id.values():
        chunk = record.get("chunk") if isinstance(record.get("chunk"), dict) else {}
        source_type = str(chunk.get("source_type") or "unknown")
        records_by_source[source_type] = records_by_source.get(source_type, 0) + 1
        if record.get("fallback_used"):
            fallback_count += 1
        prompt_version = str(record.get("prompt_version") or "unknown")
        prompt_versions[prompt_version] = prompt_versions.get(prompt_version, 0) + 1
    return {
        "schema_version": "c2c_s1_semantic_enrichment_merge_report_v1",
        "generated_at": now_utc(),
        "status": "ok",
        "records_loaded": len(records_by_id),
        "records_by_source_type": records_by_source,
        "fallback_records_loaded": fallback_count,
        "prompt_versions": prompt_versions,
        "source_paths": _dedupe_texts(source_paths, max_items=200),
        "chunks_enriched": {
            "paper": count_enriched(paper_chunks),
            "rebuttal": count_enriched(rebuttal_chunks),
            "code": count_enriched(code_chunks),
        },
        "chunk_index_entries_enriched": sum(1 for entry in entries if isinstance(entry, dict) and isinstance(entry.get("semantic_enrichment"), dict)),
    }


def _dedupe_texts(values: list[Any], *, max_items: int = 40) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= max_items:
            break
    return result


def _code_repo_map_markdown(repo_map: dict[str, Any]) -> str:
    if not isinstance(repo_map, dict):
        return "# Code Repo Map\n\nNo code repo map available.\n"
    lines = ["# Code Repo Map", ""]
    counts = repo_map.get("counts") or {}
    lines.append(f"- Files: {counts.get('files', 0)}")
    lines.append(f"- Symbols: {counts.get('symbols', 0)}")
    lines.append(f"- Chunks: {counts.get('chunks', 0)}")
    lines.append(f"- Edges: {counts.get('edges', 0)}")
    for file_entry in repo_map.get("files", [])[:80]:
        if not isinstance(file_entry, dict):
            continue
        lines.extend(["", f"## {file_entry.get('path')}"])
        lines.append(f"- edit_surface: {file_entry.get('edit_surface')}")
        for symbol in (file_entry.get("symbols") or [])[:30]:
            if isinstance(symbol, dict):
                lines.append(f"- `{symbol.get('symbol')}` ({symbol.get('kind')}) lines {symbol.get('start_line')}-{symbol.get('end_line')}")
    return "\n".join(lines).strip() + "\n"


def _s1_novelty_auditor_config(config: dict[str, Any], *, mode: str = "c2c") -> dict[str, Any]:
    cfg = ((config.get("agents") or {}).get("s1_novelty_auditor") or {})
    if not isinstance(cfg, dict):
        cfg = {}
    default_enabled = False
    return {
        "enabled": bool(cfg.get("enabled", default_enabled)),
        "threshold": float(cfg.get("threshold") or 0.58),
        "max_revision_rounds": int(cfg.get("max_revision_rounds") or 1),
        "timeout_seconds": int(cfg.get("timeout_seconds") or (config.get("llm", {}) or {}).get("timeout_seconds") or 900),
        "session_key": str(cfg.get("session_key") or ("s1:c2c_novelty_auditor" if mode == "c2c" else "s1:generic_novelty_auditor")),
        "resume_enabled": bool(cfg.get("resume_enabled", True)),
        "sandbox": str(cfg.get("sandbox") or "read-only"),
        "approval_policy": str(cfg.get("approval_policy") or "never"),
        "json_events": bool(cfg.get("json_events", True)),
        "model": cfg.get("model"),
    }


def _s1_codex_agent_config(config: dict[str, Any], *, mode: str = "c2c") -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "strategy": "codex_resume_evidence_agent",
        "session_key": "s1:c2c_evidence_direction" if mode == "c2c" else "s1:generic_evidence_direction",
        "max_json_repairs": 2,
        "resume_enabled": True,
        "duplicate_direction_reset_threshold": 1,
    }
    agent_cfg = ((config.get("agents") or {}).get("s1_evidence_agent") or {})
    c2c_cfg = ((config.get("c2c") or {}).get("s1_evidence_agent") or {}) if mode == "c2c" else {}
    generic_cfg = ((config.get("literature") or {}).get("s1_evidence_agent") or {}) if mode != "c2c" else {}
    if isinstance(agent_cfg, dict):
        cfg.update(agent_cfg)
    if isinstance(generic_cfg, dict):
        cfg.update(generic_cfg)
    if isinstance(c2c_cfg, dict):
        cfg.update(c2c_cfg)
    return cfg


def _c2c_s1_two_phase_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = (((config.get("ideation") or {}).get("c2c_s1_two_phase") or {}) if isinstance(config.get("ideation"), dict) else {})
    raw = raw if isinstance(raw, dict) else {}
    llm_timeout = int((config.get("llm", {}) or {}).get("timeout_seconds") or 900)
    request_raw = raw.get("request_agent") if isinstance(raw.get("request_agent"), dict) else {}
    direction_raw = raw.get("direction_agent") if isinstance(raw.get("direction_agent"), dict) else {}
    retriever_raw = raw.get("retriever") if isinstance(raw.get("retriever"), dict) else {}
    shared_session_key = str(raw.get("session_key") or "s1:c2c_evidence_direction")
    request_session_key = str(request_raw.get("session_key") or direction_raw.get("session_key") or shared_session_key)
    direction_session_key = str(direction_raw.get("session_key") or request_session_key)
    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_request_revision_rounds": int(raw.get("max_request_revision_rounds") or 1),
        "max_direction_followup_rounds": int(raw.get("max_direction_followup_rounds") or 1),
        "request_agent": {
            "session_key": request_session_key,
            "max_json_repairs": int(request_raw.get("max_json_repairs") or ((_s1_codex_agent_config(config, mode="c2c").get("max_json_repairs") or 2))),
            "timeout_seconds": int(request_raw.get("timeout_seconds") or llm_timeout),
            "resume_enabled": bool(request_raw.get("resume_enabled", True)),
            "sandbox": request_raw.get("sandbox") or "read-only",
            "approval_policy": request_raw.get("approval_policy") or "never",
            "json_events": bool(request_raw.get("json_events", True)),
            "model": request_raw.get("model"),
        },
        "retriever": {
            "top_k_per_request": int(retriever_raw.get("top_k_per_request") or 2),
            "min_score": float(retriever_raw.get("min_score") or 0.0),
            "max_total_items": int(retriever_raw.get("max_total_items") or 12),
            "require_deterministic_bundle": bool(retriever_raw.get("require_deterministic_bundle", True)),
        },
        "direction_agent": {
            "session_key": direction_session_key,
            "max_json_repairs": int(direction_raw.get("max_json_repairs") or ((_s1_codex_agent_config(config, mode="c2c").get("max_json_repairs") or 2))),
            "timeout_seconds": int(direction_raw.get("timeout_seconds") or llm_timeout),
            "resume_enabled": bool(direction_raw.get("resume_enabled", True)),
            "sandbox": direction_raw.get("sandbox") or "read-only",
            "approval_policy": direction_raw.get("approval_policy") or "never",
            "json_events": bool(direction_raw.get("json_events", True)),
            "model": direction_raw.get("model"),
        },
    }


def _s1_codex_evidence_prompt(
    *,
    topic: str,
    evidence_brief: dict[str, Any],
    chunk_index: dict[str, Any],
    code_intake_report: dict[str, Any],
    implementation_surface_map: dict[str, Any],
    code_retrieval_index: dict[str, Any],
    baseline: dict[str, Any],
    negative_memory: dict[str, Any],
    rebuttal_matrix: dict[str, Any],
    feedback: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    shared_method_memory = shared_method_memory_for_prompt(
        config,
        limit=12,
        query_context=shared_method_memory_query_context(
            config,
            topic=topic,
            feedback=feedback,
            negative_memory=negative_memory,
        ),
    )
    context_payload = {
        "schema_version": "c2c_s1_codex_prompt_context_v1",
        "topic": topic,
        "task": "Choose one broad C2C mechanism direction for effect-first discovery. Do not generate concrete S2/S2.5 variants.",
        "available_artifacts": [
            "intake/c2c/evidence_brief.json",
            "intake/c2c/chunk_index.json",
            "intake/c2c/chunk_index.jsonl",
            "intake/c2c/paper_chunks.jsonl",
            "intake/c2c/rebuttal_chunks.jsonl",
            "intake/c2c/code_chunks.jsonl",
            "intake/c2c/code_edges.jsonl",
            "intake/c2c/code_repo_map.md",
            "intake/c2c/implementation_surface_map.json",
            "intake/c2c/code_intake_report.json",
            "intake/c2c/code_retrieval_index.json",
            "intake/c2c/negative_result_memory.json",
            "intake/shared_method_failure_memory.json",
            "experiment/results/failure_feedback.json",
            "plan/direction_scorecard.json",
        ],
        "baseline": _compact_json_value(baseline, max_chars=4000),
        "evidence_brief": _compact_json_value(evidence_brief, max_chars=9000),
        "chunk_catalog": _compact_json_value(_summarize_chunk_index_for_prompt(chunk_index), max_chars=9000),
        "code_intake_report": _compact_json_value(code_intake_report, max_chars=5000),
        "implementation_surface_map": _compact_json_value(implementation_surface_map, max_chars=6000),
        "code_retrieval_index": _compact_json_value(code_retrieval_index, max_chars=5000),
        "negative_memory": _compact_json_value(negative_memory, max_chars=5000),
        "shared_method_failure_memory": _compact_json_value(shared_method_memory, max_chars=7000),
        "rebuttal_matrix": _compact_json_value(rebuttal_matrix, max_chars=5000),
        "method_level_failure_feedback": _compact_json_value(feedback[:12], max_chars=7000),
    }
    output_contract = {
        "schema_version": "c2c_s1_codex_direction_v1",
        "status": "ok",
        "evidence_requests": [
            {
                "query": "short search target you investigated",
                "source_type": "paper|rebuttal|code|artifact",
                "desired_evidence": "method|risk|implementation|failure",
                "why_needed": "why this evidence matters",
            }
        ],
        "evidence_bundle": {
            "items": [
                {
                    "chunk_id": "chunk id or artifact anchor",
                    "source_path": "path inspected",
                    "source_type": "paper|rebuttal|code|artifact",
                    "summary": "short factual evidence summary",
                    "supports": ["direction id or claim"],
                    "risks": ["risk or counterevidence, if any"],
                }
            ]
        },
        "direction_decision": {
            "direction_id": "stable snake_case id",
            "mechanism_direction": "short mechanism name",
            "mechanism_type": "recognized mechanism label",
            "mechanism_axis": "scoring|routing|span_selection|normalization|training_signal|fallback",
            "integration_point": "aligner|projector|wrapper|train_loss|recipe",
            "control_signal": "confidence|entropy|span_agreement|utility|pathology|semantic_similarity",
            "core_hypothesis": "effect hypothesis",
            "why_baseline_fails": "why the current baseline lacks this mechanism",
            "expected_metric_signature": {"primary_metric": "three_dataset_mean", "expected_direction": "increase", "diagnostics": []},
            "required_evidence_refs": [{"source_type": "paper|rebuttal|code|artifact", "source_label": "chunk or artifact", "claim": "supporting evidence"}],
            "counterevidence_refs": [{"source_type": "paper|rebuttal|code|artifact|failure_feedback", "source_label": "chunk or artifact", "claim": "risk evidence"}],
            "implementation_surface_refs": [{"source_type": "code", "source_label": "file or symbol", "claim": "likely implementation surface"}],
            "known_negative_memory_refs": ["memory_id or failed pattern ids"],
            "go_to_s2_conditions": ["conditions that make this direction ready for S2 variant planning"],
            "return_to_s1_conditions": ["method-level conditions that should abandon this direction"],
            "allowed_variants": ["variant families allowed inside this direction"],
            "forbidden_patterns": ["patterns not to repeat"],
            "target_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
            "failure_focus": ["what S2/S3 should diagnose"],
            "expected_files": ["high-level likely edit surfaces, not a concrete patch plan"],
            "verification_commands": ["required checks S2/S3 must preserve"],
            "rationale": "why this direction is worth trying now",
            "used_shared_memory_refs": ["memory_id values from shared_method_failure_memory that influenced this decision, or []"],
        },
        "used_shared_memory_refs": ["memory_id values from shared_method_failure_memory that influenced this decision, or []"],
        "selected_ideas": [
            {
                "id": "same as direction_decision.direction_id",
                "title": "high-level direction title",
                "selected": True,
                "hypothesis": "same high-level direction hypothesis",
                "novelty_score": 7,
                "feasibility_score": 7,
                "mechanism_type": "one of: utility_predicted_cache_routing, counterfactual_training_objective, semantic_span_graph_alignment, verifier_guided_cache_acceptance, latent_bridge_memory, pathology_conditioned_controller",
                "description": "broad direction only, not a concrete patch variant",
                "motivation": "why S2 should explore this direction",
                "reviewer_risk_response": "method-level risk and mitigation",
                "expected_files": ["likely edit surfaces, not a full patch plan"],
                "verification_commands": ["checks S2/S3 should preserve"],
                "evidence_refs": [{"source_type": "paper", "source_label": "chunk or artifact", "claim": "supporting fact"}],
                "counterevidence_refs": [{"source_type": "failure_feedback", "source_label": "chunk or artifact", "claim": "risk fact"}],
                "code_refs": [{"source_type": "code", "source_label": "file or symbol", "claim": "implementation surface"}],
                "s1_allowed_variants": ["broad variant families S2 may explore"],
                "s1_forbidden_patterns": ["method patterns S2 must avoid"],
                "used_shared_memory_refs": ["same memory ids used by this direction, or []"],
            }
        ],
        "negative_constraints": {
            "reviewer_concerns": ["method-level concern to respect"],
            "forbidden_idea_ids": ["failed method ids, not S2.5 coding errors"],
            "forbidden_patterns": ["method patterns to avoid"],
            "failure_feedback_rules": ["how S2/S3 should use method-level failures"],
        },
        "decision_chain": {
            "evidence": ["supporting fact ids"],
            "counterevidence": ["risk fact ids"],
            "conclusion": "one sentence decision",
        },
    }
    return "\n\n".join(
        [
            "You are the S1 Codex resume evidence agent for an automated C2C research loop.",
            "Work read-only. You may inspect the listed artifacts with shell commands if the brief is not enough. Do not edit files.",
            "S1 only chooses a method direction. Ignore S2.5 implementation/runtime errors as method evidence unless they reveal an actual method-level infeasibility.",
            "shared_method_failure_memory.memory_catalog/recent_entries is a lightweight retrieved error catalog, ranked by memory_retrieval.combined_score and memory_quality.priority; follow retrieval_policy/ranking_policy and prioritize high_quality_memory_ids, especially proxy_full_false_positive, full_train_failure, proxy_dataset_misprediction, cross_project_mechanism_failure, and ablation_evidence. If a catalog item seems decision-relevant, inspect the full memory via full_memory_access/read_hint before relying on detailed evidence.",
            "If shared_method_failure_memory affects the direction or forbidden patterns, copy the exact memory_id values into used_shared_memory_refs. Use [] if none affected the decision.",
            "Pick exactly one mechanism direction. Do not produce concrete patch variants; S2 will do that from direction_decision.allowed_variants.",
            "Optimize for effect-first discovery: the output should help S2/S2.5 find a runnable patch with cheap proxy upside and no evaluator contamination.",
            "selected_ideas must contain exactly one selected high-level direction card. It is the S1 output passed to S2; do not make the program infer it for you.",
            "Return only one valid JSON object. No markdown, no comments, no prose outside JSON.",
            "Context JSON:",
            json.dumps(context_payload, ensure_ascii=False, indent=2),
            "Required JSON shape:",
            json.dumps(output_contract, ensure_ascii=False, indent=2),
        ]
    )


def _generic_s1_codex_evidence_prompt(
    *,
    topic: str,
    papers: list[dict[str, Any]],
    survey: str,
    theme_map_path: str | None,
    resources: dict[str, Any],
    config: dict[str, Any],
    excluded_direction_semantic_hashes: list[str] | None = None,
) -> str:
    shared_method_memory = shared_method_memory_for_prompt(
        config,
        limit=12,
        query_context=shared_method_memory_query_context(config, topic=topic),
    )
    compact_papers = [
        {
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "source": paper.get("source"),
            "venue": paper.get("venue"),
            "abstract": (paper.get("abstract") or paper.get("text_snippet") or "")[:900],
            "local_pdf_path": paper.get("local_pdf_path"),
        }
        for paper in papers[:12]
    ]
    context_payload = {
        "schema_version": "generic_s1_codex_prompt_context_v1",
        "topic": topic,
        "task": "Choose one high-level research idea/direction for S2 to plan. Do not generate concrete experiment variants.",
        "available_artifacts": [
            "literature/papers/metadata.json",
            "literature/survey.md",
            *(["literature/theme_map.md"] if theme_map_path else []),
            "meta/negative_memory.jsonl",
            "intake/shared_method_failure_memory.json",
            "experiment/results/failure_feedback.json",
            "plan/performance_feedback.json",
        ],
        "papers": compact_papers,
        "survey_excerpt": survey[:10000],
        "theme_map_path": theme_map_path,
        "local_resources": _compact_json_value(resources, max_chars=6000),
        "shared_method_failure_memory": _compact_json_value(shared_method_memory, max_chars=7000),
        "excluded_direction_semantic_hashes": list(excluded_direction_semantic_hashes or []),
    }
    output_contract = {
        "schema_version": "generic_s1_codex_direction_v1",
        "status": "ok",
        "evidence_requests": [
            {
                "query": "evidence or artifact you inspected",
                "source_type": "paper|artifact|code|memory",
                "desired_evidence": "method|risk|implementation|benchmark",
                "why_needed": "why this evidence matters",
            }
        ],
        "evidence_bundle": {
            "items": [
                {
                    "source_path": "artifact path or paper id",
                    "source_type": "paper|artifact|code|memory",
                    "summary": "short factual evidence summary",
                    "supports": ["idea id or claim"],
                    "risks": ["risk or counterevidence, if any"],
                }
            ]
        },
        "direction_decision": {
            "direction_id": "stable snake_case id",
            "title": "high-level research idea title",
            "mechanism_axis": "method axis S2 should explore",
            "integration_point": "likely experiment/code integration point",
            "control_signal": "signal or variable S2 should control",
            "core_hypothesis": "main effect hypothesis",
            "why_baseline_fails": "why current baselines are expected to miss the effect",
            "expected_metric_signature": {"primary_metric": "primary_metric", "expected_direction": "increase", "diagnostics": []},
            "required_evidence_refs": [{"source_type": "paper|artifact|code|memory", "source_label": "paper or artifact", "claim": "supporting evidence"}],
            "counterevidence_refs": [{"source_type": "paper|artifact|code|memory", "source_label": "paper or artifact", "claim": "risk evidence"}],
            "implementation_surface_refs": [{"source_type": "artifact|code", "source_label": "resource or file", "claim": "likely S2 implementation surface"}],
            "known_negative_memory_refs": ["memory_id or failed direction ids"],
            "go_to_s2_conditions": ["conditions that make this direction ready for S2 planning"],
            "return_to_s1_conditions": ["method-level conditions that should abandon this direction"],
            "allowed_variants": ["broad variant families S2 may explore"],
            "forbidden_patterns": ["method patterns not to repeat"],
            "target_datasets": ["dataset or benchmark names if known"],
            "failure_focus": ["what S2/S3 should diagnose"],
            "rationale": "why this direction is worth trying now",
            "used_shared_memory_refs": ["memory_id values from shared_method_failure_memory that influenced this decision, or []"],
        },
        "used_shared_memory_refs": ["memory_id values from shared_method_failure_memory that influenced this decision, or []"],
        "selected_ideas": [
            {
                "id": "same as direction_decision.direction_id",
                "title": "high-level research idea title",
                "selected": True,
                "hypothesis": "main effect hypothesis",
                "novelty_score": 7,
                "feasibility_score": 7,
                "description": "broad idea only, not a concrete S2 variant",
                "motivation": "why S2 should plan this direction",
                "expected_contribution": "expected research contribution",
                "key_baselines": ["baseline names or families"],
                "required_compute": "rough compute expectation",
                "key_references": ["paper ids or artifact refs"],
                "evidence_refs": [{"source_type": "paper", "source_label": "paper id or artifact", "claim": "supporting fact"}],
                "counterevidence_refs": [{"source_type": "artifact", "source_label": "artifact or risk", "claim": "risk fact"}],
                "used_shared_memory_refs": ["same memory ids used by this idea, or []"],
            }
        ],
        "negative_constraints": {
            "forbidden_idea_ids": ["failed direction ids if any"],
            "forbidden_patterns": ["method patterns to avoid"],
            "failure_feedback_rules": ["how S2/S3 should use failures"],
        },
        "decision_chain": {
            "evidence": ["supporting fact ids"],
            "counterevidence": ["risk fact ids"],
            "conclusion": "one sentence decision",
        },
    }
    return "\n\n".join(
        [
            "You are the global S1 Codex resume evidence agent for an automated research loop.",
            "Work read-only. You may inspect listed artifacts if the brief is not enough. Do not edit files.",
            "S1 chooses one high-level research idea/direction. S2 turns it into concrete experiment candidates.",
            "shared_method_failure_memory.memory_catalog/recent_entries is a lightweight retrieved error catalog, ranked by memory_retrieval.combined_score and memory_quality.priority; follow retrieval_policy/ranking_policy and prioritize high_quality_memory_ids, especially proxy_full_false_positive, full_train_failure, proxy_dataset_misprediction, cross_project_mechanism_failure, and ablation_evidence. If a catalog item seems decision-relevant, inspect the full memory via full_memory_access/read_hint before relying on detailed evidence.",
            "If shared_method_failure_memory affects the idea or forbidden patterns, copy the exact memory_id values into used_shared_memory_refs. Use [] if none affected the decision.",
            "Return exactly one selected idea in selected_ideas. Do not let the program infer missing idea content.",
            "Return only one valid JSON object. No markdown, no comments, no prose outside JSON.",
            "Context JSON:",
            json.dumps(context_payload, ensure_ascii=False, indent=2),
            "Required JSON shape:",
            json.dumps(output_contract, ensure_ascii=False, indent=2),
        ]
    )


def _run_s1_c2c_evidence_request_agent(
    *,
    project_root: Path,
    config: dict[str, Any],
    topic: str,
    evidence_brief: dict[str, Any],
    chunk_index: dict[str, Any],
    code_intake_report: dict[str, Any],
    implementation_surface_map: dict[str, Any],
    code_retrieval_index: dict[str, Any],
    baseline: dict[str, Any],
    negative_memory: dict[str, Any],
    rebuttal_matrix: dict[str, Any],
    feedback: list[dict[str, Any]],
    shared_memory: dict[str, Any],
    retrieval_feedback: list[str] | None = None,
) -> dict[str, Any]:
    cfg = _c2c_s1_two_phase_config(config)
    agent_cfg = dict(cfg.get("request_agent") or {})
    session_key = str(agent_cfg.get("session_key") or "s1:c2c_evidence_request")
    if not shutil.which("codex"):
        return {"status": "blocked", "phase": "evidence_request_agent", "reason": "codex executable not found", "session_key": session_key, "attempts": []}
    session_record = _load_s1_codex_session_record(project_root, session_key)
    session_id = _session_id_from_record(session_record) if agent_cfg.get("resume_enabled", True) else None
    attempts: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    previous_output = ""
    previous_status = ""
    for attempt_idx in range(max(1, int(agent_cfg.get("max_json_repairs") or 2) + 1)):
        prompt = (
            _s1_c2c_json_repair_prompt("evidence_request_agent", validation_errors, previous_output, previous_status)
            if attempt_idx > 0
            else _s1_c2c_evidence_request_prompt(
                topic=topic,
                evidence_brief=evidence_brief,
                chunk_index=chunk_index,
                code_intake_report=code_intake_report,
                implementation_surface_map=implementation_surface_map,
                code_retrieval_index=code_retrieval_index,
                baseline=baseline,
                negative_memory=negative_memory,
                rebuttal_matrix=rebuttal_matrix,
                feedback=feedback,
                shared_memory=shared_memory,
                retrieval_feedback=retrieval_feedback or [],
            )
        )
        call = _run_s1_codex_cli_once(
            project_root=project_root,
            config=config,
            session_key=session_key,
            session_id=session_id,
            prompt=prompt,
            timeout_seconds=int(agent_cfg.get("timeout_seconds") or (config.get("llm", {}) or {}).get("timeout_seconds") or 900),
            repair_attempt=attempt_idx > 0,
            attempt=attempt_idx + 1,
            mode="c2c",
            task_prefix="Follow this task exactly. You are S1a evidence_request_agent. Do not choose a direction. Return only valid JSON.",
            agent_config={**_s1_codex_agent_config(config, mode="c2c"), **agent_cfg, "session_key": session_key},
        )
        if call.get("session_id"):
            session_id = str(call["session_id"])
        attempts.append(_s1_codex_attempt_summary(call))
        payload = call.get("payload") if isinstance(call.get("payload"), dict) else None
        if call.get("status") == "ok" and payload is not None:
            plan = normalize_c2c_evidence_request_plan(payload, topic=topic)
            validation_errors = validate_c2c_evidence_request_plan(plan)
            if not validation_errors:
                return {
                    "status": "ok",
                    "phase": "evidence_request_agent",
                    "evidence_request_plan": plan,
                    "session_key": session_key,
                    "session_id": session_id,
                    "repair_count": attempt_idx,
                    "attempts": attempts,
                }
            previous_status = "contract_invalid"
        else:
            validation_errors = [str(call.get("reason") or "codex output invalid")]
            previous_status = str(call.get("status") or "invalid")
            if call.get("failure_category"):
                return _s1_codex_backend_blocked_result(
                    session_key=session_key,
                    session_id=session_id,
                    used_existing_session=bool(session_record),
                    repair_count=attempt_idx,
                    attempts=attempts,
                    novelty_audits=[],
                    reset_info=None,
                    call=call,
                    validation_errors=validation_errors,
                )
        previous_output = json.dumps(payload if payload is not None else call.get("raw_text") or call.get("reason") or "", ensure_ascii=False)[-4000:]
    return {
        "status": "blocked",
        "phase": "evidence_request_agent",
        "reason": "S1a evidence request agent did not return a valid request plan",
        "session_key": session_key,
        "session_id": session_id,
        "repair_count": max(0, len(attempts) - 1),
        "attempts": attempts,
        "validation_errors": validation_errors,
    }


def _run_s1_c2c_direction_agent(
    *,
    project_root: Path,
    config: dict[str, Any],
    topic: str,
    baseline: dict[str, Any],
    negative_memory: dict[str, Any],
    feedback: list[dict[str, Any]],
    evidence_bundle: dict[str, Any],
    retrieval_trace: dict[str, Any],
    shared_memory: dict[str, Any],
) -> dict[str, Any]:
    cfg = _c2c_s1_two_phase_config(config)
    agent_cfg = dict(cfg.get("direction_agent") or {})
    novelty_cfg = _s1_novelty_auditor_config(config, mode="c2c")
    session_key = str(agent_cfg.get("session_key") or "s1:c2c_direction_decision")
    if not shutil.which("codex"):
        return {"status": "blocked", "phase": "direction_agent", "reason": "codex executable not found", "session_key": session_key, "attempts": []}
    session_record = _load_s1_codex_session_record(project_root, session_key)
    session_id = _session_id_from_record(session_record) if agent_cfg.get("resume_enabled", True) else None
    attempts: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    previous_output = ""
    previous_status = ""
    novelty_audits: list[dict[str, Any]] = []
    total_attempts = max(1, int(agent_cfg.get("max_json_repairs") or 2) + 1)
    total_attempts += max(0, int(novelty_cfg.get("max_revision_rounds") or 0)) if novelty_cfg.get("enabled") else 0
    for attempt_idx in range(total_attempts):
        if attempt_idx > 0 and previous_status == "novelty_rejected":
            prompt = previous_output
        elif attempt_idx > 0:
            prompt = _s1_c2c_json_repair_prompt("direction_agent", validation_errors, previous_output, previous_status)
        else:
            prompt = _s1_c2c_direction_prompt(
                topic=topic,
                baseline=baseline,
                negative_memory=negative_memory,
                feedback=feedback,
                evidence_bundle=evidence_bundle,
                retrieval_trace=retrieval_trace,
                shared_memory=shared_memory,
            )
        call = _run_s1_codex_cli_once(
            project_root=project_root,
            config=config,
            session_key=session_key,
            session_id=session_id,
            prompt=prompt,
            timeout_seconds=int(agent_cfg.get("timeout_seconds") or (config.get("llm", {}) or {}).get("timeout_seconds") or 1200),
            repair_attempt=attempt_idx > 0,
            attempt=attempt_idx + 1,
            mode="c2c",
            task_prefix="Follow this task exactly. You are S1c direction_agent. Use only evidence_bundle refs. Return only valid JSON.",
            agent_config={**_s1_codex_agent_config(config, mode="c2c"), **agent_cfg, "session_key": session_key},
        )
        if call.get("session_id"):
            session_id = str(call["session_id"])
        attempts.append(_s1_codex_attempt_summary(call))
        payload = call.get("payload") if isinstance(call.get("payload"), dict) else None
        if call.get("status") == "ok" and payload is not None:
            status = str(payload.get("status") or "ok").lower()
            if status in {"needs_more_evidence", "insufficient_evidence"}:
                followup_plan = _s1c_followup_evidence_request_plan(payload, topic=topic)
                validation_errors = _validate_s1c_followup_evidence_request_plan(followup_plan)
                if not validation_errors:
                    return {
                        "status": "needs_more_evidence",
                        "phase": "direction_agent",
                        "reason": str(payload.get("reason") or payload.get("request_rationale") or "S1c requested more deterministic evidence before choosing a direction."),
                        "followup_evidence_request_plan": followup_plan,
                        "session_key": session_key,
                        "session_id": session_id,
                        "repair_count": attempt_idx,
                        "attempts": attempts,
                        "novelty_audits": novelty_audits,
                    }
                previous_status = "followup_request_invalid"
                previous_output = json.dumps(payload, ensure_ascii=False)[-4000:]
                continue
            payload = _normalize_s1_c2c_direction_payload(payload, evidence_bundle=evidence_bundle)
            validation_errors = _validate_s1_c2c_direction_payload(payload, evidence_bundle=evidence_bundle)
            if not validation_errors:
                novelty_payload = {**payload, "evidence_bundle": evidence_bundle}
                novelty_audit = _run_s1_novelty_auditor(project_root=project_root, config=config, payload=novelty_payload, shared_memory=shared_memory, mode="c2c")
                novelty_audits.append(novelty_audit)
                if not novelty_audit.get("passed", True):
                    validation_errors = ["s1_novelty_audit_rejected: direction too similar to prior ideas or memories"]
                    previous_status = "novelty_rejected"
                    previous_output = _s1_revision_feedback_prompt(novelty_audit.get("audit") or novelty_audit, payload, mode="c2c")
                    continue
                return {
                    "status": "ok",
                    "phase": "direction_agent",
                    "payload": payload,
                    "session_key": session_key,
                    "session_id": session_id,
                    "repair_count": attempt_idx,
                    "attempts": attempts,
                    "novelty_audits": novelty_audits,
                }
            previous_status = "contract_invalid"
        else:
            validation_errors = [str(call.get("reason") or "codex output invalid")]
            previous_status = str(call.get("status") or "invalid")
            if call.get("failure_category"):
                return _s1_codex_backend_blocked_result(
                    session_key=session_key,
                    session_id=session_id,
                    used_existing_session=bool(session_record),
                    repair_count=attempt_idx,
                    attempts=attempts,
                    novelty_audits=novelty_audits,
                    reset_info=None,
                    call=call,
                    validation_errors=validation_errors,
                )
        previous_output = json.dumps(payload if payload is not None else call.get("raw_text") or call.get("reason") or "", ensure_ascii=False)[-4000:]
    return {
        "status": "blocked",
        "phase": "direction_agent",
        "reason": "S1c direction agent did not return valid bundle-grounded direction JSON",
        "session_key": session_key,
        "session_id": session_id,
        "repair_count": max(0, len(attempts) - 1),
        "attempts": attempts,
        "novelty_audits": novelty_audits,
        "validation_errors": validation_errors,
    }


def _s1_c2c_evidence_request_prompt(
    *,
    topic: str,
    evidence_brief: dict[str, Any],
    chunk_index: dict[str, Any],
    code_intake_report: dict[str, Any],
    implementation_surface_map: dict[str, Any],
    code_retrieval_index: dict[str, Any],
    baseline: dict[str, Any],
    negative_memory: dict[str, Any],
    rebuttal_matrix: dict[str, Any],
    feedback: list[dict[str, Any]],
    shared_memory: dict[str, Any],
    retrieval_feedback: list[str],
) -> str:
    output_contract = default_c2c_evidence_request_plan(topic=topic)
    context_payload = {
        "schema_version": "c2c_s1a_evidence_request_context_v1",
        "topic": topic,
        "task": "Create evidence requests only. Do not choose a research direction.",
        "baseline": _compact_json_value(baseline, max_chars=3000),
        "evidence_brief": _compact_json_value(evidence_brief, max_chars=6000),
        "chunk_catalog_summary": _compact_json_value(_summarize_chunk_index_for_prompt(chunk_index), max_chars=6000),
        "code_intake_summary": _compact_json_value(code_intake_report, max_chars=3500),
        "implementation_surface_summary": _compact_json_value(implementation_surface_map, max_chars=3500),
        "code_retrieval_summary": _compact_json_value(code_retrieval_index, max_chars=3000),
        "negative_memory_summary": _compact_json_value(negative_memory, max_chars=3500),
        "rebuttal_matrix_summary": _compact_json_value(rebuttal_matrix, max_chars=3500),
        "method_feedback_summary": _compact_json_value(feedback[:8], max_chars=4500),
        "shared_method_memory_summary": _compact_json_value(shared_memory, max_chars=4500),
        "retrieval_feedback_to_fix": retrieval_feedback,
        "available_source_types": ["paper", "rebuttal", "code", "failure_memory", "feedback"],
    }
    return "\n\n".join(
        [
            "You are S1a evidence_request_agent.",
            "You are not a direction selection agent.",
            "You must only produce an evidence request plan.",
            "Do not output direction_decision, selected_ideas, evidence_bundle, expected_files, or any research direction.",
            "Requests should cover paper support, code implementation surface, counterevidence, and failure memory when available.",
            "You must include competing candidate_direction_hypotheses, uncertainty_axes, discriminating_evidence_requests, and must_have_before_direction so S1b can retrieve evidence that separates plausible directions.",
            "Return only one valid JSON object.",
            "Context JSON:",
            json.dumps(context_payload, ensure_ascii=False, indent=2),
            "Required JSON shape:",
            json.dumps(output_contract, ensure_ascii=False, indent=2),
        ]
    )


def _s1_c2c_direction_prompt(
    *,
    topic: str,
    baseline: dict[str, Any],
    negative_memory: dict[str, Any],
    feedback: list[dict[str, Any]],
    evidence_bundle: dict[str, Any],
    retrieval_trace: dict[str, Any],
    shared_memory: dict[str, Any],
) -> str:
    evidence_items = [
        {
            "evidence_id": item.get("evidence_id"),
            "request_id": item.get("request_id"),
            "source_type": item.get("source_type"),
            "purpose": item.get("purpose"),
            "ref": item.get("ref"),
            "excerpt": item.get("excerpt") or item.get("text") or item.get("summary"),
            "score": item.get("score"),
        }
        for item in (evidence_bundle.get("items") or [])
        if isinstance(item, dict)
    ]
    allowed_refs = [item.get("ref") for item in evidence_items if isinstance(item.get("ref"), dict)]
    output_contract = {
        "schema_version": "c2c_s1_direction_agent_v1",
        "status": "ok",
        "direction_decision": {
            "direction_id": "stable snake_case id",
            "mechanism_direction": "human-readable high-level direction",
            "mechanism_type": "mechanism family",
            "mechanism_axis": "routing|alignment|normalization|training_signal|scoring",
            "integration_point": "wrapper|aligner|projector|train_loss|recipe",
            "control_signal": "signal S2 should manipulate",
            "core_hypothesis": "one high-level mechanism hypothesis",
            "why_baseline_fails": "why baseline misses this effect",
            "why_this_direction": "why bundle evidence supports this direction",
            "expected_metric_signature": {"primary_metric": "three_dataset_mean", "expected_direction": "increase", "diagnostics": []},
            "required_evidence_refs": ["must be copied exactly from allowed_refs"],
            "counterevidence_refs": ["must be copied exactly from allowed_refs and support counterevidence"],
            "implementation_surface_refs": ["must be copied exactly from allowed_refs with source_type=code"],
            "expected_files": ["files covered by code refs only"],
            "allowed_variants": ["broad S2 variant families"],
            "forbidden_patterns": ["patterns S2 should avoid"],
            "failure_routing_hints": ["when S2/S3 should return to S1"],
            "s2_affordance": "what S2 can instantiate from this direction",
            "verification_commands": ["py_compile", "small2048_train", "three_dataset_eval"],
            "target_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
            "failure_focus": ["diagnostics to watch"],
            "used_shared_memory_refs": ["memory ids if used, or []"],
        },
        "selected_ideas": [
            {
                "id": "same as direction_id",
                "title": "same high-level direction",
                "selected": True,
                "hypothesis": "same core hypothesis",
                "novelty_score": 7,
                "feasibility_score": 7,
                "mechanism_type": "same mechanism family",
                "description": "high-level direction only; S2 creates concrete variants",
                "motivation": "why S2 should plan this direction",
                "reviewer_risk_response": "how S2/S3 should watch counterevidence",
                "expected_files": ["code files covered by code_refs"],
                "verification_commands": ["py_compile", "small2048_train", "three_dataset_eval"],
                "evidence_refs": ["refs copied exactly from allowed_refs"],
                "counterevidence_refs": ["refs copied exactly from allowed_refs"],
                "code_refs": ["code refs copied exactly from allowed_refs"],
                "s1_allowed_variants": ["broad S2 families"],
                "s1_forbidden_patterns": ["patterns to avoid"],
                "used_shared_memory_refs": ["memory ids if used, or []"],
            }
        ],
        "candidate_direction_scorecard": {
            "schema_version": "c2c_s1_direction_candidate_scorecard_v1",
            "selected_direction_id": "same as direction_decision.direction_id",
            "candidates": [
                {
                    "direction_id": "stable candidate id",
                    "mechanism_axis": "routing|alignment|normalization|training_signal|scoring",
                    "integration_point": "wrapper|aligner|projector|train_loss|recipe",
                    "control_signal": "signal S2 would manipulate",
                    "score": 0.0,
                    "selected": False,
                    "evidence_refs": ["refs copied exactly from allowed_refs"],
                    "counterevidence_refs": ["refs copied exactly from allowed_refs when relevant"],
                    "implementation_surface_refs": ["code refs copied exactly from allowed_refs when relevant"],
                    "why_selected": "required for selected candidate only",
                    "why_not_selected": ["required for non-selected candidates"],
                }
            ],
            "comparison_axes": ["evidence support", "counterevidence resolution", "implementation surface", "negative memory"],
        },
        "negative_constraints": {"forbidden_idea_ids": [], "forbidden_patterns": [], "failure_feedback_rules": []},
            "decision_chain": {"evidence": ["evidence_id values"], "counterevidence": ["evidence_id values"], "conclusion": "one sentence"},
    }
    followup_contract = {
        "schema_version": "c2c_s1_direction_agent_v1",
        "status": "needs_more_evidence",
        "reason": "why the current deterministic bundle is insufficient",
        "followup_evidence_request_plan": {
            "evidence_requests": [
                {
                    "request_id": "specific_missing_card",
                    "source_type": "paper|rebuttal|code|failure_memory|feedback",
                    "query": "specific query for missing evidence",
                    "keywords": ["specific", "terms"],
                    "purpose": "support|counterevidence|implementation_surface|failure_memory",
                    "top_k": 2,
                    "filters": {},
                    "must_resolve": True,
                }
            ],
            "request_rationale": "why these cards are needed before selecting a direction",
        },
    }
    context_payload = {
        "schema_version": "c2c_s1c_direction_context_v1",
        "topic": topic,
        "baseline": _compact_json_value(baseline, max_chars=3000),
        "negative_memory_summary": _compact_json_value(negative_memory, max_chars=3500),
        "feedback_summary": _compact_json_value(feedback[:8], max_chars=4500),
        "shared_method_memory_summary": _compact_json_value(shared_memory, max_chars=4500),
        "evidence_bundle": {"producer": evidence_bundle.get("producer"), "retriever_version": evidence_bundle.get("retriever_version"), "items": evidence_items},
        "retrieval_trace_coverage": _compact_json_value(
            {
                key: retrieval_trace.get(key)
                for key in [
                    "coverage",
                    "candidate_counts",
                    "unfilled_must_resolve_requests",
                    "request_plan_id",
                    "candidate_direction_hypotheses",
                    "uncertainty_axes",
                    "discriminating_evidence_requests",
                    "must_have_before_direction",
                    "code_neighborhood_expansions",
                ]
            },
            max_chars=6500,
        ),
        "allowed_refs": allowed_refs,
    }
    return "\n\n".join(
        [
            "You are S1c direction_agent.",
            "You are continuing the same S1 evidence-on-demand session after S1a requested evidence and S1b deterministic retrieval returned the evidence bundle.",
            "You can only choose a high-level research direction from the supplied deterministic evidence_bundle.",
            "If the supplied bundle is not enough to choose responsibly, return status=needs_more_evidence with followup_evidence_request_plan; do not choose a direction in that response.",
            "When status=ok, you must not output evidence_requests, followup_evidence_request_plan, or evidence_bundle.",
            "You must not invent refs, papers, code facts, rebuttal facts, or failure memories outside evidence_bundle.items.",
            "All evidence_refs, counterevidence_refs, and code_refs must be copied exactly from allowed_refs.",
            "If evidence is insufficient, return status=needs_more_evidence with no selected_ideas instead of inventing evidence.",
            "expected_files must be covered by code_refs from evidence_bundle.",
            "Return only one valid JSON object.",
            "Context JSON:",
            json.dumps(context_payload, ensure_ascii=False, indent=2),
            "Required JSON shape:",
            json.dumps(output_contract, ensure_ascii=False, indent=2),
            "Alternative JSON shape only when the bundle is insufficient:",
            json.dumps(followup_contract, ensure_ascii=False, indent=2),
        ]
    )


def _s1c_followup_evidence_request_plan(payload: dict[str, Any], *, topic: str) -> dict[str, Any]:
    raw = payload.get("followup_evidence_request_plan") if isinstance(payload.get("followup_evidence_request_plan"), dict) else {}
    raw_requests = raw.get("evidence_requests") if isinstance(raw.get("evidence_requests"), list) else payload.get("followup_evidence_requests") if isinstance(payload.get("followup_evidence_requests"), list) else []
    requests: list[dict[str, Any]] = []
    for idx, request in enumerate(raw_requests):
        if not isinstance(request, dict):
            continue
        source_type = str(request.get("source_type") or "paper").strip() or "paper"
        requests.append(
            {
                "request_id": str(request.get("request_id") or f"s1c_followup_{idx + 1}_{source_type}"),
                "source_type": source_type,
                "query": str(request.get("query") or " ".join(str(item) for item in request.get("keywords") or []) or topic),
                "keywords": [str(item) for item in request.get("keywords") or [] if item],
                "purpose": str(request.get("purpose") or "support"),
                "top_k": max(1, int(request.get("top_k") or 2)),
                "filters": request.get("filters") if isinstance(request.get("filters"), dict) else {},
                "must_resolve": bool(request.get("must_resolve", True)),
            }
        )
    return {
        "schema_version": "c2c_s1_evidence_request_plan_v1",
        "request_plan_id": str(raw.get("request_plan_id") or _short_s1_hash({"topic": topic, "requests": requests})),
        "evidence_requests": requests,
        "required_source_coverage": raw.get("required_source_coverage") if isinstance(raw.get("required_source_coverage"), dict) else {},
        "retrieval_budget": raw.get("retrieval_budget") if isinstance(raw.get("retrieval_budget"), dict) else {"top_k_per_request": 2, "max_total_items": 12, "min_score": 0.0},
        "forbidden_outputs": ["direction_decision", "selected_ideas", "evidence_bundle", "expected_files"],
        "request_rationale": str(raw.get("request_rationale") or payload.get("reason") or payload.get("request_rationale") or "S1c requested additional deterministic evidence."),
    }


def _validate_s1c_followup_evidence_request_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed_sources = {"paper", "rebuttal", "code", "failure_memory", "feedback"}
    requests = plan.get("evidence_requests") if isinstance(plan, dict) else None
    if not isinstance(requests, list) or not requests:
        return ["followup_evidence_request_plan.evidence_requests must be a non-empty list"]
    seen = set()
    for idx, request in enumerate(requests):
        if not isinstance(request, dict):
            errors.append(f"followup_evidence_request_plan.evidence_requests[{idx}] must be an object")
            continue
        request_id = str(request.get("request_id") or "")
        if not request_id:
            errors.append(f"followup_evidence_request_plan.evidence_requests[{idx}].request_id missing")
        elif request_id in seen:
            errors.append(f"duplicate followup request_id: {request_id}")
        seen.add(request_id)
        if str(request.get("source_type") or "") not in allowed_sources:
            errors.append(f"followup_evidence_request_plan.evidence_requests[{idx}].source_type must be one of {sorted(allowed_sources)}")
        if not str(request.get("query") or "").strip():
            errors.append(f"followup_evidence_request_plan.evidence_requests[{idx}].query missing")
        if int(request.get("top_k") or 0) <= 0:
            errors.append(f"followup_evidence_request_plan.evidence_requests[{idx}].top_k must be positive")
    return errors


def _short_s1_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()[:12]


def _merge_s1_c2c_evidence_bundles(primary: dict[str, Any], followup: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary if isinstance(primary, dict) else {})
    items: list[dict[str, Any]] = []
    seen = set()
    for bundle in [primary, followup]:
        for item in (bundle.get("items") if isinstance(bundle, dict) else []) or []:
            if not isinstance(item, dict):
                continue
            key = _s1_evidence_item_key(item)
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
    merged["schema_version"] = str(merged.get("schema_version") or (followup or {}).get("schema_version") or "c2c_s1_deterministic_evidence_bundle_v1")
    merged["producer"] = "deterministic_retriever"
    merged["retriever_version"] = str(merged.get("retriever_version") or (followup or {}).get("retriever_version") or "")
    merged["items"] = items
    merged["merged_followup_bundle_count"] = int(merged.get("merged_followup_bundle_count") or 0) + 1
    return merged


def _merge_s1_c2c_retrieval_traces(primary: dict[str, Any], followup: dict[str, Any], *, followup_round: int) -> dict[str, Any]:
    merged = dict(primary if isinstance(primary, dict) else {})
    for key in [
        "requests",
        "evidence_requests",
        "selected_refs",
        "resolved_refs",
        "unfilled_requests",
        "unfilled_must_resolve_requests",
        "rejected_top_candidates",
        "code_neighborhood_expansions",
        "candidate_direction_hypotheses",
        "uncertainty_axes",
        "discriminating_evidence_requests",
        "must_have_before_direction",
    ]:
        merged[key] = _dedupe_trace_values([*((primary or {}).get(key) or []), *((followup or {}).get(key) or [])])
    merged["coverage_contributors"] = _merge_s1_trace_contributors((primary or {}).get("coverage_contributors"), (followup or {}).get("coverage_contributors"))
    merged["resolved_ref_count"] = len(merged.get("resolved_refs") or merged.get("selected_refs") or [])
    merged["unresolved_ref_count"] = int((primary or {}).get("unresolved_ref_count") or 0) + int((followup or {}).get("unresolved_ref_count") or 0)
    merged["deterministic"] = True
    merged["followup_rounds"] = [*((primary or {}).get("followup_rounds") or []), {"round": followup_round, "request_plan_id": (followup or {}).get("request_plan_id"), "selected_ref_count": len((followup or {}).get("selected_refs") or [])}]
    return merged


def _merge_s1_trace_contributors(primary: Any, followup: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in [primary, followup]:
        if not isinstance(payload, dict):
            continue
        for key, values in payload.items():
            if isinstance(values, list):
                merged[key] = _dedupe_trace_values([*(merged.get(key) or []), *values])
            elif values:
                merged[key] = values
    return merged


def _s1_evidence_item_key(item: dict[str, Any]) -> str:
    ref = item.get("ref") if isinstance(item.get("ref"), dict) else None
    if ref:
        return json.dumps(ref, sort_keys=True, ensure_ascii=True, default=str)
    return str(item.get("evidence_id") or item.get("chunk_id") or item.get("source_path") or json.dumps(item, sort_keys=True, ensure_ascii=True, default=str))


def _dedupe_trace_values(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _validate_s1_c2c_direction_payload(payload: dict[str, Any], *, evidence_bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["direction output must be an object"]
    for field in ["evidence_requests", "evidence_bundle"]:
        if field in payload:
            errors.append(f"direction_agent must not output {field}")
    if payload.get("followup_evidence_request_plan"):
        errors.append("direction_agent status=ok must not output followup_evidence_request_plan")
    if str(payload.get("status") or "ok").lower() not in {"ok", "direction_selected"}:
        errors.append("status must be ok or direction_selected")
    decision = payload.get("direction_decision")
    if not isinstance(decision, dict) or not decision:
        errors.append("direction_decision must be a non-empty object")
    else:
        for field in ["direction_id", "mechanism_direction", "mechanism_type", "mechanism_axis", "integration_point", "control_signal", "core_hypothesis", "why_baseline_fails", "why_this_direction", "expected_files"]:
            if decision.get(field) in (None, "", []):
                errors.append(f"direction_decision missing {field}")
        for field in ["required_evidence_refs", "counterevidence_refs", "implementation_surface_refs"]:
            if not isinstance(decision.get(field), list) or not decision.get(field):
                errors.append(f"direction_decision.{field} must be a non-empty list")
    ideas = payload.get("selected_ideas")
    if not isinstance(ideas, list) or len(ideas) != 1:
        errors.append("selected_ideas must contain exactly one high-level direction card")
    else:
        idea = ideas[0]
        if not isinstance(idea, dict):
            errors.append("selected_ideas[0] must be an object")
        else:
            for field in ["id", "title", "hypothesis", "novelty_score", "feasibility_score", "expected_files", "verification_commands", "evidence_refs", "counterevidence_refs", "code_refs", "reviewer_risk_response", "mechanism_type"]:
                if idea.get(field) in (None, "", []):
                    errors.append(f"selected_ideas[0] missing {field}")
    if not isinstance(payload.get("negative_constraints"), dict):
        errors.append("negative_constraints must be an object")
    scorecard_errors = _validate_s1_direction_candidate_scorecard(payload.get("candidate_direction_scorecard"), evidence_bundle=evidence_bundle)
    errors.extend(scorecard_errors)
    if not errors:
        report = validate_direction_refs_subset_of_bundle(payload, evidence_bundle)
        if report.get("status") != "pass":
            errors.extend(direction_bundle_ref_errors_for_repair(report))
    return errors


def _normalize_s1_c2c_direction_payload(payload: dict[str, Any], *, evidence_bundle: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    decision = dict(normalized.get("direction_decision") if isinstance(normalized.get("direction_decision"), dict) else {})
    ideas = [dict(item) if isinstance(item, dict) else item for item in (normalized.get("selected_ideas") or [])] if isinstance(normalized.get("selected_ideas"), list) else normalized.get("selected_ideas")
    if decision.get("implementation_surface_refs"):
        decision["implementation_surface_refs"] = _bundle_backed_s1_code_refs(decision.get("implementation_surface_refs"), evidence_bundle=evidence_bundle)
    if isinstance(ideas, list):
        for idea in ideas[:1]:
            if isinstance(idea, dict) and idea.get("code_refs"):
                idea["code_refs"] = _bundle_backed_s1_code_refs(idea.get("code_refs"), evidence_bundle=evidence_bundle)
            if isinstance(idea, dict) and idea.get("implementation_surface_refs"):
                idea["implementation_surface_refs"] = _bundle_backed_s1_code_refs(idea.get("implementation_surface_refs"), evidence_bundle=evidence_bundle)
    normalized["direction_decision"] = decision
    if isinstance(ideas, list):
        normalized["selected_ideas"] = ideas
    expected_files = _expected_files_from_s1_direction_payload(normalized, evidence_bundle=evidence_bundle)
    if expected_files and decision.get("expected_files") in (None, "", []):
        decision["expected_files"] = expected_files
    if isinstance(ideas, list):
        for idea in ideas[:1]:
            if isinstance(idea, dict) and idea.get("expected_files") in (None, "", []):
                idea["expected_files"] = list(decision.get("expected_files") or expected_files)
        normalized["selected_ideas"] = ideas
    normalized["direction_decision"] = decision
    normalized["candidate_direction_scorecard"] = _s1_direction_candidate_scorecard(normalized, evidence_bundle=evidence_bundle)
    return normalized


def _s1_direction_candidate_scorecard(
    payload: dict[str, Any],
    *,
    evidence_bundle: dict[str, Any],
    raw_scorecard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = raw_scorecard if isinstance(raw_scorecard, dict) else payload.get("candidate_direction_scorecard") if isinstance(payload.get("candidate_direction_scorecard"), dict) else {}
    decision = payload.get("direction_decision") if isinstance(payload.get("direction_decision"), dict) else {}
    selected_ideas = payload.get("selected_ideas") if isinstance(payload.get("selected_ideas"), list) else []
    selected_idea = next((item for item in selected_ideas if isinstance(item, dict)), {})
    selected_direction_id = str(raw.get("selected_direction_id") or decision.get("direction_id") or selected_idea.get("id") or "selected_direction")
    raw_candidates = raw.get("candidates") if isinstance(raw.get("candidates"), list) else []
    candidates = [_normalize_s1_direction_candidate(item, selected_direction_id=selected_direction_id, decision=decision, selected_idea=selected_idea) for item in raw_candidates if isinstance(item, dict)]
    candidates = [item for item in candidates if item.get("direction_id")]
    if not any(item.get("selected") for item in candidates):
        candidates.insert(0, _selected_s1_direction_candidate(selected_direction_id, decision=decision, selected_idea=selected_idea))
    candidates = _dedupe_s1_direction_candidates(candidates, selected_direction_id=selected_direction_id)
    return {
        "schema_version": "c2c_s1_direction_candidate_scorecard_v1",
        "direction_id": selected_direction_id,
        "selected_direction_id": selected_direction_id,
        "candidates": candidates,
        "comparison_axes": raw.get("comparison_axes") if isinstance(raw.get("comparison_axes"), list) and raw.get("comparison_axes") else ["evidence_support", "counterevidence_resolution", "implementation_surface", "negative_memory"],
        "coverage": {
            "candidate_count": len(candidates),
            "non_selected_count": len([item for item in candidates if not item.get("selected")]),
            "bundle_ref_count": len(bundle_ref_set(evidence_bundle)),
        },
    }


def _normalize_s1_direction_candidate(
    item: dict[str, Any],
    *,
    selected_direction_id: str,
    decision: dict[str, Any],
    selected_idea: dict[str, Any],
) -> dict[str, Any]:
    direction_id = str(item.get("direction_id") or item.get("id") or "").strip()
    selected = bool(item.get("selected") or (direction_id and direction_id == selected_direction_id))
    candidate = {
        "direction_id": direction_id,
        "mechanism_axis": str(item.get("mechanism_axis") or (decision.get("mechanism_axis") if selected else "") or "unknown"),
        "integration_point": str(item.get("integration_point") or (decision.get("integration_point") if selected else "") or "unknown"),
        "control_signal": str(item.get("control_signal") or (decision.get("control_signal") if selected else "") or "unknown"),
        "score": float(item.get("score") if item.get("score") is not None else (1.0 if selected else 0.0)),
        "selected": selected,
        "evidence_refs": _as_ref_list(item.get("evidence_refs") or (decision.get("required_evidence_refs") if selected else [])),
        "counterevidence_refs": _as_ref_list(item.get("counterevidence_refs") or (decision.get("counterevidence_refs") if selected else [])),
        "implementation_surface_refs": _as_ref_list(item.get("implementation_surface_refs") or item.get("code_refs") or (decision.get("implementation_surface_refs") if selected else [])),
        "why_selected": str(item.get("why_selected") or item.get("rationale") or (decision.get("why_this_direction") if selected else "")),
        "why_not_selected": [str(reason) for reason in item.get("why_not_selected") or ([] if selected else ["Lower deterministic evidence support than the selected direction."]) if reason],
    }
    if selected and not candidate["why_not_selected"]:
        candidate["why_not_selected"] = []
    if selected and not candidate["why_selected"]:
        candidate["why_selected"] = str(selected_idea.get("motivation") or selected_idea.get("description") or "Selected as the strongest bundle-grounded S1 direction.")
    return candidate


def _selected_s1_direction_candidate(selected_direction_id: str, *, decision: dict[str, Any], selected_idea: dict[str, Any]) -> dict[str, Any]:
    return {
        "direction_id": selected_direction_id,
        "mechanism_axis": str(decision.get("mechanism_axis") or "unknown"),
        "integration_point": str(decision.get("integration_point") or "unknown"),
        "control_signal": str(decision.get("control_signal") or "unknown"),
        "score": 1.0,
        "selected": True,
        "evidence_refs": _as_ref_list(decision.get("required_evidence_refs") or selected_idea.get("evidence_refs") or []),
        "counterevidence_refs": _as_ref_list(decision.get("counterevidence_refs") or selected_idea.get("counterevidence_refs") or []),
        "implementation_surface_refs": _as_ref_list(decision.get("implementation_surface_refs") or selected_idea.get("code_refs") or []),
        "why_selected": str(decision.get("why_this_direction") or selected_idea.get("motivation") or "Selected as the strongest bundle-grounded S1 direction."),
        "why_not_selected": [],
    }


def _dedupe_s1_direction_candidates(candidates: list[dict[str, Any]], *, selected_direction_id: str) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for candidate in sorted(candidates, key=lambda item: (not item.get("selected"), -float(item.get("score") or 0.0), str(item.get("direction_id") or ""))):
        direction_id = str(candidate.get("direction_id") or "")
        if not direction_id or direction_id in seen:
            continue
        seen.add(direction_id)
        if direction_id == selected_direction_id:
            candidate["selected"] = True
            candidate["why_not_selected"] = []
        deduped.append(candidate)
    return deduped


def _validate_s1_direction_candidate_scorecard(scorecard: Any, *, evidence_bundle: dict[str, Any]) -> list[str]:
    if not isinstance(scorecard, dict):
        return ["candidate_direction_scorecard must be an object"]
    errors: list[str] = []
    if scorecard.get("schema_version") != "c2c_s1_direction_candidate_scorecard_v1":
        errors.append("candidate_direction_scorecard.schema_version must be c2c_s1_direction_candidate_scorecard_v1")
    selected_direction_id = str(scorecard.get("selected_direction_id") or "")
    candidates = scorecard.get("candidates")
    if not selected_direction_id:
        errors.append("candidate_direction_scorecard.selected_direction_id missing")
    if not isinstance(candidates, list) or not candidates:
        errors.append("candidate_direction_scorecard.candidates must be a non-empty list")
        return errors
    selected_count = 0
    allowed_refs = bundle_ref_set(evidence_bundle)
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"candidate_direction_scorecard.candidates[{idx}] must be an object")
            continue
        for field in ["direction_id", "mechanism_axis", "integration_point", "control_signal"]:
            if candidate.get(field) in (None, "", []):
                errors.append(f"candidate_direction_scorecard.candidates[{idx}] missing {field}")
        if candidate.get("selected") is True:
            selected_count += 1
            if not candidate.get("why_selected"):
                errors.append(f"candidate_direction_scorecard.candidates[{idx}] selected candidate missing why_selected")
        elif not candidate.get("why_not_selected"):
            errors.append(f"candidate_direction_scorecard.candidates[{idx}] missing why_not_selected")
        for ref_field in ["evidence_refs", "counterevidence_refs", "implementation_surface_refs"]:
            for ref in candidate.get(ref_field) or []:
                if isinstance(ref, dict) and canonical_ref_key(ref) not in allowed_refs:
                    errors.append(f"candidate_direction_scorecard.candidates[{idx}].{ref_field} contains ref outside evidence_bundle")
                    break
    if selected_count != 1:
        errors.append("candidate_direction_scorecard must contain exactly one selected candidate")
    return errors


def _as_ref_list(value: Any) -> list[Any]:
    return [item for item in value if item] if isinstance(value, list) else []


def _bundle_backed_s1_code_refs(refs: Any, *, evidence_bundle: dict[str, Any]) -> list[Any]:
    code_items = [
        item
        for item in evidence_bundle.get("items") or []
        if isinstance(item, dict) and _s1_ref_source_type(item).lower() in {"code", "implementation_surface"}
    ]
    normalized: list[Any] = []
    for ref in refs or []:
        matched = _matched_s1_code_items(ref, code_items)
        if matched:
            normalized.extend(_s1_bundle_ref(item) for item in matched)
        else:
            normalized.append(ref)
    deduped: list[Any] = []
    seen = set()
    for ref in normalized:
        key = json.dumps(ref, sort_keys=True, ensure_ascii=True, default=str) if isinstance(ref, dict) else str(ref)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _s1_bundle_ref(item: dict[str, Any]) -> dict[str, Any]:
    ref = item.get("ref") if isinstance(item.get("ref"), dict) else item
    return dict(ref)


def _expected_files_from_s1_direction_payload(payload: dict[str, Any], *, evidence_bundle: dict[str, Any]) -> list[str]:
    decision = payload.get("direction_decision") if isinstance(payload.get("direction_decision"), dict) else {}
    refs: list[Any] = []
    refs.extend(decision.get("expected_files") or [])
    refs.extend(decision.get("implementation_surface_refs") or [])
    for idea in payload.get("selected_ideas") or []:
        if not isinstance(idea, dict):
            continue
        refs.extend(idea.get("expected_files") or [])
        refs.extend(idea.get("code_refs") or [])
        refs.extend(idea.get("implementation_surface_refs") or [])
    code_items = [
        item
        for item in evidence_bundle.get("items") or []
        if isinstance(item, dict) and _s1_ref_source_type(item).lower() in {"code", "implementation_surface"}
    ]
    files: list[str] = []
    for ref in refs:
        direct = _s1_ref_file_path(ref)
        if direct:
            files.append(direct)
            continue
        matched = _matched_s1_code_items(ref, code_items)
        files.extend(_s1_ref_file_path(item) for item in matched)
    if refs and not files:
        files.extend(_s1_ref_file_path(item) for item in code_items)
    result: list[str] = []
    seen = set()
    for file_path in files:
        if not file_path or file_path in seen:
            continue
        seen.add(file_path)
        result.append(file_path)
    return result


def _matched_s1_code_items(ref: Any, code_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ref_values = _s1_ref_match_values(ref)
    if not ref_values:
        return []
    matches = []
    for item in code_items:
        item_values = _s1_ref_match_values(item)
        if ref_values & item_values:
            matches.append(item)
    return matches


def _s1_ref_match_values(ref: Any) -> set[str]:
    values: set[str] = set()
    records = []
    if isinstance(ref, dict):
        records.append(ref)
        nested = ref.get("ref")
        if isinstance(nested, dict):
            records.append(nested)
    elif isinstance(ref, str):
        values.add(ref.strip())
        normalized_file = _normalize_s1_expected_file(ref)
        if normalized_file:
            values.add(normalized_file)
    for record in records:
        for key in ["evidence_id", "chunk_id", "source_label", "label", "source_path", "path", "file", "locator"]:
            value = record.get(key)
            if value:
                raw = str(value).strip()
                values.add(raw)
                normalized_file = _normalize_s1_expected_file(raw)
                if normalized_file:
                    values.add(normalized_file)
    return {value for value in values if value}


def _s1_ref_source_type(ref: dict[str, Any]) -> str:
    nested = ref.get("ref") if isinstance(ref.get("ref"), dict) else {}
    return str(ref.get("source_type") or nested.get("source_type") or "")


def _s1_ref_file_path(ref: Any) -> str:
    records = []
    if isinstance(ref, dict):
        records.append(ref)
        nested = ref.get("ref")
        if isinstance(nested, dict):
            records.append(nested)
    elif isinstance(ref, str):
        records.append({"source_path": ref})
    for record in records:
        for key in ["source_path", "path", "file", "source_label", "chunk_id"]:
            file_path = _normalize_s1_expected_file(record.get(key))
            if file_path:
                return file_path
    return ""


def _normalize_s1_expected_file(value: Any) -> str:
    if not value:
        return ""
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    for separator in ["::", "#"]:
        if separator in text:
            text = text.split(separator, 1)[0]
    root_markers = ["/external/c2c_snapshot/", "/C2C/"]
    for marker in root_markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    else:
        for marker, prefix in [("/rosetta/", "rosetta"), ("/script/", "script"), ("/test/", "test"), ("/tests/", "tests")]:
            if marker in text:
                text = f"{prefix}/{text.split(marker, 1)[1]}"
                break
    if not ("/" in text or text.endswith(".py")):
        return ""
    return text.removeprefix("./")


def _s1_c2c_json_repair_prompt(role: str, validation_errors: list[str], previous_output: str, previous_status: str) -> str:
    return "\n\n".join(
        [
            f"You are repairing S1 {role} JSON only.",
            f"Previous status: {previous_status or 'invalid'}",
            "Validation errors:",
            json.dumps(validation_errors[:20], ensure_ascii=False, indent=2),
            "Previous output tail:",
            str(previous_output)[-3000:],
            "Return one corrected JSON object. Do not include markdown or prose.",
        ]
    )


def _s1b_retrieval_blockers(trace: dict[str, Any]) -> list[str]:
    coverage = trace.get("coverage") if isinstance(trace.get("coverage"), dict) else {}
    blockers = []
    for item in trace.get("unfilled_must_resolve_requests") or []:
        if isinstance(item, dict):
            blockers.append(f"unfilled_must_resolve_request:{item.get('request_id')}")
    if int(coverage.get("paper") or 0) < 2:
        blockers.append("retrieval_coverage.paper<2")
    if int(coverage.get("code") or 0) < 2:
        blockers.append("retrieval_coverage.code<2")
    if int(coverage.get("counterevidence") or 0) < 1:
        blockers.append("retrieval_coverage.counterevidence<1")
    return blockers


def _blocked_c2c_two_phase_result(*, reason: str, phases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "blocked",
        "strategy": "codex_two_phase_evidence_direction",
        "blocked_reason": reason,
        "evidence_session": {
            "schema_version": "c2c_s1_two_phase_session_v1",
            "status": "blocked",
            "source": "codex_cli_and_deterministic_retriever",
            "phases": phases,
            "attempts": [attempt for phase in phases for attempt in (phase.get("attempts") or []) if isinstance(phase, dict)],
            "repair_count": sum(int(phase.get("repair_count") or 0) for phase in phases if isinstance(phase, dict)),
            "reason": reason,
        },
        "selected_ideas": [],
        "negative_constraints": {},
    }


def _c2c_two_phase_session(
    *,
    request_result: dict[str, Any],
    deterministic_trace: dict[str, Any],
    direction_result: dict[str, Any],
    quality_artifacts: dict[str, Any],
    used_shared_memory_refs: list[str],
) -> dict[str, Any]:
    phases = [
        {
            "phase": "evidence_request_agent",
            "status": request_result.get("status"),
            "session_key": request_result.get("session_key"),
            "session_id": request_result.get("session_id"),
            "repair_count": request_result.get("repair_count", 0),
            "attempts": request_result.get("attempts", []),
        },
        {
            "phase": "deterministic_retriever",
            "status": "warning" if deterministic_trace.get("bootstrap_degraded_retrieval") else "ok",
            "retriever_version": deterministic_trace.get("retriever_version"),
            "request_plan_id": deterministic_trace.get("request_plan_id"),
            "selected_ref_count": len(deterministic_trace.get("selected_refs") or []),
            "coverage": deterministic_trace.get("coverage"),
            "unfilled_must_resolve_requests": deterministic_trace.get("unfilled_must_resolve_requests") or [],
            "warnings": deterministic_trace.get("bootstrap_warnings") or [],
        },
        {
            "phase": "direction_agent",
            "status": direction_result.get("status"),
            "session_key": direction_result.get("session_key"),
            "session_id": direction_result.get("session_id"),
            "repair_count": direction_result.get("repair_count", 0),
            "attempts": direction_result.get("attempts", []),
            "novelty_audits": direction_result.get("novelty_audits", []),
        },
        {
            "phase": "quality_gate",
            "status": (quality_artifacts.get("evidence_quality_score") or {}).get("gate"),
            "failed_rules": (quality_artifacts.get("evidence_quality_score") or {}).get("failed_rules", []),
        },
    ]
    attempts = [attempt for phase in phases for attempt in (phase.get("attempts") or []) if isinstance(attempt, dict)]
    return {
        "schema_version": "c2c_s1_two_phase_session_v1",
        "status": "ok",
        "source": "codex_cli_and_deterministic_retriever",
        "phases": phases,
        "attempts": attempts,
        "repair_count": int(request_result.get("repair_count") or 0) + int(direction_result.get("repair_count") or 0),
        "used_shared_memory_refs": used_shared_memory_refs,
        "novelty_audits": direction_result.get("novelty_audits", []),
    }


def _build_c2c_s1_quality_artifacts(
    *,
    project_root: Path,
    payload: dict[str, Any],
    direction: dict[str, Any],
    evidence_bundle: dict[str, Any],
    evidence_ref_report: dict[str, Any],
    novelty_audit: dict[str, Any] | list[Any],
    shared_memory_checked: bool,
    direction_bundle_ref_report: dict[str, Any] | None = None,
    deterministic_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    direction_fingerprint = build_s1_direction_fingerprint(direction, project_root=project_root)
    evidence_quality_score = build_s1_evidence_quality_score(
        direction,
        payload=payload,
        evidence_bundle=evidence_bundle,
        evidence_ref_report=evidence_ref_report,
        novelty_audit=novelty_audit,
        direction_fingerprint=direction_fingerprint,
        direction_bundle_ref_report=direction_bundle_ref_report,
        shared_memory_checked=shared_memory_checked,
    )
    evidence_retrieval_trace = build_s1_evidence_retrieval_trace(
        direction,
        payload=payload,
        evidence_ref_report=evidence_ref_report,
        evidence_quality_score=evidence_quality_score,
        direction_fingerprint=direction_fingerprint,
        deterministic_trace=deterministic_trace,
    )
    return {
        "evidence_quality_score": evidence_quality_score,
        "evidence_retrieval_trace": evidence_retrieval_trace,
        "direction_fingerprint": direction_fingerprint,
    }


def _run_s1_codex_evidence_agent(
    *,
    project_root: Path,
    config: dict[str, Any],
    prompt: str,
    max_repairs: int,
    timeout_seconds: int,
    mode: str = "c2c",
) -> dict[str, Any]:
    cfg = _s1_codex_agent_config(config, mode=mode)
    novelty_cfg = _s1_novelty_auditor_config(config, mode=mode)
    session_key = str(cfg.get("session_key") or "s1:c2c_evidence_direction")
    if not shutil.which("codex"):
        return {
            "status": "blocked",
            "reason": "codex executable not found; S1 Codex evidence agent cannot run and no fallback is enabled",
            "session_key": session_key,
            "attempts": [],
        }
    reset_info = _maybe_reset_s1_codex_session_before_run(project_root, session_key, cfg, mode=mode)
    session_record = _load_s1_codex_session_record(project_root, session_key)
    session_id = _session_id_from_record(session_record) if cfg.get("resume_enabled", True) else None
    used_existing_session = bool(session_id)
    attempts: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    previous_output = ""
    previous_status = ""
    total_attempts = max(1, int(max_repairs) + 1)
    total_attempts += max(0, int(novelty_cfg.get("max_revision_rounds") or 0)) if novelty_cfg.get("enabled") else 0
    novelty_audits: list[dict[str, Any]] = []

    for attempt_idx in range(total_attempts):
        repair_attempt = attempt_idx > 0
        if repair_attempt and previous_status == "novelty_rejected":
            current_prompt = previous_output
        elif repair_attempt:
            current_prompt = _s1_codex_json_repair_prompt(validation_errors, previous_output, previous_status, mode=mode)
        else:
            current_prompt = prompt
        call = _run_s1_codex_cli_once(
            project_root=project_root,
            config=config,
            session_key=session_key,
            session_id=session_id,
            prompt=current_prompt,
            timeout_seconds=timeout_seconds,
            repair_attempt=repair_attempt,
            attempt=attempt_idx + 1,
            mode=mode,
        )
        if call.get("session_id"):
            session_id = str(call["session_id"])
        attempts.append(_s1_codex_attempt_summary(call))
        status = str(call.get("status") or "")
        payload = call.get("payload") if isinstance(call.get("payload"), dict) else None
        if status == "ok" and payload is not None:
            validation_errors = _validate_s1_codex_payload(payload, mode=mode)
            evidence_ref_report = resolve_s1_evidence_refs(project_root, payload, mode=mode)
            if evidence_ref_report.get("status") != "pass":
                validation_errors.extend(evidence_ref_errors_for_repair(evidence_ref_report))
            if not validation_errors:
                shared_memory = shared_method_memory_for_prompt(config, limit=12)
                novelty_audit = _run_s1_novelty_auditor(
                    project_root=project_root,
                    config=config,
                    payload=payload,
                    shared_memory=shared_memory,
                    mode=mode,
                )
                novelty_audits.append(novelty_audit)
                if not novelty_audit.get("passed", True):
                    validation_errors = ["s1_novelty_audit_rejected: direction too similar to prior ideas or memories"]
                    previous_status = "novelty_rejected"
                    previous_output = _s1_revision_feedback_prompt(novelty_audit.get("audit") or novelty_audit, payload, mode=mode)
                    continue
                quality_artifacts: dict[str, Any] = {}
                if mode == "c2c":
                    used_shared_memory_refs = collect_used_shared_memory_refs(payload, shared_memory)
                    direction_decision = dict(payload.get("direction_decision") if isinstance(payload.get("direction_decision"), dict) else {})
                    direction_decision["used_shared_memory_refs"] = used_shared_memory_refs
                    selected_ideas = _s1_codex_direction_cards(payload, used_shared_memory_refs=used_shared_memory_refs)
                    negative_constraints = dict(payload.get("negative_constraints") if isinstance(payload.get("negative_constraints"), dict) else {})
                    negative_constraints["used_shared_memory_refs"] = used_shared_memory_refs
                    direction_payload = dict(payload)
                    direction_payload["direction_decision"] = direction_decision
                    direction_payload["selected_ideas"] = selected_ideas
                    direction_payload["negative_constraints"] = negative_constraints
                    direction_payload["used_shared_memory_refs"] = used_shared_memory_refs
                    direction = build_direction_contract(direction_payload, mode="c2c", used_shared_memory_refs=used_shared_memory_refs)
                    normalized_novelty = normalize_novelty_audit(novelty_audits, direction_id=str(direction.get("direction_id") or ""))
                    quality_artifacts = _build_c2c_s1_quality_artifacts(
                        project_root=project_root,
                        payload=direction_payload,
                        direction=direction,
                        evidence_bundle=payload.get("evidence_bundle") if isinstance(payload.get("evidence_bundle"), dict) else {"items": []},
                        evidence_ref_report=evidence_ref_report,
                        novelty_audit=normalized_novelty,
                        shared_memory_checked=True,
                    )
                    quality_score = quality_artifacts.get("evidence_quality_score") if isinstance(quality_artifacts.get("evidence_quality_score"), dict) else {}
                    if quality_score.get("gate") != "pass":
                        failed_rules = quality_score.get("failed_rules") if isinstance(quality_score.get("failed_rules"), list) else []
                        validation_errors = [f"s1_evidence_quality_gate_failed: {rule}" for rule in failed_rules] or ["s1_evidence_quality_gate_failed"]
                        previous_status = "contract_invalid"
                        previous_output = json.dumps(
                            {
                                "validation_errors": validation_errors,
                                "evidence_quality_score": quality_score,
                                "evidence_retrieval_trace": quality_artifacts.get("evidence_retrieval_trace"),
                                "repair_instruction": "Return a revised S1 C2C JSON contract with enough resolved paper/code evidence, counterevidence, implementation surface coverage, and novelty to pass the deterministic quality gate.",
                            },
                            ensure_ascii=False,
                        )[-4000:]
                        continue
                post_reset = _record_s1_codex_session_health(
                    project_root,
                    session_key,
                    session_id,
                    cfg,
                    payload,
                    mode=mode,
                )
                return {
                    "status": "ok",
                    "payload": payload,
                    "evidence_ref_report": evidence_ref_report,
                    "session_key": session_key,
                    "session_id": session_id,
                    "used_existing_session": used_existing_session,
                    "repair_count": attempt_idx,
                    "attempts": attempts,
                    "novelty_audits": novelty_audits,
                    **quality_artifacts,
                    "session_reset": bool(reset_info or post_reset),
                    "session_reset_reason": (post_reset or reset_info or {}).get("reason"),
                }
            previous_status = "contract_invalid"
        else:
            validation_errors = [str(call.get("reason") or f"codex output status={status}")]
            previous_status = status or "invalid"
            if call.get("failure_category"):
                return _s1_codex_backend_blocked_result(
                    session_key=session_key,
                    session_id=session_id,
                    used_existing_session=used_existing_session,
                    repair_count=attempt_idx,
                    attempts=attempts,
                    novelty_audits=novelty_audits,
                    reset_info=reset_info,
                    call=call,
                    validation_errors=validation_errors,
                )
        previous_output = str(call.get("raw_text") or call.get("reason") or "")[-4000:]

    return {
        "status": "blocked",
        "reason": "S1 Codex evidence agent did not return valid contract JSON after repair attempts",
        "session_key": session_key,
        "session_id": session_id,
        "used_existing_session": used_existing_session,
        "repair_count": max(0, total_attempts - 1),
        "attempts": attempts,
        "novelty_audits": novelty_audits,
        "session_reset": bool(reset_info),
        "session_reset_reason": (reset_info or {}).get("reason"),
        "validation_errors": validation_errors,
        "last_output_tail": previous_output[-2000:],
    }


def _run_s1_codex_cli_once(
    *,
    project_root: Path,
    config: dict[str, Any],
    session_key: str,
    session_id: str | None,
    prompt: str,
    timeout_seconds: int,
    repair_attempt: bool,
    attempt: int,
    mode: str,
    task_prefix: str | None = None,
    agent_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    llm_cfg = config.get("llm", {}) or {}
    codex_cfg = llm_cfg.get("codex_cli") or {}
    agent_cfg = agent_config or _s1_codex_agent_config(config, mode=mode)
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as handle:
        output_path = Path(handle.name)
    command = ["codex"]
    sandbox = str(agent_cfg.get("sandbox") or codex_cfg.get("sandbox") or "read-only")
    approval_policy = str(agent_cfg.get("approval_policy") or codex_cfg.get("approval_policy") or "never")
    command.extend(["-s", sandbox, "-a", approval_policy, "exec", "--skip-git-repo-check", "--output-last-message", str(output_path)])
    if agent_cfg.get("json_events", codex_cfg.get("json_events", True)):
        command.append("--json")
    model = str(agent_cfg.get("model") or llm_cfg.get("model") or "")
    if model:
        command.extend(["-m", model])
    reasoning_effort = str(agent_cfg.get("reasoning_effort") or llm_cfg.get("reasoning_effort") or "").strip()
    if reasoning_effort and reasoning_effort != "none":
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    command.extend(["-C", str(project_root.resolve())])
    if session_id:
        command.extend(["resume", session_id, "-"])
    else:
        command.append("-")

    merged_prompt = f"{task_prefix or 'Follow this task exactly. You are selecting an S1 method direction only. Do not edit files. Return only valid JSON.'}\n\n{prompt}"
    started = now_utc()
    start_monotonic = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            input=merged_prompt,
            cwd=project_root,
            timeout=max(1, int(timeout_seconds)),
            env=codex_subprocess_env(config),
        )
        raw_text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    except subprocess.TimeoutExpired as exc:
        raw_text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        call = {
            "status": "timeout",
            "reason": f"codex timed out: {exc}",
            "session_key": session_key,
            "session_id": session_id,
            "attempt": attempt,
            "repair_attempt": repair_attempt,
            "started_at": started,
            "ended_at": now_utc(),
            "duration_seconds": round(time.monotonic() - start_monotonic, 3),
            "raw_text": raw_text,
        }
        _append_s1_codex_event(project_root, session_key, _s1_codex_attempt_summary(call), mode=mode)
        return call
    finally:
        output_path.unlink(missing_ok=True)

    parsed_session_id = _parse_s1_codex_session_id(result.stderr, result.stdout) or session_id
    call = {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "reason": None if result.returncode == 0 else (result.stderr[-1200:] or result.stdout[-1200:] or f"codex exited {result.returncode}"),
        "session_key": session_key,
        "session_id": parsed_session_id,
        "previous_session_id": session_id,
        "attempt": attempt,
        "repair_attempt": repair_attempt,
        "started_at": started,
        "ended_at": now_utc(),
        "duration_seconds": round(time.monotonic() - start_monotonic, 3),
        "raw_text": raw_text,
    }
    if result.returncode != 0:
        category = _s1_codex_backend_failure_category(call.get("reason"), result.stderr, result.stdout)
        if category:
            call["failure_category"] = category
            call["retryable"] = category in {"llm_rate_limit_or_quota", "llm_transient_backend"}
    if parsed_session_id:
        _save_s1_codex_session(project_root, session_key, parsed_session_id, config, call)
    if result.returncode == 0:
        payload = _parse_json_object(raw_text)
        if isinstance(payload, dict):
            call["payload"] = payload
        else:
            call["status"] = "invalid_json"
            call["reason"] = "Codex did not return a JSON object"
    _append_s1_codex_event(project_root, session_key, _s1_codex_attempt_summary(call), mode=mode)
    return call


def _s1_codex_json_repair_prompt(validation_errors: list[str], previous_output: str, previous_status: str, *, mode: str = "c2c") -> str:
    schema_name = "c2c_s1_codex_direction_v1" if mode == "c2c" else "generic_s1_codex_direction_v1"
    payload = {
        "schema_version": f"{mode}_s1_codex_json_repair_v1",
        "status": previous_status or "invalid",
        "errors_to_fix": validation_errors,
        "previous_output_tail": previous_output[-4000:],
        "instructions": [
            f"Return only a valid JSON object matching {schema_name}.",
            "Keep the same S1 method-direction task and reuse the context already present in this resume session.",
            "Do not switch to prose or markdown.",
            "If a field is missing, fill it from the artifacts you have already inspected or inspect the listed artifacts again.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _s1_codex_backend_failure_category(*parts: Any) -> str | None:
    text = "\n".join(str(part or "") for part in parts).lower()
    if any(marker in text for marker in ["401 unauthorized", "invalid token", "invalid api key", "incorrect api key", "authentication_error"]):
        return "llm_authentication"
    if any(marker in text for marker in ["403 forbidden", "permission denied", "access denied"]):
        return "llm_permission"
    if any(marker in text for marker in ["429", "too many requests", "rate limit", "rate_limit", "insufficient_quota", "quota exceeded", "billing limit", "payment required"]):
        return "llm_rate_limit_or_quota"
    if any(marker in text for marker in ["temporarily unavailable", "service unavailable", "retry-after", "connection reset", "connection aborted"]):
        return "llm_transient_backend"
    return None


def _s1_codex_backend_blocked_result(
    *,
    session_key: str,
    session_id: str | None,
    used_existing_session: bool,
    repair_count: int,
    attempts: list[dict[str, Any]],
    novelty_audits: list[dict[str, Any]],
    reset_info: dict[str, Any] | None,
    call: dict[str, Any],
    validation_errors: list[str],
) -> dict[str, Any]:
    category = str(call.get("failure_category") or "codex_cli_failure")
    retryable = bool(call.get("retryable"))
    reason = _s1_codex_backend_blocked_reason(category, retryable=retryable)
    return {
        "status": "blocked",
        "reason": reason,
        "failure_category": category,
        "retryable": retryable,
        "session_key": session_key,
        "session_id": session_id,
        "used_existing_session": used_existing_session,
        "repair_count": max(0, repair_count),
        "attempts": attempts,
        "novelty_audits": novelty_audits,
        "session_reset": bool(reset_info),
        "session_reset_reason": (reset_info or {}).get("reason"),
        "validation_errors": validation_errors,
        "last_output_tail": str(call.get("raw_text") or call.get("reason") or "")[-2000:],
    }


def _s1_codex_backend_blocked_reason(category: str, *, retryable: bool) -> str:
    if category == "llm_authentication":
        return (
            "S1 Codex evidence agent failed before JSON generation because the Codex backend rejected authentication "
            "(401 Unauthorized / invalid token). Refresh OPENAI_API_KEY/OPENAI_BASE_URL credentials, then resume."
        )
    if category == "llm_permission":
        return (
            "S1 Codex evidence agent failed before JSON generation because the Codex backend denied access "
            "(403/permission error). Check model, endpoint, and account permissions, then resume."
        )
    if category == "llm_rate_limit_or_quota":
        return (
            "S1 Codex evidence agent failed before JSON generation because the Codex backend hit a quota/rate-limit/billing error. "
            "Wait for quota recovery or update billing credentials, then resume."
        )
    if category == "llm_transient_backend":
        return "S1 Codex evidence agent failed before JSON generation because the Codex backend was transiently unavailable. Retry resume later."
    retry_hint = " Retry resume later." if retryable else ""
    return f"S1 Codex evidence agent failed before JSON generation due to a Codex CLI/backend failure.{retry_hint}"


def _s1_revision_feedback_prompt(audit: dict[str, Any], previous_payload: dict[str, Any], *, mode: str = "c2c") -> str:
    schema_name = "c2c_s1_codex_direction_v1" if mode == "c2c" else "generic_s1_codex_direction_v1"
    return "\n\n".join(
        [
            "Your previous S1 direction was rejected by the independent novelty auditor.",
            "Generate a revised S1 direction that stays method-level but is less similar to prior ideas/memories.",
            "Do not edit files. Return only valid JSON matching the original S1 schema.",
            "You must explicitly address novelty_audit.revision_guidance and avoid repeated similarity sources.",
            "Do not solve this by renaming the same mechanism; change the core mechanism hypothesis or allowed variant family.",
            "Novelty audit JSON:",
            json.dumps(audit, ensure_ascii=False, indent=2),
            "Previous rejected S1 payload:",
            json.dumps(previous_payload, ensure_ascii=False, indent=2)[:12000],
            "Required schema:",
            schema_name,
        ]
    )


def _s1_novelty_auditor_prompt(
    *,
    project_root: Path,
    config: dict[str, Any],
    payload: dict[str, Any],
    shared_memory: dict[str, Any],
    threshold: float,
    mode: str,
) -> str:
    ideas = payload.get("selected_ideas") if isinstance(payload.get("selected_ideas"), list) else []
    direction = payload.get("direction_decision") if isinstance(payload.get("direction_decision"), dict) else {}
    local_history = {
        "negative_memory": _safe_read_json_artifact(project_root / ("intake/c2c/negative_result_memory.json" if mode == "c2c" else "meta/negative_memory.jsonl")),
        "performance_feedback": _safe_read_json_artifact(project_root / "plan/performance_feedback.json"),
        "direction_scorecard": _safe_read_json_artifact(project_root / "plan/direction_scorecard.json"),
        "s2_planner_memory": _safe_read_json_artifact(project_root / "plan/s2_planner_memory.json"),
    }
    context = {
        "schema_version": f"{mode}_s1_novelty_audit_context_v1",
        "mode": mode,
        "threshold": threshold,
        "new_direction": direction,
        "new_selected_ideas": ideas[:1],
        "shared_method_failure_memory": _compact_json_value(shared_memory, max_chars=9000),
        "local_history": _compact_json_value(local_history, max_chars=9000),
    }
    output_contract = {
        "schema_version": f"{mode}_s1_novelty_audit_v1",
        "status": "ok",
        "novelty_score": "float 0.0-1.0 where higher means less similar to old ideas/memories",
        "max_similarity_score": "float 0.0-1.0 where higher means more similar to one old idea/memory",
        "passed": "true if novelty_score >= threshold and no high-risk near-duplicate exists",
        "threshold": threshold,
        "most_similar_sources": [
            {
                "source_type": "shared_memory|local_history|direction_scorecard|s2_planner_memory",
                "source_id": "memory_id/path/id",
                "similarity_score": "float 0.0-1.0",
                "overlap": ["mechanism", "dataset", "integration point", "failure mode"],
                "why_similar": "short factual reason",
            }
        ],
        "distinctive_elements": ["what is genuinely different"],
        "repeated_patterns": ["old pattern repeated by new idea"],
        "revision_guidance": ["concrete method-level changes S1 should make if rejected"],
        "decision": "pass|revise",
    }
    return "\n\n".join(
        [
            "You are an independent S1 novelty auditor.",
            "Your job is to compare the new S1 idea/direction against shared memory and local historical failures.",
            "Score semantic and mechanism similarity, not just string overlap.",
            "Reject if the direction is essentially a renamed old mechanism, repeats a failed mechanism/dataset/failure-mode combination, or differs only by local tuning.",
            "Pass if the core mechanism is meaningfully different enough for S2 to explore.",
            "Do not edit files. Return only one valid JSON object.",
            "Context JSON:",
            json.dumps(context, ensure_ascii=False, indent=2),
            "Required JSON shape:",
            json.dumps(output_contract, ensure_ascii=False, indent=2),
        ]
    )


def _run_s1_novelty_auditor(
    *,
    project_root: Path,
    config: dict[str, Any],
    payload: dict[str, Any],
    shared_memory: dict[str, Any],
    mode: str = "c2c",
) -> dict[str, Any]:
    cfg = _s1_novelty_auditor_config(config, mode=mode)
    threshold = float(cfg.get("threshold") or 0.58)
    if not cfg.get("enabled"):
        return {"status": "skipped", "enabled": False, "passed": True, "threshold": threshold, "reason": "disabled"}
    if not shutil.which("codex"):
        return {"status": "skipped", "enabled": True, "passed": True, "threshold": threshold, "reason": "codex executable not found"}
    session_key = str(cfg.get("session_key"))
    prompt = _s1_novelty_auditor_prompt(project_root=project_root, config=config, payload=payload, shared_memory=shared_memory, threshold=threshold, mode=mode)
    call = _run_s1_codex_cli_once(
        project_root=project_root,
        config=config,
        session_key=session_key,
        session_id=_session_id_from_record(_load_s1_codex_session_record(project_root, session_key)) if cfg.get("resume_enabled", True) else None,
        prompt=prompt,
        timeout_seconds=int(cfg.get("timeout_seconds") or 900),
        repair_attempt=False,
        attempt=1,
        mode=mode,
        task_prefix="Follow this task exactly. You are auditing S1 novelty only. Do not edit files. Return only valid JSON.",
        agent_config=cfg,
    )
    audit_payload = call.get("payload") if isinstance(call.get("payload"), dict) else {}
    errors = _validate_s1_novelty_audit_payload(audit_payload)
    if call.get("status") != "ok" or errors:
        return {
            "status": "invalid" if errors else str(call.get("status") or "failed"),
            "enabled": True,
            "passed": True,
            "threshold": threshold,
            "reason": "; ".join(errors) if errors else call.get("reason"),
            "call": _s1_codex_attempt_summary(call),
        }
    score = _float_or_default(audit_payload.get("novelty_score"), 0.0)
    passed = bool(audit_payload.get("passed")) and score >= threshold
    return {
        "status": "ok",
        "enabled": True,
        "passed": passed,
        "threshold": threshold,
        "audit": audit_payload,
        "call": _s1_codex_attempt_summary(call),
    }


def _validate_s1_novelty_audit_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict) or not payload:
        return ["audit output must be a non-empty object"]
    if str(payload.get("status") or "ok").lower() not in {"ok", "pass", "revise"}:
        errors.append("status must be ok/pass/revise")
    if not isinstance(payload.get("novelty_score"), (int, float)):
        errors.append("novelty_score must be numeric")
    if not isinstance(payload.get("max_similarity_score"), (int, float)):
        errors.append("max_similarity_score must be numeric")
    if not isinstance(payload.get("passed"), bool):
        errors.append("passed must be boolean")
    if not isinstance(payload.get("most_similar_sources"), list):
        errors.append("most_similar_sources must be a list")
    if not isinstance(payload.get("revision_guidance"), list):
        errors.append("revision_guidance must be a list")
    return errors


def _float_or_default(value: Any, default: float) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_read_json_artifact(path: Path) -> Any:
    try:
        return read_json(path, default={}) or {}
    except Exception:
        return {}


def _validate_s1_codex_payload(payload: dict[str, Any], *, mode: str = "c2c") -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["top-level output must be a JSON object"]
    if str(payload.get("status") or "ok").lower() not in {"ok", "direction_selected"}:
        errors.append("status must be ok or direction_selected")
    if not isinstance(payload.get("direction_decision"), dict) or not payload.get("direction_decision"):
        errors.append("direction_decision must be a non-empty object")
    else:
        decision = payload["direction_decision"]
        direction_fields = ["direction_id", "core_hypothesis", "allowed_variants", "forbidden_patterns", "failure_focus"]
        direction_fields.append("mechanism_direction" if mode == "c2c" else "title")
        for field in direction_fields:
            if decision.get(field) in (None, "", []):
                errors.append(f"direction_decision missing {field}")
    if not isinstance(payload.get("negative_constraints"), dict):
        errors.append("negative_constraints must be an object")
    if not isinstance(payload.get("evidence_requests"), list):
        errors.append("evidence_requests must be a list")
    evidence_bundle = payload.get("evidence_bundle")
    if not isinstance(evidence_bundle, dict):
        errors.append("evidence_bundle must be an object")
    elif not isinstance(evidence_bundle.get("items"), list) or not evidence_bundle.get("items"):
        errors.append("evidence_bundle.items must be a non-empty list")
    ideas = payload.get("selected_ideas")
    if not isinstance(ideas, list):
        errors.append("selected_ideas must be a list")
        return errors
    if len(ideas) != 1:
        errors.append("S1 selected_ideas must contain exactly one high-level direction card; S2 generates concrete variants")
    selected_count = sum(1 for idea in ideas if isinstance(idea, dict) and idea.get("selected") is True)
    if ideas and selected_count != 1:
        errors.append("the optional S1 direction card must have exactly one selected=true")
    direction_ids = set()
    for idx, idea in enumerate(ideas[:1], start=1):
        if not isinstance(idea, dict):
            errors.append(f"selected_ideas[{idx}] must be an object")
            continue
        for field in ["id", "title", "hypothesis", "novelty_score", "feasibility_score"]:
            if idea.get(field) in (None, "", []):
                errors.append(f"selected_ideas[{idx}] missing {field}")
        ref_fields = ["evidence_refs", "counterevidence_refs"]
        if mode == "c2c":
            ref_fields = ["expected_files", "verification_commands", "evidence_refs", "counterevidence_refs", "code_refs"]
        for field in ref_fields:
            if not isinstance(idea.get(field), list) or not idea.get(field):
                errors.append(f"selected_ideas[{idx}].{field} must be a non-empty list")
        if mode == "c2c" and not idea.get("reviewer_risk_response"):
            errors.append(f"selected_ideas[{idx}] missing reviewer_risk_response")
        if mode == "c2c" and not idea.get("mechanism_type"):
            errors.append(f"selected_ideas[{idx}] missing mechanism_type")
        direction_id = idea.get("s1_direction_id") or idea.get("direction_id") or (payload.get("direction_decision") or {}).get("direction_id")
        if direction_id:
            direction_ids.add(str(direction_id))
    if len(direction_ids) > 1:
        errors.append("selected_ideas must stay within one S1 direction_id")
    return errors


def _s1_codex_direction_cards(payload: dict[str, Any], *, used_shared_memory_refs: list[str] | None = None) -> list[dict[str, Any]]:
    raw = payload.get("selected_ideas")
    if not isinstance(raw, list):
        return []
    direction = payload.get("direction_decision") if isinstance(payload.get("direction_decision"), dict) else {}
    direction_id = str(direction.get("direction_id") or "")
    cards = []
    for item in raw[:1]:
        if not isinstance(item, dict):
            continue
        card = dict(item)
        if direction_id:
            card["s1_direction_id"] = str(card.get("s1_direction_id") or card.get("direction_id") or direction_id)
            card["direction_id"] = str(card.get("direction_id") or direction_id)
        card["selected"] = True
        card["used_shared_memory_refs"] = list(used_shared_memory_refs if used_shared_memory_refs is not None else card.get("used_shared_memory_refs") or [])
        card["s1_evidence_agent"] = {
            "source": str(payload.get("s1_agent_source") or "codex_resume_evidence_agent"),
            "direction_id": card.get("s1_direction_id") or card.get("direction_id") or direction_id,
            "evidence_bundle_items": len((payload.get("evidence_bundle") or {}).get("items") or []) if isinstance(payload.get("evidence_bundle"), dict) else 0,
            "used_shared_memory_refs": list(used_shared_memory_refs or []),
        }
        cards.append(card)
    return cards


def _generic_s1_codex_ideas(payload: dict[str, Any], *, used_shared_memory_refs: list[str] | None = None) -> list[dict[str, Any]]:
    raw = payload.get("selected_ideas")
    if not isinstance(raw, list):
        return []
    direction = payload.get("direction_decision") if isinstance(payload.get("direction_decision"), dict) else {}
    direction_id = str(direction.get("direction_id") or "")
    ideas = []
    for item in raw[:1]:
        if not isinstance(item, dict):
            continue
        idea = dict(item)
        if direction_id:
            idea.setdefault("direction_id", direction_id)
        idea["selected"] = True
        idea["used_shared_memory_refs"] = list(used_shared_memory_refs if used_shared_memory_refs is not None else idea.get("used_shared_memory_refs") or [])
        idea["s1_evidence_agent"] = {
            "source": "codex_resume_evidence_agent",
            "direction_id": idea.get("direction_id") or direction_id or idea.get("id"),
            "evidence_bundle_items": len((payload.get("evidence_bundle") or {}).get("items") or []) if isinstance(payload.get("evidence_bundle"), dict) else 0,
            "used_shared_memory_refs": list(used_shared_memory_refs or []),
        }
        ideas.append(idea)
    return ideas


def _attach_s1_novelty_audit_to_ideas(ideas: list[dict[str, Any]], audits: list[dict[str, Any]] | None) -> None:
    if not ideas or not audits:
        return
    latest = next((item for item in reversed(audits) if isinstance(item, dict)), {})
    audit = latest.get("audit") if isinstance(latest.get("audit"), dict) else {}
    summary = {
        "status": latest.get("status"),
        "passed": latest.get("passed"),
        "threshold": latest.get("threshold"),
        "novelty_score": audit.get("novelty_score"),
        "max_similarity_score": audit.get("max_similarity_score"),
        "most_similar_sources": (audit.get("most_similar_sources") or [])[:5] if isinstance(audit.get("most_similar_sources"), list) else [],
    }
    for idea in ideas:
        if isinstance(idea, dict):
            idea["s1_novelty_audit"] = {key: value for key, value in summary.items() if value not in (None, "", [], {})}


def _s1_codex_payload_refs(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    items = []
    bundle = payload.get("evidence_bundle") if isinstance(payload.get("evidence_bundle"), dict) else {}
    for entry in bundle.get("items") or []:
        if isinstance(entry, dict):
            items.append(entry)
    evidence_refs: list[dict[str, Any]] = []
    counter_refs: list[dict[str, Any]] = []
    code_refs: list[dict[str, Any]] = []
    for entry in items:
        ref = {
            "source_type": entry.get("source_type") or "artifact",
            "source_label": entry.get("chunk_id") or entry.get("source_path") or "evidence_bundle",
            "source_path": entry.get("source_path"),
            "claim": entry.get("summary") or entry.get("claim") or "",
        }
        if ref["source_type"] == "code":
            code_refs.append(ref)
        elif entry.get("risks"):
            counter_refs.append(ref)
        else:
            evidence_refs.append(ref)
    if not evidence_refs and items:
        entry = items[0]
        evidence_refs.append(
            {
                "source_type": entry.get("source_type") or "artifact",
                "source_label": entry.get("chunk_id") or entry.get("source_path") or "evidence_bundle",
                "source_path": entry.get("source_path"),
                "claim": entry.get("summary") or "",
            }
        )
    return evidence_refs, counter_refs, code_refs


def _evidence_bundle_from_selected_ideas(ideas: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next((idea for idea in ideas if isinstance(idea, dict) and idea.get("selected")), ideas[0] if ideas else {})
    items: list[dict[str, Any]] = []
    for ref in selected.get("evidence_refs") or []:
        if isinstance(ref, dict):
            items.append(
                {
                    "source_path": ref.get("source_path") or ref.get("source_label") or "literature/direction.json",
                    "source_type": ref.get("source_type") or "artifact",
                    "summary": ref.get("claim") or ref.get("summary") or "Evidence supporting the selected direction.",
                    "supports": [selected.get("id") or selected.get("direction_id") or "selected_direction"],
                    "risks": [],
                }
            )
    for ref in selected.get("counterevidence_refs") or []:
        if isinstance(ref, dict):
            items.append(
                {
                    "source_path": ref.get("source_path") or ref.get("source_label") or "literature/direction.json",
                    "source_type": ref.get("source_type") or "artifact",
                    "summary": ref.get("claim") or ref.get("summary") or "Counterevidence or risk for the selected direction.",
                    "supports": [],
                    "risks": [ref.get("claim") or ref.get("summary") or "risk"],
                }
            )
    if not items:
        items.append(
            {
                "source_path": "literature/direction.json",
                "source_type": "artifact",
                "summary": "Selected S1 direction compatibility artifact.",
                "supports": [selected.get("id") or "selected_direction"],
                "risks": [],
            }
        )
    return {"schema_version": "auto_research_evidence_bundle_v1", "items": items}


def _s1_codex_decision_chain(payload: dict[str, Any], ideas: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next((idea for idea in ideas if idea.get("selected")), ideas[0] if ideas else {})
    return {
        "evidence": [str(ref.get("source_label") or ref.get("claim")) for ref in selected.get("evidence_refs") or [] if isinstance(ref, dict)][:5],
        "counterevidence": [str(ref.get("source_label") or ref.get("claim")) for ref in selected.get("counterevidence_refs") or [] if isinstance(ref, dict)][:5],
        "conclusion": (payload.get("direction_decision") or {}).get("rationale") if isinstance(payload.get("direction_decision"), dict) else selected.get("hypothesis"),
    }


def _summarize_chunk_index_for_prompt(chunk_index: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(chunk_index, dict):
        return {}
    entries = chunk_index.get("entries") if isinstance(chunk_index.get("entries"), list) else []
    compact_entries = []
    for entry in entries[:180]:
        if not isinstance(entry, dict):
            continue
        compact_entries.append(
            {
                "chunk_id": entry.get("chunk_id"),
                "source_type": entry.get("source_type"),
                "source_path": entry.get("source_path"),
                "section": entry.get("section"),
                "keywords": entry.get("keywords"),
                "semantic_summary": entry.get("semantic_summary"),
                "mechanism_tags": entry.get("mechanism_tags"),
                "failure_modes": entry.get("failure_modes"),
                "retrieval_keywords": entry.get("retrieval_keywords"),
            }
        )
    return {
        "counts": chunk_index.get("counts") or {},
        "entries": compact_entries,
        "truncated_entries": max(0, len(entries) - len(compact_entries)),
    }


def _compact_json_value(value: Any, *, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return value
    return {"truncated": True, "max_chars": max_chars, "json_prefix": text[:max_chars]}


def _load_s1_codex_session_record(project_root: Path, session_key: str) -> dict[str, Any]:
    payload = read_yaml(project_root / "meta" / "codex_sessions.yaml", default={"sessions": {}}) or {"sessions": {}}
    session = (payload.get("sessions") or {}).get(session_key) if isinstance(payload, dict) else None
    return session if isinstance(session, dict) else {}


def _session_id_from_record(session: dict[str, Any]) -> str | None:
    if isinstance(session, dict) and session.get("session_id"):
        return str(session["session_id"])
    return None


def _maybe_reset_s1_codex_session_before_run(project_root: Path, session_key: str, cfg: dict[str, Any], *, mode: str) -> dict[str, Any] | None:
    if not cfg.get("resume_enabled", True):
        return None
    reason = ""
    if mode == "c2c" and _s1_direction_budget_exhausted(project_root, cfg):
        reason = "same_direction_budget_exhausted"
    if not reason:
        return None
    return _reset_s1_codex_session(project_root, session_key, reason=reason, phase="before_run")


def _s1_direction_budget_exhausted(project_root: Path, cfg: dict[str, Any]) -> bool:
    scorecard = read_json(project_root / "plan" / "direction_scorecard.json", default={}) or {}
    current = scorecard.get("current_direction") if isinstance(scorecard, dict) else {}
    summary = current.get("summary") if isinstance(current, dict) else {}
    feedback = current.get("s1_feedback") if isinstance(current, dict) else {}
    if isinstance(feedback, dict) and feedback.get("recommendation") == "return_to_s1_new_direction":
        return True
    try:
        count = int(summary.get("same_direction_failure_count") or 0)
        budget = int(summary.get("same_direction_failure_budget") or cfg.get("same_direction_failure_reset_threshold") or 5)
    except (TypeError, ValueError):
        return False
    return bool(budget and count >= budget)


def _record_s1_codex_session_health(
    project_root: Path,
    session_key: str,
    session_id: str | None,
    cfg: dict[str, Any],
    payload: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any] | None:
    if not session_id:
        return None
    direction_id = _s1_payload_direction_id(payload)
    if not direction_id:
        return None
    path = project_root / "meta" / "codex_sessions.yaml"
    sessions_payload = read_yaml(path, default={"sessions": {}}) or {"sessions": {}}
    sessions_payload.setdefault("sessions", {})
    record = sessions_payload["sessions"].get(session_key) if isinstance(sessions_payload["sessions"].get(session_key), dict) else {}
    health = record.get("health") if isinstance(record.get("health"), dict) else {}
    last_direction_id = health.get("last_direction_id")
    duplicate_streak = int(health.get("duplicate_direction_streak") or 0)
    duplicate_streak = duplicate_streak + 1 if last_direction_id == direction_id else 0
    health.update(
        {
            "last_direction_id": direction_id,
            "duplicate_direction_streak": duplicate_streak,
            "updated_at": now_utc(),
            "mode": mode,
        }
    )
    record["health"] = health
    sessions_payload["sessions"][session_key] = record
    write_yaml(path, sessions_payload)
    threshold = max(1, int(cfg.get("duplicate_direction_reset_threshold") or 2))
    if duplicate_streak >= threshold:
        return _reset_s1_codex_session(project_root, session_key, reason="duplicate_direction_streak", phase="after_run", extra={"direction_id": direction_id, "duplicate_direction_streak": duplicate_streak})
    return None


def _s1_payload_direction_id(payload: dict[str, Any]) -> str:
    decision = payload.get("direction_decision") if isinstance(payload.get("direction_decision"), dict) else {}
    direction_id = decision.get("direction_id")
    if direction_id:
        return str(direction_id)
    for item in payload.get("selected_ideas") or []:
        if isinstance(item, dict) and (item.get("s1_direction_id") or item.get("direction_id") or item.get("id")):
            return str(item.get("s1_direction_id") or item.get("direction_id") or item.get("id"))
    return ""


def _reset_s1_codex_session(project_root: Path, session_key: str, *, reason: str, phase: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    path = project_root / "meta" / "codex_sessions.yaml"
    payload = read_yaml(path, default={"sessions": {}}) or {"sessions": {}}
    payload.setdefault("sessions", {})
    previous = payload["sessions"].pop(session_key, None)
    event = {
        "event": "session_reset",
        "session_key": session_key,
        "reason": reason,
        "phase": phase,
        "previous_session_id": previous.get("session_id") if isinstance(previous, dict) else None,
        "timestamp": now_utc(),
        **(extra or {}),
    }
    payload.setdefault("session_reset_history", [])
    payload["session_reset_history"].append(event)
    payload["session_reset_history"] = payload["session_reset_history"][-50:]
    write_yaml(path, payload)
    _append_s1_codex_event(project_root, session_key, event)
    return event


def _save_s1_codex_session(project_root: Path, session_key: str, session_id: str, config: dict[str, Any], call: dict[str, Any]) -> None:
    path = project_root / "meta" / "codex_sessions.yaml"
    payload = read_yaml(path, default={"sessions": {}}) or {"sessions": {}}
    payload.setdefault("sessions", {})
    previous = payload["sessions"].get(session_key) if isinstance(payload["sessions"].get(session_key), dict) else {}
    payload["sessions"][session_key] = {
        "session_id": session_id,
        "provider": "codex_cli",
        "model": (config.get("llm") or {}).get("model"),
        "updated_at": now_utc(),
        "purpose": "s1_c2c_evidence_direction",
        "created_at": previous.get("created_at") or now_utc(),
        "health": previous.get("health") if isinstance(previous.get("health"), dict) else {},
        "last_call": _s1_codex_attempt_summary(call),
    }
    write_yaml(path, payload)


def _append_s1_codex_event(project_root: Path, session_key: str, event: dict[str, Any], *, mode: str = "c2c") -> None:
    path = project_root / "literature" / ("c2c/s1_codex_events.jsonl" if mode == "c2c" else "s1_codex_events.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"session_key": session_key, **event}, ensure_ascii=False) + "\n")


def _s1_codex_attempt_summary(call: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "status": call.get("status"),
        "returncode": call.get("returncode"),
        "reason": call.get("reason"),
        "failure_category": call.get("failure_category"),
        "retryable": call.get("retryable"),
        "session_id": call.get("session_id"),
        "previous_session_id": call.get("previous_session_id"),
        "attempt": call.get("attempt"),
        "repair_attempt": call.get("repair_attempt"),
        "started_at": call.get("started_at"),
        "ended_at": call.get("ended_at"),
        "duration_seconds": call.get("duration_seconds"),
    }
    raw_text = str(call.get("raw_text") or "")
    if raw_text:
        summary["raw_text_tail"] = raw_text[-1000:]
    if isinstance(call.get("payload"), dict):
        summary["payload_keys"] = sorted(str(key) for key in call["payload"].keys())
    return {key: value for key, value in summary.items() if value is not None}


def _parse_s1_codex_session_id(stderr: str, stdout: str = "") -> str | None:
    match = re.search(r"session id:\s*([0-9a-fA-F-]+)", stderr or "")
    if match:
        return match.group(1)
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started" and event.get("thread_id"):
            return str(event["thread_id"])
    return None


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)```", text or "", flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1).strip())
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None
    start = (text or "").find("{")
    end = (text or "").rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None
    return None
