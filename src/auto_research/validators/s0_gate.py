"""S0 static project intake gate."""

from __future__ import annotations

from .base import StageGateValidator


class S0GateValidator(StageGateValidator):
    stage_key = "S0_intake"
    validator_name = "s0_intake_gate_v1"

    def validate(self):
        if not self.config.get("c2c", {}).get("enabled"):
            self.pass_check("generic_s0_intake", message="Generic projects do not require static C2C intake artifacts.")
            return self.finalize(default_reason="S0 static intake passed")

        bundle_path = self.require_file("intake/c2c/static_bundle.json", check_name="c2c_static_bundle_exists")
        if not bundle_path:
            return self.finalize(default_reason="S0 static intake passed")

        bundle = self.read_json_artifact("intake/c2c/static_bundle.json")
        required_keys = [
            "metadata",
            "reference_result",
            "paper_full_manifest",
            "repo_manifest",
            "historical_results",
            "baseline",
            "repo_card",
            "paper_cards",
            "paper_chunks",
            "bibliography_cards",
            "rebuttal_matrix",
            "rebuttal_chunks",
            "code_cards",
            "code_file_manifest",
            "code_symbols",
            "code_chunks",
            "code_edges",
            "code_repo_map",
            "code_intake_report",
            "implementation_surface_map",
            "code_retrieval_index",
            "cache_summary",
            "chunk_index",
            "result_ledger_csv",
            "negative_memory",
            "retrieval_plan",
            "followup_bundle",
            "evidence_brief",
        ]
        missing = [key for key in required_keys if not isinstance(bundle, dict) or key not in bundle]
        if missing:
            self.retry_check(
                "c2c_static_bundle_schema",
                "S0 C2C static bundle missing required keys",
                artifact="intake/c2c/static_bundle.json",
                details={"missing": missing},
            )
        else:
            self.pass_check("c2c_static_bundle_schema", artifact="intake/c2c/static_bundle.json")

        for rel_path in [
            "intake/c2c/repo_manifest.json",
            "intake/c2c/repo_card.json",
            "intake/c2c/baseline_evidence.json",
            "intake/c2c/paper_full_manifest.json",
            "intake/c2c/paper_cards.json",
            "intake/c2c/paper_chunks.jsonl",
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
            "intake/c2c/evidence_brief.json",
        ]:
            self.require_file(rel_path, check_name=f"{rel_path}_exists")

        chunk_index = bundle.get("chunk_index") if isinstance(bundle, dict) else None
        if isinstance(chunk_index, dict):
            self._validate_chunk_index(chunk_index)
        if isinstance(bundle, dict):
            self._validate_paper_full_manifest(bundle)
            self._validate_code_intake(bundle)

        return self.finalize(default_reason="S0 static intake passed")

    def _validate_paper_full_manifest(self, bundle: dict) -> None:
        cards = ((bundle.get("reference_result") or {}).get("cards") or [])
        pdf_cards = [
            card
            for card in cards
            if isinstance(card, dict) and card.get("kind") == "ref_paper" and str(card.get("local_path") or "").lower().endswith(".pdf")
        ]
        if not pdf_cards:
            self.pass_check("c2c_paper_full_manifest_optional", message="No PDF ref_paper inputs detected.")
            return
        manifest = bundle.get("paper_full_manifest") or []
        if not isinstance(manifest, list) or not manifest:
            self.retry_check(
                "c2c_paper_full_manifest_nonempty",
                "PDF ref_paper inputs require MinerU paper_full_manifest entries",
                artifact="intake/c2c/paper_full_manifest.json",
            )
            return
        missing = []
        for item in manifest:
            if not isinstance(item, dict):
                continue
            rel_path = item.get("paper_full_md_path")
            if not rel_path or not (self.project_root / str(rel_path)).exists():
                missing.append({"paper_id": item.get("paper_id"), "paper_full_md_path": rel_path})
                continue
            text = (self.project_root / str(rel_path)).read_text(encoding="utf-8", errors="ignore")
            if "#" not in text:
                missing.append({"paper_id": item.get("paper_id"), "paper_full_md_path": rel_path, "reason": "no markdown heading"})
        if missing:
            self.retry_check(
                "c2c_paper_full_manifest_files",
                "MinerU paper_full.md outputs are missing or malformed",
                artifact="intake/c2c/paper_full_manifest.json",
                details={"missing": missing[:5]},
            )
        else:
            self.pass_check("c2c_paper_full_manifest_files", artifact="intake/c2c/paper_full_manifest.json", details={"count": len(manifest)})

    def _validate_code_intake(self, bundle: dict) -> None:
        file_manifest = bundle.get("code_file_manifest") or {}
        symbols = bundle.get("code_symbols") or []
        chunks = bundle.get("code_chunks") or []
        repo_map = bundle.get("code_repo_map") or {}
        report = bundle.get("code_intake_report") or {}
        surface_map = bundle.get("implementation_surface_map") or {}
        retrieval_index = bundle.get("code_retrieval_index") or {}
        if not isinstance(file_manifest, dict) or not file_manifest.get("files"):
            self.retry_check("c2c_code_file_manifest_nonempty", "S0 code_file_manifest has no files", artifact="intake/c2c/code_file_manifest.json")
        else:
            self.pass_check("c2c_code_file_manifest_nonempty", artifact="intake/c2c/code_file_manifest.json", details={"file_count": len(file_manifest.get("files", []))})
        if not isinstance(symbols, list) or not symbols:
            self.retry_check("c2c_code_symbols_nonempty", "S0 tree-sitter code_symbols is empty", artifact="intake/c2c/code_symbols.jsonl")
        else:
            self.pass_check("c2c_code_symbols_nonempty", artifact="intake/c2c/code_symbols.jsonl", details={"symbol_count": len(symbols)})
        if not isinstance(chunks, list) or not chunks:
            self.retry_check("c2c_code_chunks_nonempty", "S0 code_chunks is empty", artifact="intake/c2c/code_chunks.jsonl")
            return
        required = ["chunk_id", "path", "node_type", "symbol", "start_line", "end_line", "edit_surface", "text"]
        bad = []
        for chunk in chunks[:20]:
            if not isinstance(chunk, dict):
                bad.append({"chunk": chunk, "missing": required})
                continue
            missing = [field for field in required if chunk.get(field) in (None, "", [])]
            if missing:
                bad.append({"chunk_id": chunk.get("chunk_id"), "missing": missing})
        if bad:
            self.retry_check("c2c_code_chunk_schema", "S0 code_chunks missing tree-sitter retrieval fields", artifact="intake/c2c/code_chunks.jsonl", details={"bad": bad[:5]})
        else:
            self.pass_check("c2c_code_chunk_schema", artifact="intake/c2c/code_chunks.jsonl", details={"chunk_count": len(chunks)})
        counts = repo_map.get("counts") if isinstance(repo_map, dict) else {}
        if not counts or int(counts.get("chunks") or 0) <= 0:
            self.retry_check("c2c_code_repo_map_counts", "S0 code_repo_map has no chunk count", artifact="intake/c2c/code_repo_map.json")
        else:
            self.pass_check("c2c_code_repo_map_counts", artifact="intake/c2c/code_repo_map.json", details={"counts": counts})
        report_counts = report.get("counts") if isinstance(report, dict) else {}
        if not report_counts or int(report_counts.get("chunks") or 0) <= 0:
            self.retry_check("c2c_code_intake_report_counts", "S0 code_intake_report has no chunk diagnostics", artifact="intake/c2c/code_intake_report.json")
        else:
            self.pass_check("c2c_code_intake_report_counts", artifact="intake/c2c/code_intake_report.json", details={"counts": report_counts})
        surfaces = surface_map.get("surfaces") if isinstance(surface_map, dict) else {}
        if not isinstance(surfaces, dict) or not surfaces:
            self.retry_check("c2c_implementation_surface_map_nonempty", "S0 implementation_surface_map is empty", artifact="intake/c2c/implementation_surface_map.json")
        else:
            self.pass_check("c2c_implementation_surface_map_nonempty", artifact="intake/c2c/implementation_surface_map.json", details={"surface_count": len(surfaces)})
        default_queries = retrieval_index.get("default_queries") if isinstance(retrieval_index, dict) else []
        if not isinstance(default_queries, list) or not default_queries:
            self.retry_check("c2c_code_retrieval_index_nonempty", "S0 code_retrieval_index has no default queries", artifact="intake/c2c/code_retrieval_index.json")
        else:
            self.pass_check("c2c_code_retrieval_index_nonempty", artifact="intake/c2c/code_retrieval_index.json", details={"query_count": len(default_queries)})

    def _validate_chunk_index(self, chunk_index: dict) -> None:
        counts = chunk_index.get("counts") or {}
        entries = chunk_index.get("entries") or []
        missing_sources = [source for source in ["paper", "rebuttal", "code"] if int(counts.get(source) or 0) <= 0]
        if not isinstance(entries, list) or not entries:
            self.retry_check("c2c_chunk_index_nonempty", "S0 chunk_index has no entries", artifact="intake/c2c/chunk_index.json")
            return
        if missing_sources:
            self.retry_check(
                "c2c_chunk_index_source_coverage",
                "S0 chunk_index is missing required source types",
                artifact="intake/c2c/chunk_index.json",
                details={"missing_sources": missing_sources, "counts": counts},
            )
        else:
            self.pass_check("c2c_chunk_index_source_coverage", artifact="intake/c2c/chunk_index.json", details={"counts": counts})
        required_fields = ["chunk_id", "source_type", "source_path", "section", "keywords", "text_preview"]
        bad_entries = []
        for entry in entries[:20]:
            if not isinstance(entry, dict):
                bad_entries.append({"entry": entry, "missing": required_fields})
                continue
            missing = [field for field in required_fields if entry.get(field) in (None, "", [])]
            if missing:
                bad_entries.append({"chunk_id": entry.get("chunk_id"), "missing": missing})
        if bad_entries:
            self.retry_check(
                "c2c_chunk_index_entry_schema",
                "S0 chunk_index entries are missing retrieval fields",
                artifact="intake/c2c/chunk_index.json",
                details={"bad_entries": bad_entries[:5]},
            )
        else:
            self.pass_check("c2c_chunk_index_entry_schema", artifact="intake/c2c/chunk_index.json")
