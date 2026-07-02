"""Optional DeepSeek semantic enrichment for S0 chunks."""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from .artifacts import ArtifactManager
from .shared_cache import shared_cache_root
from .utils import ensure_dir, now_utc, read_json, write_json


S0_SEMANTIC_ENRICHMENT_SCHEMA_VERSION = "s0_semantic_enrichment_sample_v1"
S0_SEMANTIC_ENRICHMENT_PROMPT_VERSION = "deepseek_s0_semantic_enrichment_v1"
S0_CODE_SEMANTIC_ENRICHMENT_PROMPT_VERSION = "deepseek_s0_code_semantic_enrichment_v2"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEEPSEEK_V4_FLASH_PRICING_USD_PER_MTOK = {
    "input_cache_hit": 0.0028,
    "input_cache_miss": 0.14,
    "output": 0.28,
}
DEEPSEEK_V4_FLASH_PRICING_CNY_PER_MTOK = {
    "input_cache_hit": 0.02,
    "input_cache_miss": 1.0,
    "output": 2.0,
}


class S0SemanticEnrichmentError(RuntimeError):
    """Raised when S0 semantic enrichment cannot run."""


class DeepSeekS0SemanticEnricher:
    """Enrich S0 chunks without changing deterministic chunk ids or raw text."""

    def __init__(self, project_root: Path, config: dict[str, Any]):
        self.project_root = project_root
        self.config = config
        self.stage_key = "S0_intake"
        self.artifacts = ArtifactManager(project_root)

    def run(
        self,
        *,
        limit: int | str = 8,
        source_types: list[str] | None = None,
        dry_run: bool = False,
        refresh: bool = False,
        workers: int = 1,
    ) -> dict[str, Any]:
        chunks = _load_s0_chunks(self.project_root)
        result = self.enrich_chunk_list(
            chunks,
            limit=limit,
            source_types=source_types,
            dry_run=dry_run,
            refresh=refresh,
            workers=workers,
            write_artifacts=True,
        )
        return {"status": "ok", "artifacts": result["artifacts"], "report": result["report"]}

    def enrich_c2c_chunks(
        self,
        *,
        paper_chunks: list[dict[str, Any]],
        rebuttal_chunks: list[dict[str, Any]],
        code_chunks: list[dict[str, Any]],
        limit: int | str | None = None,
        source_types: list[str] | None = None,
        dry_run: bool | None = None,
        refresh: bool = False,
        workers: int = 1,
        write_artifacts: bool = True,
    ) -> dict[str, Any]:
        chunks: list[dict[str, Any]] = []
        for source_type, items in [("paper", paper_chunks), ("rebuttal", rebuttal_chunks), ("code", code_chunks)]:
            for item in items:
                chunk = dict(item)
                chunk.setdefault("source_type", source_type)
                chunks.append(chunk)
        result = self.enrich_chunk_list(
            chunks,
            limit=limit,
            source_types=source_types,
            dry_run=dry_run,
            refresh=refresh,
            workers=workers,
            write_artifacts=write_artifacts,
        )
        enriched_by_id = {
            ((record.get("chunk") or {}).get("chunk_id")): record
            for record in result["report"].get("records", [])
            if isinstance(record, dict) and (record.get("chunk") or {}).get("chunk_id")
        }
        return {
            **result,
            "paper_chunks": _apply_records_to_chunks(paper_chunks, enriched_by_id, source_type="paper"),
            "rebuttal_chunks": _apply_records_to_chunks(rebuttal_chunks, enriched_by_id, source_type="rebuttal"),
            "code_chunks": _apply_records_to_chunks(code_chunks, enriched_by_id, source_type="code"),
        }

    def enrich_chunk_list(
        self,
        chunks: list[dict[str, Any]],
        *,
        limit: int | str | None = None,
        source_types: list[str] | None = None,
        dry_run: bool | None = None,
        refresh: bool = False,
        workers: int = 1,
        write_artifacts: bool = True,
    ) -> dict[str, Any]:
        cfg = _enrichment_config(self.config)
        model = str(cfg.get("model") or DEFAULT_DEEPSEEK_MODEL)
        base_url = str(cfg.get("base_url") or DEFAULT_DEEPSEEK_BASE_URL)
        max_input_chars = int(cfg.get("max_input_chars") or 6000)
        code_max_input_chars = int(cfg.get("code_max_input_chars") or 3000)
        temperature = float(cfg.get("temperature", 0.1))
        max_tokens = int(cfg.get("max_tokens") or 1200)
        code_max_tokens = int(cfg.get("code_max_tokens") or 800)
        selected_source_types = source_types or list(cfg.get("source_types") or ["paper", "rebuttal", "code"])
        selected_source_types = [str(item) for item in selected_source_types if str(item)]
        selected_limit = _resolve_limit(limit if limit is not None else cfg.get("limit"), total=len(chunks), default=8)
        selected_dry_run = bool(cfg.get("dry_run", False)) if dry_run is None else bool(dry_run)
        selected_workers = max(1, int(workers or cfg.get("workers") or 1))
        if selected_limit <= 0:
            raise S0SemanticEnrichmentError("limit must be positive")

        chunks = [_annotate_enrichment_priority(chunk) for chunk in chunks if str(chunk.get("source_type") or "") in set(selected_source_types)]
        eligible_chunks = [chunk for chunk in chunks if not chunk.get("semantic_enrichment_skip")]
        selected = _select_balanced_chunks(eligible_chunks, limit=selected_limit, source_types=selected_source_types)
        local_cache_root = self.project_root / ".cache" / "auto_research" / "s0_semantic_enrichment" / _cache_safe(model)
        shared_root = shared_cache_root(self.project_root, self.config) / "s0_semantic_enrichment" / _cache_safe(model)
        cache_roots = _dedupe_paths([local_cache_root, shared_root, *_legacy_semantic_cache_roots(self.project_root, model)])
        write_cache_roots = _dedupe_paths([local_cache_root, shared_root])

        records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        tasks = [
            {
                "ordinal": ordinal,
                "chunk": chunk,
                "max_input_chars": code_max_input_chars if str(chunk.get("source_type") or "") == "code" else max_input_chars,
                "max_tokens": code_max_tokens if str(chunk.get("source_type") or "") == "code" else max_tokens,
            }
            for ordinal, chunk in enumerate(selected)
        ]
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not selected_dry_run and not api_key and not _all_enrichment_tasks_cached(tasks, cache_roots=cache_roots, model=model, refresh=refresh):
            raise S0SemanticEnrichmentError("DEEPSEEK_API_KEY is missing and selected chunks are not fully available in the shared S0 semantic cache.")
        client = None if selected_dry_run or not api_key else OpenAI(api_key=api_key, base_url=base_url)
        if selected_dry_run or selected_workers == 1:
            results = [
                _process_enrichment_task(
                    task,
                    cache_roots=cache_roots,
                    write_cache_roots=write_cache_roots,
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    project_id=self.project_root.name,
                    refresh=refresh,
                    dry_run=selected_dry_run,
                    client=client,
                )
                for task in tasks
            ]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=selected_workers) as pool:
                futures = [
                    pool.submit(
                        _process_enrichment_task,
                        task,
                        cache_roots=cache_roots,
                        write_cache_roots=write_cache_roots,
                        model=model,
                        base_url=base_url,
                        temperature=temperature,
                        project_id=self.project_root.name,
                        refresh=refresh,
                        dry_run=False,
                        client=client,
                    )
                    for task in tasks
                ]
                for future in as_completed(futures):
                    results.append(future.result())
        for result in sorted(results, key=lambda item: int(item.get("ordinal") or 0)):
            records.append(result["record"])
            if result.get("failure"):
                failures.append(result["failure"])

        cost_summary = _cost_summary(records, total_available_chunks=len(chunks))
        fallback_count = sum(1 for record in records if record.get("fallback_used"))
        report = {
            "schema_version": S0_SEMANTIC_ENRICHMENT_SCHEMA_VERSION,
            "generated_at": now_utc(),
            "project_id": self.project_root.name,
            "mode": "dry_run" if selected_dry_run else "api",
            "provider": "deepseek",
            "model": model,
            "base_url": base_url,
            "prompt_version": S0_SEMANTIC_ENRICHMENT_PROMPT_VERSION,
            "source_types": selected_source_types,
            "limit": selected_limit,
            "max_input_chars": max_input_chars,
            "code_max_input_chars": code_max_input_chars,
            "max_tokens": max_tokens,
            "code_max_tokens": code_max_tokens,
            "workers": selected_workers,
            "selected_count": len(selected),
            "success_count": len(records),
            "failure_count": len(failures),
            "fallback_count": fallback_count,
            "total_available_chunks": len(chunks),
            "eligible_chunks": len(eligible_chunks),
            "skipped_chunks": len(chunks) - len(eligible_chunks),
            "selection_policy": _selection_policy_summary(),
            "cost_summary": cost_summary,
            "failures": failures,
            "records": records,
        }
        if not write_artifacts:
            return {"status": "ok", "artifacts": [], "report": report}
        json_record = self.artifacts.write_json(
            self.stage_key,
            "c2c/semantic_enrichment_sample.json",
            report,
            artifact_type="c2c_s0_semantic_enrichment_sample",
            summary="Small-batch DeepSeek semantic enrichment sample for S0 chunks",
            source_paths=["intake/c2c/static_bundle.json", "intake/c2c/chunk_index.json"],
            metadata={
                "model": model,
                "limit": selected_limit,
                "success_count": len(records),
                "estimated_usd": cost_summary["actual_sample_cost_usd"],
                "estimated_cny": cost_summary["actual_sample_cost_cny"],
            },
        )
        jsonl_record = self.artifacts.write_text(
            self.stage_key,
            "c2c/semantic_enrichment_sample.jsonl",
            _jsonl(records),
            artifact_type="c2c_s0_semantic_enrichment_sample_jsonl",
            summary="Line-delimited DeepSeek semantic enrichment sample records",
            source_paths=[json_record["path"]],
            metadata={"record_count": len(records)},
        )
        return {"status": "ok", "artifacts": [json_record["path"], jsonl_record["path"]], "report": report}


def _enrichment_config(config: dict[str, Any]) -> dict[str, Any]:
    intake = config.get("intake") if isinstance(config, dict) else {}
    semantic = (intake or {}).get("semantic_enrichment") if isinstance(intake, dict) else {}
    return semantic if isinstance(semantic, dict) else {}


def semantic_enrichment_enabled(config: dict[str, Any]) -> bool:
    cfg = _enrichment_config(config)
    return bool(cfg.get("enabled", False))


def _load_s0_chunks(project_root: Path) -> list[dict[str, Any]]:
    bundle = read_json(project_root / "intake" / "c2c" / "static_bundle.json", default={}) or {}
    if isinstance(bundle, dict) and bundle.get("schema_version") == "c2c_static_intake_bundle_v1":
        chunks: list[dict[str, Any]] = []
        for source_type, key in [("paper", "paper_chunks"), ("rebuttal", "rebuttal_chunks"), ("code", "code_chunks")]:
            for chunk in bundle.get(key) or []:
                if isinstance(chunk, dict):
                    item = dict(chunk)
                    item.setdefault("source_type", source_type)
                    chunks.append(item)
        if chunks:
            return chunks
    chunks = []
    for source_type, rel in [
        ("paper", "intake/c2c/paper_chunks.jsonl"),
        ("rebuttal", "intake/c2c/rebuttal_chunks.jsonl"),
        ("code", "intake/c2c/code_chunks.jsonl"),
    ]:
        chunks.extend(_read_jsonl_chunks(project_root / rel, source_type=source_type))
    if not chunks:
        raise S0SemanticEnrichmentError("No S0 chunks found. Run S0_intake first.")
    return chunks


def _read_jsonl_chunks(path: Path, *, source_type: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            item.setdefault("source_type", source_type)
            chunks.append(item)
    return chunks


def _apply_records_to_chunks(chunks: list[dict[str, Any]], enriched_by_id: dict[str, dict[str, Any]], *, source_type: str) -> list[dict[str, Any]]:
    updated = []
    for chunk in chunks:
        item = dict(chunk)
        chunk_id = str(item.get("chunk_id") or "")
        record = enriched_by_id.get(chunk_id)
        if record:
            enrichment = record.get("enrichment") or {}
            item["semantic_enrichment"] = {
                "schema_version": S0_SEMANTIC_ENRICHMENT_SCHEMA_VERSION,
                "provider": record.get("provider"),
                "model": record.get("model"),
                "prompt_version": record.get("prompt_version"),
                "cache_status": record.get("cache_status"),
                "generated_at": record.get("generated_at"),
                **enrichment,
            }
            item["semantic_summary"] = enrichment.get("semantic_summary") or ""
            item["mechanism_tags"] = enrichment.get("mechanism_tags") or []
            item["failure_modes"] = enrichment.get("failure_modes") or []
            item["retrieval_keywords"] = _dedupe_strings([*(item.get("keywords") or []), *(enrichment.get("retrieval_keywords") or []), *(enrichment.get("mechanism_tags") or [])])
        else:
            item.setdefault("source_type", source_type)
        updated.append(item)
    return updated


def _resolve_limit(value: Any, *, total: int, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "full", "none", "-1"}:
            return max(1, total)
        try:
            return int(text)
        except ValueError as exc:
            raise S0SemanticEnrichmentError(f"Invalid enrichment limit: {value}") from exc
    return int(value)


def _select_balanced_chunks(chunks: list[dict[str, Any]], *, limit: int, source_types: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {source_type: [] for source_type in source_types}
    for chunk in chunks:
        source_type = str(chunk.get("source_type") or "")
        if source_type in grouped:
            grouped[source_type].append(chunk)
    for source_type, items in grouped.items():
        grouped[source_type] = sorted(items, key=lambda item: (-float(item.get("semantic_enrichment_priority") or 0.0), str(item.get("chunk_id") or "")))
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < limit:
        added = False
        for source_type in source_types:
            items = grouped.get(source_type) or []
            if index < len(items):
                selected.append(items[index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        index += 1
    return selected


def _process_enrichment_task(
    task: dict[str, Any],
    *,
    cache_roots: list[Path],
    write_cache_roots: list[Path],
    model: str,
    base_url: str,
    temperature: float,
    project_id: str,
    refresh: bool,
    dry_run: bool,
    client: OpenAI | None,
) -> dict[str, Any]:
    prepared = _prepare_chunk(task["chunk"], max_input_chars=int(task["max_input_chars"]))
    cache_key = _chunk_cache_key(prepared, model=model)
    cached, cache_source = _read_cached_enrichment_record(cache_roots, cache_key, prepared=prepared, model=model, refresh=refresh)
    if cached is not None:
        record = _cached_record_for_prepared(cached, prepared)
        record["cache_status"] = "hit"
        record["cache_source"] = cache_source
        _write_enrichment_record_to_roots(write_cache_roots, cache_key, record)
        return {"ordinal": task["ordinal"], "record": record}
    if dry_run:
        return {"ordinal": task["ordinal"], "record": _dry_run_record(prepared, model=model, base_url=base_url)}
    if client is None:
        raise S0SemanticEnrichmentError("DEEPSEEK_API_KEY is missing and selected chunks are not fully available in the shared S0 semantic cache.")
    try:
        record = _call_deepseek(
            client,
            prepared,
            model=model,
            temperature=temperature,
            max_tokens=int(task["max_tokens"]),
            project_id=project_id,
        )
    except Exception as exc:  # pragma: no cover - exact SDK exceptions vary.
        record = _fallback_record(prepared, model=model, error=str(exc))
        record["cache_status"] = "fallback"
        return {
            "ordinal": task["ordinal"],
            "record": record,
            "failure": {"chunk_id": prepared["chunk_id"], "source_type": prepared["source_type"], "error": str(exc), "fallback_used": True},
        }
    record["cache_status"] = "miss"
    _write_enrichment_record_to_roots(write_cache_roots, cache_key, record)
    return {"ordinal": task["ordinal"], "record": record}


def _all_enrichment_tasks_cached(tasks: list[dict[str, Any]], *, cache_roots: list[Path], model: str, refresh: bool) -> bool:
    if refresh:
        return False
    for task in tasks:
        prepared = _prepare_chunk(task["chunk"], max_input_chars=int(task["max_input_chars"]))
        cache_key = _chunk_cache_key(prepared, model=model)
        cached, _ = _read_cached_enrichment_record(cache_roots, cache_key, prepared=prepared, model=model, refresh=False)
        if cached is None:
            return False
    return True


def _read_cached_enrichment_record(
    cache_roots: list[Path],
    cache_key: str,
    *,
    prepared: dict[str, Any],
    model: str,
    refresh: bool,
) -> tuple[dict[str, Any] | None, str]:
    if refresh:
        return None, ""
    for cache_root in cache_roots:
        cache_path = cache_root / f"{cache_key}.json"
        if not cache_path.exists():
            continue
        cached = read_json(cache_path, default={}) or {}
        if isinstance(cached, dict) and cached.get("schema_version") == S0_SEMANTIC_ENRICHMENT_SCHEMA_VERSION:
            return dict(cached), str(cache_path)
    for cache_root in cache_roots:
        if not cache_root.exists():
            continue
        for cache_path in sorted(cache_root.glob("*.json")):
            cached = read_json(cache_path, default={}) or {}
            if _cached_record_matches_prepared(cached, prepared, model=model):
                return dict(cached), str(cache_path)
    return None, ""


def _cached_record_matches_prepared(record: Any, prepared: dict[str, Any], *, model: str) -> bool:
    if not isinstance(record, dict) or record.get("schema_version") != S0_SEMANTIC_ENRICHMENT_SCHEMA_VERSION:
        return False
    chunk = record.get("chunk") if isinstance(record.get("chunk"), dict) else {}
    return (
        str(record.get("model") or "") == model
        and str(record.get("prompt_version") or "") == _prompt_version_for_chunk(prepared)
        and str(chunk.get("source_type") or "") == str(prepared.get("source_type") or "")
        and str(chunk.get("text_sha256") or "") == str(prepared.get("text_sha256") or "")
    )


def _cached_record_for_prepared(record: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    updated["chunk"] = dict(prepared)
    return updated


def _write_enrichment_record_to_roots(cache_roots: list[Path], cache_key: str, record: dict[str, Any]) -> None:
    for cache_root in cache_roots:
        ensure_dir(cache_root)
        write_json(cache_root / f"{cache_key}.json", record)


def _legacy_semantic_cache_roots(project_root: Path, model: str) -> list[Path]:
    model_dir = _cache_safe(model)
    roots: list[Path] = []
    for candidate in sorted(project_root.parent.glob("*/.cache/auto_research/s0_semantic_enrichment/*")):
        if candidate.parent.parent.parent.parent == project_root:
            continue
        if candidate.name == model_dir and candidate.is_dir():
            roots.append(candidate)
    default_dir = _cache_safe(DEFAULT_DEEPSEEK_MODEL)
    if default_dir != model_dir:
        for candidate in sorted(project_root.parent.glob(f"*/.cache/auto_research/s0_semantic_enrichment/{default_dir}")):
            if candidate.parent.parent.parent.parent != project_root and candidate.is_dir():
                roots.append(candidate)
    return roots


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _annotate_enrichment_priority(chunk: dict[str, Any]) -> dict[str, Any]:
    item = dict(chunk)
    source_type = str(item.get("source_type") or "")
    if source_type == "code":
        priority, reasons, skip_reason = _code_enrichment_priority(item)
    elif source_type == "paper":
        priority, reasons, skip_reason = _paper_enrichment_priority(item)
    elif source_type == "rebuttal":
        priority, reasons, skip_reason = _rebuttal_enrichment_priority(item)
    else:
        priority, reasons, skip_reason = 0.1, ["unknown_source_type"], ""
    item["semantic_enrichment_priority"] = round(priority, 3)
    item["semantic_enrichment_priority_reasons"] = reasons
    if skip_reason:
        item["semantic_enrichment_skip"] = True
        item["semantic_enrichment_skip_reason"] = skip_reason
    return item


def _paper_enrichment_priority(chunk: dict[str, Any]) -> tuple[float, list[str], str]:
    if _is_low_information_text_chunk(chunk):
        return 0.0, ["paper", "low_information_heading"], "low_information_heading"
    text = _priority_text(chunk)
    section = str(chunk.get("section") or "").lower()
    score = 1.0
    reasons = ["paper"]
    if any(term in section for term in ["abstract", "method", "approach", "experiment", "result", "analysis"]):
        score += 2.5
        reasons.append("high_value_section")
    if any(term in text for term in ["cache", "kv", "gate", "projection", "fusion", "alignment", "tokenizer", "latency", "accuracy"]):
        score += 1.5
        reasons.append("c2c_terms")
    if len(str(chunk.get("text") or "")) < 200 and not any(term in section for term in ["abstract", "method"]):
        score -= 1.0
        reasons.append("short_low_context")
    return score, reasons, ""


def _rebuttal_enrichment_priority(chunk: dict[str, Any]) -> tuple[float, list[str], str]:
    if _is_low_information_text_chunk(chunk):
        return 0.0, ["rebuttal", "low_information_heading"], "low_information_heading"
    text = _priority_text(chunk)
    score = 2.0
    reasons = ["rebuttal"]
    if any(term in text for term in ["concern", "reviewer", "weakness", "rebuttal", "limitation", "baseline", "failure", "ablation"]):
        score += 2.0
        reasons.append("review_risk_terms")
    if any(term in text for term in ["cache", "kv", "gate", "projection", "fusion", "latency", "accuracy"]):
        score += 1.2
        reasons.append("c2c_terms")
    return score, reasons, ""


def _code_enrichment_priority(chunk: dict[str, Any]) -> tuple[float, list[str], str]:
    path = str(chunk.get("path") or chunk.get("source_path") or "")
    path_lc = path.lower()
    text = _priority_text(chunk)
    if path_lc.startswith((".github/", "htmlcov/", "wandb/", ".git/")):
        return 0.0, ["excluded_path"], "non_mechanism_infra_path"
    if any(part in f"/{path_lc}/" for part in ["/local/checkpoints/", "/local/snapshots/", "/local/final_results/", "/__pycache__/"]):
        return 0.0, ["excluded_generated_path"], "generated_or_large_local_path"
    score = 0.5
    reasons = ["code"]
    if path_lc.startswith(("rosetta/", "script/", "recipe/", "test/", "tests/")):
        score += 2.0
        reasons.append("c2c_edit_surface_prefix")
    if path_lc.startswith(("rosetta/model/", "script/train/", "script/evaluation/", "recipe/train_recipe/", "recipe/eval_recipe/")):
        score += 2.0
        reasons.append("mechanism_or_experiment_surface")
    tags = set(str(tag) for tag in (chunk.get("risk_tags") or []))
    mechanism_tags = tags.intersection({"alignment_core", "projector_core", "runtime_path", "training_path"})
    if mechanism_tags:
        score += 2.5
        reasons.extend(sorted(mechanism_tags))
    if "evaluation_path" in tags:
        score -= 0.8
        reasons.append("evaluation_path")
    if "test_path" in tags:
        score -= 0.6
        reasons.append("test_path")
    if chunk.get("edit_surface") in {"allowed", "allowed_prefix"}:
        score += 1.0
        reasons.append(f"edit_surface:{chunk.get('edit_surface')}")
    if any(term in text for term in ["cache", "valid_mask", "align", "projector", "wrapper", "gate", "confidence", "loss", "recipe", "ablation"]):
        score += 1.5
        reasons.append("mechanism_terms")
    if score < 1.0:
        return score, reasons, "low_mechanism_relevance"
    return score, reasons, ""


def _priority_text(chunk: dict[str, Any]) -> str:
    values = [
        chunk.get("chunk_id"),
        chunk.get("source_path"),
        chunk.get("path"),
        chunk.get("section"),
        chunk.get("symbol"),
        " ".join(str(item) for item in chunk.get("keywords") or []),
        " ".join(str(item) for item in chunk.get("risk_tags") or []),
        str(chunk.get("text") or "")[:2000],
    ]
    return " ".join(str(value or "") for value in values).lower()


def _is_low_information_text_chunk(chunk: dict[str, Any]) -> bool:
    raw_text = str(chunk.get("text") or "").strip()
    compact = re.sub(r"\s+", " ", raw_text)
    if not compact:
        return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9_\\-]*", compact)
    lower = compact.lower()
    mechanism_hits = sum(1 for term in ["cache", "kv", "gate", "projection", "fusion", "alignment", "tokenizer", "latency", "accuracy", "baseline", "failure", "ablation"] if term in lower)
    section = str(chunk.get("section") or "").lower()
    high_value_section = any(term in section for term in ["abstract", "method", "approach", "experiment", "result", "analysis"])
    if len(compact) < 80 and len(words) <= 10 and mechanism_hits == 0:
        return True
    if len(compact) < 140 and len(words) <= 14 and high_value_section and mechanism_hits == 0:
        return True
    if len(compact) < 180 and re.fullmatch(r"(section\s+)?\d+(\.\d+)*\s*[:.\-]?\s*[A-Za-z0-9_ /-]+", compact, flags=re.I):
        return True
    return False


def _selection_policy_summary() -> dict[str, Any]:
    return {
        "strategy": "balanced_by_source_type_then_priority",
        "code_excluded_prefixes": [".github/", "htmlcov/", "wandb/", ".git/", "local/checkpoints/", "local/snapshots/", "local/final_results/"],
        "code_priority_prefixes": ["rosetta/model/", "script/train/", "script/evaluation/", "recipe/train_recipe/", "recipe/eval_recipe/"],
        "code_priority_tags": ["alignment_core", "projector_core", "runtime_path", "training_path"],
        "paper_priority_sections": ["abstract", "method", "approach", "experiment", "result", "analysis"],
        "paper_rebuttal_filters": ["empty_text", "short_heading_without_mechanism_terms", "section_marker_heading"],
        "rebuttal_priority_terms": ["concern", "reviewer", "weakness", "limitation", "baseline", "failure", "ablation"],
    }


def _dedupe_strings(values: list[Any], *, max_items: int = 40) -> list[str]:
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


def _prepare_chunk(chunk: dict[str, Any], *, max_input_chars: int) -> dict[str, Any]:
    text = str(chunk.get("text") or chunk.get("content") or chunk.get("text_preview") or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_input_chars:
        text = text[:max_input_chars]
    source_type = str(chunk.get("source_type") or "")
    return {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "source_type": source_type,
        "source_path": str(chunk.get("source_path") or chunk.get("path") or chunk.get("local_path") or ""),
        "paper_id": str(chunk.get("paper_id") or ""),
        "title": str(chunk.get("title") or ""),
        "section": str(chunk.get("section") or ("file" if source_type == "code" else "")),
        "path": str(chunk.get("path") or chunk.get("source_path") or ""),
        "symbol": str(chunk.get("symbol") or ""),
        "symbol_kind": str(chunk.get("symbol_kind") or ""),
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "keywords": chunk.get("keywords") or [],
        "risk_tags": chunk.get("risk_tags") or [],
        "edit_surface": chunk.get("edit_surface") or "",
        "semantic_enrichment_priority": chunk.get("semantic_enrichment_priority"),
        "semantic_enrichment_priority_reasons": chunk.get("semantic_enrichment_priority_reasons") or [],
        "tokens_estimate": chunk.get("tokens_estimate") or max(1, len(text) // 4),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _chunk_cache_key(chunk: dict[str, Any], *, model: str) -> str:
    payload = {
        "prompt_version": _prompt_version_for_chunk(chunk),
        "model": model,
        "chunk_id": chunk.get("chunk_id"),
        "source_type": chunk.get("source_type"),
        "text_sha256": chunk.get("text_sha256"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _call_deepseek(client: OpenAI, chunk: dict[str, Any], *, model: str, temperature: float, max_tokens: int, project_id: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=_messages_for_chunk(chunk),
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        stream=False,
        extra_body={"user_id": _safe_user_id(f"auto-research-{project_id}")},
    )
    content = response.choices[0].message.content or "{}"
    usage = _usage_dict(getattr(response, "usage", None))
    repaired = False
    try:
        enrichment = _parse_json_object(content)
    except Exception:
        repair_response = client.chat.completions.create(
            model=model,
            messages=_repair_messages_for_chunk(chunk, content),
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            stream=False,
            extra_body={"user_id": _safe_user_id(f"auto-research-{project_id}")},
        )
        repaired = True
        content = repair_response.choices[0].message.content or "{}"
        enrichment = _parse_json_object(content)
        usage = _merge_usage(usage, _usage_dict(getattr(repair_response, "usage", None)))
    normalized = _normalize_enrichment(enrichment)
    if str(chunk.get("source_type") or "") == "code" and _empty_semantic_enrichment(normalized):
        normalized = _deterministic_code_enrichment(chunk, reason="empty_llm_code_enrichment")
    return {
        "schema_version": S0_SEMANTIC_ENRICHMENT_SCHEMA_VERSION,
        "generated_at": now_utc(),
        "provider": "deepseek",
        "model": model,
        "prompt_version": _prompt_version_for_chunk(chunk),
        "json_repaired": repaired,
        "chunk": _chunk_ref(chunk),
        "enrichment": normalized,
        "usage": usage,
        "estimated_cost": _estimate_cost(usage),
    }


def _prompt_version_for_chunk(chunk: dict[str, Any]) -> str:
    if str(chunk.get("source_type") or "") == "code":
        return S0_CODE_SEMANTIC_ENRICHMENT_PROMPT_VERSION
    return S0_SEMANTIC_ENRICHMENT_PROMPT_VERSION


def _messages_for_chunk(chunk: dict[str, Any]) -> list[dict[str, str]]:
    if str(chunk.get("source_type") or "") == "code":
        return _messages_for_code_chunk(chunk)
    system = (
        "You enrich static research evidence chunks for an automated research pipeline. "
        "Return JSON only. Do not invent citations, file paths, line numbers, metrics, or claims not supported by the chunk. "
        "Keep chunk_id/source fields unchanged. Summaries should be specific and useful for S1 mechanism direction selection and S2 patch planning."
    )
    user = {
        "task": "Add semantic fields for this S0 chunk.",
        "required_json_schema": {
            "semantic_summary": "1-3 sentence grounded summary",
            "mechanism_tags": ["short mechanism tags"],
            "method_claims": ["supported method claims"],
            "failure_modes": ["possible or mentioned failure modes"],
            "implementation_relevance": "none|low|medium|high plus short reason",
            "dataset_relevance": [{"dataset": "name", "relevance": "none|low|medium|high", "reason": "short reason"}],
            "reviewer_risk_notes": ["risks relevant to rebuttal or reviewer concerns"],
            "retrieval_keywords": ["expanded search terms"],
            "s1_direction_utility": "how useful this chunk is for choosing a mechanism direction",
            "s2_patch_utility": "how useful this chunk is for concrete patch planning",
            "evidence_quality": "low|medium|high plus reason",
            "code_patch_surface_notes": "for code chunks only; otherwise empty string",
        },
        "chunk": chunk,
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _messages_for_code_chunk(chunk: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You summarize a code chunk for an automated research pipeline. Return compact JSON only. "
        "Use short strings and arrays. Do not quote code. Do not invent paths, symbols, tests, or metrics. "
        "If unsure, say unknown. Focus on patch planning surfaces."
    )
    structure = _code_structure_summary(chunk)
    user = {
        "task": "Fill this compact code summary from the provided structure and short code excerpt.",
        "required_json_schema": {
            "purpose": "one short sentence",
            "mechanism_role": "none|alignment|projection|runtime|training|config|evaluation|other",
            "patch_surface": "exact symbol/config knob to edit, or unknown",
            "ablation_hooks": ["up to 4 visible switches/knobs"],
            "inputs_outputs": ["up to 4 visible input/output tensors or args"],
            "risk_flags": ["up to 4 implementation risks"],
            "search_terms": ["up to 8 code search terms"],
            "s1_utility": "low|medium|high",
            "s2_utility": "low|medium|high",
            "confidence": "low|medium|high",
        },
        "chunk_ref": {
            key: chunk.get(key)
            for key in [
                "chunk_id",
                "source_type",
                "source_path",
                "path",
                "symbol",
                "symbol_kind",
                "start_line",
                "end_line",
                "keywords",
                "risk_tags",
                "edit_surface",
                "semantic_enrichment_priority_reasons",
            ]
            if chunk.get(key) not in (None, "", [])
        },
        "structure": structure,
        "code_excerpt": structure["code_excerpt"],
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _code_structure_summary(chunk: dict[str, Any]) -> dict[str, Any]:
    text = str(chunk.get("text") or "")
    compact_lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    signatures = []
    for line in compact_lines:
        stripped = line.strip()
        if re.match(r"^(class|def)\s+[A-Za-z_][A-Za-z0-9_]*", stripped):
            signatures.append(stripped[:180])
        if len(signatures) >= 8:
            break
    assignments = re.findall(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\s*=", text)
    args = []
    for match in re.finditer(r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*\(([^)]*)\)", text):
        for arg in match.group(1).split(","):
            name = arg.strip().split(":", 1)[0].split("=", 1)[0].strip()
            if name and name not in {"self", "cls"}:
                args.append(name)
    calls = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    keys = re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_.-]{2,})['\"]", text)
    tensor_terms = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:cache|mask|hidden|logit|loss|score|weight|gate|span|token|align|project)[A-Za-z0-9_]*\b", text, flags=re.I)
    excerpt = "\n".join(compact_lines[:24])
    if len(excerpt) > 1800:
        excerpt = excerpt[:1800]
    return {
        "symbol": chunk.get("symbol"),
        "symbol_kind": chunk.get("symbol_kind"),
        "path": chunk.get("path") or chunk.get("source_path"),
        "line_range": [chunk.get("start_line"), chunk.get("end_line")],
        "risk_tags": chunk.get("risk_tags") or [],
        "edit_surface": chunk.get("edit_surface"),
        "signatures": _dedupe_strings(signatures, max_items=8),
        "init_or_self_fields": _dedupe_strings(assignments, max_items=16),
        "args": _dedupe_strings(args, max_items=16),
        "calls": _dedupe_strings(calls, max_items=20),
        "string_keys": _dedupe_strings(keys, max_items=20),
        "tensor_terms": _dedupe_strings(tensor_terms, max_items=20),
        "code_excerpt": excerpt,
    }


def _repair_messages_for_chunk(chunk: dict[str, Any], bad_content: str) -> list[dict[str, str]]:
    system = (
        "You repair malformed JSON for a research evidence enrichment pipeline. "
        "Return one valid JSON object only. Preserve the semantic meaning of the draft. "
        "Do not add unsupported citations, paths, metrics, or line numbers."
    )
    user = {
        "task": "Repair this malformed enrichment response into the required JSON object.",
        "required_keys": [
            "semantic_summary",
            "mechanism_tags",
            "method_claims",
            "failure_modes",
            "implementation_relevance",
            "dataset_relevance",
            "reviewer_risk_notes",
            "retrieval_keywords",
            "s1_direction_utility",
            "s2_patch_utility",
            "evidence_quality",
            "code_patch_surface_notes",
        ],
        "chunk_ref": _chunk_ref(chunk),
        "malformed_response": bad_content[:12000],
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("DeepSeek response was not a JSON object")
    return value


def _normalize_enrichment(value: dict[str, Any]) -> dict[str, Any]:
    if _looks_like_compact_code_enrichment(value):
        return _normalize_compact_code_enrichment(value)
    return {
        "semantic_summary": _string(value.get("semantic_summary"), max_chars=1200),
        "mechanism_tags": _string_list(value.get("mechanism_tags"), max_items=12),
        "method_claims": _string_list(value.get("method_claims"), max_items=12),
        "failure_modes": _string_list(value.get("failure_modes"), max_items=12),
        "implementation_relevance": _string(value.get("implementation_relevance"), max_chars=600),
        "dataset_relevance": _dataset_relevance(value.get("dataset_relevance")),
        "reviewer_risk_notes": _string_list(value.get("reviewer_risk_notes"), max_items=12),
        "retrieval_keywords": _string_list(value.get("retrieval_keywords"), max_items=20),
        "s1_direction_utility": _string(value.get("s1_direction_utility"), max_chars=700),
        "s2_patch_utility": _string(value.get("s2_patch_utility"), max_chars=700),
        "evidence_quality": _string(value.get("evidence_quality"), max_chars=400),
        "code_patch_surface_notes": _string(value.get("code_patch_surface_notes"), max_chars=700),
    }


def _looks_like_compact_code_enrichment(value: dict[str, Any]) -> bool:
    return any(key in value for key in ["purpose", "mechanism_role", "patch_surface", "ablation_hooks", "inputs_outputs", "risk_flags", "search_terms"])


def _normalize_compact_code_enrichment(value: dict[str, Any]) -> dict[str, Any]:
    purpose = _string(value.get("purpose"), max_chars=500)
    role = _string(value.get("mechanism_role"), max_chars=80)
    patch_surface = _string(value.get("patch_surface"), max_chars=240)
    s1 = _string(value.get("s1_utility"), max_chars=80)
    s2 = _string(value.get("s2_utility"), max_chars=80)
    confidence = _string(value.get("confidence"), max_chars=80)
    tags = _string_list([role, *(_as_list(value.get("search_terms"))[:6])], max_items=8)
    risk_flags = _string_list(value.get("risk_flags"), max_items=8)
    search_terms = _string_list(value.get("search_terms"), max_items=12)
    hooks = _string_list(value.get("ablation_hooks"), max_items=8)
    io_terms = _string_list(value.get("inputs_outputs"), max_items=8)
    return {
        "semantic_summary": purpose,
        "mechanism_tags": tags,
        "method_claims": _string_list([f"role={role}", f"patch_surface={patch_surface}", *io_terms], max_items=8),
        "failure_modes": risk_flags,
        "implementation_relevance": f"{s2 or 'unknown'}: role={role or 'unknown'}; surface={patch_surface or 'unknown'}",
        "dataset_relevance": [],
        "reviewer_risk_notes": risk_flags[:3],
        "retrieval_keywords": _string_list([*search_terms, *hooks, role, patch_surface], max_items=20),
        "s1_direction_utility": s1,
        "s2_patch_utility": s2,
        "evidence_quality": confidence,
        "code_patch_surface_notes": "; ".join(item for item in [patch_surface, f"hooks={', '.join(hooks)}" if hooks else "", f"io={', '.join(io_terms)}" if io_terms else ""] if item)[:700],
    }


def _empty_semantic_enrichment(value: dict[str, Any]) -> bool:
    fields = ["semantic_summary", "mechanism_tags", "method_claims", "failure_modes", "implementation_relevance", "retrieval_keywords", "s1_direction_utility", "s2_patch_utility", "evidence_quality", "code_patch_surface_notes"]
    return not any(value.get(field) for field in fields)


def _deterministic_code_enrichment(chunk: dict[str, Any], *, reason: str) -> dict[str, Any]:
    structure = _code_structure_summary(chunk)
    symbol = str(structure.get("symbol") or chunk.get("symbol") or chunk.get("chunk_id") or "code chunk")
    role = _infer_code_mechanism_role(chunk)
    surface = symbol if symbol else str(chunk.get("path") or chunk.get("source_path") or "unknown")
    hooks = _dedupe_strings([*(structure.get("init_or_self_fields") or []), *(structure.get("string_keys") or [])], max_items=8)
    io_terms = _dedupe_strings([*(structure.get("args") or []), *(structure.get("tensor_terms") or [])], max_items=8)
    terms = _dedupe_strings([role, symbol, *(structure.get("calls") or []), *(structure.get("tensor_terms") or []), *(chunk.get("risk_tags") or [])], max_items=16)
    summary = f"{symbol} is a {role} code surface inferred from path, symbol, and local structure."
    risks = [reason]
    if "evaluation_path" in set(chunk.get("risk_tags") or []):
        risks.append("evaluation_path")
    if "test_path" in set(chunk.get("risk_tags") or []):
        risks.append("test_path")
    return {
        "semantic_summary": summary,
        "mechanism_tags": _string_list([role, *(chunk.get("risk_tags") or [])], max_items=8),
        "method_claims": _string_list([f"symbol={symbol}", f"role={role}", *io_terms], max_items=8),
        "failure_modes": _string_list(risks, max_items=8),
        "implementation_relevance": f"medium: deterministic structure summary for {role}",
        "dataset_relevance": [],
        "reviewer_risk_notes": _string_list(risks, max_items=3),
        "retrieval_keywords": _string_list([*terms, *hooks], max_items=20),
        "s1_direction_utility": "medium" if role in {"alignment", "projection", "runtime", "training"} else "low",
        "s2_patch_utility": "medium: inspect exact symbol and raw chunk",
        "evidence_quality": "low: deterministic code fallback",
        "code_patch_surface_notes": "; ".join(item for item in [surface, f"hooks={', '.join(hooks)}" if hooks else "", f"io={', '.join(io_terms)}" if io_terms else ""] if item)[:700],
    }


def _infer_code_mechanism_role(chunk: dict[str, Any]) -> str:
    tags = set(str(tag) for tag in chunk.get("risk_tags") or [])
    text = _priority_text(chunk)
    if "alignment_core" in tags or "align" in text:
        return "alignment"
    if "projector_core" in tags or "project" in text:
        return "projection"
    if "runtime_path" in tags or "wrapper" in text or "cache" in text:
        return "runtime"
    if "training_path" in tags or "loss" in text or "train" in text:
        return "training"
    if "evaluation_path" in tags:
        return "evaluation"
    if chunk.get("symbol_kind") == "config" or "recipe" in text:
        return "config"
    return "other"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dataset_relevance(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value[:12]:
        if isinstance(item, dict):
            items.append(
                {
                    "dataset": _string(item.get("dataset"), max_chars=80),
                    "relevance": _string(item.get("relevance"), max_chars=120),
                    "reason": _string(item.get("reason"), max_chars=300),
                }
            )
    return items


def _string(value: Any, *, max_chars: int) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()[:max_chars]


def _string_list(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_string(item, max_chars=180) for item in value[:max_items] if _string(item, max_chars=180)]


def _chunk_ref(chunk: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "chunk_id",
        "source_type",
        "source_path",
        "paper_id",
        "title",
        "section",
        "path",
        "symbol",
        "symbol_kind",
        "start_line",
        "end_line",
        "keywords",
        "risk_tags",
        "edit_surface",
        "semantic_enrichment_priority",
        "semantic_enrichment_priority_reasons",
        "tokens_estimate",
        "text_sha256",
    ]
    return {key: chunk.get(key) for key in keys if chunk.get(key) not in (None, "", [])}


def _dry_run_record(chunk: dict[str, Any], *, model: str, base_url: str) -> dict[str, Any]:
    prompt_tokens = _estimate_tokens(json.dumps(_messages_for_chunk(chunk), ensure_ascii=False))
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 600,
        "total_tokens": prompt_tokens + 600,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": prompt_tokens,
        "estimated": True,
    }
    return {
        "schema_version": S0_SEMANTIC_ENRICHMENT_SCHEMA_VERSION,
        "generated_at": now_utc(),
        "provider": "deepseek",
        "model": model,
        "base_url": base_url,
        "prompt_version": _prompt_version_for_chunk(chunk),
        "cache_status": "dry_run",
        "chunk": _chunk_ref(chunk),
        "enrichment": {
            "semantic_summary": f"Dry-run placeholder summary for {chunk.get('source_type')} chunk {chunk.get('chunk_id')}.",
            "mechanism_tags": _string_list(chunk.get("keywords") or [chunk.get("source_type")], max_items=6),
            "method_claims": [],
            "failure_modes": [],
            "implementation_relevance": "dry_run: not evaluated",
            "dataset_relevance": [],
            "reviewer_risk_notes": [],
            "retrieval_keywords": _string_list([*(chunk.get("keywords") or []), chunk.get("source_type"), chunk.get("section"), chunk.get("symbol")], max_items=12),
            "s1_direction_utility": "dry_run: not evaluated",
            "s2_patch_utility": "dry_run: not evaluated",
            "evidence_quality": "dry_run: not evaluated",
            "code_patch_surface_notes": "dry_run: not evaluated" if chunk.get("source_type") == "code" else "",
        },
        "usage": usage,
        "estimated_cost": _estimate_cost(usage),
    }


def _fallback_record(chunk: dict[str, Any], *, model: str, error: str) -> dict[str, Any]:
    summary_subject = chunk.get("symbol") or chunk.get("section") or chunk.get("chunk_id")
    keywords = _string_list([*(chunk.get("keywords") or []), *(chunk.get("risk_tags") or []), chunk.get("symbol"), chunk.get("path")], max_items=12)
    source_type = str(chunk.get("source_type") or "")
    enrichment = (
        _deterministic_code_enrichment(chunk, reason="llm_json_enrichment_failed")
        if source_type == "code"
        else {
            "semantic_summary": f"Fallback semantic summary for {source_type} chunk {summary_subject}. DeepSeek returned malformed JSON; use raw chunk text and retrieval keywords for final evidence.",
            "mechanism_tags": keywords[:6],
            "method_claims": [],
            "failure_modes": ["llm_json_enrichment_failed"],
            "implementation_relevance": "low: fallback based on metadata only",
            "dataset_relevance": [],
            "reviewer_risk_notes": [],
            "retrieval_keywords": keywords,
            "s1_direction_utility": "low: fallback enrichment requires reading raw chunk",
            "s2_patch_utility": "low: fallback enrichment requires reading raw chunk",
            "evidence_quality": "low: deterministic fallback after malformed LLM JSON",
            "code_patch_surface_notes": "",
        }
    )
    return {
        "schema_version": S0_SEMANTIC_ENRICHMENT_SCHEMA_VERSION,
        "generated_at": now_utc(),
        "provider": "deepseek",
        "model": model,
        "prompt_version": _prompt_version_for_chunk(chunk),
        "json_repaired": False,
        "fallback_used": True,
        "fallback_reason": error[:800],
        "chunk": _chunk_ref(chunk),
        "enrichment": enrichment,
        "usage": {},
        "estimated_cost": _estimate_cost({}),
    }


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = {key: getattr(usage, key) for key in dir(usage) if not key.startswith("_") and isinstance(getattr(usage, key), (int, float, str, type(None)))}
    prompt_tokens = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
    cache_hit = int(raw.get("prompt_cache_hit_tokens") or raw.get("cache_hit_tokens") or 0)
    cache_miss = int(raw.get("prompt_cache_miss_tokens") or raw.get("cache_miss_tokens") or 0)
    if prompt_tokens and not cache_hit and not cache_miss:
        cache_miss = prompt_tokens
    total_tokens = int(raw.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cache_hit_tokens": cache_hit,
        "prompt_cache_miss_tokens": cache_miss,
    }


def _merge_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = ["prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"]
    return {key: int(left.get(key) or 0) + int(right.get(key) or 0) for key in keys}


def _estimate_cost(usage: dict[str, Any]) -> dict[str, float]:
    hit = float(usage.get("prompt_cache_hit_tokens") or 0)
    miss = float(usage.get("prompt_cache_miss_tokens") or usage.get("prompt_tokens") or 0)
    output = float(usage.get("completion_tokens") or 0)
    usd = (
        hit * DEEPSEEK_V4_FLASH_PRICING_USD_PER_MTOK["input_cache_hit"]
        + miss * DEEPSEEK_V4_FLASH_PRICING_USD_PER_MTOK["input_cache_miss"]
        + output * DEEPSEEK_V4_FLASH_PRICING_USD_PER_MTOK["output"]
    ) / 1_000_000
    cny = (
        hit * DEEPSEEK_V4_FLASH_PRICING_CNY_PER_MTOK["input_cache_hit"]
        + miss * DEEPSEEK_V4_FLASH_PRICING_CNY_PER_MTOK["input_cache_miss"]
        + output * DEEPSEEK_V4_FLASH_PRICING_CNY_PER_MTOK["output"]
    ) / 1_000_000
    return {
        "usd": round(usd, 8),
        "cny": round(cny, 8),
        "pricing": {
            "usd_per_1m_tokens": DEEPSEEK_V4_FLASH_PRICING_USD_PER_MTOK,
            "cny_per_1m_tokens": DEEPSEEK_V4_FLASH_PRICING_CNY_PER_MTOK,
        },
    }


def _cost_summary(records: list[dict[str, Any]], *, total_available_chunks: int) -> dict[str, Any]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }
    usd = 0.0
    cny = 0.0
    api_records = 0
    for record in records:
        usage = record.get("usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
        cost = record.get("estimated_cost") or {}
        usd += float(cost.get("usd") or 0.0)
        cny += float(cost.get("cny") or 0.0)
        if record.get("cache_status") in {"miss", "dry_run"}:
            api_records += 1
    sample_count = max(1, len(records))
    projection_multiplier = float(total_available_chunks) / float(sample_count) if total_available_chunks else 0.0
    return {
        "pricing_note": "DeepSeek V4 Flash prices are per 1M tokens; cached input is much cheaper than cache-miss input.",
        "actual_sample_cost_usd": round(usd, 8),
        "actual_sample_cost_cny": round(cny, 8),
        "actual_api_record_count": api_records,
        "sample_record_count": len(records),
        "total_available_chunks": total_available_chunks,
        "projected_full_cost_usd": round(usd * projection_multiplier, 8),
        "projected_full_cost_cny": round(cny * projection_multiplier, 8),
        "usage_totals": totals,
        "usd_per_1m_tokens": DEEPSEEK_V4_FLASH_PRICING_USD_PER_MTOK,
        "cny_per_1m_tokens": DEEPSEEK_V4_FLASH_PRICING_CNY_PER_MTOK,
    }


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _jsonl(items: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + ("\n" if items else "")


def _cache_safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "model"


def _safe_user_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9\-_]+", "-", value).strip("-")
    return safe[:512] or "auto-research"
