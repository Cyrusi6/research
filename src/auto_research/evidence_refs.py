"""Evidence reference resolution for stage contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import read_json


REF_FIELDS = ("evidence_refs", "counterevidence_refs", "code_refs")


def resolve_s1_evidence_refs(project_root: Path, payload: dict[str, Any], *, mode: str = "c2c") -> dict[str, Any]:
    """Resolve S1 evidence/code refs against S0 chunks and local artifacts."""
    catalog = _load_reference_catalog(project_root)
    errors: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    strict = mode == "c2c" and bool(catalog["chunk_ids"] or catalog["source_paths"] or catalog["code_files"])

    for kind, ref, owner in _iter_payload_refs(payload):
        result = _resolve_ref(project_root, catalog, ref, kind=kind, strict=strict)
        entry = {"kind": kind, "owner": owner, "ref": _compact_ref(ref), **result}
        if result["status"] == "resolved":
            resolved.append(entry)
        elif result["status"] == "unresolved":
            errors.append(entry)

    status = "pass" if not errors else "fail"
    return {
        "schema_version": "s1_evidence_ref_report_v1",
        "status": status,
        "mode": mode,
        "counts": {
            "resolved": len(resolved),
            "unresolved": len(errors),
            "chunk_ids": len(catalog["chunk_ids"]),
            "source_paths": len(catalog["source_paths"]),
            "code_files": len(catalog["code_files"]),
            "code_symbols": len(catalog["code_symbols"]),
        },
        "errors": errors,
        "resolved": resolved[:80],
    }


def evidence_ref_errors_for_repair(report: dict[str, Any], *, limit: int = 12) -> list[str]:
    errors = []
    for item in report.get("errors") or []:
        if not isinstance(item, dict):
            continue
        ref = item.get("ref") if isinstance(item.get("ref"), dict) else {}
        parts = [
            str(item.get("kind") or "ref"),
            str(item.get("owner") or "unknown_owner"),
            str(item.get("reason") or "unresolved"),
        ]
        if ref.get("chunk_id"):
            parts.append(f"chunk_id={ref.get('chunk_id')}")
        if ref.get("source_path"):
            parts.append(f"source_path={ref.get('source_path')}")
        if ref.get("source_label"):
            parts.append(f"source_label={ref.get('source_label')}")
        errors.append(" | ".join(part for part in parts if part))
        if len(errors) >= limit:
            break
    return errors


def _iter_payload_refs(payload: dict[str, Any]):
    bundle = payload.get("evidence_bundle") if isinstance(payload.get("evidence_bundle"), dict) else {}
    for idx, item in enumerate(bundle.get("items") or []):
        if isinstance(item, dict):
            yield "evidence_bundle.items", item, f"evidence_bundle.items[{idx}]"
    for idea_idx, idea in enumerate(payload.get("selected_ideas") or []):
        if not isinstance(idea, dict):
            continue
        owner = str(idea.get("id") or idea.get("title") or f"selected_ideas[{idea_idx}]")
        for field in REF_FIELDS:
            for idx, ref in enumerate(idea.get(field) or []):
                if isinstance(ref, dict):
                    yield field, ref, f"{owner}.{field}[{idx}]"


def _resolve_ref(project_root: Path, catalog: dict[str, set[str]], ref: dict[str, Any], *, kind: str, strict: bool) -> dict[str, str]:
    chunk_id = _ref_chunk_id(ref)
    source_path = _ref_source_path(ref)
    source_label = _ref_source_label(ref)
    source_type = str(ref.get("source_type") or "").lower()
    is_code_ref = kind == "code_refs" or source_type == "code"

    if chunk_id:
        if chunk_id in catalog["chunk_ids"]:
            return {"status": "resolved", "reason": "chunk_id_in_chunk_index"}
        if strict and source_type in {"paper", "rebuttal", "code"}:
            return {"status": "unresolved", "reason": "chunk_id_not_in_chunk_index"}

    if source_path:
        if _path_exists(project_root, source_path) or source_path in catalog["source_paths"] or _path_suffix_match(source_path, catalog["source_paths"]):
            return {"status": "resolved", "reason": "source_path_exists_or_indexed"}
        if strict:
            return {"status": "unresolved", "reason": "source_path_missing"}

    if is_code_ref:
        code_target = source_label or source_path or chunk_id
        if _matches_code_target(code_target, catalog):
            return {"status": "resolved", "reason": "code_ref_matches_file_or_symbol"}
        if strict:
            return {"status": "unresolved", "reason": "code_ref_missing_file_or_symbol"}

    if source_label:
        if source_label in catalog["chunk_ids"]:
            return {"status": "resolved", "reason": "source_label_matches_chunk_id"}
        if _path_exists(project_root, source_label) or source_label in catalog["source_paths"] or _path_suffix_match(source_label, catalog["source_paths"]):
            return {"status": "resolved", "reason": "source_label_matches_artifact_or_source_path"}
        if _label_anchor_known(source_label, catalog):
            return {"status": "resolved", "reason": "source_label_anchor_known"}

    if strict:
        return {"status": "unresolved", "reason": "no_resolvable_chunk_path_or_code_ref"}
    return {"status": "unknown", "reason": "no_authoritative_index_for_soft_ref"}


def _load_reference_catalog(project_root: Path) -> dict[str, set[str]]:
    catalog = {"chunk_ids": set(), "source_paths": set(), "code_files": set(), "code_symbols": set(), "artifact_paths": set()}
    for rel in [
        "intake/c2c/chunk_index.json",
        "literature/c2c/chunk_index.json",
    ]:
        _add_chunk_index(catalog, read_json(project_root / rel, default={}) or {})
    bundle = read_json(project_root / "intake/c2c/static_bundle.json", default={}) or {}
    if isinstance(bundle, dict):
        _add_chunk_index(catalog, bundle.get("chunk_index") or {})
        _add_code_manifest(catalog, bundle.get("code_file_manifest") or {})
        _add_code_symbols(catalog, bundle.get("code_symbols") or [])
        for chunk in bundle.get("code_chunks") or []:
            _add_code_chunk(catalog, chunk)
    for rel in [
        "intake/c2c/code_file_manifest.json",
        "literature/c2c/code_file_manifest.json",
    ]:
        _add_code_manifest(catalog, read_json(project_root / rel, default={}) or {})
    for rel in [
        "intake/c2c/code_symbols.jsonl",
        "literature/c2c/code_symbols.jsonl",
    ]:
        _add_code_symbols(catalog, _read_jsonl(project_root / rel))
    for rel in [
        "intake/c2c/code_chunks.jsonl",
        "literature/c2c/code_chunks.jsonl",
    ]:
        for chunk in _read_jsonl(project_root / rel):
            _add_code_chunk(catalog, chunk)
    for rel in [
        "literature/survey.md",
        "literature/theme_map.md",
        "literature/papers/metadata.json",
        "meta/negative_memory.jsonl",
        "experiment/results/failure_feedback.json",
        "plan/performance_feedback.json",
        "plan/direction_scorecard.json",
        "intake/c2c/negative_result_memory.json",
        "literature/c2c/negative_result_memory.json",
        "intake/c2c/result_ledger.csv",
        "literature/c2c/result_ledger.csv",
    ]:
        if (project_root / rel).exists():
            catalog["artifact_paths"].add(rel)
            catalog["source_paths"].add(rel)
    return catalog


def _add_chunk_index(catalog: dict[str, set[str]], chunk_index: Any) -> None:
    if not isinstance(chunk_index, dict):
        return
    for entry in chunk_index.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        chunk_id = entry.get("chunk_id")
        if chunk_id:
            catalog["chunk_ids"].add(str(chunk_id))
        for key in ("source_path", "path", "local_path"):
            value = entry.get(key)
            if value:
                catalog["source_paths"].add(str(value))
        _add_code_chunk(catalog, entry)


def _add_code_manifest(catalog: dict[str, set[str]], manifest: Any) -> None:
    files = manifest.get("files") if isinstance(manifest, dict) else manifest
    if not isinstance(files, list):
        return
    for item in files:
        if not isinstance(item, dict):
            continue
        for key in ("path", "rel_path", "file_path", "source_path"):
            value = item.get(key)
            if value:
                catalog["code_files"].add(str(value))


def _add_code_symbols(catalog: dict[str, set[str]], symbols: Any) -> None:
    if not isinstance(symbols, list):
        return
    for item in symbols:
        if not isinstance(item, dict):
            continue
        for key in ("symbol", "qualified_name", "name", "symbol_id"):
            value = item.get(key)
            if value:
                catalog["code_symbols"].add(str(value))
        for key in ("path", "file_path", "source_path"):
            value = item.get(key)
            if value:
                catalog["code_files"].add(str(value))


def _add_code_chunk(catalog: dict[str, set[str]], chunk: Any) -> None:
    if not isinstance(chunk, dict):
        return
    for key in ("chunk_id",):
        value = chunk.get(key)
        if value:
            catalog["chunk_ids"].add(str(value))
    for key in ("path", "source_path"):
        value = chunk.get(key)
        if value:
            catalog["code_files"].add(str(value))
            catalog["source_paths"].add(str(value))
    symbol = chunk.get("symbol")
    if symbol:
        catalog["code_symbols"].add(str(symbol))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            items.append(payload)
    return items


def _ref_chunk_id(ref: dict[str, Any]) -> str:
    for key in ("chunk_id", "chunk"):
        value = ref.get(key)
        if value:
            return str(value)
    return ""


def _ref_source_path(ref: dict[str, Any]) -> str:
    value = ref.get("source_path") or ref.get("path") or ref.get("file")
    return str(value) if value else ""


def _ref_source_label(ref: dict[str, Any]) -> str:
    value = ref.get("source_label") or ref.get("label") or ref.get("source")
    return str(value) if value else ""


def _path_exists(project_root: Path, value: str) -> bool:
    path = Path(_strip_ref_anchor(value))
    if path.is_absolute():
        return path.exists()
    return (project_root / path).exists()


def _path_suffix_match(value: str, candidates: set[str]) -> bool:
    normalized = _strip_ref_anchor(value).removeprefix("code:")
    return any(item == normalized or item.endswith("/" + normalized) or normalized.endswith("/" + item) for item in candidates)


def _matches_code_target(value: str, catalog: dict[str, set[str]]) -> bool:
    if not value:
        return False
    normalized = _strip_ref_anchor(value).removeprefix("code:")
    if normalized in catalog["code_files"] or normalized in catalog["code_symbols"] or normalized in catalog["chunk_ids"]:
        return True
    return _path_suffix_match(normalized, catalog["code_files"]) or any(
        symbol == normalized or symbol.endswith("." + normalized) or normalized.endswith("." + symbol)
        for symbol in catalog["code_symbols"]
    )


def _label_anchor_known(value: str, catalog: dict[str, set[str]]) -> bool:
    if ":" not in value:
        return False
    prefix, suffix = value.split(":", 1)
    if prefix in {"code", "paper", "rebuttal"}:
        return value in catalog["chunk_ids"] or _path_suffix_match(suffix, catalog["source_paths"])
    if prefix in {"feedback", "failure_feedback"}:
        return any("negative_result_memory" in item or "failure_feedback" in item or "performance_feedback" in item for item in catalog["source_paths"])
    return False


def _strip_ref_anchor(value: str) -> str:
    text = str(value)
    if "#" not in text:
        return text
    before, after = text.split("#", 1)
    if before and after:
        return before
    return text


def _compact_ref(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ref.get(key)
        for key in ("chunk_id", "source_path", "source_type", "source_label", "claim")
        if ref.get(key) not in (None, "", [])
    }
