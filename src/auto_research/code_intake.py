"""Tree-sitter based repository code intake."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import ensure_dir, now_utc, read_json, sanitize_filename, sha256_file, write_json


CODE_INTAKE_SCHEMA_VERSION = "code_intake_v1"
CODE_TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".sh", ".cfg", ".ini", ".toml"}
CODE_SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "htmlcov",
    "wandb",
    "checkpoints",
    "snapshots",
    "auto_research_runs",
}
MAX_CODE_FILE_BYTES = 512_000


@dataclass(frozen=True)
class CodeIntakeResult:
    file_manifest: dict[str, Any]
    symbols: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    repo_map: dict[str, Any]
    repo_map_md: str
    report: dict[str, Any]
    report_md: str
    surface_map: dict[str, Any]
    retrieval_index: dict[str, Any]


def build_code_intake(
    repo_root: Path,
    *,
    allowed_files: list[str] | None = None,
    allowed_prefixes: list[str] | None = None,
    cache_dir: Path | None = None,
) -> CodeIntakeResult:
    files = _collect_code_files(repo_root)
    allowed_files = allowed_files or []
    allowed_prefixes = allowed_prefixes or []
    cache_dir = cache_dir.expanduser() if cache_dir else None
    if cache_dir:
        ensure_dir(cache_dir)
    manifest_entries = []
    symbols: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    cache_events: list[dict[str, Any]] = []
    for rel in files:
        path = repo_root / rel
        language = _language_for_path(path)
        file_sha = sha256_file(path)
        file_entry = {
            "path": rel,
            "language": language,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha,
            "edit_surface": _edit_surface(rel, allowed_files=allowed_files, allowed_prefixes=allowed_prefixes),
        }
        manifest_entries.append(file_entry)
        cached = _read_code_file_cache(cache_dir, file_entry) if cache_dir else None
        if cached:
            parsed = cached
            file_entry["cache_status"] = "hit"
            cache_events.append({"path": rel, "sha256": file_sha, "status": "hit"})
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix == ".py":
                parsed = _parse_python_file(text, rel, file_entry["edit_surface"])
            else:
                config_chunks = _non_python_chunks(text, rel, language, file_entry["edit_surface"])
                parsed = {
                    "symbols": [_file_symbol_from_chunk(chunk) for chunk in config_chunks],
                    "chunks": config_chunks,
                    "edges": [],
                }
            file_entry["cache_status"] = "miss" if cache_dir else "disabled"
            cache_events.append({"path": rel, "sha256": file_sha, "status": file_entry["cache_status"]})
            _write_code_file_cache(cache_dir, file_entry, parsed) if cache_dir else None
        symbols.extend(parsed.get("symbols") or [])
        chunks.extend(parsed.get("chunks") or [])
        edges.extend(parsed.get("edges") or [])
    edges.extend(_same_file_neighbor_edges(chunks))
    edges.extend(_tested_by_edges(chunks))
    edges.extend(_resolved_call_edges(chunks, symbols))
    edges.extend(_config_key_edges(chunks))
    edges = _dedupe_edges(edges)
    repo_map = _build_repo_map(manifest_entries, symbols, chunks, edges)
    report = _build_code_intake_report(
        files=manifest_entries,
        symbols=symbols,
        chunks=chunks,
        edges=edges,
        allowed_files=allowed_files,
        allowed_prefixes=allowed_prefixes,
        cache_events=cache_events,
    )
    surface_map = _build_implementation_surface_map(symbols=symbols, chunks=chunks, edges=edges)
    retrieval_index = _build_code_retrieval_index(chunks=chunks, symbols=symbols, edges=edges, surface_map=surface_map)
    return CodeIntakeResult(
        file_manifest={"schema_version": CODE_INTAKE_SCHEMA_VERSION, "files": manifest_entries},
        symbols=symbols,
        chunks=chunks,
        edges=edges,
        repo_map=repo_map,
        repo_map_md=_repo_map_markdown(repo_map),
        report=report,
        report_md=_code_intake_report_markdown(report),
        surface_map=surface_map,
        retrieval_index=retrieval_index,
    )


def retrieve_code_chunks(
    *,
    query: str,
    chunks: list[dict[str, Any]],
    symbols: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
    top_k: int = 12,
    prefer_editable: bool = True,
    allowed_risk_tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank code chunks for S1/S2 prompts using local static metadata."""
    del symbols, edges
    terms = _query_terms(query)
    if not terms:
        return []
    allowed_risk_tags = allowed_risk_tags or []
    ranked = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        score, reasons = _score_code_chunk(chunk, terms, prefer_editable=prefer_editable, allowed_risk_tags=allowed_risk_tags)
        if score <= 0:
            continue
        ranked.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "path": chunk.get("path"),
                "symbol": chunk.get("symbol"),
                "node_type": chunk.get("node_type"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "edit_surface": chunk.get("edit_surface"),
                "risk_tags": chunk.get("risk_tags", []),
                "score": round(score, 3),
                "match_reasons": reasons[:10],
                "text_preview": chunk.get("text_preview", ""),
                "keywords": chunk.get("keywords", [])[:20],
            }
        )
    return sorted(ranked, key=lambda item: (-float(item.get("score") or 0.0), str(item.get("path") or ""), str(item.get("symbol") or "")))[:top_k]


def _collect_code_files(repo_root: Path) -> list[str]:
    paths = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root).as_posix()
        if not _is_code_file(path, rel):
            continue
        paths.append(rel)
    return paths


def _is_code_file(path: Path, rel: str) -> bool:
    if path.suffix.lower() not in CODE_TEXT_SUFFIXES:
        return False
    if path.stat().st_size > MAX_CODE_FILE_BYTES:
        return False
    if set(Path(rel).parts).intersection(CODE_SKIP_PARTS):
        return False
    return True


def _read_code_file_cache(cache_dir: Path | None, file_entry: dict[str, Any]) -> dict[str, Any] | None:
    if not cache_dir:
        return None
    path = _code_file_cache_path(cache_dir, str(file_entry.get("path") or ""), str(file_entry.get("sha256") or ""))
    if not path.exists():
        return None
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != CODE_INTAKE_SCHEMA_VERSION:
        return None
    if payload.get("path") != file_entry.get("path") or payload.get("sha256") != file_entry.get("sha256"):
        return None
    parsed = payload.get("parsed")
    if not isinstance(parsed, dict):
        return None
    return {
        "symbols": parsed.get("symbols") if isinstance(parsed.get("symbols"), list) else [],
        "chunks": parsed.get("chunks") if isinstance(parsed.get("chunks"), list) else [],
        "edges": parsed.get("edges") if isinstance(parsed.get("edges"), list) else [],
    }


def _write_code_file_cache(cache_dir: Path | None, file_entry: dict[str, Any], parsed: dict[str, Any]) -> None:
    if not cache_dir:
        return
    path = _code_file_cache_path(cache_dir, str(file_entry.get("path") or ""), str(file_entry.get("sha256") or ""))
    ensure_dir(path.parent)
    write_json(
        path,
        {
            "schema_version": CODE_INTAKE_SCHEMA_VERSION,
            "created_at": now_utc(),
            "path": file_entry.get("path"),
            "sha256": file_entry.get("sha256"),
            "language": file_entry.get("language"),
            "edit_surface": file_entry.get("edit_surface"),
            "parsed": parsed,
        },
    )


def _code_file_cache_path(cache_dir: Path, rel_path: str, sha256: str) -> Path:
    slug = sanitize_filename(rel_path.replace("/", "__"), max_length=96)
    return cache_dir / f"{slug}.{sha256[:16]}.json"


def _parse_python_file(text: str, rel_path: str, edit_surface: str) -> dict[str, Any]:
    imports = _imports_with_ast(text)
    chunks = [_file_prelude_chunk(text, rel_path, imports, edit_surface)]
    symbols = []
    edges = []
    try:
        tree = _python_tree_sitter_tree(text)
    except Exception:
        return _parse_python_file_with_ast(text, rel_path, edit_surface, imports)
    if tree is None:
        return _parse_python_file_with_ast(text, rel_path, edit_surface, imports)
    root = tree.root_node
    class_nodes = _direct_children(root, {"class_definition"})
    function_nodes = _direct_children(root, {"function_definition", "async_function_definition"})
    for class_node in class_nodes:
        class_symbol, class_chunk, method_nodes = _class_symbol_and_chunk(text, rel_path, class_node, imports, edit_surface)
        symbols.append(class_symbol)
        chunks.append(class_chunk)
        edges.append(_contains_edge(_file_symbol_id(rel_path), class_symbol["symbol_id"]))
        for method_node in method_nodes:
            method_symbol, method_chunk, method_edges = _function_symbol_and_chunk(
                text,
                rel_path,
                method_node,
                imports,
                edit_surface,
                parent_symbol=class_symbol["symbol"],
                parent_symbol_id=class_symbol["symbol_id"],
            )
            symbols.append(method_symbol)
            chunks.append(method_chunk)
            edges.append(_contains_edge(class_symbol["symbol_id"], method_symbol["symbol_id"]))
            edges.extend(method_edges)
    for function_node in function_nodes:
        symbol, chunk, function_edges = _function_symbol_and_chunk(text, rel_path, function_node, imports, edit_surface)
        symbols.append(symbol)
        chunks.append(chunk)
        edges.append(_contains_edge(_file_symbol_id(rel_path), symbol["symbol_id"]))
        edges.extend(function_edges)
    if len(chunks) == 1:
        fallback = _module_chunk(text, rel_path, imports, edit_surface)
        chunks.append(fallback)
        symbols.append(_file_symbol_from_chunk(fallback))
    return {"symbols": symbols, "chunks": chunks, "edges": edges}


def _python_tree_sitter_tree(text: str):
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython

    parser = Parser(Language(tspython.language()))
    return parser.parse(text.encode("utf-8"))


def _direct_children(node: Any, types: set[str]) -> list[Any]:
    return [child for child in node.children if child.type in types]


def _node_text(text: str, node: Any) -> str:
    return text.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _node_name(text: str, node: Any) -> str:
    name_node = node.child_by_field_name("name")
    return _node_text(text, name_node).strip() if name_node else ""


def _node_lines(node: Any) -> tuple[int, int]:
    return int(node.start_point[0]) + 1, int(node.end_point[0]) + 1


def _signature_from_node(text: str, node: Any) -> str:
    node_text = _node_text(text, node).strip()
    first_line = node_text.splitlines()[0] if node_text else ""
    return first_line.strip()


def _class_symbol_and_chunk(text: str, rel_path: str, node: Any, imports: list[str], edit_surface: str) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    name = _node_name(text, node)
    start_line, end_line = _node_lines(node)
    body_node = node.child_by_field_name("body")
    method_nodes = _direct_children(body_node, {"function_definition", "async_function_definition"}) if body_node else []
    method_names = [_node_name(text, method) for method in method_nodes]
    class_text = _node_text(text, node)
    docstring = _docstring_from_python_source(class_text)
    symbol_id = f"{rel_path}::{name}"
    skeleton = "\n".join([_signature_from_node(text, node), *(f"    def {method_name}(...)" for method_name in method_names if method_name)])
    chunk = _code_chunk(
        rel_path=rel_path,
        node_type=node.type,
        symbol=name,
        parent_symbol="",
        symbol_kind="class",
        start_line=start_line,
        end_line=end_line,
        signature=_signature_from_node(text, node),
        docstring=docstring,
        imports=imports,
        text=skeleton,
        edit_surface=edit_surface,
        extra={"methods": method_names, "chunk_role": "class_skeleton"},
    )
    symbol = _symbol_from_chunk(chunk, symbol_id=symbol_id)
    return symbol, chunk, method_nodes


def _function_symbol_and_chunk(
    text: str,
    rel_path: str,
    node: Any,
    imports: list[str],
    edit_surface: str,
    *,
    parent_symbol: str = "",
    parent_symbol_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    name = _node_name(text, node)
    symbol = f"{parent_symbol}.{name}" if parent_symbol else name
    start_line, end_line = _node_lines(node)
    body_text = _node_text(text, node)
    docstring = _docstring_from_python_source(body_text)
    calls = _calls_with_ast(body_text)
    references = _references_with_ast(body_text)
    config_keys = _config_keys(body_text)
    chunk = _code_chunk(
        rel_path=rel_path,
        node_type=node.type,
        symbol=symbol,
        parent_symbol=parent_symbol,
        symbol_kind="method" if parent_symbol else "function",
        start_line=start_line,
        end_line=end_line,
        signature=_signature_from_node(text, node),
        docstring=docstring,
        imports=imports,
        text=body_text,
        edit_surface=edit_surface,
        extra={
            "calls": calls,
            "references": references,
            "config_keys": config_keys,
            "chunk_role": "function_body",
        },
    )
    symbol_id = f"{rel_path}::{symbol}"
    symbol_payload = _symbol_from_chunk(chunk, symbol_id=symbol_id)
    edges = []
    for call in calls[:80]:
        edges.append({"edge_type": "calls", "src": symbol_id, "dst": call, "confidence": "static_name_match"})
    for key in config_keys[:40]:
        edges.append({"edge_type": "reads_config", "src": symbol_id, "dst": key, "confidence": "static_literal_match"})
    if parent_symbol_id:
        edges.append(_contains_edge(parent_symbol_id, symbol_id))
    return symbol_payload, chunk, edges


def _parse_python_file_with_ast(text: str, rel_path: str, edit_surface: str, imports: list[str]) -> dict[str, Any]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        chunk = _module_chunk(text[:2400], rel_path, imports, edit_surface)
        return {"symbols": [_file_symbol_from_chunk(chunk)], "chunks": [chunk], "edges": []}
    lines = text.splitlines()
    chunks = [_file_prelude_chunk(text, rel_path, imports, edit_surface)]
    symbols = []
    edges = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_symbol, class_chunk = _ast_symbol_and_chunk(
                lines,
                rel_path,
                node,
                imports,
                edit_surface,
                symbol=node.name,
                parent_symbol="",
                symbol_kind="class",
            )
            chunks.append(class_chunk)
            symbols.append(class_symbol)
            edges.append(_contains_edge(_file_symbol_id(rel_path), class_symbol["symbol_id"]))
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                method_symbol, method_chunk = _ast_symbol_and_chunk(
                    lines,
                    rel_path,
                    child,
                    imports,
                    edit_surface,
                    symbol=f"{node.name}.{child.name}",
                    parent_symbol=node.name,
                    symbol_kind="method",
                )
                chunks.append(method_chunk)
                symbols.append(method_symbol)
                edges.append(_contains_edge(class_symbol["symbol_id"], method_symbol["symbol_id"]))
                edges.extend(_semantic_edges_from_chunk(method_symbol["symbol_id"], method_chunk))
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        symbol, chunk = _ast_symbol_and_chunk(
            lines,
            rel_path,
            node,
            imports,
            edit_surface,
            symbol=node.name,
            parent_symbol="",
            symbol_kind="function",
        )
        chunks.append(chunk)
        symbols.append(symbol)
        edges.append(_contains_edge(_file_symbol_id(rel_path), symbol["symbol_id"]))
        edges.extend(_semantic_edges_from_chunk(symbol["symbol_id"], chunk))
    if len(chunks) == 1:
        fallback = _module_chunk(text, rel_path, imports, edit_surface)
        chunks.append(fallback)
        symbols.append(_file_symbol_from_chunk(fallback))
    return {"symbols": symbols, "chunks": chunks, "edges": edges}


def _ast_symbol_and_chunk(
    lines: list[str],
    rel_path: str,
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    imports: list[str],
    edit_surface: str,
    *,
    symbol: str,
    parent_symbol: str,
    symbol_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = max(1, getattr(node, "lineno", 1))
    end = max(start, getattr(node, "end_lineno", start))
    source = "\n".join(lines[start - 1 : end])
    node_type = (
        "class_definition"
        if isinstance(node, ast.ClassDef)
        else "async_function_definition" if isinstance(node, ast.AsyncFunctionDef) else "function_definition"
    )
    chunk = _code_chunk(
        rel_path=rel_path,
        node_type=node_type,
        symbol=symbol,
        parent_symbol=parent_symbol,
        symbol_kind=symbol_kind,
        start_line=start,
        end_line=end,
        signature=source.splitlines()[0].strip() if source else "",
        docstring=ast.get_docstring(node) or "",
        imports=imports,
        text=source,
        edit_surface=edit_surface,
        extra={
            "calls": _calls_with_ast(source),
            "references": _references_with_ast(source),
            "config_keys": _config_keys(source),
            "chunk_role": "class_body" if symbol_kind == "class" else "function_body",
        },
    )
    return _symbol_from_chunk(chunk, symbol_id=f"{rel_path}::{symbol}"), chunk


def _semantic_edges_from_chunk(symbol_id: str, chunk: dict[str, Any]) -> list[dict[str, Any]]:
    edges = []
    for call in (chunk.get("calls") or [])[:80]:
        edges.append({"edge_type": "calls", "src": symbol_id, "dst": call, "confidence": "static_name_match"})
    for key in (chunk.get("config_keys") or [])[:40]:
        edges.append({"edge_type": "reads_config", "src": symbol_id, "dst": key, "confidence": "static_literal_match"})
    return edges


def _file_prelude_chunk(text: str, rel_path: str, imports: list[str], edit_surface: str) -> dict[str, Any]:
    lines = text.splitlines()
    prelude_lines = []
    for line in lines[:120]:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) or not stripped or stripped.startswith(('"""', "'''", "#")) or re.match(r"^[A-Z_][A-Z0-9_]*\\s*=", stripped):
            prelude_lines.append(line)
            continue
        if prelude_lines:
            break
    prelude = "\n".join(prelude_lines).strip() or "\n".join(lines[:40])
    return _code_chunk(
        rel_path=rel_path,
        node_type="file_prelude",
        symbol=Path(rel_path).stem,
        parent_symbol="",
        symbol_kind="file",
        start_line=1,
        end_line=max(1, len(prelude.splitlines())),
        signature=f"module {rel_path}",
        docstring=_docstring_from_python_source(text),
        imports=imports,
        text=prelude,
        edit_surface=edit_surface,
        extra={"chunk_role": "file_prelude"},
    )


def _module_chunk(text: str, rel_path: str, imports: list[str], edit_surface: str) -> dict[str, Any]:
    return _code_chunk(
        rel_path=rel_path,
        node_type="module",
        symbol=Path(rel_path).stem,
        parent_symbol="",
        symbol_kind="file",
        start_line=1,
        end_line=len(text.splitlines()),
        signature=f"module {rel_path}",
        docstring=_docstring_from_python_source(text),
        imports=imports,
        text=text[:2400],
        edit_surface=edit_surface,
        extra={"chunk_role": "module_body"},
    )


def _non_python_chunks(text: str, rel_path: str, language: str, edit_surface: str) -> list[dict[str, Any]]:
    chunks = []
    if language in {"json", "yaml"}:
        chunks.extend(_config_chunks(text, rel_path, language, edit_surface))
    if chunks:
        return chunks
    for idx, chunk_text in enumerate(_chunk_text(text, max_chars=2200, overlap=200)):
        chunks.append(
            _code_chunk(
                rel_path=rel_path,
                node_type=f"{language}_text",
                symbol=f"{Path(rel_path).stem}:{idx}",
                parent_symbol="",
                symbol_kind="file",
                start_line=_line_number_for_offset(text, text.find(chunk_text)),
                end_line=_line_number_for_offset(text, text.find(chunk_text) + len(chunk_text)),
                signature=f"{language} chunk {idx}",
                docstring="",
                imports=[],
                text=chunk_text,
                edit_surface=edit_surface,
                extra={"chunk_role": "text_chunk"},
            )
        )
    return chunks


def _config_chunks(text: str, rel_path: str, language: str, edit_surface: str) -> list[dict[str, Any]]:
    chunks = []
    key_paths = _json_key_paths(text) if language == "json" else _yaml_like_key_paths(text)
    if not key_paths:
        return []
    chunk = _code_chunk(
        rel_path=rel_path,
        node_type=f"{language}_config",
        symbol=Path(rel_path).stem,
        parent_symbol="",
        symbol_kind="config",
        start_line=1,
        end_line=len(text.splitlines()),
        signature=f"{language} config {rel_path}",
        docstring="",
        imports=[],
        text=text[:4000],
        edit_surface=edit_surface,
        extra={"chunk_role": "config_chunk", "config_keys": key_paths[:120]},
    )
    chunks.append(chunk)
    return chunks


def _code_chunk(
    *,
    rel_path: str,
    node_type: str,
    symbol: str,
    parent_symbol: str,
    symbol_kind: str,
    start_line: int,
    end_line: int,
    signature: str,
    docstring: str,
    imports: list[str],
    text: str,
    edit_surface: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = extra or {}
    text = text.rstrip() + ("\n" if text else "")
    chunk_id = f"{rel_path}::{symbol or Path(rel_path).stem}::{start_line}-{end_line}"
    keywords = _code_keywords(text, extra_terms=[rel_path, symbol, parent_symbol, symbol_kind, node_type, *(extra.get("config_keys") or [])])
    return {
        "chunk_id": chunk_id,
        "source_type": "code",
        "language": _language_for_path(Path(rel_path)),
        "path": rel_path,
        "source_path": rel_path,
        "node_type": node_type,
        "symbol": symbol,
        "parent_symbol": parent_symbol,
        "symbol_kind": symbol_kind,
        "section": symbol or Path(rel_path).stem,
        "start_line": start_line,
        "end_line": end_line,
        "start_byte": None,
        "end_byte": None,
        "signature": signature,
        "decorators": _decorators_from_text(text),
        "docstring": docstring,
        "imports": imports,
        "calls": extra.get("calls") or [],
        "references": extra.get("references") or [],
        "config_keys": extra.get("config_keys") or [],
        "risk_tags": _risk_tags(rel_path, text),
        "edit_surface": edit_surface,
        "keywords": keywords,
        "text": text,
        "text_preview": " ".join(text.split())[:500],
        "tokens_estimate": max(1, len(text) // 4),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        **{key: value for key, value in extra.items() if key not in {"calls", "references", "config_keys"}},
    }


def _symbol_from_chunk(chunk: dict[str, Any], *, symbol_id: str) -> dict[str, Any]:
    return {
        "symbol_id": symbol_id,
        "path": chunk["path"],
        "symbol": chunk["symbol"],
        "parent_symbol": chunk.get("parent_symbol", ""),
        "kind": chunk.get("symbol_kind", ""),
        "node_type": chunk.get("node_type", ""),
        "signature": chunk.get("signature", ""),
        "docstring": chunk.get("docstring", ""),
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "chunk_id": chunk.get("chunk_id"),
        "edit_surface": chunk.get("edit_surface"),
        "risk_tags": chunk.get("risk_tags", []),
        "keywords": chunk.get("keywords", []),
    }


def _file_symbol_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return _symbol_from_chunk(chunk, symbol_id=_file_symbol_id(chunk["path"]))


def _file_symbol_id(rel_path: str) -> str:
    return f"{rel_path}::<module>"


def _contains_edge(src: str, dst: str) -> dict[str, Any]:
    return {"edge_type": "contains", "src": src, "dst": dst, "confidence": "syntax"}


def _same_file_neighbor_edges(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    by_file: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_file.setdefault(chunk.get("path", ""), []).append(chunk)
    for file_chunks in by_file.values():
        ordered = sorted(file_chunks, key=lambda item: int(item.get("start_line") or 0))
        for left, right in zip(ordered, ordered[1:]):
            edges.append({"edge_type": "same_file_neighbor", "src": left["chunk_id"], "dst": right["chunk_id"], "confidence": "line_order"})
    return edges


def _tested_by_edges(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_chunks = [chunk for chunk in chunks if "/test" in f"/{chunk.get('path', '')}" or Path(str(chunk.get("path", ""))).name.startswith("test_")]
    code_chunks = [chunk for chunk in chunks if chunk not in test_chunks]
    edges = []
    for test_chunk in test_chunks:
        text = " ".join([test_chunk.get("symbol", ""), test_chunk.get("text_preview", "")]).lower()
        for code_chunk in code_chunks:
            symbol = str(code_chunk.get("symbol") or "").split(".")[-1].lower()
            if symbol and symbol in text:
                edges.append({"edge_type": "tested_by", "src": code_chunk["chunk_id"], "dst": test_chunk["chunk_id"], "confidence": "symbol_name_match"})
                edges.append({"edge_type": "tests_symbol", "src": test_chunk["chunk_id"], "dst": code_chunk["chunk_id"], "confidence": "symbol_name_match"})
    return edges


def _resolved_call_edges(chunks: list[dict[str, Any]], symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_full_symbol = {str(symbol.get("symbol")): symbol for symbol in symbols if symbol.get("symbol")}
    by_short_symbol: dict[str, list[dict[str, Any]]] = {}
    by_path_and_short: dict[tuple[str, str], dict[str, Any]] = {}
    class_method_by_path: dict[tuple[str, str], dict[str, Any]] = {}
    for symbol in symbols:
        name = str(symbol.get("symbol") or "")
        if not name:
            continue
        short = name.split(".")[-1]
        by_short_symbol.setdefault(short, []).append(symbol)
        by_path_and_short[(str(symbol.get("path") or ""), short)] = symbol
        parent = str(symbol.get("parent_symbol") or "")
        if parent:
            class_method_by_path[(str(symbol.get("path") or ""), f"{parent}.{short}")] = symbol
    edges = []
    for chunk in chunks:
        src = chunk.get("chunk_id")
        if not src:
            continue
        parent = str(chunk.get("parent_symbol") or "")
        path = str(chunk.get("path") or "")
        imports = _import_alias_map(chunk.get("imports") or [])
        for call in chunk.get("calls") or []:
            call_name = str(call)
            target = None
            confidence = "unresolved"
            if call_name.startswith("self.") and parent:
                target = class_method_by_path.get((path, f"{parent}.{call_name.split('.', 1)[1]}"))
                confidence = "self_method_resolution" if target else confidence
            if target is None and call_name in by_full_symbol:
                target = by_full_symbol[call_name]
                confidence = "full_symbol_match"
            if target is None and "." in call_name:
                owner, attr = call_name.rsplit(".", 1)
                imported_owner = imports.get(owner)
                if imported_owner:
                    candidates = [
                        symbol
                        for symbol in by_short_symbol.get(attr, [])
                        if str(symbol.get("path") or "").replace("/", ".").endswith(imported_owner.replace(".", "/"))
                    ]
                    if len(candidates) == 1:
                        target = candidates[0]
                        confidence = "import_alias_symbol_match"
                if target is None:
                    target = by_path_and_short.get((path, attr))
                    confidence = "same_file_attr_match" if target else confidence
            if target is None:
                candidates = by_short_symbol.get(call_name, [])
                if len(candidates) == 1:
                    target = candidates[0]
                    confidence = "unique_short_symbol_match"
                elif by_path_and_short.get((path, call_name)):
                    target = by_path_and_short[(path, call_name)]
                    confidence = "same_file_symbol_match"
            if target:
                edges.append(
                    {
                        "edge_type": "resolved_call",
                        "src": src,
                        "dst": target.get("chunk_id") or target.get("symbol_id"),
                        "symbol_id": target.get("symbol_id"),
                        "call": call_name,
                        "confidence": confidence,
                    }
                )
    return edges


def _config_key_edges(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config_chunks = [chunk for chunk in chunks if chunk.get("symbol_kind") == "config" or chunk.get("chunk_role") == "config_chunk"]
    key_to_config: dict[str, list[dict[str, Any]]] = {}
    for chunk in config_chunks:
        for key in chunk.get("config_keys") or []:
            key_text = str(key)
            key_to_config.setdefault(key_text, []).append(chunk)
            key_to_config.setdefault(key_text.split(".")[-1], []).append(chunk)
    edges = []
    for chunk in chunks:
        src = chunk.get("chunk_id")
        if not src or chunk in config_chunks:
            continue
        for key in chunk.get("config_keys") or []:
            matches = key_to_config.get(str(key)) or key_to_config.get(str(key).split(".")[-1]) or []
            for match in matches[:8]:
                edges.append(
                    {
                        "edge_type": "config_key_defined_in",
                        "src": src,
                        "dst": match.get("chunk_id"),
                        "config_key": key,
                        "confidence": "config_key_match",
                    }
                )
    return edges


def _import_alias_map(imports: list[str]) -> dict[str, str]:
    aliases = {}
    for item in imports:
        text = str(item)
        if " as " in text:
            module, alias = [part.strip() for part in text.split(" as ", 1)]
            aliases[alias] = module
        else:
            aliases[text.rsplit(".", 1)[-1]] = text
    return aliases


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for edge in edges:
        key = (edge.get("edge_type"), edge.get("src"), edge.get("dst"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _build_repo_map(files: list[dict[str, Any]], symbols: list[dict[str, Any]], chunks: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    by_file = []
    symbol_by_file: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        symbol_by_file.setdefault(symbol.get("path", ""), []).append(symbol)
    for file_entry in files:
        file_symbols = symbol_by_file.get(file_entry["path"], [])
        by_file.append(
            {
                **file_entry,
                "symbol_count": len(file_symbols),
                "symbols": [
                    {
                        "symbol": item.get("symbol"),
                        "kind": item.get("kind"),
                        "start_line": item.get("start_line"),
                        "end_line": item.get("end_line"),
                        "edit_surface": item.get("edit_surface"),
                    }
                    for item in file_symbols[:80]
                ],
            }
        )
    return {
        "schema_version": CODE_INTAKE_SCHEMA_VERSION,
        "counts": {
            "files": len(files),
            "symbols": len(symbols),
            "chunks": len(chunks),
            "edges": len(edges),
        },
        "files": by_file,
        "top_editable_symbols": [
            item
            for item in symbols
            if item.get("edit_surface") == "allowed"
        ][:80],
    }


def _repo_map_markdown(repo_map: dict[str, Any]) -> str:
    lines = ["# Code Repo Map", ""]
    counts = repo_map.get("counts") or {}
    lines.append(f"- Files: {counts.get('files', 0)}")
    lines.append(f"- Symbols: {counts.get('symbols', 0)}")
    lines.append(f"- Chunks: {counts.get('chunks', 0)}")
    lines.append(f"- Edges: {counts.get('edges', 0)}")
    lines.append("")
    for file_entry in repo_map.get("files", []):
        lines.append(f"## {file_entry.get('path')}")
        lines.append(f"- language: {file_entry.get('language')}")
        lines.append(f"- edit_surface: {file_entry.get('edit_surface')}")
        for symbol in file_entry.get("symbols", [])[:40]:
            lines.append(f"- `{symbol.get('symbol')}` ({symbol.get('kind')}) lines {symbol.get('start_line')}-{symbol.get('end_line')}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_code_intake_report(
    *,
    files: list[dict[str, Any]],
    symbols: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    allowed_files: list[str],
    allowed_prefixes: list[str],
    cache_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    files_by_path = {item.get("path"): item for item in files}
    python_files = [item for item in files if item.get("language") == "python"]
    python_with_function_chunks = {
        chunk.get("path")
        for chunk in chunks
        if chunk.get("language") == "python" and chunk.get("symbol_kind") in {"class", "method", "function"}
    }
    chunks_by_role: dict[str, int] = {}
    chunks_by_surface: dict[str, int] = {}
    chunks_by_risk: dict[str, int] = {}
    for chunk in chunks:
        chunks_by_role[str(chunk.get("chunk_role") or "unknown")] = chunks_by_role.get(str(chunk.get("chunk_role") or "unknown"), 0) + 1
        chunks_by_surface[str(chunk.get("edit_surface") or "unknown")] = chunks_by_surface.get(str(chunk.get("edit_surface") or "unknown"), 0) + 1
        for tag in chunk.get("risk_tags") or []:
            chunks_by_risk[str(tag)] = chunks_by_risk.get(str(tag), 0) + 1
    empty_chunks = [chunk.get("chunk_id") for chunk in chunks if not str(chunk.get("text") or "").strip()]
    oversized_chunks = [
        {"chunk_id": chunk.get("chunk_id"), "tokens_estimate": chunk.get("tokens_estimate")}
        for chunk in chunks
        if int(chunk.get("tokens_estimate") or 0) > 1400
    ]
    duplicate_hashes = _duplicate_chunk_hashes(chunks)
    missing_allowed_files = [path for path in allowed_files if path not in files_by_path]
    allowed_file_coverage = [
        {
            "path": path,
            "present": path in files_by_path,
            "chunk_count": sum(1 for chunk in chunks if chunk.get("path") == path),
            "symbol_count": sum(1 for symbol in symbols if symbol.get("path") == path),
            "function_level_chunk_count": sum(
                1 for chunk in chunks if chunk.get("path") == path and chunk.get("symbol_kind") in {"class", "method", "function"}
            ),
        }
        for path in allowed_files
    ]
    quality_flags = []
    if python_files and len(python_with_function_chunks) / max(1, len(python_files)) < 0.25:
        quality_flags.append("low_python_function_chunk_coverage")
    if missing_allowed_files:
        quality_flags.append("missing_allowed_files")
    if empty_chunks:
        quality_flags.append("empty_code_chunks")
    if oversized_chunks:
        quality_flags.append("oversized_code_chunks")
    if duplicate_hashes:
        quality_flags.append("duplicate_code_chunks")
    if not any(chunk.get("calls") for chunk in chunks):
        quality_flags.append("no_static_call_signals")
    if not any(chunk.get("config_keys") for chunk in chunks):
        quality_flags.append("no_config_key_signals")
    cache_events = cache_events or []
    cache_counts: dict[str, int] = {}
    for event in cache_events:
        status = str(event.get("status") or "unknown")
        cache_counts[status] = cache_counts.get(status, 0) + 1
    return {
        "schema_version": CODE_INTAKE_SCHEMA_VERSION,
        "counts": {
            "files": len(files),
            "python_files": len(python_files),
            "symbols": len(symbols),
            "chunks": len(chunks),
            "edges": len(edges),
            "function_level_chunks": sum(1 for chunk in chunks if chunk.get("symbol_kind") in {"class", "method", "function"}),
            "editable_chunks": sum(1 for chunk in chunks if chunk.get("edit_surface") in {"allowed", "allowed_prefix"}),
            "chunks_with_calls": sum(1 for chunk in chunks if chunk.get("calls")),
            "chunks_with_config_keys": sum(1 for chunk in chunks if chunk.get("config_keys")),
            "chunks_with_risk_tags": sum(1 for chunk in chunks if chunk.get("risk_tags")),
        },
        "coverage": {
            "allowed_files": allowed_file_coverage,
            "allowed_prefixes": allowed_prefixes,
            "missing_allowed_files": missing_allowed_files,
            "python_function_file_ratio": round(len(python_with_function_chunks) / max(1, len(python_files)), 4),
        },
        "distributions": {
            "chunks_by_role": dict(sorted(chunks_by_role.items())),
            "chunks_by_edit_surface": dict(sorted(chunks_by_surface.items())),
            "chunks_by_risk_tag": dict(sorted(chunks_by_risk.items())),
        },
        "diagnostics": {
            "quality_flags": quality_flags,
            "empty_chunks": empty_chunks[:20],
            "oversized_chunks": oversized_chunks[:20],
            "duplicate_chunk_hashes": duplicate_hashes[:20],
        },
        "cache": {
            "enabled": bool(cache_events) and not all(event.get("status") == "disabled" for event in cache_events),
            "counts": dict(sorted(cache_counts.items())),
            "events": cache_events[:200],
        },
    }


def _duplicate_chunk_hashes(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[str]] = {}
    for chunk in chunks:
        sha = str(chunk.get("sha256") or "")
        chunk_id = str(chunk.get("chunk_id") or "")
        if sha and chunk_id:
            by_hash.setdefault(sha, []).append(chunk_id)
    return [{"sha256": sha, "chunk_ids": ids[:12], "count": len(ids)} for sha, ids in by_hash.items() if len(ids) > 1]


def _code_intake_report_markdown(report: dict[str, Any]) -> str:
    counts = report.get("counts") or {}
    coverage = report.get("coverage") or {}
    diagnostics = report.get("diagnostics") or {}
    lines = ["# Code Intake Report", ""]
    lines.append(f"- Files: {counts.get('files', 0)}")
    lines.append(f"- Python files: {counts.get('python_files', 0)}")
    lines.append(f"- Symbols: {counts.get('symbols', 0)}")
    lines.append(f"- Chunks: {counts.get('chunks', 0)}")
    lines.append(f"- Edges: {counts.get('edges', 0)}")
    lines.append(f"- Editable chunks: {counts.get('editable_chunks', 0)}")
    lines.append(f"- Python function file ratio: {coverage.get('python_function_file_ratio', 0)}")
    flags = diagnostics.get("quality_flags") or []
    lines.append(f"- Quality flags: {', '.join(flags) if flags else 'none'}")
    cache = report.get("cache") or {}
    lines.append(f"- Cache: enabled={cache.get('enabled')} counts={cache.get('counts', {})}")
    lines.extend(["", "## Allowed File Coverage"])
    for item in coverage.get("allowed_files") or []:
        lines.append(
            f"- {item.get('path')}: present={item.get('present')} chunks={item.get('chunk_count')} "
            f"symbols={item.get('symbol_count')} function_chunks={item.get('function_level_chunk_count')}"
        )
    return "\n".join(lines).strip() + "\n"


def _build_implementation_surface_map(*, symbols: list[dict[str, Any]], chunks: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    del symbols
    surfaces = {
        "alignment_core": [],
        "projector_core": [],
        "runtime_path": [],
        "training_path": [],
        "configuration": [],
        "evaluation_or_test": [],
        "other_editable": [],
    }
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        item = _surface_item_from_chunk(chunk)
        if not item:
            continue
        tags = set(chunk.get("risk_tags") or [])
        if "test_path" in tags or "evaluation_path" in tags or chunk.get("edit_surface") == "forbidden":
            surfaces["evaluation_or_test"].append(item)
        if chunk.get("symbol_kind") == "config" or chunk.get("config_keys"):
            surfaces["configuration"].append(item)
        if "alignment_core" in tags:
            surfaces["alignment_core"].append(item)
        if "projector_core" in tags:
            surfaces["projector_core"].append(item)
        if "runtime_path" in tags:
            surfaces["runtime_path"].append(item)
        if "training_path" in tags:
            surfaces["training_path"].append(item)
        if chunk.get("edit_surface") in {"allowed", "allowed_prefix"} and not tags.intersection({"alignment_core", "projector_core", "runtime_path", "training_path"}):
            surfaces["other_editable"].append(item)
    for key, values in surfaces.items():
        surfaces[key] = _dedupe_surface_items(values)[:80]
    relation_summary: dict[str, int] = {}
    for edge in edges:
        relation_summary[str(edge.get("edge_type") or "unknown")] = relation_summary.get(str(edge.get("edge_type") or "unknown"), 0) + 1
    return {
        "schema_version": CODE_INTAKE_SCHEMA_VERSION,
        "surfaces": surfaces,
        "relation_summary": dict(sorted(relation_summary.items())),
        "guidance": [
            "Prefer alignment_core/projector_core/runtime_path chunks with edit_surface=allowed for S2.5 mechanism patches.",
            "Treat evaluation_or_test as evidence only unless a targeted test addition is explicitly needed.",
            "Use configuration entries to wire ablation switches and recipe-level defaults.",
        ],
    }


def _surface_item_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    if chunk.get("chunk_role") == "file_prelude" and chunk.get("symbol_kind") == "file":
        return {}
    return {
        "chunk_id": chunk.get("chunk_id"),
        "path": chunk.get("path"),
        "symbol": chunk.get("symbol"),
        "symbol_kind": chunk.get("symbol_kind"),
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "edit_surface": chunk.get("edit_surface"),
        "risk_tags": chunk.get("risk_tags", []),
        "config_keys": chunk.get("config_keys", [])[:20],
        "calls": chunk.get("calls", [])[:20],
        "text_preview": chunk.get("text_preview", ""),
    }


def _dedupe_surface_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        key = item.get("chunk_id")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return sorted(deduped, key=lambda item: (str(item.get("path") or ""), int(item.get("start_line") or 0), str(item.get("symbol") or "")))


def _build_code_retrieval_index(
    *,
    chunks: list[dict[str, Any]],
    symbols: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    surface_map: dict[str, Any],
) -> dict[str, Any]:
    del symbols, edges
    default_queries = [
        "alignment valid_mask span overlap confidence gate token cache routing",
        "projector projection residual hidden states dtype device",
        "runtime wrapper forward cache attention past_key_values",
        "training loss optimizer recipe ablation switch",
        "evaluation metric benchmark dataset openbookqa mmlu arc",
    ]
    return {
        "schema_version": CODE_INTAKE_SCHEMA_VERSION,
        "default_queries": [
            {"query": query, "results": retrieve_code_chunks(query=query, chunks=chunks, top_k=8)}
            for query in default_queries
        ],
        "surface_queries": [
            {"surface": surface, "chunk_ids": [item.get("chunk_id") for item in items[:30] if item.get("chunk_id")]}
            for surface, items in (surface_map.get("surfaces") or {}).items()
        ],
        "usage": {
            "function": "auto_research.code_intake.retrieve_code_chunks",
            "inputs": ["query", "chunks", "top_k", "prefer_editable", "allowed_risk_tags"],
            "recommended_top_k": 8,
        },
    }


def _query_terms(query: str) -> list[str]:
    raw = re.split(r"[^A-Za-z0-9_\\-]+", query.lower())
    stop = {"the", "and", "for", "with", "from", "into", "this", "that", "candidate", "idea", "patch"}
    return _dedupe_strings([term for term in raw if len(term) >= 3 and term not in stop])


def _score_code_chunk(
    chunk: dict[str, Any],
    terms: list[str],
    *,
    prefer_editable: bool,
    allowed_risk_tags: list[str],
) -> tuple[float, list[str]]:
    haystacks = {
        "symbol": " ".join([str(chunk.get("symbol") or ""), str(chunk.get("signature") or "")]).lower(),
        "path": str(chunk.get("path") or "").lower(),
        "keywords": " ".join(str(item) for item in chunk.get("keywords") or []).lower(),
        "risk_tags": " ".join(str(item) for item in chunk.get("risk_tags") or []).lower(),
        "config_keys": " ".join(str(item) for item in chunk.get("config_keys") or []).lower(),
        "calls": " ".join(str(item) for item in chunk.get("calls") or []).lower(),
        "references": " ".join(str(item) for item in chunk.get("references") or []).lower(),
        "text_preview": str(chunk.get("text_preview") or "").lower(),
    }
    weights = {
        "symbol": 4.0,
        "path": 2.0,
        "keywords": 3.0,
        "risk_tags": 3.0,
        "config_keys": 3.0,
        "calls": 2.0,
        "references": 1.5,
        "text_preview": 1.0,
    }
    score = 0.0
    reasons = []
    for term in terms:
        for field, haystack in haystacks.items():
            if term in haystack:
                score += weights[field]
                reasons.append(f"{field}:{term}")
    if prefer_editable and chunk.get("edit_surface") in {"allowed", "allowed_prefix"}:
        score += 1.5
        reasons.append(f"edit_surface:{chunk.get('edit_surface')}")
    if allowed_risk_tags:
        overlap = set(allowed_risk_tags).intersection(set(chunk.get("risk_tags") or []))
        if overlap:
            score += 2.0 * len(overlap)
            reasons.extend(f"risk_tag:{tag}" for tag in sorted(overlap))
    if chunk.get("edit_surface") == "forbidden":
        score -= 2.0
        reasons.append("penalty:forbidden")
    if chunk.get("chunk_role") == "file_prelude":
        score -= 0.5
    return score, _dedupe_strings(reasons)


def _imports_with_ast(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"{alias.name} as {alias.asname}" if alias.asname else alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                name = f"{module}.{alias.name}" if module else alias.name
                imports.append(f"{name} as {alias.asname}" if alias.asname else name)
    return imports[:120]


def _calls_with_ast(text: str) -> list[str]:
    try:
        tree = ast.parse(textwrap.dedent(text))
    except SyntaxError:
        return []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls.append(_call_name(node.func))
    return _dedupe_strings([call for call in calls if call])[:120]


def _references_with_ast(text: str) -> list[str]:
    try:
        tree = ast.parse(textwrap.dedent(text))
    except SyntaxError:
        return []
    refs = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    return _dedupe_strings(refs)[:160]


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _call_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


def _docstring_from_python_source(text: str) -> str:
    try:
        node = ast.parse(textwrap.dedent(text))
    except SyntaxError:
        return ""
    return ast.get_docstring(node) or ""


def _decorators_from_text(text: str) -> list[str]:
    return [line.strip()[1:] for line in text.splitlines() if line.strip().startswith("@")][:20]


def _config_keys(text: str) -> list[str]:
    keys = re.findall(r"['\"]([a-zA-Z_][a-zA-Z0-9_]*(?:alignment|confidence|soft|gate|top_k|entropy|span|token|cache|routing|mode)[a-zA-Z0-9_]*)['\"]", text)
    return _dedupe_strings(keys)[:80]


def _json_key_paths(text: str) -> list[str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    keys = []
    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                keys.append(path)
                walk(child, path)
        elif isinstance(value, list):
            for idx, child in enumerate(value[:8]):
                walk(child, f"{prefix}[{idx}]")
    walk(payload)
    return keys[:200]


def _yaml_like_key_paths(text: str) -> list[str]:
    keys = []
    stack: list[tuple[int, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^(?P<indent>\\s*)(?P<key>[A-Za-z_][A-Za-z0-9_\\-]*)\\s*:", line)
        if not match:
            continue
        indent = len(match.group("indent"))
        key = match.group("key")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = ".".join(item[1] for item in stack)
        full = f"{parent}.{key}" if parent else key
        keys.append(full)
        stack.append((indent, key))
    return keys[:200]


def _risk_tags(rel_path: str, text: str) -> list[str]:
    lowered = f"{rel_path}\n{text}".lower()
    tags = []
    if any(marker in lowered for marker in ["aligner", "alignment", "span", "valid_mask"]):
        tags.append("alignment_core")
    if any(marker in lowered for marker in ["projector", "projection"]):
        tags.append("projector_core")
    if any(marker in lowered for marker in ["wrapper", "forward", "cache"]):
        tags.append("runtime_path")
    if any(marker in lowered for marker in ["train", "loss", "optimizer"]):
        tags.append("training_path")
    if any(marker in lowered for marker in ["eval", "benchmark", "metric"]):
        tags.append("evaluation_path")
    if Path(rel_path).name.startswith("test_") or "/test/" in f"/{rel_path}/":
        tags.append("test_path")
    return _dedupe_strings(tags)


def _edit_surface(rel_path: str, *, allowed_files: list[str], allowed_prefixes: list[str]) -> str:
    if rel_path in set(allowed_files):
        return "allowed"
    if any(rel_path.startswith(prefix) for prefix in allowed_prefixes):
        return "allowed_prefix"
    if "evaluation" in rel_path or Path(rel_path).name.startswith("test_") or "/test/" in f"/{rel_path}/":
        return "forbidden"
    return "risky"


def _code_keywords(text: str, *, extra_terms: list[Any]) -> list[str]:
    candidates = []
    for value in extra_terms:
        if value:
            candidates.extend(re.split(r"[^A-Za-z0-9_\\-]+", str(value)))
    candidates.extend(_config_keys(text))
    candidates.extend(re.findall(r"\\b[A-Za-z_][A-Za-z0-9_]{3,}\\b", text))
    return _dedupe_strings([item for item in candidates if len(item) <= 48])[:32]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _line_number_for_offset(text: str, offset: int) -> int:
    if offset < 0:
        return 1
    return text.count("\n", 0, offset) + 1


def _language_for_path(path: Path) -> str:
    if path.suffix == ".py":
        return "python"
    if path.suffix == ".json":
        return "json"
    if path.suffix in {".yaml", ".yml"}:
        return "yaml"
    if path.suffix == ".md":
        return "markdown"
    if path.suffix == ".sh":
        return "shell"
    return "text"
