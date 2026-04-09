"""Import helpers for external research notes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .artifacts import ArtifactManager
from .utils import compact_markdown


class ConsensusImporter:
    stage_key = "S1_literature"

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.artifacts = ArtifactManager(project_root)

    def import_file(self, source_path: Path, *, label: str | None = None) -> dict[str, Any]:
        label = label or source_path.stem
        raw_text = source_path.read_text(encoding="utf-8")
        normalized = self._normalize_text(source_path, raw_text)
        extracted = self._extract_signals(source_path, raw_text, normalized)
        safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label).strip("._") or "consensus_import"

        raw_record = self.artifacts.write_text(
            self.stage_key,
            f"imports/consensus/{safe_label}.raw.txt",
            raw_text,
            artifact_type="consensus_raw",
            summary="Imported Consensus transcript",
        )
        normalized_record = self.artifacts.write_text(
            self.stage_key,
            f"imports/consensus/{safe_label}.normalized.md",
            normalized,
            artifact_type="consensus_normalized",
            summary="Normalized Consensus transcript",
            source_paths=[raw_record["path"]],
        )
        extracted_record = self.artifacts.write_json(
            self.stage_key,
            f"imports/consensus/{safe_label}.extracted.json",
            extracted,
            artifact_type="consensus_extracted",
            summary="Extracted Consensus signals",
            source_paths=[normalized_record["path"]],
        )
        return {
            "raw": raw_record["path"],
            "normalized": normalized_record["path"],
            "extracted": extracted_record["path"],
            "summary": extracted,
        }

    def list_imports(self) -> list[dict[str, Any]]:
        imports_dir = self.project_root / "literature" / "imports" / "consensus"
        if not imports_dir.exists():
            return []
        records = []
        for path in sorted(imports_dir.glob("*.extracted.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
        return records

    @staticmethod
    def _normalize_text(source_path: Path, raw_text: str) -> str:
        if source_path.suffix.lower() == ".ris":
            entries = parse_ris(raw_text)
            lines = ["# Consensus RIS Import", ""]
            for idx, entry in enumerate(entries, start=1):
                authors = ", ".join(entry.get("authors", [])) or "Unknown authors"
                lines.extend(
                    [
                        f"## Entry {idx}",
                        f"- Title: {entry.get('title', 'Untitled')}",
                        f"- Authors: {authors}",
                        f"- Year: {entry.get('year', 'n/a')}",
                        f"- DOI: {entry.get('doi', 'n/a')}",
                        f"- URL: {entry.get('url', 'n/a')}",
                        "",
                    ]
                )
            return compact_markdown("\n".join(lines))
        if source_path.suffix.lower() == ".json":
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                return compact_markdown(raw_text)
            lines = []
            if isinstance(payload, dict):
                messages = payload.get("messages") or payload.get("conversation") or payload.get("items")
                if isinstance(messages, list):
                    for item in messages:
                        if isinstance(item, dict):
                            role = item.get("role") or item.get("speaker") or "message"
                            content = item.get("content") or item.get("text") or item.get("message") or ""
                            if isinstance(content, list):
                                content = " ".join(str(x) for x in content)
                            lines.append(f"## {role}\n\n{content}")
                else:
                    lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
            elif isinstance(payload, list):
                for idx, item in enumerate(payload, start=1):
                    lines.append(f"## message_{idx}\n\n{item}")
            return compact_markdown("\n\n".join(lines))
        return compact_markdown(raw_text)

    @staticmethod
    def _extract_signals(source_path: Path, raw_text: str, text: str) -> dict[str, Any]:
        if source_path.suffix.lower() == ".ris":
            entries = parse_ris(raw_text)
            titles = [entry.get("title", "") for entry in entries if entry.get("title")]
            urls = sorted({entry.get("url", "") for entry in entries if entry.get("url")})
            dois = sorted({entry.get("doi", "") for entry in entries if entry.get("doi")})
            queries = [title for title in titles if _looks_itr_relevant(title)]
            return {
                "import_type": "consensus_ris",
                "urls": urls,
                "arxiv_ids": sorted(set(re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", " ".join(urls)))),
                "dois": dois,
                "paper_title_candidates": titles[:50],
                "queries": queries[:20],
                "entries": entries,
            }
        urls = sorted(set(re.findall(r"https?://\S+", text)))
        arxiv_ids = sorted(set(re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text)))
        dois = sorted(set(re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, flags=re.IGNORECASE)))
        quoted_titles = sorted(set(re.findall(r"[\"“](.+?)[\"”]", text)))
        queries = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith(("query:", "search:", "topic:")):
                queries.append(stripped.split(":", 1)[1].strip())
        return {
            "import_type": "consensus_dialogue",
            "urls": urls,
            "arxiv_ids": arxiv_ids,
            "dois": dois,
            "paper_title_candidates": quoted_titles[:30],
            "queries": queries[:20],
        }


def parse_ris(raw_text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("ER"):
            if current:
                entries.append(_normalize_ris_entry(current))
                current = {}
            continue
        if "  - " not in line:
            continue
        tag, value = line.split("  - ", 1)
        tag = tag.strip()
        value = value.strip()
        if tag in {"AU", "KW"}:
            current.setdefault(tag, []).append(value)
        else:
            current[tag] = value
    if current:
        entries.append(_normalize_ris_entry(current))
    return entries


def _normalize_ris_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": entry.get("TI", ""),
        "authors": entry.get("AU", []),
        "year": entry.get("PY") or entry.get("DA", "")[:4],
        "journal": entry.get("JO", ""),
        "doi": entry.get("DO", ""),
        "url": entry.get("UR", ""),
        "keywords": entry.get("KW", []),
        "raw_entry": entry,
    }


def _looks_itr_relevant(title: str) -> bool:
    title_lc = title.lower()
    keep_keywords = [
        "image-text retrieval",
        "text-image retrieval",
        "cross-modal retrieval",
        "cross modal retrieval",
        "multimodal alignment",
        "modality-specific adaptive scaling and attention network for cross-modal retrieval",
    ]
    return any(keyword in title_lc for keyword in keep_keywords)
