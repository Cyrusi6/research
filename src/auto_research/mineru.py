"""MinerU API client for PDF-to-Markdown intake."""

from __future__ import annotations

import os
import hashlib
import json
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

from .utils import ensure_dir, now_utc, write_json


class MinerUError(RuntimeError):
    """Raised when MinerU parsing cannot produce a usable Markdown artifact."""


@dataclass
class MinerUPdfClient:
    api_key: str | None = None
    base_url: str = "https://mineru.net/api/v4"
    model_version: str = "vlm"
    language: str = "en"
    enable_formula: bool = True
    enable_table: bool = True
    is_ocr: bool = False
    timeout_seconds: int = 900
    poll_interval_seconds: int = 5
    request_timeout_seconds: int = 60
    session: Any | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("MINERU_API_KEY")
        self.session = self.session or requests.Session()

    def parse_pdf(self, pdf_path: Path, output_dir: Path, *, data_id: str, title: str = "") -> dict[str, Any]:
        if not self.api_key:
            raise MinerUError("MINERU_API_KEY is not set; cannot parse PDF with MinerU.")
        pdf_path = pdf_path.expanduser().resolve()
        if not pdf_path.exists():
            raise MinerUError(f"PDF file does not exist: {pdf_path}")
        ensure_dir(output_dir)

        started_at = now_utc()
        batch_payload = self._request_upload_url(pdf_path, data_id=data_id)
        batch_id = str(batch_payload.get("batch_id") or "")
        file_urls = batch_payload.get("file_urls") or []
        if not batch_id or not file_urls:
            raise MinerUError("MinerU upload URL response missing batch_id or file_urls.")

        self._upload_pdf(pdf_path, str(file_urls[0]))
        extract_result = self._poll_batch(batch_id, data_id=data_id, file_name=pdf_path.name)
        full_zip_url = str(extract_result.get("full_zip_url") or "")
        if not full_zip_url:
            raise MinerUError("MinerU completed without full_zip_url.")
        markdown = self._download_full_markdown(full_zip_url)
        markdown = normalize_mineru_markdown(markdown, title=title or pdf_path.stem)
        if not markdown.strip():
            raise MinerUError("MinerU full.md is empty after normalization.")

        paper_full_path = output_dir / "paper_full.md"
        paper_full_path.write_text(markdown, encoding="utf-8")
        result = {
            "provider": "mineru",
            "schema_version": "mineru_pdf_parse_result_v1",
            "created_at": now_utc(),
            "started_at": started_at,
            "state": extract_result.get("state"),
            "batch_id": batch_id,
            "data_id": data_id,
            "file_name": pdf_path.name,
            "model_version": self.model_version,
            "client_version": "mineru_api_v4",
            "prompt_schema_version": "c2c_paper_full_markdown_v1",
            "parser_config_hash": _mineru_client_config_hash(
                {
                    "base_url": self.base_url,
                    "model_version": self.model_version,
                    "language": self.language,
                    "enable_formula": self.enable_formula,
                    "enable_table": self.enable_table,
                    "is_ocr": self.is_ocr,
                }
            ),
            "language": self.language,
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
            "is_ocr": self.is_ocr,
            "full_zip_url": full_zip_url,
            "paper_full_md_path": paper_full_path.name,
            "err_msg": extract_result.get("err_msg") or "",
            "trace": {
                "api_base_url": self.base_url,
            },
        }
        write_json(output_dir / "mineru_result.json", result)
        return result

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "*/*",
        }

    def _request_upload_url(self, pdf_path: Path, *, data_id: str) -> dict[str, Any]:
        payload = {
            "files": [
                {
                    "name": pdf_path.name,
                    "data_id": data_id,
                    "is_ocr": self.is_ocr,
                }
            ],
            "model_version": self.model_version,
            "language": self.language,
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
        }
        try:
            response = self.session.post(
                f"{self.base_url}/file-urls/batch",
                headers=self._headers,
                json=payload,
                timeout=self.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise MinerUError(f"MinerU request upload URL failed: {exc}") from exc
        return self._data_or_raise(response, "request upload URL")

    def _upload_pdf(self, pdf_path: Path, file_url: str) -> None:
        try:
            with pdf_path.open("rb") as handle:
                response = self.session.put(file_url, data=handle, timeout=self.request_timeout_seconds)
        except requests.RequestException as exc:
            raise MinerUError(f"MinerU upload failed: {exc}") from exc
        if response.status_code not in {200, 201}:
            raise MinerUError(f"MinerU upload failed with HTTP {response.status_code}.")

    def _poll_batch(self, batch_id: str, *, data_id: str, file_name: str) -> dict[str, Any]:
        deadline = time.monotonic() + max(1, int(self.timeout_seconds))
        last_result: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                response = self.session.get(
                    f"{self.base_url}/extract-results/batch/{batch_id}",
                    headers=self._headers,
                    timeout=self.request_timeout_seconds,
                )
            except requests.RequestException as exc:
                raise MinerUError(f"MinerU poll batch failed: {exc}") from exc
            data = self._data_or_raise(response, "poll batch")
            results = data.get("extract_result") or []
            result = _select_extract_result(results, data_id=data_id, file_name=file_name)
            if result:
                last_result = result
                state = str(result.get("state") or "")
                if state == "done":
                    return result
                if state == "failed":
                    raise MinerUError(f"MinerU parse failed: {result.get('err_msg') or result.get('err_code') or 'unknown error'}")
            time.sleep(max(1, int(self.poll_interval_seconds)))
        raise MinerUError(f"MinerU batch polling timed out after {self.timeout_seconds}s. Last result: {last_result}")

    def _download_full_markdown(self, full_zip_url: str) -> str:
        try:
            response = self.session.get(full_zip_url, timeout=self.request_timeout_seconds)
        except requests.RequestException as exc:
            raise MinerUError(f"MinerU result zip download failed: {exc}") from exc
        if response.status_code != 200:
            raise MinerUError(f"MinerU result zip download failed with HTTP {response.status_code}.")
        try:
            with zipfile.ZipFile(BytesIO(response.content)) as archive:
                names = archive.namelist()
                full_md_name = next((name for name in names if Path(name).name == "full.md"), "")
                if not full_md_name:
                    raise MinerUError("MinerU result zip does not contain full.md.")
                return archive.read(full_md_name).decode("utf-8", errors="replace")
        except zipfile.BadZipFile as exc:
            raise MinerUError(f"MinerU result is not a valid zip: {exc}") from exc

    @staticmethod
    def _data_or_raise(response: Any, action: str) -> dict[str, Any]:
        if response.status_code != 200:
            raise MinerUError(f"MinerU {action} failed with HTTP {response.status_code}.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise MinerUError(f"MinerU {action} returned invalid JSON.") from exc
        if payload.get("code") != 0:
            raise MinerUError(f"MinerU {action} failed: {payload.get('msg') or payload.get('code')}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise MinerUError(f"MinerU {action} response missing data object.")
        return data


def normalize_mineru_markdown(markdown: str, *, title: str) -> str:
    text = markdown.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines).strip()
    if normalized and not any(line.lstrip().startswith("#") for line in normalized.splitlines()[:12]):
        normalized = f"# {title.strip() or 'Paper'}\n\n{normalized}"
    return normalized + ("\n" if normalized else "")


def _mineru_client_config_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _select_extract_result(results: Any, *, data_id: str, file_name: str) -> dict[str, Any]:
    if not isinstance(results, list):
        return {}
    for result in results:
        if isinstance(result, dict) and result.get("data_id") == data_id:
            return result
    for result in results:
        if isinstance(result, dict) and result.get("file_name") == file_name:
            return result
    first = results[0] if results else {}
    return first if isinstance(first, dict) else {}
