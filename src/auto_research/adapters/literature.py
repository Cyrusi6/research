"""Literature retrieval and PDF download."""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import requests

from ..utils import sanitize_filename


SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API_URL = "http://export.arxiv.org/api/query"


@dataclass
class LiteratureProvider:
    config: dict[str, Any]

    def __post_init__(self) -> None:
        literature_config = self.config.get("literature", {})
        self.timeout = literature_config.get("request_timeout_seconds", 20)
        self.max_papers = literature_config.get("max_papers", 12)
        self.arxiv_max_results = literature_config.get("arxiv_max_results", 8)
        self.api_key = literature_config.get("semantic_scholar_api_key") or ""
        self.session = requests.Session()

    def search(self, topic: str) -> list[dict[str, Any]]:
        papers = []
        for query in self._candidate_queries(topic):
            papers.extend(self.search_semantic_scholar(query))
            papers.extend(self.search_arxiv(query))
            deduped = self._deduplicate(papers)
            if len(deduped) >= 6:
                return deduped
        return self._deduplicate(papers)

    @staticmethod
    def _candidate_queries(topic: str) -> list[str]:
        topic_lc = topic.lower()
        queries = [topic]
        if "image-text retrieval" in topic_lc or ("image" in topic_lc and "text" in topic_lc and "retrieval" in topic_lc):
            queries.extend(
                [
                    "image-text retrieval",
                    "text-image retrieval",
                    "cross-modal retrieval",
                    "image-text retrieval hard negatives",
                ]
            )
        if "multimodal" in topic_lc and "retrieval" in topic_lc:
            queries.extend(["multimodal retrieval", "vision-language retrieval"])
        if "document" in topic_lc and "retrieval" in topic_lc:
            queries.extend(["visual document retrieval", "multimodal document retrieval"])
        seen = []
        for query in queries:
            if query not in seen:
                seen.append(query)
        return seen

    def search_semantic_scholar(self, query: str) -> list[dict[str, Any]]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        params = {
            "query": query,
            "limit": self.max_papers,
            "fields": ",".join(
                [
                    "title",
                    "authors",
                    "year",
                    "abstract",
                    "citationCount",
                    "externalIds",
                    "url",
                    "openAccessPdf",
                    "venue",
                ]
            ),
        }
        try:
            response = self.session.get(
                SEMANTIC_SCHOLAR_URL,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException:
            return []
        data = response.json().get("data", [])
        papers = []
        for paper in data:
            external_ids = paper.get("externalIds") or {}
            arxiv_id = external_ids.get("ArXiv")
            open_access = paper.get("openAccessPdf") or {}
            papers.append(
                {
                    "paper_id": external_ids.get("CorpusId") or arxiv_id or sanitize_filename(paper.get("title", "")),
                    "title": paper.get("title"),
                    "authors": [item.get("name", "") for item in paper.get("authors", [])],
                    "year": paper.get("year"),
                    "abstract": paper.get("abstract", ""),
                    "citation_count": paper.get("citationCount", 0),
                    "source": "semantic_scholar",
                    "venue": paper.get("venue"),
                    "pdf_url": open_access.get("url") or (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None),
                    "arxiv_id": arxiv_id,
                    "url": paper.get("url"),
                }
            )
        return papers

    def search_arxiv(self, query: str) -> list[dict[str, Any]]:
        params = {"search_query": f"all:{query}", "start": 0, "max_results": self.arxiv_max_results}
        try:
            response = self.session.get(ARXIV_API_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            return []
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = []
        for entry in root.findall("atom:entry", ns):
            entry_id = entry.findtext("atom:id", default="", namespaces=ns)
            arxiv_id = entry_id.rsplit("/", 1)[-1]
            title = html.unescape((entry.findtext("atom:title", default="", namespaces=ns) or "").strip())
            summary = html.unescape((entry.findtext("atom:summary", default="", namespaces=ns) or "").strip())
            published = entry.findtext("atom:published", default="", namespaces=ns)
            authors = [author.findtext("atom:name", default="", namespaces=ns) for author in entry.findall("atom:author", ns)]
            entries.append(
                {
                    "paper_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "year": published[:4] if published else None,
                    "abstract": summary,
                    "citation_count": 0,
                    "source": "arxiv",
                    "venue": "arXiv",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None,
                    "arxiv_id": arxiv_id,
                    "url": entry_id,
                }
            )
        return entries

    def download_pdf(self, pdf_url: str) -> bytes | None:
        try:
            response = self.session.get(pdf_url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            return None
        if "pdf" not in response.headers.get("content-type", "").lower() and not pdf_url.endswith(".pdf"):
            return None
        return response.content

    @staticmethod
    def pdf_filename(paper: dict[str, Any]) -> str:
        author = sanitize_filename((paper.get("authors") or ["unknown"])[0].split()[-1].lower())
        year = str(paper.get("year") or "unknown")
        title = sanitize_filename(paper.get("title") or "paper")
        paper_id = sanitize_filename(paper.get("paper_id") or title)
        return f"{author}_{year}_{title}_{paper_id}.pdf"

    @staticmethod
    def _deduplicate(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = {}
        for paper in papers:
            key = re.sub(r"\W+", "", (paper.get("title") or "").lower())
            if not key:
                continue
            if key not in seen or paper.get("citation_count", 0) > seen[key].get("citation_count", 0):
                seen[key] = paper
        ordered = sorted(seen.values(), key=lambda item: (-int(item.get("citation_count") or 0), item.get("title") or ""))
        return ordered
