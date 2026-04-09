"""Literature stage."""

from __future__ import annotations

import json
from typing import Any

from ..adapters.literature import LiteratureProvider
from ..agents.idea_review import IdeaReviewAgent
from ..importers import ConsensusImporter
from ..itr_ideas import build_itr_theme_map, build_laps_candidate_ideas, collect_consensus_entries, theme_map_markdown
from ..resources import discover_local_mm_resources
from ..utils import compact_markdown, sanitize_filename, split_sentences
from .base import AgentContext


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
            raw_ideas = build_laps_candidate_ideas(theme_map, discover_local_mm_resources(self.context.config), topic=topic)
            review_payload = IdeaReviewAgent(self.context).review(ideas=raw_ideas, theme_map=theme_map)
            ideas = review_payload["ideas"]
        else:
            ideas = self._build_ideas(topic, ingested)
        ideas_record = self.context.artifacts.write_json(
            self.stage_key,
            "ideas.json",
            ideas,
            artifact_type="ideas",
            summary=f"{len(ideas)} candidate ideas",
            source_paths=[metadata_record["path"], *( [theme_map_record["path"]] if theme_map_record else [] )],
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
