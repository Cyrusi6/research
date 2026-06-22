import json
from types import SimpleNamespace

import pytest

from auto_research.cli import main
from auto_research.s0_enrichment import DeepSeekS0SemanticEnricher, S0SemanticEnrichmentError
from auto_research.utils import write_json, write_yaml


def _write_project(tmp_path):
    project = tmp_path / "proj_enrich"
    (project / "meta").mkdir(parents=True)
    (project / "intake" / "c2c").mkdir(parents=True)
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": "proj_enrich",
            "research_topic": "cross tokenizer cache",
            "status": "running",
            "current_stage": "S1_literature",
            "iteration": 1,
            "blocked_reason": None,
            "stages": {},
        },
    )
    write_yaml(project / "meta" / "project_config.yaml", {"project": {"workspace_root": str(tmp_path)}})
    bundle = {
        "schema_version": "c2c_static_intake_bundle_v1",
        "paper_chunks": [
            {"chunk_id": "paper:0001", "source_type": "paper", "source_path": "paper.md", "section": "method", "text": "Cache routing should preserve useful transferred states while avoiding noisy spans."},
            {"chunk_id": "paper:0002", "source_type": "paper", "source_path": "paper.md", "section": "experiments", "text": "OpenbookQA and MMLU expose different failure behavior."},
        ],
        "rebuttal_chunks": [
            {"chunk_id": "rebuttal:0001", "source_type": "rebuttal", "source_path": "rebuttal.md", "section": "concern", "text": "Reviewers worry that hard gates collapse coverage."}
        ],
        "code_chunks": [
            {
                "chunk_id": "code:aligner",
                "source_type": "code",
                "source_path": "rosetta/model/aligner.py",
                "path": "rosetta/model/aligner.py",
                "symbol": "align",
                "start_line": 1,
                "end_line": 20,
                "text": "def align(cache, valid_mask): return cache",
            }
        ],
        "chunk_index": {"entries": [], "counts": {"paper": 2, "rebuttal": 1, "code": 1, "total": 4}},
    }
    write_json(project / "intake" / "c2c" / "static_bundle.json", bundle)
    return project


def test_s0_enrichment_dry_run_writes_costed_sample(tmp_path) -> None:
    project = _write_project(tmp_path)

    result = DeepSeekS0SemanticEnricher(project, {"intake": {"semantic_enrichment": {"model": "deepseek-v4-flash"}}}).run(limit=3, dry_run=True)

    report = result["report"]
    assert result["status"] == "ok"
    assert report["mode"] == "dry_run"
    assert report["selected_count"] == 3
    assert report["cost_summary"]["projected_full_cost_cny"] > 0
    assert (project / "intake" / "c2c" / "semantic_enrichment_sample.json").exists()
    assert (project / "intake" / "c2c" / "semantic_enrichment_sample.jsonl").exists()
    assert {record["chunk"]["source_type"] for record in report["records"]} == {"paper", "rebuttal", "code"}


def test_s0_enrichment_requires_api_key_for_api_mode(tmp_path, monkeypatch) -> None:
    project = _write_project(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(S0SemanticEnrichmentError, match="DEEPSEEK_API_KEY"):
        DeepSeekS0SemanticEnricher(project, {}).run(limit=1, dry_run=False)


def test_s0_enrichment_api_mode_caches_records(tmp_path, monkeypatch) -> None:
    project = _write_project(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            content = json.dumps(
                {
                    "semantic_summary": "Grounded cache routing evidence.",
                    "mechanism_tags": ["cache routing"],
                    "method_claims": ["routing should preserve useful states"],
                    "failure_modes": ["coverage collapse"],
                    "implementation_relevance": "medium: mentions valid_mask path",
                    "dataset_relevance": [{"dataset": "mmlu-redux", "relevance": "medium", "reason": "reasoning benchmark"}],
                    "reviewer_risk_notes": ["avoid hard gates"],
                    "retrieval_keywords": ["utility routing", "coverage"],
                    "s1_direction_utility": "Useful for selecting a coverage-preserving direction.",
                    "s2_patch_utility": "Useful for locating routing code.",
                    "evidence_quality": "high: direct chunk evidence",
                    "code_patch_surface_notes": "aligner surface",
                }
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=80, total_tokens=180, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=100),
            )

    import auto_research.s0_enrichment as module

    monkeypatch.setattr(module, "OpenAI", FakeClient)

    first = DeepSeekS0SemanticEnricher(project, {}).run(limit=1)
    second = DeepSeekS0SemanticEnricher(project, {}).run(limit=1)

    assert len(calls) == 1
    assert first["report"]["records"][0]["cache_status"] == "miss"
    assert second["report"]["records"][0]["cache_status"] == "hit"
    assert first["report"]["records"][0]["enrichment"]["mechanism_tags"] == ["cache routing"]


def test_s0_enrichment_filters_infra_code_and_prioritizes_mechanism_chunks(tmp_path) -> None:
    project = _write_project(tmp_path)
    chunks = [
        {"chunk_id": "paper:method", "source_type": "paper", "section": "method", "text": "KV cache projection and fusion."},
        {"chunk_id": "rebuttal:risk", "source_type": "rebuttal", "text": "Reviewer concern about baseline and failure."},
        {
            "chunk_id": "code:ci",
            "source_type": "code",
            "path": ".github/workflows/fix.yml",
            "source_path": ".github/workflows/fix.yml",
            "text": "sed replacement workflow",
        },
        {
            "chunk_id": "code:aligner",
            "source_type": "code",
            "path": "rosetta/model/aligner.py",
            "source_path": "rosetta/model/aligner.py",
            "risk_tags": ["alignment_core", "runtime_path"],
            "edit_surface": "allowed_prefix",
            "text": "def align(cache, valid_mask): return cache",
        },
    ]

    report = DeepSeekS0SemanticEnricher(project, {"intake": {"semantic_enrichment": {"dry_run": True, "limit": 3}}}).enrich_chunk_list(
        chunks,
        write_artifacts=False,
    )["report"]

    selected_ids = [(record["chunk"]["chunk_id"], record["chunk"]["source_type"]) for record in report["records"]]
    assert ("code:aligner", "code") in selected_ids
    assert ("code:ci", "code") not in selected_ids
    assert report["skipped_chunks"] == 1
    code_record = next(record for record in report["records"] if record["chunk"]["source_type"] == "code")
    assert "alignment_core" in code_record["chunk"]["semantic_enrichment_priority_reasons"]


def test_s0_enrichment_filters_low_information_paper_headings(tmp_path) -> None:
    project = _write_project(tmp_path)
    chunks = [
        {"chunk_id": "paper:heading", "source_type": "paper", "section": "method", "text": "3 Method"},
        {"chunk_id": "paper:abstract", "source_type": "paper", "section": "abstract", "text": "C2C uses KV cache projection, fusion, and gating to transfer semantic information between models."},
        {"chunk_id": "rebuttal:risk", "source_type": "rebuttal", "text": "Reviewer concern: baseline fairness and ablation failure risk."},
    ]

    report = DeepSeekS0SemanticEnricher(project, {"intake": {"semantic_enrichment": {"dry_run": True, "limit": 3}}}).enrich_chunk_list(
        chunks,
        write_artifacts=False,
    )["report"]

    selected_ids = {record["chunk"]["chunk_id"] for record in report["records"]}
    assert "paper:heading" not in selected_ids
    assert "paper:abstract" in selected_ids
    assert report["skipped_chunks"] == 1


def test_s0_enrichment_applies_semantic_fields_to_chunk_lists(tmp_path) -> None:
    project = _write_project(tmp_path)
    enricher = DeepSeekS0SemanticEnricher(
        project,
        {"intake": {"semantic_enrichment": {"dry_run": True, "limit": 3, "model": "deepseek-v4-flash"}}},
    )
    paper_chunks = [{"chunk_id": "paper:0001", "source_type": "paper", "section": "abstract", "text": "KV cache projection and gating can transfer semantic information between models."}]
    rebuttal_chunks = [{"chunk_id": "rebuttal:0001", "source_type": "rebuttal", "text": "Reviewer concern about baseline fairness, failure analysis, and ablation."}]
    code_chunks = [{"chunk_id": "code:aligner", "source_type": "code", "path": "rosetta/model/aligner.py", "text": "def align(): pass"}]

    result = enricher.enrich_c2c_chunks(
        paper_chunks=paper_chunks,
        rebuttal_chunks=rebuttal_chunks,
        code_chunks=code_chunks,
        write_artifacts=False,
    )

    assert result["report"]["selected_count"] == 3
    assert result["paper_chunks"][0]["semantic_enrichment"]["schema_version"] == "s0_semantic_enrichment_sample_v1"
    assert result["rebuttal_chunks"][0]["semantic_enrichment"]["cache_status"] == "dry_run"
    assert result["code_chunks"][0]["retrieval_keywords"]


def test_s0_enrichment_repairs_malformed_json_response(tmp_path, monkeypatch) -> None:
    project = _write_project(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                content = '{"semantic_summary": "broken'
            else:
                content = json.dumps(
                    {
                        "semantic_summary": "Repaired grounded summary.",
                        "mechanism_tags": ["cache routing"],
                        "method_claims": [],
                        "failure_modes": [],
                        "implementation_relevance": "medium",
                        "dataset_relevance": [],
                        "reviewer_risk_notes": [],
                        "retrieval_keywords": ["cache"],
                        "s1_direction_utility": "medium",
                        "s2_patch_utility": "medium",
                        "evidence_quality": "medium",
                        "code_patch_surface_notes": "",
                    }
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=10),
            )

    import auto_research.s0_enrichment as module

    monkeypatch.setattr(module, "OpenAI", FakeClient)

    report = DeepSeekS0SemanticEnricher(project, {}).run(limit=1, refresh=True)["report"]

    assert len(calls) == 2
    assert report["success_count"] == 1
    assert report["failure_count"] == 0
    assert report["records"][0]["json_repaired"] is True
    assert report["records"][0]["usage"]["total_tokens"] == 30


def test_s0_enrichment_uses_compact_code_prompt_and_limits_code_input(tmp_path, monkeypatch) -> None:
    project = _write_project(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            content = json.dumps(
                {
                    "semantic_summary": "Code chunk configures cache alignment.",
                    "mechanism_tags": ["alignment"],
                    "method_claims": [],
                    "failure_modes": [],
                    "implementation_relevance": "high",
                    "reviewer_risk_notes": [],
                    "retrieval_keywords": ["valid_mask"],
                    "s1_direction_utility": "high",
                    "s2_patch_utility": "high",
                    "evidence_quality": "high",
                    "code_patch_surface_notes": "TokenAligner.__init__",
                }
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=20),
            )

    import auto_research.s0_enrichment as module

    monkeypatch.setattr(module, "OpenAI", FakeClient)
    long_code = "def align(cache, valid_mask):\n    return cache\n" * 200
    chunks = [
        {
            "chunk_id": "code:aligner",
            "source_type": "code",
            "path": "rosetta/model/aligner.py",
            "source_path": "rosetta/model/aligner.py",
            "risk_tags": ["alignment_core"],
            "edit_surface": "allowed_prefix",
            "text": long_code,
        }
    ]

    report = DeepSeekS0SemanticEnricher(project, {"intake": {"semantic_enrichment": {"code_max_input_chars": 100, "code_max_tokens": 222}}}).enrich_chunk_list(
        chunks,
        limit=1,
        refresh=True,
        write_artifacts=False,
    )["report"]

    payload = json.loads(calls[0]["messages"][1]["content"])
    assert calls[0]["max_tokens"] == 222
    assert "structure" in payload
    assert "code_excerpt" in payload
    assert len(payload["code_excerpt"]) <= 100
    assert "dataset_relevance" not in payload["required_json_schema"]
    assert "patch_surface" in payload["required_json_schema"]
    assert report["records"][0]["enrichment"]["code_patch_surface_notes"] == "TokenAligner.__init__"
    assert report["records"][0]["prompt_version"] == "deepseek_s0_code_semantic_enrichment_v2"


def test_s0_enrichment_maps_compact_code_response_to_common_fields(tmp_path, monkeypatch) -> None:
    project = _write_project(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            content = json.dumps(
                {
                    "purpose": "Initializes projector routing.",
                    "mechanism_role": "projection",
                    "patch_surface": "Projector.__init__",
                    "ablation_hooks": ["use_gate", "projector_type"],
                    "inputs_outputs": ["hidden_states", "projected_cache"],
                    "risk_flags": ["shape mismatch"],
                    "search_terms": ["projector", "gate"],
                    "s1_utility": "medium",
                    "s2_utility": "high",
                    "confidence": "high",
                }
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=20),
            )

    import auto_research.s0_enrichment as module

    monkeypatch.setattr(module, "OpenAI", FakeClient)
    report = DeepSeekS0SemanticEnricher(project, {}).enrich_chunk_list(
        [
            {
                "chunk_id": "code:projector",
                "source_type": "code",
                "path": "rosetta/model/projector.py",
                "source_path": "rosetta/model/projector.py",
                "risk_tags": ["projector_core"],
                "edit_surface": "allowed_prefix",
                "text": "class Projector:\n    def __init__(self, projector_type, use_gate=True):\n        self.use_gate = use_gate\n",
            }
        ],
        limit=1,
        refresh=True,
        write_artifacts=False,
    )["report"]

    enrichment = report["records"][0]["enrichment"]
    assert enrichment["semantic_summary"] == "Initializes projector routing."
    assert enrichment["mechanism_tags"][0] == "projection"
    assert "Projector.__init__" in enrichment["code_patch_surface_notes"]
    assert "shape mismatch" in enrichment["failure_modes"]


def test_s0_enrichment_falls_back_when_json_repair_fails(tmp_path, monkeypatch) -> None:
    project = _write_project(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"semantic_summary": "broken'))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=10),
            )

    import auto_research.s0_enrichment as module

    monkeypatch.setattr(module, "OpenAI", FakeClient)

    report = DeepSeekS0SemanticEnricher(project, {}).run(limit=1, refresh=True)["report"]

    assert report["selected_count"] == 1
    assert report["success_count"] == 1
    assert report["failure_count"] == 1
    assert report["fallback_count"] == 1
    assert report["records"][0]["cache_status"] == "fallback"
    assert "llm_json_enrichment_failed" in report["records"][0]["enrichment"]["failure_modes"]


def test_s0_enrichment_fills_empty_code_response_deterministically(tmp_path, monkeypatch) -> None:
    project = _write_project(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({})))],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=20),
            )

    import auto_research.s0_enrichment as module

    monkeypatch.setattr(module, "OpenAI", FakeClient)
    report = DeepSeekS0SemanticEnricher(project, {}).enrich_chunk_list(
        [
            {
                "chunk_id": "code:aligner-soft",
                "source_type": "code",
                "path": "rosetta/model/aligner.py",
                "source_path": "rosetta/model/aligner.py",
                "symbol": "TokenAligner._soft_alignment_confidence",
                "risk_tags": ["alignment_core"],
                "edit_surface": "allowed_prefix",
                "text": "def _soft_alignment_confidence(self, scores, valid_mask):\n    return scores.mean()\n",
            }
        ],
        limit=1,
        refresh=True,
        write_artifacts=False,
    )["report"]

    enrichment = report["records"][0]["enrichment"]
    assert enrichment["semantic_summary"]
    assert "alignment" in enrichment["mechanism_tags"]
    assert "empty_llm_code_enrichment" in enrichment["failure_modes"]
    assert "TokenAligner._soft_alignment_confidence" in enrichment["code_patch_surface_notes"]


def test_enrich_s0_cli_outputs_summary(tmp_path, monkeypatch, capsys) -> None:
    _write_project(tmp_path)

    import auto_research.config as config_module
    import auto_research.orchestrator as orchestrator_module

    config = {"project": {"workspace_root": str(tmp_path)}}
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)

    main(["enrich-s0", "--project-id", "proj_enrich", "--limit", "2", "--dry-run"])
    output = capsys.readouterr().out
    assert "S0 semantic enrichment: ok" in output
    assert "sample cost:" in output

    main(["enrich-s0", "--project-id", "proj_enrich", "--limit", "1", "--dry-run", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"]["mode"] == "dry_run"

    main(["enrich-s0", "--project-id", "proj_enrich", "--limit", "all", "--workers", "2", "--dry-run", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"]["limit"] >= payload["report"]["selected_count"]
    assert payload["report"]["workers"] == 2
