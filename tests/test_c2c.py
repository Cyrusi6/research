import json
import time
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import yaml

import auto_research.config as config_module
import auto_research.agents.literature as literature_module
import auto_research.agents.experiment as experiment_module
import auto_research.orchestrator as orchestrator_module
from auto_research.adapters.runner import ExperimentRunner
from auto_research.agents.debate import MultiAgentReasoningService
from auto_research.agents.experiment import ExperimentAgent
from auto_research.agents.plan import PlanAgent
from auto_research.agents.base import AgentContext
from auto_research.failure_log import build_c2c_feedback_bundle, load_c2c_feedback_bundle
from auto_research.artifacts import ArtifactManager
from auto_research.c2c import C2CAdapter, C2CPatchGuard, c2c_idea_novelty_report, default_c2c_ideas
from auto_research.validators.s2_gate import S2GateValidator
from auto_research.code_patch import CodePatchAgent, CodexPatchBackend, DynamicEditPolicy, FrozenPatchGuard
from auto_research.llm import ModelClient
from auto_research.mineru import MinerUError, MinerUPdfClient
from auto_research.code_intake import retrieve_code_chunks
from auto_research.orchestrator import Orchestrator
from auto_research.utils import sha256_file
from auto_research.workspace import init_workspace


def _base_config(tmp_path: Path, *, simulate: bool = True) -> dict:
    return {
        "project": {"workspace_root": str(tmp_path), "target_venue": "TestConf", "language": "en"},
        "llm": {"provider": "openai", "use_real_api": False, "model": "mock"},
        "literature": {"download_pdfs": False, "request_timeout_seconds": 1, "max_papers": 0, "arxiv_max_results": 0},
        "experiment": {"simulate": simulate, "random_seeds": [42]},
        "writing": {"claim_verification": {"enabled": True, "min_pass_rate": 0.8}, "require_compile": False},
        "review": {"pass_threshold": 7.0, "max_iterations": 1},
        "orchestration": {"judge_max_retries": 1, "auto_mode": True},
    }


def _fake_c2c_repo(tmp_path: Path) -> Path:
    root = tmp_path / "C2C"
    for rel in [
        "rosetta/model",
        "script/train",
        "script/evaluation",
        "recipe/train_recipe",
        "recipe/eval_recipe",
        "local/final_results/route1_alignment_v22/small_loop_summary",
        "local/final_results/demo/mmlu-redux",
        "local/tmp/train_recipes/route1_alignment_v22",
        "local/tmp/eval_configs/route1_alignment_v22",
        "test",
        "wandb/offline-run",
        "local/checkpoints/demo",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    for name in ["README.md", "RUNBOOK.md", "C2C_跨Tokenizer柔性对齐改进方向研究备忘.md"]:
        (root / name).write_text(f"# {name}\nC2C cross-tokenizer cache communication.\n", encoding="utf-8")
    (root / "local/final_results/EXPERIMENT_RECORD.md").write_text("E0 baseline\nE20 v2.2 token_mlp\n", encoding="utf-8")
    (root / "rosetta/model/aligner.py").write_text("VALUE = 'aligner'\n", encoding="utf-8")
    (root / "rosetta/model/projector.py").write_text("VALUE = 'projector'\n", encoding="utf-8")
    (root / "rosetta/model/wrapper.py").write_text("VALUE = 'wrapper'\n", encoding="utf-8")
    (root / "script/train/SFT_train.py").write_text("print('train')\n", encoding="utf-8")
    (root / "script/evaluation/unified_evaluator.py").write_text("print('eval')\n", encoding="utf-8")
    (root / "test/test_aligner_span_overlap.py").write_text("def test_span():\n    assert True\n", encoding="utf-8")
    (root / "recipe/train_recipe/C2C_0.6+0.5.json").write_text(
        json.dumps({"output": {}, "data": {"kwargs": {}}, "training": {}, "model": {}}),
        encoding="utf-8",
    )
    (root / "recipe/eval_recipe/unified_eval.yaml").write_text(
        yaml.safe_dump({"model": {"rosetta_config": {}}, "output": {}, "eval": {"dataset": "mmlu-redux"}}),
        encoding="utf-8",
    )
    scores = root / "local/final_results/route1_alignment_v22/small_loop_summary/route1_v22_small_loop_scores.csv"
    scores.write_text(
        "\n".join(
            [
                "method,receiver,sharer,alignment_strategy,confidence_gate,train_samples,final_train_loss,mid_eval_loss,final_eval_loss,mmlu_redux,ai2_arc_challenge,openbookqa,mean,delta_mean_vs_v21_entropy050",
                "v2.2_token_mlp_entropy050,Qwen,Tiny,soft_span_overlap_v2,token_mlp,2048,0.37,0.17,0.16,47.07,54.78,50.60,50.82,1.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "model": "Rosetta",
        "dataset": "mmlu-redux",
        "answer_method": "generate",
        "overall_accuracy": 0.4707,
    }
    (root / "local/final_results/demo/mmlu-redux/Rosetta_mmlu-redux_generate_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (root / "wandb/offline-run/skip.txt").write_text("skip", encoding="utf-8")
    (root / "local/checkpoints/demo/model.pth").write_text("skip", encoding="utf-8")
    return root


def _fake_git_c2c_repo(tmp_path: Path) -> Path:
    repo = _fake_c2c_repo(tmp_path)
    (repo / ".git").mkdir(parents=True, exist_ok=True)
    return repo


def _s1_codex_direction_payload() -> dict:
    return {
        "schema_version": "c2c_s1_codex_direction_v1",
        "status": "ok",
        "evidence_requests": [
            {
                "query": "utility cache routing implementation surface",
                "source_type": "code",
                "desired_evidence": "implementation",
                "why_needed": "S2 needs a bounded place to turn the direction into a patch.",
            }
        ],
        "evidence_bundle": {
            "items": [
                {
                    "chunk_id": "code:rosetta/model/aligner.py",
                    "source_path": "intake/c2c/code_chunks.jsonl",
                    "source_type": "code",
                    "summary": "Alignment and cache transfer surfaces are localized in rosetta/model files.",
                    "supports": ["utility_predicted_cache_routing"],
                    "risks": [],
                },
                {
                    "chunk_id": "feedback:coverage_collapse",
                    "source_path": "intake/c2c/negative_result_memory.json",
                    "source_type": "failure_feedback",
                    "summary": "Prior hard-gate style changes risk all-dataset collapse.",
                    "supports": [],
                    "risks": ["hard_gate_stack"],
                },
            ]
        },
        "direction_decision": {
            "direction_id": "utility_predicted_cache_routing",
            "mechanism_direction": "Utility Predicted Cache Routing",
            "mechanism_type": "utility_predicted_cache_routing",
            "core_hypothesis": "Predict downstream utility for transferred cache states and let S2 explore soft routing mechanisms that preserve baseline coverage.",
            "allowed_variants": ["soft residual utility scaling", "coverage-preserving utility modulation"],
            "forbidden_patterns": ["extra hard accept/reject gate", "evaluator changes"],
            "target_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
            "failure_focus": ["dataset-level coverage collapse", "mmlu-redux regression"],
            "expected_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
            "verification_commands": ["py_compile", "small2048_train", "three_dataset_eval"],
            "rationale": "The direction is mechanism-level and leaves concrete patch variants to S2.",
        },
        "selected_ideas": [
            {
                "id": "utility_predicted_cache_routing",
                "title": "Utility Predicted Cache Routing",
                "selected": True,
                "hypothesis": "Predict downstream utility for transferred cache states and route them without reducing baseline transfer coverage.",
                "novelty_score": 7,
                "feasibility_score": 7,
                "mechanism_type": "utility_predicted_cache_routing",
                "description": "High-level S1 direction only; S2 will generate concrete implementation candidates.",
                "motivation": "Baseline transfer lacks a downstream utility signal and previous failures warn against hard gating.",
                "reviewer_risk_response": "Track transfer coverage and per-dataset regressions; forbid evaluator edits and hard-gate stacking.",
                "expected_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
                "verification_commands": ["py_compile", "small2048_train", "three_dataset_eval"],
                "evidence_refs": [{"source_type": "code", "source_label": "code:rosetta/model/aligner.py", "claim": "bounded implementation surface"}],
                "counterevidence_refs": [{"source_type": "failure_feedback", "source_label": "feedback:coverage_collapse", "claim": "avoid hard gate collapse"}],
                "code_refs": [{"source_type": "code", "source_label": "rosetta/model/aligner.py", "claim": "alignment/cache routing surface"}],
                "s1_allowed_variants": ["soft residual utility scaling", "coverage-preserving utility modulation"],
                "s1_forbidden_patterns": ["extra hard accept/reject gate", "evaluator changes"],
            }
        ],
        "negative_constraints": {
            "reviewer_concerns": ["failure_modes_ood", "coverage collapse"],
            "forbidden_idea_ids": ["hard_gate_stack"],
            "forbidden_patterns": ["extra hard accept/reject gate", "evaluator changes"],
            "failure_feedback_rules": ["Use method-level failures only; ignore S2.5 coding noise in S1."],
        },
        "decision_chain": {
            "evidence": ["code:rosetta/model/aligner.py"],
            "counterevidence": ["feedback:coverage_collapse"],
            "conclusion": "Use utility-predicted cache routing as the S1 direction and let S2 choose concrete variants.",
        },
    }


def test_init_c2c_creates_snapshot_and_config(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.txt"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_text("paper text", encoding="utf-8")
    ref_rebuttal.write_text("review text", encoding="utf-8")
    config = _base_config(tmp_path / "workspace")
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)

    project_id = Orchestrator().init_c2c_project(
        "cross tokenizer cache",
        target_repo=source_repo,
        ref_paper=ref_paper,
        ref_rebuttal=ref_rebuttal,
        env_python=Path("/usr/bin/python3"),
        project_id="proj_c2c",
        simulate=True,
    )

    root = tmp_path / "workspace" / project_id
    project_config = yaml.safe_load((root / "meta/project_config.yaml").read_text(encoding="utf-8"))
    assert project_config["c2c"]["enabled"] is True
    assert (root / "external/c2c_snapshot/rosetta/model/aligner.py").exists()
    assert not (root / "external/c2c_snapshot/wandb").exists()
    assert not (root / "external/c2c_snapshot/local/checkpoints").exists()
    assert (root / "experiment/c2c/repo_snapshot_manifest.json").exists()
    manifest = json.loads((root / "experiment/c2c/repo_snapshot_manifest.json").read_text(encoding="utf-8"))
    assert "source_git_commit" in manifest


def test_c2c_importers_parse_refs_and_historical_results(tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.txt"
    ref_rebuttal = tmp_path / "rebuttal.json"
    ref_paper.write_text(
        "Abstract\npaper method core for KV cache tokenizer communication.\n"
        "Method\nThe alignment method shares cache states across tokenizer boundaries.\n"
        "Experiments\nThe benchmark reports baseline accuracy and ablation results.\n"
        "References\n[1] unrelated prior work.\n",
        encoding="utf-8",
    )
    ref_rebuttal.write_text(
        json.dumps({"review": "needs stronger tokenizer mismatch evidence and baseline fairness discussion"}),
        encoding="utf-8",
    )
    paths = init_workspace(_base_config(tmp_path / "workspace"), "topic", project_id="proj", simulate=True)
    config_patch = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(source_repo),
            "ref_paper": str(ref_paper),
            "ref_rebuttal": str(ref_rebuttal),
            "env_python": "/usr/bin/python3",
        }
    }
    (paths.root / "meta/project_config.yaml").write_text(yaml.safe_dump(config_patch), encoding="utf-8")

    adapter = C2CAdapter(paths.root, config_patch)
    refs = adapter.import_reference_materials()
    history = adapter.import_historical_results()
    baseline = adapter.baseline_evidence(history)
    repo_manifest = adapter.build_repo_manifest()
    repo_card = adapter.build_repo_card(repo_manifest, history)
    paper_cards = adapter.build_paper_cards(refs["cards"])
    paper_chunks = adapter.build_paper_chunks(refs["cards"])
    bibliography = adapter.build_bibliography_cards(refs["cards"])
    rebuttal_matrix = adapter.build_rebuttal_concern_matrix(refs["cards"])
    rebuttal_chunks = adapter.build_rebuttal_chunks(refs["cards"])
    code_cards = adapter.build_code_cards(repo_manifest)
    code_intake = adapter.build_code_intake()
    code_chunks = code_intake.chunks
    chunk_index = adapter.build_chunk_index(
        paper_chunks=paper_chunks,
        rebuttal_chunks=rebuttal_chunks,
        code_chunks=code_chunks,
    )
    negative_memory = adapter.build_negative_result_memory(history, baseline)
    retrieval_plan = adapter.build_research_retrieval_plan(
        topic="cross tokenizer cache",
        repo_card=repo_card,
        paper_cards=paper_cards,
        paper_chunks=paper_chunks,
        rebuttal_matrix=rebuttal_matrix,
        rebuttal_chunks=rebuttal_chunks,
        code_cards=code_cards,
        code_chunks=code_chunks,
        negative_memory=negative_memory,
        baseline=baseline,
    )
    result_ledger = adapter.build_result_ledger_csv(history, baseline)

    assert refs["status"] == "ok"
    assert len(refs["cards"]) == 2
    assert history["counts"]["small_loop_rows"] == 1
    assert baseline["mean"] == 50.06
    assert repo_card["baseline"]["name"] == "paper_original_rosetta_fuser"
    assert len(paper_cards) == 1
    assert paper_chunks
    assert paper_chunks[0]["keywords"]
    assert "References" not in paper_chunks[-1]["text"]
    assert bibliography[0]["entry_count"] == 1
    assert rebuttal_chunks
    assert rebuttal_chunks[0]["keywords"]
    assert code_cards
    assert code_chunks
    assert code_chunks[0]["keywords"]
    assert code_intake.file_manifest["files"]
    assert code_intake.repo_map["counts"]["chunks"] == len(code_chunks)
    assert code_intake.repo_map["counts"]["symbols"] == len(code_intake.symbols)
    assert code_intake.report["counts"]["chunks"] == len(code_chunks)
    assert "surfaces" in code_intake.surface_map
    assert code_intake.retrieval_index["default_queries"]
    assert any(chunk["edit_surface"] in {"allowed", "allowed_prefix"} for chunk in code_chunks)
    assert any(edge["edge_type"] == "same_file_neighbor" for edge in code_intake.edges)
    assert chunk_index["counts"]["paper"] == len(paper_chunks)
    assert chunk_index["counts"]["rebuttal"] == len(rebuttal_chunks)
    assert chunk_index["counts"]["code"] == len(code_chunks)
    assert chunk_index["entries"][0]["text_preview"]
    assert chunk_index["entries"][0]["keywords"]
    assert retrieval_plan["paper_targets"]
    assert retrieval_plan["rebuttal_targets"]
    assert retrieval_plan["code_targets"]
    assert retrieval_plan["code_symbols"]
    assert retrieval_plan["code_symbols"][0]["path"]
    assert retrieval_plan["questions"]
    assert "method,kind,route_family" in result_ledger
    assert "baseline_fairness" in {item["concern_id"] for item in rebuttal_matrix["matrix"]}
    assert rebuttal_matrix["structured_concerns"][0]["source_snippet"]
    assert rebuttal_matrix["structured_concerns"][0]["experiment_implication"]
    assert rebuttal_matrix["structured_concerns"][0]["next_round_constraint"]
    assert "blocked_idea_patterns" in negative_memory


def test_tree_sitter_code_intake_builds_symbol_chunks_and_edges(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    aligner = repo / "rosetta/model/aligner.py"
    aligner.write_text(
        "\n".join(
            [
                "from rosetta.model.projector import Projector",
                "",
                "class CacheRouter:",
                "    def __init__(self, cfg):",
                "        self.cfg = cfg",
                "        self.projector = Projector()",
                "",
                "    def route(self, hidden, valid_mask):",
                "        gate = self.cfg.get('confidence_gate')",
                "        if valid_mask is None:",
                "            return hidden",
                "        return self.projector(hidden) * gate",
                "",
                "def build_router(cfg):",
                "    return CacheRouter(cfg)",
                "",
                "def run_route(router, hidden, valid_mask):",
                "    return router.route(hidden, valid_mask)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "test/test_cache_router.py").write_text(
        "from rosetta.model.aligner import CacheRouter\n\n"
        "def test_cache_router_route():\n"
        "    router = CacheRouter({'confidence_gate': 1.0})\n"
        "    assert router.route(1, True) == 1\n",
        encoding="utf-8",
    )
    (repo / "recipe/train_recipe/C2C_0.6+0.5.json").write_text(
        json.dumps({"alignment": {"confidence_gate": 1.0}, "output": {}, "data": {"kwargs": {}}}),
        encoding="utf-8",
    )
    paths = init_workspace(_base_config(tmp_path / "workspace"), "topic", project_id="proj_intake", simulate=True)
    config_patch = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "ref_paper": str(tmp_path / "missing_paper.txt"),
            "ref_rebuttal": str(tmp_path / "missing_rebuttal.txt"),
            "env_python": "/usr/bin/python3",
        }
    }
    adapter = C2CAdapter(paths.root, config_patch)

    intake = adapter.build_code_intake()

    symbols = {item["symbol"]: item for item in intake.symbols}
    assert "CacheRouter" in symbols
    assert "CacheRouter.route" in symbols
    assert "build_router" in symbols
    route_chunk = next(chunk for chunk in intake.chunks if chunk["symbol"] == "CacheRouter.route")
    assert route_chunk["node_type"] == "function_definition"
    assert route_chunk["edit_surface"] == "allowed"
    assert "valid_mask" in route_chunk["references"]
    assert "confidence_gate" in route_chunk["config_keys"]
    assert "alignment_core" in route_chunk["risk_tags"]
    assert any(edge["edge_type"] == "contains" and edge["dst"].endswith("CacheRouter.route") for edge in intake.edges)
    assert any(edge["edge_type"] == "calls" and edge["dst"] == "self.cfg.get" for edge in intake.edges)
    assert any(edge["edge_type"] == "resolved_call" and edge["call"] == "CacheRouter" for edge in intake.edges)
    assert any(edge["edge_type"] == "config_key_defined_in" and edge["config_key"] == "confidence_gate" for edge in intake.edges)
    assert any(edge["edge_type"] == "tests_symbol" for edge in intake.edges)
    assert intake.file_manifest["schema_version"] == "code_intake_v1"
    assert intake.repo_map["counts"]["symbols"] == len(intake.symbols)
    assert intake.report["counts"]["chunks_with_config_keys"] >= 1
    assert intake.report["cache"]["counts"]["miss"] > 0
    assert not intake.report["coverage"]["missing_allowed_files"]
    alignment_surface = intake.surface_map["surfaces"]["alignment_core"]
    assert any(item["symbol"] == "CacheRouter.route" for item in alignment_surface)
    retrieved = retrieve_code_chunks(query="confidence gate valid_mask routing", chunks=intake.chunks, top_k=3)
    assert retrieved[0]["symbol"] == "CacheRouter.route"
    assert any("confidence" in reason or "valid_mask" in reason for reason in retrieved[0]["match_reasons"])

    second_intake = adapter.build_code_intake()
    assert second_intake.report["cache"]["counts"]["hit"] >= intake.report["cache"]["counts"]["miss"]


class _FakeMinerUResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, content: bytes = b""):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content

    def json(self) -> dict:
        return self._payload


class _FakeMinerUSession:
    def __init__(self, zip_content: bytes):
        self.zip_content = zip_content
        self.requests: list[dict] = []

    def post(self, url: str, *, headers=None, json=None, timeout=None):
        self.requests.append({"method": "POST", "url": url, "headers": headers, "json": json, "timeout": timeout})
        return _FakeMinerUResponse(
            payload={
                "code": 0,
                "msg": "ok",
                "data": {"batch_id": "batch-1", "file_urls": ["https://upload.example/file.pdf"]},
            }
        )

    def put(self, url: str, *, data=None, timeout=None):
        self.requests.append({"method": "PUT", "url": url, "timeout": timeout})
        return _FakeMinerUResponse(status_code=200)

    def get(self, url: str, *, headers=None, timeout=None):
        self.requests.append({"method": "GET", "url": url, "headers": headers, "timeout": timeout})
        if "extract-results" in url:
            return _FakeMinerUResponse(
                payload={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "batch_id": "batch-1",
                        "extract_result": [
                            {
                                "file_name": "paper.pdf",
                                "data_id": "paper-id",
                                "state": "done",
                                "full_zip_url": "https://download.example/result.zip",
                            }
                        ],
                    },
                }
            )
        return _FakeMinerUResponse(content=self.zip_content)


def _zip_with_full_md(markdown: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nested/full.md", markdown)
    return buffer.getvalue()


def test_mineru_pdf_client_writes_paper_full_without_leaking_key(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    session = _FakeMinerUSession(_zip_with_full_md("# Title\n\n## Method\n\n$E=mc^2$\n"))
    client = MinerUPdfClient(
        api_key="secret-token",
        session=session,
        poll_interval_seconds=1,
        timeout_seconds=5,
    )

    result = client.parse_pdf(pdf_path, tmp_path / "out", data_id="paper-id", title="Fallback Title")

    paper_full = tmp_path / "out" / "paper_full.md"
    mineru_result = tmp_path / "out" / "mineru_result.json"
    assert result["state"] == "done"
    assert paper_full.exists()
    assert "## Method" in paper_full.read_text(encoding="utf-8")
    assert "secret-token" not in mineru_result.read_text(encoding="utf-8")
    assert session.requests[0]["headers"]["Authorization"] == "Bearer secret-token"


def test_mineru_pdf_client_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MINERU_API_KEY", raising=False)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    client = MinerUPdfClient(api_key="", session=_FakeMinerUSession(_zip_with_full_md("# T\n")))

    try:
        client.parse_pdf(pdf_path, tmp_path / "out", data_id="paper-id")
    except MinerUError as exc:
        assert "MINERU_API_KEY" in str(exc)
    else:
        raise AssertionError("MinerUPdfClient should fail without an API key")


def test_c2c_pdf_ref_uses_mineru_paper_full(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.pdf"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_bytes(b"%PDF-1.4 fake")
    ref_rebuttal.write_text("review text", encoding="utf-8")
    paths = init_workspace(_base_config(tmp_path / "workspace"), "topic", project_id="proj_pdf", simulate=True)
    config_patch = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(source_repo),
            "ref_paper": str(ref_paper),
            "ref_rebuttal": str(ref_rebuttal),
            "env_python": "/usr/bin/python3",
            "pdf_ingest": {"provider": "mineru"},
        }
    }
    (paths.root / "meta/project_config.yaml").write_text(yaml.safe_dump(config_patch), encoding="utf-8")

    def fake_parse(self, pdf_path: Path, output_dir: Path, *, data_id: str, title: str = ""):
        del self, pdf_path, data_id, title
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "paper_full.md").write_text("# Paper\n\n## Method\n\n$$x+y$$\n", encoding="utf-8")
        result = {"provider": "mineru", "state": "done", "paper_full_md_path": "paper_full.md"}
        (output_dir / "mineru_result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr("auto_research.c2c.MinerUPdfClient.parse_pdf", fake_parse)

    refs = C2CAdapter(paths.root, config_patch).import_reference_materials()

    assert refs["status"] == "ok"
    paper_card = next(card for card in refs["cards"] if card["kind"] == "ref_paper")
    assert paper_card["parser"] == "mineru"
    assert paper_card["paper_full_md_path"].endswith("paper_full.md")
    assert "## Method" in paper_card["text"]
    assert refs["paper_full_manifest"][0]["paper_full_md_path"] == paper_card["paper_full_md_path"]


def test_c2c_pdf_ref_reuses_mineru_sha_cache(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.pdf"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_bytes(b"%PDF-1.4 fake cache")
    ref_rebuttal.write_text("review text", encoding="utf-8")
    paths = init_workspace(_base_config(tmp_path / "workspace"), "topic", project_id="proj_pdf_cache", simulate=True)
    config_patch = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(source_repo),
            "ref_paper": str(ref_paper),
            "ref_rebuttal": str(ref_rebuttal),
            "env_python": "/usr/bin/python3",
            "pdf_ingest": {"provider": "mineru"},
        }
    }
    calls = {"count": 0}

    def fake_parse(self, pdf_path: Path, output_dir: Path, *, data_id: str, title: str = ""):
        del self, pdf_path, data_id, title
        calls["count"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "paper_full.md").write_text("# Cached Paper\n\n## Method\n\ncache text\n", encoding="utf-8")
        result = {"provider": "mineru", "state": "done", "paper_full_md_path": "paper_full.md"}
        (output_dir / "mineru_result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr("auto_research.c2c.MinerUPdfClient.parse_pdf", fake_parse)
    first = C2CAdapter(paths.root, config_patch).import_reference_materials()

    def fail_parse(self, pdf_path: Path, output_dir: Path, *, data_id: str, title: str = ""):
        del self, pdf_path, output_dir, data_id, title
        raise AssertionError("MinerU API should not be called when sha cache is available")

    monkeypatch.setattr("auto_research.c2c.MinerUPdfClient.parse_pdf", fail_parse)
    second = C2CAdapter(paths.root, config_patch).import_reference_materials()

    assert calls["count"] == 1
    assert first["paper_full_manifest"][0]["cache_status"] == "miss"
    assert second["paper_full_manifest"][0]["cache_status"] in {"local_hit", "sha_hit"}
    assert "Cached Paper" in second["cards"][0]["text"]


def test_c2c_strong_reference_comparison_is_s3_only(tmp_path: Path) -> None:
    adapter = C2CAdapter(
        tmp_path,
        {
            "c2c": {
                "strong_references": [
                    {
                        "name": "strong_local_v22",
                        "mean": 50.82,
                        "datasets": {"mmlu-redux": 47.07, "ai2-arc": 54.78, "openbookqa": 50.60},
                        "visible_to_ideation": False,
                        "reference_role": "s3_strong_reference_only",
                    },
                    {
                        "name": "ideation_visible_ref",
                        "mean": 99.0,
                        "datasets": {"mmlu-redux": 99.0},
                        "visible_to_ideation": True,
                    },
                ]
            }
        },
    )
    best = {
        "id": "winner",
        "metrics": {"mean": 51.0, "datasets": {"mmlu-redux": 48.0, "ai2-arc": 55.0, "openbookqa": 50.0}},
    }

    comparisons = ExperimentAgent._c2c_strong_reference_comparisons(best, adapter)

    assert [item["name"] for item in comparisons] == ["strong_local_v22"]
    assert comparisons[0]["delta_vs_reference"] == 0.18
    assert comparisons[0]["used_for_acceptance"] is False
    assert comparisons[0]["visible_to_ideation"] is False


def test_c2c_patch_guard_rejects_out_of_scope_and_applies_allowed(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    guard = C2CPatchGuard(["rosetta/model/aligner.py"], ["recipe/", "local/auto_research_runs/"])

    rejected = guard.apply_edits(repo, [{"path": "script/train/SFT_train.py", "old": "train", "new": "patch"}])
    applied = guard.apply_edits(repo, [{"path": "rosetta/model/aligner.py", "old": "aligner", "new": "patched"}])

    assert rejected["status"] == "rejected"
    assert applied["status"] == "applied"
    assert "patched" in (repo / "rosetta/model/aligner.py").read_text(encoding="utf-8")


def test_dynamic_edit_policy_allows_expected_scope_and_rejects_forbidden_paths(tmp_path: Path) -> None:
    policy = DynamicEditPolicy.from_config()

    allowed = [
        "rosetta/model/aligner.py",
        "script/train/SFT_train.py",
        "recipe/train_recipe/demo.json",
        "recipe/eval_recipe/demo.yaml",
        "test/test_aligner_span_overlap.py",
        "tests/test_patch.py",
        "pyproject.toml",
        "requirements-dev.txt",
    ]
    rejected = [
        "/tmp/outside.py",
        "../outside.py",
        "local/final_results/old.py",
        "local/checkpoints/model.py",
        "data/cache.py",
        "datasets/mmlu.py",
        "models/model.py",
        "rosetta/model/weights.bin",
        "foo/requirements-dev.txt",
    ]

    assert all(policy.allowed(path) for path in allowed)
    assert not any(policy.allowed(path) for path in rejected)

    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escape.py").write_text("VALUE = 1\n", encoding="utf-8")
    repo.mkdir()
    (repo / "rosetta").symlink_to(outside, target_is_directory=True)

    assert not policy.allowed("rosetta/escape.py", repo_root=repo)


def test_frozen_patch_guard_requires_sha_and_restores_added_files(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    policy = DynamicEditPolicy.from_config()
    guard = FrozenPatchGuard(policy)
    aligner = repo / "rosetta/model/aligner.py"
    original = aligner.read_text(encoding="utf-8")
    added = repo / "test/test_added_patch.py"

    result = guard.apply(
        repo,
        {
            "operations": [
                {
                    "op": "replace_file",
                    "path": "rosetta/model/aligner.py",
                    "old_sha256": sha256_file(aligner),
                    "new": "VALUE = 'patched'\n",
                },
                {
                    "op": "add_file",
                    "path": "test/test_added_patch.py",
                    "new": "def test_added_patch():\n    assert True\n",
                },
            ]
        },
    )

    assert result["status"] == "applied"
    assert "patched" in aligner.read_text(encoding="utf-8")
    assert added.exists()

    guard.restore(repo, result["restore_state"])

    assert aligner.read_text(encoding="utf-8") == original
    assert not added.exists()

    missing_sha = guard.apply(
        repo,
        {
            "operations": [
                {"op": "replace_file", "path": "rosetta/model/aligner.py", "new": "VALUE = 'bad'\n"}
            ]
        },
    )
    bad_sha = guard.apply(
        repo,
        {
            "operations": [
                {
                    "op": "replace_file",
                    "path": "rosetta/model/aligner.py",
                    "old_sha256": "not-the-current-sha",
                    "new": "VALUE = 'bad'\n",
                }
            ]
        },
    )
    forbidden = guard.apply(
        repo,
        {
            "operations": [
                {"op": "add_file", "path": "local/final_results/old.py", "new": "VALUE = 'bad'\n"}
            ]
        },
    )

    assert missing_sha["status"] == "rejected"
    assert bad_sha["status"] == "rejected"
    assert forbidden["status"] == "rejected"
    assert aligner.read_text(encoding="utf-8") == original


def _code_patch_test_config(workspace_root: Path, repo: Path, *, require_targeted_tests: bool = False) -> dict:
    config = _base_config(workspace_root, simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {"enabled": False},
        },
    }
    config["code_patch"] = {
        "enabled": True,
        "backend": "mock_codex",
        "timeout_seconds": 1800,
        "max_candidates": 3,
        "variants_per_candidate": 1,
        "validation": {
            "require_py_compile": True,
            "require_targeted_tests": require_targeted_tests,
            "mechanism_self_review": {"enabled": False},
        },
    }
    return config


def test_code_patch_agent_generates_artifacts_from_temp_repo_without_polluting_source(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class MockBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            assert implementation_contract["candidate_id"] == "idea_patch"
            assert "implementation_targets" in implementation_contract
            assert edit_policy.allowed("script/train/SFT_train.py", repo_root=temp_repo)
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'patched aligner'\n", encoding="utf-8")
            (temp_repo / "script/train/SFT_train.py").write_text("print('patched train')\n", encoding="utf-8")
            (temp_repo / ".pytest_cache/v/cache").mkdir(parents=True)
            (temp_repo / ".pytest_cache/v/cache/nodeids").write_text("[]\n", encoding="utf-8")
            (temp_repo / "rosetta/model/__pycache__").mkdir(parents=True)
            (temp_repo / "rosetta/model/__pycache__/aligner.cpython-310.pyc").write_bytes(b"cache")
            (temp_repo / ".coverage").write_text("coverage-data\n", encoding="utf-8")
            (temp_repo / "htmlcov").mkdir()
            (temp_repo / "htmlcov/index.html").write_text("<html></html>\n", encoding="utf-8")
            (temp_repo / "test/test_patch_backend.py").write_text(
                "def test_patch_backend():\n    assert True\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Implemented a multi-file patch in the temporary repo."}

    ideas = [{"id": "idea_patch", "title": "Patch Idea", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=MockBackend()).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert ideas[0]["code_patch"]["has_executable_change"] is True
    assert set(ideas[0]["code_patch"]["changed_files"]) == {
        "rosetta/model/aligner.py",
        "script/train/SFT_train.py",
        "test/test_patch_backend.py",
    }
    assert (paths.root / "plan/code_patches/idea_patch/patch.json").exists()
    assert (paths.root / "plan/code_patches/idea_patch/patch.diff").exists()
    assert (paths.root / "plan/code_patches/idea_patch/rationale.md").exists()
    assert (paths.root / "plan/code_patches/idea_patch/validation.json").exists()
    assert (paths.root / "plan/code_patches/idea_patch/implementation_contract.json").exists()
    assert (paths.root / "plan/code_patches/idea_patch/codex_prompt.md").exists()
    assert (paths.root / "plan/code_patches/patch_manifest.json").exists()
    prompt = (paths.root / "plan/code_patches/idea_patch/codex_prompt.md").read_text(encoding="utf-8")
    assert "Implementation contract" in prompt
    assert "aligner" in (repo / "rosetta/model/aligner.py").read_text(encoding="utf-8")
    assert "patched" not in (repo / "rosetta/model/aligner.py").read_text(encoding="utf-8")
    assert "patched" not in (repo / "script/train/SFT_train.py").read_text(encoding="utf-8")


def test_code_patch_delta_ignores_c2c_generated_runtime_artifacts(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_patch_runtime_artifacts", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class RuntimeArtifactBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'patched aligner'\n", encoding="utf-8")
            baseline_final = temp_repo / "local/auto_research_runs/proxy_baseline/checkpoints/final"
            baseline_final.mkdir(parents=True, exist_ok=True)
            (baseline_final / "projector_0.json").write_text('{"generated": true}\n', encoding="utf-8")
            baseline_results = temp_repo / "local/auto_research_runs/proxy_baseline/results/mmlu-redux"
            baseline_results.mkdir(parents=True, exist_ok=True)
            (baseline_results / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                '{"overall_accuracy": 0.5}\n',
                encoding="utf-8",
            )
            candidate_root = temp_repo / "local/auto_research_runs/idea_patch"
            candidate_root.mkdir(parents=True, exist_ok=True)
            (candidate_root / "train_recipe.json").write_text('{"generated": true}\n', encoding="utf-8")
            (candidate_root / "eval_mmlu-redux.yaml").write_text("generated: true\n", encoding="utf-8")
            (candidate_root / "run_state.json").write_text('{"generated": true}\n', encoding="utf-8")
            candidate_final = temp_repo / "local/auto_research_runs/idea_patch/checkpoints/final"
            candidate_final.mkdir(parents=True, exist_ok=True)
            (candidate_final / "adapter.bin").write_bytes(b"generated")
            return {"status": "ok", "rationale": "Patched model code and generated runtime artifacts."}

    ideas = [{"id": "idea_patch", "title": "Patch Idea", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=RuntimeArtifactBackend()).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["changed_files"] == ["rosetta/model/aligner.py"]
    patch = json.loads((paths.root / "plan/code_patches/idea_patch/patch.json").read_text(encoding="utf-8"))
    assert [operation["path"] for operation in patch["operations"]] == ["rosetta/model/aligner.py"]
    diff = (paths.root / "plan/code_patches/idea_patch/patch.diff").read_text(encoding="utf-8")
    assert "local/auto_research_runs" not in diff


def test_code_patch_persistent_backend_uses_git_worktree_and_codex_resume(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["codex_json_events"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_persistent", simulate=False)
    artifacts = ArtifactManager(paths.root)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append([str(part) for part in command])
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            worktree_repo = Path(command[-2])
            shutil.copytree(snapshot, worktree_repo)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command and command[0] == "codex":
            output_path = Path(command[command.index("--output-last-message") + 1])
            if "resume" in command:
                (Path(kwargs["cwd"]) / "rosetta/model/aligner.py").write_text("VALUE = 'persistent'\n", encoding="utf-8")
                output_path.write_text("patched\n", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout='{"type":"event","stage":"patch"}\n', stderr="session id: 123e4567-e89b-12d3-a456-426614174000\n")
            output_path.write_text('{"files":["rosetta/model/aligner.py"]}\n', encoding="utf-8")
            return SimpleNamespace(
                returncode=0,
                stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174000"}\n{"type":"event","stage":"preload"}\n',
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "idea/session path", "title": "Persistent", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    session_dir = paths.root / "plan/code_worktrees/idea-session_path/v1"
    assert (session_dir / "repo").exists()
    assert (session_dir / "codex_session.json").exists()
    assert (session_dir / "codex_events.jsonl").exists()
    assert (session_dir / "patch_blueprint.json").exists()
    assert "123e4567-e89b-12d3-a456-426614174000" in (paths.root / "meta/codex_sessions.yaml").read_text(encoding="utf-8")
    codex_commands = [command for command in commands if command and command[0] == "codex"]
    assert len(codex_commands) == 2
    assert "resume" not in codex_commands[0]
    assert "resume" in codex_commands[1]
    assert "--json" in codex_commands[0]
    assert "--json" in codex_commands[1]
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))
    assert patch["code_worktree"]["branch"] == "auto-research/proj_persistent/idea-session_path/v1"
    assert ideas[0]["code_patch"]["codex_session_id"] == "123e4567-e89b-12d3-a456-426614174000"


def test_code_patch_persistent_backend_materializes_snapshot_over_git_head(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    baseline_text = (snapshot / "rosetta/model/aligner.py").read_text(encoding="utf-8")
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_materialize_snapshot", simulate=False)
    artifacts = ArtifactManager(paths.root)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="stalehead\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            worktree_repo = Path(command[-2])
            shutil.copytree(snapshot, worktree_repo)
            (worktree_repo / "rosetta/model/aligner.py").write_text("VALUE = 'stale-head'\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command and command[0] == "codex":
            worktree_repo = Path(kwargs["cwd"])
            assert (worktree_repo / "rosetta/model/aligner.py").read_text(encoding="utf-8") == baseline_text
            output_path = Path(command[command.index("--output-last-message") + 1])
            if "resume" in command:
                (worktree_repo / "rosetta/model/aligner.py").write_text(baseline_text + "PATCHED = True\n", encoding="utf-8")
                output_path.write_text("patched\n", encoding="utf-8")
            else:
                output_path.write_text('{"files":["rosetta/model/aligner.py"]}\n', encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="session id: 123e4567-e89b-12d3-a456-426614174000\n")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "materialize", "title": "Materialize", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    metadata = json.loads((paths.root / "plan/code_worktrees/materialize/v1/worktree_metadata.json").read_text(encoding="utf-8"))
    assert metadata["baseline_materialized_from_snapshot"] is True
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))
    operation = patch["operations"][0]
    assert operation["path"] == "rosetta/model/aligner.py"
    assert operation["old_sha256"] == sha256_file(snapshot / "rosetta/model/aligner.py")


def test_code_patch_persistent_backend_reuses_existing_worktree(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_reuse_worktree", simulate=False)
    artifacts = ArtifactManager(paths.root)
    existing_repo = paths.root / "plan/code_worktrees/reuse/v1/repo"
    shutil.copytree(snapshot, existing_repo)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append([str(part) for part in command])
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            raise AssertionError("worktree add should not run for existing worktree")
        if command and command[0] == "codex":
            output_path = Path(command[command.index("--output-last-message") + 1])
            if "resume" in command:
                (Path(kwargs["cwd"]) / "rosetta/model/aligner.py").write_text("VALUE = 'reuse'\n", encoding="utf-8")
            output_path.write_text("ok\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="session id: 123e4567-e89b-12d3-a456-426614174000\n")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "reuse", "title": "Reuse", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert not any(command[:5] == ["git", "-C", str(repo), "worktree", "add"] for command in commands)


def test_code_patch_persistent_backend_resume_failure_falls_back_to_new_session(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    config["code_patch"]["validation"]["require_py_compile"] = False
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_resume_fallback", simulate=False)
    artifacts = ArtifactManager(paths.root)
    session_dir = paths.root / "plan/code_worktrees/fallback/v1"
    session_dir.mkdir(parents=True)
    (session_dir / "codex_session.json").write_text(json.dumps({"session_id": "old-session"}), encoding="utf-8")
    resume_attempts = 0

    def fake_run(command, **kwargs):
        nonlocal resume_attempts
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            shutil.copytree(snapshot, Path(command[-2]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command and command[0] == "codex":
            output_path = Path(command[command.index("--output-last-message") + 1])
            if "resume" in command:
                resume_attempts += 1
                output_path.write_text("", encoding="utf-8")
                return SimpleNamespace(returncode=1, stdout="", stderr="session not found\n")
            (Path(kwargs["cwd"]) / "rosetta/model/aligner.py").write_text("VALUE = 'fallback'\n", encoding="utf-8")
            output_path.write_text("patched\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="session id: 123e4567-e89b-12d3-a456-426614174000\n")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "fallback", "title": "Fallback", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert resume_attempts == 1
    actions = ideas[0]["code_patch"].get("recovery_actions") or []
    assert any(action.get("action") == "retry_codex_with_new_persistent_session" for action in actions)
    assert "123e4567-e89b-12d3-a456-426614174000" in (session_dir / "codex_session.json").read_text(encoding="utf-8")


def test_code_patch_persistent_backend_rejects_non_git_target_repo(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    paths = init_workspace(config, "topic", project_id="proj_non_git", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [{"id": "non_git", "title": "Non Git", "hypothesis": "h"}]

    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "no_valid_patch"
    assert "Git worktree requires c2c.target_repo to be a git repo" in ideas[0]["code_patch"]["reason"]


def test_code_patch_persistent_backend_rejects_target_repo_that_differs_from_snapshot(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_git_c2c_repo(tmp_path)
    snapshot = tmp_path / "snapshot"
    import shutil

    shutil.copytree(repo, snapshot, ignore=lambda directory, names: {".git"} & set(names))
    (repo / "rosetta/model/aligner.py").write_text("VALUE = 'not baseline'\n", encoding="utf-8")
    config = _code_patch_test_config(tmp_path / "workspace", snapshot)
    config["c2c"]["target_repo"] = str(repo)
    config["code_patch"]["backend"] = "codex_persistent_cli"
    config["code_patch"]["persistent_session"] = True
    config["code_patch"]["use_git_worktree"] = True
    paths = init_workspace(config, "topic", project_id="proj_baseline_guard", simulate=False)
    artifacts = ArtifactManager(paths.root)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[:3] == ["git", "-C", str(repo)] and command[3:5] == ["worktree", "add"]:
            raise AssertionError("worktree add must not run when baseline guard fails")
        raise AssertionError(f"unexpected command: {command}")

    import auto_research.code_patch as code_patch_module

    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)
    ideas = [{"id": "baseline_guard", "title": "Baseline Guard", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "no_valid_patch"
    assert "does not match the baseline snapshot" in ideas[0]["code_patch"]["reason"]


def test_code_patch_agent_filters_run_artifacts_from_patch_contract(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_contract_filter", simulate=False)
    artifacts = ArtifactManager(paths.root)
    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured["implementation_contract"] = implementation_contract
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'contract filter'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Patched only editable files."}

    ideas = [
        {
            "id": "contract_filter",
            "title": "Contract Filter",
            "hypothesis": "h",
            "experiment_contract": {
                "expected_files": [
                    "rosetta/model/aligner.py",
                    "local/auto_research_runs/contract_filter/main_results.json",
                    "local/auto_research_runs/contract_filter/train_overrides.json",
                ]
            },
        }
    ]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    contract = captured["implementation_contract"]
    assert isinstance(contract, dict)
    targets = contract["implementation_targets"]
    assert targets["expected_files"] == ["rosetta/model/aligner.py"]
    blocked_paths = {item["path"] for item in targets["blocked_expected_files"]}
    assert "local/auto_research_runs/contract_filter/main_results.json" in blocked_paths
    assert "local/auto_research_runs/contract_filter/train_overrides.json" in blocked_paths


def test_code_patch_contract_includes_c2c_mechanism_and_ablation(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_mechanism_contract", simulate=False)
    artifacts = ArtifactManager(paths.root)
    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured.setdefault("implementation_contract", implementation_contract)
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'mechanism contract'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Patched mechanism target."}

    ideas = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    contract = captured["implementation_contract"]
    assert isinstance(contract, dict)
    mechanism = contract["mechanism_contract"]
    assert mechanism["mechanism_type"] == "utility_predicted_cache_routing"
    assert mechanism["ablation_switch"] == "ablation_disable_utility_predicted_cache_routing"
    assert mechanism["coverage_diagnostics"]["required"] is True
    assert mechanism["matched_coverage_ablation"]["required"] is True
    assert mechanism["novelty_gate"]["status"] == "pass"
    scope = contract["implementation_scope"]
    assert scope["scope"] == "medium"
    assert "rosetta/model/utility_predicted_cache_routing.py" in scope["required_new_files"]
    assert scope["integration_points"]
    assert any("medium-scope" in requirement for requirement in contract["s2_5_requirements"])
    assert any("mechanism-level" in requirement for requirement in contract["s2_5_requirements"])
    assert any("matched-coverage" in requirement for requirement in contract["s2_5_requirements"])


def test_code_patch_contract_large_scope_uses_mvp_slice(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_large_scope_contract", simulate=False)
    artifacts = ArtifactManager(paths.root)
    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured.setdefault("implementation_contract", implementation_contract)
            (temp_repo / "rosetta/model/projector.py").write_text("VALUE = 'large scope mvp'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Patched first MVP slice."}

    ideas = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})
    candidate = ideas[1]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": [candidate]}, [candidate])

    assert manifest["status"] == "ok"
    contract = captured["implementation_contract"]
    assert isinstance(contract, dict)
    assert contract["implementation_scope"]["scope"] == "large"
    assert contract["implementation_scope"]["mvp_slice"]
    assert any("large-scope" in requirement for requirement in contract["s2_5_requirements"])


def test_code_patch_agent_repairs_validation_failed_patch_once(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo, require_targeted_tests=True)
    paths = init_workspace(config, "topic", project_id="proj_patch_validation_repair", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class RepairingBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            if len(self.calls) == 1:
                (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'needs repair'\n", encoding="utf-8")
                (temp_repo / "test/test_aligner_span_overlap.py").write_text(
                    "def test_span():\n    assert False\n",
                    encoding="utf-8",
                )
                return {"status": "ok", "rationale": "Initial patch has a failing focused test."}
            assert implementation_contract["validation_failure_feedback"]["failed_checks"]
            (temp_repo / "test/test_aligner_span_overlap.py").write_text(
                "def test_span():\n    assert True\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Repaired the focused test failure."}

    backend = RepairingBackend()
    ideas = [{"id": "validation_repair", "title": "Validation Repair", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.calls) == 2
    assert "validation_failure_feedback" in backend.calls[1]
    assert validation["status"] == "ok"
    assert validation["recovery_actions"][0]["action"] == "retry_codex_after_validation_failure"
    assert patch["recovery_actions"][0]["action"] == "retry_codex_after_validation_failure"


def test_code_patch_runtime_smoke_repairs_dtype_failure_before_proxy_train(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "script/train/SFT_train.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                "import os",
                "import sys",
                "if os.environ.get('WANDB_DISABLED') != 'true':",
                "    raise RuntimeError('wandb smoke must be disabled')",
                "config_path = Path(sys.argv[sys.argv.index('--config') + 1])",
                "cfg = json.loads(config_path.read_text(encoding='utf-8'))",
                "if cfg['data']['kwargs']['num_samples'] < 2:",
                "    raise RuntimeError('runtime smoke train sample split would be empty')",
                "if cfg['data']['train_ratio'] >= 0.99:",
                "    raise RuntimeError('runtime smoke train ratio was not hardened')",
                "aligner = Path('rosetta/model/aligner.py').read_text(encoding='utf-8')",
                "if 'runtime repaired' not in aligner:",
                "    raise RuntimeError('expected scalar type Float but found BFloat16')",
                "print('first batch ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["validation"]["runtime_smoke"] = {"enabled": True, "train_samples": 1, "timeout_seconds": 20, "gpu_ids": []}
    config["code_patch"]["validation"]["max_repair_attempts"] = 1
    paths = init_workspace(config, "topic", project_id="proj_patch_runtime_smoke", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class RuntimeRepairBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            if len(self.calls) == 1:
                (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'runtime needs repair'\n", encoding="utf-8")
                return {"status": "ok", "rationale": "Initial patch leaves dtype mismatch in first batch."}
            feedback = implementation_contract["validation_failure_feedback"]
            assert feedback["failed_checks"][0]["name"] == "runtime_smoke:first_batch_train"
            assert feedback["failed_checks"][0]["failure_category"] == "dtype_mismatch"
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'runtime repaired'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Repaired dtype handling for first-batch smoke."}

    ideas = [{"id": "runtime_smoke_repair", "title": "Runtime Smoke Repair", "hypothesis": "h"}]
    backend = RuntimeRepairBackend()
    CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.calls) == 2
    assert any(check["name"] == "runtime_smoke:first_batch_train" and check["returncode"] == 0 for check in validation["checks"])
    assert any(action["action"] == "retry_codex_after_validation_failure" for action in validation["recovery_actions"])


def test_code_patch_validation_preserves_pytest_failure_context(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo, require_targeted_tests=True)
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    paths = init_workspace(config, "topic", project_id="proj_patch_failure_context", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class FailingTestBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'failure context'\n", encoding="utf-8")
            (temp_repo / "test/test_aligner_span_overlap.py").write_text(
                "def test_span():\n"
                "    assert 'actual boundary score' == 'expected boundary score'\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Patch with an intentional test failure."}

    ideas = [{"id": "failure_context", "title": "Failure Context", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=FailingTestBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert validation["status"] == "validation_failed"
    failed = [check for check in validation["checks"] if check["returncode"] != 0]
    assert failed
    assert "actual boundary score" in failed[0]["stdout"]
    assert "expected boundary score" in failed[0]["stdout"]


def test_code_patch_validation_records_pytest_timeout(tmp_path: Path, monkeypatch) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo, require_targeted_tests=True)
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    paths = init_workspace(config, "topic", project_id="proj_patch_pytest_timeout", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class TimeoutBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'timeout context'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Patch whose focused test times out."}

    import auto_research.code_patch as code_patch_module

    real_run = code_patch_module.subprocess.run

    def fake_run(command, *args, **kwargs):
        if "-m" in command and "pytest" in command:
            raise code_patch_module.subprocess.TimeoutExpired(command, kwargs.get("timeout"), output=b"partial pytest stdout", stderr=b"partial pytest stderr")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(code_patch_module.subprocess, "run", fake_run)

    ideas = [{"id": "pytest_timeout", "title": "Pytest Timeout", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=TimeoutBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert ideas[0]["code_patch"]["status"] == "validation_failed"
    timeout_check = next(check for check in validation["checks"] if check["name"].startswith("pytest:"))
    assert timeout_check["returncode"] == 124
    assert timeout_check["failure_category"] == "pytest_timeout"
    assert timeout_check["timeout_seconds"] == 180
    assert "partial pytest stdout" in timeout_check["stdout"]
    assert "Command timed out after 180s" in timeout_check["stderr"]


def test_code_patch_agent_marks_py_compile_failure_as_validation_failed(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_bad", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class BadBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("def broken(:\n", encoding="utf-8")
            return {"status": "ok", "rationale": "This patch has a syntax error."}

    ideas = [{"id": "bad_patch", "title": "Bad Patch", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=BadBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert ideas[0]["code_patch"]["status"] == "validation_failed"
    assert validation["status"] == "validation_failed"
    assert any(check["returncode"] != 0 for check in validation["checks"])


def test_code_patch_agent_blocks_unactivated_new_config_parameter(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/projector.py").write_text(
        "class C2CProjector:\n"
        "    def __init__(self, hidden_dim: int = 4):\n"
        "        self.hidden_dim = hidden_dim\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_activation", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class NewParamBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class C2CProjector:\n"
                "    def __init__(\n"
                "        self,\n"
                "        hidden_dim: int = 4,\n"
                "        alignment_confidence_head_group_count: int = 2,\n"
                "    ):\n"
                "        self.hidden_dim = hidden_dim\n"
                "        self.alignment_confidence_head_group_count = alignment_confidence_head_group_count\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Added a configurable head group count."}

    ideas = [{"id": "unactivated_param", "title": "Unactivated Param", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=NewParamBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert ideas[0]["code_patch"]["status"] == "config_activation_missing"
    assert ideas[0]["code_patch"]["has_executable_change"] is False
    assert validation["activation_check"]["missing_parameters"] == ["alignment_confidence_head_group_count"]


def test_code_patch_agent_repairs_unactivated_new_config_parameter(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/projector.py").write_text(
        "class C2CProjector:\n"
        "    def __init__(self, hidden_dim: int = 4):\n"
        "        self.hidden_dim = hidden_dim\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_repair", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class RepairingParamBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            if len(self.calls) == 1:
                (temp_repo / "rosetta/model/projector.py").write_text(
                    "class C2CProjector:\n"
                    "    def __init__(\n"
                    "        self,\n"
                    "        hidden_dim: int = 4,\n"
                    "        alignment_confidence_head_group_count: int = 2,\n"
                    "    ):\n"
                    "        self.hidden_dim = hidden_dim\n"
                    "        self.alignment_confidence_head_group_count = alignment_confidence_head_group_count\n",
                    encoding="utf-8",
                )
                return {"status": "ok", "rationale": "Added an unactivated configurable head group count."}
            feedback = implementation_contract["validation_failure_feedback"]
            assert feedback["activation_check"]["status"] == "config_activation_missing"
            assert feedback["activation_check"]["missing_parameters"] == ["alignment_confidence_head_group_count"]
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class C2CProjector:\n"
                "    def __init__(self, hidden_dim: int = 4):\n"
                "        self.hidden_dim = hidden_dim\n"
                "        self.alignment_confidence_head_group_count = 2\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Repaired by keeping the value internal for the MVP."}

    backend = RepairingParamBackend()
    ideas = [{"id": "activation_repair", "title": "Activation Repair", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.calls) == 2
    assert validation["activation_check"]["status"] == "ok"
    assert validation["recovery_actions"][0]["action"] == "retry_codex_after_validation_failure"
    assert patch["recovery_actions"][0]["failed_checks"] == []


def test_code_patch_agent_accepts_new_config_parameter_when_contract_activates_it(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/projector.py").write_text(
        "class C2CProjector:\n"
        "    def __init__(self, hidden_dim: int = 4):\n"
        "        self.hidden_dim = hidden_dim\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_activation_ok", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class NewParamBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            assert implementation_contract["experiment_contract"]["config_overrides"]["train"]["model"]["projector"]["params"]["alignment_confidence_head_group_count"] == 2
            del edit_policy
            (temp_repo / "rosetta/model/projector.py").write_text(
                "class C2CProjector:\n"
                "    def __init__(\n"
                "        self,\n"
                "        hidden_dim: int = 4,\n"
                "        alignment_confidence_head_group_count: int = 2,\n"
                "    ):\n"
                "        self.hidden_dim = hidden_dim\n"
                "        self.alignment_confidence_head_group_count = alignment_confidence_head_group_count\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Added and activated a configurable head group count."}

    ideas = [
        {
            "id": "activated_param",
            "title": "Activated Param",
            "hypothesis": "h",
            "experiment_contract": {
                "config_overrides": {
                    "train": {
                        "model": {
                            "projector": {
                                "params": {
                                    "alignment_confidence_head_group_count": 2,
                                }
                            }
                        }
                    }
                }
            },
        }
    ]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=NewParamBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert validation["activation_check"]["activated_parameters"] == ["alignment_confidence_head_group_count"]


def test_code_patch_agent_ignores_standard_projector_list_constructor_param(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/wrapper.py").write_text(
        "class RosettaModel:\n"
        "    def __init__(self, model_list=None):\n"
        "        self.model_list = model_list\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_projector_list", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class ProjectorListBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/wrapper.py").write_text(
                "class RosettaModel:\n"
                "    def __init__(self, model_list=None, projector_list=None):\n"
                "        self.model_list = model_list\n"
                "        self.projector_list = projector_list or []\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Threaded an existing constructor dependency."}

    ideas = [{"id": "projector_list", "title": "Projector List", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=ProjectorListBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert validation["activation_check"]["status"] == "ok"
    assert validation["activation_check"]["introduced_config_parameters"] == []


def test_code_patch_agent_ignores_local_type_annotations_for_activation(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    (repo / "rosetta/model/aligner.py").write_text(
        "from typing import List, Tuple\n\n"
        "def score_spans():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_local_annotations", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class LocalAnnotationBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text(
                "from typing import List, Tuple\n\n"
                "def score_spans():\n"
                "    selected_token_spans: List[Tuple[int, int]] = []\n"
                "    target_span: Tuple[int, int] = (0, 1)\n"
                "    selected_token_spans.append(target_span)\n"
                "    return len(selected_token_spans)\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Added local span bookkeeping."}

    ideas = [{"id": "local_annotations", "title": "Local annotations", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=LocalAnnotationBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert validation["activation_check"]["introduced_config_parameters"] == []


def test_code_patch_agent_includes_previous_patch_failure_in_retry_contract(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_retry_feedback", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class FailingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            (temp_repo / "rosetta/model/aligner.py").write_text("def broken(:\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Broken first attempt."}

    ideas = [{"id": "retry_feedback", "title": "Retry Feedback", "hypothesis": "h"}]
    first_manifest = CodePatchAgent(paths.root, config, artifacts, backend=FailingBackend()).run({"candidate_ideas": ideas}, ideas)
    assert first_manifest["status"] == "no_valid_patch"

    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured["previous_patch_failure"] = implementation_contract.get("previous_patch_failure")
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'retry ok'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Retry used failure feedback."}

    retry_ideas = [{"id": "retry_feedback", "title": "Retry Feedback", "hypothesis": "h"}]
    retry_manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": retry_ideas}, retry_ideas)

    assert retry_manifest["status"] == "ok"
    previous = captured["previous_patch_failure"]
    assert isinstance(previous, dict)
    assert previous["status"] == "validation_failed"
    assert previous["failed_checks"]


def test_code_patch_agent_includes_proxy_effect_repair_contract(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_proxy_effect_repair_feedback", simulate=False)
    artifacts = ArtifactManager(paths.root)
    results_dir = paths.root / "experiment" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    proxy_contract = {
        "mode": "effect_first_proxy_repair",
        "reason": "proxy mean delta 0.0 below soft threshold 0.1",
        "proxy_dataset_deltas": {"mmlu-redux": -0.2, "openbookqa": 0.1},
        "proxy_dataset_regressions": {"mmlu-redux": 0.2, "openbookqa": 0.0},
        "dragging_datasets": [{"dataset": "mmlu-redux", "delta": -0.2, "regression": 0.2}],
        "patch_risk_labels": ["test_change"],
        "repair_priorities": ["Target dragging proxy datasets: mmlu-redux"],
    }
    (results_dir / "main_results.json").write_text(
        json.dumps(
            {
                "candidate_results": [
                    {
                        "id": "proxy_retry",
                        "title": "Proxy Retry",
                        "decision": "proxy_repairable",
                        "patch_result": {"changed_files": ["rosetta/model/aligner.py"]},
                        "proxy_screen": {
                            "status": "repairable_proxy_risk",
                            "reason": proxy_contract["reason"],
                            "repair_hint": "effect repair only",
                            "proxy_effect_repair_contract": proxy_contract,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class InspectingBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            captured["implementation_contract"] = implementation_contract
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'proxy repair ok'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Used proxy evidence."}

    ideas = [{"id": "proxy_retry", "title": "Proxy Retry", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=InspectingBackend()).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    contract = captured["implementation_contract"]
    assert isinstance(contract, dict)
    proxy_effect = contract["proxy_effect_repair_contract"]
    assert isinstance(proxy_effect, dict)
    assert proxy_effect["mode"] == "effect_first_proxy_repair"
    assert proxy_effect["dragging_datasets"][0]["dataset"] == "mmlu-redux"
    requirements = contract["s2_5_requirements"]
    assert any("effect-first cheap-proxy repair" in requirement for requirement in requirements)
    assert any("mmlu-redux" in requirement for requirement in requirements)
    assert any("paperization-only" in requirement for requirement in requirements)


def test_code_patch_agent_marks_codex_429_as_retryable(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_429", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class RateLimitedBackend:
        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, temp_repo, edit_policy
            return {
                "status": "codex_failed",
                "reason": "ERROR: exceeded retry limit, last status: 429 Too Many Requests",
                "stderr": "429 Too Many Requests",
            }

    ideas = [{"id": "rate_limited", "title": "Rate Limited", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=RateLimitedBackend()).run({"candidate_ideas": ideas}, ideas)
    plan_dir = paths.root / "plan"
    (plan_dir / "plan.yaml").write_text(
        yaml.safe_dump(
            {
                "hypotheses": [{"id": "h1"}],
                "baselines": [{"name": "base"}, {"name": "candidate"}],
                "datasets": [{"name": "mmlu-redux"}],
                "task_graph": {},
                "resource_budget": {"peak_concurrent_gpus": 1},
                "execution": {
                    "collector": "c2c_small_loop",
                    "min_delta_to_pass": 0.1,
                    "max_dataset_regression": 2.0,
                    "selected_gpu_ids": [0],
                },
                "acceptance_criteria": {
                    "minimum_mean_delta": 0.1,
                    "coverage_diagnostics_required": True,
                    "matched_coverage_ablation_required": True,
                },
                "ablation_matrix": [
                    {"experiment": "matched transfer coverage control", "matched_coverage_ablation": {"required": True}}
                ],
                "reviewer_risk_controls": {"top_concerns": []},
            }
        ),
        encoding="utf-8",
    )
    (plan_dir / "short_loop_plan.yaml").write_text("run: true\n", encoding="utf-8")
    (plan_dir / "candidate_ideas.json").write_text(json.dumps(default_c2c_ideas("topic", config["c2c"]["baseline"])), encoding="utf-8")
    report = S2GateValidator(paths.root, config).validate().to_dict()

    assert manifest["status"] == "retryable_no_valid_patch"
    assert manifest["retryable_patch_count"] == 1
    assert ideas[0]["code_patch"]["status"] == "retryable_codex_failed"
    assert ideas[0]["code_patch"]["retryable"] is True
    assert report["status"] == "NEEDS_RETRY"
    patch_status = next(check for check in report["checks"] if check["name"] == "s2_5_patch_manifest_status")
    assert patch_status["status"] == "NEEDS_RETRY"


def test_code_patch_agent_retries_noop_sandbox_error_with_fallback_sandbox(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["codex_sandbox"] = "workspace-write"
    config["code_patch"]["codex_sandbox_fallback"] = "danger-full-access"
    paths = init_workspace(config, "topic", project_id="proj_patch_sandbox_retry", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class SandboxBackend(CodexPatchBackend):
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, temp_repo, edit_policy
            self.calls.append("primary")
            return {
                "status": "ok",
                "sandbox": "workspace-write",
                "rationale": "Blocked by the execution sandbox: bwrap: Can't bind mount /oldroot/ on /newroot/: No such device",
            }

        def _run_codex_once(self, implementation_contract, temp_repo, edit_policy, *, sandbox):
            del implementation_contract, edit_policy
            self.calls.append(sandbox)
            assert sandbox == "danger-full-access"
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'fallback patched'\n", encoding="utf-8")
            return {"status": "ok", "sandbox": sandbox, "rationale": "Fallback sandbox patch succeeded."}

    backend = SandboxBackend()
    ideas = [{"id": "sandbox_retry", "title": "Sandbox Retry", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert backend.calls == ["primary", "danger-full-access"]
    assert validation["recovery_actions"][0]["action"] == "retry_codex_noop_with_fallback_sandbox"
    assert patch["backend_sandbox"] == "danger-full-access"
    assert patch["recovery_actions"][0]["fallback_sandbox"] == "danger-full-access"
    assert "fallback patched" not in (repo / "rosetta/model/aligner.py").read_text(encoding="utf-8")


def test_code_patch_agent_repairs_blocked_no_executable_change(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    paths = init_workspace(config, "topic", project_id="proj_patch_noop_contract_repair", simulate=False)
    artifacts = ArtifactManager(paths.root)

    class NoopThenPatchBackend:
        def __init__(self):
            self.calls = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.calls.append(implementation_contract)
            if len(self.calls) == 1:
                return {"status": "ok", "rationale": "Initial attempt did not edit files."}
            feedback = implementation_contract["contract_failure_feedback"]
            assert feedback["status"] == "blocked_no_executable_change"
            assert "no allowed file changes" in feedback["reason"]
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'contract repaired'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "Repaired by editing an allowed integration point."}

    backend = NoopThenPatchBackend()
    ideas = [{"id": "noop_contract_repair", "title": "Noop Contract Repair", "hypothesis": "h"}]
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert manifest["valid_patch_count"] == 1
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.calls) == 2
    assert validation["status"] == "ok"
    assert patch["recovery_actions"][0]["action"] == "retry_codex_after_contract_failure"
    assert patch["recovery_actions"][0]["failed_status"] == "blocked_no_executable_change"


def test_code_patch_agent_uses_second_variant_after_first_validation_failure(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 2
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    config["code_patch"]["validation"]["max_contract_repair_attempts"] = 0
    paths = init_workspace(config, "topic", project_id="proj_best_of_n", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [{"id": "best_of_n", "title": "Best Of N", "hypothesis": "h"}]

    class VariantBackend:
        def __init__(self):
            self.contracts = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.contracts.append(implementation_contract)
            if len(self.contracts) == 1:
                (temp_repo / "rosetta/model/aligner.py").write_text("def broken(:\n", encoding="utf-8")
                return {"status": "ok", "rationale": "broken first variant"}
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'second variant'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "valid second variant"}

    backend = VariantBackend()
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert len(backend.contracts) == 2
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert ideas[0]["code_patch"]["selected_variant"] == 2
    assert ideas[0]["code_patch"]["patch_json"].endswith("plan/code_patches/best_of_n/variants/v2/patch.json")
    assert backend.contracts[1]["patch_variant"]["previous_variant_attempts"][0]["status"] == "validation_failed"


def test_code_patch_agent_scores_all_ok_variants_and_selects_best_quality(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 2
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    config["code_patch"]["validation"]["mechanism_self_review"] = {"enabled": True}
    paths = init_workspace(config, "topic", project_id="proj_best_quality", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [
        {
            "id": "quality_best",
            "title": "Quality Best",
            "hypothesis": "h",
            "mechanism_type": "utility_predicted_cache_routing",
            "ablation_plan": {"switch": "ablation_disable_quality_best"},
            "coverage_diagnostics": {"required": True},
            "matched_coverage_ablation": {"required": True},
            "experiment_contract": {
                "ablation_switch": "ablation_disable_quality_best",
                "coverage_diagnostics": {"required": True},
                "matched_coverage_ablation": {"required": True},
                "expected_files": ["rosetta/model/projector.py"],
            },
        }
    ]

    class QualityBackend:
        def __init__(self):
            self.calls = 0

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del implementation_contract, edit_policy
            self.calls += 1
            if self.calls == 1:
                (temp_repo / "rosetta/model/projector.py").write_text(
                    "QUALITY = 'ok but weaker core mechanism path'\n",
                    encoding="utf-8",
                )
                return {"status": "ok", "rationale": "First valid but low-evidence variant."}
            (temp_repo / "rosetta/model/projector.py").write_text(
                "\n".join(
                    [
                        "ablation_disable_quality_best = False",
                        "def route_cache(control_mode='matched_coverage_quality_best'):",
                        "    coverage_diagnostics = {'accepted_span_rate': 0.5}",
                        "    matched_coverage_delta = 0.0",
                        "    return coverage_diagnostics, matched_coverage_delta",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return {"status": "ok", "rationale": "Second valid variant has explicit mechanism evidence."}

    backend = QualityBackend()
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)

    assert manifest["status"] == "ok"
    assert backend.calls == 2
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert ideas[0]["code_patch"]["selected_variant"] == 2
    assert ideas[0]["code_patch"]["selection_reason"] == "quality_score"
    attempts = ideas[0]["code_patch"]["variant_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["status"] == "ok"
    assert "ablation_switch_not_wired" in attempts[0]["quality_score"]["soft_issues"]
    assert attempts[1]["quality_score"]["score"] > attempts[0]["quality_score"]["score"]
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    assert validation["mechanism_review"]["status"] == "ok"
    assert validation["mechanism_review"]["mechanism_evidence_map"]


def test_code_patch_mechanism_self_review_keeps_diagnostics_soft_by_default(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["max_repair_attempts"] = 1
    config["code_patch"]["validation"]["mechanism_self_review"] = {"enabled": True}
    paths = init_workspace(config, "topic", project_id="proj_mechanism_review_soft", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [
        {
            "id": "review_repair",
            "title": "Review Repair",
            "hypothesis": "h",
            "mechanism_type": "utility_predicted_cache_routing",
            "ablation_plan": {"switch": "ablation_disable_review_repair"},
            "coverage_diagnostics": {"required": True},
            "matched_coverage_ablation": {"required": True},
            "experiment_contract": {
                "ablation_switch": "ablation_disable_review_repair",
                "coverage_diagnostics": {"required": True},
                "matched_coverage_ablation": {"required": True},
                "expected_files": ["rosetta/model/projector.py"],
            },
        }
    ]

    class SoftReviewBackend:
        def __init__(self):
            self.contracts = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.contracts.append(implementation_contract)
            if len(self.contracts) == 1:
                (temp_repo / "rosetta/model/projector.py").write_text(
                    "VALUE = 'mechanism shell without diagnostics'\n",
                    encoding="utf-8",
                )
                return {"status": "ok", "rationale": "Missing ablation and diagnostics."}

    backend = SoftReviewBackend()
    manifest = CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.contracts) == 1
    assert validation["mechanism_review"]["status"] == "ok"
    assert "ablation_switch_not_wired" in validation["mechanism_review"]["soft_issues"]
    assert validation["mechanism_review"]["quality_repair"]["needed"] is False
    assert validation["mechanism_review"]["quality_repair"]["deferred"] is True
    assert validation["mechanism_review"]["quality_repair"]["mode"] == "paperization_after_effect"
    assert ideas[0]["code_patch"]["quality_score"]["soft_issues"]


def test_code_patch_mechanism_self_review_can_be_strict_for_diagnostics(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["max_repair_attempts"] = 0
    config["code_patch"]["validation"]["mechanism_self_review"] = {
        "enabled": True,
        "require_ablation_wired": True,
        "require_coverage_evidence": True,
        "require_matched_coverage_evidence": True,
    }
    paths = init_workspace(config, "topic", project_id="proj_mechanism_review_strict", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [
        {
            "id": "strict_review",
            "title": "Strict Review",
            "hypothesis": "h",
            "mechanism_type": "utility_predicted_cache_routing",
            "ablation_plan": {"switch": "ablation_disable_strict_review"},
            "coverage_diagnostics": {"required": True},
            "matched_coverage_ablation": {"required": True},
            "experiment_contract": {
                "ablation_switch": "ablation_disable_strict_review",
                "coverage_diagnostics": {"required": True},
                "matched_coverage_ablation": {"required": True},
                "expected_files": ["rosetta/model/projector.py"],
            },
        }
    ]

    class StrictBackend:
            def generate(self, implementation_contract, temp_repo, edit_policy):
                del implementation_contract, edit_policy
                (temp_repo / "rosetta/model/projector.py").write_text(
                    "VALUE = 'mechanism shell only'\n",
                    encoding="utf-8",
                )
                return {"status": "ok", "rationale": "Missing strict diagnostics."}

    manifest = CodePatchAgent(paths.root, config, artifacts, backend=StrictBackend()).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "no_valid_patch"
    assert ideas[0]["code_patch"]["status"] == "mechanism_self_review_failed"
    assert "ablation_switch_not_wired" in validation["mechanism_review"]["issues"]
    assert "missing_coverage_diagnostics_evidence" in validation["mechanism_review"]["issues"]


def test_code_patch_agent_repairs_evaluator_proxy_risk(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    paths = init_workspace(config, "topic", project_id="proj_eval_repair", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [{"id": "eval_repair", "title": "Eval Repair", "hypothesis": "h"}]

    class EvalRiskBackend:
        def __init__(self):
            self.contracts = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.contracts.append(implementation_contract)
            if len(self.contracts) == 1:
                (temp_repo / "script/evaluation/unified_evaluator.py").write_text("print('contaminated')\n", encoding="utf-8")
                return {"status": "ok", "rationale": "bad evaluator hook"}
            (temp_repo / "script/evaluation/unified_evaluator.py").write_text("print('eval')\n", encoding="utf-8")
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'repaired mechanism'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "safe repair"}

    backend = EvalRiskBackend()
    CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))
    patch = json.loads((paths.root / ideas[0]["code_patch"]["patch_json"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "ok"
    assert len(backend.contracts) == 2
    assert backend.contracts[1]["contract_failure_feedback"]["status"] == "proxy_risk_repair_required"
    assert patch["changed_files"] == ["rosetta/model/aligner.py"]
    assert validation["risk_check"]["status"] == "ok"
    assert any(action["failed_status"] == "proxy_risk_repair_required" for action in validation["recovery_actions"])
    assert any(action["action"] == "restore_evaluator_files_before_repair" for action in validation["recovery_actions"])


def test_code_patch_agent_blocks_evaluator_repair_that_recontaminates(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _code_patch_test_config(tmp_path / "workspace", repo)
    config["code_patch"]["variants_per_candidate"] = 1
    config["code_patch"]["validation"]["max_contract_repair_attempts"] = 1
    paths = init_workspace(config, "topic", project_id="proj_eval_repair_recontaminate", simulate=False)
    artifacts = ArtifactManager(paths.root)
    ideas = [{"id": "eval_recontaminate", "title": "Eval Recontaminate", "hypothesis": "h"}]

    class RecontaminatingBackend:
        def __init__(self):
            self.contracts = []

        def generate(self, implementation_contract, temp_repo, edit_policy):
            del edit_policy
            self.contracts.append(implementation_contract)
            (temp_repo / "script/evaluation/unified_evaluator.py").write_text("print('still contaminated')\n", encoding="utf-8")
            (temp_repo / "rosetta/model/aligner.py").write_text("VALUE = 'mechanism attempt'\n", encoding="utf-8")
            return {"status": "ok", "rationale": "still touched evaluator"}

    backend = RecontaminatingBackend()
    CodePatchAgent(paths.root, config, artifacts, backend=backend).run({"candidate_ideas": ideas}, ideas)
    validation = json.loads((paths.root / ideas[0]["code_patch"]["validation"]).read_text(encoding="utf-8"))

    assert ideas[0]["code_patch"]["status"] == "proxy_risk_repair_required"
    assert len(backend.contracts) == 2
    assert backend.contracts[1]["forbidden_repair_files"] == ["script/evaluation/"]
    assert validation["risk_check"]["risk_labels"] == ["evaluation_code_changed"]
    assert any(action["action"] == "restore_evaluator_files_before_repair" for action in validation["recovery_actions"])


def test_c2c_pipeline_blocks_with_missing_reference_path(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "missing_paper.pdf"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_rebuttal.write_text("review text", encoding="utf-8")
    config = _base_config(tmp_path / "workspace", simulate=True)
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    orchestrator = Orchestrator()
    project_id = orchestrator.init_c2c_project(
        "cross tokenizer cache",
        target_repo=source_repo,
        ref_paper=ref_paper,
        ref_rebuttal=ref_rebuttal,
        env_python=Path("/usr/bin/python3"),
        project_id="proj_c2c_missing_ref",
        simulate=True,
    )

    result = orchestrator.start(project_id)

    assert result["status"] == "blocked"
    assert result["stage"] == "S0_intake"
    assert "ref_paper" in result["reason"]


def test_c2c_pipeline_runs_to_s3_with_mock_small_loop(monkeypatch, tmp_path: Path) -> None:
    source_repo = _fake_c2c_repo(tmp_path)
    ref_paper = tmp_path / "paper.txt"
    ref_rebuttal = tmp_path / "rebuttal.md"
    ref_paper.write_text("paper text", encoding="utf-8")
    ref_rebuttal.write_text("review text", encoding="utf-8")
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["agents"] = {"s2_directional_planner": {"resume_enabled": False}}
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    s1_codex_commands = []
    s1_codex_prompts = []
    original_subprocess_run = literature_module.subprocess.run

    def fake_s1_codex_run(command, **kwargs):
        if not command or command[0] != "codex":
            return original_subprocess_run(command, **kwargs)
        s1_codex_commands.append(command)
        s1_codex_prompts.append(kwargs.get("input") or "")
        output_path = Path(command[command.index("--output-last-message") + 1])
        if len(s1_codex_commands) == 1:
            output_path.write_text("not json", encoding="utf-8")
            stdout = '{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174001"}\n'
        else:
            assert "resume" in command
            output_path.write_text(json.dumps(_s1_codex_direction_payload()), encoding="utf-8")
            stdout = ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(literature_module.subprocess, "run", fake_s1_codex_run)
    orchestrator = Orchestrator()
    project_id = orchestrator.init_c2c_project(
        "cross tokenizer cache",
        target_repo=source_repo,
        ref_paper=ref_paper,
        ref_rebuttal=ref_rebuttal,
        env_python=Path("/usr/bin/python3"),
        project_id="proj_c2c_loop",
        simulate=True,
    )
    result = orchestrator.start(project_id)
    root = tmp_path / "workspace" / project_id

    assert result["status"] == "completed"
    main_results = json.loads((root / "experiment/results/main_results.json").read_text(encoding="utf-8"))
    assert main_results["best_candidate"]["decision"] == "candidate_win"
    assert main_results["acceptance"]["passed"] is True
    assert main_results["acceptance"]["delta"] >= main_results["acceptance"]["min_delta_to_pass"]
    assert (root / "intake/c2c/static_bundle.json").exists()
    assert (root / "intake/c2c/evidence_brief.json").exists()
    assert (root / "intake/c2c/chunk_index.json").exists()
    assert (root / "intake/c2c/chunk_index.jsonl").exists()
    assert (root / "intake/c2c/code_intake_report.json").exists()
    assert (root / "intake/c2c/implementation_surface_map.json").exists()
    assert (root / "intake/c2c/code_retrieval_index.json").exists()
    assert (root / "plan/short_loop_plan.yaml").exists()
    assert (root / "literature/idea_debate.json").exists()
    assert (root / "literature/negative_constraints.json").exists()
    assert (root / "literature/c2c/evidence_requests.json").exists()
    assert (root / "literature/c2c/evidence_bundle.json").exists()
    assert (root / "literature/c2c/direction_decision.json").exists()
    assert (root / "literature/c2c/evidence_session.json").exists()
    assert (root / "literature/c2c/repo_card.json").exists()
    assert (root / "literature/c2c/rebuttal_concern_matrix.json").exists()
    assert (root / "literature/c2c/negative_result_memory.json").exists()
    assert (root / "literature/c2c/paper_chunks.jsonl").exists()
    assert (root / "literature/c2c/bibliography.json").exists()
    assert (root / "literature/c2c/rebuttal_chunks.jsonl").exists()
    assert (root / "literature/c2c/code_cards.json").exists()
    assert (root / "literature/c2c/code_chunks.jsonl").exists()
    assert (root / "literature/c2c/code_intake_report.json").exists()
    assert (root / "literature/c2c/implementation_surface_map.json").exists()
    assert (root / "literature/c2c/code_retrieval_index.json").exists()
    assert (root / "literature/c2c/chunk_index.json").exists()
    assert (root / "literature/c2c/retrieval_plan.json").exists()
    assert (root / "literature/c2c/retrieval_followup.json").exists()
    bundle = json.loads((root / "intake/c2c/static_bundle.json").read_text(encoding="utf-8"))
    assert bundle["chunk_index"]["counts"]["paper"] > 0
    assert bundle["chunk_index"]["counts"]["rebuttal"] > 0
    assert bundle["chunk_index"]["counts"]["code"] > 0
    assert bundle["code_intake_report"]["counts"]["chunks"] > 0
    assert bundle["implementation_surface_map"]["surfaces"]
    assert bundle["code_retrieval_index"]["default_queries"]
    ideas = json.loads((root / "literature/ideas.json").read_text(encoding="utf-8"))
    assert len(ideas) == 1
    assert ideas[0]["id"] == "utility_predicted_cache_routing"
    assert ideas[0]["s1_evidence_agent"]["source"] == "codex_resume_evidence_agent"
    direction = json.loads((root / "literature/c2c/direction_decision.json").read_text(encoding="utf-8"))
    assert direction["direction_id"] == "utility_predicted_cache_routing"
    evidence_session = json.loads((root / "literature/c2c/evidence_session.json").read_text(encoding="utf-8"))
    assert evidence_session["repair_count"] == 1
    assert len(evidence_session["attempts"]) == 2
    assert "resume" in s1_codex_commands[1]
    assert "errors_to_fix" in s1_codex_prompts[1]


def test_gpu_selector_auto_limits_to_six(monkeypatch) -> None:
    snapshot = [
        {"index": idx, "memory_total_mb": 80000, "memory_free_mb": 10000 + idx * 1000, "memory_used_mb": 0, "utilization_gpu": idx}
        for idx in range(8)
    ]
    monkeypatch.setattr(ExperimentRunner, "_gpu_snapshot", staticmethod(lambda: snapshot))
    selection = ExperimentRunner({"experiment": {"gpu_policy": {"max_gpus": 6, "min_free_mb": 0}}, "c2c": {"small_loop": {"gpu_ids": "auto"}}}).select_gpus()
    assert selection.selected_ids == [7, 6, 5, 4, 3, 2]
    assert len(selection.selected_ids) == 6


def test_c2c_preflight_repairs_broken_model_symlink(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    model_root = tmp_path / "models"
    broken = model_root / "Qwen3-0.6B"
    model_root.mkdir()
    broken.symlink_to(model_root / "missing", target_is_directory=True)
    hf_home = tmp_path / "hf_home"
    snapshot = hf_home / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: hf_home))
    monkeypatch.setattr(C2CAdapter, "_offline_model_load_errors", staticmethod(lambda model_path: []))

    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "model_map": {"Qwen/Qwen3-0.6B": str(broken)},
            "datasets": ["mmlu-redux"],
            "small_loop": {"eval_datasets": ["mmlu-redux"], "strict_dataset_cache": False},
        },
        "experiment": {"gpu_policy": {"max_gpus": 1}},
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    candidate = {"id": "idea", "title": "Idea"}
    run_spec = adapter.materialize_candidate_configs(candidate)
    result = adapter.preflight(run_spec)
    assert result["status"] == "ok"
    assert broken.resolve() == snapshot
    assert any(action["status"] == "ok" for action in result["recovery_actions"])


def test_c2c_materializes_candidate_config_overrides(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux"],
            "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
        }
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    candidate = {
        "id": "override_idea",
        "experiment_contract": {
            "config_overrides": {
                "train": {
                    "model": {
                        "soft_alignment_top_k": 2,
                        "soft_alignment_confidence_floor": 0.2,
                    }
                },
                "eval": {
                    "model": {
                        "rosetta_config": {
                            "soft_alignment_top_k": 2,
                            "soft_alignment_confidence_floor": 0.2,
                        }
                    }
                },
            }
        },
    }
    run_spec = adapter.materialize_candidate_configs(candidate)
    train = json.loads(Path(run_spec["train_config"]).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load(next(iter(run_spec["eval_configs"].values())).read_text(encoding="utf-8"))

    assert run_spec["has_executable_change"] is True
    assert train["model"]["soft_alignment_top_k"] == 2
    assert train["model"]["soft_alignment_confidence_floor"] == 0.2
    assert eval_cfg["model"]["rosetta_config"]["soft_alignment_top_k"] == 2
    assert eval_cfg["model"]["rosetta_config"]["soft_alignment_confidence_floor"] == 0.2


def test_c2c_materialized_train_configs_disable_wandb_without_service_token(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux"],
            "small_loop": {
                "eval_datasets": ["mmlu-redux"],
                "train_samples": 1,
                "gpu_ids": [0],
                "proxy_screen": {
                    "enabled": True,
                    "train_samples": 2,
                    "eval_datasets": ["mmlu-redux"],
                    "per_device_train_batch_size": 1,
                },
            },
        }
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    run_spec = adapter.materialize_candidate_configs({"id": "idea"})
    baseline_spec = adapter.materialize_proxy_baseline_configs()

    train = json.loads(Path(run_spec["train_config"]).read_text(encoding="utf-8"))
    proxy_train = json.loads(Path(run_spec["proxy_screen"]["train_config"]).read_text(encoding="utf-8"))
    baseline_train = json.loads(Path(baseline_spec["train_config"]).read_text(encoding="utf-8"))

    for payload in [train, proxy_train, baseline_train]:
        wandb_config = payload["output"]["wandb_config"]
        assert wandb_config["mode"] == "disabled"
        assert wandb_config["entity"] is None
    for payload in [proxy_train, baseline_train]:
        assert payload["training"]["per_device_train_batch_size"] == 1
        assert payload["training"]["gradient_accumulation_steps"] == 1

    combined_commands = "\n".join(
        [
            run_spec["commands"]["train"],
            run_spec["proxy_screen"]["commands"]["train"],
            baseline_spec["commands"]["train"],
        ]
    )
    assert "WANDB_DISABLED=true" in combined_commands
    assert "WANDB_SERVICE=" not in combined_commands
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in combined_commands


def test_c2c_proxy_batch_auto_uses_gpu_memory(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    snapshot = [
        {"index": 0, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 0},
        {"index": 1, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 0},
    ]
    monkeypatch.setattr(ExperimentRunner, "_gpu_snapshot", staticmethod(lambda: snapshot))
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux"],
            "small_loop": {
                "eval_datasets": ["mmlu-redux"],
                "gpu_ids": [0, 1],
                "proxy_screen": {"enabled": True, "train_samples": 2, "eval_datasets": ["mmlu-redux"]},
            },
        }
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    selection = ExperimentRunner(config).select_gpus({"gpu_ids": [0, 1], "max_gpus": 2})
    run_spec = adapter.materialize_candidate_configs({"id": "idea"}, selection)
    proxy_train = json.loads(Path(run_spec["proxy_screen"]["train_config"]).read_text(encoding="utf-8"))
    baseline = adapter.materialize_proxy_baseline_configs(selection)
    baseline_train = json.loads(Path(baseline["train_config"]).read_text(encoding="utf-8"))

    assert proxy_train["training"]["per_device_train_batch_size"] == 2
    assert baseline_train["training"]["per_device_train_batch_size"] == 2


def test_c2c_proxy_batch_explicit_override_wins(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    monkeypatch.setattr(
        ExperimentRunner,
        "_gpu_snapshot",
        staticmethod(lambda: [{"index": 0, "memory_total_mb": 24564, "memory_free_mb": 24000, "memory_used_mb": 0, "utilization_gpu": 0}]),
    )
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "datasets": ["mmlu-redux"],
            "small_loop": {
                "eval_datasets": ["mmlu-redux"],
                "gpu_ids": [0],
                "proxy_screen": {
                    "enabled": True,
                    "train_samples": 2,
                    "eval_datasets": ["mmlu-redux"],
                    "per_device_train_batch_size": 1,
                },
            },
        }
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    selection = ExperimentRunner(config).select_gpus({"gpu_ids": [0], "max_gpus": 1})
    run_spec = adapter.materialize_candidate_configs({"id": "idea"}, selection)
    proxy_train = json.loads(Path(run_spec["proxy_screen"]["train_config"]).read_text(encoding="utf-8"))

    assert proxy_train["training"]["per_device_train_batch_size"] == 1


def test_c2c_materialization_localizes_runtime_model_literals(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    dataset_adapter = repo / "rosetta/train/dataset_adapters.py"
    dataset_adapter.parent.mkdir(parents=True, exist_ok=True)
    dataset_adapter.write_text(
        'from transformers import AutoTokenizer\n'
        'TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")\n',
        encoding="utf-8",
    )
    local_model = tmp_path / "models/Qwen3-0.6B"
    local_model.mkdir(parents=True)
    config = {
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": "/usr/bin/python3",
            "model_map": {"Qwen/Qwen3-0.6B": str(local_model)},
            "datasets": ["mmlu-redux"],
            "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
        }
    }

    run_spec = C2CAdapter(tmp_path / "project", config).materialize_candidate_configs({"id": "idea"})

    updated = dataset_adapter.read_text(encoding="utf-8")
    assert "Qwen/Qwen3-0.6B" not in updated
    assert str(local_model) in updated
    assert run_spec["runtime_localization"]["status"] == "ok"
    assert run_spec["runtime_localization"]["files"][0]["path"] == "rosetta/train/dataset_adapters.py"


def test_experiment_runner_preserves_command_output_head_and_tail(tmp_path: Path) -> None:
    runner = ExperimentRunner({})
    command = (
        "python - <<'PY'\n"
        "import sys\n"
        "sys.stderr.write('TRACEBACK_START\\n' + 'x' * 15000 + '\\nROOT_CAUSE\\n')\n"
        "raise SystemExit(1)\n"
        "PY"
    )

    result = runner.run_step(name="fail", command=command, working_dir=tmp_path)
    stderr = result["attempts"][0]["stderr"]

    assert result["status"] == "failed"
    assert "TRACEBACK_START" in stderr
    assert "ROOT_CAUSE" in stderr
    assert "truncated" in stderr


def test_experiment_runner_times_out_process_group(tmp_path: Path) -> None:
    runner = ExperimentRunner({})
    command = (
        "python - <<'PY'\n"
        "import time\n"
        "print('started', flush=True)\n"
        "time.sleep(5)\n"
        "PY"
    )

    result = runner.run_step(name="slow", command=command, working_dir=tmp_path, retry_policy={"timeout_seconds": 1})
    attempt = result["attempts"][0]

    assert result["status"] == "failed"
    assert result["returncode"] == 124
    assert attempt["timed_out"] is True
    assert attempt["timeout_seconds"] == 1
    assert "timed out" in attempt["stderr"]


def test_c2c_debate_avoids_failed_feedback_ideas(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["ideation"] = {"debate": {"enabled": True, "rounds": 1}}
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    feedback = [
        {
            "kind": "c2c_feedback_summary",
            "failed_idea_ids": [
                "entropy_calibrated_span_gate",
                "headwise_alignment_confidence",
                "length_aware_topk_alignment",
            ],
            "failed_titles": [
                "Entropy-calibrated span confidence gate",
                "Headwise alignment confidence modulation",
                "Length-aware top-k soft alignment",
            ],
            "avoid_repeat_rules": ["Do not repeat this mechanism without addressing mmlu-redux regression."],
            "failure_modes": ["mmlu_regression"],
            "dataset_regressions": {"mmlu-redux": 2.7},
            "summary_text": "latest=c2c_failure_feedback:not_viable | reason=mmlu-redux regression",
        }
    ]

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=feedback,
    )
    idea_ids = {idea["id"] for idea in debate["selected_ideas"]}

    assert "entropy_calibrated_span_gate" not in idea_ids
    assert "headwise_alignment_confidence" not in idea_ids
    assert "length_aware_topk_alignment" not in idea_ids
    assert "verifier_guided_cache_acceptance" in idea_ids
    assert debate["negative_constraints"]["forbidden_idea_ids"]
    assert debate["negative_constraints"]["failure_feedback_rules"] == [
        "Do not repeat this mechanism without addressing mmlu-redux regression."
    ]
    assert debate["negative_constraints"]["failure_modes"] == ["mmlu_regression"]
    selected = debate["selected_ideas"][0]
    assert selected["failure_feedback_refs"]
    assert selected["failure_feedback_refs"][0]["source_type"] == "failure_feedback"
    assert "mmlu-redux regression" in selected["failure_feedback_refs"][0]["snippet"]
    assert selected["novelty_gate"]["status"] == "pass"
    assert selected["implementation_scope_gate"]["status"] == "pass"


def test_c2c_debate_emits_decision_chain(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["ideation"] = {"debate": {"enabled": True, "rounds": 1}}
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    assert "decision_chain" in debate
    assert debate["decision_chain"]["evidence"]
    assert debate["decision_chain"]["counterevidence"]
    assert debate["decision_chain"]["conclusion"]
    assert debate["selected_ideas"][0]["decision_chain"]["evidence"]
    assert debate["selected_ideas"][0]["decision_chain"]["counterevidence"]
    assert debate["selected_ideas"][0]["evidence_refs"]
    assert isinstance(debate["selected_ideas"][0]["evidence_refs"][0], dict)


def test_c2c_s1_codex_evidence_agent_blocks_without_fallback(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {"enabled": True}
    config["agents"] = {"s1_evidence_agent": {"max_json_repairs": 1, "timeout_seconds": 5}}
    project_root = tmp_path / "workspace" / "p"
    project_root.mkdir(parents=True)
    monkeypatch.setattr(literature_module.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    original_subprocess_run = literature_module.subprocess.run

    def fake_bad_codex(command, **kwargs):
        if not command or command[0] != "codex":
            return original_subprocess_run(command, **kwargs)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("still not json", encoding="utf-8")
        stdout = '{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174002"}\n' if "resume" not in command else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(literature_module.subprocess, "run", fake_bad_codex)

    result = literature_module._run_s1_codex_evidence_agent(
        project_root=project_root,
        config=config,
        prompt="return valid json",
        max_repairs=1,
        timeout_seconds=5,
    )

    assert result["status"] == "blocked"
    assert result["repair_count"] == 1
    assert "fallback" not in json.dumps(result)
    assert (project_root / "literature/c2c/s1_codex_events.jsonl").exists()


def test_c2c_s2_directional_planner_falls_back_without_real_llm(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py", "script/train/SFT_train.py", "test/test_aligner_span_overlap.py"],
    }
    config["agents"] = {"s2_directional_planner": {"resume_enabled": False}}
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_planner_fallback", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")

    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    result = PlanAgent(context).run()

    planned = result["plan"]["candidate_ideas"]
    assert planned[0]["id"] == "utility_predicted_cache_routing"
    assert result["plan"]["directional_planning"]["status"] == "fallback_no_real_llm"


def test_c2c_s2_directional_planner_uses_direction_variants(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py", "script/train/SFT_train.py", "test/test_aligner_span_overlap.py"],
    }
    config["agents"] = {"s2_directional_planner": {"resume_enabled": False}}
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_planner_variants", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")

    class PlannerLLM(ModelClient):
        def __init__(self, config, project_root=None):
            super().__init__(config, project_root=project_root)
            self.use_real_api = True
            self.prompts = []

        def generate_json_with_schema(self, **kwargs):
            self.prompts.append(kwargs.get("prompt", ""))
            return {
                "planner_summary": "Create two utility-routing implementation variants within the S1 direction.",
                "planning_mode": "same_direction_variant",
                "candidates": [
                    {
                        "id": "utility_router_soft_residual_variant",
                        "title": "Utility router soft residual variant",
                        "description": "Keep baseline transfer as the default path and use predicted utility to scale a residual correction rather than adding a hard accept/reject gate.",
                        "motivation": "Previous proxy failures collapsed all datasets, so the variant should preserve baseline coverage while only attenuating harmful residual transfer.",
                        "hypothesis": "A soft residual utility router improves proxy mean without lowering transfer coverage across mmlu-redux, ai2-arc, and openbookqa.",
                        "mechanism_type": "utility_predicted_cache_routing",
                        "paper_claim": "Receiver utility should modulate residual cache transfer instead of replacing the original C2C path.",
                        "why_baseline_fails": "The baseline lacks a downstream utility signal for residual cache injection.",
                        "expected_signature": {"primary": "utility-positive spans keep baseline coverage while harmful residuals shrink", "stats": ["utility_residual_scale", "baseline_transfer_coverage"]},
                        "experiment_contract": {
                            "config_overrides": {
                                "train": {"model": {"cache_routing_mode": "utility_soft_residual", "cache_routing_loss_weight": 0.05}},
                                "eval": {"model": {"rosetta_config": {"cache_routing_mode": "utility_soft_residual"}}},
                            }
                        },
                        "failure_avoidance": ["Do not add another hard gate", "Preserve baseline transfer coverage"],
                        "failure_feedback_refs": [{"source_type": "failure_feedback", "source_label": "proxy collapsed all datasets"}],
                    }
                ],
            }

    llm = PlannerLLM(config, project_root=paths.root)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), llm)
    result = PlanAgent(context).run()

    planned = result["plan"]["candidate_ideas"]
    assert result["plan"]["directional_planning"]["status"] == "ok"
    assert result["plan"]["directional_planning"]["memory_entry_count"] == 1
    assert planned[0]["id"] == "utility_router_soft_residual_variant"
    assert planned[0]["mechanism_type"] == "utility_predicted_cache_routing"
    assert planned[0]["selected"] is True
    assert planned[0]["s2_planner"]["source"] == "directional_planner"
    saved = json.loads((paths.root / "plan" / "candidate_ideas.json").read_text(encoding="utf-8"))
    assert saved[0]["id"] == "utility_router_soft_residual_variant"
    memory = json.loads((paths.root / "plan" / "s2_planner_memory.json").read_text(encoding="utf-8"))
    assert memory["entry_count"] == 1
    assert memory["entries"][0]["selected_candidate"]["id"] == "utility_router_soft_residual_variant"

    second = PlanAgent(context).run()

    assert second["plan"]["directional_planning"]["memory_entry_count"] == 2
    assert "utility_router_soft_residual_variant" in llm.prompts[-1]
    memory = json.loads((paths.root / "plan" / "s2_planner_memory.json").read_text(encoding="utf-8"))
    assert memory["entry_count"] == 2


def test_c2c_s2_resume_planner_uses_codex_session(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["codex_cli"] = {"use_resume": True, "sandbox": "read-only", "approval_policy": "never", "json_events": True}
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py", "script/train/SFT_train.py", "test/test_aligner_span_overlap.py"],
    }
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_resume_planner", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")

    monkeypatch.setattr("auto_research.agents.plan.shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    commands = []
    prompts = []

    def fake_run(command, **kwargs):
        commands.append(command)
        prompts.append(kwargs.get("input") or "")
        assert command[0] == "codex"
        assert "-s" in command and command[command.index("-s") + 1] == "read-only"
        assert "--json" in command
        assert command[-1] == "-"
        output_path = Path(command[command.index("--output-last-message") + 1])
        variant_id = "utility_resume_memory_variant" if "resume" in command else "utility_resume_soft_residual"
        payload = {
            "planner_summary": "Resume planner inspected memory and made a soft residual variant.",
            "planning_mode": "same_direction_variant",
            "candidates": [
                {
                    "id": variant_id,
                    "title": "Utility resume soft residual",
                    "description": "Use utility prediction to softly scale residual transfer while preserving baseline cache coverage.",
                    "motivation": "The previous direction collapsed all datasets, so preserve coverage and only modulate residual transfer.",
                    "hypothesis": "Soft residual utility routing avoids all-dataset collapse in cheap proxy.",
                    "mechanism_type": "utility_predicted_cache_routing",
                    "paper_claim": "Receiver utility should modulate residual cache transfer rather than hard-filtering spans.",
                    "why_baseline_fails": "The baseline lacks a downstream utility signal for residual transfer.",
                    "expected_signature": {
                        "primary": "residual scale changes without coverage collapse",
                        "stats": ["utility_residual_scale", "baseline_transfer_coverage"],
                    },
                    "experiment_contract": {
                        "config_overrides": {
                            "train": {"model": {"cache_routing_mode": variant_id}},
                            "eval": {"model": {"rosetta_config": {"cache_routing_mode": variant_id}}},
                        }
                    },
                    "failure_avoidance": ["preserve baseline coverage"],
                    "failure_feedback_refs": [{"source_type": "failure_feedback", "source_label": "memory"}],
                }
            ],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174111"}\n',
            stderr="",
        )

    import auto_research.agents.plan as plan_module

    monkeypatch.setattr(plan_module.subprocess, "run", fake_run)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    result = PlanAgent(context).run()

    assert result["plan"]["directional_planning"]["source"] == "codex_resume_planner"
    assert result["plan"]["directional_planning"]["session_id"] == "123e4567-e89b-12d3-a456-426614174111"
    assert result["plan"]["candidate_ideas"][0]["id"] == "utility_resume_soft_residual"
    sessions = yaml.safe_load((paths.root / "meta" / "codex_sessions.yaml").read_text(encoding="utf-8"))
    assert sessions["sessions"]["s2_planner:utility_predicted_cache_routing"]["session_id"] == "123e4567-e89b-12d3-a456-426614174111"
    events = (paths.root / "plan" / "logs" / "s2_planner_codex_events.jsonl").read_text(encoding="utf-8")
    assert "s2_planner:utility_predicted_cache_routing" in events
    assert "resume" not in commands[0]

    second = PlanAgent(context).run()

    assert second["plan"]["directional_planning"]["used_existing_session"] is True
    assert second["plan"]["candidate_ideas"][0]["id"] == "utility_resume_memory_variant"
    assert "resume" in commands[1]
    assert commands[1][commands[1].index("resume") + 1] == "123e4567-e89b-12d3-a456-426614174111"
    assert "utility_resume_soft_residual" in prompts[1]
    memory = json.loads((paths.root / "plan" / "s2_planner_memory.json").read_text(encoding="utf-8"))
    assert memory["entry_count"] == 2


def test_c2c_s2_resume_planner_resets_duplicate_session_but_keeps_memory(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["codex_cli"] = {"use_resume": True, "sandbox": "read-only", "approval_policy": "never", "json_events": True}
    config["agents"] = {"s2_directional_planner": {"session_reset_duplicate_streak": 1}}
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {"eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"], "gpu_ids": [0], "max_candidates": 2},
        "allowed_files": ["rosetta/model/projector.py", "rosetta/model/wrapper.py", "script/train/SFT_train.py", "test/test_aligner_span_overlap.py"],
    }
    config["code_patch"] = {"enabled": False}
    paths = init_workspace(config, "topic", project_id="proj_s2_resume_reset", simulate=False)
    ideas = default_c2c_ideas("topic", config["c2c"]["baseline"])
    ArtifactManager(paths.root).write_json("S1_literature", "ideas.json", ideas, artifact_type="ideas", summary="ideas")
    ArtifactManager(paths.root).write_json("S1_literature", "c2c/baseline_evidence.json", config["c2c"]["baseline"], artifact_type="baseline", summary="baseline")

    monkeypatch.setattr("auto_research.agents.plan.shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)

    def payload_for_variant(variant_id: str) -> dict:
        return {
            "planner_summary": "Duplicate reset test.",
            "planning_mode": "same_direction_variant",
            "candidates": [
                {
                    "id": variant_id,
                    "title": "Utility duplicate candidate",
                    "description": "Use utility prediction to softly scale residual transfer while preserving baseline cache coverage.",
                    "hypothesis": "Soft residual utility routing avoids all-dataset collapse in cheap proxy.",
                    "mechanism_type": "utility_predicted_cache_routing",
                    "paper_claim": "Receiver utility should modulate residual cache transfer rather than hard-filtering spans.",
                    "why_baseline_fails": "The baseline lacks a downstream utility signal for residual transfer.",
                    "expected_signature": {"primary": "residual scale changes without coverage collapse", "stats": ["utility_residual_scale"]},
                    "experiment_contract": {
                        "config_overrides": {
                            "train": {"model": {"cache_routing_mode": variant_id}},
                            "eval": {"model": {"rosetta_config": {"cache_routing_mode": variant_id}}},
                        }
                    },
                    "failure_avoidance": ["preserve baseline coverage"],
                }
            ],
        }

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps(payload_for_variant("utility_duplicate_candidate")), encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"123e4567-e89b-12d3-a456-426614174222"}\n',
            stderr="",
        )

    import auto_research.agents.plan as plan_module

    monkeypatch.setattr(plan_module.subprocess, "run", fake_run)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    first = PlanAgent(context).run()
    second = PlanAgent(context).run()

    assert first["plan"]["directional_planning"]["source"] == "codex_resume_planner"
    assert first["plan"]["candidate_ideas"][0]["id"] == "utility_duplicate_candidate"
    assert second["plan"]["directional_planning"]["status"] == "fallback_no_real_llm"
    assert second["plan"]["directional_planning"]["resume_planner"]["session_reset"] is True
    assert second["plan"]["directional_planning"]["resume_planner"]["session_reset_reason"] == "duplicate_output_streak"
    assert "resume" in commands[1]
    sessions = yaml.safe_load((paths.root / "meta" / "codex_sessions.yaml").read_text(encoding="utf-8"))
    assert "s2_planner:utility_predicted_cache_routing" not in sessions.get("sessions", {})
    memory = json.loads((paths.root / "plan" / "s2_planner_memory.json").read_text(encoding="utf-8"))
    assert memory["entry_count"] == 2
    assert memory["entries"][0]["selected_candidate"]["id"] == "utility_duplicate_candidate"
    events = (paths.root / "plan" / "logs" / "s2_planner_codex_events.jsonl").read_text(encoding="utf-8")
    assert "session_reset" in events


def test_c2c_novelty_report_rejects_pure_local_tuning() -> None:
    report = c2c_idea_novelty_report(
        {
            "id": "local_topk_tuning",
            "title": "Local top-k confidence floor tuning",
            "selected": True,
            "experiment_contract": {
                "primary_metric": "three_dataset_mean",
                "baseline": "base",
                "config_overrides": {
                    "train": {"model": {"soft_alignment_top_k": 2, "soft_alignment_confidence_floor": 0.2}},
                    "eval": {"model": {"rosetta_config": {"soft_alignment_top_k": 2}}},
                },
            },
        }
    )

    assert report["status"] == "reject"
    assert report["local_tuning_flags"]


def test_c2c_novelty_report_rejects_hard_gate_without_coverage_controls() -> None:
    report = c2c_idea_novelty_report(
        {
            "id": "hard_gate_stack",
            "title": "Utility hard gate stack",
            "description": "Add an additional hard gate that rejects transferred spans whenever utility is below a fixed threshold.",
            "mechanism_type": "utility_predicted_cache_routing",
            "paper_claim": "Cache routing should use utility.",
            "why_baseline_fails": "The baseline accepts harmful spans.",
            "expected_signature": {"primary": "fewer bad spans"},
            "ablation_plan": {"switch": "ablation_disable_hard_gate_stack"},
            "expected_files": ["rosetta/model/projector.py"],
            "experiment_contract": {"ablation_switch": "ablation_disable_hard_gate_stack"},
        }
    )

    assert report["status"] == "reject"
    assert report["hard_gate_stack_flags"]
    assert set(report["missing_required_fields"]) == {"coverage_diagnostics", "matched_coverage_ablation"}


def test_c2c_novelty_report_accepts_default_mechanism_idea() -> None:
    idea = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})[0]
    report = c2c_idea_novelty_report(idea)

    assert report["status"] == "pass"
    assert report["mechanism_type"] == "utility_predicted_cache_routing"
    assert "coverage_diagnostics" in report["signals"]
    assert "matched_coverage_ablation" in report["signals"]


def test_s2_gate_requires_coverage_controls(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "plan").mkdir(parents=True)
    (project / "plan" / "short_loop_plan.yaml").write_text("collector: c2c_small_loop\n", encoding="utf-8")
    ideas = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})
    (project / "plan" / "candidate_ideas.json").write_text(json.dumps(ideas), encoding="utf-8")
    plan = {
        "selected_idea": ideas[0],
        "candidate_ideas": ideas,
        "hypotheses": [],
        "baselines": [],
        "datasets": [],
        "metrics": [],
        "statistical_testing": {},
        "ablation_matrix": [],
        "task_graph": {},
        "resource_budget": {"peak_concurrent_gpus": 0},
        "execution": {
            "collector": "c2c_small_loop",
            "min_delta_to_pass": 0.1,
            "max_dataset_regression": 2.0,
            "selected_gpu_ids": [],
        },
        "acceptance_criteria": {
            "minimum_mean_delta": 0.1,
            "max_dataset_regression": 2.0,
        },
        "reviewer_risk_controls": {"top_concerns": []},
    }
    (project / "plan" / "plan.yaml").write_text(yaml.safe_dump(plan), encoding="utf-8")

    report = S2GateValidator(project, {}).validate()

    assert report.status == "NEEDS_RETRY"
    check_names = {check.name for check in report.checks if check.status == "NEEDS_RETRY"}
    assert "c2c_coverage_control_requirements" in check_names
    assert "c2c_matched_coverage_ablation" in check_names


def test_c2c_debate_structures_fallback_refs(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["ideation"] = {"debate": {"enabled": True, "rounds": 1}}
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    fallback_idea = debate["selected_ideas"][0]
    assert isinstance(fallback_idea["evidence_refs"][0], dict)
    assert fallback_idea["evidence_refs"][0]["source_path"]
    assert fallback_idea["counterevidence_refs"][0]["source_type"] in {"repo_artifact", "summary"}
    assert fallback_idea["code_refs"][0]["source_type"] == "code"


def test_c2c_debate_parallel_timeout_falls_back(tmp_path: Path, monkeypatch) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = True
    config["llm"]["timeout_seconds"] = 20
    config["ideation"] = {"debate": {"enabled": True, "rounds": 1, "parallel": True, "agent_timeout_seconds": 1}}
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    context.llm.use_real_api = True

    def slow_worker(*args, **kwargs):
        time.sleep(30)

    monkeypatch.setattr("auto_research.agents.debate._run_role_worker", slow_worker)

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    statuses = [output.get("status") for output in debate["rounds"][0]["outputs"]]
    assert statuses == ["timeout_fallback"] * 6
    assert debate["selected_ideas"]
    progress_path = tmp_path / "workspace" / "p" / "literature" / "c2c" / "idea_debate_progress.jsonl"
    assert progress_path.exists()
    assert "timeout_fallback" in progress_path.read_text(encoding="utf-8")


def test_c2c_debate_role_specific_timeout_and_recovery_flag(tmp_path: Path, monkeypatch) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = True
    config["ideation"] = {
        "debate": {
            "enabled": True,
            "rounds": 2,
            "parallel": True,
            "agent_timeout_seconds": 1,
            "role_timeout_seconds": {"method_inventor": 2},
        }
    }
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    context.llm.use_real_api = True

    def role_worker(queue, config, project_root_text, role, context_payload, prior_round, round_idx, fallback):
        if role == "method_inventor" and round_idx == 1:
            time.sleep(1.4)
        if role == "systems_feasibility" and round_idx == 1:
            time.sleep(3)
        output = dict(fallback)
        output["status"] = "ok"
        queue.put({"status": "ok", "output": output})

    monkeypatch.setattr("auto_research.agents.debate._run_role_worker", role_worker)

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    assert debate["rounds"][0]["outputs"][2]["role"] == "method_inventor"
    assert debate["rounds"][0]["outputs"][2]["status"] == "ok"
    assert debate["rounds"][0]["outputs"][4]["role"] == "systems_feasibility"
    assert debate["rounds"][0]["outputs"][4]["status"] == "timeout_fallback"
    assert debate["rounds"][1]["outputs"][4]["status"] == "ok"
    assert any(
        flag.get("type") == "gpt_recovered_after_timeout" and flag.get("role") == "systems_feasibility"
        for flag in debate["quality_flags"]
    )
    progress_path = tmp_path / "workspace" / "p" / "literature" / "c2c" / "idea_debate_progress.jsonl"
    progress = progress_path.read_text(encoding="utf-8")
    assert '"role": "method_inventor"' in progress
    assert '"timeout_seconds": 2' in progress
    assert "systems_feasibility timed out after 1s" in progress


def test_c2c_debate_meta_timeout_falls_back(tmp_path: Path, monkeypatch) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = True
    config["ideation"] = {
        "debate": {
            "enabled": True,
            "rounds": 1,
            "parallel": True,
            "agent_timeout_seconds": 5,
            "meta_timeout_seconds": 1,
        }
    }
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    context.llm.use_real_api = True

    def fast_role_worker(queue, config, project_root_text, role, context_payload, prior_round, round_idx, fallback):
        queue.put({"status": "ok", "output": fallback})

    def slow_meta_worker(*args, **kwargs):
        time.sleep(30)

    monkeypatch.setattr("auto_research.agents.debate._run_role_worker", fast_role_worker)
    monkeypatch.setattr("auto_research.agents.debate._run_meta_worker", slow_meta_worker)

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    assert debate["meta_judge"]["status"] == "timeout_fallback"
    assert any(flag.get("type") == "meta_timeout_fallback" for flag in debate["quality_flags"])
    assert debate["selected_ideas"]
    progress_path = tmp_path / "workspace" / "p" / "literature" / "c2c" / "idea_debate_progress.jsonl"
    assert "meta_judge" in progress_path.read_text(encoding="utf-8")


def test_c2c_debate_meta_receives_compressed_round_summaries(tmp_path: Path, monkeypatch) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = True
    config["ideation"] = {
        "debate": {
            "enabled": True,
            "rounds": 2,
            "parallel": False,
            "agent_timeout_seconds": 5,
            "meta_timeout_seconds": 5,
        }
    }
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    context.llm.use_real_api = True

    def fake_role(role, context_payload, prior_round, round_idx):
        return {
            "role": role,
            "status": "ok",
            "score": 7,
            "claims": [f"{role} claim"],
            "evidence_refs": [{"source_type": "paper", "source_label": "paper"}],
            "counterevidence_refs": [{"source_type": "failure_feedback", "source_label": "feedback"}],
            "code_refs": [{"source_type": "code", "source_label": "code"}],
            "failure_feedback_refs": [{"source_type": "failure_feedback", "source_label": "feedback"}],
            "proposed_ideas": [{"id": "idea_a", "title": "Idea A", "selected": role == "literature_scout", "novelty_score": 7, "feasibility_score": 8}],
            "decision_chain": {"evidence": ["e1"], "counterevidence": ["c1"], "conclusion": "ok"},
            "risks": ["r1"],
        }

    captured = {}

    def fake_meta(context_payload, round_summaries, fallback_ideas, fallback):
        captured["round_summaries"] = round_summaries
        return fallback

    monkeypatch.setattr("auto_research.agents.debate.MultiAgentReasoningService._run_role", lambda self, role, context_payload, prior_round, round_idx: fake_role(role, context_payload, prior_round, round_idx))
    monkeypatch.setattr("auto_research.agents.debate.MultiAgentReasoningService._run_meta_judge_sync", lambda self, context_payload, round_summaries, fallback_ideas, fallback: fake_meta(context_payload, round_summaries, fallback_ideas, fallback))

    debate = MultiAgentReasoningService(context).run_c2c_debate(
        topic="cross tokenizer cache",
        repo_card={},
        paper_cards=[],
        rebuttal_matrix={"top_concerns": ["failure_modes_ood"]},
        negative_memory={},
        baseline={"name": "base", "mean": 50.0, "datasets": {}},
        feedback=[],
    )

    assert captured["round_summaries"]
    assert captured["round_summaries"][0]["role_summaries"][0]["top_evidence"]
    assert captured["round_summaries"][0]["selected_idea_ids"] == ["idea_a"]
    assert debate["selected_ideas"]


def test_c2c_feedback_bundle_expands_round_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    feedback_dir = project_root / "literature" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    round_payload = {
        "created_at": "2026-05-19T00:00:00Z",
        "project_id": "proj_x",
        "iteration": 2,
        "kind": "c2c_feedback_summary",
        "summary_entry": {
            "timestamp": "2026-05-19T00:00:00Z",
            "project_id": "proj_x",
            "iteration": 2,
            "kind": "c2c_feedback_summary",
            "failed_idea_ids": ["idea_a"],
            "failed_titles": ["Idea A"],
            "avoid_repeat_rules": ["Avoid A"],
            "summary_text": "latest=c2c_failure_feedback:not_viable",
        },
        "entries": [
            {
                "timestamp": "2026-05-19T00:00:00Z",
                "project_id": "proj_x",
                "iteration": 2,
                "kind": "c2c_failure_feedback",
                "idea_id": "idea_a",
                "title": "Idea A",
                "decision": "not_viable",
                "failure_mode": "not_viable",
                "reason": "bad",
                "avoid_repeat_rule": "Avoid A",
            }
        ],
        "iteration_traces": [
            {
                "timestamp": "2026-05-19T00:00:01Z",
                "from_stage": "S3_experiment",
                "to_stage": "S1_literature",
                "iteration": 2,
                "reason": "bad",
                "result_status": "not_viable",
            }
        ],
        "feedback_items": [
            {
                "timestamp": "2026-05-19T00:00:00Z",
                "project_id": "proj_x",
                "iteration": 2,
                "kind": "c2c_feedback_summary",
                "failed_idea_ids": ["idea_a"],
                "failed_titles": ["Idea A"],
                "avoid_repeat_rules": ["Avoid A"],
            }
        ],
    }
    (feedback_dir / "failed_ideas_round_002.json").write_text(json.dumps(round_payload), encoding="utf-8")

    bundle = load_c2c_feedback_bundle(project_root)
    assert bundle["summary"]["failed_idea_ids"] == ["idea_a"]
    assert bundle["entries"]
    assert bundle["iteration_traces"]
    assert any(item.get("kind") == "c2c_failure_feedback" for item in bundle["feedback_items"])


def test_c2c_feedback_bundle_includes_direction_scorecard_method_view(tmp_path: Path) -> None:
    project_root = tmp_path / "project_direction_scorecard"
    path = project_root / "plan" / "direction_scorecard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
            "schema_version": "c2c_direction_scorecard_v1",
            "project_id": "project_direction_scorecard",
            "current_direction_id": "utility_predicted_cache_routing",
            "current_direction": {
                "direction_id": "utility_predicted_cache_routing",
                "title": "Utility-predicted cache routing",
                "mechanism_type": "utility_predicted_cache_routing",
                "summary": {
                    "status": "budget_exhausted",
                    "attempt_count": 5,
                    "same_direction_failure_count": 5,
                    "same_direction_failure_budget": 5,
                    "best_proxy_delta": -1.2,
                    "positive_dataset_signal_attempts": 1,
                    "runtime_stable_attempts": 4,
                    "low_patch_risk_attempts": 4,
                    "all_dataset_collapse_attempts": 3,
                    "health_score": -8.4,
                    "direction_quality": "poor_direction_evidence",
                },
                "s1_feedback": {
                    "recommendation": "return_to_s1_new_direction",
                    "conclusion": "Direction failed after five attempts with repeated all-dataset collapse.",
                    "avoid_repeat_rule": "Do not repeat this S1 direction without a mechanism-level change.",
                },
            },
            }
        ),
        encoding="utf-8",
    )

    bundle = load_c2c_feedback_bundle(project_root, view="method")

    entries = bundle["entries"]
    scorecards = [item.get("direction_scorecard") for item in entries if item.get("direction_scorecard")]
    assert scorecards
    assert scorecards[0]["direction_id"] == "utility_predicted_cache_routing"
    assert scorecards[0]["summary"]["best_proxy_delta"] == -1.2
    assert scorecards[0]["s1_feedback"]["recommendation"] == "return_to_s1_new_direction"


def test_c2c_feedback_bundle_includes_proxy_calibration_method_view(tmp_path: Path) -> None:
    project_root = tmp_path / "project_proxy_calibration"
    path = project_root / "experiment" / "results" / "proxy_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "c2c_proxy_calibration_v1",
                "project_id": "project_proxy_calibration",
                "summary": {
                    "candidate_count": 2,
                    "proxy_false_positive_count": 1,
                    "proxy_false_positive_rate": 0.5,
                    "dataset_error_summary": {
                        "mmlu-redux": {
                            "mean_abs_proxy_full_delta_error": 2.4,
                            "max_abs_proxy_full_delta_error": 2.4,
                            "misprediction_count": 1,
                            "count": 1,
                        }
                    },
                    "mechanism_false_positive_summary": {
                        "utility_predicted_cache_routing": {
                            "count": 1,
                            "false_positive_count": 1,
                            "false_positive_rate": 1.0,
                        }
                    },
                },
                "current_iteration": {
                    "iteration": 3,
                    "acceptance_passed": False,
                    "candidate_count": 1,
                    "proxy_false_positive_count": 1,
                    "proxy_false_positive_rate": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = load_c2c_feedback_bundle(project_root, view="method")

    calibrations = [item.get("proxy_calibration") for item in bundle["entries"] if item.get("proxy_calibration")]
    assert calibrations
    assert calibrations[0]["summary"]["proxy_false_positive_rate"] == 0.5
    assert calibrations[0]["summary"]["dataset_error_summary"]["mmlu-redux"]["misprediction_count"] == 1
    assert calibrations[0]["summary"]["mechanism_false_positive_summary"]["utility_predicted_cache_routing"]["false_positive_rate"] == 1.0


def test_c2c_feedback_bundle_summary_builder() -> None:
    bundle = build_c2c_feedback_bundle(
        [
            {
                "timestamp": "2026-05-19T00:00:00Z",
                "project_id": "proj_x",
                "iteration": 2,
                "kind": "c2c_failure_feedback",
                "idea_id": "idea_a",
                "title": "Idea A",
                "decision": "not_viable",
                "failure_mode": "not_viable",
                "reason": "bad",
                "avoid_repeat_rule": "Avoid A",
            }
        ],
        project_id="proj_x",
        iteration=2,
        traces=[
            {
                "timestamp": "2026-05-19T00:00:01Z",
                "from_stage": "S3_experiment",
                "to_stage": "S1_literature",
                "iteration": 2,
                "reason": "bad",
                "result_status": "not_viable",
            }
        ],
        sources=["meta/negative_memory.jsonl"],
    )
    assert bundle["summary"]["failed_idea_ids"] == ["idea_a"]
    assert bundle["summary_entry"]["kind"] == "c2c_feedback_summary"
    assert bundle["feedback_items"][0]["kind"] == "c2c_feedback_summary"
    assert bundle["feedback_items"][1]["idea_id"] == "idea_a"


def test_c2c_feedback_bundle_preserves_failure_attribution() -> None:
    bundle = build_c2c_feedback_bundle(
        [
            {
                "timestamp": "2026-05-19T00:00:00Z",
                "project_id": "proj_x",
                "iteration": 2,
                "kind": "c2c_failure_feedback",
                "idea_id": "idea_a",
                "title": "Idea A",
                "decision": "not_viable",
                "failure_mode": "not_viable",
                "failure_attribution": {
                    "primary_failure": "mmlu-redux_regression",
                    "dragging_datasets": [
                        {"dataset": "mmlu-redux", "sample_family": "multi_domain_knowledge_reasoning", "regression": 3.2}
                    ],
                    "sample_type_failures": [
                        {"sample_family": "multi_domain_knowledge_reasoning", "dataset": "mmlu-redux"}
                    ],
                    "mixed_gain_patterns": ["openbookqa_gain_mmlu_redux_regression"],
                    "patch_risk": {
                        "risk_labels": ["projector_mechanism_changed"],
                        "risk_files": [{"path": "rosetta/model/projector.py", "reasons": ["projector mechanism changed"]}],
                    },
                },
            }
        ],
        project_id="proj_x",
        iteration=2,
    )

    summary = bundle["summary"]
    assert summary["dragging_datasets"][0]["dataset"] == "mmlu-redux"
    assert summary["sample_type_failures"] == ["multi_domain_knowledge_reasoning"]
    assert summary["patch_risk_files"] == ["rosetta/model/projector.py"]
    assert summary["mixed_gain_patterns"] == ["openbookqa_gain_mmlu_redux_regression"]
    assert "dragging_datasets=mmlu-redux" in summary["summary_text"]


def test_c2c_train_failure_with_checkpoint_continues_eval(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["llm"]["use_real_api"] = False
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {"enabled": False},
        },
        "allowed_files": ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"],
        "allowed_prefixes": ["recipe/", "local/auto_research_runs/"],
    }
    paths = init_workspace(config, "topic", project_id="proj_recovery", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        if name == "train":
            run_id = "idea"
            final = repo / "local" / "auto_research_runs" / run_id / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
            return {"step": name, "status": "failed", "attempts": [{"stdout": "", "stderr": "train crashed", "returncode": 1}], "returncode": 1}
        if name.startswith("eval_"):
            dataset = name.replace("eval_", "")
            out = repo / "local" / "auto_research_runs" / "idea" / "results" / dataset
            out.mkdir(parents=True, exist_ok=True)
            (out / f"Rosetta_{dataset}_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": dataset, "answer_method": "generate", "overall_accuracy": 0.51}),
                encoding="utf-8",
            )
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    candidate = {
        "id": "idea",
        "title": "Idea",
        "hypothesis": "h",
        "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
    }
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate=candidate,
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )
    assert result["command_status"] in {"partial", "ok"}
    assert result["metrics"]["mean"] == 51.0
    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))
    assert any(action["action"] == "skip_failed_train_with_existing_final_checkpoint" for action in state["recovery_actions"])


def test_deterministic_s3_blocks_noop_candidate(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {"enabled": False},
        },
        "allowed_files": ["rosetta/model/aligner.py"],
        "allowed_prefixes": ["recipe/", "local/auto_research_runs/"],
    }
    paths = init_workspace(config, "topic", project_id="proj_noop", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    def fail_run_step(**kwargs):
        raise AssertionError("noop candidate must be blocked before running commands")

    monkeypatch.setattr(agent.runner, "run_step", fail_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={"id": "noop", "title": "Noop"},
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    assert result["decision"] == "blocked"
    assert result["command_status"] == "blocked"
    assert result["has_executable_change"] is False


def test_s3_applies_frozen_patch_archives_snapshot_and_does_not_call_llm(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
    }
    paths = init_workspace(config, "topic", project_id="proj_s3_patch", simulate=True)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))

    class BombLLM:
        use_real_api = True

        def generate(self, **kwargs):
            raise AssertionError("S3 execution must not call LLM")

        def generate_json(self, **kwargs):
            raise AssertionError("S3 execution must not call LLM")

        def generate_json_with_schema(self, **kwargs):
            raise AssertionError("S3 execution must not call LLM")

    context.llm = BombLLM()
    patch_dir = paths.root / "plan/code_patches/frozen_idea"
    patch_dir.mkdir(parents=True)
    aligner = repo / "rosetta/model/aligner.py"
    patch_payload = {
        "schema_version": 1,
        "candidate_id": "frozen_idea",
        "title": "Frozen Idea",
        "operations": [
            {
                "op": "replace_file",
                "path": "rosetta/model/aligner.py",
                "old_sha256": sha256_file(aligner),
                "new": "VALUE = 'frozen patch'\n",
            }
        ],
        "changed_files": ["rosetta/model/aligner.py"],
        "rationale": "Frozen test patch.",
    }
    (patch_dir / "patch.json").write_text(json.dumps(patch_payload), encoding="utf-8")
    candidate = {
        "id": "frozen_idea",
        "title": "Frozen Idea",
        "hypothesis": "h",
        "code_patch": {
            "status": "ok",
            "patch_json": "plan/code_patches/frozen_idea/patch.json",
            "changed_files": ["rosetta/model/aligner.py"],
            "has_executable_change": True,
        },
    }

    agent = ExperimentAgent(context)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate=candidate,
        index=0,
        simulate=True,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    snapshot = paths.root / "experiment/code_snapshots/frozen_idea/rosetta/model/aligner.py"
    manifest = paths.root / "experiment/code_snapshots/frozen_idea/manifest.json"
    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))

    assert result["patch_result"]["status"] == "applied"
    assert result["code_snapshot"]["status"] == "ok"
    assert result["command_status"] == "mocked"
    assert snapshot.exists()
    assert "frozen patch" in snapshot.read_text(encoding="utf-8")
    assert manifest.exists()
    assert state["code_snapshot"]["status"] == "ok"


def test_s3_runs_ablation_switch_disabled_eval(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["experiment"]["disable_llm_during_execution"] = True
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {"enabled": False},
        },
        "allowed_prefixes": ["local/auto_research_runs/"],
    }
    paths = init_workspace(config, "topic", project_id="proj_s3_ablation", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    seen_steps: list[str] = []

    def write_summary(root: Path, dataset: str, accuracy: float) -> None:
        out = root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f"Rosetta_{dataset}_generate_summary.json").write_text(
            json.dumps({"model": "Rosetta", "dataset": dataset, "answer_method": "generate", "overall_accuracy": accuracy}),
            encoding="utf-8",
        )

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del command, working_dir, retry_policy
        seen_steps.append(name)
        run_root = repo / "local" / "auto_research_runs" / "mechanism"
        if name == "train":
            final = run_root / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        elif name.startswith("eval_"):
            dataset = name.replace("eval_", "")
            write_summary(run_root / "results", dataset, 0.55)
        elif name.startswith("ablation_eval_"):
            dataset = name.replace("ablation_eval_", "")
            write_summary(run_root / "ablation_disabled" / "results", dataset, 0.50)
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    candidate = {
        "id": "mechanism",
        "title": "Mechanism",
        "hypothesis": "h",
        "experiment_contract": {
            "ablation_switch": "disable_mechanism",
            "config_overrides": {
                "train": {"model": {"mechanism_enabled": True}},
                "eval": {"model": {"rosetta_config": {"mechanism_enabled": True}}},
            },
        },
    }

    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate=candidate,
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    ablation = result["ablation"]
    comparison = ablation["comparison"]
    disabled_eval = yaml.safe_load((repo / "local/auto_research_runs/mechanism/ablation_disabled/eval_mmlu-redux.yaml").read_text(encoding="utf-8"))

    assert "ablation_eval_mmlu-redux" in seen_steps
    assert result["metrics"]["mean"] == 55.0
    assert ablation["status"] == "ok"
    assert ablation["metrics"]["mean"] == 50.0
    assert comparison["enabled_minus_disabled_mean"] == 5.0
    assert comparison["mechanism_supported"] is True
    assert disabled_eval["model"]["rosetta_config"]["disable_mechanism"] is True
    assert disabled_eval["output"]["output_dir"] == "local/auto_research_runs/mechanism/ablation_disabled/results/mmlu-redux"


def test_c2c_ablation_payload_and_verification_report_supported(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    best = {
        "id": "mechanism",
        "title": "Mechanism",
        "decision": "candidate_win",
        "command_status": "ok",
        "metrics": {"mean": 55.0, "datasets": {"mmlu-redux": 55.0}},
        "delta_vs_baseline": 5.0,
        "worst_dataset_regression": 0.0,
        "ablation": {
            "enabled": True,
            "status": "ok",
            "switch": "disable_mechanism",
            "metrics": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
            "comparison": {
                "status": "ok",
                "enabled_mean": 55.0,
                "disabled_mean": 50.0,
                "enabled_minus_disabled_mean": 5.0,
                "dataset_enabled_minus_disabled": {"mmlu-redux": 5.0},
                "mechanism_supported": True,
            },
        },
    }
    payload = {
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "best_candidate": best,
        "candidate_results": [best],
    }

    ablation_payload = ExperimentAgent._c2c_ablation_payload(payload, adapter)
    verification = ExperimentAgent._c2c_verification_md(best, 50.0, [best], 0.1, 2.0)

    assert ablation_payload["status"] == "ok"
    assert ablation_payload["best_supported"] is True
    assert ablation_payload["best_delta_enabled_vs_disabled"] == 5.0
    assert ablation_payload["candidate_ablations"][0]["supported"] is True
    assert "H2: supported" in verification
    assert "disable_mechanism" in verification


def test_c2c_ablation_payload_distinguishes_declared_switch_from_reached_stage(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=True)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
    }
    adapter = C2CAdapter(tmp_path / "project", config)
    candidate = {
        "id": "proxy_rejected",
        "title": "Proxy Rejected",
        "decision": "proxy_rejected",
        "command_status": "proxy_rejected",
        "experiment_contract": {"ablation_switch": "disable_proxy_rejected"},
        "metrics": None,
        "ablation": {"enabled": False, "status": "skipped", "reason": "not run"},
    }

    ablation_payload = ExperimentAgent._c2c_ablation_payload(
        {"baseline": config["c2c"]["baseline"], "best_candidate": None, "candidate_results": [candidate]},
        adapter,
    )

    assert ablation_payload["status"] == "skipped"
    assert ablation_payload["reason"] == "candidate ablation switches were declared, but no candidate reached full eval before ablation"
    assert ablation_payload["candidate_ablations"][0]["declared_switch"] == "disable_proxy_rejected"
    assert ablation_payload["candidate_ablations"][0]["reached_ablation_stage"] is False


def test_c2c_failure_analysis_accepts_grouped_posthoc_suggestions() -> None:
    payload = {
        "acceptance": {"passed": False, "reason": "no candidate metrics"},
        "best_candidate": None,
    }
    posthoc = {
        "failure_modes": [{"observed": "Patch rejected before preflight."}],
        "next_round_suggestions": {
            "S1": [{"constraint": "Only submit contract-safe candidates."}],
            "S2.5": [{"action": "Keep new knobs fixed or explicitly activated."}],
        },
        "avoid_repeat_rules": {"S2": [{"rule": "Do not repeat config-unsafe patches."}]},
    }

    markdown = ExperimentAgent._c2c_failure_analysis_md(payload, posthoc)

    assert "Patch rejected before preflight." in markdown
    assert "S1: Only submit contract-safe candidates." in markdown
    assert "S2.5: Keep new knobs fixed or explicitly activated." in markdown
    assert "S2: Do not repeat config-unsafe patches." in markdown


def test_s3_rejects_validation_failed_code_patch_before_training(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
    }
    paths = init_workspace(config, "topic", project_id="proj_s3_bad_patch", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    def fail_run_step(**kwargs):
        raise AssertionError("validation_failed patch must be blocked before training")

    monkeypatch.setattr(agent.runner, "run_step", fail_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "bad_patch",
            "title": "Bad Patch",
            "code_patch": {"status": "validation_failed", "reason": "py_compile failed"},
            "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    assert result["decision"] == "patch_rejected"
    assert result["command_status"] == "patch_rejected"
    assert result["patch_result"]["patch_status"] == "validation_failed"
    assert "py_compile failed" in result["patch_result"]["errors"][0]


def test_c2c_static_proxy_rejects_evaluator_patch_before_training(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "static",
                "reject_eval_code_changes": True,
                "reject_if_no_executable_change": True,
            },
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_static", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    patch_dir = paths.root / "plan/code_patches/eval_risk"
    patch_dir.mkdir(parents=True)
    evaluator = repo / "script/evaluation/unified_evaluator.py"
    (patch_dir / "patch.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": "eval_risk",
                "title": "Eval Risk",
                "operations": [
                    {
                        "op": "replace_file",
                        "path": "script/evaluation/unified_evaluator.py",
                        "old_sha256": sha256_file(evaluator),
                        "new": "print('changed eval')\n",
                    }
                ],
                "changed_files": ["script/evaluation/unified_evaluator.py"],
            }
        ),
        encoding="utf-8",
    )
    agent = ExperimentAgent(context)
    seen_steps = []

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        seen_steps.append(name)
        if name == "train" or name.startswith("eval_"):
            raise AssertionError("proxy rejected candidate must not reach full train/eval")
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "eval_risk",
            "title": "Eval Risk",
            "code_patch": {
                "status": "ok",
                "patch_json": "plan/code_patches/eval_risk/patch.json",
                "changed_files": ["script/evaluation/unified_evaluator.py"],
                "has_executable_change": True,
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    state = json.loads(Path(result["run_state_path"]).read_text(encoding="utf-8"))
    assert result["decision"] == "proxy_repairable"
    assert result["command_status"] == "proxy_repairable"
    assert result["proxy_screen"]["status"] == "repairable_proxy_risk"
    assert result["failure_attribution"]["primary_failure"] == "repairable_proxy_risk_before_full_training"
    assert "evaluation_code_changed" in result["failure_attribution"]["patch_risk"]["risk_labels"]
    assert "train" not in seen_steps
    assert state["train"] is None


def test_c2c_proxy_carries_instrumentation_quality_repair_request(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "static",
                "require_proxy_metrics": False,
                "require_paired_baseline": False,
                "soft_proxy_mean_delta": None,
                "soft_max_proxy_dataset_regression": None,
                "soft_min_proxy_score": None,
            },
        },
    }
    config["code_patch"] = {
        "enabled": True,
        "dynamic_whitelist": {
            "include_prefixes": ["rosetta/"],
            "include_extensions": [".py"],
            "exclude_prefixes": [],
            "exclude_extensions": [],
            "include_root_globs": [],
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_quality_repair_proxy", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    patch_dir = paths.root / "plan" / "code_patches" / "quality"
    patch_dir.mkdir(parents=True)
    projector = repo / "rosetta/model/projector.py"
    (patch_dir / "patch.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": "quality",
                "title": "Quality",
                "operations": [
                    {
                        "op": "replace_file",
                        "path": "rosetta/model/projector.py",
                        "old_sha256": sha256_file(projector),
                        "new": "VALUE = 'quality runnable mechanism'\n",
                    }
                ],
                "changed_files": ["rosetta/model/projector.py"],
                "mechanism_review": {
                    "status": "ok",
                    "soft_issues": ["missing_coverage_diagnostics_evidence"],
                    "quality_repair": {
                        "needed": False,
                        "deferred": True,
                        "mode": "paperization_after_effect",
                        "issues": ["missing_coverage_diagnostics_evidence"],
                        "constraints": ["Only add instrumentation."],
                        "ablation_switch": "ablation_disable_quality",
                    },
                },
                "quality_score": {"score": 70, "soft_issues": ["missing_coverage_diagnostics_evidence"]},
            }
        ),
        encoding="utf-8",
    )
    agent = ExperimentAgent(context)
    monkeypatch.setattr(agent.runner, "run_step", lambda **kwargs: {"step": kwargs["name"], "status": "ok", "returncode": 0, "stdout": "", "stderr": "", "attempts": []})
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "quality",
            "title": "Quality",
            "code_patch": {
                "status": "ok",
                "patch_json": "plan/code_patches/quality/patch.json",
                "changed_files": ["rosetta/model/projector.py"],
                "has_executable_change": True,
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    assert result["proxy_screen"]["status"] == "passed"
    assert result["proxy_screen"]["quality_repair"]["needed"] is False
    assert result["proxy_screen"]["quality_repair"]["deferred"] is True
    assert result["proxy_screen"]["quality_repair"]["repair_route"] == "paperization"
    assert result["proxy_screen"]["quality_repair"]["mode"] == "paperization_after_effect"
    assert result["failure_attribution"]["quality_repair"]["acceptance_guard"]["rerun_same_proxy_subset"] is True


def test_s3_reuses_completed_proxy_rejected_run_state_without_rerun(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "train_samples": 1,
                "eval_limit": 1,
                "eval_datasets": ["mmlu-redux"],
                "min_proxy_mean_delta": -0.3,
                "require_paired_baseline": True,
                "baseline_cache_path": "experiment/results/c2c_proxy_baseline.json",
            },
        },
        "allowed_files": ["rosetta/model/projector.py"],
        "allowed_prefixes": ["recipe/", "local/auto_research_runs/"],
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_resume_reuse", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    adapter = C2CAdapter(paths.root, config)
    gpu_selection = agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1})
    candidate = {
        "id": "proxy_resume",
        "title": "Proxy Resume",
        "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
    }
    run_spec = adapter.materialize_candidate_configs(candidate, gpu_selection)
    patch_fingerprint = ExperimentAgent._c2c_patch_fingerprint(adapter, {"status": "skipped", "changed_files": []}, run_spec)
    state_path = Path(run_spec["run_state_path"])
    state_path.write_text(
        json.dumps(
            {
                "candidate_id": "proxy_resume",
                "run_id": "proxy_resume",
                "preflight": {"status": "ok"},
                "proxy_screen": {
                    "enabled": True,
                    "status": "rejected",
                    "reason": "proxy mean delta -1.0 below hard threshold -0.3",
                    "metrics": {"mean": 49.0, "datasets": {"mmlu-redux": 49.0}},
                    "baseline_metrics": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
                    "proxy_delta_vs_baseline": -1.0,
                    "proxy_dataset_deltas": {"mmlu-redux": -1.0},
                    "proxy_dataset_regressions": {"mmlu-redux": 1.0},
                    "proxy_worst_dataset_regression": 1.0,
                    "proxy_score": -1.5,
                    "patch_fingerprint": patch_fingerprint,
                },
                "metrics": None,
                "attempts": [],
                "frozen_hashes": run_spec["frozen_hashes"],
                "config_overrides": run_spec["config_overrides"],
                "has_executable_change": True,
                "patch_fingerprint": patch_fingerprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fail_run_step(**kwargs):
        raise AssertionError("completed proxy_rejected run_state should be reused")

    monkeypatch.setattr(agent.runner, "run_step", fail_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=adapter,
        candidate=candidate,
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=gpu_selection,
    )

    assert result["decision"] == "proxy_rejected"
    assert result["command_status"] == "proxy_rejected"
    assert result["proxy_screen"]["proxy_delta_vs_baseline"] == -1.0
    assert result["patch_fingerprint"] == patch_fingerprint
    assert result["metrics"] is None
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["proxy_screen"]["status"] == "rejected"
    assert saved["patch_fingerprint"] == patch_fingerprint


def test_s3_proxy_reuse_requires_matching_patch_fingerprint(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [0]},
        "allowed_files": ["rosetta/model/projector.py"],
        "allowed_prefixes": ["recipe/", "local/auto_research_runs/"],
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_resume_fingerprint", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    adapter = C2CAdapter(paths.root, config)
    gpu_selection = agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1})
    candidate = {
        "id": "proxy_resume",
        "title": "Proxy Resume",
        "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
    }
    run_spec = adapter.materialize_candidate_configs(candidate, gpu_selection)
    state_path = Path(run_spec["run_state_path"])
    state_path.write_text(
        json.dumps(
            {
                "candidate_id": "proxy_resume",
                "run_id": "proxy_resume",
                "preflight": {"status": "ok"},
                "proxy_screen": {
                    "enabled": True,
                    "status": "rejected",
                    "metrics": {"mean": 49.0, "datasets": {"mmlu-redux": 49.0}},
                    "patch_fingerprint": "old_patch",
                },
                "metrics": None,
                "frozen_hashes": run_spec["frozen_hashes"],
                "patch_fingerprint": "old_patch",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert agent._load_reusable_c2c_proxy_state(run_spec, "new_patch") is None


def test_c2c_proxy_metric_near_threshold_is_repairable() -> None:
    decision = ExperimentAgent._c2c_proxy_metric_decision(
        metrics={"mean": 49.8, "datasets": {"mmlu-redux": 49.8}},
        baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        proxy_baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        patch_risk={"risk_labels": []},
        proxy_cfg={
            "require_paired_baseline": True,
            "min_proxy_mean_delta": -0.1,
            "repairable_proxy_mean_margin": 0.25,
            "max_proxy_dataset_regression": 1.5,
        },
    )

    assert decision["status"] == "repairable_proxy_risk"
    assert decision["repair_route"] == "S2_plan"
    assert decision["proxy_delta_vs_baseline"] == -0.2


def test_c2c_proxy_soft_zero_delta_is_repairable_by_default() -> None:
    decision = ExperimentAgent._c2c_proxy_metric_decision(
        metrics={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        proxy_baseline={"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        patch_risk={"risk_labels": []},
        proxy_cfg={
            "require_paired_baseline": True,
            "min_proxy_mean_delta": -0.3,
            "soft_proxy_mean_delta": 0.1,
            "repair_soft_proxy_fail": True,
            "max_proxy_dataset_regression": 1.5,
        },
    )

    assert decision["status"] == "repairable_proxy_risk"
    assert decision["soft_fail"] is True
    assert decision["repair_route"] == "S2_plan"
    assert decision["repair_mode"] == "effect_first_proxy_repair"
    assert "paperization" in decision["repair_hint"]
    assert "proxy mean delta" in decision["reason"]
    repair_contract = decision["proxy_effect_repair_contract"]
    assert repair_contract["mode"] == "effect_first_proxy_repair"
    assert repair_contract["proxy_delta_vs_baseline"] == 0.0
    assert "proxy mean delta" in repair_contract["soft_flags"][0]
    assert any("paperization" in item for item in repair_contract["forbidden"])


def test_c2c_proxy_command_failure_classifies_runtime_errors() -> None:
    dtype_failure = ExperimentAgent._c2c_proxy_command_failure(
        {
            "step": "proxy_command_0",
            "returncode": 1,
            "attempts": [
                {
                    "stdout": "",
                    "stderr": "RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16",
                }
            ],
        }
    )
    shape_failure = ExperimentAgent._c2c_proxy_command_failure(
        {
            "step": "proxy_command_0",
            "returncode": 1,
            "attempts": [{"stdout": "", "stderr": "TypeError: must be real number, not list"}],
        }
    )

    assert dtype_failure["category"] == "dtype_mismatch"
    assert "dtype/device" in dtype_failure["repair_hint"]
    assert shape_failure["category"] == "schema_shape_mismatch"


def test_c2c_proxy_command_failure_classifies_timeout() -> None:
    failure = ExperimentAgent._c2c_proxy_command_failure(
        {
            "step": "proxy_command_1",
            "returncode": 124,
            "attempts": [
                {
                    "stdout": "started",
                    "stderr": "Command timed out after 1200s",
                    "timed_out": True,
                    "elapsed_seconds": 1200.1,
                    "timeout_seconds": 1200,
                }
            ],
        }
    )

    assert failure["category"] == "proxy_timeout"
    assert "inference/training cost" in failure["repair_hint"]
    assert failure["timeout_seconds"] == 1200


def test_c2c_result_payload_compacts_patch_state_and_proxy_logs(tmp_path: Path, monkeypatch) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "train_samples": 1,
                "eval_limit": 1,
                "reject_on_command_failure": True,
            },
        },
    }
    config["code_patch"] = {
        "enabled": True,
        "dynamic_whitelist": {
            "include_prefixes": ["rosetta/", "script/", "recipe/", "test/", "tests/"],
            "include_extensions": [".py", ".json", ".yaml", ".yml", ".toml", ".txt"],
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_compact_payload", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    adapter = C2CAdapter(paths.root, config)

    patch_path = paths.root / "plan" / "code_patches" / "compact" / "patch.json"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    old_sha = sha256_file(adapter.repo_root / "rosetta/model/aligner.py")
    patch_path.write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "op": "replace_file",
                        "path": "rosetta/model/aligner.py",
                        "old_sha256": old_sha,
                        "new": "VALUE = 'compact payload patch'\n",
                    }
                ],
                "changed_files": ["rosetta/model/aligner.py"],
            }
        ),
        encoding="utf-8",
    )

    def fake_preflight(run_spec, gpu_selection):
        del gpu_selection
        payload = {"status": "ok", "checks": [], "errors": [], "warnings": [], "recovery_actions": []}
        Path(run_spec["preflight_path"]).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        del command, working_dir, retry_policy
        if name.startswith("preflight_command_") or name.startswith("proxy_baseline_"):
            if name == "proxy_baseline_train":
                final = repo / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
                final.mkdir(parents=True, exist_ok=True)
                (final / "marker.txt").write_text("ok", encoding="utf-8")
            if name == "proxy_baseline_eval_mmlu-redux":
                out = repo / "local" / "auto_research_runs" / "proxy_baseline" / "results" / "mmlu-redux"
                out.mkdir(parents=True, exist_ok=True)
                (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                    json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.5}),
                    encoding="utf-8",
                )
            return {"step": name, "status": "ok", "returncode": 0, "stdout": "ok", "stderr": "", "attempts": []}
        return {
            "step": name,
            "status": "failed",
            "returncode": 1,
            "stdout": "x" * 9000,
            "stderr": "RuntimeError: compact failure\n" + "y" * 9000,
            "attempts": [{"stdout": "nested" * 1000, "stderr": "nested-err" * 1000}],
        }

    monkeypatch.setattr(adapter, "preflight", fake_preflight)
    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=adapter,
        candidate={
            "id": "compact",
            "title": "Compact",
            "code_patch": {
                "status": "ok",
                "patch_json": "plan/code_patches/compact/patch.json",
                "changed_files": ["rosetta/model/aligner.py"],
                "has_executable_change": True,
            },
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    rendered = json.dumps(result, ensure_ascii=False)
    command_log = json.loads((paths.root / "experiment" / "logs" / "c2c_compact_commands.json").read_text(encoding="utf-8"))

    assert "restore_state" not in result["patch_result"]
    assert result["patch_result"]["restore_state_omitted"] is True
    assert len(rendered) < 30000
    assert "x" * 5000 not in rendered
    assert "y" * 5000 not in rendered
    assert "stdout_tail" in result["proxy_screen"]["attempts"][0]
    assert "stdout" not in result["proxy_screen"]["attempts"][0]
    assert "stdout_tail" in command_log["runs"][0]
    assert "stdout" not in command_log["runs"][0]


def test_c2c_replay_proxy_runs_paired_baseline_before_full_training(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path)
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["c2c"] = {
        "enabled": True,
        "snapshot_path": str(repo),
        "env_python": "/usr/bin/python3",
        "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "datasets": ["mmlu-redux"],
        "small_loop": {
            "eval_datasets": ["mmlu-redux"],
            "train_samples": 1,
            "gpu_ids": [0],
            "proxy_screen": {
                "enabled": True,
                "mode": "replay",
                "eval_datasets": ["mmlu-redux"],
                "eval_limit": 2,
                "train_samples": 2,
                "require_proxy_metrics": True,
                "require_paired_baseline": True,
                "run_baseline_if_missing": True,
                "min_proxy_mean_delta": -0.3,
                "max_proxy_dataset_regression": 1.5,
            },
        },
    }
    paths = init_workspace(config, "topic", project_id="proj_proxy_replay", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    seen_steps = []

    def fake_run_step(*, name, command, working_dir, retry_policy=None):
        seen_steps.append(name)
        if name == "proxy_baseline_train":
            final = repo / "local" / "auto_research_runs" / "proxy_baseline" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        if name == "proxy_baseline_eval_mmlu-redux":
            out = repo / "local" / "auto_research_runs" / "proxy_baseline" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.5}),
                encoding="utf-8",
            )
        if name == "proxy_command_1":
            out = repo / "local" / "auto_research_runs" / "idea_proxy_replay" / "proxy" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.505}),
                encoding="utf-8",
            )
        if name == "train":
            final = repo / "local" / "auto_research_runs" / "idea_proxy_replay" / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            (final / "marker.txt").write_text("ok", encoding="utf-8")
        if name == "eval_mmlu-redux":
            out = repo / "local" / "auto_research_runs" / "idea_proxy_replay" / "results" / "mmlu-redux"
            out.mkdir(parents=True, exist_ok=True)
            (out / "Rosetta_mmlu-redux_generate_summary.json").write_text(
                json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.51}),
                encoding="utf-8",
            )
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}

    monkeypatch.setattr(agent.runner, "run_step", fake_run_step)
    result = agent._run_single_c2c_candidate(
        adapter=C2CAdapter(paths.root, config),
        candidate={
            "id": "idea_proxy_replay",
            "title": "Idea Proxy Replay",
            "experiment_contract": {"config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}}},
        },
        index=0,
        simulate=False,
        baseline_mean=50.0,
        min_delta=0.1,
        max_regression=2.0,
        gpu_selection=agent.runner.select_gpus({"gpu_ids": [0], "max_gpus": 1}),
    )

    proxy = result["proxy_screen"]
    baseline_cache = paths.root / "experiment" / "results" / "c2c_proxy_baseline.json"
    assert baseline_cache.exists()
    assert seen_steps.index("proxy_baseline_train") < seen_steps.index("proxy_command_0")
    assert seen_steps.index("proxy_command_1") < seen_steps.index("train")
    assert proxy["status"] == "passed"
    assert proxy["proxy_delta_vs_baseline"] == 0.5
    assert proxy["proxy_decision_mode"] == "paired_baseline"
    assert result["command_status"] == "ok"


def test_c2c_failure_attribution_records_dataset_sample_and_patch_risk() -> None:
    candidate = {
        "id": "mixed_tradeoff",
        "decision": "not_viable",
        "metrics": {"mean": 50.0, "datasets": {"mmlu-redux": 43.0, "ai2-arc": 55.0, "openbookqa": 52.0}},
        "delta_vs_baseline": -0.82,
        "patch_result": {"changed_files": ["rosetta/model/projector.py", "script/evaluation/unified_evaluator.py"]},
        "config_overrides": {"train": {"model": {"soft_alignment_top_k": 2}}},
    }
    baseline = {"mean": 50.82, "datasets": {"mmlu-redux": 47.0, "ai2-arc": 54.0, "openbookqa": 50.0}}

    attribution = ExperimentAgent._c2c_failure_attribution(candidate, baseline)

    assert attribution["primary_failure"] == "mmlu-redux_regression"
    assert attribution["dragging_datasets"][0]["dataset"] == "mmlu-redux"
    assert attribution["sample_type_failures"][0]["sample_family"] == "multi_domain_knowledge_reasoning"
    assert "openbookqa_gain_mmlu_redux_regression" in attribution["mixed_gain_patterns"]
    assert "evaluation_code_changed" in attribution["patch_risk"]["risk_labels"]
    assert "projector_mechanism_changed" in attribution["patch_risk"]["risk_labels"]
    assert "train.model.soft_alignment_top_k" in attribution["patch_risk"]["config_override_keys"]


def test_c2c_failure_attribution_uses_proxy_dataset_deltas_without_full_metrics() -> None:
    candidate = {
        "id": "proxy_tradeoff",
        "decision": "proxy_repairable",
        "metrics": {},
        "proxy_screen": {
            "status": "repairable_proxy_risk",
            "metrics": {"mean": 32.6, "datasets": {"ai2-arc": 32.5, "mmlu-redux": 32.0, "openbookqa": 33.3}},
            "proxy_baseline": {"mean": 32.0, "datasets": {"ai2-arc": 33.8, "mmlu-redux": 30.7, "openbookqa": 31.5}},
            "proxy_dataset_deltas": {"ai2-arc": -1.3, "mmlu-redux": 1.3, "openbookqa": 1.8},
        },
    }

    attribution = ExperimentAgent._c2c_failure_attribution(candidate, {"mean": 50.0, "datasets": {}})

    assert attribution["primary_failure"] == "repairable_proxy_risk_before_full_training"
    assert attribution["dragging_datasets"][0]["dataset"] == "ai2-arc"
    assert attribution["dragging_datasets"][0]["source"] == "proxy_screen"
    assert attribution["sample_type_failures"][0]["sample_family"] == "science_reasoning_challenge"
    assert "cross_dataset_tradeoff" in attribution["mixed_gain_patterns"]


def test_c2c_ablation_no_effect_is_failure_attribution() -> None:
    candidate = {
        "id": "noop_mechanism",
        "decision": "not_viable",
        "metrics": {"mean": 55.0, "datasets": {"mmlu-redux": 55.0}},
        "delta_vs_baseline": 5.0,
        "ablation": {
            "enabled": True,
            "status": "ok",
            "switch": "disable_noop",
            "comparison": {
                "status": "ok",
                "enabled_minus_disabled_mean": 0.0,
                "dataset_enabled_minus_disabled": {"mmlu-redux": 0.0},
                "mechanism_supported": False,
            },
        },
    }
    baseline = {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}}

    attribution = ExperimentAgent._c2c_failure_attribution(candidate, baseline)
    posthoc = ExperimentAgent._c2c_deterministic_posthoc_review(
        {
            "acceptance": {"passed": False, "reason": "mechanism ablation support not met"},
            "baseline": baseline,
            "best_candidate": candidate,
            "candidate_results": [candidate],
        }
    )

    assert attribution["primary_failure"] == "ablation_no_effect"
    assert attribution["ablation_evidence"]["status"] == "no_effect"
    assert any("ablation switch did not change metrics" in item for item in posthoc["failure_modes"])


def test_c2c_acceptance_requires_ablation_support_when_configured() -> None:
    best = {
        "decision": "not_viable",
        "metrics": {"mean": 55.0, "datasets": {"mmlu-redux": 55.0}},
        "worst_dataset_regression": 0.0,
        "acceptance_rule": {"require_ablation_support": True},
        "mechanism_supported": False,
    }

    comparison = ExperimentAgent._c2c_acceptance_comparison(
        best,
        {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        min_delta=0.1,
        max_regression=2.0,
    )

    assert comparison["passed"] is False
    assert comparison["reason"] == "mechanism ablation support not met"


def test_c2c_acceptance_defaults_to_effect_first_without_ablation_support() -> None:
    best = {
        "decision": "not_viable",
        "metrics": {"mean": 55.0, "datasets": {"mmlu-redux": 55.0}},
        "worst_dataset_regression": 0.0,
        "acceptance_rule": {"require_ablation_support": False},
        "mechanism_supported": False,
    }

    comparison = ExperimentAgent._c2c_acceptance_comparison(
        best,
        {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        min_delta=0.1,
        max_regression=2.0,
    )

    assert comparison["passed"] is True
    assert comparison["reason"] == "accepted"
    assert comparison["require_ablation_support"] is False


def test_c2c_proxy_calibration_marks_false_positive_and_dataset_errors() -> None:
    payload = {
        "baseline": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0, "ai2-arc": 50.0, "openbookqa": 50.0}},
        "acceptance": {"passed": False, "reason": "mean delta or dataset regression threshold not met"},
        "candidate_results": [
            {
                "id": "utility_proxy_pass_full_fail",
                "title": "Utility proxy pass full fail",
                "mechanism_type": "utility_predicted_cache_routing",
                "decision": "not_viable",
                "metrics": {"mean": 49.8, "datasets": {"mmlu-redux": 48.5, "ai2-arc": 50.2, "openbookqa": 50.7}},
                "delta_vs_baseline": -0.2,
                "proxy_screen": {
                    "status": "passed",
                    "proxy_delta_vs_baseline": 0.8,
                    "proxy_score": 0.6,
                    "proxy_dataset_deltas": {"mmlu-redux": 0.9, "ai2-arc": 0.2, "openbookqa": 0.3},
                    "metrics": {"mean": 50.8, "datasets": {"mmlu-redux": 50.9, "ai2-arc": 50.2, "openbookqa": 50.3}},
                },
            }
        ],
    }

    iteration = experiment_module._c2c_proxy_calibration_iteration(payload, iteration=3)
    summary = experiment_module._c2c_proxy_calibration_summary([iteration])

    candidate = iteration["candidates"][0]
    assert candidate["proxy_false_positive"] is True
    assert candidate["mispredicted_datasets"] == ["mmlu-redux"]
    assert candidate["dataset_calibration"]["mmlu-redux"]["proxy_delta"] == 0.9
    assert candidate["dataset_calibration"]["mmlu-redux"]["full_delta"] == -1.5
    assert summary["proxy_false_positive_rate"] == 1.0
    assert summary["dataset_error_summary"]["mmlu-redux"]["misprediction_count"] == 1
    assert summary["mechanism_false_positive_summary"]["utility_predicted_cache_routing"]["false_positive_rate"] == 1.0


def test_c2c_paperization_readiness_after_effect_win() -> None:
    readiness = experiment_module._c2c_paperization_readiness(
        {
            "id": "winner",
            "proxy_screen": {
                "quality_repair": {
                    "issues": ["missing_coverage_diagnostics_evidence", "missing_matched_coverage_evidence"]
                }
            },
            "patch_result": {
                "mechanism_review": {
                    "soft_issues": ["ablation_switch_not_wired"]
                }
            },
        },
        {"passed": True},
    )

    assert readiness["status"] == "ready"
    assert readiness["next_stage"] == "paperization"
    assert readiness["candidate_id"] == "winner"
    assert any("coverage diagnostics" in item for item in readiness["tasks"])
    assert any("ablation switch" in item for item in readiness["tasks"])


def test_disable_llm_during_execution_skips_patch_llm(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    config["experiment"]["disable_llm_during_execution"] = True
    context = AgentContext(tmp_path / "workspace" / "p", config, ArtifactManager(tmp_path / "workspace" / "p"), ModelClient(config, project_root=tmp_path))
    agent = ExperimentAgent(context)

    class BombLLM:
        use_real_api = True

        def generate_json(self, **kwargs):
            raise AssertionError("training execution must not call LLM patch generation")

    context.llm = BombLLM()
    adapter = C2CAdapter(tmp_path, {"c2c": {"snapshot_path": str(_fake_c2c_repo(tmp_path)), "env_python": "/usr/bin/python3"}})
    patch = agent._generate_c2c_patch({"id": "x"}, adapter)
    assert patch["operations"] == []
    assert patch["status"] == "missing"
    assert "frozen S2.5 patch" in patch["summary"]


def test_c2c_posthoc_review_degrades_to_deterministic_feedback(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_posthoc", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    class FailingLLM:
        use_real_api = True

        def generate_json_with_schema(self, **kwargs):
            raise RuntimeError("429 Too Many Requests")

    context.llm = FailingLLM()
    payload = {
        "baseline": {"mean": 50.82, "datasets": {"mmlu-redux": 47.07, "ai2-arc": 54.78, "openbookqa": 50.6}},
        "acceptance": {
            "passed": False,
            "baseline_mean": 50.82,
            "best_mean": 50.0666,
            "delta": -0.7534,
            "min_delta_to_pass": 0.1,
            "max_dataset_regression": 2.0,
            "reason": "mean delta or dataset regression threshold not met",
        },
        "best_candidate": {
            "id": "mmlu_safe_low_confidence_gate",
            "title": "MMLU-safe low-confidence transfer gate",
            "decision": "not_viable",
            "metrics": {"mean": 50.0666, "datasets": {"mmlu-redux": 45.3303, "ai2-arc": 54.8696, "openbookqa": 50.0}},
            "dataset_regressions": {"mmlu-redux": 1.7397, "ai2-arc": 0.0, "openbookqa": 0.6},
        },
        "candidate_results": [
            {
                "id": "mmlu_safe_low_confidence_gate",
                "title": "MMLU-safe low-confidence transfer gate",
                "decision": "not_viable",
                "metrics": {"mean": 50.0666, "datasets": {"mmlu-redux": 45.3303}},
                "dataset_regressions": {"mmlu-redux": 1.7397},
            }
        ],
    }

    review = agent._c2c_posthoc_review(payload)

    assert review["status"] == "degraded"
    assert "GPT posthoc review unavailable" in review["reason"]
    assert review["failure_modes"]
    assert review["next_round_suggestions"]
    assert review["avoid_repeat_rules"]
    assert review["feedback_entries"][0]["idea_id"] == "mmlu_safe_low_confidence_gate"


def test_c2c_posthoc_review_uses_deterministic_feedback_without_llm(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_posthoc_no_llm", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    class NoLLM:
        use_real_api = False

    context.llm = NoLLM()
    payload = {
        "baseline": {"mean": 50.82, "datasets": {"mmlu-redux": 47.07}},
        "acceptance": {
            "passed": False,
            "baseline_mean": 50.82,
            "best_mean": 48.0,
            "delta": -2.82,
            "min_delta_to_pass": 0.1,
            "max_dataset_regression": 2.0,
            "reason": "mean delta or dataset regression threshold not met",
        },
        "best_candidate": {
            "id": "weak_gate",
            "title": "Weak gate",
            "decision": "not_viable",
            "metrics": {"mean": 48.0, "datasets": {"mmlu-redux": 43.0}},
            "dataset_regressions": {"mmlu-redux": 4.07},
        },
        "candidate_results": [
            {
                "id": "weak_gate",
                "title": "Weak gate",
                "decision": "not_viable",
                "metrics": {"mean": 48.0, "datasets": {"mmlu-redux": 43.0}},
                "dataset_regressions": {"mmlu-redux": 4.07},
            }
        ],
    }

    review = agent._c2c_posthoc_review(payload)

    assert review["status"] == "deterministic_no_llm"
    assert review["next_round_suggestions"]
    assert review["avoid_repeat_rules"]
    assert review["feedback_entries"][0]["idea_id"] == "weak_gate"


def test_c2c_posthoc_review_summarizes_proxy_repairable_failures(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_posthoc_proxy", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    class NoLLM:
        use_real_api = False

    context.llm = NoLLM()
    payload = {
        "baseline": {"mean": 50.0, "datasets": {}},
        "acceptance": {"passed": False, "reason": "no candidate metrics"},
        "best_candidate": None,
        "candidate_results": [
            {
                "id": "proxy_tradeoff",
                "title": "Proxy tradeoff",
                "decision": "proxy_repairable",
                "proxy_screen": {
                    "status": "repairable_proxy_risk",
                    "reason": "proxy worst dataset regression",
                    "proxy_dataset_deltas": {"ai2-arc": -1.3, "openbookqa": 1.8},
                },
                "failure_attribution": {
                    "primary_failure": "repairable_proxy_risk_before_full_training",
                    "dragging_datasets": [{"dataset": "ai2-arc", "regression": 1.3}],
                    "patch_risk": {"risk_labels": ["projector_mechanism_changed"]},
                },
            }
        ],
    }

    review = agent._c2c_posthoc_review(payload)

    assert review["status"] == "deterministic_proxy_feedback"
    assert "cheap proxy blocked all candidates" in review["failure_modes"][0]
    assert "proxy_dataset_deltas" in " ".join(review["next_round_suggestions"])
    assert review["feedback_entries"][0]["reason"] == "proxy worst dataset regression"
    assert review["feedback_entries"][0]["proxy_screen"]["proxy_dataset_deltas"]["ai2-arc"] == -1.3


def test_c2c_posthoc_review_skips_llm_for_proxy_only_failures(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_posthoc_proxy_skip", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    class BombLLM:
        use_real_api = True

        def generate_json_with_schema(self, **kwargs):
            raise AssertionError("proxy-only posthoc should not call the LLM")

    context.llm = BombLLM()
    payload = {
        "baseline": {"mean": 50.0, "datasets": {}},
        "acceptance": {
            "passed": False,
            "reason": "proxy mean delta -1.2 below hard threshold -0.3",
            "proxy_best_mean": 48.8,
            "proxy_delta": -1.2,
        },
        "best_candidate": None,
        "best_proxy_candidate": {
            "id": "proxy_tradeoff",
            "title": "Proxy tradeoff",
            "decision": "proxy_rejected",
            "proxy_screen": {
                "status": "rejected",
                "metrics": {"mean": 48.8, "datasets": {"ai2-arc": 47.0}},
                "proxy_delta_vs_baseline": -1.2,
                "proxy_dataset_deltas": {"ai2-arc": -1.3},
                "reason": "proxy mean delta -1.2 below hard threshold -0.3",
            },
        },
        "candidate_results": [
            {
                "id": "proxy_tradeoff",
                "title": "Proxy tradeoff",
                "decision": "proxy_rejected",
                "proxy_screen": {
                    "status": "rejected",
                    "metrics": {"mean": 48.8, "datasets": {"ai2-arc": 47.0}},
                    "proxy_delta_vs_baseline": -1.2,
                    "proxy_dataset_deltas": {"ai2-arc": -1.3},
                    "reason": "proxy mean delta -1.2 below hard threshold -0.3",
                },
                "failure_attribution": {
                    "primary_failure": "cheap_proxy_rejected_before_full_training",
                    "dragging_datasets": [{"dataset": "ai2-arc", "regression": 1.3}],
                },
            }
        ],
    }

    review = agent._c2c_posthoc_review(payload)

    assert review["status"] == "deterministic_proxy_feedback"
    assert "skipped GPT posthoc" in review["reason"]
    assert review["feedback_entries"][0]["idea_id"] == "proxy_tradeoff"


def test_c2c_iteration_history_appends_best_and_consecutive_counts(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_history", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    registry_path = paths.root / "meta" / "registry.yaml"

    first = {
        "baseline": {"mean": 50.0},
        "acceptance": {"passed": False, "best_mean": 48.5, "delta": -1.5, "reason": "below"},
        "best_candidate": {
            "id": "idea_a",
            "title": "Idea A",
            "decision": "not_viable",
            "metrics": {"mean": 48.5, "datasets": {}},
            "delta_vs_baseline": -1.5,
        },
        "candidate_results": [{"id": "idea_a"}],
    }
    history = agent._append_c2c_iteration_history(first)
    assert history["iteration_count"] == 1
    assert history["best_candidate_id"] == "idea_a"
    assert history["consecutive_not_viable"] == 1

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["iteration"] = 2
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    second = {
        "baseline": {"mean": 50.0},
        "acceptance": {"passed": False, "best_mean": 49.2, "delta": -0.8, "reason": "below"},
        "best_candidate": {
            "id": "idea_b",
            "title": "Idea B",
            "decision": "not_viable",
            "metrics": {"mean": 49.2, "datasets": {}},
            "delta_vs_baseline": -0.8,
        },
        "candidate_results": [{"id": "idea_b"}],
    }
    history = agent._append_c2c_iteration_history(second)

    assert history["iteration_count"] == 2
    assert history["best_candidate_id"] == "idea_b"
    assert history["best_delta_so_far"] == -0.8
    assert history["consecutive_not_viable"] == 2
    saved = json.loads((paths.root / "experiment/results/c2c_iteration_history.json").read_text(encoding="utf-8"))
    assert [item["iteration"] for item in saved["iterations"]] == [1, 2]


def test_c2c_iteration_history_records_proxy_rejected_metrics(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "workspace", simulate=False)
    paths = init_workspace(config, "topic", project_id="proj_history_proxy", simulate=False)
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)
    candidate = {
        "id": "proxy_tradeoff",
        "title": "Proxy tradeoff",
        "decision": "proxy_rejected",
        "proxy_screen": {
            "status": "rejected",
            "metrics": {"mean": 48.8, "datasets": {"ai2-arc": 47.0}},
            "proxy_delta_vs_baseline": -1.2,
            "proxy_score": -1.9,
            "proxy_worst_dataset_regression": 1.3,
            "proxy_dataset_deltas": {"ai2-arc": -1.3},
        },
    }
    payload = {
        "baseline": {"mean": 50.0},
        "acceptance": {
            "passed": False,
            "baseline_mean": 50.0,
            "best_mean": None,
            "delta": None,
            "proxy_best_mean": 48.8,
            "proxy_delta": -1.2,
            "proxy_score": -1.9,
            "reason": "cheap proxy blocked candidate before full S3",
        },
        "best_candidate": None,
        "best_proxy_candidate": candidate,
        "candidate_results": [candidate],
    }

    history = agent._append_c2c_iteration_history(payload)

    assert history["best_candidate_id"] is None
    assert history["best_proxy_candidate_id"] == "proxy_tradeoff"
    assert history["best_proxy_mean_so_far"] == 48.8
    assert history["best_proxy_delta_so_far"] == -1.2
    saved = json.loads((paths.root / "experiment/results/c2c_iteration_history.json").read_text(encoding="utf-8"))
    entry = saved["iterations"][0]
    assert entry["acceptance"]["proxy_best_mean"] == 48.8
    assert entry["best_proxy_candidate"]["proxy_dataset_deltas"]["ai2-arc"] == -1.3
