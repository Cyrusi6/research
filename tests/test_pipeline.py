import json
from pathlib import Path

import auto_research.config as config_module
import auto_research.orchestrator as orchestrator_module
from auto_research.adapters.literature import LiteratureProvider
from auto_research.orchestrator import Orchestrator


def _test_config(tmp_path: Path, simulate: bool) -> dict:
    return {
        "project": {"workspace_root": str(tmp_path), "target_venue": "TestConf", "language": "en"},
        "llm": {"use_real_api": False, "model": "mock"},
        "literature": {"download_pdfs": True, "request_timeout_seconds": 1, "max_papers": 2, "arxiv_max_results": 1},
        "plan": {"min_hypotheses": 1, "min_baselines": 2, "min_datasets": 1},
        "experiment": {"simulate": simulate, "random_seeds": [42, 123, 456]},
        "writing": {"claim_verification": {"enabled": True, "min_pass_rate": 0.8}, "require_compile": False},
        "review": {"pass_threshold": 7.0, "max_iterations": 2},
        "orchestration": {"judge_max_retries": 1},
    }


def test_simulated_pipeline_runs_to_completion(monkeypatch, tmp_path: Path) -> None:
    config = _test_config(tmp_path, simulate=True)
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(LiteratureProvider, "search", lambda self, topic: [])
    monkeypatch.setattr(LiteratureProvider, "download_pdf", lambda self, url: None)

    orchestrator = Orchestrator()
    project_id = orchestrator.init_project("retrieval benchmark", project_id="proj_pipeline", simulate=True)
    result = orchestrator.start(project_id)

    assert result["status"] == "completed"
    review_dispatch = tmp_path / project_id / "review" / "revision_dispatch.yaml"
    assert review_dispatch.exists()
    registry = (tmp_path / project_id / "meta" / "registry.yaml").read_text(encoding="utf-8")
    assert "current_stage: DONE" in registry
    references_manifest = json.loads((tmp_path / project_id / "references" / "papers" / "manifest.json").read_text(encoding="utf-8"))
    assert references_manifest["papers"]


def test_real_mode_blocks_at_experiment_stage(monkeypatch, tmp_path: Path) -> None:
    config = _test_config(tmp_path, simulate=False)
    monkeypatch.setattr(config_module, "load_root_config", lambda: config)
    monkeypatch.setattr(orchestrator_module, "load_root_config", lambda: config)
    monkeypatch.setattr(LiteratureProvider, "search", lambda self, topic: [])
    monkeypatch.setattr(LiteratureProvider, "download_pdf", lambda self, url: None)

    orchestrator = Orchestrator()
    project_id = orchestrator.init_project("real run topic", project_id="proj_blocked", simulate=False)
    result = orchestrator.start(project_id)

    assert result["status"] == "blocked"
    assert result["stage"] == "S3_experiment"
