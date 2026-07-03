import json
from pathlib import Path

from auto_research.c2c_e2e import build_c2c_e2e_readiness_report
from auto_research.utils import sha256_file, write_json


def _config(tmp_path: Path, *, simulate: bool = False, target_repo: Path | None = None, env_python: Path | None = None) -> dict:
    repo = target_repo or tmp_path / "C2C"
    paper = tmp_path / "paper.md"
    rebuttal = tmp_path / "rebuttal.md"
    dataset_root = tmp_path / "datasets"
    paper.write_text("paper", encoding="utf-8")
    rebuttal.write_text("rebuttal", encoding="utf-8")
    dataset_root.mkdir(exist_ok=True)
    return {
        "project": {"workspace_root": str(tmp_path)},
        "llm": {"provider": "openai", "reasoning_provider": "openai", "use_real_api": False},
        "experiment": {"simulate": simulate},
        "c2c": {
            "enabled": True,
            "target_repo": str(repo),
            "snapshot_path": "external/c2c_snapshot",
            "ref_paper": str(paper),
            "ref_rebuttal": str(rebuttal),
            "env_python": str(env_python or Path("/usr/bin/python3")),
            "dataset_root": str(dataset_root),
            "small_loop": {"strict_dataset_cache": True, "proxy_screen": {"gpu_policy": {"gpu_ids": "auto"}}},
        },
    }


def test_c2c_readiness_fails_when_target_repo_missing(tmp_path: Path) -> None:
    project = tmp_path / "workspace" / "proj"
    project.mkdir(parents=True)

    report = build_c2c_e2e_readiness_report(project, _config(tmp_path, target_repo=tmp_path / "missing_repo"))

    assert report["gate"] == "fail"
    assert report["checks"]["target_repo_exists"] is False
    assert "target_repo_exists" in report["blocking_reasons"]


def test_c2c_readiness_fails_when_env_python_missing(tmp_path: Path) -> None:
    repo = tmp_path / "C2C"
    repo.mkdir()
    project = tmp_path / "workspace" / "proj"
    project.mkdir(parents=True)

    report = build_c2c_e2e_readiness_report(project, _config(tmp_path, target_repo=repo, env_python=tmp_path / "missing_python"))

    assert report["gate"] == "fail"
    assert report["checks"]["env_python_executable"] is False
    assert "env_python_executable" in report["blocking_reasons"]


def test_c2c_readiness_simulate_mode_does_not_require_real_dataset_gpu_or_llm(tmp_path: Path) -> None:
    project = tmp_path / "workspace" / "proj"
    project.mkdir(parents=True)
    config = _config(tmp_path, simulate=True, target_repo=tmp_path / "missing_repo", env_python=tmp_path / "missing_python")
    config["llm"] = {"provider": "openai", "reasoning_provider": "openai", "use_real_api": True}
    config["c2c"]["dataset_root"] = str(tmp_path / "missing_datasets")

    report = build_c2c_e2e_readiness_report(project, config)

    assert report["gate"] == "pass"
    assert report["checks"]["llm_config_ready"] is True
    assert report["checks"]["dataset_paths_ready"] is True
    assert report["checks"]["gpu_policy_ready"] is True


def test_c2c_readiness_real_openai_provider_requires_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    repo = tmp_path / "C2C"
    repo.mkdir()
    project = tmp_path / "workspace" / "proj"
    project.mkdir(parents=True)
    config = _config(tmp_path, target_repo=repo)
    config["llm"] = {"provider": "openai", "reasoning_provider": "openai", "use_real_api": True}

    report = build_c2c_e2e_readiness_report(project, config)

    assert report["gate"] == "fail"
    assert report["checks"]["llm_config_ready"] is False
    assert "llm_config_ready" in report["blocking_reasons"]


def test_c2c_readiness_warns_on_s0_cache_fingerprint_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "C2C"
    (repo / "rosetta" / "model").mkdir(parents=True)
    (repo / "rosetta" / "model" / "aligner.py").write_text("new", encoding="utf-8")
    project = tmp_path / "workspace" / "proj"
    snapshot = project / "external" / "c2c_snapshot" / "rosetta" / "model"
    snapshot.mkdir(parents=True)
    snapshot_file = snapshot / "aligner.py"
    snapshot_file.write_text("current", encoding="utf-8")
    write_json(
        project / "intake" / "c2c" / "static_bundle.json",
        {
            "schema_version": "c2c_static_intake_bundle_v1",
            "repo_manifest": {"core_files": [{"path": "rosetta/model/aligner.py", "sha256": sha256_file(repo / "rosetta" / "model" / "aligner.py")}]},
        },
    )
    config = _config(tmp_path, target_repo=repo)

    report = build_c2c_e2e_readiness_report(project, config)

    assert report["gate"] == "warn"
    assert report["checks"]["s0_cache_compatible"] is False
    assert "s0_cache_fingerprint_mismatch_or_unchecked" in report["warnings"]
