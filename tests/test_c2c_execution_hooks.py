import json
import sys
from pathlib import Path

from auto_research.c2c_e2e import build_c2c_e2e_readiness_report, build_c2c_execution_hooks_report, write_c2c_execution_hooks_report
from auto_research.utils import read_json


def _repo(root: Path) -> Path:
    repo = root / "C2C"
    (repo / "rosetta").mkdir(parents=True)
    (repo / "rosetta" / "__init__.py").write_text("", encoding="utf-8")
    eval_script = repo / "script" / "evaluation" / "unified_evaluator.py"
    eval_script.parent.mkdir(parents=True)
    eval_script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--config')\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    return repo


def _config(tmp_path: Path, repo: Path, dataset_root: Path) -> dict:
    return {
        "project": {"workspace_root": str(tmp_path / "workspace")},
        "llm": {"provider": "openai", "use_real_api": False},
        "experiment": {"simulate": False},
        "c2c": {
            "enabled": True,
            "target_repo": str(repo),
            "snapshot_path": str(repo),
            "ref_paper": str(tmp_path / "paper.md"),
            "ref_rebuttal": str(tmp_path / "rebuttal.md"),
            "env_python": sys.executable,
            "dataset_root": str(dataset_root),
            "small_loop": {"proxy_screen": {"command_timeout_seconds": 30, "gpu_policy": {"gpu_ids": "auto"}}},
        },
        "orchestration": {"c2c_e2e": {"execution_hook_timeout_seconds": 10}},
    }


def test_execution_hooks_report_passes_cheap_real_probes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    (dataset_root / "sample.jsonl").write_text(json.dumps({"question": "q", "answer": "a"}) + "\n", encoding="utf-8")
    (tmp_path / "paper.md").write_text("paper", encoding="utf-8")
    (tmp_path / "rebuttal.md").write_text("rebuttal", encoding="utf-8")
    project = tmp_path / "workspace" / "proj"
    project.mkdir(parents=True)

    report = write_c2c_execution_hooks_report(project, _config(tmp_path, repo, dataset_root))

    assert report["gate"] == "pass"
    assert report["checks"]["env_python_runs"] is True
    assert report["checks"]["target_repo_importable"] is True
    assert report["checks"]["eval_help_command_passed"] is True
    assert report["checks"]["dataset_one_example_loadable"] is True
    assert read_json(project / "meta" / "c2c_execution_hooks_report.json")["schema_version"] == "c2c_execution_hooks_report_v1"


def test_readiness_fails_when_execution_hooks_gate_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    (tmp_path / "paper.md").write_text("paper", encoding="utf-8")
    (tmp_path / "rebuttal.md").write_text("rebuttal", encoding="utf-8")
    project = tmp_path / "workspace" / "proj"
    project.mkdir(parents=True)
    config = _config(tmp_path, repo, dataset_root)

    hooks = build_c2c_execution_hooks_report(project, config)
    assert hooks["gate"] == "fail"
    assert "dataset_one_example_loadable" in hooks["blocking_reasons"]

    from auto_research.utils import write_json

    write_json(project / "meta" / "c2c_execution_hooks_report.json", hooks)
    readiness = build_c2c_e2e_readiness_report(project, config)

    assert readiness["gate"] == "fail"
    assert readiness["checks"]["real_execution_hooks_ready"] is False
    assert "real_execution_hooks_ready" in readiness["blocking_reasons"]
