from pathlib import Path
from types import SimpleNamespace

import auto_research.cli as cli_module
from auto_research.cli import _smoke_c2c_command
from auto_research.cli import _apply_c2c_run_overrides
from auto_research.config import load_project_config
from auto_research.utils import read_yaml, write_json, write_yaml


def test_smoke_c2c_bootstraps_missing_project_with_user_overrides(monkeypatch, tmp_path: Path) -> None:
    project = tmp_path / "workspace" / "smoke_bootstrap"
    seen: list[dict] = []

    class FakeOrchestrator:
        def _project_root(self, project_id: str) -> Path:
            return project

        def doctor_c2c(self, project_id: str) -> dict:
            write_yaml(project / "meta" / "registry.yaml", {"project_id": project.name, "current_stage": "S0_intake", "iteration": 1, "status": "initialized", "stages": {}})
            write_json(project / "meta" / "c2c_e2e_readiness_report.json", {"gate": "pass", "blocking_reasons": [], "warnings": []})
            write_json(project / "meta" / "c2c_execution_hooks_report.json", {"gate": "pass", "blocking_reasons": []})
            return {"status": "pass", "readiness_report": {"gate": "pass"}, "execution_hooks_report": {"gate": "pass"}}

    def fake_load_project_config(project_root: Path) -> dict:
        return {"c2c": {"enabled": True}, "experiment": {"simulate": False}}

    def fake_run_c2c(args, orchestrator):
        seen.append(vars(args).copy())
        project.mkdir(parents=True, exist_ok=True)
        write_yaml(project / "meta" / "project_config.yaml", {"c2c": {"enabled": True}, "experiment": {"simulate": False}})
        write_json(project / "meta" / "c2c_e2e_run_manifest.json", {"final_status": "completed", "stage_boundaries": {"S3_experiment": {"status": "completed"}}})
        return {"status": "prepared" if args.prepare_only else "completed", "project_id": args.project_id}

    monkeypatch.setattr(cli_module, "_run_c2c_command", fake_run_c2c)
    monkeypatch.setattr(cli_module, "load_project_config", fake_load_project_config)

    args = SimpleNamespace(
        project_id="smoke_bootstrap",
        topic="custom topic",
        target_repo="/tmp/C2C",
        ref_paper="/tmp/paper.pdf",
        ref_rebuttal="/tmp/rebuttal.pdf",
        env_python="/tmp/python",
        from_stage="S3_experiment",
        audit_scope="completed",
        no_s0_cache=True,
        s0_cache_project=None,
        s0_cache_path=None,
        s0_force_refresh=True,
        prepare_only=True,
    )
    result = _smoke_c2c_command(args, FakeOrchestrator())

    assert result["status"] == "prepared"
    assert [step["name"] for step in result["steps"]] == ["prepare-c2c", "doctor-c2c"]
    assert seen[0]["prepare_only"] is True
    assert seen[0]["topic"] == "custom topic"
    assert seen[0]["target_repo"] == "/tmp/C2C"
    assert seen[0]["ref_paper"] == "/tmp/paper.pdf"
    assert seen[0]["ref_rebuttal"] == "/tmp/rebuttal.pdf"
    assert seen[0]["env_python"] == "/tmp/python"
    assert seen[0]["no_s0_cache"] is True
    assert seen[0]["s0_force_refresh"] is True


def test_smoke_c2c_help_exposes_override_flags() -> None:
    parser = cli_module.build_parser()
    args = parser.parse_args(
        [
            "smoke-c2c",
            "--project-id",
            "p",
            "--topic",
            "t",
            "--target-repo",
            "/repo",
            "--ref-paper",
            "/paper",
            "--ref-rebuttal",
            "/rebuttal",
            "--env-python",
            "/python",
            "--no-s0-cache",
            "--s0-force-refresh",
            "--prepare-only",
        ]
    )

    assert args.command == "smoke-c2c"
    assert args.topic == "t"
    assert args.target_repo == "/repo"
    assert args.ref_paper == "/paper"
    assert args.ref_rebuttal == "/rebuttal"
    assert args.env_python == "/python"
    assert args.no_s0_cache is True
    assert args.s0_force_refresh is True
    assert args.prepare_only is True


def test_bootstrap_profile_forces_single_iteration_and_s3_proxy_overlay(monkeypatch, tmp_path: Path) -> None:
    project = tmp_path / "workspace" / "bootstrap_profile"
    write_yaml(project / "meta" / "project_config.yaml", {"orchestration": {"profile": "standard"}})
    write_yaml(project / "meta" / "registry.yaml", {"max_iterations": 9})
    monkeypatch.setattr("auto_research.config.load_root_config", lambda: {"project": {"workspace_root": str(tmp_path / "workspace")}})

    result = _apply_c2c_run_overrides(
        project,
        max_iterations=9,
        stop_after_stage="S4_writing",
        auto_mode=True,
        profile="bootstrap",
    )

    persisted = read_yaml(project / "meta" / "project_config.yaml", default={})
    effective = load_project_config(project)
    assert result["max_iterations"] == 1
    assert result["stop_after_stage"] == "S3_experiment"
    assert persisted["orchestration"]["profile"] == "bootstrap"
    assert "code_patch" not in persisted
    assert effective["review"]["max_iterations"] == 1
    assert effective["orchestration"]["stop_after_stage"] == "S3_experiment"
    assert effective["orchestration"]["bootstrap"]["cached_s0_only"] is True
    assert effective["code_patch"]["validation"]["require_py_compile"] is True
    assert effective["code_patch"]["validation"]["require_targeted_tests"] is True
    assert effective["code_patch"]["validation"]["require_config_activation"] is False
    assert effective["code_patch"]["validation"]["runtime_smoke"]["enabled"] is False
    assert effective["c2c"]["small_loop"]["max_candidates"] == 1
    assert effective["c2c"]["small_loop"]["proxy_screen"]["activation_smoke"]["enabled"] is False


def test_smoke_profile_defaults_to_standard() -> None:
    args = cli_module.build_parser().parse_args(["smoke-c2c", "--project-id", "p"])
    assert args.profile == "standard"
