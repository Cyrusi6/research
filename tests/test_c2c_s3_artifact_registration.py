import json
from pathlib import Path

from auto_research.agents.base import AgentContext
from auto_research.agents.experiment import ExperimentAgent
from auto_research.artifacts import ArtifactManager
from auto_research.llm import ModelClient
from auto_research.utils import write_json
from auto_research.workspace import init_workspace


def _config(workspace: Path) -> dict:
    return {
        "project": {"workspace_root": str(workspace), "target_venue": "TestConf", "language": "en"},
        "llm": {"provider": "openai", "use_real_api": False, "model": "mock"},
        "literature": {"download_pdfs": False},
        "experiment": {"simulate": False, "random_seeds": [42]},
        "writing": {"require_compile": False},
        "review": {"pass_threshold": 7.0, "max_iterations": 1},
        "orchestration": {"judge_max_retries": 1, "auto_mode": True},
        "c2c": {
            "enabled": True,
            "snapshot_path": "external/c2c_snapshot",
            "env_python": "/usr/bin/python3",
            "model_map": {},
            "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
            "datasets": ["mmlu-redux"],
            "small_loop": {
                "proxy_screen": {
                    "enabled": True,
                    "mode": "replay",
                    "eval_datasets": ["mmlu-redux"],
                    "eval_limit": 2,
                    "train_samples": 2,
                    "activation_smoke": {"enabled": False},
                }
            },
        },
    }


def test_c2c_proxy_contract_writes_are_registered_in_experiment_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path / "workspace")
    paths = init_workspace(config, "topic", project_id="proj_proxy_manifest", simulate=False)
    write_json(paths.root / "plan" / "s2_planner" / "variant_scorecard.json", {"ranking": []})
    write_json(paths.root / "plan" / "s2_planner" / "next_variant.json", {"variant_id": "v1"})
    write_json(paths.root / "plan" / "code_patches" / "patch_gate_report.json", {"gate": "pass"})
    context = AgentContext(paths.root, config, ArtifactManager(paths.root), ModelClient(config, project_root=paths.root))
    agent = ExperimentAgent(context)

    agent._write_c2c_proxy_policy_contracts(plan={"id": "plan"}, execution={"id": "exec"}, include_baseline=True)

    manifest = json.loads((paths.root / "experiment" / "stage_manifest.json").read_text(encoding="utf-8"))
    paths_in_manifest = {item["path"] for item in manifest["artifacts"]}
    assert "experiment/results/c2c_proxy_calibration_policy.json" in paths_in_manifest
    assert "experiment/results/c2c_effective_proxy_policy.json" in paths_in_manifest
    assert "experiment/results/c2c_proxy_baseline_fingerprint.json" in paths_in_manifest
    assert "experiment/results/c2c_proxy_cache_report.json" in paths_in_manifest
