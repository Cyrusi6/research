from pathlib import Path

from auto_research.research_state import ResearchEventLedger
from auto_research.utils import write_json
from auto_research.validators import run_stage_gate
from test_authoritative_state_machine import _complete, _direction, _initialize, _reserve, _variant


def _write_s3_outputs(root: Path, *, bootstrap: bool = False) -> tuple[dict, dict, dict]:
    direction = _direction()
    variant = _variant(direction, 1)
    write_json(root / "literature" / "direction.json", direction)
    write_json(root / "plan" / "variant.json", variant)
    ledger = ResearchEventLedger(root)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant, profile="bootstrap" if bootstrap else "standard")
    _complete(ledger, attempt, outcome="accepted")
    trial = ledger.state()["trial_results"][attempt["attempt_id"]]
    write_json(root / "experiment" / "results" / "trial_result.json", trial)
    write_json(
        root / "experiment" / "results" / "main_results.json",
        {"candidate_results": [{"id": variant["variant_id"], "metrics": {"mean": 51.0}}], "acceptance": {"passed": True}},
    )
    write_json(root / "experiment" / "results" / "ablation_results.json", {"status": "ok"})
    (root / "experiment" / "results" / "hypothesis_verification.md").write_text("ok\n", encoding="utf-8")
    if bootstrap:
        write_json(
            root / "experiment" / "results" / "bootstrap_proxy_completion.json",
            {"schema_version": "bootstrap_proxy_completion_v1", "bootstrap_proxy_complete": True},
        )
    return direction, variant, attempt


def test_s3_gate_passes_strict_trial_result(tmp_path: Path) -> None:
    _write_s3_outputs(tmp_path)
    report = run_stage_gate("S3_experiment", tmp_path, {}).to_dict()
    assert report["status"] == "PASS"
    assert next(check for check in report["checks"] if check["name"] == "trial_result_v1")["status"] == "PASS"


def test_s3_gate_retries_without_trial_result(tmp_path: Path) -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    write_json(tmp_path / "literature" / "direction.json", direction)
    write_json(tmp_path / "plan" / "variant.json", variant)
    report = run_stage_gate("S3_experiment", tmp_path, {}).to_dict()
    assert report["status"] == "NEEDS_RETRY"
    assert any(check.get("artifact") == "experiment/results/trial_result.json" for check in report["checks"])


def test_s3_bootstrap_gate_passes_once_without_budget_consumption(tmp_path: Path) -> None:
    direction, _, attempt = _write_s3_outputs(tmp_path, bootstrap=True)
    config = {"orchestration": {"profile": "bootstrap"}}
    report = run_stage_gate("S3_experiment", tmp_path, config).to_dict()
    state = ResearchEventLedger(tmp_path).state()
    assert report["status"] == "PASS"
    assert state["attempts"][attempt["attempt_id"]]["consumes_direction_budget"] is False
    assert state["directions"][direction["direction_hash"]]["budget"]["consumed"] == 0
