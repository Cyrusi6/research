from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from auto_research.agents.experiment import (
    _stage_evidence_inventory,
)
from auto_research.agents.plan import _trial_spec_from_plan
from auto_research.contract_store import ContractStore
from auto_research.domain_contracts import canonical_hash, trial_spec_hash
from auto_research.evidence import EVIDENCE_SCHEMA_VERSIONS, encode_canonical_evidence
from auto_research.s3_validation import S3ValidationError, _validate_trial_spec_projection_drift
from auto_research.utils import write_json


def _variant() -> dict:
    return {
        "direction_id": "direction-1",
        "direction_semantic_hash": "d" * 64,
        "direction_spec_hash": "e" * 64,
        "variant_id": "variant-1",
        "variant_semantic_hash": "a" * 64,
        "variant_spec_hash": "f" * 64,
        "ablation": {"remove_core": True},
        "expected_metric_signature": {"primary_metric": "accuracy"},
    }


def _plan() -> dict:
    return {
        "hypotheses": [{"id": "H1", "statement": "candidate improves accuracy"}],
        "baselines": [{"name": "baseline"}],
        "datasets": [{"name": "dataset-a", "split": "validation", "sample_count": 2}],
        "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [7], "aggregation": "mean", "require_complete_seed_coverage": True},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 0.05},
        "ablation_matrix": [],
        "execution": {"mode": "simulate", "collector": "generic", "commands": []},
        "resource_budget": {"wall_clock_minutes": 20},
    }


def _real_plan(project_root: Path) -> dict:
    sample_bytes = b'{"id":"sample-a","value":1}\n'
    sample_path = project_root / "samples" / "dataset-a.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_bytes(sample_bytes)
    evaluator_path = project_root / "evaluator.py"
    evaluator_path.write_text("def evaluate(value):\n    return float(value)\n", encoding="utf-8")
    plan = _plan()
    plan["datasets"] = [{
        "name": "dataset-a",
        "split": "validation",
        "sample_count": 1,
        "source_revision": "fixture-source-v1",
        "ordered_sample_ids": [hashlib.sha256(sample_bytes).hexdigest()],
    }]
    plan["execution"] = {
        "mode": "real",
        "collector": "external_manifest",
        "commands": [{"argv": ["true"]}],
        "workdir": str(project_root),
        "phase_manifest_path": "runner/phase_manifest.json",
        "evaluator_id": "fixture-evaluator",
        "evaluator_source_paths": ["evaluator.py"],
        "dependency_payloads": [{"name": "python", "lock": "fixture-lock-v1"}],
    }
    return plan


def _attempt(trial_spec: dict) -> dict:
    return {
        "attempt_id": "attempt-12345678",
        "direction_semantic_hash": "d" * 64,
        "direction_spec_hash": "e" * 64,
        "variant_semantic_hash": "a" * 64,
        "variant_spec_hash": "f" * 64,
        "trial_spec_hash": trial_spec_hash(trial_spec),
        "protocol_hash": canonical_hash(trial_spec["protocol"]),
        "sample_manifest_hash": canonical_hash(trial_spec["sample_manifest"]),
        "evaluator_hash": trial_spec["execution_contract"]["evaluator_hash"],
        "attempt_kind": "proxy_full",
        "seeds": [7],
        "lifecycle_generation": 0,
        "implementation_hash": "b" * 64,
        "attempt_input_hash": "c" * 64,
        "phase_executions": {
            "proxy": None,
            "full": {
                "phase_execution_id": "phase-full-0001",
                "phase_start_event_id": "phase-start-full",
                "producer_run_id": "producer-run-1",
                "command_plan_hash": next(
                    item["command_plan_hash"] for item in trial_spec["phase_contracts"] if item["phase"] == "full"
                ),
            },
        },
    }


def _measurement(attempt: dict, *, rows: list[dict] | None = None) -> dict:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSIONS["main_results"],
        "evidence_kind": "main_results",
        "evidence_id": "evidence:main:producer-run-1",
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": "producer-run-1",
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "cross_references": {},
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": "full",
        "phase_execution_id": attempt["phase_executions"]["full"]["phase_execution_id"],
        "phase_start_event_id": attempt["phase_executions"]["full"]["phase_start_event_id"],
        "rows": rows or [],
    }


def _activation(attempt: dict) -> dict:
    execution = attempt["phase_executions"]["full"]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSIONS["activation_evidence"],
        "evidence_kind": "activation_evidence",
        "evidence_id": "evidence:activation:producer-run-1",
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": "producer-run-1",
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "cross_references": {},
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": "full",
        "phase_execution_id": execution["phase_execution_id"],
        "phase_start_event_id": execution["phase_start_event_id"],
        "probe_id": "fixture-forward-probe",
        "status": "passed",
        "command_status": "completed",
        "exit_code": 0,
        "implementation_surface_ids": ["src/router.py"],
    }


def _write_inventory(tmp_path: Path, attempt: dict, main_payload: dict) -> list[dict]:
    runner = tmp_path / "runner"
    runner.mkdir(parents=True, exist_ok=True)
    (runner / "main.json").write_bytes(encode_canonical_evidence(main_payload))
    (runner / "activation.json").write_bytes(encode_canonical_evidence(_activation(attempt)))
    return [
        {"kind": "activation_evidence", "source_path": "runner/activation.json", "producer_run_id": "producer-run-1"},
        {"kind": "main_results", "source_path": "runner/main.json", "producer_run_id": "producer-run-1"},
    ]


def _row(attempt: dict, *, role: str, value: float) -> dict:
    return {
        "phase": "full",
        "role": role,
        "dataset_id": "dataset-a",
        "metric_id": "accuracy",
        "seed": 7,
        "metric_value": value,
        "command_status": "completed",
        "attempt_id": attempt["attempt_id"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "producer_run_id": "producer-run-1",
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase_execution_id": attempt["phase_executions"]["full"]["phase_execution_id"],
        "phase_start_event_id": attempt["phase_executions"]["full"]["phase_start_event_id"],
    }


def test_identity_only_main_evidence_cannot_create_observations(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_plan(), _variant(), project_root=tmp_path)
    attempt = _attempt(trial_spec)
    inventory = _write_inventory(tmp_path, attempt, _measurement(attempt))

    with pytest.raises(S3ValidationError, match=r"rows: \[\] is too short"):
        _stage_evidence_inventory(
            project_root=tmp_path,
            attempt=attempt,
            trial_spec=trial_spec,
            inventory=inventory,
        )
    assert not list((tmp_path / "experiment" / "attempts").glob("**/main_results/*.json"))


def test_real_generic_trial_spec_uses_readable_content_addressed_contracts(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_real_plan(tmp_path), _variant(), project_root=tmp_path)
    store = ContractStore(tmp_path)

    sample_manifest = store.read_contract(
        trial_spec["sample_manifest_ref"],
        contract_kind="sample_manifest",
        schema_file="sample_manifest_v4.schema.json",
    )
    evaluator_manifest = store.read_contract(
        trial_spec["execution_contract"]["evaluator_manifest_ref"],
        contract_kind="evaluator_manifest",
        schema_file="evaluator_manifest_v2.schema.json",
    )

    assert sample_manifest["provenance_mode"] == "real"
    assert evaluator_manifest["provenance_mode"] == "real"
    assert store.read_bytes(evaluator_manifest["source_blobs"][0]) == (tmp_path / "evaluator.py").read_bytes()
    assert all((tmp_path / ref["relative_path"]).is_file() for ref in sample_manifest["datasets"][0]["raw_sample_refs"])


def test_only_explicit_inventory_is_staged_even_when_legacy_fixed_path_exists(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_plan(), _variant(), project_root=tmp_path)
    attempt = _attempt(trial_spec)
    legacy = tmp_path / "experiment" / "results" / "main_results.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(encode_canonical_evidence(_measurement(attempt, rows=[_row(attempt, role="baseline", value=0.0), _row(attempt, role="candidate", value=1.0)])))

    with pytest.raises(S3ValidationError, match="inventory is empty|required evidence|missing"):
        _stage_evidence_inventory(
            project_root=tmp_path,
            attempt=attempt,
            trial_spec=trial_spec,
            inventory=[],
        )
    assert not (tmp_path / "experiment" / "attempts").exists()


def test_staged_rows_are_attempt_scoped_and_content_addressed(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_plan(), _variant(), project_root=tmp_path)
    attempt = _attempt(trial_spec)
    payload = _measurement(attempt, rows=[_row(attempt, role="baseline", value=0.0), _row(attempt, role="candidate", value=1.0)])
    inventory = _write_inventory(tmp_path, attempt, payload)

    completion = _stage_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        inventory=inventory,
    )
    assert set(completion) == {
        "schema_version",
        "attempt_id",
        "trial_spec_hash",
        "lifecycle_generation",
        "implementation_hash",
        "attempt_input_hash",
        "phase",
        "phase_execution_id",
        "producer_run_id",
        "command_plan_hash",
        "entries",
    }
    assert completion["schema_version"] == "auto_research_completion_evidence_v3"
    entry = next(item for item in completion["entries"] if item["kind"] == "main_results")
    staged_payload = json.loads((tmp_path / entry["relative_path"]).read_text(encoding="utf-8"))
    assert entry["relative_path"].startswith(f"experiment/attempts/{attempt['attempt_id']}/producer-run-1/main_results/")
    assert entry["relative_path"].endswith(f"{entry['content_hash']}.json")
    assert {item["role"] for item in staged_payload["rows"]} == {"baseline", "candidate"}


def test_cross_attempt_inventory_is_rejected(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_plan(), _variant(), project_root=tmp_path)
    attempt = _attempt(trial_spec)
    payload = _measurement(attempt, rows=[_row(attempt, role="baseline", value=0.0), _row(attempt, role="candidate", value=1.0)])
    payload["attempt_id"] = "attempt-other"
    inventory = _write_inventory(tmp_path, attempt, payload)

    with pytest.raises(S3ValidationError, match="attempt_id"):
        _stage_evidence_inventory(
            project_root=tmp_path,
            attempt=attempt,
            trial_spec=trial_spec,
            inventory=inventory,
        )


def test_missing_trial_spec_projection_is_integrity_error(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_plan(), _variant(), project_root=tmp_path)
    attempt = _attempt(trial_spec)
    errors: list[str] = []

    _validate_trial_spec_projection_drift(errors, tmp_path, attempt, deepcopy(trial_spec))

    assert errors == ["canonical TrialSpec projection is missing"]
