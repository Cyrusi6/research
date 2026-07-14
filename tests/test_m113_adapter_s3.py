from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.agents.experiment import (
    _decode_staged_execution_observations,
    _stage_evidence_inventory,
)
from auto_research.agents.plan import _trial_spec_from_plan
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
    trial_spec = _trial_spec_from_plan(_plan(), _variant())
    attempt = _attempt(trial_spec)
    source = tmp_path / "runner" / "main.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(encode_canonical_evidence(_measurement(attempt)))

    with pytest.raises(S3ValidationError, match=r"rows: \[\] is too short"):
        _stage_evidence_inventory(
            project_root=tmp_path,
            attempt=attempt,
            trial_spec=trial_spec,
            inventory=[{"kind": "main_results", "source_path": "runner/main.json", "producer_run_id": "producer-run-1"}],
        )
    assert not (tmp_path / "experiment" / "attempts").exists()


def test_only_explicit_inventory_is_staged_even_when_legacy_fixed_path_exists(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_plan(), _variant())
    attempt = _attempt(trial_spec)
    legacy = tmp_path / "experiment" / "results" / "main_results.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(encode_canonical_evidence(_measurement(attempt, rows=[_row(attempt, role="baseline", value=0.0), _row(attempt, role="candidate", value=1.0)])))

    with pytest.raises(S3ValidationError, match="required evidence|missing"):
        _stage_evidence_inventory(
            project_root=tmp_path,
            attempt=attempt,
            trial_spec=trial_spec,
            inventory=[],
        )
    assert not (tmp_path / "experiment" / "attempts").exists()


def test_staged_rows_are_attempt_scoped_and_content_addressed(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_plan(), _variant())
    attempt = _attempt(trial_spec)
    payload = _measurement(attempt, rows=[_row(attempt, role="baseline", value=0.0), _row(attempt, role="candidate", value=1.0)])
    source = tmp_path / "runner" / "main.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(encode_canonical_evidence(payload))

    completion = _stage_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        inventory=[{"kind": "main_results", "source_path": "runner/main.json", "producer_run_id": "producer-run-1"}],
    )
    observations = _decode_staged_execution_observations(tmp_path, attempt, trial_spec, completion)

    assert set(completion) == {"schema_version", "attempt_id", "trial_spec_hash", "lifecycle_generation", "implementation_hash", "attempt_input_hash", "entries"}
    assert completion["schema_version"] == "auto_research_completion_evidence_v2"
    entry = completion["entries"][0]
    assert entry["relative_path"].startswith(f"experiment/attempts/{attempt['attempt_id']}/producer-run-1/main_results/")
    assert entry["relative_path"].endswith(f"{entry['content_hash']}.json")
    assert {item["role"] for item in observations} == {"baseline", "candidate"}
    assert {item["raw_artifact_path"] for item in observations} == {entry["relative_path"]}
    assert {item["raw_artifact_hash"] for item in observations} == {entry["content_hash"]}


def test_cross_attempt_inventory_is_rejected(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_plan(), _variant())
    attempt = _attempt(trial_spec)
    payload = _measurement(attempt, rows=[_row(attempt, role="baseline", value=0.0), _row(attempt, role="candidate", value=1.0)])
    payload["attempt_id"] = "attempt-other"
    source = tmp_path / "runner" / "main.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(encode_canonical_evidence(payload))

    with pytest.raises(S3ValidationError, match="attempt_id"):
        _stage_evidence_inventory(
            project_root=tmp_path,
            attempt=attempt,
            trial_spec=trial_spec,
            inventory=[{"kind": "main_results", "source_path": "runner/main.json", "producer_run_id": "producer-run-1"}],
        )


def test_missing_trial_spec_projection_is_integrity_error(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_plan(), _variant())
    errors: list[str] = []

    _validate_trial_spec_projection_drift(errors, tmp_path, deepcopy(trial_spec))

    assert errors == ["canonical TrialSpec projection is missing"]
