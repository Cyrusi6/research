from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import (
    acceptance_contract_hash,
    canonical_hash,
    classify_trial_result,
    trial_spec_hash,
    validate_contract,
    validate_trial_spec,
)
from auto_research.evidence import (
    CONTENT_ADDRESSED_EVIDENCE_SCHEMA_VERSIONS,
    EVIDENCE_MANIFEST_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSIONS,
    TRANSACTION_EVIDENCE_SCHEMA_VERSIONS,
    content_addressed_evidence_path,
    encode_canonical_evidence,
    evidence_content_hash,
)


def _trial_spec_facts(*, objective: str = "maximize", datasets: tuple[str, ...] = ("d1",), seeds: tuple[int, ...] = (7,)) -> dict:
    sample_datasets = []
    datasets_spec = []
    for dataset_id in datasets:
        sample_ids = [canonical_hash({"dataset": dataset_id, "sample": index}) for index in range(2)]
        content_hash = canonical_hash(
            {"dataset_id": dataset_id, "source_revision": "revision-1", "split": "test", "ordered_sample_ids": sample_ids}
        )
        sample_datasets.append(
            {
                "dataset_id": dataset_id,
                "source_revision": "revision-1",
                "split": "test",
                "sample_count": len(sample_ids),
                "ordered_sample_ids": sample_ids,
                "content_hash": content_hash,
            }
        )
        datasets_spec.append(
            {"dataset_id": dataset_id, "split": "test", "sample_count": len(sample_ids), "sample_hash": content_hash}
        )
    manifest = {
        "manifest_id": "sample-manifest-1",
        "provenance_mode": "real",
        "datasets": sample_datasets,
    }
    evaluator = {
        "provenance_mode": "real",
        "evaluator_id": "accuracy-evaluator",
        "source_digest": canonical_hash({"source": "evaluator.py"}),
        "config_hash": canonical_hash({"threshold": 0.5}),
        "dependency_digest": canonical_hash({"python": "3.11"}),
    }
    runtime = {"device": "cpu"}
    spec = {
        "protocol": {"protocol_id": "protocol-1", "required_phases": ["full"], "terminal_phases": ["full"], "proxy_terminal_allowed": False, "aggregation": "mean"},
        "sample_manifest": manifest,
        "datasets": datasets_spec,
        "metrics": [{"metric_id": "score", "objective": objective, "aggregation": "mean", "role": "primary"}],
        "primary_metric_id": "score",
        "statistical_testing": {"method": "paired", "seeds": list(seeds), "require_complete_seed_coverage": True},
        "required_roles": ["baseline", "candidate"],
        "acceptance_constraints": [
            {"constraint_id": "primary-delta", "kind": "minimum_mean_delta", "hard": True, "metric_id": "score", "threshold": 0.05, "objective": objective},
            {"constraint_id": "dataset-regression", "kind": "per_dataset_maximum_regression", "hard": True, "metric_id": "score", "threshold": 0.1, "objective": objective},
        ],
        "execution_contract": {"runtime_config": runtime, "runtime_config_hash": canonical_hash(runtime), "evaluator_provenance": evaluator, "evaluator_hash": canonical_hash(evaluator), "command_contract_hash": canonical_hash(["evaluate"])},
        "required_artifacts": ["main_results"],
        "evidence_requirements": [{"requirement_id": "main-results", "kind": "main_results", "required": True, "applicable_phases": ["full"]}],
    }
    return spec



def _trial_spec(
    *,
    objective: str = "maximize",
    datasets: tuple[str, ...] = ("d1",),
    seeds: tuple[int, ...] = (7,),
    project_root: Path | None = None,
) -> dict:
    from support.authoritative_evidence import build_trial_spec_v8
    return build_trial_spec_v8(
        _trial_spec_facts(objective=objective, datasets=datasets, seeds=seeds),
        project_root=project_root,
    )

def _attempt(spec: dict) -> dict:
    attempt = {
        "direction_id": "direction-1", "direction_semantic_hash": "1" * 64, "direction_spec_hash": "2" * 64,
        "variant_id": "variant-1", "variant_semantic_hash": "3" * 64, "variant_spec_hash": "4" * 64,
        "attempt_id": "attempt-1", "trial_spec_hash": trial_spec_hash(spec), "acceptance_contract_hash": acceptance_contract_hash(spec),
        "protocol_hash": canonical_hash(spec["protocol"]), "attempt_input_hash": "5" * 64,
        "implementation_hash": "6" * 64, "lifecycle_generation": 0,
        "sample_manifest_hash": canonical_hash(spec["sample_manifest"]), "evaluator_hash": spec["execution_contract"]["evaluator_hash"],
        "seeds": spec["statistical_testing"]["seeds"],
    }
    attempt["phase_executions"] = {"full": {"phase_execution_id": "phase-full-0001", "phase_start_event_id": "phase-start-full", "producer_run_id": "producer-run-1"}, "proxy": None}
    attempt["committed_proxy_outcome"] = None
    return attempt


def _main_inventory(spec: dict, values: dict[tuple[str, int], tuple[float, float]]) -> tuple[dict, dict[str, bytes]]:
    attempt = _attempt(spec)
    evidence_id = "evidence:main:attempt-1"
    producer_run_id = "producer-run-1"
    execution = attempt["phase_executions"]["full"]
    common = {"lifecycle_generation": 0, "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"], "phase": "full", "phase_execution_id": execution["phase_execution_id"], "phase_start_event_id": execution["phase_start_event_id"]}
    rows = []
    for (dataset_id, seed), (baseline, candidate) in sorted(values.items()):
        for role, value in (("baseline", baseline), ("candidate", candidate)):
            rows.append({"phase": "full", "role": role, "dataset_id": dataset_id, "metric_id": "score", "seed": seed, "metric_value": value, "command_status": "completed", "attempt_id": attempt["attempt_id"], "variant_semantic_hash": attempt["variant_semantic_hash"], "variant_spec_hash": attempt["variant_spec_hash"], "trial_spec_hash": attempt["trial_spec_hash"], "sample_manifest_hash": attempt["sample_manifest_hash"], "evaluator_hash": attempt["evaluator_hash"], "producer_run_id": producer_run_id, **common})
    payload = {"schema_version": EVIDENCE_SCHEMA_VERSIONS["main_results"], "evidence_kind": "main_results", "evidence_id": evidence_id, "attempt_id": attempt["attempt_id"], "producer_run_id": producer_run_id, "direction_semantic_hash": attempt["direction_semantic_hash"], "direction_spec_hash": attempt["direction_spec_hash"], "variant_semantic_hash": attempt["variant_semantic_hash"], "variant_spec_hash": attempt["variant_spec_hash"], "trial_spec_hash": attempt["trial_spec_hash"], "protocol_hash": attempt["protocol_hash"], "sample_manifest_hash": attempt["sample_manifest_hash"], "evaluator_hash": attempt["evaluator_hash"], "cross_references": {}, **common, "rows": rows}
    raw = encode_canonical_evidence(payload)
    digest = evidence_content_hash(payload)
    path = content_addressed_evidence_path(attempt_id=attempt["attempt_id"], producer_run_id=producer_run_id, evidence_kind="main_results", content_hash=digest)
    blob = lambda value: {"schema_version": "auto_research_contract_blob_v1", "algorithm": "sha256", "digest": value, "size_bytes": 1, "relative_path": f"meta/contracts/sha256/{value[:2]}/{value}.json"}
    entry = {"evidence_id": evidence_id, "kind": "main_results", "relative_path": path, "content_hash": digest, "schema_version": payload["schema_version"], "attempt_id": attempt["attempt_id"], "producer_run_id": producer_run_id, "direction_semantic_hash": attempt["direction_semantic_hash"], "direction_spec_hash": attempt["direction_spec_hash"], "variant_semantic_hash": attempt["variant_semantic_hash"], "variant_spec_hash": attempt["variant_spec_hash"], "trial_spec_hash": attempt["trial_spec_hash"], "protocol_hash": attempt["protocol_hash"], "sample_manifest_hash": attempt["sample_manifest_hash"], "evaluator_hash": attempt["evaluator_hash"], **common, "cross_references": {}, "command_id": "command-main-attempt-1", "command_hash": "a" * 64, "command_plan_hash": "b" * 64, "receipt_ref": blob("c" * 64), "receipt_hash": "c" * 64, "output_ref": blob(digest), "completed_event_id": "event:command:completed:main", "derivation_ref": blob("d" * 64), "derivation_hash": "d" * 64}
    manifest = {
        "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "attempt_id": attempt["attempt_id"],
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "phase": "full",
        "phase_execution_id": execution["phase_execution_id"],
        "producer_run_id": producer_run_id,
        "derivation_ref": blob("d" * 64),
        "derivation_hash": "d" * 64,
        "derive_receipt_ref": blob("c" * 64),
        "derive_receipt_hash": "c" * 64,
        "lifecycle_generation": 0,
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "entries": [entry],
    }
    return manifest, {evidence_id: raw}


def _classify(spec: dict, values: dict[tuple[str, int], tuple[float, float]], **kwargs) -> dict:
    manifest, evidence_bytes = _main_inventory(spec, values)
    return classify_trial_result(attempt=_attempt(spec), trial_spec=spec, evidence_manifest=manifest, evidence_bytes=evidence_bytes, **kwargs)


def test_identity_only_main_artifact_cannot_authorize_forged_observations() -> None:
    spec = _trial_spec()
    manifest, evidence_bytes = _main_inventory(spec, {("d1", 7): (0.0, 1.0)})
    assert _classify(spec, {("d1", 7): (0.0, 1.0)})["outcome_classification"] == "accepted"
    identity_only = {"schema_version": "auto_research_main_results_v2", "attempt_id": "attempt-1"}
    raw = encode_canonical_evidence(identity_only)
    entry = manifest["entries"][0]
    entry["content_hash"] = canonical_hash(identity_only)
    entry["relative_path"] = content_addressed_evidence_path(attempt_id="attempt-1", producer_run_id=entry["producer_run_id"], evidence_kind="main_results", content_hash=entry["content_hash"])
    evidence_bytes[entry["evidence_id"]] = raw
    with pytest.raises(ValueError, match="required property|evidence_kind|rows"):
        classify_trial_result(attempt=_attempt(spec), trial_spec=spec, evidence_manifest=manifest, evidence_bytes=evidence_bytes)


@pytest.mark.parametrize("objective,values,expected", [
    ("maximize", {("d1", 7): (0.4, 0.6), ("d2", 7): (0.5, 0.7)}, {"d1": 0.2, "d2": 0.2}),
    ("maximize", {("d1", 7): (0.4, 0.6), ("d2", 7): (0.5, 0.3)}, {"d1": 0.2, "d2": -0.2}),
    ("maximize", {("d1", 7): (0.6, 0.4), ("d2", 7): (0.7, 0.5)}, {"d1": -0.2, "d2": -0.2}),
    ("minimize", {("d1", 7): (0.6, 0.4), ("d2", 7): (0.7, 0.5)}, {"d1": 0.2, "d2": 0.2}),
    ("minimize", {("d1", 7): (0.6, 0.4), ("d2", 7): (0.5, 0.7)}, {"d1": 0.2, "d2": -0.2}),
    ("minimize", {("d1", 7): (0.4, 0.6), ("d2", 7): (0.5, 0.7)}, {"d1": -0.2, "d2": -0.2}),
])
def test_paired_dataset_improvement_respects_metric_objective(objective: str, values: dict, expected: dict) -> None:
    spec = _trial_spec(objective=objective, datasets=("d1", "d2"))
    assert _classify(spec, {("d1", 7): (0.4, 0.6), ("d2", 7): (0.5, 0.7)}) if objective == "maximize" else _classify(spec, {("d1", 7): (0.6, 0.4), ("d2", 7): (0.7, 0.5)})
    result = _classify(spec, values)
    regression = next(item for item in result["constraint_results"] if item["kind"] == "per_dataset_maximum_regression")
    assert regression["observed"]["deltas"] == pytest.approx(expected)


@pytest.mark.parametrize("attack", ["missing_seed", "extra_seed", "extra_dataset", "duplicate", "wrong_role", "global_scalar", "bool"])
def test_quantitative_row_attacks_fail_after_valid_baseline(attack: str) -> None:
    spec = _trial_spec(seeds=(7, 8))
    values = {("d1", 7): (0.0, 1.0), ("d1", 8): (0.0, 1.0)}
    assert _classify(spec, values)["outcome_classification"] == "accepted"
    manifest, evidence_bytes = _main_inventory(spec, values)
    payload = json.loads(evidence_bytes[manifest["entries"][0]["evidence_id"]])
    if attack == "missing_seed": payload["rows"] = [row for row in payload["rows"] if row["seed"] != 8]
    elif attack == "extra_seed": payload["rows"][0]["seed"] = 9
    elif attack == "extra_dataset": payload["rows"][0]["dataset_id"] = "d2"
    elif attack == "duplicate": payload["rows"].append(deepcopy(payload["rows"][0]))
    elif attack == "wrong_role": payload["rows"][0]["role"] = "ablation"
    elif attack == "global_scalar": payload["global_mean"] = 0.5
    elif attack == "bool": payload["rows"][0]["metric_value"] = True
    raw = encode_canonical_evidence(payload)
    digest = evidence_content_hash(payload)
    entry = manifest["entries"][0]
    entry["content_hash"] = digest
    entry["relative_path"] = content_addressed_evidence_path(attempt_id="attempt-1", producer_run_id=entry["producer_run_id"], evidence_kind="main_results", content_hash=digest)
    evidence_bytes[entry["evidence_id"]] = raw
    with pytest.raises(ValueError):
        classify_trial_result(attempt=_attempt(spec), trial_spec=spec, evidence_manifest=manifest, evidence_bytes=evidence_bytes)


@pytest.mark.parametrize("field", ["constraint_results", "outcome_classification", "primary_metric_summary", "all_hard_constraints_passed", "observations", "raw_artifacts"])
def test_every_caller_derived_field_must_equal_canonical_result(field: str) -> None:
    spec = _trial_spec()
    result = _classify(spec, {("d1", 7): (0.0, 1.0)})
    diagnostic = deepcopy(result)
    if isinstance(diagnostic[field], bool): diagnostic[field] = not diagnostic[field]
    elif isinstance(diagnostic[field], str): diagnostic[field] = "rejected"
    elif isinstance(diagnostic[field], list): diagnostic[field] = []
    elif isinstance(diagnostic[field], dict): diagnostic[field] = {}
    manifest, evidence_bytes = _main_inventory(spec, {("d1", 7): (0.0, 1.0)})
    with pytest.raises(ValueError, match="diagnostic TrialResult"):
        classify_trial_result(attempt=_attempt(spec), trial_spec=spec, evidence_manifest=manifest, evidence_bytes=evidence_bytes, diagnostic_result=diagnostic)


@pytest.mark.parametrize("field,value", [
    ("metric_value", 9.0), ("role", "baseline"), ("dataset_id", "forged"),
    ("seed", 99), ("metric_id", "forged"), ("raw_artifact_path", "experiment/attempts/forged.json"),
])
def test_diagnostic_observation_cannot_disagree_with_decoded_raw_row(field: str, value) -> None:
    spec = _trial_spec()
    result = _classify(spec, {("d1", 7): (0.0, 1.0)})
    diagnostic = deepcopy(result)
    diagnostic["observations"][0][field] = "candidate" if field == "role" else value
    manifest, evidence_bytes = _main_inventory(spec, {("d1", 7): (0.0, 1.0)})
    with pytest.raises(ValueError, match="diagnostic TrialResult"):
        classify_trial_result(attempt=_attempt(spec), trial_spec=spec, evidence_manifest=manifest, evidence_bytes=evidence_bytes, diagnostic_result=diagnostic)


@pytest.mark.parametrize("mutation", ["sample_id", "sample_artifact_hash", "evaluator_source", "evaluator_hash"])
def test_trial_spec_provenance_mutations_are_rejected_after_valid_baseline(mutation: str) -> None:
    spec = _trial_spec()
    validate_trial_spec(spec)
    changed = deepcopy(spec)
    if mutation == "sample_id":
        changed["sample_manifest"]["datasets"][0]["ordered_sample_ids"][1] = changed["sample_manifest"]["datasets"][0]["ordered_sample_ids"][0]
    elif mutation == "sample_artifact_hash":
        changed["sample_manifest"]["artifact_hash"] = "f" * 64
    elif mutation == "evaluator_source":
        changed["execution_contract"]["evaluator_provenance"]["source_digest"] = "f" * 64
    else:
        changed["execution_contract"]["evaluator_hash"] = "f" * 64
    with pytest.raises(ValueError):
        validate_trial_spec(changed)


def test_all_evidence_kind_schemas_are_closed_and_versioned() -> None:
    for kind, version in CONTENT_ADDRESSED_EVIDENCE_SCHEMA_VERSIONS.items():
        schema_name = f"{kind}_v{version.rsplit('_v', 1)[1]}.schema.json"
        schema = json.loads((__import__("pathlib").Path(__file__).parents[1] / "src/auto_research/schemas" / schema_name).read_text())
        assert schema["additionalProperties"] is False
        assert schema["properties"]["schema_version"]["const"] == version


def _nonquantitative_payload(kind: str) -> dict:
    version = EVIDENCE_SCHEMA_VERSIONS[kind]
    base = {
        "schema_version": version, "evidence_kind": kind, "evidence_id": f"evidence:{kind}",
        "attempt_id": "attempt-1", "producer_run_id": "producer-run-1",
        "direction_semantic_hash": "1" * 64, "direction_spec_hash": "2" * 64,
        "variant_semantic_hash": "3" * 64, "variant_spec_hash": "4" * 64,
        "trial_spec_hash": "5" * 64, "protocol_hash": "6" * 64,
        "sample_manifest_hash": "7" * 64, "evaluator_hash": "8" * 64,
        "lifecycle_generation": 0, "implementation_hash": "9" * 64,
        "attempt_input_hash": "a" * 64, "phase": "proxy",
        "phase_execution_id": "phase-proxy-0001", "phase_start_event_id": "phase-start-proxy",
        "cross_references": {},
    }
    extras = {
        "activation_evidence": {
            "probe_id": "forward-probe",
            "status": "activated",
            "command_status": "completed",
            "exit_code": 0,
            "expected_surface_ids": ["src/model.py"],
            "observed_surface_ids": ["src/model.py"],
            "activation_delta_threshold": 0.1,
            "surface_measurements": [{
                "surface_id": "src/model.py",
                "enabled_value": 1.0,
                "disabled_value": 0.0,
                "delta": 1.0,
                "threshold": 0.1,
                "status": "ACTIVATED",
            }],
        },
        "proxy_baseline_fingerprint": {"baseline_hash": "9" * 64, "dataset_ids": ["d1"], "seeds": [7], "fingerprint_inputs": {"sample_manifest_hash": "7" * 64, "evaluator_hash": "8" * 64, "protocol_hash": "6" * 64, "phase_execution_id": "phase-proxy-0001"}},
        "proxy_cache_report": {"cross_references": {"proxy_baseline_fingerprint_hash": "a" * 64}, "cache_key": "b" * 64, "baseline_hash": "9" * 64, "cache_entry_hash": "c" * 64, "status": "hit"},
        "full_s3_readiness": {
            "cross_references": {"activation_evidence_hash": "a" * 64, "proxy_results_hash": "b" * 64},
            "readiness_check_plan_ref": {
                "schema_version": "auto_research_contract_blob_v1",
                "algorithm": "sha256",
                "digest": "c" * 64,
                "size_bytes": 1,
                "relative_path": f"meta/contracts/sha256/cc/{'c' * 64}.json",
            },
            "readiness_check_plan_hash": "c" * 64,
            "ready": True,
            "classification": "PASS",
            "checks": [{
                "check_id": "activation",
                "status": "PASS",
                "measurement": True,
                "comparator": "eq",
                "threshold": True,
            }],
        },
        "bootstrap_completion": {"cross_references": {"activation_evidence_hash": "a" * 64, "proxy_results_hash": "b" * 64}, "completion_status": "verified"},
    }
    base.update(extras[kind])
    return base


@pytest.mark.parametrize("kind", sorted(set(EVIDENCE_SCHEMA_VERSIONS) - {"main_results", "proxy_results", "ablation_results", "coverage_results", "matched_control_results"}))
def test_nonquantitative_evidence_schema_baseline_then_version_and_extra_property_attack(kind: str) -> None:
    payload = _nonquantitative_payload(kind)
    schema_name = f"{kind}_v{payload['schema_version'].rsplit('_v', 1)[1]}.schema.json"
    validate_contract(payload, schema_name)
    wrong_version = deepcopy(payload)
    wrong_version["schema_version"] += "-forged"
    with pytest.raises(ValueError):
        validate_contract(wrong_version, schema_name)
    extra = deepcopy(payload)
    extra["forged"] = True
    with pytest.raises(ValueError):
        validate_contract(extra, schema_name)


def test_transaction_evidence_is_not_completion_manifest_evidence() -> None:
    assert TRANSACTION_EVIDENCE_SCHEMA_VERSIONS == {
        "failure_evidence": "auto_research_failure_evidence_v6",
        "resource_probe": "auto_research_resource_probe_evidence_v4",
        "resume_evidence": "auto_research_resume_evidence_v5",
    }
    assert set(TRANSACTION_EVIDENCE_SCHEMA_VERSIONS).isdisjoint(EVIDENCE_SCHEMA_VERSIONS)
    assert "effective_proxy_policy" not in CONTENT_ADDRESSED_EVIDENCE_SCHEMA_VERSIONS
    assert "proxy_calibration_policy" not in CONTENT_ADDRESSED_EVIDENCE_SCHEMA_VERSIONS


@pytest.mark.parametrize(
    "kind",
    [
        "effective_proxy_policy",
        "proxy_calibration_policy",
        "failure_evidence",
        "resource_probe",
        "resume_evidence",
    ],
)
def test_completion_manifest_rejects_noncompletion_evidence_kinds(kind: str) -> None:
    spec = _trial_spec()
    manifest, _ = _main_inventory(spec, {("d1", 7): (0.0, 1.0)})
    forged = deepcopy(manifest)
    forged["entries"][0]["kind"] = kind
    with pytest.raises(ValueError):
        validate_contract(forged, "evidence_manifest_v6.schema.json")


def test_evidence_manifest_v6_rejects_replaced_inline_derivation() -> None:
    spec = _trial_spec()
    manifest, _ = _main_inventory(spec, {("d1", 7): (0.0, 1.0)})
    forged = deepcopy(manifest)
    forged["entries"][0]["derivation"] = {
        "decoder_id": "legacy-decoder",
        "decoder_version": "1",
        "decoder_hash": "e" * 64,
        "source_output_hashes": ["f" * 64],
    }
    with pytest.raises(ValueError):
        validate_contract(forged, "evidence_manifest_v6.schema.json")


def test_replaced_authoritative_schema_files_are_absent() -> None:
    schema_root = Path(__file__).parents[1] / "src" / "auto_research" / "schemas"
    replaced = {
        "trial_spec_v6.schema.json",
        "phase_command_plan_v1.schema.json",
        "phase_command_v2.schema.json",
        "phase_run_receipt_v3.schema.json",
        "evidence_manifest_v4.schema.json",
        "sample_manifest_v3.schema.json",
        "failure_evidence_v5.schema.json",
        "event_v7.schema.json",
        "attempt_record_v7.schema.json",
        "research_state_v7.schema.json",
    }
    assert not any((schema_root / name).exists() for name in replaced)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_quantitative_rows_fail_after_valid_baseline(value: float) -> None:
    spec = _trial_spec()
    assert _classify(spec, {("d1", 7): (0.0, 1.0)})
    manifest, evidence_bytes = _main_inventory(spec, {("d1", 7): (0.0, 1.0)})
    evidence_id = manifest["entries"][0]["evidence_id"]
    payload = json.loads(evidence_bytes[evidence_id])
    payload["rows"][0]["metric_value"] = value
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    digest = __import__("hashlib").sha256(raw).hexdigest()
    entry = manifest["entries"][0]
    entry["content_hash"] = digest
    entry["relative_path"] = content_addressed_evidence_path(attempt_id="attempt-1", producer_run_id=entry["producer_run_id"], evidence_kind="main_results", content_hash=digest)
    evidence_bytes[evidence_id] = raw
    with pytest.raises(ValueError):
        classify_trial_result(attempt=_attempt(spec), trial_spec=spec, evidence_manifest=manifest, evidence_bytes=evidence_bytes)
