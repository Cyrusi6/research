from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from auto_research.agents.experiment import (
    ExperimentAgent,
    _c2c_strict_evidence_inventory,
)
from auto_research.contract_store import ContractStore
from auto_research.derivation_contracts import build_readiness_check_plan
from auto_research.derivation_validation import ReceiptBoundSources
from auto_research.domain_contracts import canonical_hash
from auto_research.evidence import encode_canonical_evidence
from auto_research.proxy_classifier import (
    build_proxy_decision_policy,
    classify_proxy_outcome,
    derive_readiness_from_receipts,
)
from auto_research.research_state import IntegrityError, ResearchEventLedger
from auto_research.validators import run_stage_gate
from support.authoritative_evidence import (
    record_completed_evidence_command,
    stage_authoritative_completion,
)
from support.local_c2c_execution import (
    build_c2c_context,
    create_local_c2c_repo,
    install_fake_gpu,
)
from test_m114_authoritative_phase_transactions import _c2c_inputs


_READINESS_PLAN_HASH = "d" * 64
_READINESS_PLAN_REF = {
    "schema_version": "auto_research_contract_blob_v1",
    "algorithm": "sha256",
    "digest": _READINESS_PLAN_HASH,
    "size_bytes": 1,
    "relative_path": f"meta/contracts/sha256/dd/{_READINESS_PLAN_HASH}.json",
}


def _policy() -> dict[str, Any]:
    return build_proxy_decision_policy(
        primary_metric_id="accuracy",
        objective="maximize",
        aggregation="paired_mean",
        datasets=["dataset-a"],
        seeds=[7],
        metric_ids=["accuracy"],
        roles=["baseline", "candidate"],
        aggregate_improvement_threshold=0.05,
        per_dataset_maximum_regression=0.01,
        activation_delta_threshold=0.01,
        activation_surface_ids=["src/model.py", "src/router.py"],
        readiness_check_ids=["runtime-ready"],
        readiness_check_plan_ref=_READINESS_PLAN_REF,
        readiness_check_plan_hash=_READINESS_PLAN_HASH,
        evidence_kinds=["activation_evidence", "full_s3_readiness", "proxy_results"],
        mode="gate_to_full",
    )


def _binding(policy: dict[str, Any]) -> dict[str, Any]:
    binding = {
        "schema_version": "auto_research_proxy_evaluation_binding_v1",
        "policy_hash": policy["policy_hash"],
        "attempt_id": "attempt-m1153",
        "direction_semantic_hash": "1" * 64,
        "direction_spec_hash": "2" * 64,
        "variant_semantic_hash": "3" * 64,
        "variant_spec_hash": "4" * 64,
        "trial_spec_hash": "5" * 64,
        "lifecycle_generation": 0,
        "implementation_hash": "a" * 64,
        "attempt_input_hash": "b" * 64,
        "phase_execution_id": "phase-proxy-m1153",
        "phase_start_event_id": "event:proxy:m1153",
        "producer_run_id": "producer-m1153",
        "command_plan_hash": "6" * 64,
        "phase_contract_hash": "7" * 64,
        "sample_contract_ref": {"artifact_hash": "8" * 64},
        "evaluator_contract_ref": {"artifact_hash": "9" * 64},
        "provenance_mode": "local-external",
        "expected_evidence_kinds": policy["evidence_kinds"],
    }
    binding["binding_hash"] = canonical_hash(binding)
    return binding


def _identity(kind: str, evidence_id: str) -> dict[str, Any]:
    return {
        "evidence_kind": kind,
        "evidence_id": evidence_id,
        "attempt_id": "attempt-m1153",
        "lifecycle_generation": 0,
        "implementation_hash": "a" * 64,
        "attempt_input_hash": "b" * 64,
        "phase": "proxy",
        "phase_execution_id": "phase-proxy-m1153",
        "phase_start_event_id": "event:proxy:m1153",
        "producer_run_id": "producer-m1153",
        "direction_semantic_hash": "1" * 64,
        "direction_spec_hash": "2" * 64,
        "variant_semantic_hash": "3" * 64,
        "variant_spec_hash": "4" * 64,
        "trial_spec_hash": "5" * 64,
        "protocol_hash": "c" * 64,
        "sample_manifest_hash": "8" * 64,
        "evaluator_hash": "9" * 64,
        "cross_references": {},
    }


def _decoded_evidence(*, candidate: float = 0.60) -> dict[str, dict[str, Any]]:
    return {
        "proxy": {
            "schema_version": "auto_research_proxy_results_v1",
            **_identity("proxy_results", "evidence-proxy-m1153"),
            "rows": [
                {
                    "phase": "proxy",
                    "role": "baseline",
                    "dataset_id": "dataset-a",
                    "metric_id": "accuracy",
                    "seed": 7,
                    "metric_value": 0.50,
                    "command_status": "completed",
                    "attempt_id": "attempt-m1153",
                    "variant_semantic_hash": "3" * 64,
                    "variant_spec_hash": "4" * 64,
                    "trial_spec_hash": "5" * 64,
                    "sample_manifest_hash": "8" * 64,
                    "evaluator_hash": "9" * 64,
                    "producer_run_id": "producer-m1153",
                    "lifecycle_generation": 0,
                    "implementation_hash": "a" * 64,
                    "attempt_input_hash": "b" * 64,
                    "phase_execution_id": "phase-proxy-m1153",
                    "phase_start_event_id": "event:proxy:m1153",
                },
                {
                    "phase": "proxy",
                    "role": "candidate",
                    "dataset_id": "dataset-a",
                    "metric_id": "accuracy",
                    "seed": 7,
                    "metric_value": candidate,
                    "command_status": "completed",
                    "attempt_id": "attempt-m1153",
                    "variant_semantic_hash": "3" * 64,
                    "variant_spec_hash": "4" * 64,
                    "trial_spec_hash": "5" * 64,
                    "sample_manifest_hash": "8" * 64,
                    "evaluator_hash": "9" * 64,
                    "producer_run_id": "producer-m1153",
                    "lifecycle_generation": 0,
                    "implementation_hash": "a" * 64,
                    "attempt_input_hash": "b" * 64,
                    "phase_execution_id": "phase-proxy-m1153",
                    "phase_start_event_id": "event:proxy:m1153",
                },
            ],
        },
        "activation": {
            "schema_version": "auto_research_activation_evidence_v4",
            **_identity("activation_evidence", "evidence-activation-m1153"),
            "probe_id": "activation-probe-m1153",
            "status": "activated",
            "command_status": "completed",
            "exit_code": 0,
            "expected_surface_ids": ["src/model.py", "src/router.py"],
            "observed_surface_ids": ["src/model.py", "src/router.py"],
            "activation_delta_threshold": 0.01,
            "surface_measurements": [
                {
                    "surface_id": surface_id,
                    "enabled_value": 1.0,
                    "disabled_value": 0.0,
                    "delta": 1.0,
                    "threshold": 0.01,
                    "status": "ACTIVATED",
                }
                for surface_id in ["src/model.py", "src/router.py"]
            ],
        },
        "readiness": {
            "schema_version": "auto_research_full_s3_readiness_v4",
            **_identity("full_s3_readiness", "evidence-readiness-m1153"),
            "cross_references": {
                "activation_evidence_hash": "a" * 64,
                "proxy_results_hash": "b" * 64,
            },
            "readiness_check_plan_ref": _READINESS_PLAN_REF,
            "readiness_check_plan_hash": _READINESS_PLAN_HASH,
            "ready": True,
            "classification": "PASS",
            "checks": [
                {
                    "check_id": "runtime-ready",
                    "status": "PASS",
                    "measurement": True,
                    "comparator": "eq",
                    "threshold": True,
                },
            ],
        },
    }


def _classify(evidence: dict[str, dict[str, Any]], *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    frozen_policy = policy or _policy()
    return classify_proxy_outcome(
        frozen_policy=frozen_policy,
        evaluation_binding=_binding(frozen_policy),
        decoded_evidence=evidence,
        evidence_manifest_hash="f" * 64,
        derivation_manifest_hash="e" * 64,
    )


@pytest.fixture(scope="module")
def real_c2c_pass(tmp_path_factory: pytest.TempPathFactory):
    parent = tmp_path_factory.mktemp("m1153-real-c2c")
    monkeypatch = pytest.MonkeyPatch()
    install_fake_gpu(parent, monkeypatch)
    repo = create_local_c2c_repo(parent, proxy_accuracy=0.51)
    root = parent / "project"
    root.mkdir()
    context = build_c2c_context(root, repo, profile="standard")
    result = ExperimentAgent(context).run()
    ledger = ResearchEventLedger(root)
    events = ledger.events()
    event_types = [event["event_type"] for event in events]
    proxy_event = next(event for event in events if event["event_type"] == "ProxyEvidenceCommitted")
    assert proxy_event["payload"]["proxy_outcome"]["decision"] == "RUN_FULL", result
    assert event_types.index("ProxyEvidenceCommitted") < event_types.index("FullPhaseStarted")
    assert run_stage_gate("S3_experiment", root, context.config).to_dict()["status"] == "PASS"
    yield root, ledger, proxy_event
    monkeypatch.undo()


def _event_evidence(root: Path, proxy_event: dict[str, Any], kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    entry = next(
        item
        for item in proxy_event["payload"]["evidence_manifest"]["entries"]
        if item["kind"] == kind
    )
    payload = json.loads((root / entry["relative_path"]).read_text(encoding="utf-8"))
    return entry, payload


def _proxy_raw_payloads(root: Path, ledger: ResearchEventLedger) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    store = ContractStore(root)
    payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in ledger.state()["phase_commands"].values():
        command = record["command"]
        if command["phase"] != "proxy" or record["status"] != "completed":
            continue
        if command["command_spec_id"].endswith("derive-evidence"):
            continue
        receipt = store.read_json(record["receipt_ref"], schema_file="phase_run_receipt_v5.schema.json")
        for output in receipt.get("raw_outputs") or []:
            raw = store.read_bytes(output["contract_ref"])
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append((output, payload))
    return payloads


def _observed_check_ids(raw_payloads: list[tuple[dict[str, Any], dict[str, Any]]]) -> set[str]:
    observed: set[str] = set()
    for output, payload in raw_payloads:
        for candidate in (output, payload):
            check_id = candidate.get("check_id")
            if isinstance(check_id, str) and check_id:
                observed.add(check_id)
            checks = candidate.get("checks")
            if isinstance(checks, list):
                observed.update(
                    str(item["check_id"])
                    for item in checks
                    if isinstance(item, dict) and isinstance(item.get("check_id"), str)
                )
    return observed


def _observed_surface_ids(raw_payloads: list[tuple[dict[str, Any], dict[str, Any]]]) -> set[str]:
    observed: set[str] = set()
    for output, payload in raw_payloads:
        for candidate in (output, payload):
            values = candidate.get("observed_surface_ids")
            if isinstance(values, list):
                observed.update(str(item) for item in values if isinstance(item, str) and item)
    return observed


def test_real_activation_exit_zero_cannot_batch_synthesize_all_readiness_pass(
    real_c2c_pass,
) -> None:
    root, ledger, proxy_event = real_c2c_pass
    entry, readiness = _event_evidence(root, proxy_event, "full_s3_readiness")
    required = set(proxy_event["payload"]["proxy_outcome"]["readiness_check_ids"])
    assert readiness["ready"] is True
    assert {item["check_id"] for item in readiness["checks"]} == required
    assert all(item["status"] == "PASS" for item in readiness["checks"])

    observed = _observed_check_ids(_proxy_raw_payloads(root, ledger))
    assert observed == required, (
        "RUN_FULL readiness must be decoded from independent physical check outputs; "
        f"evidence_hash={entry['content_hash']} required={sorted(required)} observed={sorted(observed)}"
    )


def test_real_expected_activation_surfaces_cannot_be_reported_without_observation(
    real_c2c_pass,
) -> None:
    root, ledger, proxy_event = real_c2c_pass
    entry, activation = _event_evidence(root, proxy_event, "activation_evidence")
    expected = set(proxy_event["payload"]["proxy_outcome"]["activation_surface_ids"])
    assert activation["status"] == "activated"
    assert set(activation["expected_surface_ids"]) == expected
    assert set(activation["observed_surface_ids"]) == expected

    observed = _observed_surface_ids(_proxy_raw_payloads(root, ledger))
    assert observed == expected, (
        "activation surfaces must come from raw observed coverage, not the frozen expected list; "
        f"evidence_hash={entry['content_hash']} expected={sorted(expected)} observed={sorted(observed)}"
    )


@pytest.mark.parametrize("attack", ["readiness", "missing_surface", "activation_delta"])
def test_real_receipt_semantic_block_repairs_without_full_and_replays(
    tmp_path: Path,
    real_c2c_pass,
    attack: str,
) -> None:
    del real_c2c_pass
    monkeypatch = pytest.MonkeyPatch()
    install_fake_gpu(tmp_path, monkeypatch)
    try:
        repo = create_local_c2c_repo(tmp_path / "fixture", proxy_accuracy=0.51)
        control_path = repo / "local_execution_control.json"
        control = json.loads(control_path.read_text(encoding="utf-8"))
        if attack == "readiness":
            control["readiness_checks"]["proxy-ready-for-full"]["measurement"] = 0.0
        elif attack == "missing_surface":
            control["activation_observed_surfaces"] = []
        else:
            control["activation_disabled_accuracy"] = control["proxy_accuracy"]
        control_path.write_text(json.dumps(control, sort_keys=True), encoding="utf-8")

        root = tmp_path / f"blocked-{attack}"
        root.mkdir()
        context = build_c2c_context(root, repo, profile="standard")
        result = ExperimentAgent(context).run()
        ledger = ResearchEventLedger(root)
        events = ledger.events()
        proxy_event = next(event for event in events if event["event_type"] == "ProxyEvidenceCommitted")

        assert proxy_event["payload"]["proxy_outcome"]["decision"] == "REPAIR_IMPLEMENTATION"
        assert result["route_outcome"]["next_action"] == "REPAIR_IMPLEMENTATION"
        assert result["attempt"]["state"] == "IMPLEMENTATION_REPAIR"
        assert "FullPhaseStarted" not in [event["event_type"] for event in events]
        state = ledger.state()
        assert state["directions"][result["attempt"]["direction_semantic_hash"]]["budget"] == {
            "target": 5,
            "reserved": 1,
            "consumed": 0,
        }
        assert state["trial_results"] == {}
        assert state["method_tried_history"] == []
        markers = list((root / "experiment" / "execution_repos").rglob("local_command_invocations.jsonl"))
        assert markers
        marker_bytes = {path: path.read_bytes() for path in markers}
        assert all(
            "/proxy/" in str(record.get("config") or record.get("output") or "")
            or "proxy_" in str(record.get("config") or record.get("output") or "")
            for path in markers
            for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        )
        assert run_stage_gate("S3_experiment", root, context.config).to_dict()["status"] == "PASS"

        before_events = ledger.events()
        replay = ExperimentAgent(context).run()
        assert replay["route_outcome"] == result["route_outcome"]
        assert ResearchEventLedger(root).events() == before_events
        assert {path: path.read_bytes() for path in markers} == marker_bytes
    finally:
        monkeypatch.undo()


def _readiness_sources(*, include_measurements: bool) -> tuple[dict[str, Any], ReceiptBoundSources]:
    decoder_ref = {
        "schema_version": "auto_research_contract_blob_v1",
        "algorithm": "sha256",
        "digest": "e" * 64,
        "size_bytes": 1,
        "relative_path": f"meta/contracts/sha256/ee/{'e' * 64}.json",
    }
    decoder = {
        "schema_version": "auto_research_decoder_descriptor_v1",
        "decoder_id": "canonical-identity",
        "decoder_version": "1",
        "semantic_hash": "f" * 64,
        "implementation_hash": "e" * 64,
        "immutable_ref": decoder_ref,
    }
    checks: list[dict[str, Any]] = []
    raw_facts: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_lineage: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ordinal, check_id in enumerate(("runtime-ready", "storage-ready")):
        command_spec_id = f"proxy-check-{ordinal:02d}"
        output_id = f"proxy-check-output-{ordinal:02d}"
        binding = {
            "source_ordinal": 0,
            "source_phase": "proxy",
            "command_spec_id": command_spec_id,
            "output_id": output_id,
            "output_kind": "raw_readiness_check",
            "output_schema_version": "auto_research_raw_readiness_check_v1",
        }
        checks.append(
            {
                "ordinal": ordinal,
                "check_id": check_id,
                "check_kind": "raw_measurement",
                "source_bindings": [binding],
                "predicate": {
                    "field_path": f"{output_id}.ready",
                    "comparator": "eq",
                    "threshold": True,
                },
                "required_coverage": {"mode": "exact", "expected_surface_ids": []},
                "decoder_descriptor": decoder,
                "blocked_classification": "IMPLEMENTATION_BLOCKED",
                "blocked_route": "REPAIR_IMPLEMENTATION",
            }
        )
        key = ("proxy", command_spec_id, output_id)
        raw_facts[key] = {"ready": True} if include_measurements else {"exit_code": 0}
        raw_lineage[key] = {
            "source_phase": "proxy",
            "command_spec_id": command_spec_id,
            "output_id": output_id,
            "output_kind": "raw_readiness_check",
            "command_status": "completed",
            "exit_code": 0,
            "receipt_hash": "1" * 64,
            "receipt_ref": {"digest": "1" * 64},
            "output_ref": {"digest": "2" * 64},
            "completed_event_id": f"event:completed:{ordinal}",
        }
    plan = build_readiness_check_plan(
        plan_id="readiness-plan-m1153",
        phase="proxy",
        checks=checks,
    )
    sources = ReceiptBoundSources(
        raw_facts=raw_facts,
        raw_fact_lineage=raw_lineage,
        surface_checks=(),
        physical_inputs=(),
    )
    return plan, sources


def test_activation_command_exit_zero_without_check_measurements_is_not_readiness_pass() -> None:
    plan, sources = _readiness_sources(include_measurements=True)
    valid = derive_readiness_from_receipts(
        readiness_check_plan=plan,
        receipt_bound_sources=sources,
    )
    assert valid == {
        "readiness_check_plan_hash": canonical_hash(plan),
        "ready": True,
        "classification": "PASS",
        "checks": [
            {
                "check_id": "runtime-ready",
                "status": "PASS",
                "measurement": True,
                "comparator": "eq",
                "threshold": True,
            },
            {
                "check_id": "storage-ready",
                "status": "PASS",
                "measurement": True,
                "comparator": "eq",
                "threshold": True,
            },
        ],
    }

    exit_only_plan, exit_only_sources = _readiness_sources(include_measurements=False)
    with pytest.raises(ValueError, match="receipt|predicate|measurement|readiness"):
        derive_readiness_from_receipts(
            readiness_check_plan=exit_only_plan,
            receipt_bound_sources=exit_only_sources,
        )


def test_blocked_readiness_has_deterministic_repair_proxy_outcome_and_replay() -> None:
    baseline = _classify(_decoded_evidence())
    assert baseline["decision"] == "RUN_FULL"

    blocked = _decoded_evidence()
    blocked["readiness"]["ready"] = False
    blocked["readiness"]["classification"] = "BLOCKED"
    blocked["readiness"]["checks"][0]["status"] = "BLOCKED"
    blocked["readiness"]["checks"][0]["measurement"] = False
    first = _classify(blocked)
    second = _classify(deepcopy(blocked))
    assert first == second
    assert first["decision"] == "REPAIR_IMPLEMENTATION"
    assert "proxy_readiness_blocked" in first["reason_codes"]


@pytest.mark.parametrize("attack", ["missing", "extra", "duplicate", "wrong_id", "cross_attempt", "generation"])
def test_readiness_inventory_attacks_are_zero_write_after_valid_baseline(
    tmp_path: Path,
    attack: str,
) -> None:
    baseline_root = tmp_path / "baseline"
    baseline_attempt, baseline_trial, baseline_comparison, baseline_values = _c2c_inputs(baseline_root)
    baseline_ledger = ResearchEventLedger(baseline_root)
    baseline_inventory = _c2c_strict_evidence_inventory(
        project_root=baseline_root,
        attempt=baseline_attempt,
        trial_spec=baseline_trial,
        comparison_candidate=baseline_comparison,
        baseline=baseline_values,
        simulate=True,
    )
    baseline_completion = stage_authoritative_completion(
        baseline_root, baseline_attempt, baseline_trial, baseline_inventory
    )
    record_completed_evidence_command(
        baseline_root, baseline_ledger, baseline_attempt, baseline_completion
    )
    _, baseline_route = baseline_ledger.commit_proxy_evidence(baseline_completion)
    assert baseline_route["next_action"] == "RUN_FULL"

    attack_root = tmp_path / "attack"
    attempt, trial_spec, comparison, baseline_values = _c2c_inputs(attack_root)
    ledger = ResearchEventLedger(attack_root)
    inventory = _c2c_strict_evidence_inventory(
        project_root=attack_root,
        attempt=attempt,
        trial_spec=trial_spec,
        comparison_candidate=comparison,
        baseline=baseline_values,
        simulate=True,
    )
    completion = stage_authoritative_completion(attack_root, attempt, trial_spec, inventory)
    record_completed_evidence_command(attack_root, ledger, attempt, completion)
    attacked = deepcopy(completion)
    readiness_index = next(
        index for index, item in enumerate(attacked["entries"]) if item["kind"] == "full_s3_readiness"
    )
    if attack == "missing":
        attacked["entries"].pop(readiness_index)
    elif attack == "extra":
        extra = deepcopy(attacked["entries"][readiness_index])
        extra["evidence_id"] = "evidence-extra-readiness-m1153"
        extra["kind"] = "unexpected_readiness"
        attacked["entries"].append(extra)
    elif attack == "duplicate":
        attacked["entries"].append(deepcopy(attacked["entries"][readiness_index]))
    elif attack == "wrong_id":
        attacked["entries"][readiness_index]["evidence_id"] = "evidence-wrong-readiness-m1153"
    elif attack == "cross_attempt":
        attacked["entries"][readiness_index]["attempt_id"] = baseline_attempt["attempt_id"]
    else:
        attacked["lifecycle_generation"] += 1

    before_events = ledger.events()
    before_state = ledger.state()
    with pytest.raises((IntegrityError, ValueError, KeyError)):
        ledger.commit_proxy_evidence(attacked)
    assert ledger.events() == before_events
    assert ledger.state() == before_state
    assert not any(event["event_type"] == "ProxyEvidenceCommitted" for event in before_events)


def _blocked_readiness_inventory(root: Path, inventory: list[dict[str, str]]) -> list[dict[str, str]]:
    attacked = deepcopy(inventory)
    item = next(entry for entry in attacked if entry["kind"] == "full_s3_readiness")
    path = root / item["source_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ready"] = False
    payload["classification"] = "BLOCKED"
    payload["checks"][0]["status"] = "BLOCKED"
    comparator = payload["checks"][0]["comparator"]
    threshold = payload["checks"][0]["threshold"]
    if comparator == "eq":
        payload["checks"][0]["measurement"] = not threshold if isinstance(threshold, bool) else f"not-{threshold}"
    elif comparator == "exact_set":
        payload["checks"][0]["measurement"] = []
    else:
        payload["checks"][0]["measurement"] = float(threshold) - 1.0
    path.write_bytes(encode_canonical_evidence(payload))
    return attacked


def test_activation_pass_with_independent_readiness_block_repairs_without_full_and_replays(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline"
    baseline_attempt, baseline_trial, baseline_comparison, baseline_values = _c2c_inputs(baseline_root)
    baseline_ledger = ResearchEventLedger(baseline_root)
    baseline_inventory = _c2c_strict_evidence_inventory(
        project_root=baseline_root,
        attempt=baseline_attempt,
        trial_spec=baseline_trial,
        comparison_candidate=baseline_comparison,
        baseline=baseline_values,
        simulate=True,
    )
    baseline_completion = stage_authoritative_completion(
        baseline_root, baseline_attempt, baseline_trial, baseline_inventory
    )
    record_completed_evidence_command(
        baseline_root, baseline_ledger, baseline_attempt, baseline_completion
    )
    _, baseline_route = baseline_ledger.commit_proxy_evidence(baseline_completion)
    assert baseline_route["next_action"] == "RUN_FULL"

    root = tmp_path / "blocked"
    attempt, trial_spec, comparison, baseline_values = _c2c_inputs(root)
    ledger = ResearchEventLedger(root)
    inventory = _c2c_strict_evidence_inventory(
        project_root=root,
        attempt=attempt,
        trial_spec=trial_spec,
        comparison_candidate=comparison,
        baseline=baseline_values,
        simulate=True,
    )
    blocked_inventory = _blocked_readiness_inventory(root, inventory)
    completion = stage_authoritative_completion(root, attempt, trial_spec, blocked_inventory)
    record_completed_evidence_command(root, ledger, attempt, completion)

    repaired, route = ledger.commit_proxy_evidence(completion)
    assert repaired["state"] == "IMPLEMENTATION_REPAIR"
    assert route["next_action"] == "REPAIR_IMPLEMENTATION"
    assert "FullPhaseStarted" not in [event["event_type"] for event in ledger.events()]
    assert ledger.state()["directions"][attempt["direction_semantic_hash"]]["budget"] == {
        "target": 5,
        "reserved": 1,
        "consumed": 0,
    }

    before = ledger.events()
    replayed_attempt, replayed_route = ledger.commit_proxy_evidence(completion)
    assert ledger.events() == before
    assert replayed_attempt == repaired
    assert replayed_route == route
