from __future__ import annotations

from copy import deepcopy

import pytest

from auto_research.failure_validation import (
    FAILURE_EVIDENCE_SCHEMA_VERSION,
    RESOURCE_PROBE_SCHEMA_VERSION,
    RESUME_EVIDENCE_SCHEMA_VERSION,
    canonical_evidence_bytes,
    evidence_bytes_hash,
    validate_failure_evidence,
    validate_resume_evidence,
)


def _receipt_ref(digest: str = "e" * 64) -> dict:
    return {
        "schema_version": "auto_research_contract_blob_v1",
        "algorithm": "sha256",
        "digest": digest,
        "size_bytes": 128,
        "relative_path": f"meta/contracts/sha256/{digest[:2]}/{digest}.json",
    }


def _command_binding() -> dict:
    return {
        "command_id": "command-proxy-0001",
        "command_hash": "d" * 64,
        "command_plan_hash": "e" * 64,
        "receipt_ref": _receipt_ref(),
        "receipt_hash": "f" * 64,
    }


def _identity(*, phase: str = "proxy", generation: int = 0, producer: str = "producer-proxy-0001") -> dict:
    return {
        "attempt_id": "attempt-m115-1",
        "direction_semantic_hash": "1" * 64,
        "direction_spec_hash": "2" * 64,
        "variant_semantic_hash": "3" * 64,
        "variant_spec_hash": "4" * 64,
        "trial_spec_hash": "5" * 64,
        "protocol_hash": "6" * 64,
        "sample_manifest_hash": "7" * 64,
        "evaluator_hash": "8" * 64,
        "lifecycle_generation": generation,
        "implementation_hash": "9" * 64,
        "attempt_input_hash": "a" * 64,
        "phase": phase,
        "phase_execution_id": f"phase-{phase}-0001",
        "phase_start_event_id": f"event:phase:{phase}:1",
        "producer_run_id": producer,
    }


def _command_receipt(identity: dict, *, exit_code: int = 2) -> dict:
    stdout_ref = _receipt_ref("b" * 64)
    stderr_ref = _receipt_ref("c" * 64)
    return {
        "schema_version": "auto_research_phase_run_receipt_v4",
        "command_id": "command-proxy-0001",
        "command_hash": "d" * 64,
        "command_spec_id": "command-spec-proxy-0001",
        "command_plan_hash": "e" * 64,
        "started_event_id": "event:command:started:1",
        "started_event_hash": "f" * 64,
        "attempt_id": identity["attempt_id"],
        "lifecycle_generation": identity["lifecycle_generation"],
        "phase": "proxy" if identity["phase"] == "activation" else identity["phase"],
        "phase_execution_id": identity["phase_execution_id"],
        "phase_start_event_id": identity["phase_start_event_id"],
        "producer_run_id": identity["producer_run_id"],
        "implementation_hash": identity["implementation_hash"],
        "attempt_input_hash": identity["attempt_input_hash"],
        "provenance_mode": "synthetic",
        "receipt_locator": f"meta/contracts/sha256/{'f' * 2}/{'f' * 64}.json",
        "started_at": "2026-07-15T00:00:00Z",
        "completed_at": "2026-07-15T00:00:01Z",
        "exit_code": exit_code,
        "stdout_hash": "b" * 64,
        "stderr_hash": "c" * 64,
        "stdout_ref": stdout_ref,
        "stderr_ref": stderr_ref,
        "external_job_id": None,
        "outputs": [],
    }


def _failure(identity: dict, *, failure_class: str, referenced_hash: str, exit_code: int = 2) -> dict:
    resource_failure = failure_class in {"resource_pause", "oom_retry"}
    return {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": "failure_evidence",
        "evidence_id": "evidence:failure:m115",
        **identity,
        "cross_references": {
            "resource_probe_hash" if resource_failure else "phase_run_receipt_hash": referenced_hash
        },
        "source_state": "PROXY_RUNNING",
        "source_phase": identity["phase"],
        "failure_class": failure_class,
        "command_status": "resource_paused" if resource_failure else "failed",
        "exit_code": exit_code,
        "reason": "authoritative failure",
        "observed_at": "2026-07-15T00:00:01Z",
        "log_hash": "d" * 64 if resource_failure else "c" * 64,
        **{
            **_command_binding(),
            "receipt_hash": referenced_hash if not resource_failure else _command_binding()["receipt_hash"],
        },
    }


def _probe(identity: dict, *, status: str, observed: float, required: float = 10.0) -> dict:
    return {
        "schema_version": RESOURCE_PROBE_SCHEMA_VERSION,
        "evidence_kind": "resource_probe",
        "evidence_id": "evidence:resource:m115",
        **identity,
        "resource_type": "gpu_memory",
        "resource_id": "gpu:0",
        "required_capacity": required,
        "observed_capacity": observed,
        "unit": "bytes",
        "probe_status": status,
        "observed_at": "2026-07-15T00:00:02Z",
        **_command_binding(),
    }


def _resume(resume_identity: dict, pause_identity: dict, probe: dict, pause_hash: str) -> dict:
    return {
        "schema_version": RESUME_EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": "resume_evidence",
        "evidence_id": "evidence:resume:m115",
        **resume_identity,
        "cross_references": {"resource_probe_hash": evidence_bytes_hash(canonical_evidence_bytes(probe))},
        "pause_event_id": "event:pause:m115:1",
        "pause_evidence_hash": pause_hash,
        "pause_phase": pause_identity["phase"],
        "pause_phase_execution_id": pause_identity["phase_execution_id"],
        "pause_producer_run_id": pause_identity["producer_run_id"],
        "resource_type": probe["resource_type"],
        "resource_id": probe["resource_id"],
        "required_capacity": probe["required_capacity"],
        "observed_capacity": probe["observed_capacity"],
        "unit": probe["unit"],
        "probe_status": probe["probe_status"],
        "observed_at": probe["observed_at"],
        **_command_binding(),
    }


def test_non_resource_failure_requires_exact_canonical_command_receipt() -> None:
    identity = _identity()
    receipt_raw = canonical_evidence_bytes(_command_receipt(identity))
    failure_raw = canonical_evidence_bytes(
        _failure(identity, failure_class="activation_failure", referenced_hash=evidence_bytes_hash(receipt_raw))
    )

    decoded = validate_failure_evidence(identity, failure_raw, phase_run_receipt_raw=receipt_raw)

    assert decoded["failure_evidence"]["failure_class"] == "activation_failure"
    assert decoded["phase_run_receipt"]["exit_code"] == 2
    with pytest.raises(ValueError, match="requires canonical PhaseRunReceipt"):
        validate_failure_evidence(identity, failure_raw)


def test_non_resource_failure_rejects_legacy_command_result_authority() -> None:
    identity = _identity()
    receipt_raw = canonical_evidence_bytes(_command_receipt(identity))
    failure = _failure(
        identity,
        failure_class="activation_failure",
        referenced_hash=evidence_bytes_hash(receipt_raw),
    )
    failure["cross_references"] = {
        "command_result_evidence_hash": evidence_bytes_hash(receipt_raw),
    }

    with pytest.raises(ValueError, match="schema violation"):
        validate_failure_evidence(
            identity,
            canonical_evidence_bytes(failure),
            phase_run_receipt_raw=receipt_raw,
        )


def test_non_resource_failure_rejects_raw_bytes_tamper_even_when_json_still_decodes() -> None:
    identity = _identity()
    receipt = _command_receipt(identity)
    receipt_raw = canonical_evidence_bytes(receipt)
    failure = _failure(identity, failure_class="implementation_failure", referenced_hash=evidence_bytes_hash(receipt_raw))
    failure_raw = canonical_evidence_bytes(failure)
    assert validate_failure_evidence(identity, failure_raw, phase_run_receipt_raw=receipt_raw)

    with pytest.raises(ValueError, match="not canonical JSON"):
        validate_failure_evidence(identity, failure_raw, phase_run_receipt_raw=receipt_raw + b"\n")

    tampered_receipt = deepcopy(receipt)
    tampered_receipt["command_hash"] = "0" * 64
    tampered_raw = canonical_evidence_bytes(tampered_receipt)
    with pytest.raises(ValueError, match="referenced hash"):
        validate_failure_evidence(identity, failure_raw, phase_run_receipt_raw=tampered_raw)

    forged_failure = deepcopy(failure)
    tampered_receipt["exit_code"] = 9
    tampered_raw = canonical_evidence_bytes(tampered_receipt)
    forged_failure["cross_references"]["phase_run_receipt_hash"] = evidence_bytes_hash(tampered_raw)
    forged_failure["receipt_hash"] = evidence_bytes_hash(tampered_raw)
    with pytest.raises(ValueError, match="exit_code"):
        validate_failure_evidence(identity, canonical_evidence_bytes(forged_failure), phase_run_receipt_raw=tampered_raw)


@pytest.mark.parametrize(
    ("target", "field"),
    [
        ("failure", "attempt_id"), ("failure", "phase"),
        ("failure", "phase_execution_id"), ("failure", "producer_run_id"),
        ("receipt", "attempt_id"), ("receipt", "phase_execution_id"),
        ("receipt", "producer_run_id"),
    ],
)
def test_non_resource_failure_rejects_identity_phase_and_producer_drift(target: str, field: str) -> None:
    identity = _identity()
    receipt = _command_receipt(identity)
    receipt_raw = canonical_evidence_bytes(receipt)
    failure = _failure(identity, failure_class="activation_failure", referenced_hash=evidence_bytes_hash(receipt_raw))
    attacked = failure if target == "failure" else receipt
    attacked[field] = {
        "attempt_id": "attempt-forged",
        "phase": "full",
        "phase_execution_id": "phase-proxy-9999",
        "producer_run_id": "producer-forged-0001",
    }[field]
    receipt_raw = canonical_evidence_bytes(receipt)
    failure["cross_references"]["phase_run_receipt_hash"] = evidence_bytes_hash(receipt_raw)
    failure["receipt_hash"] = evidence_bytes_hash(receipt_raw)

    with pytest.raises(ValueError, match=field):
        validate_failure_evidence(identity, canonical_evidence_bytes(failure), phase_run_receipt_raw=receipt_raw)


def test_resource_pause_requires_insufficient_probe_and_strict_capacity_gap() -> None:
    identity = _identity()
    probe = _probe(identity, status="insufficient", observed=9.0)
    probe_raw = canonical_evidence_bytes(probe)
    failure_raw = canonical_evidence_bytes(
        _failure(identity, failure_class="resource_pause", referenced_hash=evidence_bytes_hash(probe_raw), exit_code=137)
    )

    assert validate_failure_evidence(identity, failure_raw, resource_probe_raw=probe_raw)

    for status, observed in (("available", 9.0), ("insufficient", 10.0), ("insufficient", 11.0)):
        attacked = _probe(identity, status=status, observed=observed)
        attacked_raw = canonical_evidence_bytes(attacked)
        failure = _failure(
            identity,
            failure_class="resource_pause",
            referenced_hash=evidence_bytes_hash(attacked_raw),
            exit_code=137,
        )
        with pytest.raises(ValueError, match="insufficient|observed_capacity"):
            validate_failure_evidence(identity, canonical_evidence_bytes(failure), resource_probe_raw=attacked_raw)


def test_resource_pause_rejects_probe_producer_and_raw_bytes_tamper() -> None:
    identity = _identity()
    probe = _probe(identity, status="insufficient", observed=1.0)
    probe_raw = canonical_evidence_bytes(probe)
    failure = _failure(
        identity,
        failure_class="oom_retry",
        referenced_hash=evidence_bytes_hash(probe_raw),
        exit_code=137,
    )
    failure_raw = canonical_evidence_bytes(failure)

    forged_probe = deepcopy(probe)
    forged_probe["producer_run_id"] = "producer-forged-0001"
    forged_raw = canonical_evidence_bytes(forged_probe)
    failure["cross_references"]["resource_probe_hash"] = evidence_bytes_hash(forged_raw)
    with pytest.raises(ValueError, match="producer_run_id"):
        validate_failure_evidence(identity, canonical_evidence_bytes(failure), resource_probe_raw=forged_raw)

    with pytest.raises(ValueError, match="not canonical JSON"):
        validate_failure_evidence(identity, failure_raw, resource_probe_raw=b" " + probe_raw)


def test_resume_requires_available_probe_and_allows_exact_capacity_boundary() -> None:
    pause_identity = _identity()
    pause_probe_raw = canonical_evidence_bytes(_probe(pause_identity, status="insufficient", observed=4.0))
    pause = _failure(
        pause_identity,
        failure_class="resource_pause",
        referenced_hash=evidence_bytes_hash(pause_probe_raw),
        exit_code=137,
    )
    pause_raw = canonical_evidence_bytes(pause)
    resume_identity = _identity(phase="resume", generation=1, producer="producer-resume-0001")
    probe = _probe(resume_identity, status="available", observed=10.0)
    probe_raw = canonical_evidence_bytes(probe)
    resume = _resume(resume_identity, pause_identity, probe, evidence_bytes_hash(pause_raw))

    decoded = validate_resume_evidence(
        resume_identity,
        canonical_evidence_bytes(resume),
        resource_probe_raw=probe_raw,
        expected_pause_identity=pause_identity,
        pause_failure_raw=pause_raw,
        pause_resource_probe_raw=pause_probe_raw,
    )

    assert decoded["resource_probe"]["observed_capacity"] == decoded["resource_probe"]["required_capacity"]
    assert decoded["pause_failure_evidence"]["producer_run_id"] == pause_identity["producer_run_id"]


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("status", "available"),
        ("capacity", "observed_capacity"),
        ("phase", "phase"),
        ("producer", "producer_run_id"),
        ("pause_producer", "pause_producer_run_id"),
        ("resource_id", "resource_id"),
    ],
)
def test_resume_rejects_resource_identity_phase_and_producer_attacks(attack: str, message: str) -> None:
    pause_identity = _identity()
    pause_probe_raw = canonical_evidence_bytes(_probe(pause_identity, status="insufficient", observed=4.0))
    pause_raw = canonical_evidence_bytes(
        _failure(
            pause_identity,
            failure_class="resource_pause",
            referenced_hash=evidence_bytes_hash(pause_probe_raw),
            exit_code=137,
        )
    )
    resume_identity = _identity(phase="resume", generation=1, producer="producer-resume-0001")
    probe = _probe(resume_identity, status="available", observed=12.0)
    resume = _resume(resume_identity, pause_identity, probe, evidence_bytes_hash(pause_raw))

    if attack == "status":
        probe["probe_status"] = "insufficient"
        resume["probe_status"] = "insufficient"
    elif attack == "capacity":
        probe["observed_capacity"] = 9.0
        resume["observed_capacity"] = 9.0
    elif attack == "phase":
        probe["phase"] = "proxy"
    elif attack == "producer":
        probe["producer_run_id"] = "producer-forged-0001"
    elif attack == "pause_producer":
        resume["pause_producer_run_id"] = "producer-forged-0001"
    else:
        resume["resource_id"] = "gpu:1"
    probe_raw = canonical_evidence_bytes(probe)
    resume["cross_references"]["resource_probe_hash"] = evidence_bytes_hash(probe_raw)

    with pytest.raises(ValueError, match=message):
        validate_resume_evidence(
            resume_identity,
            canonical_evidence_bytes(resume),
            resource_probe_raw=probe_raw,
            expected_pause_identity=pause_identity,
            pause_failure_raw=pause_raw,
            pause_resource_probe_raw=pause_probe_raw,
        )


def test_resume_rejects_raw_probe_and_pause_evidence_tamper() -> None:
    pause_identity = _identity()
    pause_probe_raw = canonical_evidence_bytes(_probe(pause_identity, status="insufficient", observed=4.0))
    pause_raw = canonical_evidence_bytes(
        _failure(
            pause_identity,
            failure_class="resource_pause",
            referenced_hash=evidence_bytes_hash(pause_probe_raw),
            exit_code=137,
        )
    )
    resume_identity = _identity(phase="resume", generation=1, producer="producer-resume-0001")
    probe = _probe(resume_identity, status="available", observed=12.0)
    probe_raw = canonical_evidence_bytes(probe)
    resume_raw = canonical_evidence_bytes(_resume(resume_identity, pause_identity, probe, evidence_bytes_hash(pause_raw)))

    with pytest.raises(ValueError, match="not canonical JSON"):
        validate_resume_evidence(resume_identity, resume_raw, resource_probe_raw=probe_raw + b"\n")
    with pytest.raises(ValueError, match="referenced hash"):
        validate_resume_evidence(
            resume_identity,
            resume_raw,
            resource_probe_raw=probe_raw,
            expected_pause_identity=pause_identity,
            pause_failure_raw=pause_raw.replace(b"attempt-m115-1", b"attempt-forged"),
            pause_resource_probe_raw=pause_probe_raw,
        )
