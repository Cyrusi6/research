from __future__ import annotations

import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from auto_research.command_journal import (
    CommandExecutionResult,
    CommandJournalResult,
    LedgerCommandJournal,
)
from auto_research.contract_store import ContractStore, canonical_contract_bytes
from auto_research.derivation_validation import (
    derive_evidence_deterministically,
    validated_physical_receipt_inputs,
)
from auto_research.domain_contracts import TRIAL_SPEC_SCHEMA_VERSION, canonical_hash, validate_trial_spec
from auto_research.evidence import (
    EVIDENCE_SCHEMA_VERSIONS,
    EvidenceStore,
    content_addressed_evidence_path,
    encode_canonical_evidence,
    stage_completion_evidence,
)
from auto_research.evidence_lineage import (
    manifest_from_completion_evidence,
    validate_receipt_bound_evidence,
)
from auto_research.phase_execution import ResearchLedgerPhaseAuthority
from auto_research.phase_command_plan import build_phase_command_plan, store_phase_command_plan
from auto_research.proxy_classifier import build_proxy_decision_policy
from auto_research.research_state import FAILURE_EVIDENCE_SCHEMA_VERSION, ResearchEventLedger


_DEFAULT_CONTRACT_ROOT = Path(tempfile.mkdtemp(prefix="auto-research-test-contracts-"))


def _record_failed_phase_command(
    project_root: Path,
    attempt: dict,
    *,
    suffix: str,
    exit_code: int,
    stderr: str,
) -> dict:
    running_phase = next(phase for phase in ("proxy", "full") if attempt["phases"].get(phase) == "RUNNING")
    ledger = ResearchEventLedger(project_root)
    context = ResearchLedgerPhaseAuthority(ledger).context_for_attempt(project_root, attempt["attempt_id"], running_phase)
    plan = ContractStore(project_root).read_json(
        context.command_plan_hash,
        schema_file="phase_command_plan_v4.schema.json",
    )
    used = {
        record["command"]["command_spec_id"]
        for record in ledger.state()["phase_commands"].values()
        if record["command"]["attempt_id"] == attempt["attempt_id"]
        and record["command"]["lifecycle_generation"] == attempt["lifecycle_generation"]
        and record["command"]["phase_execution_id"] == context.phase_execution_id
    }
    command_spec = next(
        command
        for command in plan["commands"]
        if command["authority_role"] == "physical" and command["command_spec_id"] not in used
    )
    stdout = ""
    result = LedgerCommandJournal(project_root, ledger).run_once(
        context,
        command_id=f"failure-{running_phase}-{attempt['lifecycle_generation']}-{suffix}",
        command_spec_id=command_spec["command_spec_id"],
        argv=tuple(command_spec["argv"]),
        cwd=command_spec["cwd"],
        source_snapshot_hash=command_spec["source_snapshot_hash"],
        expected_outputs=tuple(output["kind"] for output in command_spec["expected_outputs"]),
        environment=command_spec["environment"],
        inherited_environment=command_spec["inherited_environment"],
        retry_policy=command_spec["retry_policy"],
        resource_policy=command_spec["resource_policy"],
        resume_policy=command_spec["resume_policy"],
        runner=lambda: CommandExecutionResult(
            exit_code=exit_code,
            stdout_hash=hashlib.sha256(stdout.encode()).hexdigest(),
            stderr_hash=hashlib.sha256(stderr.encode()).hexdigest(),
            outputs=(),
            stdout=stdout,
            stderr=stderr,
        ),
    )
    return {
        "command_id": result.receipt["command_id"],
        "command_hash": result.receipt["command_hash"],
        "command_plan_hash": result.receipt["command_plan_hash"],
        "receipt_ref": dict(result.receipt_ref),
        "receipt_hash": result.receipt_ref["digest"],
    }


def build_trial_spec_v9(
    spec: dict,
    *,
    project_root: Path | None = None,
    adapter_id: str = "fixture-phase-adapter",
    command_provenance_mode: str = "synthetic",
) -> dict:
    """Build the current strict TrialSpec from test fixture facts.

    This helper is deliberately not a runtime compatibility reader.  It
    materializes immutable synthetic/fixture sample and evaluator bytes, then
    freezes the same PhaseCommandPlan/derivation contracts used in production.
    """

    result = deepcopy(spec)
    root = Path(project_root) if project_root is not None else _DEFAULT_CONTRACT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    store = ContractStore(root)
    result["schema_version"] = TRIAL_SPEC_SCHEMA_VERSION
    legacy_manifest = result["sample_manifest"]
    sample_datasets = []
    dataset_specs = []
    for dataset in legacy_manifest["datasets"]:
        dataset_id = dataset["dataset_id"]
        declared_sample_ids = list(dataset["ordered_sample_ids"])
        supplied_refs = dataset.get("raw_sample_refs")
        if supplied_refs:
            raw_sample_refs = [dict(item) for item in supplied_refs]
            for reference in raw_sample_refs:
                store.verify(reference)
        else:
            raw_sample_refs = [
                store.put_bytes(
                    canonical_contract_bytes(
                        {
                            "fixture_sample": sample_id,
                            "ordinal": ordinal,
                            "dataset_id": dataset_id,
                        }
                    )
                )
                for ordinal, sample_id in enumerate(declared_sample_ids)
            ]
        ordered_sample_ids = [item["digest"] for item in raw_sample_refs]
        content_digest = store.digest_referenced_bytes(raw_sample_refs)
        sample_datasets.append(
            {
                "dataset_id": dataset_id,
                "source_revision": dataset["source_revision"],
                "split": dataset["split"],
                "sample_count": len(ordered_sample_ids),
                "ordered_sample_ids": ordered_sample_ids,
                "raw_sample_refs": raw_sample_refs,
                "content_digest": content_digest,
                "record_format": "jsonl-record-bytes-v1",
                "canonicalization_contract": "preserve-selected-record-bytes-v1",
            }
        )
        dataset_specs.append(
            {
                "dataset_id": dataset_id,
                "split": dataset["split"],
                "sample_count": len(ordered_sample_ids),
                "sample_hash": content_digest,
            }
        )
    manifest_id = str(legacy_manifest.get("manifest_id") or "test-sample-manifest")
    if len(manifest_id) < 8:
        manifest_id = f"fixture-{manifest_id}"
    sample_manifest = {
        "schema_version": "auto_research_sample_manifest_v4",
        "manifest_id": manifest_id,
        "provenance_mode": legacy_manifest.get("provenance_mode", "synthetic"),
        "datasets": sample_datasets,
    }
    result["sample_manifest"] = sample_manifest
    result["sample_manifest_ref"] = store.put_contract(
        sample_manifest,
        contract_kind="sample_manifest",
        schema_file="sample_manifest_v4.schema.json",
    )
    result["datasets"] = dataset_specs

    legacy_evaluator = result["execution_contract"]["evaluator_provenance"]
    source_blob = store.put_bytes(
        canonical_contract_bytes(
            {
                "source": legacy_evaluator.get("source_digest"),
                "evaluator_id": legacy_evaluator.get("evaluator_id"),
            }
        )
    )
    dependency_blob = store.put_bytes(
        canonical_contract_bytes({"dependencies": legacy_evaluator.get("dependency_digest")})
    )
    config_blob = store.put_bytes(
        canonical_contract_bytes(
            {"config": legacy_evaluator.get("config_hash") or legacy_evaluator.get("config_digest")}
        )
    )
    evaluator_id = str(legacy_evaluator.get("evaluator_id") or "test-evaluator")
    if len(evaluator_id) < 8:
        evaluator_id = f"fixture-{evaluator_id}"
    evaluator_manifest = {
        "schema_version": "auto_research_evaluator_manifest_v2",
        "evaluator_id": evaluator_id,
        "provenance_mode": legacy_evaluator.get("provenance_mode", "synthetic"),
        "source_blobs": [source_blob],
        "dependency_blobs": [dependency_blob],
        "config_blob": config_blob,
        "config_digest": store.digest_referenced_bytes([config_blob]),
        "source_digest": store.digest_referenced_bytes([source_blob]),
        "dependency_digest": store.digest_referenced_bytes([dependency_blob]),
    }
    evaluator_ref = store.put_contract(
        evaluator_manifest,
        contract_kind="evaluator_manifest",
        schema_file="evaluator_manifest_v2.schema.json",
    )
    result["execution_contract"]["evaluator_provenance"] = evaluator_manifest
    result["execution_contract"]["evaluator_manifest_ref"] = evaluator_ref
    result["execution_contract"]["evaluator_hash"] = canonical_hash(evaluator_manifest)

    requirements = []
    for requirement in result.get("evidence_requirements") or []:
        if requirement["kind"] in {"effective_proxy_policy", "proxy_calibration_policy", "proxy_decision_report"}:
            continue
        if set(result["protocol"]["required_phases"]) == {"proxy"} and requirement["kind"] == "main_results":
            requirement["kind"] = "proxy_results"
            requirement["requirement_id"] = "proxy-results"
        requirement["schema_version"] = EVIDENCE_SCHEMA_VERSIONS[requirement["kind"]]
        requirements.append(requirement)
    result["evidence_requirements"] = requirements

    required_phases = list(result["protocol"]["required_phases"])
    datasets = [item["dataset_id"] for item in result["datasets"]]
    seeds = list(result["statistical_testing"]["seeds"])
    metrics = [item["metric_id"] for item in result["metrics"]]
    if "proxy" in required_phases:
        mandatory = {
            "proxy_results": ("proxy-results", "auto_research_proxy_results_v1"),
            "activation_evidence": ("activation", "auto_research_activation_evidence_v4"),
            "proxy_baseline_fingerprint": ("proxy-baseline", "auto_research_proxy_baseline_fingerprint_v3"),
            "proxy_cache_report": ("proxy-cache", "auto_research_proxy_cache_report_v3"),
            "bootstrap_completion" if result["protocol"]["proxy_terminal_allowed"] else "full_s3_readiness": (
                "bootstrap" if result["protocol"]["proxy_terminal_allowed"] else "readiness",
                "auto_research_bootstrap_completion_v3"
                if result["protocol"]["proxy_terminal_allowed"]
                else "auto_research_full_s3_readiness_v4",
            ),
        }
        present = {item["kind"] for item in requirements}
        for kind, (requirement_id, schema_version) in mandatory.items():
            if kind not in present:
                requirements.append(
                    {
                        "requirement_id": requirement_id,
                        "kind": kind,
                        "required": True,
                        "applicable_phases": ["proxy"],
                        "schema_version": schema_version,
                    }
                )
    result["required_artifacts"] = list(dict.fromkeys(item["kind"] for item in requirements))
    result["execution_contract"].pop("command_contract_hash", None)
    _rebuild_phase_authority(
        result,
        project_root=root,
        adapter_id=adapter_id,
        adapter_version="1",
        provenance_mode=command_provenance_mode,
        variant_spec_hash=canonical_hash({"fixture": "variant"}),
        source_snapshot_hash=evaluator_manifest["source_digest"],
    )
    validate_trial_spec(result)
    return result


def refresh_phase_command_plans(spec: dict, *, project_root: Path | None = None) -> dict:
    root = Path(project_root) if project_root is not None else _DEFAULT_CONTRACT_ROOT
    existing_plan = next(
        (item.get("command_plan") for item in spec.get("phase_contracts") or [] if item.get("command_plan")),
        {},
    )
    adapter = existing_plan.get("adapter_identity") or {}
    _rebuild_phase_authority(
        spec,
        project_root=root,
        adapter_id=str(adapter.get("adapter_id") or "fixture-phase-adapter"),
        adapter_version=str(adapter.get("adapter_version") or "1"),
        provenance_mode=str(adapter.get("provenance_mode") or "synthetic"),
        variant_spec_hash=str(existing_plan.get("variant_spec_hash") or canonical_hash({"fixture": "variant"})),
        source_snapshot_hash=str(spec["execution_contract"]["evaluator_provenance"]["source_digest"]),
    )
    return spec


def _rebuild_phase_authority(
    spec: dict,
    *,
    project_root: Path,
    adapter_id: str,
    adapter_version: str,
    provenance_mode: str,
    variant_spec_hash: str,
    source_snapshot_hash: str,
) -> None:
    existing = {item["phase"]: item for item in spec.get("phase_contracts") or []}
    datasets = [item["dataset_id"] for item in spec["datasets"]]
    seeds = list(spec["statistical_testing"]["seeds"])
    metrics = [spec["primary_metric_id"]]
    terminal_phases = set(spec["protocol"]["terminal_phases"])
    activation_surfaces = list(
        ((spec.get("proxy_decision_policy") or {}).get("activation_surface_ids") or ["src/model.py"])
    )
    activation_threshold = float(
        (spec.get("proxy_decision_policy") or {}).get("activation_delta_threshold", 0.0)
    )
    phase_contracts = []
    for phase in spec["protocol"]["required_phases"]:
        requirements = [
            item
            for item in spec["evidence_requirements"]
            if phase in item["applicable_phases"] or "always" in item["applicable_phases"]
        ]
        evidence_kinds = [item["kind"] for item in requirements]
        previous = existing.get(phase) or {}
        roles = list(
            previous.get("roles")
            or (["baseline", "candidate"] if phase == "proxy" else spec["required_roles"])
        )
        raw_outputs = [
            {
                "output_id": f"raw-{item['kind'].replace('_', '-')}",
                "kind": item["kind"],
                "schema_version": item["schema_version"],
                "locator": f"fixture/{phase}/{item['kind']}.json",
                "locator_type": "file",
                "dataset_id": None,
                "role": None,
                "required": True,
                "normalized_kinds": [item["kind"]],
            }
            for item in requirements
        ]
        readiness_checks: list[dict[str, Any]] = []
        if "activation_evidence" in evidence_kinds:
            readiness_checks.append(
                {
                    "check_id": "activation-mechanism",
                    "check_kind": "activation_delta",
                    "predicate": {
                        "field_path": "surface_measurements.delta",
                        "comparator": "delta_gte",
                        "threshold": activation_threshold,
                    },
                    "required_coverage": {
                        "mode": "exact",
                        "expected_surface_ids": activation_surfaces,
                    },
                }
            )
        if "full_s3_readiness" in evidence_kinds:
            readiness_checks.append(
                {
                    "check_id": "proxy-ready-for-full",
                    "check_kind": "raw_measurement",
                    "predicate": {
                        "field_path": "ready",
                        "comparator": "eq",
                        "threshold": True,
                    },
                    "required_coverage": {"mode": "exact", "expected_surface_ids": []},
                }
            )
        plan = build_phase_command_plan(
            phase=phase,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            provenance_mode=provenance_mode,
            variant_spec_hash=variant_spec_hash,
            source_snapshot_hash=source_snapshot_hash,
            command_values=[
                {
                    "command_spec_id": "full-train" if phase == "full" else "proxy-fixture-physical",
                    "authority_role": "physical",
                    "argv": ["auto-research-adapter", "fixture-evidence", phase],
                    "cwd": str(project_root),
                    "physical_raw_outputs": raw_outputs,
                }
            ],
            expected_evidence=requirements,
            default_cwd=str(project_root),
            project_root=project_root,
            coverage_contract={
                "mode": "exact_cartesian",
                "datasets": list(previous.get("datasets") or datasets),
                "seeds": list(previous.get("seeds") or seeds),
                "metrics": list(previous.get("metrics") or metrics),
                "roles": roles,
            },
            readiness_checks=readiness_checks,
            decoder_id="canonical-identity",
            decoder_version="1",
        )
        plan_ref, plan_hash = store_phase_command_plan(project_root, plan)
        terminal = phase in terminal_phases
        bootstrap_terminal = (
            terminal
            and phase == "proxy"
            and "bootstrap" in str(spec["protocol"]["protocol_id"]).lower()
        )
        phase_contracts.append(
            {
                "phase": phase,
                "datasets": list(previous.get("datasets") or datasets),
                "seeds": list(previous.get("seeds") or seeds),
                "roles": roles,
                "metrics": list(previous.get("metrics") or metrics),
                "evidence_kinds": evidence_kinds,
                "terminal": terminal,
                "consumes_direction_budget": bool(
                    previous.get("consumes_direction_budget", terminal and not bootstrap_terminal)
                ),
                "command_plan": plan,
                "command_plan_ref": plan_ref,
                "command_plan_hash": plan_hash,
                "derivation_plan": plan["derivation_plan"],
                "derivation_plan_ref": plan["derivation_plan_ref"],
                "derivation_plan_hash": plan["derivation_plan_hash"],
                "readiness_check_plan": plan["readiness_check_plan"],
                "readiness_check_plan_ref": plan["readiness_check_plan_ref"],
                "readiness_check_plan_hash": plan["readiness_check_plan_hash"],
            }
        )
    spec["phase_contracts"] = phase_contracts
    spec["execution_contract"]["phase_command_plan_hashes"] = {
        item["phase"]: item["command_plan_hash"] for item in phase_contracts
    }
    if "proxy" not in set(spec["protocol"]["required_phases"]):
        spec["proxy_decision_policy"] = None
        return
    proxy_contract = next(item for item in phase_contracts if item["phase"] == "proxy")
    primary_constraint = next(
        item
        for item in spec["acceptance_constraints"]
        if item["kind"] == "minimum_mean_delta" and item.get("metric_id") == spec["primary_metric_id"]
    )
    regression_constraint = next(
        (item for item in spec["acceptance_constraints"] if item["kind"] == "per_dataset_maximum_regression"),
        None,
    )
    readiness_plan = proxy_contract["readiness_check_plan"]
    if not isinstance(readiness_plan, Mapping):
        raise ValueError("proxy fixture requires a frozen ReadinessCheckPlan")
    readiness_ids = [
        item["check_id"] for item in readiness_plan["checks"] if item["check_kind"] == "raw_measurement"
    ]
    spec["proxy_decision_policy"] = build_proxy_decision_policy(
        primary_metric_id=spec["primary_metric_id"],
        objective=next(
            item["objective"] for item in spec["metrics"] if item["metric_id"] == spec["primary_metric_id"]
        ),
        aggregation="paired_mean",
        datasets=proxy_contract["datasets"],
        seeds=proxy_contract["seeds"],
        metric_ids=proxy_contract["metrics"],
        roles=proxy_contract["roles"],
        aggregate_improvement_threshold=float(primary_constraint["threshold"]),
        per_dataset_maximum_regression=float(
            regression_constraint["threshold"] if regression_constraint else 0.0
        ),
        activation_delta_threshold=activation_threshold,
        activation_surface_ids=activation_surfaces,
        readiness_check_ids=readiness_ids,
        readiness_check_plan_ref=proxy_contract["readiness_check_plan_ref"],
        readiness_check_plan_hash=proxy_contract["readiness_check_plan_hash"],
        evidence_kinds=proxy_contract["evidence_kinds"],
        mode=(
            "terminal_bootstrap"
            if spec["protocol"]["proxy_terminal_allowed"]
            else "gate_to_full"
        ),
    )


def start_attempt_phase(ledger, attempt: dict, phase: str) -> dict:
    phase_execution_id = f"{phase}-{attempt['attempt_id'][:12]}-g{attempt['lifecycle_generation']}"
    producer_run_id = f"producer-{phase}-{attempt['attempt_id'][:8]}-g{attempt['lifecycle_generation']}"
    if phase == "proxy":
        return ledger.start_proxy_phase(attempt["attempt_id"], phase_execution_id=phase_execution_id, producer_run_id=producer_run_id)
    return ledger.start_full_phase(attempt["attempt_id"], phase_execution_id=phase_execution_id, producer_run_id=producer_run_id)


def stage_authoritative_completion(
    project_root: Path,
    attempt: dict,
    trial_spec: dict,
    inventory: list[dict],
) -> dict:
    """Stage production-shaped raw evidence without constructing TrialResult fields."""

    return stage_completion_evidence(
        project_root=project_root,
        attempt=attempt,
        trial_spec=trial_spec,
        inventory=inventory,
    )


def record_completed_evidence_command(
    project_root: Path,
    ledger,
    attempt: dict,
    completion: dict,
) -> dict:
    """Commit the physical receipt and its one constrained derivation receipt.

    CompletionEvidence remains a staging request.  Its bytes first become raw
    outputs of the frozen physical fixture command; the production pure decoder
    then recomputes normalized outputs and a structured derivation manifest.
    """

    phase = completion["entries"][0]["phase"]
    state = ledger.state()
    current_attempt = state["attempts"][attempt["attempt_id"]]
    context = ResearchLedgerPhaseAuthority(ledger).context_for_attempt(
        project_root,
        current_attempt["attempt_id"],
        phase,
    )
    phase_contract = next(
        item
        for item in current_attempt["frozen_trial_spec"]["phase_contracts"]
        if item["phase"] == phase
    )
    plan = phase_contract["command_plan"]
    if plan["derivation_plan"]["decoder_descriptor"]["decoder_id"] != "canonical-identity":
        raise ValueError("the common authoritative fixture requires the frozen canonical-identity decoder")
    entries_by_kind = {entry["kind"]: entry for entry in completion["entries"]}
    store = ContractStore(project_root)
    evidence_store = EvidenceStore(project_root)
    journal = LedgerCommandJournal(project_root, ledger)

    physical_commands = [
        item for item in plan["commands"] if item["authority_role"] == "physical"
    ]
    for command_spec in physical_commands:
        if command_spec["expected_outputs"]:
            raise ValueError("fixture physical commands must publish raw_outputs, not normalized outputs")
        raw_outputs = []
        for raw_spec in command_spec["physical_raw_outputs"]:
            normalized_kinds = list(raw_spec["normalized_kinds"])
            if len(normalized_kinds) != 1:
                raise ValueError("canonical fixture raw output must map to exactly one normalized kind")
            kind = normalized_kinds[0]
            entry = entries_by_kind.get(kind)
            if entry is None:
                raise ValueError(f"CompletionEvidence lacks frozen physical output source: {kind}")
            raw = evidence_store.read_entry(entry, current_attempt)
            reference = store.put_bytes(raw)
            raw_outputs.append(
                {
                    "output_id": raw_spec["output_id"],
                    "kind": raw_spec["kind"],
                    "schema_version": raw_spec["schema_version"],
                    "contract_ref": reference,
                    "locator": raw_spec["locator"],
                    "locator_type": raw_spec["locator_type"],
                    "dataset_id": raw_spec["dataset_id"],
                    "role": raw_spec["role"],
                }
            )
        stdout = f"physical:{current_attempt['attempt_id']}:{phase}:{command_spec['command_spec_id']}"
        physical_result = journal.run_once(
            context,
            command_id=(
                f"fixture-physical-{phase}-{command_spec['ordinal']:02d}-"
                f"{current_attempt['attempt_id'][:12]}"
            ),
            command_spec_id=command_spec["command_spec_id"],
            argv=tuple(command_spec["argv"]),
            cwd=command_spec["cwd"],
            source_snapshot_hash=command_spec["source_snapshot_hash"],
            expected_outputs=(),
            environment=command_spec["environment"],
            inherited_environment=command_spec["inherited_environment"],
            retry_policy=command_spec["retry_policy"],
            resource_policy=command_spec["resource_policy"],
            resume_policy=command_spec["resume_policy"],
            runner=lambda raw_outputs=tuple(raw_outputs), stdout=stdout, command_spec=command_spec: CommandExecutionResult(
                exit_code=0,
                stdout_hash=hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                stderr_hash=hashlib.sha256(b"").hexdigest(),
                outputs=(),
                raw_outputs=raw_outputs,
                stdout=stdout,
                stderr="",
                external_job_id=(
                    f"fixture-physical-{phase}-{command_spec['ordinal']:02d}-"
                    f"{current_attempt['attempt_id'][:8]}"
                ),
            ),
        )
        if physical_result.get("status") != "completed":
            raise AssertionError(f"fixture physical command did not complete: {physical_result}")

    state = ledger.state()
    current_attempt = state["attempts"][attempt["attempt_id"]]
    derivation_plan = plan["derivation_plan"]
    derive_command = next(
        item for item in plan["commands"] if item["authority_role"] == "derivation"
    )

    def derive() -> CommandExecutionResult:
        derive_state = ledger.state()
        derive_attempt = derive_state["attempts"][attempt["attempt_id"]]
        physical_inputs = validated_physical_receipt_inputs(
            project_root=project_root,
            attempt=derive_attempt,
            phase_commands=derive_state["phase_commands"],
            phase=phase,
            derivation_plan=derivation_plan,
        )
        deterministic = derive_evidence_deterministically(
            attempt=derive_attempt,
            trial_spec=derive_attempt["frozen_trial_spec"],
            phase=phase,
            derivation_plan=derivation_plan,
            physical_inputs=physical_inputs,
            decoder_implementation_bytes=store.read_bytes(
                derivation_plan["decoder_descriptor"]["immutable_ref"]
            ),
        )
        for output in deterministic.normalized_outputs:
            staged_entry = entries_by_kind[output.kind]
            if evidence_store.read_entry(staged_entry, derive_attempt) != output.raw_bytes:
                raise ValueError(
                    f"fixture staged evidence differs from deterministic decoder output: {output.kind}"
                )
        outputs = []
        normalized_outputs = []
        for ordinal, output in enumerate(deterministic.normalized_outputs):
            reference = store.put_bytes(output.raw_bytes)
            outputs.append(
                {
                    "output_id": output.output_id,
                    "kind": output.kind,
                    "schema_version": output.schema_version,
                    "contract_ref": reference,
                }
            )
            normalized_outputs.append(
                {
                    "ordinal": ordinal,
                    "output_id": output.output_id,
                    "kind": output.kind,
                    "schema_version": output.schema_version,
                    "contract_ref": deepcopy(reference),
                    "content_hash": reference["digest"],
                }
            )
        derivation_manifest = {
            "schema_version": "auto_research_evidence_derivation_manifest_v3",
            **deepcopy(deterministic.manifest_facts),
            "derivation_plan_ref": deepcopy(plan["derivation_plan_ref"]),
            "derivation_plan_hash": plan["derivation_plan_hash"],
            "normalized_outputs": normalized_outputs,
        }
        derivation_ref = store.put_json(
            derivation_manifest,
            schema_file="evidence_derivation_manifest_v3.schema.json",
        )
        summary = json.dumps(
            {
                "derivation": "fixture-receipt-bound-core-v2",
                "phase_execution_id": context.phase_execution_id,
                "source_receipt_hashes": [item.receipt_hash for item in physical_inputs],
                "output_hashes": [item["contract_ref"]["digest"] for item in outputs],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return CommandExecutionResult(
            exit_code=0,
            stdout_hash=hashlib.sha256(summary.encode("utf-8")).hexdigest(),
            stderr_hash=hashlib.sha256(b"").hexdigest(),
            outputs=tuple(outputs),
            derivation_ref=derivation_ref,
            derivation_hash=derivation_ref["digest"],
            stdout=summary,
            stderr="",
            external_job_id=f"fixture-derive-{phase}-{derive_attempt['attempt_id'][:8]}",
        )

    result = journal.run_once(
        context,
        command_id=f"fixture-{phase}-{current_attempt['attempt_id'][:12]}",
        command_spec_id=derive_command["command_spec_id"],
        argv=tuple(derive_command["argv"]),
        cwd=derive_command["cwd"],
        source_snapshot_hash=derive_command["source_snapshot_hash"],
        expected_outputs=tuple(item["kind"] for item in derive_command["expected_outputs"]),
        environment=derive_command["environment"],
        inherited_environment=derive_command["inherited_environment"],
        retry_policy=derive_command["retry_policy"],
        resource_policy=derive_command["resource_policy"],
        resume_policy=derive_command["resume_policy"],
        runner=derive,
    )
    if not isinstance(result, CommandJournalResult) or result.get("status") != "completed":
        raise AssertionError(f"fixture derivation command did not complete: {result}")
    return dict(result)


def validate_authoritative_completion(
    project_root: Path,
    ledger,
    attempt: dict,
    completion: dict,
):
    """Return the canonical manifest derived from the committed command receipt."""

    state = ledger.state()
    current_attempt = state["attempts"][attempt["attempt_id"]]
    staged_manifest = manifest_from_completion_evidence(
        attempt=current_attempt,
        completion_evidence=completion,
    )
    return validate_receipt_bound_evidence(
        project_root=project_root,
        attempt=current_attempt,
        trial_spec=current_attempt["frozen_trial_spec"],
        manifest=staged_manifest,
        phase_commands=state["phase_commands"],
        phase=completion["entries"][0]["phase"],
    )


def build_quantitative_completion(
    project_root: Path,
    attempt: dict,
    *,
    role_values: dict[str, float],
    dataset_id: str,
    metric_id: str,
    seed: int,
    phase: str,
    producer_run_id: str | None = None,
    evidence_kind: str | None = None,
) -> dict:
    phase_execution = attempt["phase_executions"][phase]
    if not isinstance(phase_execution, dict):
        raise ValueError("Attempt phase must be started before building evidence")
    producer_run_id = producer_run_id or phase_execution["producer_run_id"]
    if producer_run_id != phase_execution["producer_run_id"]:
        raise ValueError("producer_run_id must match PhaseExecutionManifest")
    evidence_kind = evidence_kind or ("proxy_results" if phase == "proxy" else "main_results")
    phase_contract = next(
        item
        for item in attempt["frozen_trial_spec"]["phase_contracts"]
        if item["phase"] == phase
    )
    if dataset_id not in phase_contract["datasets"]:
        raise ValueError("dataset_id is not frozen in the phase contract")
    if metric_id not in phase_contract["metrics"]:
        raise ValueError("metric_id is not frozen in the phase contract")
    if seed not in phase_contract["seeds"]:
        raise ValueError("seed is not frozen in the phase contract")
    if evidence_kind not in phase_contract["evidence_kinds"]:
        raise ValueError("evidence_kind is not frozen in the phase contract")
    payloads = _build_phase_payloads(
        attempt,
        phase=phase,
        role_values=role_values,
        preferred_kind=evidence_kind,
    )
    return _stage_payloads(
        project_root,
        attempt,
        attempt["frozen_trial_spec"],
        payloads,
    )


def build_bootstrap_completion(
    project_root: Path,
    attempt: dict,
    trial_spec: dict,
    *,
    baseline_value: float = 0.5,
    candidate_value: float = 0.7,
) -> dict:
    """Build strict bootstrap evidence without caller-authored TrialResult facts."""

    payloads = _build_phase_payloads(
        attempt,
        phase="proxy",
        role_values={"baseline": baseline_value, "candidate": candidate_value},
        preferred_kind="proxy_results",
    )
    return _stage_payloads(project_root, attempt, trial_spec, payloads)


def _build_phase_payloads(
    attempt: Mapping[str, Any],
    *,
    phase: str,
    role_values: Mapping[str, float],
    preferred_kind: str,
) -> list[dict[str, Any]]:
    trial_spec = attempt["frozen_trial_spec"]
    phase_contract = next(
        item for item in trial_spec["phase_contracts"] if item["phase"] == phase
    )
    phase_execution = attempt["phase_executions"][phase]
    if not isinstance(phase_execution, Mapping):
        raise ValueError("Attempt phase must be started before building evidence")
    payloads: dict[str, dict[str, Any]] = {}
    quantitative_roles = {
        "main_results": ("baseline", "candidate"),
        "proxy_results": ("baseline", "candidate"),
        "ablation_results": ("ablation",),
        "coverage_results": ("coverage",),
        "matched_control_results": ("matched_control",),
    }
    baseline_value = float(role_values.get("baseline", 0.0))
    candidate_value = float(role_values.get("candidate", baseline_value))
    auxiliary_values = {
        "ablation": baseline_value,
        "coverage": 1.0,
        "matched_control": baseline_value,
    }
    for kind in phase_contract["evidence_kinds"]:
        if kind not in quantitative_roles:
            continue
        rows = []
        for dataset_id in phase_contract["datasets"]:
            for seed in phase_contract["seeds"]:
                for metric_id in phase_contract["metrics"]:
                    for role in quantitative_roles[kind]:
                        value = float(role_values.get(role, auxiliary_values.get(role, candidate_value)))
                        rows.append(
                            _measurement_row(
                                attempt,
                                phase=phase,
                                phase_execution=phase_execution,
                                role=role,
                                dataset_id=dataset_id,
                                metric_id=metric_id,
                                seed=seed,
                                value=value,
                            )
                        )
        payloads[kind] = {
            **_evidence_identity(attempt, phase, phase_execution, kind),
            "rows": rows,
        }

    policy = trial_spec.get("proxy_decision_policy") or {}
    if "activation_evidence" in phase_contract["evidence_kinds"]:
        threshold = float(policy.get("activation_delta_threshold", 0.0))
        surfaces = list(policy.get("activation_surface_ids") or ["src/model.py"])
        delta = candidate_value - baseline_value
        activated = delta >= threshold
        payloads["activation_evidence"] = {
            **_evidence_identity(attempt, phase, phase_execution, "activation_evidence"),
            "probe_id": "activation-mechanism",
            "status": "activated" if activated else "not_activated",
            "command_status": "completed",
            "exit_code": 0,
            "expected_surface_ids": surfaces,
            "observed_surface_ids": surfaces,
            "activation_delta_threshold": threshold,
            "surface_measurements": [
                {
                    "surface_id": surface_id,
                    "enabled_value": candidate_value,
                    "disabled_value": baseline_value,
                    "delta": delta,
                    "threshold": threshold,
                    "status": "ACTIVATED" if activated else "NOT_ACTIVATED",
                }
                for surface_id in surfaces
            ],
        }

    if "proxy_baseline_fingerprint" in phase_contract["evidence_kinds"]:
        fingerprint_inputs = {
            "sample_manifest_hash": attempt["sample_manifest_hash"],
            "evaluator_hash": attempt["evaluator_hash"],
            "protocol_hash": attempt["protocol_hash"],
            "phase_execution_id": phase_execution["phase_execution_id"],
        }
        payloads["proxy_baseline_fingerprint"] = {
            **_evidence_identity(
                attempt, phase, phase_execution, "proxy_baseline_fingerprint"
            ),
            "baseline_hash": canonical_hash(fingerprint_inputs),
            "dataset_ids": list(phase_contract["datasets"]),
            "seeds": list(phase_contract["seeds"]),
            "fingerprint_inputs": fingerprint_inputs,
        }

    if "proxy_cache_report" in phase_contract["evidence_kinds"]:
        fingerprint = payloads["proxy_baseline_fingerprint"]
        fingerprint_hash = canonical_hash(fingerprint)
        baseline_hash = fingerprint["baseline_hash"]
        payloads["proxy_cache_report"] = {
            **_evidence_identity(attempt, phase, phase_execution, "proxy_cache_report"),
            "cross_references": {
                "proxy_baseline_fingerprint_hash": fingerprint_hash
            },
            "cache_key": canonical_hash(
                {
                    "attempt_input_hash": attempt["attempt_input_hash"],
                    "baseline_hash": baseline_hash,
                }
            ),
            "baseline_hash": baseline_hash,
            "cache_entry_hash": baseline_hash,
            "status": "created",
        }

    proxy_results = payloads.get("proxy_results")
    activation = payloads.get("activation_evidence")
    proxy_cross_references = {}
    if proxy_results is not None:
        proxy_cross_references["proxy_results_hash"] = canonical_hash(proxy_results)
    if activation is not None:
        proxy_cross_references["activation_evidence_hash"] = canonical_hash(activation)

    if "full_s3_readiness" in phase_contract["evidence_kinds"]:
        readiness_plan = phase_contract["readiness_check_plan"]
        checks = []
        for check in readiness_plan["checks"]:
            if check["check_kind"] != "raw_measurement":
                continue
            predicate = check["predicate"]
            measurement = _passing_measurement(
                predicate["comparator"], predicate["threshold"]
            )
            checks.append(
                {
                    "check_id": check["check_id"],
                    "status": "PASS",
                    "measurement": measurement,
                    "comparator": predicate["comparator"],
                    "threshold": predicate["threshold"],
                }
            )
        payloads["full_s3_readiness"] = {
            **_evidence_identity(attempt, phase, phase_execution, "full_s3_readiness"),
            "cross_references": proxy_cross_references,
            "readiness_check_plan_ref": phase_contract["readiness_check_plan_ref"],
            "readiness_check_plan_hash": phase_contract["readiness_check_plan_hash"],
            "ready": True,
            "classification": "PASS",
            "checks": checks,
        }

    if "bootstrap_completion" in phase_contract["evidence_kinds"]:
        payloads["bootstrap_completion"] = {
            **_evidence_identity(attempt, phase, phase_execution, "bootstrap_completion"),
            "cross_references": proxy_cross_references,
            "completion_status": "verified",
        }

    missing = set(phase_contract["evidence_kinds"]) - set(payloads)
    if missing:
        raise ValueError(f"fixture cannot construct frozen evidence kinds: {sorted(missing)}")
    if preferred_kind not in payloads:
        raise ValueError("preferred evidence kind is absent from the frozen phase")
    return [payloads[kind] for kind in phase_contract["evidence_kinds"]]


def _evidence_identity(
    attempt: Mapping[str, Any],
    phase: str,
    phase_execution: Mapping[str, Any],
    kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSIONS[kind],
        "evidence_kind": kind,
        "evidence_id": f"evidence:{kind}:{phase_execution['producer_run_id']}",
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": phase_execution["producer_run_id"],
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
        "phase": phase,
        "phase_execution_id": phase_execution["phase_execution_id"],
        "phase_start_event_id": phase_execution["phase_start_event_id"],
    }


def _measurement_row(
    attempt: Mapping[str, Any],
    *,
    phase: str,
    phase_execution: Mapping[str, Any],
    role: str,
    dataset_id: str,
    metric_id: str,
    seed: int,
    value: float,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "role": role,
        "dataset_id": dataset_id,
        "metric_id": metric_id,
        "seed": seed,
        "metric_value": value,
        "command_status": "completed",
        "attempt_id": attempt["attempt_id"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "producer_run_id": phase_execution["producer_run_id"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase_execution_id": phase_execution["phase_execution_id"],
        "phase_start_event_id": phase_execution["phase_start_event_id"],
    }


def _passing_measurement(comparator: str, threshold: Any) -> Any:
    if comparator in {"eq", "exact_set", "gte", "delta_gte", "lte"}:
        return deepcopy(threshold)
    if comparator == "gt":
        return float(threshold) + 1.0
    if comparator == "lt":
        return float(threshold) - 1.0
    raise ValueError(f"unsupported fixture readiness comparator: {comparator}")


def _stage_payloads(
    project_root: Path,
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    payloads: list[dict[str, Any]],
) -> dict:
    staging_root = (
        Path(".tmp")
        / "authoritative-evidence"
        / str(attempt["attempt_id"])
        / str(payloads[0]["phase_execution_id"])
    )
    inventory = []
    for payload in payloads:
        raw = encode_canonical_evidence(payload)
        digest = hashlib.sha256(raw).hexdigest()
        relative_path = staging_root / f"{payload['evidence_kind']}-{digest}.json"
        target = project_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        inventory.append(
            {"kind": payload["evidence_kind"], "source_path": relative_path.as_posix()}
        )
    return stage_authoritative_completion(
        project_root,
        dict(attempt),
        dict(trial_spec),
        inventory,
    )


def build_failure_evidence_v6(
    project_root: Path,
    attempt: dict,
    *,
    failure_class: str,
    suffix: str,
    exit_code: int = 1,
) -> dict:
    """Stage canonical non-resource failure bound to a committed PhaseRunReceipt."""

    if failure_class not in {
        "implementation_failure",
        "activation_failure",
        "integrity_failure",
        "safety_failure",
    }:
        raise ValueError("build_failure_evidence_v6 only supports non-resource failures")
    running_phase = next(
        (phase for phase in ("proxy", "full") if attempt["phases"].get(phase) == "RUNNING"),
        None,
    )
    if running_phase is None:
        raise ValueError("Attempt must have an authoritative running phase")
    phase_execution = attempt["phase_executions"][running_phase]
    if not isinstance(phase_execution, dict):
        raise ValueError("Attempt running phase is missing its execution manifest")
    source_phase = "activation" if failure_class == "activation_failure" else running_phase
    producer_run_id = phase_execution["producer_run_id"]
    identity = {
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": producer_run_id,
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": source_phase,
        "phase_execution_id": phase_execution["phase_execution_id"],
        "phase_start_event_id": phase_execution["phase_start_event_id"],
    }
    stderr_text = "b" * 64
    command_binding = _record_failed_phase_command(
        project_root,
        attempt,
        suffix=suffix,
        exit_code=exit_code,
        stderr=stderr_text,
    )

    command_status = "integrity_blocked" if failure_class in {"integrity_failure", "safety_failure"} else "failed"
    evidence = {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": "failure_evidence",
        "evidence_id": f"failure-evidence-{attempt['lifecycle_generation']}-{suffix}",
        **identity,
        "cross_references": {"phase_run_receipt_hash": command_binding["receipt_hash"]},
        "source_state": attempt["state"],
        "source_phase": source_phase,
        "failure_class": failure_class,
        "command_status": command_status,
        "exit_code": exit_code,
        "reason": f"verified {failure_class}",
        "observed_at": "2026-07-14T00:00:01Z",
        "log_hash": hashlib.sha256(stderr_text.encode()).hexdigest(),
        **command_binding,
    }
    evidence_raw = encode_canonical_evidence(evidence)
    evidence_hash = hashlib.sha256(evidence_raw).hexdigest()
    evidence_path = (
        project_root
        / "experiment"
        / "attempts"
        / attempt["attempt_id"]
        / producer_run_id
        / "failure_evidence"
        / f"{evidence_hash}.json"
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(evidence_raw)
    return evidence


build_failure_evidence_v4 = build_failure_evidence_v6


def build_resource_failure_evidence_v4(
    project_root: Path,
    attempt: dict,
    *,
    failure_class: str,
    suffix: str,
    resource_type: str = "system_memory",
    resource_id: str = "memory-0",
    required_capacity: float = 10.0,
    observed_capacity: float = 1.0,
    unit: str = "bytes",
    exit_code: int = 137,
) -> dict:
    """Stage canonical resource-pause failure and its immutable probe."""

    if failure_class not in {"resource_pause", "oom_retry"}:
        raise ValueError("build_resource_failure_evidence_v4 only supports resource failures")
    running_phase = next(
        (phase for phase in ("proxy", "full") if attempt["phases"].get(phase) == "RUNNING"),
        None,
    )
    if running_phase is None:
        raise ValueError("Attempt must have an authoritative running phase")
    phase_execution = attempt["phase_executions"][running_phase]
    if not isinstance(phase_execution, dict):
        raise ValueError("Attempt running phase is missing its execution manifest")
    producer_run_id = phase_execution["producer_run_id"]
    identity = {
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": producer_run_id,
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": running_phase,
        "phase_execution_id": phase_execution["phase_execution_id"],
        "phase_start_event_id": phase_execution["phase_start_event_id"],
    }
    command_binding = _record_failed_phase_command(
        project_root,
        attempt,
        suffix=suffix,
        exit_code=exit_code,
        stderr="resource capacity insufficient",
    )
    probe = {
        "schema_version": "auto_research_resource_probe_evidence_v4",
        "evidence_kind": "resource_probe",
        "evidence_id": f"resource-probe-{attempt['lifecycle_generation']}-{suffix}",
        **identity,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "required_capacity": required_capacity,
        "observed_capacity": observed_capacity,
        "unit": unit,
        "probe_status": "insufficient",
        "observed_at": "2026-07-14T00:00:00Z",
        **command_binding,
    }
    probe_raw = encode_canonical_evidence(probe)
    probe_hash = hashlib.sha256(probe_raw).hexdigest()
    probe_path = project_root / content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=producer_run_id,
        evidence_kind="resource_probe",
        content_hash=probe_hash,
    )
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_bytes(probe_raw)

    evidence = {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": "failure_evidence",
        "evidence_id": f"failure-evidence-{attempt['lifecycle_generation']}-{suffix}",
        **identity,
        "cross_references": {"resource_probe_hash": probe_hash},
        "source_state": attempt["state"],
        "source_phase": running_phase,
        "failure_class": failure_class,
        "command_status": "resource_paused",
        "exit_code": exit_code,
        "reason": f"verified {failure_class}",
        "observed_at": "2026-07-14T00:00:01Z",
        "log_hash": probe_hash,
        **command_binding,
    }
    evidence_raw = encode_canonical_evidence(evidence)
    evidence_hash = hashlib.sha256(evidence_raw).hexdigest()
    evidence_path = project_root / content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=producer_run_id,
        evidence_kind="failure_evidence",
        content_hash=evidence_hash,
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(evidence_raw)
    return evidence
