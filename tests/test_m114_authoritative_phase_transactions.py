from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.agents.experiment import (
    ExperimentAgent,
    _c2c_strict_evidence_inventory,
    _generic_external_evidence_inventory,
    _stage_evidence_inventory,
)
from auto_research.agents.plan import _trial_spec_from_plan
from auto_research.agents.base import AgentContext
from auto_research.artifacts import ArtifactManager
from auto_research.contract_store import ContractStore
from auto_research.llm import ModelClient
from auto_research.validators import run_stage_gate
from auto_research.domain_contracts import canonical_hash
from auto_research.evidence import content_addressed_evidence_path, encode_canonical_evidence
from auto_research.research_state import IntegrityError, ResearchEventLedger, _event_hash, canonical_json
from auto_research.s3_validation import S3ValidationError
from support.authoritative_evidence import record_completed_evidence_command
from test_m112_ledger_authority import _direction as named_direction
from test_m112_ledger_authority import _variant as named_variant
from test_m112_experiment_flow import _authoritative_direction_and_variant
from test_m113_ledger_closure import (
    _direction,
    _failure_evidence,
    _resume_evidence,
    _running_attempt,
    _receipt_backed_completion,
    _scoped_artifact,
    _trial_spec,
    _trial_spec_facts,
    _valid_completion,
    _variant,
)


def _c2c_inputs(tmp_path: Path, *, profile: str = "standard") -> tuple[dict, dict, dict, dict]:
    from support.authoritative_evidence import start_attempt_phase, build_trial_spec_v8
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction)
    variant_path = tmp_path / "plan" / "variant.json"
    variant_path.parent.mkdir(parents=True, exist_ok=True)
    variant_path.write_text(json.dumps(variant, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    trial_spec = _trial_spec_facts()
    trial_spec["protocol"]["required_phases"] = ["proxy"] if profile == "bootstrap" else ["proxy", "full"]
    trial_spec["protocol"]["terminal_phases"] = ["proxy"] if profile == "bootstrap" else ["full"]
    trial_spec["protocol"]["proxy_terminal_allowed"] = profile == "bootstrap"
    trial_spec["execution_contract"]["runtime_config"]["collector"] = "c2c_small_loop"
    trial_spec["execution_contract"]["runtime_config_hash"] = canonical_hash(
        trial_spec["execution_contract"]["runtime_config"]
    )
    trial_spec["evidence_requirements"] = [
        {"requirement_id": "proxy-results", "kind": "proxy_results", "required": True, "applicable_phases": ["proxy"], "schema_version": "auto_research_proxy_results_v1"},
        {"requirement_id": "activation", "kind": "activation_evidence", "required": True, "applicable_phases": ["proxy"], "schema_version": "auto_research_activation_evidence_v4"},
        {"requirement_id": "proxy-baseline", "kind": "proxy_baseline_fingerprint", "required": True, "applicable_phases": ["proxy"], "schema_version": "auto_research_proxy_baseline_fingerprint_v3"},
        {"requirement_id": "proxy-cache", "kind": "proxy_cache_report", "required": True, "applicable_phases": ["proxy"], "schema_version": "auto_research_proxy_cache_report_v3"},
        {"requirement_id": "bootstrap" if profile == "bootstrap" else "readiness", "kind": "bootstrap_completion" if profile == "bootstrap" else "full_s3_readiness", "required": True, "applicable_phases": ["proxy"], "schema_version": "auto_research_bootstrap_completion_v3" if profile == "bootstrap" else "auto_research_full_s3_readiness_v4"},
    ]
    if profile == "standard":
        trial_spec["evidence_requirements"].append({"requirement_id": "main", "kind": "main_results", "required": True, "applicable_phases": ["full"], "schema_version": "auto_research_main_results_v3"})
    trial_spec["required_artifacts"] = [item["kind"] for item in trial_spec["evidence_requirements"]]
    trial_spec = build_trial_spec_v8(
        trial_spec,
        project_root=tmp_path,
        adapter_id="auto-research-c2c",
        command_provenance_mode="local-external",
    )
    attempt = ledger.reserve_attempt(profile=profile, direction=direction, variant=variant, implementation_hash=canonical_hash({"impl": profile}), attempt_kind="bootstrap_proxy" if profile == "bootstrap" else "proxy_full", trial_spec=trial_spec)
    attempt = start_attempt_phase(ledger, attempt, "proxy")
    trial_spec = deepcopy(attempt["frozen_trial_spec"])
    comparison = {
        "metrics": {"mean": 1.0, "datasets": {"fake": 1.0}},
        "proxy_screen": {
            "metrics": {"mean": 1.0, "datasets": {"fake": 1.0}},
            "baseline_metrics": {"mean": 0.0, "datasets": {"fake": 0.0}},
        },
        "activation_smoke": {"status": "passed", "attempts": [{"status": "ok"}], "implementation_surface_ids": ["src/router.py"]},
        "full_s3_readiness": {"status": "ready", "full_train_allowed": True},
        "bootstrap": {"status": "proxy_reached"},
        "ablation": {"metrics": {"datasets": {"fake": 0.5}}},
        "matched_control_metrics": {"datasets": {"fake": 0.8}},
        "coverage_metrics": {"datasets": {"fake": 1.0}},
    }
    baseline = {"mean": 0.0, "datasets": {"fake": 0.0}}
    return attempt, trial_spec, comparison, baseline


def _generic_external_command(argv: list[str], cwd: Path) -> dict:
    raw_outputs = [
        {
            "output_id": "raw-main-results",
            "kind": "main_results",
            "schema_version": "auto_research_main_results_v3",
            "locator": "runner/main.json",
            "locator_type": "file",
            "dataset_id": None,
            "role": None,
            "required": True,
            "normalized_kinds": ["main_results"],
        },
        {
            "output_id": "raw-activation-evidence",
            "kind": "activation_evidence",
            "schema_version": "auto_research_activation_evidence_v4",
            "locator": "runner/activation.json",
            "locator_type": "file",
            "dataset_id": None,
            "role": None,
            "required": True,
            "normalized_kinds": ["activation_evidence"],
        },
    ]
    return {
        "argv": argv,
        "cwd": str(cwd),
        "environment": {
            "AUTO_RESEARCH_C2C_RAW_OUTPUT_SPECS": json.dumps(raw_outputs, sort_keys=True, separators=(",", ":")),
        },
        "physical_raw_outputs": raw_outputs,
    }


def test_c2c_real_inventory_rejects_mutable_results_without_authoritative_receipts(tmp_path: Path) -> None:
    attempt, trial_spec, comparison, baseline = _c2c_inputs(tmp_path)
    before = list((tmp_path / "experiment").rglob("*.json"))
    with pytest.raises(S3ValidationError, match="authoritative command receipts"):
        _c2c_strict_evidence_inventory(
            project_root=tmp_path,
            attempt=attempt,
            trial_spec=trial_spec,
            comparison_candidate=comparison,
            baseline=baseline,
            simulate=False,
        )
    assert list((tmp_path / "experiment").rglob("*.json")) == before


def test_c2c_bootstrap_inventory_is_proxy_only_and_contains_completion(tmp_path: Path) -> None:
    attempt, trial_spec, comparison, baseline = _c2c_inputs(tmp_path, profile="bootstrap")
    inventory = _c2c_strict_evidence_inventory(
        project_root=tmp_path,
        attempt=attempt,
        trial_spec=trial_spec,
        comparison_candidate=comparison,
        baseline=baseline,
        simulate=True,
    )
    kinds = {item["kind"] for item in inventory}
    assert "bootstrap_completion" in kinds
    assert "full_s3_readiness" not in kinds
    main_source = next(item for item in inventory if item["kind"] == "proxy_results")
    main_payload = json.loads((tmp_path / main_source["source_path"]).read_text(encoding="utf-8"))
    assert {row["phase"] for row in main_payload["rows"]} == {"proxy"}


def test_repair_generation_rejects_old_generation_completion(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    stale_completion = _valid_completion(tmp_path, attempt)
    failure = _failure_evidence(tmp_path, attempt, failure_class="activation_failure", exit_code=1)
    repaired, _ = ledger.disposition_failure(failure)
    revised = ledger.revise_implementation(
        repaired["attempt_id"], implementation_hash=canonical_hash({"implementation": 2})
    )
    from support.authoritative_evidence import start_attempt_phase
    revised = start_attempt_phase(ledger, revised, "full")
    before = ledger.state()
    with pytest.raises(IntegrityError, match="generation|implementation|input"):
        ledger.complete_attempt(stale_completion)
    after = ledger.state()
    assert after["last_sequence"] == before["last_sequence"]
    assert after["trial_results"] == before["trial_results"]
    assert after["directions"][attempt["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 1, "consumed": 0}


def _rehash_chain(ledger: ResearchEventLedger, sequence: int, payload: dict) -> None:
    with sqlite3.connect(ledger.db_path) as connection:
        row = connection.execute("SELECT * FROM events WHERE sequence = ?", (sequence,)).fetchone()
        assert row is not None
        event = {
            "schema_version": row[7], "event_id": row[1], "sequence": row[0],
            "event_type": row[2], "previous_event_hash": row[4], "created_at": row[6],
            "payload": payload,
        }
        event_hash = _event_hash(event)
        connection.execute("UPDATE events SET payload_json = ?, event_hash = ? WHERE sequence = ?", (canonical_json(payload), event_hash, sequence))
        previous_hash = event_hash
        for following in connection.execute("SELECT * FROM events WHERE sequence > ? ORDER BY sequence", (sequence,)).fetchall():
            next_event = {
                "schema_version": following[7], "event_id": following[1], "sequence": following[0],
                "event_type": following[2], "previous_event_hash": previous_hash,
                "created_at": following[6], "payload": json.loads(following[3]),
            }
            next_hash = _event_hash(next_event)
            connection.execute("UPDATE events SET previous_event_hash = ?, event_hash = ? WHERE sequence = ?", (previous_hash, next_hash, following[0]))
            previous_hash = next_hash


@pytest.mark.parametrize("field", ["outcome_classification", "all_hard_constraints_passed", "primary_metric_summary"])
def test_reducer_rejects_forged_finalization_derivatives(tmp_path: Path, field: str) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    completion = _receipt_backed_completion(tmp_path, ledger, attempt)
    ledger.complete_attempt(completion)
    event = next(item for item in ledger.events() if item["event_type"] == "AttemptFinalized")
    payload = deepcopy(event["payload"])
    trial = payload["trial_result"]
    if field == "outcome_classification":
        trial[field] = "rejected" if trial[field] == "accepted" else "accepted"
    elif field == "all_hard_constraints_passed":
        trial[field] = not trial[field]
    else:
        trial[field] = {**trial[field], "delta": trial[field]["delta"] + 9.0}
    _rehash_chain(ledger, event["sequence"], payload)
    with pytest.raises(IntegrityError):
        ledger.rebuild()


def test_evidence_parent_symlink_escape_is_rejected(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    completion = _receipt_backed_completion(tmp_path, ledger, attempt)
    entry = completion["entries"][0]
    artifact = tmp_path / entry["relative_path"]
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    moved = outside / artifact.name
    shutil.copy2(artifact, moved)
    shutil.rmtree(artifact.parent)
    os.symlink(outside, artifact.parent)
    before = len(ledger.events())
    with pytest.raises(IntegrityError, match="symlink|outside|evidence root"):
        ledger.complete_attempt(completion)
    assert len(ledger.events()) == before


def test_resume_rejects_probe_with_other_attempt_identity(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    paused, _ = ledger.disposition_failure(
        _failure_evidence(tmp_path, attempt, failure_class="resource_pause", exit_code=137)
    )
    evidence = _resume_evidence(tmp_path, ledger, paused, resource_type="gpu_memory")
    old_hash = evidence["cross_references"]["resource_probe_hash"]
    producer = evidence["producer_run_id"]
    root = tmp_path / "experiment" / "attempts" / paused["attempt_id"] / producer / "resource_probe"
    probe_path = next(root.glob(f"{old_hash}.json"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["attempt_id"] = "other-attempt"
    raw = encode_canonical_evidence(probe)
    new_hash = hashlib.sha256(raw).hexdigest()
    new_path = root / f"{new_hash}.json"
    new_path.write_bytes(raw)
    evidence["cross_references"]["resource_probe_hash"] = new_hash
    _scoped_artifact(tmp_path, paused, producer, "resume_evidence", evidence)
    with pytest.raises(IntegrityError, match="identity|attempt|authority|completed authoritative command"):
        ledger.resume_attempt(evidence)


def test_exact_replay_uses_attempt_scoped_trial_spec_not_global_projection(tmp_path: Path) -> None:
    ledger, attempt_a = _running_attempt(tmp_path)
    completion = _receipt_backed_completion(tmp_path, ledger, attempt_a)
    completed, route = ledger.complete_attempt(completion)
    direction_b = named_direction("direction-b")
    variant_b = named_variant(direction_b, "variant-b")
    ledger.select_direction(direction_b)
    ledger.plan_variant(variant_b)
    trial_spec_b = deepcopy(_trial_spec(tmp_path))
    trial_spec_b["protocol"]["protocol_id"] = "m114-full-b"
    ledger.reserve_attempt(
        profile="standard", direction=direction_b, variant=variant_b,
        implementation_hash=canonical_hash({"implementation": "b"}), attempt_kind="full", trial_spec=trial_spec_b,
    )
    before = len(ledger.events())
    replayed, replayed_route = ledger.complete_attempt(completion)
    assert replayed == completed
    assert replayed_route == route
    assert len(ledger.events()) == before


def test_bootstrap_and_standard_cannot_both_be_active(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction)
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    ledger.reserve_attempt(
        profile="bootstrap", direction=direction, variant=variant,
        implementation_hash=canonical_hash({"bootstrap": 1}), attempt_kind="bootstrap_proxy", trial_spec=_trial_spec(tmp_path),
    )
    with pytest.raises(IntegrityError, match="execution_width=1"):
        ledger.reserve_attempt(
            profile="standard", direction=direction, variant=variant,
            implementation_hash=canonical_hash({"standard": 1}), attempt_kind="full", trial_spec=_trial_spec(tmp_path),
        )


def test_complete_attempt_honors_explicit_event_id(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    completion = _receipt_backed_completion(tmp_path, ledger, attempt)
    ledger.complete_attempt(completion, event_id="explicit-finalization-id")
    assert ledger.events()[-1]["event_id"] == "explicit-finalization-id"


def test_unregistered_optional_evidence_is_rejected_before_commit(tmp_path: Path) -> None:
    ledger, attempt = _running_attempt(tmp_path)
    completion = _receipt_backed_completion(tmp_path, ledger, attempt)
    producer_run_id = attempt["phase_executions"]["full"]["producer_run_id"]
    evidence_id = "evidence:unregistered-activation"
    payload = {
        "schema_version": "auto_research_activation_evidence_v4",
        "evidence_kind": "activation_evidence",
        "evidence_id": evidence_id,
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
        "phase": "full",
        "phase_execution_id": attempt["phase_executions"]["full"]["phase_execution_id"],
        "phase_start_event_id": attempt["phase_executions"]["full"]["phase_start_event_id"],
        "cross_references": {},
        "probe_id": "unregistered-probe",
        "status": "activated",
        "command_status": "completed",
        "exit_code": 0,
        "expected_surface_ids": ["src/model.py"],
        "observed_surface_ids": ["src/model.py"],
        "activation_delta_threshold": 0.0,
        "surface_measurements": [{
            "surface_id": "src/model.py",
            "enabled_value": 1.0,
            "disabled_value": 0.0,
            "delta": 1.0,
            "threshold": 0.0,
            "status": "ACTIVATED",
        }],
    }
    raw = encode_canonical_evidence(payload)
    digest = hashlib.sha256(raw).hexdigest()
    relative_path = content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"], producer_run_id=producer_run_id,
        evidence_kind="activation_evidence", content_hash=digest,
    )
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    completion["entries"].append({
        "evidence_id": evidence_id,
        "kind": "activation_evidence",
        "relative_path": relative_path,
        "content_hash": digest,
        "schema_version": "auto_research_activation_evidence_v4",
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
        "phase": "full",
        "phase_execution_id": attempt["phase_executions"]["full"]["phase_execution_id"],
        "phase_start_event_id": attempt["phase_executions"]["full"]["phase_start_event_id"],
    })
    before = ledger.state()
    with pytest.raises(IntegrityError, match="unregistered|evidence requirement|artifact"):
        ledger.complete_attempt(completion)
    after = ledger.state()
    assert after["last_sequence"] == before["last_sequence"]
    assert after["trial_results"] == before["trial_results"]


def test_generic_non_simulate_external_manifest_commits_strict_trial(tmp_path: Path) -> None:
    direction, variant = _authoritative_direction_and_variant()
    project_root = tmp_path / "generic-real"
    project_root.mkdir()
    sample_bytes = b'{"id":"sample-1","label":1}\n'
    sample_source = project_root / "samples" / "dataset-a.jsonl"
    sample_source.parent.mkdir(parents=True)
    sample_source.write_bytes(sample_bytes)
    evaluator_source = project_root / "evaluator.py"
    evaluator_source.write_text(
        "def evaluate(baseline, candidate):\n    return candidate - baseline\n",
        encoding="utf-8",
    )
    plan = {
        "hypotheses": [{"id": "H1", "statement": "candidate improves accuracy"}],
        "baselines": [{"name": "baseline"}],
        "datasets": [{"name": "dataset-a", "split": "validation", "sample_count": 1, "source_revision": "local-fixture-v1", "ordered_sample_ids": [hashlib.sha256(sample_bytes).hexdigest()]}],
        "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [7], "aggregation": "mean", "require_complete_seed_coverage": True},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 0.0},
        "ablation_matrix": [],
        "execution": {
            "mode": "real", "collector": "external_manifest", "commands": [_generic_external_command([sys.executable, "producer.py"], project_root)],
            "workdir": str(project_root), "phase_manifest_path": "runner/phase_manifest.json",
            "evaluator_id": "fixture-evaluator", "evaluator_source_paths": ["evaluator.py"],
            "dependency_payloads": [{"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}],
        },
        "resource_budget": {"wall_clock_minutes": 5},
    }
    trial_spec = _trial_spec_from_plan(plan, variant, project_root=project_root)
    contract_store = ContractStore(project_root)
    sample_manifest = contract_store.read_contract(
        trial_spec["sample_manifest_ref"],
        contract_kind="sample_manifest",
        schema_file="sample_manifest_v4.schema.json",
    )
    evaluator_manifest = contract_store.read_contract(
        trial_spec["execution_contract"]["evaluator_manifest_ref"],
        contract_kind="evaluator_manifest",
        schema_file="evaluator_manifest_v2.schema.json",
    )
    assert sample_manifest["datasets"][0]["ordered_sample_ids"] == [hashlib.sha256(sample_bytes).hexdigest()]
    assert contract_store.read_bytes(evaluator_manifest["source_blobs"][0]) == evaluator_source.read_bytes()
    assert evaluator_manifest["provenance_mode"] == "real"
    projections = {
        "literature/direction.json": direction,
        "plan/variant.json": variant,
        "plan/trial_spec.json": trial_spec,
        "plan/code_patches/implementation_contract.json": {"schema_version": "implementation_contract_test_v1", "variant_spec_hash": variant["variant_spec_hash"]},
        "plan/code_patches/patch_manifest.json": {"schema_version": "auto_research_patch_manifest_v1", "status": "disabled", "selected_candidate_id": variant["variant_id"], "variant_spec_hash": variant["variant_spec_hash"]},
        "plan/code_patches/patch_gate_report.json": {"schema_version": "auto_research_patch_gate_v1", "gate": "pass", "variant_id": variant["variant_id"], "variant_spec_hash": variant["variant_spec_hash"], "checks": {"activation": True}},
    }
    for relative_path, payload in projections.items():
        target = project_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
    (project_root / "producer.py").write_text(
        """import json
from pathlib import Path
root=Path('.')
attempt=json.loads(next((root/'meta/attempts').glob('*.json')).read_text())
phase=attempt['phase_executions']['full']; producer=phase['producer_run_id']
common={'lifecycle_generation':attempt['lifecycle_generation'],'implementation_hash':attempt['implementation_hash'],'attempt_input_hash':attempt['attempt_input_hash'],'phase':'full','phase_execution_id':phase['phase_execution_id'],'phase_start_event_id':phase['phase_start_event_id']}
identity={'attempt_id':attempt['attempt_id'],'producer_run_id':producer,'direction_semantic_hash':attempt['direction_semantic_hash'],'direction_spec_hash':attempt['direction_spec_hash'],'variant_semantic_hash':attempt['variant_semantic_hash'],'variant_spec_hash':attempt['variant_spec_hash'],'trial_spec_hash':attempt['trial_spec_hash'],'protocol_hash':attempt['protocol_hash'],'sample_manifest_hash':attempt['sample_manifest_hash'],'evaluator_hash':attempt['evaluator_hash'],'cross_references':{},**common}
rows=[]
for role,value in [('baseline',0.5),('candidate',0.8)]: rows.append({'phase':'full','role':role,'dataset_id':'dataset-a','metric_id':'accuracy','seed':7,'metric_value':value,'command_status':'completed','attempt_id':attempt['attempt_id'],'variant_semantic_hash':attempt['variant_semantic_hash'],'variant_spec_hash':attempt['variant_spec_hash'],'trial_spec_hash':attempt['trial_spec_hash'],'sample_manifest_hash':attempt['sample_manifest_hash'],'evaluator_hash':attempt['evaluator_hash'],'producer_run_id':producer,**{k:v for k,v in common.items() if k!='phase'}})
main={'schema_version':'auto_research_main_results_v3','evidence_kind':'main_results','evidence_id':'evidence-main-real',**identity,'rows':rows}
activation={'schema_version':'auto_research_activation_evidence_v4','evidence_kind':'activation_evidence','evidence_id':'evidence-activation-real',**identity,'probe_id':'forward-probe-real','status':'activated','command_status':'completed','exit_code':0,'expected_surface_ids':['src/router.py'],'observed_surface_ids':['src/router.py'],'activation_delta_threshold':0.0,'surface_measurements':[{'surface_id':'src/router.py','enabled_value':1.0,'disabled_value':0.0,'delta':1.0,'threshold':0.0,'status':'ACTIVATED'}]}
(root/'runner').mkdir(); (root/'runner/main.json').write_text(json.dumps(main,sort_keys=True,separators=(',',':'))); (root/'runner/activation.json').write_text(json.dumps(activation,sort_keys=True,separators=(',',':')))
manifest={**phase,'sample_contract_ref':attempt['frozen_trial_spec']['sample_manifest_ref'],'evaluator_contract_ref':attempt['frozen_trial_spec']['execution_contract']['evaluator_manifest_ref'],'artifacts':[{'kind':'main_results','source_path':'runner/main.json','producer_run_id':producer},{'kind':'activation_evidence','source_path':'runner/activation.json','producer_run_id':producer}]}
(root/'runner/phase_manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':')))
""",
        encoding="utf-8",
    )
    config = {"experiment": {"simulate": False}, "orchestration": {"profile": "standard"}, "llm": {"use_real_api": False}}
    context = AgentContext(project_root, config, ArtifactManager(project_root), ModelClient(config, project_root=project_root))
    result = ExperimentAgent(context).run()
    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    state = ResearchEventLedger(project_root).state()
    trial = next(iter(state["trial_results"].values()))
    assert trial["outcome_classification"] == "accepted"
    assert {item["role"] for item in trial["observations"]} == {"baseline", "candidate"}
    assert state["directions"][direction["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 0, "consumed": 1}
    assert run_stage_gate("S3_experiment", project_root, config).to_dict()["status"] == "PASS"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.update(schema_version="auto_research_phase_execution_manifest_v1"), "PhaseExecutionManifest v2"),
        (lambda manifest: manifest.update(command_plan_hash="0" * 64), "Ledger phase authorization"),
        (lambda manifest: manifest.update(sample_contract_ref=manifest["evaluator_contract_ref"]), "sample ContractRef mismatch"),
        (lambda manifest: manifest.update(expected_evidence_kinds=["unexpected_results"]), "Ledger phase authorization"),
    ],
)
def test_generic_external_manifest_rejects_non_authoritative_v2_bindings(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    _, attempt = _running_attempt(tmp_path)
    phase_manifest = deepcopy(attempt["phase_executions"]["full"])
    producer_run_id = phase_manifest["producer_run_id"]
    artifacts = [
        {"kind": kind, "source_path": f"runner/{kind}.json", "producer_run_id": producer_run_id}
        for kind in phase_manifest["expected_evidence_kinds"]
    ]
    manifest = {
        **phase_manifest,
        "sample_contract_ref": deepcopy(attempt["frozen_trial_spec"]["sample_manifest_ref"]),
        "evaluator_contract_ref": deepcopy(attempt["frozen_trial_spec"]["execution_contract"]["evaluator_manifest_ref"]),
        "artifacts": artifacts,
    }
    manifest_path = tmp_path / "runner" / "phase_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    execution = {"phase_manifest_path": "runner/phase_manifest.json"}
    assert {item["kind"] for item in _generic_external_evidence_inventory(project_root=tmp_path, attempt=attempt, execution=execution)} == set(
        phase_manifest["expected_evidence_kinds"]
    )

    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(S3ValidationError, match=message):
        _generic_external_evidence_inventory(project_root=tmp_path, attempt=attempt, execution=execution)


def test_real_evaluator_manifest_tamper_rejects_reservation_without_write(tmp_path: Path) -> None:
    direction, variant = _authoritative_direction_and_variant()
    evaluator_source = tmp_path / "evaluator.py"
    evaluator_source.write_text("def evaluate(value):\n    return value\n", encoding="utf-8")
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "dataset-a.jsonl").write_text('{"id":"sample-a","text":"fixture"}\n', encoding="utf-8")
    plan = {
        "hypotheses": [{"id": "H1", "statement": "candidate improves accuracy"}],
        "baselines": [{"name": "baseline"}],
        "datasets": [{"name": "dataset-a", "split": "validation", "sample_count": 1, "source_revision": "fixture-v1", "sample_source_path": "samples/dataset-a.jsonl"}],
        "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [7], "aggregation": "mean", "require_complete_seed_coverage": True},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 0.0},
        "ablation_matrix": [],
        "execution": {
            "mode": "real", "collector": "external_manifest", "commands": [_generic_external_command(["true"], tmp_path)], "workdir": str(tmp_path),
            "phase_manifest_path": "runner/phase_manifest.json", "evaluator_id": "fixture-evaluator",
            "evaluator_source_paths": ["evaluator.py"], "dependency_payloads": [{"lock": "fixture-v1"}],
        },
        "resource_budget": {"wall_clock_minutes": 5},
    }
    trial_spec = _trial_spec_from_plan(plan, variant, project_root=tmp_path)
    ledger = ResearchEventLedger(tmp_path)
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    before = ledger.state()
    evaluator_ref = trial_spec["execution_contract"]["evaluator_manifest_ref"]["blob"]
    evaluator_path = tmp_path / evaluator_ref["relative_path"]
    evaluator_path.write_bytes(evaluator_path.read_bytes() + b" ")
    with pytest.raises(IntegrityError, match="ContractRef rejected"):
        ledger.reserve_attempt(
            profile="standard", direction=direction, variant=variant,
            implementation_hash=canonical_hash({"implementation": "fixture"}),
            attempt_kind="full", trial_spec=trial_spec,
        )
    after = ledger.state()
    assert after["last_sequence"] == before["last_sequence"]
    assert after["attempts"] == before["attempts"]


def test_proxy_evidence_transaction_gates_full_without_budget_consumption(tmp_path: Path) -> None:
    attempt, trial_spec, comparison, baseline = _c2c_inputs(tmp_path)
    ledger = ResearchEventLedger(tmp_path)
    inventory = _c2c_strict_evidence_inventory(
        project_root=tmp_path, attempt=attempt, trial_spec=trial_spec,
        comparison_candidate=comparison, baseline=baseline, simulate=True,
    )
    completion = _stage_evidence_inventory(project_root=tmp_path, attempt=attempt, trial_spec=trial_spec, inventory=inventory)
    record_completed_evidence_command(tmp_path, ledger, attempt, completion)
    proxy_attempt, route = ledger.commit_proxy_evidence(completion)
    assert proxy_attempt["state"] == "PROXY_COMPLETED"
    assert route["next_action"] == "RUN_FULL"
    assert ledger.state()["directions"][attempt["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 1, "consumed": 0}
    full_attempt = ledger.start_full_phase(attempt["attempt_id"], phase_execution_id="phase-full-transaction", producer_run_id="producer-full-transaction")
    assert full_attempt["state"] == "FULL_RUNNING"
    assert [event["event_type"] for event in ledger.events() if event["event_type"] in {"ProxyPhaseStarted", "ProxyEvidenceCommitted", "FullPhaseStarted"}] == ["ProxyPhaseStarted", "ProxyEvidenceCommitted", "FullPhaseStarted"]


def test_proxy_reject_prevents_full_start_and_releases_reservation(tmp_path: Path) -> None:
    attempt, trial_spec, comparison, baseline = _c2c_inputs(tmp_path)
    comparison["proxy_screen"]["metrics"]["datasets"]["fake"] = -1.0
    ledger = ResearchEventLedger(tmp_path)
    inventory = _c2c_strict_evidence_inventory(
        project_root=tmp_path, attempt=attempt, trial_spec=trial_spec,
        comparison_candidate=comparison, baseline=baseline, simulate=True,
    )
    completion = _stage_evidence_inventory(project_root=tmp_path, attempt=attempt, trial_spec=trial_spec, inventory=inventory)
    record_completed_evidence_command(tmp_path, ledger, attempt, completion)
    rejected, route = ledger.commit_proxy_evidence(completion)
    assert rejected["state"] == "ABANDONED"
    assert route["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert ledger.state()["directions"][attempt["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 0, "consumed": 0}
    with pytest.raises(IntegrityError, match="PROXY_COMPLETED|RUN_FULL|full phase"):
        ledger.start_full_phase(attempt["attempt_id"], phase_execution_id="phase-full-forbidden", producer_run_id="producer-full-forbidden")
    assert all(event["event_type"] != "FullPhaseStarted" for event in ledger.events())


def test_rebuild_rejects_forged_proxy_outcome_derivatives(tmp_path: Path) -> None:
    attempt, trial_spec, comparison, baseline = _c2c_inputs(tmp_path)
    ledger = ResearchEventLedger(tmp_path)
    inventory = _c2c_strict_evidence_inventory(project_root=tmp_path, attempt=attempt, trial_spec=trial_spec, comparison_candidate=comparison, baseline=baseline, simulate=True)
    completion = _stage_evidence_inventory(project_root=tmp_path, attempt=attempt, trial_spec=trial_spec, inventory=inventory)
    record_completed_evidence_command(tmp_path, ledger, attempt, completion)
    ledger.commit_proxy_evidence(completion)
    event = next(item for item in ledger.events() if item["event_type"] == "ProxyEvidenceCommitted")
    payload = deepcopy(event["payload"])
    payload["proxy_outcome"]["observed_delta"] = 999.0
    _rehash_chain(ledger, event["sequence"], payload)
    with pytest.raises(IntegrityError, match="ProxyOutcome|immutable evidence"):
        ledger.rebuild()
