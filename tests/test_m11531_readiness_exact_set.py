from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from auto_research.agents.experiment import ExperimentAgent
from auto_research.contract_store import ContractStore
from auto_research.derivation_contracts import (
    build_readiness_check_plan,
    freeze_decoder_descriptor,
)
from auto_research.derivation_validation import ReceiptBoundSources
from auto_research.domain_contracts import canonical_hash
from auto_research.orchestrator import Orchestrator
from auto_research.proxy_classifier import derive_readiness_from_receipts
from auto_research.research_state import ResearchEventLedger
from auto_research.validators import run_stage_gate
from support.m1152_local_subprocess import (
    GENERIC_MARKER,
    create_generic_project,
    generic_invocations,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_identity() -> dict[str, Any]:
    return {
        "attempt_id": "attempt-m11531-readiness",
        "direction_semantic_hash": "1" * 64,
        "direction_spec_hash": "2" * 64,
        "variant_semantic_hash": "3" * 64,
        "variant_spec_hash": "4" * 64,
        "trial_spec_hash": "5" * 64,
        "protocol_hash": "6" * 64,
        "sample_manifest_hash": "7" * 64,
        "evaluator_hash": "8" * 64,
        "lifecycle_generation": 0,
        "implementation_hash": "9" * 64,
        "attempt_input_hash": "a" * 64,
        "phase": "proxy",
        "phase_execution_id": "phase-proxy-m11531",
        "phase_start_event_id": "event:proxy:m11531",
        "producer_run_id": "producer-m11531",
    }


def _receipt_bound_readiness_fact(
    store: ContractStore,
    *,
    command_spec_id: str,
    output_id: str,
    check_id: str,
) -> tuple[tuple[str, str, str], dict[str, Any], dict[str, Any]]:
    identity = _safe_identity()
    payload = {
        "schema_version": "auto_research_raw_readiness_check_v1",
        **identity,
        "command_spec_id": command_spec_id,
        "output_id": output_id,
        "check_id": check_id,
        "measurement": True,
        "ready": True,
    }
    output_ref = store.put_json(payload)
    receipt_payload = {
        "schema_version": "auto_research_test_receipt_lineage_v1",
        **identity,
        "command_id": f"cmd-{command_spec_id}",
        "command_spec_id": command_spec_id,
        "command_status": "completed",
        "exit_code": 0,
        "outputs": [
            {
                "output_id": output_id,
                "kind": "raw_readiness_check",
                "schema_version": payload["schema_version"],
                "contract_ref": output_ref,
                "content_hash": output_ref["digest"],
            }
        ],
    }
    receipt_ref = store.put_json(receipt_payload)
    lineage = {
        **identity,
        "source_phase": "proxy",
        "command_id": receipt_payload["command_id"],
        "command_spec_id": command_spec_id,
        "output_id": output_id,
        "output_kind": "raw_readiness_check",
        "output_schema_version": payload["schema_version"],
        "authority_roles": ["readiness"],
        "readiness_check_ids": [check_id],
        "command_status": "completed",
        "exit_code": 0,
        "receipt_hash": receipt_ref["digest"],
        "receipt_ref": receipt_ref,
        "output_hash": output_ref["digest"],
        "output_ref": output_ref,
        "completed_event_id": f"event:completed:{command_spec_id}",
    }
    assert store.read_json(output_ref) == payload
    assert store.read_json(receipt_ref) == receipt_payload
    return ("proxy", command_spec_id, output_id), payload, lineage


def _readiness_plan(root: Path) -> dict[str, Any]:
    decoder = freeze_decoder_descriptor(
        root,
        decoder_id="canonical-identity",
        decoder_version="1",
        semantic_contract={
            "canonicalization": {
                "encoding": "utf-8",
                "object_key_order": "lexicographic",
                "row_order": ["phase", "role", "dataset_id", "metric_id", "seed"],
                "duplicate_policy": "reject",
                "numeric_policy": "finite_non_boolean",
            },
            "coverage_contract": {
                "mode": "exact_cartesian",
                "datasets": ["readiness-dataset"],
                "seeds": [0],
                "metrics": ["readiness"],
                "roles": ["readiness"],
            },
        },
        authority_role_contract={
            "source_bindings": [
                {
                    "source_ordinal": 0,
                    "source_phase": "proxy",
                    "command_spec_id": "proxy-readiness-runtime",
                    "output_id": "proxy-readiness-runtime-output",
                    "authority_roles": ["readiness"],
                    "readiness_check_ids": ["runtime-ready-m11531"],
                }
            ]
        },
        output_contract={
            "expected_normalized_outputs": [
                {
                    "ordinal": 0,
                    "output_id": "normalized-readiness-m11531",
                    "kind": "full_s3_readiness",
                    "schema_version": "auto_research_full_s3_readiness_v4",
                }
            ]
        },
    )
    return build_readiness_check_plan(
        plan_id="readiness-plan-m11531",
        phase="proxy",
        checks=[
            {
                "ordinal": 0,
                "check_id": "runtime-ready-m11531",
                "check_kind": "raw_measurement",
                "source_bindings": [
                    {
                        "source_ordinal": 0,
                        "source_phase": "proxy",
                        "command_spec_id": "proxy-readiness-runtime",
                        "output_id": "proxy-readiness-runtime-output",
                        "output_kind": "raw_readiness_check",
                        "output_schema_version": "auto_research_raw_readiness_check_v1",
                        "required_authority_roles": ["readiness"],
                        "check_id": "runtime-ready-m11531",
                    }
                ],
                "predicate": {
                    "field_path": "proxy-readiness-runtime-output.ready",
                    "comparator": "eq",
                    "threshold": True,
                },
                "required_coverage": {"mode": "exact", "expected_surface_ids": []},
                "decoder_descriptor": decoder,
                "blocked_classification": "IMPLEMENTATION_BLOCKED",
                "blocked_route": "REPAIR_IMPLEMENTATION",
            }
        ],
    )


def test_extra_valid_lineage_readiness_raw_fact_is_rejected_exact_set(tmp_path: Path) -> None:
    store = ContractStore(tmp_path)
    plan = _readiness_plan(tmp_path)
    registered_key, registered_fact, registered_lineage = _receipt_bound_readiness_fact(
        store,
        command_spec_id="proxy-readiness-runtime",
        output_id="proxy-readiness-runtime-output",
        check_id="runtime-ready-m11531",
    )
    baseline_sources = ReceiptBoundSources(
        raw_facts={registered_key: registered_fact},
        raw_fact_lineage={registered_key: registered_lineage},
        surface_checks=(),
        physical_inputs=(),
    )
    baseline = derive_readiness_from_receipts(
        readiness_check_plan=plan,
        receipt_bound_sources=baseline_sources,
    )
    assert baseline["ready"] is True
    assert baseline["classification"] == "PASS"

    extra_key, extra_fact, extra_lineage = _receipt_bound_readiness_fact(
        store,
        command_spec_id="proxy-readiness-unregistered",
        output_id="proxy-readiness-unregistered-output",
        check_id="unregistered-readiness-m11531",
    )
    assert extra_key not in {
        (
            binding["source_phase"],
            binding["command_spec_id"],
            binding["output_id"],
        )
        for check in plan["checks"]
        for binding in check["source_bindings"]
    }
    attacked_sources = ReceiptBoundSources(
        raw_facts={registered_key: registered_fact, extra_key: extra_fact},
        raw_fact_lineage={registered_key: registered_lineage, extra_key: extra_lineage},
        surface_checks=(),
        physical_inputs=(),
    )

    with pytest.raises(ValueError, match="exact|extra|frozen|registered"):
        derive_readiness_from_receipts(
            readiness_check_plan=plan,
            receipt_bound_sources=attacked_sources,
        )


def _event_trace(root: Path) -> tuple[tuple[int, str, str, str], ...]:
    database = root / "meta" / "research_events.sqlite3"
    if not database.is_file():
        return ()
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        return tuple(
            (int(sequence), str(event_id), str(event_type), str(event_hash))
            for sequence, event_id, event_type, event_hash in connection.execute(
                "SELECT sequence, event_id, event_type, event_hash FROM events ORDER BY sequence"
            )
        )
    finally:
        connection.close()


def _agent_then_orchestrator_authority(context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    result = ExperimentAgent(context).run()
    route = Orchestrator._authoritative_s3_route(context.project_root, result)
    return result, route


_COLD_RESTART = r"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["M11531_TESTS"])

from auto_research.agents.base import AgentContext
from auto_research.agents.experiment import ExperimentAgent
from auto_research.artifacts import ArtifactManager
from auto_research.llm import ModelClient
from auto_research.orchestrator import Orchestrator
from auto_research.research_state import ResearchEventLedger

root = Path(os.environ["M11531_ROOT"])
config = json.loads(os.environ["M11531_CONFIG"])
context = AgentContext(root, config, ArtifactManager(root), ModelClient(config, project_root=root))
payload = {"pid": os.getpid(), "route": None, "historical_route": None, "error": None}
try:
    result = ExperimentAgent(context).run()
    route = Orchestrator._authoritative_s3_route(root, result)
    payload["route"] = route
    historical = ResearchEventLedger(root).query_operation_result(route["source"]["event_id"])
    payload["historical_route"] = historical.get("route_outcome")
except BaseException as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
"""


def _cold_restart(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "M11531_ROOT": str(root),
            "M11531_CONFIG": json.dumps(config, sort_keys=True),
            "M11531_TESTS": str(_PROJECT_ROOT / "tests"),
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(_PROJECT_ROOT / "tests"), environment.get("PYTHONPATH", "")))
            ),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _COLD_RESTART],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def test_prederive_integrity_failure_commits_replayable_block_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_root = tmp_path / "control"
    control_root.mkdir()
    control_context, _, _ = create_generic_project(control_root)
    control_result, control_route = _agent_then_orchestrator_authority(control_context)
    assert control_route == control_result["route_outcome"]
    assert control_route["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert run_stage_gate("S3_experiment", control_root, control_context.config).to_dict()["status"] == "PASS"

    attacked_root = tmp_path / "attacked"
    attacked_root.mkdir()
    attacked_context, _, _ = create_generic_project(attacked_root)
    original = ExperimentAgent._commit_phase_evidence_derivation
    attack: dict[str, Any] = {}

    def remove_decoder_after_physical(self: ExperimentAgent):
        context = self._active_phase_context
        assert context is not None
        state = ResearchEventLedger(self.context.project_root).state()
        completed_physical = [
            record
            for record in state["phase_commands"].values()
            if record["status"] == "completed"
            and record["command"]["phase"] == context.phase
            and not record["command"]["command_spec_id"].endswith("derive-evidence")
        ]
        assert completed_physical, "attack must occur after a committed physical receipt"
        store = ContractStore(self.context.project_root)
        command_plan = store.read_json(
            context.command_plan_hash,
            schema_file="phase_command_plan_v4.schema.json",
        )
        reference = deepcopy(command_plan["derivation_plan"]["decoder_descriptor"]["immutable_ref"])
        decoder_path = self.context.project_root / reference["relative_path"]
        attack.update(
            {
                "decoder_ref": reference,
                "decoder_bytes_hash": reference["digest"],
                "physical_completed_event_ids": [
                    record["completed_event_id"] for record in completed_physical
                ],
            }
        )
        decoder_path.unlink()
        assert not decoder_path.exists()
        return original(self)

    monkeypatch.setattr(
        ExperimentAgent,
        "_commit_phase_evidence_derivation",
        remove_decoder_after_physical,
    )
    first_result: dict[str, Any] | None = None
    first_route: dict[str, Any] | None = None
    first_error: str | None = None
    try:
        first_result, first_route = _agent_then_orchestrator_authority(attacked_context)
    except BaseException as exc:
        first_error = f"{type(exc).__name__}: {exc}"
    finally:
        monkeypatch.setattr(
            ExperimentAgent,
            "_commit_phase_evidence_derivation",
            original,
        )

    assert attack, "pre-derive decoder attack did not execute"
    before_restart_trace = _event_trace(attacked_root)
    before_restart_invocations = generic_invocations(attacked_root)
    first_restart = _cold_restart(attacked_root, attacked_context.config)
    after_first_trace = _event_trace(attacked_root)
    second_restart = _cold_restart(attacked_root, attacked_context.config)
    after_second_trace = _event_trace(attacked_root)
    after_restart_invocations = generic_invocations(attacked_root)

    routes = [
        candidate
        for candidate in (
            first_route,
            first_restart.get("route"),
            first_restart.get("historical_route"),
            second_restart.get("route"),
            second_restart.get("historical_route"),
        )
        if isinstance(candidate, dict)
    ]
    diagnostic = {
        "attack": attack,
        "first_error": first_error,
        "first_result": first_result,
        "first_restart": first_restart,
        "second_restart": second_restart,
        "event_trace_before_restart": before_restart_trace,
        "event_trace_after_first_restart": after_first_trace,
        "event_trace_after_second_restart": after_second_trace,
        "physical_invocations_before": before_restart_invocations,
        "physical_invocations_after": after_restart_invocations,
    }
    assert routes and all(route["next_action"] == "BLOCK_INTEGRITY" for route in routes), (
        "pre-derive integrity failure has no SQLite-backed typed BLOCK_INTEGRITY route; "
        + json.dumps(diagnostic, sort_keys=True, default=str)
    )
    assert first_restart["route"] == first_restart["historical_route"]
    assert second_restart["route"] == first_restart["route"]
    assert second_restart["historical_route"] == first_restart["route"]
    assert before_restart_trace == after_first_trace == after_second_trace
    assert before_restart_invocations == after_restart_invocations
    assert len(before_restart_invocations) == 1
    assert all(record.get("phase") == "full" for record in before_restart_invocations)
    assert not any(event_type == "AttemptFinalized" for _, _, event_type, _ in after_second_trace)
    assert any(
        route["source"]["event_id"] in {event_id for _, event_id, _, _ in after_second_trace}
        for route in routes
    )
    assert (attacked_root / GENERIC_MARKER).is_file()
    assert canonical_hash(first_restart["route"]) == canonical_hash(second_restart["route"])
