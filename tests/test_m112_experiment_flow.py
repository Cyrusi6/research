from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.agents import experiment as experiment_module
from auto_research.agents.experiment import ExperimentAgent
from auto_research.agents.plan import _trial_spec_from_plan
from auto_research.domain_contracts import TRIAL_SPEC_SCHEMA_VERSION, validate_trial_spec


def _variant() -> dict:
    return {
        "direction_id": "direction-1",
        "direction_semantic_hash": "d" * 64,
        "direction_spec_hash": "e" * 64,
        "variant_id": "variant-1",
        "variant_semantic_hash": "v" * 64,
        "variant_spec_hash": "f" * 64,
        "ablation": {"remove_core": True},
        "expected_metric_signature": {"primary_metric": "accuracy"},
    }


def _plan() -> dict:
    return {
        "hypotheses": [{"id": "H1", "statement": "candidate improves accuracy"}],
        "baselines": [{"name": "baseline"}],
        "datasets": [{"name": "dataset-a", "split": "validation"}],
        "metrics": [
            {"name": "accuracy", "primary": True, "higher_is_better": True},
            {"name": "latency", "primary": False, "higher_is_better": False},
        ],
        "statistical_testing": {"seeds": [7, 11], "aggregation": "mean", "require_complete_seed_coverage": True},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 0.05},
        "ablation_matrix": [{"id": "remove-core", "required": True}],
        "execution": {"mode": "simulate", "collector": "generic", "commands": []},
        "resource_budget": {"wall_clock_minutes": 20},
    }


def _attempt(trial_spec: dict) -> dict:
    return {
        "attempt_id": "attempt-1",
        "attempt_kind": "proxy_full",
        "state": "READY",
        "lifecycle_generation": 0,
        "implementation_hash": "1" * 64,
        "attempt_input_hash": "2" * 64,
        "sample_manifest_hash": "3" * 64,
        "evaluator_hash": "4" * 64,
        "seeds": trial_spec["statistical_testing"]["seeds"],
        "phases": {"proxy": "PENDING", "full": "PENDING"},
    }


def test_plan_builds_complete_frozen_trial_spec(tmp_path: Path) -> None:
    trial_spec = _trial_spec_from_plan(_plan(), _variant(), project_root=tmp_path)

    assert trial_spec["schema_version"] == TRIAL_SPEC_SCHEMA_VERSION
    assert trial_spec["primary_metric_id"] == "accuracy"
    assert trial_spec["protocol"]["required_phases"] == ["full"]
    assert trial_spec["protocol"]["terminal_phases"] == ["full"]
    assert trial_spec["required_roles"] == ["baseline", "candidate", "ablation"]
    assert {item["kind"] for item in trial_spec["acceptance_constraints"]} >= {
        "minimum_mean_delta",
        "per_dataset_maximum_regression",
        "required_ablation_contrast",
    }
    assert [item["dataset_id"] for item in trial_spec["sample_manifest"]["datasets"]] == ["dataset-a"]
    assert trial_spec["sample_manifest"]["datasets"][0]["ordered_sample_ids"]
    assert trial_spec["sample_manifest"]["datasets"][0]["content_digest"]
    assert trial_spec["sample_manifest_ref"]["contract_kind"] == "sample_manifest"
    assert trial_spec["execution_contract"]["evaluator_manifest_ref"]["contract_kind"] == "evaluator_manifest"
    assert trial_spec["datasets"][0]["dataset_id"] == "dataset-a"
    assert trial_spec["required_artifacts"] == ["main_results", "ablation_results"]
    assert {item["kind"] for item in trial_spec["evidence_requirements"]} >= {"main_results", "activation_evidence", "ablation_results"}
    validate_trial_spec(trial_spec)


def test_trial_spec_authoritative_fields_are_deep_copied(tmp_path: Path) -> None:
    plan = _plan()
    trial_spec = _trial_spec_from_plan(plan, _variant(), project_root=tmp_path)
    plan["datasets"][0]["name"] = "mutated"
    plan["statistical_testing"]["seeds"].append(99)

    assert trial_spec["datasets"][0]["dataset_id"] == "dataset-a"
    assert trial_spec["statistical_testing"]["seeds"] == [7, 11]


def test_callers_cannot_construct_typed_observations_from_summary_results() -> None:
    assert not hasattr(experiment_module, "_typed_execution_observations")


class _ResumeLedger:
    def __init__(self, attempt: dict) -> None:
        self.calls: list[tuple] = []
        self.attempt = deepcopy(attempt)

    def state(self) -> dict:
        return {"attempts": {self.attempt["attempt_id"]: deepcopy(self.attempt)}}

    def events(self) -> list[dict]:
        paused_phase = self.attempt["paused_phase"]
        phase_execution = self.attempt["phase_executions"][paused_phase]
        pause_evidence = {
            "attempt_id": self.attempt["attempt_id"],
            "failure_class": "resource_pause",
            "phase": paused_phase,
            "phase_execution_id": phase_execution["phase_execution_id"],
            "producer_run_id": phase_execution["producer_run_id"],
        }
        return [{"event_id": f"event:pause:{self.attempt['lifecycle_generation']}", "event_type": "AttemptDispositioned", "payload": {"failure_evidence": pause_evidence}}]

    def resume_resource_attempt(self, attempt_id: str, *, measurement_provider=None) -> dict:
        self.calls.append(("resume", attempt_id))
        self.attempt["state"] = "READY"
        self.attempt["lifecycle_generation"] += 1
        return deepcopy(self.attempt)

    def start_proxy_phase(self, attempt_id: str, *, phase_execution_id: str, producer_run_id: str) -> dict:
        self.calls.append(("start_proxy", attempt_id, phase_execution_id, producer_run_id))
        self.attempt["state"] = "PROXY_RUNNING"
        self.attempt["phases"]["proxy"] = "RUNNING"
        self.attempt["phase_executions"]["proxy"] = {"phase_execution_id": phase_execution_id, "producer_run_id": producer_run_id}
        return deepcopy(self.attempt)

    def start_full_phase(self, attempt_id: str, *, phase_execution_id: str, producer_run_id: str) -> dict:
        self.calls.append(("start_full", attempt_id, phase_execution_id, producer_run_id))
        self.attempt["state"] = "FULL_RUNNING"
        self.attempt["phases"]["full"] = "RUNNING"
        return deepcopy(self.attempt)


def _resume_attempt(generation: int, *, state: str = "RESOURCE_PAUSED") -> dict:
    return {
        "attempt_id": "attempt-1",
        "attempt_kind": "proxy_full",
        "state": state,
        "lifecycle_generation": generation,
        "implementation_hash": "1" * 64,
        "attempt_input_hash": "2" * 64,
        "direction_semantic_hash": "3" * 64,
        "direction_spec_hash": "4" * 64,
        "variant_semantic_hash": "5" * 64,
        "variant_spec_hash": "6" * 64,
        "trial_spec_hash": "7" * 64,
        "protocol_hash": "8" * 64,
        "sample_manifest_hash": "9" * 64,
        "evaluator_hash": "a" * 64,
        "phases": {"proxy": "PENDING", "full": "PENDING"},
        "phase_executions": {"proxy": {"phase_execution_id": "phase-proxy-pause", "phase_start_event_id": "phase-start-pause", "producer_run_id": "pause-producer"}, "full": None},
        "paused_phase": "proxy",
    }


@pytest.mark.parametrize("generation", [0, 1, 2])
def test_resource_resume_precedes_execution_transition(tmp_path: Path, generation: int) -> None:
    attempt = _resume_attempt(generation)
    ledger = _ResumeLedger(attempt)
    agent = ExperimentAgent.__new__(ExperimentAgent)
    agent._resource_measurement_provider = lambda resource_type, resource_id, unit: 2

    resumed = agent._prepare_attempt_execution(
        ledger,
        attempt,
        project_root=tmp_path,
    )

    assert [call[0] for call in ledger.calls] == ["resume", "start_proxy"]
    assert ledger.calls[0][1] == attempt["attempt_id"]
    assert ledger.calls[1][0] == "start_proxy"
    assert resumed["state"] == "PROXY_RUNNING"


def test_ready_attempt_does_not_emit_resume(tmp_path: Path) -> None:
    attempt = _resume_attempt(3, state="READY")
    attempt["phases"]["proxy"] = "COMPLETED"
    ledger = _ResumeLedger(attempt)
    agent = ExperimentAgent.__new__(ExperimentAgent)
    agent._resource_measurement_provider = lambda resource_type, resource_id, unit: 2

    agent._prepare_attempt_execution(ledger, attempt, project_root=tmp_path)

    attempt["state"] = "PROXY_COMPLETED"
    assert [call[0] for call in ledger.calls] == ["start_full"]


def _authoritative_direction_and_variant():
    from auto_research.domain_contracts import build_direction_spec, build_variant_spec, canonical_hash

    direction = build_direction_spec(
        {
            "direction_id": "resource-direction",
            "research_question": "Does the registered method improve accuracy?",
            "mechanism_invariants": {
                "causal_hypothesis": "The intervention improves routing quality.",
                "target_mediator": "routing_quality",
                "invariants": ["same benchmark", "same mediator"],
            },
            "falsification_conditions": ["accuracy does not improve"],
            "support_claim_ids": ["support-1"],
            "counter_claim_ids": ["counter-1"],
            "implementation_surface_ids": ["src/router.py"],
            "metric_signature": {"primary": "accuracy", "direction": "increase"},
            "benchmark_contract_hash": canonical_hash({"datasets": ["dataset-a"]}),
            "variant_space": {"mutable_axes": ["intervention"], "immutable_axes": ["benchmark"], "forbidden_combinations": []},
            "s2_entry_conditions": ["gate pass"],
            "return_to_s1_conditions": ["five rejected outcomes"],
            "lineage": {"s1_run_id": "s1", "iteration": 1, "input_manifest_hash": canonical_hash({"input": 1})},
        }
    )
    variant = build_variant_spec(
        direction,
        {
            "variant_id": "resource-variant",
            "variation_coordinates": {"intervention": {"strength": 1}},
            "intervention": {"summary": "Apply routing", "algorithm_operations": ["apply-routing"], "configuration": {"strength": 1}},
            "hypothesis": "Routing improves accuracy.",
            "null_hypothesis": "Routing does not improve accuracy.",
            "alternative_hypothesis": "Routing improves accuracy.",
            "controlled_variables": {"dataset": "dataset-a"},
            "nuisance_variables": ["runtime noise"],
            "implementation_surface_ids": ["src/router.py"],
            "expected_metric_signature": {"primary": "accuracy", "direction": "increase"},
            "falsification_conditions": ["accuracy does not improve"],
            "ablation": {"switch": "disable-routing"},
            "resource_budget": {"max_wall_seconds": 60, "max_retries": 3},
            "failure_routing": {"implementation": "REPAIR_IMPLEMENTATION", "method": "PROPOSE_NEXT_VARIANT"},
            "lineage": {"s2_run_id": "s2", "iteration": 1, "direction_spec_hash": direction["direction_spec_hash"], "feedback_from_attempt_ids": []},
        },
    )
    return direction, variant


def test_real_experiment_agent_three_resource_resumes_then_single_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import MethodType

    from auto_research.agents.base import AgentContext
    from auto_research.artifacts import ArtifactManager
    from auto_research.llm import ModelClient
    from auto_research.research_state import ResearchEventLedger
    from auto_research.utils import write_json

    direction, variant = _authoritative_direction_and_variant()
    plan = _plan()
    plan["ablation_matrix"] = []
    plan["statistical_testing"] = {"seeds": [7], "method": "none", "require_complete_seed_coverage": True}
    project_root = tmp_path / "project"
    trial_spec = _trial_spec_from_plan(plan, variant, project_root=project_root)
    write_json(project_root / "literature" / "direction.json", direction)
    write_json(project_root / "plan" / "variant.json", variant)
    write_json(project_root / "plan" / "trial_spec.json", trial_spec)
    write_json(
        project_root / "plan" / "code_patches" / "implementation_contract.json",
        {"schema_version": "implementation_contract_test_v1", "variant_spec_hash": variant["variant_spec_hash"]},
    )
    write_json(
        project_root / "plan" / "code_patches" / "patch_manifest.json",
        {"schema_version": "auto_research_patch_manifest_v1", "status": "disabled", "selected_candidate_id": variant["variant_id"], "variant_spec_hash": variant["variant_spec_hash"]},
    )
    write_json(
        project_root / "plan" / "code_patches" / "patch_gate_report.json",
        {"schema_version": "auto_research_patch_gate_v1", "gate": "pass", "variant_id": variant["variant_id"], "variant_spec_hash": variant["variant_spec_hash"], "checks": {"activation": True}},
    )
    config = {"experiment": {"simulate": True, "random_seeds": [7]}, "orchestration": {"profile": "standard"}, "llm": {"use_real_api": False}}
    artifacts = ArtifactManager(project_root)
    context = AgentContext(project_root, config, artifacts, ModelClient(config, project_root=project_root))
    agent = ExperimentAgent(
        context,
        resource_measurement_provider=lambda resource_type, resource_id, unit: 1,
    )
    monkeypatch.setattr(
        agent.runner,
        "env_report",
        lambda: {
            "python": "test",
            "tmux": False,
            "gpu": [],
            "resource_probe": {
                "resource_type": "quota",
                "resource_id": "simulated-execution-slot",
                "required_capacity": 1,
                "observed_capacity": 1,
                "unit": "count",
                "probe_status": "available",
                "observed_at": "2026-07-14T00:01:00Z",
            },
        },
    )
    original_simulated = agent._run_simulated
    calls = {"count": 0}

    def resource_then_success(self, plan_payload, env_source, revision_source, *, attempt, trial_spec):
        calls["count"] += 1
        if calls["count"] <= 3:
            producer_run_id = attempt["phase_executions"]["full"]["producer_run_id"]
            probe = experiment_module._identity_evidence_payload(
                attempt=attempt,
                producer_run_id=producer_run_id,
                evidence_kind="resource_probe",
                phase="full",
                fields={
                    "resource_type": "quota",
                    "resource_id": "simulated-execution-slot",
                    "required_capacity": 1,
                    "observed_capacity": 0,
                    "unit": "count",
                    "probe_status": "insufficient",
                    "observed_at": f"2026-07-14T00:00:0{calls['count']}Z",
                },
            )
            probe["schema_version"] = "auto_research_resource_probe_evidence_v3"
            probe.pop("cross_references")
            relative_path = f"experiment/results/resource_pause_{calls['count']}.json"
            artifact_path = project_root / relative_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(experiment_module.encode_canonical_evidence(probe))
            return {
                "artifacts": [relative_path],
                "status": "resource_paused",
                "failure_class": "resource_pause",
                "failure_evidence": {
                    "command_status": "resource_paused",
                    "exit_code": 137,
                    "artifact_path": relative_path,
                    "producer_run_id": producer_run_id,
                    "reason": "Hermetic resource probe reported temporary exhaustion.",
                },
            }
        return original_simulated(
            plan_payload,
            env_source,
            revision_source,
            attempt=attempt,
            trial_spec=trial_spec,
        )

    agent._run_simulated = MethodType(resource_then_success, agent)
    results = [agent.run() for _ in range(4)]
    state = ResearchEventLedger(project_root).state()
    attempts = list(state["attempts"].values())

    assert [item["route_outcome"]["next_action"] for item in results[:3]] == ["PAUSE_RESOURCE"] * 3
    assert len(attempts) == 1
    assert attempts[0]["lifecycle_generation"] == 3
    assert attempts[0]["state"] == "METHOD_COMPLETED"
    assert len(state["trial_results"]) == 1
    assert state["directions"][direction["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 0, "consumed": 1}
