from __future__ import annotations

from copy import deepcopy

import pytest

from auto_research.agents.plan import _trial_spec_from_plan
from auto_research.domain_contracts import canonical_hash, validate_trial_spec
from auto_research.phase_command_plan import (
    build_phase_command_plan,
    phase_command_plan_for_phase,
    store_phase_command_plan,
    validate_phase_command_plan,
)


def _outputs() -> list[dict]:
    return [
        {
            "kind": "main_results",
            "schema_version": "auto_research_main_results_v3",
            "required": True,
        }
    ]


def _coverage() -> dict:
    return {
        "mode": "exact_cartesian",
        "datasets": ["fixture-dataset"],
        "seeds": [7],
        "metrics": ["score"],
        "roles": ["baseline", "candidate"],
    }


def _producer_command(tmp_path, argv: list[str]) -> dict:
    return {
        "argv": argv,
        "cwd": str(tmp_path),
        "physical_raw_outputs": [
            {
                "output_id": "raw-main-results",
                "kind": "raw_main_results",
                "schema_version": "auto_research_main_results_v3",
                "locator": "runner/main-results.json",
                "locator_type": "file",
                "dataset_id": None,
                "role": None,
                "required": True,
                "normalized_kinds": ["main_results"],
            }
        ],
    }


def test_phase_command_plan_is_content_addressed_and_ordered(tmp_path) -> None:
    source_hash = canonical_hash({"source": "snapshot"})
    plan = build_phase_command_plan(
        phase="full",
        adapter_id="generic-external-adapter",
        adapter_version="1",
        provenance_mode="local-external",
        variant_spec_hash=canonical_hash({"variant": "one"}),
        source_snapshot_hash=source_hash,
        command_values=[
            {"argv": ["python", "prepare.py"], "cwd": str(tmp_path)},
            _producer_command(tmp_path, ["python", "producer.py"]),
        ],
        expected_evidence=_outputs(),
        default_cwd=str(tmp_path),
        project_root=tmp_path,
        coverage_contract=_coverage(),
    )
    reference, digest = store_phase_command_plan(tmp_path, plan)
    assert reference["digest"] == digest == canonical_hash(plan)
    assert plan["commands"][1]["dependencies"] == [plan["commands"][0]["command_spec_id"]]
    assert plan["commands"][1]["physical_raw_outputs"][0]["normalized_kinds"] == ["main_results"]
    assert plan["commands"][2]["dependencies"] == [plan["commands"][1]["command_spec_id"]]
    assert [item["kind"] for item in plan["commands"][2]["expected_outputs"]] == ["main_results"]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda value: value["commands"][1].__setitem__("ordinal", 9), "ordinals"),
        (lambda value: value["commands"][0].__setitem__("dependencies", [value["commands"][1]["command_spec_id"]]), "precede"),
        (lambda value: value["commands"][1].__setitem__("phase", "proxy"), "phase mismatch"),
        (lambda value: value["commands"][1].__setitem__("source_snapshot_hash", "0" * 64), "source snapshot"),
    ],
)
def test_phase_command_plan_rejects_dag_and_identity_attacks(tmp_path, mutate, message) -> None:
    plan = build_phase_command_plan(
        phase="full",
        adapter_id="generic-external-adapter",
        adapter_version="1",
        provenance_mode="local-external",
        variant_spec_hash=canonical_hash({"variant": "one"}),
        source_snapshot_hash=canonical_hash({"source": "snapshot"}),
        command_values=[
            {"argv": ["python", "prepare.py"], "cwd": str(tmp_path)},
            _producer_command(tmp_path, ["python", "producer.py"]),
        ],
        expected_evidence=_outputs(),
        default_cwd=str(tmp_path),
        project_root=tmp_path,
        coverage_contract=_coverage(),
    )
    attacked = deepcopy(plan)
    mutate(attacked)
    with pytest.raises(ValueError, match=message):
        validate_phase_command_plan(attacked, expected_evidence_kinds=["main_results"])


def test_trial_spec_v8_embeds_nonempty_frozen_synthetic_plan(tmp_path) -> None:
    variant = {
        "variant_id": "variant-command-plan",
        "variant_spec_hash": canonical_hash({"variant": "command-plan"}),
        "implementation_surface_ids": ["surface-main"],
    }
    plan = {
        "datasets": [
            {
                "name": "fixture-dataset",
                "split": "test",
                "sample_count": 1,
            }
        ],
        "metrics": [{"name": "score", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [7]},
        "execution": {
            "mode": "simulate",
            "collector": "generic",
            "commands": [],
            "evaluator_source_payloads": [{"source": "synthetic"}],
        },
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 0.0},
        "ablation_matrix": [],
    }
    trial_spec = _trial_spec_from_plan(plan, variant, project_root=tmp_path)
    validate_trial_spec(trial_spec)
    command_plan = phase_command_plan_for_phase(trial_spec, "full")
    assert trial_spec["schema_version"] == "auto_research_trial_spec_v8"
    assert command_plan["commands"][0]["argv"] == ["auto-research-adapter", "synthetic", "full"]
    assert trial_spec["phase_contracts"][0]["command_plan_hash"] == canonical_hash(command_plan)

    attacked = deepcopy(trial_spec)
    attacked["phase_contracts"][0]["command_plan"]["commands"][0]["argv"] = ["python", "-c", "pass"]
    with pytest.raises(ValueError, match="command_plan_hash mismatch"):
        validate_trial_spec(attacked)


def test_generic_real_phase_requires_explicit_command(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires at least one frozen command"):
        build_phase_command_plan(
            phase="full",
            adapter_id="generic-external-adapter",
            adapter_version="1",
            provenance_mode="local-external",
            variant_spec_hash=canonical_hash({"variant": "one"}),
            source_snapshot_hash=canonical_hash({"source": "snapshot"}),
            command_values=[],
            expected_evidence=_outputs(),
            default_cwd=str(tmp_path),
            project_root=tmp_path,
            coverage_contract=_coverage(),
        )


def test_phase_command_plan_rejects_removed_command_result_evidence(tmp_path) -> None:
    with pytest.raises(ValueError):
        build_phase_command_plan(
            phase="full",
            adapter_id="generic-external-adapter",
            adapter_version="1",
            provenance_mode="local-external",
            variant_spec_hash=canonical_hash({"variant": "one"}),
            source_snapshot_hash=canonical_hash({"source": "snapshot"}),
            command_values=[["python", "producer.py"]],
            expected_evidence=[
                {
                    "kind": "command_result_evidence",
                    "schema_version": "auto_research_command_result_evidence_v1",
                    "required": True,
                }
            ],
            default_cwd=str(tmp_path),
            project_root=tmp_path,
            coverage_contract=_coverage(),
        )


def test_generic_command_plan_freezes_exact_argv_cwd_and_source(tmp_path) -> None:
    source_hash = canonical_hash({"source": "generic-producer"})
    command = {
        "argv": ["python", "producer.py", "--manifest", "runner/phase.json"],
        "cwd": str(tmp_path / "worktree"),
    }
    plan = build_phase_command_plan(
        phase="full",
        adapter_id="generic-external-adapter",
        adapter_version="1",
        provenance_mode="local-external",
        variant_spec_hash=canonical_hash({"variant": "one"}),
        source_snapshot_hash=source_hash,
        command_values=[_producer_command(tmp_path / "worktree", command["argv"])],
        expected_evidence=_outputs(),
        default_cwd=str(tmp_path),
        project_root=tmp_path,
        coverage_contract=_coverage(),
    )

    frozen = plan["commands"][0]
    assert frozen["argv"] == command["argv"]
    assert frozen["cwd"] == command["cwd"]
    assert frozen["source_snapshot_hash"] == source_hash


def test_phase_command_plan_rejects_symlinked_cwd_component(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        build_phase_command_plan(
            phase="full",
            adapter_id="generic-external-adapter",
            adapter_version="1",
            provenance_mode="local-external",
            variant_spec_hash=canonical_hash({"variant": "one"}),
            source_snapshot_hash=canonical_hash({"source": "snapshot"}),
            command_values=[{**_producer_command(linked, ["python", "producer.py"]), "cwd": str(linked)}],
            expected_evidence=_outputs(),
            default_cwd=str(tmp_path),
            project_root=tmp_path,
            coverage_contract=_coverage(),
        )
