from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.agents.plan import _trial_spec_from_plan
from auto_research.contract_store import ContractStore
from auto_research.domain_contracts import canonical_hash
from auto_research.research_state import IntegrityError, ResearchEventLedger
from test_m112_ledger_authority import _direction, _variant


def _plan() -> dict:
    return {
        "hypotheses": [{"id": "H1", "statement": "candidate improves accuracy"}],
        "baselines": [{"name": "baseline"}],
        "datasets": [{"name": "dataset-a", "split": "validation", "sample_count": 1}],
        "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [7], "aggregation": "mean", "require_complete_seed_coverage": True},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 0.0},
        "ablation_matrix": [],
        "execution": {"mode": "simulate", "collector": "generic", "commands": []},
        "resource_budget": {"wall_clock_minutes": 5},
    }


def _prepared(tmp_path: Path) -> tuple[ResearchEventLedger, dict, dict, dict]:
    direction = _direction("contract-direction")
    variant = _variant(direction, "contract-variant")
    trial_spec = _trial_spec_from_plan(_plan(), variant, project_root=tmp_path)
    ledger = ResearchEventLedger(tmp_path)
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    return ledger, direction, variant, trial_spec


def _reserve(ledger: ResearchEventLedger, direction: dict, variant: dict, trial_spec: dict) -> dict:
    return ledger.reserve_attempt(
        profile="standard",
        direction=direction,
        variant=variant,
        implementation_hash=canonical_hash({"implementation": "contract-reservation"}),
        attempt_kind="full",
        trial_spec=trial_spec,
    )


def test_plan_contract_refs_are_immutable_and_reservation_rebuilds(tmp_path: Path) -> None:
    ledger, direction, variant, trial_spec = _prepared(tmp_path)
    sample_ref = trial_spec["sample_manifest_ref"]
    evaluator_ref = trial_spec["execution_contract"]["evaluator_manifest_ref"]
    assert sample_ref["contract_kind"] == "sample_manifest"
    assert evaluator_ref["contract_kind"] == "evaluator_manifest"
    assert "artifact_path" not in trial_spec["sample_manifest"]
    assert "artifact_path" not in trial_spec["execution_contract"]["evaluator_provenance"]
    store = ContractStore(tmp_path)
    assert store.read_contract(sample_ref, contract_kind="sample_manifest", schema_file="sample_manifest_v2.schema.json") == trial_spec["sample_manifest"]
    assert store.read_contract(evaluator_ref, contract_kind="evaluator_manifest", schema_file="evaluator_manifest_v2.schema.json") == trial_spec["execution_contract"]["evaluator_provenance"]

    attempt = _reserve(ledger, direction, variant, trial_spec)
    rebuilt = ledger.rebuild()
    assert rebuilt["attempts"][attempt["attempt_id"]]["trial_spec_hash"] == attempt["trial_spec_hash"]


def test_contract_ref_content_drift_rejects_reservation_without_event(tmp_path: Path) -> None:
    ledger, direction, variant, trial_spec = _prepared(tmp_path)
    before = ledger.state()
    path = tmp_path / trial_spec["sample_manifest_ref"]["blob"]["relative_path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(IntegrityError, match="ContractRef rejected"):
        _reserve(ledger, direction, variant, trial_spec)
    after = ledger.state()
    assert after["last_sequence"] == before["last_sequence"]
    assert after["attempts"] == before["attempts"]


@pytest.mark.parametrize("attack", ["ancestor_symlink", "leaf_symlink", "hard_link"])
def test_contract_path_attacks_reject_reservation_without_event(tmp_path: Path, attack: str) -> None:
    ledger, direction, variant, trial_spec = _prepared(tmp_path)
    before = ledger.state()
    reference = trial_spec["sample_manifest_ref"]["blob"]
    path = tmp_path / reference["relative_path"]
    if attack == "ancestor_symlink":
        contracts = tmp_path / "meta" / "contracts"
        outside = tmp_path / "outside-contracts"
        contracts.rename(outside)
        os.symlink(outside, contracts)
    elif attack == "leaf_symlink":
        outside = tmp_path / "outside.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        os.symlink(outside, path)
    else:
        os.link(path, tmp_path / "contract-hard-link.json")
    with pytest.raises(IntegrityError, match="ContractRef rejected"):
        _reserve(ledger, direction, variant, trial_spec)
    after = ledger.state()
    assert after["last_sequence"] == before["last_sequence"]
    assert after["attempts"] == before["attempts"]


def test_rebuild_rejects_contract_tamper_after_reservation(tmp_path: Path) -> None:
    ledger, direction, variant, trial_spec = _prepared(tmp_path)
    _reserve(ledger, direction, variant, trial_spec)
    path = tmp_path / trial_spec["execution_contract"]["evaluator_manifest_ref"]["blob"]["relative_path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(IntegrityError, match="ContractRef rejected"):
        ledger.rebuild()


def test_trial_spec_contract_ref_cannot_point_to_different_valid_manifest(tmp_path: Path) -> None:
    ledger, direction, variant, trial_spec = _prepared(tmp_path)
    baseline = deepcopy(trial_spec)
    modified = deepcopy(trial_spec["sample_manifest"])
    modified["manifest_id"] = "different-sample-manifest"
    modified_ref = ContractStore(tmp_path).put_contract(
        modified,
        contract_kind="sample_manifest",
        schema_file="sample_manifest_v2.schema.json",
    )
    trial_spec["sample_manifest_ref"] = modified_ref
    before = ledger.state()
    with pytest.raises(IntegrityError, match="content mismatch"):
        _reserve(ledger, direction, variant, trial_spec)
    after = ledger.state()
    assert after["last_sequence"] == before["last_sequence"]
    assert after["attempts"] == before["attempts"]
    assert baseline["sample_manifest_ref"] != modified_ref
