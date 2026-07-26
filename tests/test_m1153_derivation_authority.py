from __future__ import annotations

import hashlib
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from auto_research.agents.experiment import ExperimentAgent
from auto_research.contract_store import ContractStore, canonical_contract_bytes
from auto_research.derivation_validation import validate_derive_receipt_precommit
from auto_research.research_state import (
    IntegrityError,
    ResearchEventLedger,
)
from auto_research.validators import run_stage_gate
from support.local_c2c_execution import (
    build_c2c_context,
    create_local_c2c_repo,
    install_fake_gpu,
)


@dataclass(frozen=True)
class _BaselineProject:
    root: Path
    config: dict[str, Any]


@pytest.fixture(scope="module")
def m1153_baseline(tmp_path_factory: pytest.TempPathFactory) -> _BaselineProject:
    fixture_root = tmp_path_factory.mktemp("m1153-derivation-authority")
    monkeypatch = pytest.MonkeyPatch()
    install_fake_gpu(fixture_root, monkeypatch)
    repo = create_local_c2c_repo(fixture_root, proxy_accuracy=0.51)
    project_root = fixture_root / "project"
    project_root.mkdir()
    context = build_c2c_context(project_root, repo, profile="standard")
    result = ExperimentAgent(context).run()
    assert result.get("route_outcome", {}).get("next_action") == "PROPOSE_NEXT_VARIANT", result
    _assert_valid_baseline(project_root, context.config)
    try:
        yield _BaselineProject(project_root, deepcopy(context.config))
    finally:
        monkeypatch.undo()


@pytest.fixture
def authoritative_project(
    tmp_path: Path,
    m1153_baseline: _BaselineProject,
) -> _BaselineProject:
    project_root = tmp_path / "project"
    shutil.copytree(m1153_baseline.root, project_root)
    config = deepcopy(m1153_baseline.config)
    config["project"]["workspace_root"] = str(project_root.parent)
    return _BaselineProject(project_root, config)


def test_physical_derivation_ref_is_the_final_evidence_manifest_ref(
    authoritative_project: _BaselineProject,
) -> None:
    root, config = authoritative_project.root, authoritative_project.config
    _assert_valid_baseline(root, config)
    physical_ref, _, _, _ = _physical_derivation(root)
    final_refs = {
        entry["derivation_ref"]["digest"]
        for entry in _final_evidence_manifest(root)["entries"]
    }

    assert final_refs == {physical_ref["digest"]}, (
        "the derive receipt structured manifest is not the final EvidenceManifest authority; "
        f"physical={physical_ref['digest']} final={sorted(final_refs)}"
    )


def test_final_derivation_uses_the_frozen_physical_command_exact_set(
    authoritative_project: _BaselineProject,
) -> None:
    root, config = authoritative_project.root, authoritative_project.config
    _assert_valid_baseline(root, config)
    _, physical_manifest, derive_record, _ = _physical_derivation(root)
    expected_sources = _source_identity(physical_manifest["source_commands"])
    final_manifests = _final_derivation_manifests(root)
    observed_sources = {
        manifest["derivation_id"]: _source_identity(manifest["source_commands"])
        for manifest in final_manifests
    }

    assert all(
        sources == expected_sources
        and derive_record["command"]["command_id"] not in {item[0] for item in sources}
        for sources in observed_sources.values()
    ), (
        "final derivation is a self/direct proof sourced by the derive command instead of the frozen "
        f"physical exact-set; derive={derive_record['command']['command_id']} "
        f"expected={expected_sources} observed={observed_sources}"
    )


def test_missing_physical_derivation_manifest_fails_state_rebuild_and_gate_without_repair(
    authoritative_project: _BaselineProject,
) -> None:
    root, config = authoritative_project.root, authoritative_project.config
    _assert_valid_baseline(root, config)
    physical_ref, _, _, _ = _physical_derivation(root)
    physical_path = root / physical_ref["relative_path"]
    physical_path.unlink()
    cas_before = _cas_snapshot(root)

    outcomes = {
        reader: _authority_read(root, config, reader)
        for reader in ("state", "rebuild", "gate")
    }
    cas_after = _cas_snapshot(root)

    assert all(outcome != "PASS" for outcome in outcomes.values()) and cas_after == cas_before, (
        "missing physical derivation manifest was ignored by an authority reader; "
        f"physical={physical_ref['digest']} outcomes={outcomes} "
        f"cas_added={sorted(set(cas_after) - set(cas_before))}"
    )


def test_missing_event_referenced_receipt_derivation_is_not_recreated_by_authority_reads(
    authoritative_project: _BaselineProject,
) -> None:
    root, config = authoritative_project.root, authoritative_project.config
    _assert_valid_baseline(root, config)
    direct_ref = _final_evidence_manifest(root)["entries"][0]["derivation_ref"]
    direct_path = root / direct_ref["relative_path"]
    observations: dict[str, dict[str, Any]] = {}

    for reader in ("state", "rebuild", "gate"):
        direct_path.unlink(missing_ok=True)
        cas_before = _cas_snapshot(root)
        outcome = _authority_read(root, config, reader)
        cas_after = _cas_snapshot(root)
        observations[reader] = {
            "outcome": outcome,
            "recreated": direct_path.exists(),
            "added": sorted(set(cas_after) - set(cas_before)),
            "changed": sorted(
                path for path in set(cas_before) & set(cas_after) if cas_before[path] != cas_after[path]
            ),
        }

    assert all(
        item["outcome"] != "PASS"
        and item["recreated"] is False
        and not item["added"]
        and not item["changed"]
        for item in observations.values()
    ), (
        "state/rebuild/Gate repaired an event-referenced receipt derivation instead of failing read-only; "
        f"direct={direct_ref['digest']} observations={observations}"
    )


@pytest.mark.parametrize(
    ("attack", "mutate", "reason"),
    [
        pytest.param(
            "derive-self-source",
            lambda sources, self_source: [self_source],
            "self source",
            id="derive-self-source",
        ),
        pytest.param(
            "missing-source",
            lambda sources, self_source: sources[:-1],
            "missing source",
            id="missing-source",
        ),
        pytest.param(
            "duplicate-source",
            lambda sources, self_source: sources + [deepcopy(sources[0])],
            "duplicate source",
            id="duplicate-source",
        ),
        pytest.param(
            "extra-source",
            lambda sources, self_source: sources + [_extra_source(sources[0])],
            "extra source",
            id="extra-source",
        ),
        pytest.param(
            "reordered-source",
            lambda sources, self_source: list(reversed(sources)),
            "source order",
            id="reordered-source",
        ),
    ],
)
def test_physical_derivation_source_exact_set_attack_is_rejected(
    authoritative_project: _BaselineProject,
    attack: str,
    mutate: Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]],
    reason: str,
) -> None:
    root, config = authoritative_project.root, authoritative_project.config
    _assert_valid_baseline(root, config)
    _, physical_manifest, derive_record, derive_receipt = _physical_derivation(root)
    ledger = ResearchEventLedger(root)
    state = ledger.state()
    attempt = next(iter(state["attempts"].values()))
    original = validate_derive_receipt_precommit(
        project_root=root,
        attempt=attempt,
        trial_spec=attempt["frozen_trial_spec"],
        phase_commands=state["phase_commands"],
        phase="full",
        derive_record=derive_record,
        receipt_ref=derive_record["receipt_ref"],
    )
    assert original.derivation_hash == derive_receipt["derivation_hash"]
    original_sources = deepcopy(physical_manifest["source_commands"])
    assert len(original_sources) >= 2, original_sources
    attacked_sources = mutate(original_sources, _derive_self_source(derive_record, derive_receipt))
    attacked_manifest = deepcopy(physical_manifest)
    attacked_manifest["source_commands"] = attacked_sources
    store = ContractStore(root)
    attacked_manifest_ref = store.put_json(
        attacked_manifest,
        schema_file="evidence_derivation_manifest_v3.schema.json",
    )
    attacked_receipt = deepcopy(derive_receipt)
    attacked_receipt["derivation_ref"] = attacked_manifest_ref
    attacked_receipt["derivation_hash"] = attacked_manifest_ref["digest"]
    attacked_receipt_ref = store.put_json(
        attacked_receipt,
        schema_file="phase_run_receipt_v5.schema.json",
    )
    assert attacked_receipt_ref["digest"] == hashlib.sha256(
        canonical_contract_bytes(attacked_receipt)
    ).hexdigest()

    before_events = ledger.events()
    with pytest.raises(ValueError, match=reason):
        validate_derive_receipt_precommit(
            project_root=root,
            attempt=attempt,
            trial_spec=attempt["frozen_trial_spec"],
            phase_commands=state["phase_commands"],
            phase="full",
            derive_record=derive_record,
            receipt_ref=attacked_receipt_ref,
        )
    assert ledger.events() == before_events
    assert _authority_read(root, config, "state") == "PASS"


def _assert_valid_baseline(root: Path, config: dict[str, Any]) -> None:
    ledger = ResearchEventLedger(root)
    cas_before = _cas_snapshot(root)
    state = ledger.state()
    rebuilt = ledger.rebuild()
    gate = run_stage_gate("S3_experiment", root, config).to_dict()
    assert state["last_sequence"] == rebuilt["last_sequence"]
    assert len(state["trial_results"]) == 1
    assert gate["status"] == "PASS", gate
    assert _cas_snapshot(root) == cas_before, "an intact authority read unexpectedly mutated CAS"


def _physical_derivation(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    ledger = ResearchEventLedger(root)
    state = ledger.state()
    matches = [
        record
        for record in state["phase_commands"].values()
        if record["status"] == "completed"
        and record["command"]["phase"] == "full"
        and record["command"]["command_spec_id"] == "full-derive-evidence"
    ]
    assert len(matches) == 1, matches
    record = matches[0]
    store = ContractStore(root)
    receipt = store.read_json(record["receipt_ref"], schema_file="phase_run_receipt_v5.schema.json")
    physical_ref = receipt["derivation_ref"]
    assert isinstance(physical_ref, dict)
    assert receipt["derivation_hash"] == physical_ref["digest"]
    physical_manifest = store.read_json(
        physical_ref,
        schema_file="evidence_derivation_manifest_v3.schema.json",
    )
    return physical_ref, physical_manifest, record, receipt


def _final_evidence_manifest(root: Path) -> dict[str, Any]:
    state = ResearchEventLedger(root).state()
    assert len(state["trial_results"]) == 1
    return deepcopy(next(iter(state["trial_results"].values()))["evidence_manifest"])


def _final_derivation_manifests(root: Path) -> list[dict[str, Any]]:
    store = ContractStore(root)
    return [
        store.read_json(entry["derivation_ref"], schema_file="evidence_derivation_manifest_v3.schema.json")
        for entry in _final_evidence_manifest(root)["entries"]
    ]


def _derive_self_source(
    record: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    output = receipt["outputs"][0]
    return {
        "source_ordinal": 0,
        "command_id": record["command"]["command_id"],
        "command_spec_id": record["command"]["command_spec_id"],
        "command_hash": record["command"]["command_hash"],
        "completed_event_id": record["completed_event_id"],
        "receipt_ref": deepcopy(record["receipt_ref"]),
        "receipt_hash": record["receipt_ref"]["digest"],
        "output_id": output["output_id"],
        "output_kind": output["kind"],
        "output_schema_version": output["schema_version"],
        "output_ref": deepcopy(output["contract_ref"]),
        "output_hash": output["content_hash"],
        "authority_roles": ["normalized_evidence_source"],
        "readiness_check_ids": [],
    }


def _extra_source(source: dict[str, Any]) -> dict[str, Any]:
    extra = deepcopy(source)
    extra["command_id"] = "cmd-unregistered-extra-source"
    return extra


def _authority_read(root: Path, config: dict[str, Any], reader: str) -> str:
    try:
        if reader == "state":
            ResearchEventLedger(root).state()
            return "PASS"
        if reader == "rebuild":
            ResearchEventLedger(root).rebuild()
            return "PASS"
        if reader == "gate":
            report = run_stage_gate("S3_experiment", root, config).to_dict()
            return "PASS" if report["status"] == "PASS" else f"REJECTED:{report}"
    except (IntegrityError, OSError, TypeError, ValueError) as exc:
        return f"REJECTED:{type(exc).__name__}:{exc}"
    raise AssertionError(f"unknown authority reader: {reader}")


def _source_identity(
    sources: list[dict[str, Any]],
) -> tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...]:
    return tuple(
        (
            str(item["command_id"]),
            str(item["receipt_hash"]),
            str(item["output_hash"]),
            tuple(item["authority_roles"]),
            tuple(item["readiness_check_ids"]),
        )
        for item in sources
    )


def _cas_snapshot(root: Path) -> dict[str, str]:
    contract_root = root / "meta" / "contracts" / "sha256"
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(contract_root.rglob("*.json"))
    }
