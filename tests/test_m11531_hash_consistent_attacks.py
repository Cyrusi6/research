from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from support.m11531_attack_helpers import (
    DerivationAttackBaseline,
    assert_candidate_rejected_without_writes,
    assert_completed_manifest_rejected_without_writes,
    build_derivation_attack_baseline,
    build_manifest_candidate,
    build_mismatched_evidence_manifest,
    clone_project,
    mutate_activation_from_nonactivation_authority,
    mutate_cross_attempt,
    mutate_cross_generation,
    mutate_cross_phase,
    mutate_cross_producer,
    mutate_duplicate_source,
    mutate_extra_readiness_check,
    mutate_extra_readiness_source,
    mutate_extra_source,
    mutate_missing_source,
    mutate_reordered_source,
    mutate_second_manifest,
    mutate_self_source,
    mutate_wrong_decoder_artifact,
    mutate_wrong_normalized_bytes,
    mutate_wrong_source_command_id,
    mutate_wrong_source_output_id,
)


ManifestMutator = Callable[[dict[str, Any], dict[str, Any], Any], None]


@dataclass(frozen=True)
class AttackCase:
    attack: str
    mutate: ManifestMutator
    expected_tokens: tuple[str, ...]


@pytest.fixture(scope="module")
def hash_consistent_baseline(
    tmp_path_factory: pytest.TempPathFactory,
) -> DerivationAttackBaseline:
    return build_derivation_attack_baseline(
        tmp_path_factory.mktemp("m11531-hash-consistent-attacks")
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            AttackCase("self-source", mutate_self_source, ("self", "source")),
            id="self-source",
        ),
        pytest.param(
            AttackCase("missing-source", mutate_missing_source, ("missing", "source")),
            id="missing-source",
        ),
        pytest.param(
            AttackCase("extra-source", mutate_extra_source, ("extra", "source")),
            id="extra-source",
        ),
        pytest.param(
            AttackCase("duplicate-source", mutate_duplicate_source, ("duplicate", "source")),
            id="duplicate-source",
        ),
        pytest.param(
            AttackCase("reordered-source", mutate_reordered_source, ("order", "source")),
            id="reordered-source",
        ),
        pytest.param(
            AttackCase("cross-attempt", mutate_cross_attempt, ("attempt_id", "mismatch")),
            id="cross-attempt",
        ),
        pytest.param(
            AttackCase("cross-phase", mutate_cross_phase, ("phase", "mismatch")),
            id="cross-phase",
        ),
        pytest.param(
            AttackCase(
                "cross-generation",
                mutate_cross_generation,
                ("lifecycle_generation", "mismatch"),
            ),
            id="cross-generation",
        ),
        pytest.param(
            AttackCase("cross-producer", mutate_cross_producer, ("producer_run_id", "mismatch")),
            id="cross-producer",
        ),
        pytest.param(
            AttackCase(
                "wrong-command-id",
                mutate_wrong_source_command_id,
                ("command", "source", "mismatch"),
            ),
            id="wrong-command-id",
        ),
        pytest.param(
            AttackCase(
                "wrong-output-id",
                mutate_wrong_source_output_id,
                ("output", "source", "mismatch"),
            ),
            id="wrong-output-id",
        ),
        pytest.param(
            AttackCase(
                "wrong-decoder-artifact",
                mutate_wrong_decoder_artifact,
                ("decoder", "frozen"),
            ),
            id="wrong-decoder-artifact",
        ),
        pytest.param(
            AttackCase(
                "raw-normalized-mismatch",
                mutate_wrong_normalized_bytes,
                ("normalized", "deterministic"),
            ),
            id="raw-normalized-mismatch",
        ),
        pytest.param(
            AttackCase(
                "second-hash-consistent-manifest",
                mutate_second_manifest,
                ("derivation", "manifest", "canonical"),
            ),
            id="second-hash-consistent-manifest",
        ),
        pytest.param(
            AttackCase(
                "extra-readiness-source",
                mutate_extra_readiness_source,
                ("readiness", "extra", "source"),
            ),
            id="extra-readiness-source",
        ),
        pytest.param(
            AttackCase(
                "extra-readiness-check",
                mutate_extra_readiness_check,
                ("readiness", "extra", "check"),
            ),
            id="extra-readiness-check",
        ),
        pytest.param(
            AttackCase(
                "activation-from-nonactivation-authority",
                mutate_activation_from_nonactivation_authority,
                ("activation", "authority"),
            ),
            id="activation-from-nonactivation-authority",
        ),
    ],
)
def test_hash_consistent_derive_candidate_attacks_are_zero_write_rejected(
    tmp_path: Path,
    hash_consistent_baseline: DerivationAttackBaseline,
    case: AttackCase,
) -> None:
    root = clone_project(
        hash_consistent_baseline.started_root,
        tmp_path / f"attack-{case.attack}",
    )
    candidate = build_manifest_candidate(
        root,
        case.mutate,
        attack=case.attack,
    )
    assert candidate.receipt_ref["digest"] != candidate.facts["valid_receipt_hash"]
    assert candidate.manifest_ref["digest"] != candidate.facts["valid_manifest_hash"]
    error = assert_candidate_rejected_without_writes(
        root,
        candidate,
        expected_tokens=case.expected_tokens,
    )
    assert case.attack.split("-")[0] in error.lower() or case.expected_tokens[0] in error.lower()


def test_evidence_manifest_must_reference_the_exact_manifest_committed_by_derive_receipt(
    tmp_path: Path,
    hash_consistent_baseline: DerivationAttackBaseline,
) -> None:
    root = clone_project(
        hash_consistent_baseline.completed_root,
        tmp_path / "attack-evidence-manifest-binding",
    )
    attacked = build_mismatched_evidence_manifest(root)
    error = assert_completed_manifest_rejected_without_writes(
        root,
        attacked,
        expected_tokens=("evidencemanifest", "different", "derivation"),
    )
    assert attacked["derivation_hash"] in str(attacked)
    assert "different" in error.lower()


def test_attack_fixture_legal_baseline_is_authoritative_before_any_mutation(
    hash_consistent_baseline: DerivationAttackBaseline,
) -> None:
    assert hash_consistent_baseline.started_root.is_dir()
    assert hash_consistent_baseline.completed_root.is_dir()
    assert hash_consistent_baseline.phase == "proxy"
    assert hash_consistent_baseline.derive_command_id
    assert deepcopy(hash_consistent_baseline.valid_receipt_ref)["digest"]
