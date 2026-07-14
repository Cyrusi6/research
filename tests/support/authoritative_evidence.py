from __future__ import annotations

import hashlib
from pathlib import Path

from auto_research.evidence import content_addressed_evidence_path, encode_canonical_evidence


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
) -> dict:
    """Write strict row evidence and return the public CompletionEvidence request."""

    producer_run_id = producer_run_id or f"producer-{attempt['attempt_id'][:8]}"
    evidence_id = f"evidence:{attempt['attempt_id'][:8]}:main"
    rows = [
        {
            "phase": phase,
            "role": role,
            "command_status": "completed",
            "dataset_id": dataset_id,
            "metric_id": metric_id,
            "metric_value": value,
            "sample_manifest_hash": attempt["sample_manifest_hash"],
            "evaluator_hash": attempt["evaluator_hash"],
            "seed": seed,
            "attempt_id": attempt["attempt_id"],
            "variant_semantic_hash": attempt["variant_semantic_hash"],
            "variant_spec_hash": attempt["variant_spec_hash"],
            "trial_spec_hash": attempt["trial_spec_hash"],
            "producer_run_id": producer_run_id,
        }
        for role, value in role_values.items()
    ]
    payload = {
        "schema_version": "auto_research_main_results_v2",
        "evidence_kind": "main_results",
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
        "cross_references": {},
        "rows": rows,
    }
    raw = encode_canonical_evidence(payload)
    content_hash = hashlib.sha256(raw).hexdigest()
    relative_path = content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=producer_run_id,
        evidence_kind="main_results",
        content_hash=content_hash,
    )
    artifact = project_root / relative_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(raw)
    return {
        "schema_version": "auto_research_completion_evidence_v1",
        "attempt_id": attempt["attempt_id"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "entries": [
            {
                "evidence_id": evidence_id,
                "kind": "main_results",
                "relative_path": relative_path,
                "content_hash": content_hash,
                "schema_version": "auto_research_main_results_v2",
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
            }
        ],
    }
