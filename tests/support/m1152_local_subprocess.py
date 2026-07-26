from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from auto_research.agents.base import AgentContext
from auto_research.agents.plan import _trial_spec_from_plan
from auto_research.artifacts import ArtifactManager
from auto_research.contract_store import ContractStore
from auto_research.domain_contracts import build_direction_spec, build_variant_spec, canonical_hash
from auto_research.llm import ModelClient
from auto_research.research_state import ResearchEventLedger
from auto_research.utils import write_json
from support.local_c2c_execution import (
    build_c2c_context as _build_c2c_context,
    create_local_c2c_repo as _create_local_c2c_repo,
    install_fake_gpu,
    invocation_records as c2c_invocation_records,
)


GENERIC_MARKER = "runner/local_subprocess_invocations.jsonl"


def create_generic_project(root: Path) -> tuple[AgentContext, dict[str, Any], dict[str, Any]]:
    direction = build_direction_spec(
        {
            "direction_id": "generic-local-direction",
            "research_question": "Does a local operation improve paired accuracy?",
            "mechanism_invariants": {
                "causal_hypothesis": "the operation changes prediction quality",
                "target_mediator": "prediction_quality",
                "invariants": ["fixed samples", "fixed evaluator"],
            },
            "falsification_conditions": ["paired accuracy does not improve"],
            "support_claim_ids": ["support-local"],
            "counter_claim_ids": ["counter-local"],
            "implementation_surface_ids": ["src/router.py"],
            "metric_signature": {"primary": "accuracy", "direction": "increase"},
            "benchmark_contract_hash": canonical_hash({"dataset": "dataset-a"}),
            "variant_space": {
                "mutable_axes": ["strength"],
                "immutable_axes": ["samples", "evaluator"],
                "forbidden_combinations": [],
            },
            "s2_entry_conditions": ["gate"],
            "return_to_s1_conditions": ["five rejects"],
            "lineage": {
                "s1_run_id": "s1-generic-local",
                "iteration": 1,
                "input_manifest_hash": canonical_hash({"input": "local"}),
            },
        }
    )
    variant = generic_variant(direction, 1)
    _write_generic_inputs(root, direction, variant)
    config = {
        "experiment": {"simulate": False},
        "orchestration": {"profile": "standard"},
        "llm": {"provider": "openai", "use_real_api": False, "model": "mock"},
    }
    context = AgentContext(
        root,
        config,
        ArtifactManager(root),
        ModelClient(config, project_root=root),
    )
    return context, direction, variant


def generic_variant(direction: dict[str, Any], index: int) -> dict[str, Any]:
    return build_variant_spec(
        direction,
        {
            "variant_id": f"generic-local-variant-{index}",
            "variation_coordinates": {"strength": index},
            "intervention": {
                "summary": f"Apply local operation strength {index}",
                "algorithm_operations": [f"apply-local-operation-{index}"],
                "configuration": {"strength": index},
            },
            "hypothesis": f"Strength {index} improves paired accuracy.",
            "null_hypothesis": f"Strength {index} does not improve paired accuracy.",
            "alternative_hypothesis": f"Strength {index} improves paired accuracy.",
            "controlled_variables": {"dataset": "dataset-a"},
            "nuisance_variables": ["local process noise"],
            "implementation_surface_ids": ["src/router.py"],
            "expected_metric_signature": {"primary": "accuracy", "direction": "increase"},
            "falsification_conditions": ["paired accuracy does not improve"],
            "ablation": {"switch": f"disable-strength-{index}"},
            "resource_budget": {"max_wall_seconds": 60, "max_retries": 1},
            "failure_routing": {
                "implementation": "REPAIR_IMPLEMENTATION",
                "method": "PROPOSE_NEXT_VARIANT",
            },
            "lineage": {
                "s2_run_id": f"s2-generic-local-{index}",
                "iteration": index,
                "direction_spec_hash": direction["direction_spec_hash"],
                "feedback_from_attempt_ids": [],
            },
        },
    )


def activate_generic_variant(root: Path, direction: dict[str, Any], index: int) -> dict[str, Any]:
    variant = generic_variant(direction, index)
    _write_generic_inputs(root, direction, variant)
    return variant


def create_c2c_project(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    profile: str,
    proxy_accuracy: float,
    name: str,
) -> tuple[Path, Path, AgentContext, dict[str, Any]]:
    install_fake_gpu(tmp_path, monkeypatch)
    repo = _create_local_c2c_repo(tmp_path / f"fixture-{name}", proxy_accuracy=proxy_accuracy)
    root = tmp_path / name
    root.mkdir()
    context = _build_c2c_context(root, repo, profile=profile)
    direction = json.loads((root / "literature" / "direction.json").read_text(encoding="utf-8"))
    return root, repo, context, direction


def activate_c2c_variant(
    root: Path,
    direction: dict[str, Any],
    index: int,
    *,
    profile: str = "standard",
) -> dict[str, Any]:
    variant = c2c_variant(direction, index)
    raw = (root / "samples" / "mmlu-redux.jsonl").read_bytes()
    trial_spec = _trial_spec_from_plan(
        {
            "datasets": [
                {
                    "name": "mmlu-redux",
                    "split": "test",
                    "sample_count": 1,
                    "source_revision": "local-c2c-v1",
                    "ordered_sample_ids": [hashlib.sha256(raw).hexdigest()],
                }
            ],
            "metrics": [{"name": "three_dataset_mean", "primary": True, "higher_is_better": True}],
            "statistical_testing": {
                "seeds": [42],
                "aggregation": "mean",
                "require_complete_seed_coverage": True,
            },
            "acceptance_criteria": {
                "minimum_mean_delta": 0.1,
                "maximum_dataset_regression": 2.0,
            },
            "ablation_matrix": [],
            "execution": {
                "mode": "real",
                "collector": "c2c_small_loop",
                "commands": [],
                "evaluator_id": "c2c-local",
                "evaluator_source_paths": ["evaluator.py"],
                "dependency_payloads": [{"name": "python", "version": "local"}],
            },
        },
        variant,
        profile=profile,
        project_root=root,
    )
    write_json(root / "plan" / "variant.json", variant)
    write_json(root / "plan" / "trial_spec.json", trial_spec)
    _write_implementation_projection(root, variant)
    return variant


def c2c_variant(direction: dict[str, Any], index: int) -> dict[str, Any]:
    mutable_axes = list((direction.get("variant_space") or {}).get("mutable_axes") or [])
    if len(mutable_axes) != 1:
        raise ValueError("local C2C fixture requires exactly one mutable routing axis")
    routing_axis = str(mutable_axes[0])
    variant = build_variant_spec(
        direction,
        {
            "variant_id": f"c2c-local-variant-{index}",
            "variation_coordinates": {routing_axis: index},
            "intervention": {
                "summary": f"Apply utility routing strength {index}",
                "algorithm_operations": [f"route-utility-{index}"],
                "configuration": {
                    "train": {"model": {"soft_alignment_top_k": index + 1}},
                    "eval": {"model": {"rosetta_config": {"soft_alignment_top_k": index + 1}}},
                },
            },
            "hypothesis": f"Utility routing {index} improves paired C2C accuracy.",
            "null_hypothesis": f"Utility routing {index} does not improve paired C2C accuracy.",
            "alternative_hypothesis": f"Utility routing {index} improves paired C2C accuracy.",
            "controlled_variables": {"dataset": "mmlu-redux"},
            "nuisance_variables": ["local process noise"],
            "implementation_surface_ids": ["rosetta/model/wrapper.py"],
            "expected_metric_signature": {"primary": "three_dataset_mean", "direction": "increase"},
            "falsification_conditions": ["paired accuracy does not improve"],
            "ablation": {"switch": f"disable-utility-{index}"},
            "resource_budget": {"max_wall_seconds": 60, "max_retries": 1},
            "failure_routing": {
                "implementation": "REPAIR_IMPLEMENTATION",
                "method": "PROPOSE_NEXT_VARIANT",
            },
            "lineage": {
                "s2_run_id": f"s2-c2c-production-{index}",
                "iteration": index,
                "direction_spec_hash": direction["direction_spec_hash"],
                "feedback_from_attempt_ids": [],
            },
        },
    )
    return variant


def generic_invocations(root: Path) -> list[dict[str, Any]]:
    marker = root / GENERIC_MARKER
    if not marker.exists():
        return []
    return [json.loads(line) for line in marker.read_text(encoding="utf-8").splitlines() if line.strip()]


def c2c_invocations(root: Path) -> list[dict[str, Any]]:
    return c2c_invocation_records(root)


def invocation_counts(records: list[dict[str, Any]], *, key: str = "kind") -> Counter[str]:
    return Counter(str(record[key]) for record in records)


def event_types(root: Path) -> list[str]:
    ledger = ResearchEventLedger(root)
    events = ledger.events()
    assert ledger.rebuild()["last_sequence"] == len(events)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    return [str(event["event_type"]) for event in events]


def direction_budget(root: Path) -> dict[str, int]:
    state = ResearchEventLedger(root).state()
    direction = next(iter(state["directions"].values()))
    return dict(direction["budget"])


def command_lineage(root: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    ledger = ResearchEventLedger(root)
    state = ledger.state()
    store = ContractStore(root)
    lineage: dict[str, tuple[str, tuple[str, ...]]] = {}
    for command_id, record in state["phase_commands"].items():
        if record["status"] != "completed":
            continue
        receipt = store.read_json(record["receipt_ref"], schema_file="phase_run_receipt_v5.schema.json")
        lineage[command_id] = (
            record["receipt_ref"]["digest"],
            tuple(output["contract_ref"]["digest"] for output in receipt["outputs"]),
        )
    return lineage


def assert_trial_lineage(root: Path) -> None:
    ledger = ResearchEventLedger(root)
    state = ledger.state()
    store = ContractStore(root)
    assert ledger.rebuild() == state
    for trial_result in state["trial_results"].values():
        for entry in trial_result["evidence_manifest"]["entries"]:
            receipt = store.read_json(entry["receipt_ref"], schema_file="phase_run_receipt_v5.schema.json")
            derivation = store.read_json(
                entry["derivation_ref"],
                schema_file="evidence_derivation_manifest_v3.schema.json",
            )
            assert any(output["contract_ref"]["digest"] == entry["content_hash"] for output in receipt["outputs"])
            normalized = next(
                output for output in derivation["normalized_outputs"] if output["kind"] == entry["kind"]
            )
            assert normalized["contract_ref"]["digest"] == entry["content_hash"]
            assert receipt["derivation_ref"] == entry["derivation_ref"]
            assert receipt["derivation_hash"] == entry["derivation_hash"]
            assert derivation["source_commands"]


def _write_generic_inputs(root: Path, direction: dict[str, Any], variant: dict[str, Any]) -> None:
    sample = root / "samples" / "dataset-a.jsonl"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample_bytes = b'{"id":"sample-1","label":1}\n'
    sample.write_bytes(sample_bytes)
    (root / "evaluator.py").write_text("def evaluate(baseline, candidate):\n    return candidate - baseline\n", encoding="utf-8")
    (root / "producer.py").write_text(_GENERIC_PRODUCER, encoding="utf-8")
    trial_spec = _trial_spec_from_plan(
        {
            "hypotheses": [{"id": "H1", "statement": "candidate improves accuracy"}],
            "baselines": [{"name": "baseline"}],
            "datasets": [
                {
                    "name": "dataset-a",
                    "split": "validation",
                    "sample_count": 1,
                    "source_revision": "local-generic-v1",
                    "ordered_sample_ids": [hashlib.sha256(sample_bytes).hexdigest()],
                }
            ],
            "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
            "statistical_testing": {
                "seeds": [7],
                "aggregation": "mean",
                "require_complete_seed_coverage": True,
            },
            "acceptance_criteria": {
                "minimum_mean_delta": 0.1,
                "maximum_dataset_regression": 0.0,
            },
            "ablation_matrix": [],
            "execution": {
                "mode": "real",
                "collector": "external_manifest",
                "commands": [{"argv": [sys.executable, "producer.py"]}],
                "workdir": str(root),
                "phase_manifest_path": "runner/phase_manifest.json",
                "evaluator_id": "generic-local-evaluator",
                "evaluator_source_paths": ["evaluator.py"],
                "dependency_payloads": [
                    {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}
                ],
            },
            "resource_budget": {"wall_clock_minutes": 5},
        },
        variant,
        project_root=root,
    )
    write_json(root / "literature" / "direction.json", direction)
    write_json(root / "plan" / "variant.json", variant)
    write_json(root / "plan" / "trial_spec.json", trial_spec)
    _write_implementation_projection(root, variant)


def _write_implementation_projection(root: Path, variant: dict[str, Any]) -> None:
    write_json(
        root / "plan" / "code_patches" / "implementation_contract.json",
        {
            "schema_version": "implementation_contract_test_v1",
            "variant_spec_hash": variant["variant_spec_hash"],
        },
    )
    write_json(
        root / "plan" / "code_patches" / "patch_manifest.json",
        {
            "schema_version": "auto_research_patch_manifest_v1",
            "status": "disabled",
            "selected_candidate_id": variant["variant_id"],
            "variant_spec_hash": variant["variant_spec_hash"],
        },
    )
    write_json(
        root / "plan" / "code_patches" / "patch_gate_report.json",
        {
            "schema_version": "auto_research_patch_gate_v1",
            "gate": "pass",
            "variant_id": variant["variant_id"],
            "variant_spec_hash": variant["variant_spec_hash"],
            "checks": {"activation": True},
        },
    )


_GENERIC_PRODUCER = r'''import json
from pathlib import Path

root = Path('.')
attempts = [json.loads(path.read_text()) for path in (root / 'meta' / 'attempts').glob('*.json')]
attempt = next(item for item in attempts if item['state'] == 'FULL_RUNNING')
phase = attempt['phase_executions']['full']
producer = phase['producer_run_id']
marker = root / 'runner' / 'local_subprocess_invocations.jsonl'
marker.parent.mkdir(parents=True, exist_ok=True)
with marker.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps({'attempt_id': attempt['attempt_id'], 'phase': 'full', 'producer_run_id': producer}, sort_keys=True) + '\n')
common = {
    'lifecycle_generation': attempt['lifecycle_generation'],
    'implementation_hash': attempt['implementation_hash'],
    'attempt_input_hash': attempt['attempt_input_hash'],
    'phase': 'full',
    'phase_execution_id': phase['phase_execution_id'],
    'phase_start_event_id': phase['phase_start_event_id'],
}
identity = {
    'attempt_id': attempt['attempt_id'],
    'producer_run_id': producer,
    'direction_semantic_hash': attempt['direction_semantic_hash'],
    'direction_spec_hash': attempt['direction_spec_hash'],
    'variant_semantic_hash': attempt['variant_semantic_hash'],
    'variant_spec_hash': attempt['variant_spec_hash'],
    'trial_spec_hash': attempt['trial_spec_hash'],
    'protocol_hash': attempt['protocol_hash'],
    'sample_manifest_hash': attempt['sample_manifest_hash'],
    'evaluator_hash': attempt['evaluator_hash'],
    'cross_references': {},
    **common,
}
rows = []
for role, value in [('baseline', 0.5), ('candidate', 0.8)]:
    rows.append({
        'phase': 'full',
        'role': role,
        'dataset_id': 'dataset-a',
        'metric_id': 'accuracy',
        'seed': 7,
        'metric_value': value,
        'command_status': 'completed',
        'attempt_id': attempt['attempt_id'],
        'variant_semantic_hash': attempt['variant_semantic_hash'],
        'variant_spec_hash': attempt['variant_spec_hash'],
        'trial_spec_hash': attempt['trial_spec_hash'],
        'sample_manifest_hash': attempt['sample_manifest_hash'],
        'evaluator_hash': attempt['evaluator_hash'],
        'producer_run_id': producer,
        **{key: value for key, value in common.items() if key != 'phase'},
    })
run_dir = root / 'runner' / attempt['attempt_id'] / producer
run_dir.mkdir(parents=True, exist_ok=True)
main_path = run_dir / 'main.json'
activation_path = run_dir / 'activation.json'
main_path.write_text(json.dumps({
    'schema_version': 'auto_research_main_results_v3',
    'evidence_kind': 'main_results',
    'evidence_id': 'main-' + producer,
    **identity,
    'rows': rows,
}, sort_keys=True, separators=(',', ':')))
activation_path.write_text(json.dumps({
    'schema_version': 'auto_research_activation_evidence_v4',
    'evidence_kind': 'activation_evidence',
    'evidence_id': 'activation-' + producer,
    **identity,
    'probe_id': 'forward-probe-' + producer,
    'status': 'activated',
    'command_status': 'completed',
    'exit_code': 0,
    'expected_surface_ids': ['src/router.py'],
    'observed_surface_ids': ['src/router.py'],
    'activation_delta_threshold': 0.01,
    'surface_measurements': [{
        'surface_id': 'src/router.py',
        'enabled_value': 1.0,
        'disabled_value': 0.0,
        'delta': 1.0,
        'threshold': 0.01,
        'status': 'ACTIVATED',
    }],
}, sort_keys=True, separators=(',', ':')))
manifest = {
    **phase,
    'sample_contract_ref': attempt['frozen_trial_spec']['sample_manifest_ref'],
    'evaluator_contract_ref': attempt['frozen_trial_spec']['execution_contract']['evaluator_manifest_ref'],
    'artifacts': [
        {'kind': 'main_results', 'source_path': str(main_path.relative_to(root)), 'producer_run_id': producer},
        {'kind': 'activation_evidence', 'source_path': str(activation_path.relative_to(root)), 'producer_run_id': producer},
    ],
}
(root / 'runner' / 'phase_manifest.json').write_text(json.dumps(manifest, sort_keys=True, separators=(',', ':')))
'''
