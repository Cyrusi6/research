from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from auto_research.agents.base import AgentContext
from auto_research.agents.experiment import ExperimentAgent
from auto_research.agents.plan import _trial_spec_from_plan
from auto_research.artifacts import ArtifactManager
from auto_research.contract_store import ContractStore
from auto_research.domain_contracts import build_variant_spec
from auto_research.llm import ModelClient
from auto_research.research_state import ResearchEventLedger
from auto_research.validators import run_stage_gate
from auto_research.utils import write_json
from test_m114_authoritative_phase_transactions import _authoritative_direction_and_variant


@pytest.fixture(autouse=True)
def _hermetic_c2c_dataset_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hf_home = tmp_path / "hf-home"
    dataset_cache = hf_home / "datasets"
    for dataset_dir in ["mmlu-redux", "ai2-arc", "openbookqa"]:
        (dataset_cache / dataset_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(dataset_cache))


def _write_projection(root: Path, relative: str, payload: dict) -> None:
    write_json(root / relative, payload)


def _implementation_projection(root: Path, variant: dict) -> None:
    _write_projection(root, "plan/code_patches/implementation_contract.json", {
        "schema_version": "implementation_contract_test_v1",
        "variant_spec_hash": variant["variant_spec_hash"],
    })
    _write_projection(root, "plan/code_patches/patch_manifest.json", {
        "schema_version": "auto_research_patch_manifest_v1",
        "status": "disabled",
        "selected_candidate_id": variant["variant_id"],
        "variant_spec_hash": variant["variant_spec_hash"],
    })
    _write_projection(root, "plan/code_patches/patch_gate_report.json", {
        "schema_version": "auto_research_patch_gate_v1",
        "gate": "pass",
        "variant_id": variant["variant_id"],
        "variant_spec_hash": variant["variant_spec_hash"],
        "checks": {"activation": True},
    })


def _generic_project(root: Path) -> tuple[AgentContext, dict, dict]:
    direction, variant = _authoritative_direction_and_variant()
    sample = root / "samples/dataset-a.jsonl"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample_bytes = b'{"id":"sample-1","label":1}\n'
    sample.write_bytes(sample_bytes)
    (root / "evaluator.py").write_text("def evaluate(a, b):\n    return b-a\n", encoding="utf-8")
    plan = {
        "hypotheses": [{"id": "H1", "statement": "candidate improves accuracy"}],
        "baselines": [{"name": "baseline"}],
        "datasets": [{
            "name": "dataset-a", "split": "validation", "sample_count": 1,
            "source_revision": "local-fixture-v1",
            "ordered_sample_ids": [hashlib.sha256(sample_bytes).hexdigest()],
        }],
        "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [7], "aggregation": "mean", "require_complete_seed_coverage": True},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 0.0},
        "ablation_matrix": [],
        "execution": {
            "mode": "real", "collector": "external_manifest", "commands": [{"argv": [sys.executable, "producer.py"]}],
            "workdir": str(root), "phase_manifest_path": "runner/phase_manifest.json",
            "evaluator_id": "fixture-evaluator", "evaluator_source_paths": ["evaluator.py"],
            "dependency_payloads": [{"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}],
        },
        "resource_budget": {"wall_clock_minutes": 5},
    }
    trial_spec = _trial_spec_from_plan(plan, variant, project_root=root)
    for rel, payload in {
        "literature/direction.json": direction,
        "plan/variant.json": variant,
        "plan/trial_spec.json": trial_spec,
    }.items():
        _write_projection(root, rel, payload)
    _implementation_projection(root, variant)
    (root / "producer.py").write_text(_GENERIC_PRODUCER, encoding="utf-8")
    config = {"experiment": {"simulate": False}, "orchestration": {"profile": "standard"}, "llm": {"use_real_api": False}}
    return AgentContext(root, config, ArtifactManager(root), ModelClient(config, project_root=root)), direction, variant


_GENERIC_PRODUCER = r'''import json
from pathlib import Path
root=Path('.')
attempt=next(json.loads(path.read_text()) for path in (root/'meta/attempts').glob('*.json') if json.loads(path.read_text())['state']=='FULL_RUNNING')
phase=attempt['phase_executions']['full']; producer=phase['producer_run_id']
marker=root/'runner/invocations.jsonl'; marker.parent.mkdir(parents=True,exist_ok=True)
with marker.open('a',encoding='utf-8') as stream: stream.write(json.dumps({'attempt_id':attempt['attempt_id'],'phase':'full','producer_run_id':producer},sort_keys=True)+'\n')
common={'lifecycle_generation':attempt['lifecycle_generation'],'implementation_hash':attempt['implementation_hash'],'attempt_input_hash':attempt['attempt_input_hash'],'phase':'full','phase_execution_id':phase['phase_execution_id'],'phase_start_event_id':phase['phase_start_event_id']}
identity={'attempt_id':attempt['attempt_id'],'producer_run_id':producer,'direction_semantic_hash':attempt['direction_semantic_hash'],'direction_spec_hash':attempt['direction_spec_hash'],'variant_semantic_hash':attempt['variant_semantic_hash'],'variant_spec_hash':attempt['variant_spec_hash'],'trial_spec_hash':attempt['trial_spec_hash'],'protocol_hash':attempt['protocol_hash'],'sample_manifest_hash':attempt['sample_manifest_hash'],'evaluator_hash':attempt['evaluator_hash'],'cross_references':{},**common}
rows=[]
for role,value in [('baseline',0.5),('candidate',0.8)]: rows.append({'phase':'full','role':role,'dataset_id':'dataset-a','metric_id':'accuracy','seed':7,'metric_value':value,'command_status':'completed','attempt_id':attempt['attempt_id'],'variant_semantic_hash':attempt['variant_semantic_hash'],'variant_spec_hash':attempt['variant_spec_hash'],'trial_spec_hash':attempt['trial_spec_hash'],'sample_manifest_hash':attempt['sample_manifest_hash'],'evaluator_hash':attempt['evaluator_hash'],'producer_run_id':producer,**{k:v for k,v in common.items() if k!='phase'}})
main={'schema_version':'auto_research_main_results_v3','evidence_kind':'main_results','evidence_id':'evidence-main-real',**identity,'rows':rows}
activation={'schema_version':'auto_research_activation_evidence_v4','evidence_kind':'activation_evidence','evidence_id':'evidence-activation-real',**identity,'probe_id':'forward-probe-real','status':'activated','command_status':'completed','exit_code':0,'expected_surface_ids':['src/router.py'],'observed_surface_ids':['src/router.py'],'activation_delta_threshold':0.01,'surface_measurements':[{'surface_id':'src/router.py','enabled_value':1.0,'disabled_value':0.0,'delta':1.0,'threshold':0.01,'status':'ACTIVATED'}]}
(root/'runner').mkdir(exist_ok=True); (root/'runner/main.json').write_text(json.dumps(main,sort_keys=True,separators=(',',':'))); (root/'runner/activation.json').write_text(json.dumps(activation,sort_keys=True,separators=(',',':')))
manifest={**phase,'sample_contract_ref':attempt['frozen_trial_spec']['sample_manifest_ref'],'evaluator_contract_ref':attempt['frozen_trial_spec']['execution_contract']['evaluator_manifest_ref'],'artifacts':[{'kind':'main_results','source_path':'runner/main.json','producer_run_id':producer},{'kind':'activation_evidence','source_path':'runner/activation.json','producer_run_id':producer}]}
(root/'runner/phase_manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':')))
'''


def _assert_generic_receipt_chain(root: Path, *, expected_attempts: int) -> None:
    ledger = ResearchEventLedger(root)
    state = ledger.state()
    store = ContractStore(root)
    records = list(state["phase_commands"].values())
    physical = [
        item for item in records
        if item["command"]["command_spec_id"] != "full-derive-evidence"
    ]
    derived = [
        item for item in records
        if item["command"]["command_spec_id"] == "full-derive-evidence"
    ]
    assert len(physical) == expected_attempts
    assert len(derived) == expected_attempts
    for record in physical:
        receipt = store.read_json(
            record["receipt_ref"],
            schema_file="phase_run_receipt_v5.schema.json",
        )
        assert receipt["outputs"] == []
        assert {item["kind"] for item in receipt["raw_outputs"]} == {
            "activation_evidence",
            "main_results",
        }
        assert receipt["derivation_ref"] is None
        assert receipt["derivation_hash"] is None
    for record in derived:
        receipt = store.read_json(
            record["receipt_ref"],
            schema_file="phase_run_receipt_v5.schema.json",
        )
        assert receipt["raw_outputs"] == []
        assert {item["kind"] for item in receipt["outputs"]} == {
            "activation_evidence",
            "main_results",
        }
        assert receipt["derivation_ref"]["digest"] == receipt["derivation_hash"]
        trial = state["trial_results"][record["command"]["attempt_id"]]
        manifest = trial["evidence_manifest"]
        assert manifest["derive_receipt_ref"] == record["receipt_ref"]
        assert manifest["derive_receipt_hash"] == record["receipt_ref"]["digest"]
        assert manifest["derivation_ref"] == receipt["derivation_ref"]
        assert manifest["derivation_hash"] == receipt["derivation_hash"]
    invocations = [
        json.loads(line)
        for line in (root / "runner" / "invocations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(invocations) == expected_attempts
    assert len({item["attempt_id"] for item in invocations}) == expected_attempts


def test_generic_full_only_uses_authority_journal_and_gate(tmp_path: Path) -> None:
    root = tmp_path / "generic"
    root.mkdir()
    context, _, _ = _generic_project(root)
    result = ExperimentAgent(context).run()
    ledger = ResearchEventLedger(root)
    events = ledger.events()
    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert [event["event_type"] for event in events if event["event_type"].startswith("PhaseCommand")] == [
        "PhaseCommandStarted",
        "PhaseCommandCompleted",
        "PhaseCommandStarted",
        "PhaseCommandCompleted",
    ]
    assert all(
        command["status"] == "completed"
        and command["command"]["provenance_mode"] == "production"
        for command in ledger.state()["phase_commands"].values()
    )
    _assert_generic_receipt_chain(root, expected_attempts=1)
    assert run_stage_gate("S3_experiment", root, context.config).to_dict()["status"] == "PASS"


def _generic_variant(direction: dict, index: int) -> dict:
    return build_variant_spec(direction, {
        "variant_id": f"resource-variant-{index}",
        "variation_coordinates": {"intervention": {"strength": index}},
        "intervention": {"summary": f"Apply routing strength {index}", "algorithm_operations": [f"apply-routing-{index}"], "configuration": {"strength": index}},
        "hypothesis": f"Routing strength {index} improves accuracy.",
        "null_hypothesis": f"Routing strength {index} does not improve accuracy.",
        "alternative_hypothesis": f"Routing strength {index} improves accuracy.",
        "controlled_variables": {"dataset": "dataset-a"}, "nuisance_variables": ["runtime noise"],
        "implementation_surface_ids": ["src/router.py"],
        "expected_metric_signature": {"primary": "accuracy", "direction": "increase"},
        "falsification_conditions": ["accuracy does not improve"], "ablation": {"switch": f"disable-routing-{index}"},
        "resource_budget": {"max_wall_seconds": 60, "max_retries": 1},
        "failure_routing": {"implementation": "REPAIR_IMPLEMENTATION", "method": "PROPOSE_NEXT_VARIANT"},
        "lineage": {"s2_run_id": f"s2-{index}", "iteration": index, "direction_spec_hash": direction["direction_spec_hash"], "feedback_from_attempt_ids": []},
    })


def _generic_trial_spec(root: Path, variant: dict) -> dict:
    sample_bytes = (root / "samples/dataset-a.jsonl").read_bytes()
    return _trial_spec_from_plan({
        "hypotheses": [{"id": "H1", "statement": "candidate improves accuracy"}], "baselines": [{"name": "baseline"}],
        "datasets": [{"name": "dataset-a", "split": "validation", "sample_count": 1, "source_revision": "local-fixture-v1", "ordered_sample_ids": [hashlib.sha256(sample_bytes).hexdigest()]}],
        "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [7], "aggregation": "mean", "require_complete_seed_coverage": True},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 0.0}, "ablation_matrix": [],
        "execution": {"mode": "real", "collector": "external_manifest", "commands": [{"argv": [sys.executable, "producer.py"]}], "workdir": str(root), "phase_manifest_path": "runner/phase_manifest.json", "evaluator_id": "fixture-evaluator", "evaluator_source_paths": ["evaluator.py"], "dependency_payloads": [{"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}]},
        "resource_budget": {"wall_clock_minutes": 5},
    }, variant, project_root=root)


def test_generic_non_simulated_five_variants_and_sixth_precommand_rejection(tmp_path: Path) -> None:
    root = tmp_path / "generic-five"; root.mkdir(); context, direction, _ = _generic_project(root)
    routes = []; semantic_hashes = []
    for index in range(1, 6):
        variant = _generic_variant(direction, index); semantic_hashes.append(variant["variant_semantic_hash"])
        _write_projection(root, "plan/variant.json", variant); _write_projection(root, "plan/trial_spec.json", _generic_trial_spec(root, variant)); _implementation_projection(root, variant)
        routes.append(ExperimentAgent(context).run()["route_outcome"]["next_action"])
    state = ResearchEventLedger(root).state(); direction_state = state["directions"][direction["direction_semantic_hash"]]
    assert len(set(semantic_hashes)) == 5
    assert routes[:4] == ["PROPOSE_NEXT_VARIANT"] * 4 and routes[4] == "FINISH_DIRECTION"
    assert direction_state["budget"] == {"target": 5, "reserved": 0, "consumed": 5}
    assert len(state["trial_results"]) == 5 and state["latest_direction_aggregate"]["selection"]["status"] in {"selected", "inconclusive"}
    _assert_generic_receipt_chain(root, expected_attempts=5)
    command_count = len(state["phase_commands"])
    sixth = _generic_variant(direction, 6); _write_projection(root, "plan/variant.json", sixth); _write_projection(root, "plan/trial_spec.json", _generic_trial_spec(root, sixth)); _implementation_projection(root, sixth)
    with pytest.raises(Exception, match="closed|FINISHED|direction|budget"):
        ExperimentAgent(context).run()
    assert len(ResearchEventLedger(root).state()["phase_commands"]) == command_count
