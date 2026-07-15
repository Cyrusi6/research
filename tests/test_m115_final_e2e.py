from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from auto_research.agents.base import AgentContext
from auto_research.agents.experiment import ExperimentAgent
from auto_research.agents.plan import _trial_spec_from_plan
from auto_research.artifacts import ArtifactManager
from auto_research.c2c import C2CAdapter
from auto_research.domain_contracts import build_direction_spec, build_variant_spec, canonical_hash
from auto_research.llm import ModelClient
from auto_research.research_state import ResearchEventLedger
from auto_research.validators import run_stage_gate
from auto_research.utils import write_json
from test_c2c import _base_config, _fake_c2c_repo
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
            "mode": "real", "collector": "external_manifest", "commands": [f"{sys.executable} producer.py"],
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
common={'lifecycle_generation':attempt['lifecycle_generation'],'implementation_hash':attempt['implementation_hash'],'attempt_input_hash':attempt['attempt_input_hash'],'phase':'full','phase_execution_id':phase['phase_execution_id'],'phase_start_event_id':phase['phase_start_event_id']}
identity={'attempt_id':attempt['attempt_id'],'producer_run_id':producer,'direction_semantic_hash':attempt['direction_semantic_hash'],'direction_spec_hash':attempt['direction_spec_hash'],'variant_semantic_hash':attempt['variant_semantic_hash'],'variant_spec_hash':attempt['variant_spec_hash'],'trial_spec_hash':attempt['trial_spec_hash'],'protocol_hash':attempt['protocol_hash'],'sample_manifest_hash':attempt['sample_manifest_hash'],'evaluator_hash':attempt['evaluator_hash'],'cross_references':{},**common}
rows=[]
for role,value in [('baseline',0.5),('candidate',0.8)]: rows.append({'phase':'full','role':role,'dataset_id':'dataset-a','metric_id':'accuracy','seed':7,'metric_value':value,'command_status':'completed','attempt_id':attempt['attempt_id'],'variant_semantic_hash':attempt['variant_semantic_hash'],'variant_spec_hash':attempt['variant_spec_hash'],'trial_spec_hash':attempt['trial_spec_hash'],'sample_manifest_hash':attempt['sample_manifest_hash'],'evaluator_hash':attempt['evaluator_hash'],'producer_run_id':producer,**{k:v for k,v in common.items() if k!='phase'}})
main={'schema_version':'auto_research_main_results_v3','evidence_kind':'main_results','evidence_id':'evidence-main-real',**identity,'rows':rows}
activation={'schema_version':'auto_research_activation_evidence_v3','evidence_kind':'activation_evidence','evidence_id':'evidence-activation-real',**identity,'probe_id':'forward-probe-real','status':'passed','command_status':'completed','exit_code':0,'implementation_surface_ids':['src/router.py']}
(root/'runner').mkdir(exist_ok=True); (root/'runner/main.json').write_text(json.dumps(main,sort_keys=True,separators=(',',':'))); (root/'runner/activation.json').write_text(json.dumps(activation,sort_keys=True,separators=(',',':')))
manifest={**phase,'sample_contract_ref':attempt['frozen_trial_spec']['sample_manifest_ref'],'evaluator_contract_ref':attempt['frozen_trial_spec']['execution_contract']['evaluator_manifest_ref'],'artifacts':[{'kind':'main_results','source_path':'runner/main.json','producer_run_id':producer},{'kind':'activation_evidence','source_path':'runner/activation.json','producer_run_id':producer}]}
(root/'runner/phase_manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':')))
'''


def test_generic_full_only_uses_authority_journal_and_gate(tmp_path: Path) -> None:
    root = tmp_path / "generic"
    root.mkdir()
    context, _, _ = _generic_project(root)
    result = ExperimentAgent(context).run()
    ledger = ResearchEventLedger(root)
    events = ledger.events()
    assert result["route_outcome"]["next_action"] == "PROPOSE_NEXT_VARIANT"
    assert [event["event_type"] for event in events if event["event_type"].startswith("PhaseCommand")] == [
        "PhaseCommandStarted", "PhaseCommandCompleted"
    ]
    command = next(iter(ledger.state()["phase_commands"].values()))
    assert command["status"] == "completed"
    assert command["command"]["provenance_mode"] == "production"
    assert run_stage_gate("S3_experiment", root, context.config).to_dict()["status"] == "PASS"


def _c2c_contracts(root: Path, *, profile: str) -> tuple[dict, dict, dict]:
    direction = build_direction_spec({
        "direction_id": "c2c-local-direction",
        "research_question": "Does a local routing operation improve paired C2C accuracy?",
        "mechanism_invariants": {"causal_hypothesis": "routing changes cache utility", "target_mediator": "cache_utility", "invariants": ["fixed data", "fixed evaluator"]},
        "falsification_conditions": ["paired accuracy does not improve"],
        "support_claim_ids": ["support-1"], "counter_claim_ids": ["counter-1"],
        "implementation_surface_ids": ["rosetta/model/wrapper.py"],
        "metric_signature": {"primary": "three_dataset_mean", "direction": "increase"},
        "benchmark_contract_hash": canonical_hash({"dataset": "mmlu-redux"}),
        "variant_space": {"mutable_axes": ["routing"], "immutable_axes": ["data", "evaluator"], "forbidden_combinations": []},
        "s2_entry_conditions": ["gate"], "return_to_s1_conditions": ["five rejects"],
        "lineage": {"s1_run_id": "s1-local", "iteration": 1, "input_manifest_hash": canonical_hash({"input": 1})},
    })
    variant = build_variant_spec(direction, {
        "variant_id": "c2c-local-variant",
        "variation_coordinates": {"routing": "utility"},
        "intervention": {"summary": "utility route", "algorithm_operations": ["route utility"], "configuration": {"train": {"model": {"soft_alignment_top_k": 2}}, "eval": {"model": {"rosetta_config": {"soft_alignment_top_k": 2}}}}},
        "hypothesis": "utility routing improves accuracy", "null_hypothesis": "no improvement", "alternative_hypothesis": "improvement",
        "controlled_variables": {"dataset": "mmlu-redux"}, "nuisance_variables": ["process noise"],
        "implementation_surface_ids": ["rosetta/model/wrapper.py"],
        "expected_metric_signature": {"primary": "three_dataset_mean", "direction": "increase"},
        "falsification_conditions": ["no paired improvement"], "ablation": {"switch": "disable_utility"},
        "resource_budget": {"max_wall_seconds": 60, "max_retries": 1},
        "failure_routing": {"implementation": "REPAIR_IMPLEMENTATION", "method": "PROPOSE_NEXT_VARIANT"},
        "lineage": {"s2_run_id": "s2-local", "iteration": 1, "direction_spec_hash": direction["direction_spec_hash"], "feedback_from_attempt_ids": []},
    })
    sample = root / "samples/mmlu-redux.jsonl"; sample.parent.mkdir(parents=True, exist_ok=True); raw=b'{"id":"q1","answer":"A"}\n'; sample.write_bytes(raw)
    evaluator = root / "evaluator.py"; evaluator.write_text("def score(x): return x\n", encoding="utf-8")
    plan = {
        "datasets": [{"name": "mmlu-redux", "split": "test", "sample_count": 1, "source_revision": "local-c2c-v1", "ordered_sample_ids": [hashlib.sha256(raw).hexdigest()]}],
        "metrics": [{"name": "three_dataset_mean", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [42], "aggregation": "mean", "require_complete_seed_coverage": True},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 2.0},
        "ablation_matrix": [],
        "execution": {"mode": "real", "collector": "c2c_small_loop", "commands": [], "evaluator_id": "c2c-local", "evaluator_source_paths": ["evaluator.py"], "dependency_payloads": [{"name": "python", "version": "local"}]},
    }
    return direction, variant, _trial_spec_from_plan(plan, variant, profile=profile, project_root=root)


def _c2c_context(root: Path, repo: Path, *, profile: str) -> AgentContext:
    direction, variant, trial_spec = _c2c_contracts(root, profile=profile)
    for rel, payload in {"literature/direction.json": direction, "plan/variant.json": variant, "plan/trial_spec.json": trial_spec}.items(): _write_projection(root, rel, payload)
    _implementation_projection(root, variant)
    config = _base_config(root.parent / "workspace", simulate=False)
    config["orchestration"]["profile"] = profile
    config["c2c"] = {
        "enabled": True, "snapshot_path": str(repo), "env_python": sys.executable, "model_map": {},
        "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}}, "datasets": ["mmlu-redux"],
        "small_loop": {"eval_datasets": ["mmlu-redux"], "train_samples": 1, "gpu_ids": [], "proxy_screen": {"enabled": True, "train_samples": 1, "eval_datasets": ["mmlu-redux"], "min_delta_to_pass": 0.1}},
    }
    if profile == "bootstrap": config["orchestration"]["bootstrap"] = {"proxy_only": True}
    return AgentContext(root, config, ArtifactManager(root), ModelClient(config, project_root=root))


def _install_local_c2c_runner(
    agent: ExperimentAgent,
    *,
    proxy_accuracy: float,
    trace: list[dict],
    verify_callback_state: bool = True,
) -> None:
    def run_step(*, name, command, working_dir, retry_policy=None, **kwargs):
        del command, retry_policy, kwargs
        context = agent._active_phase_context
        if verify_callback_state:
            state = ResearchEventLedger(agent.context.project_root).state()
            attempt = state["attempts"][context.attempt_id]
            sequence = state["last_sequence"]
            authoritative_state = attempt["state"]
        else:
            with sqlite3.connect(ResearchEventLedger(agent.context.project_root).db_path) as connection:
                sequence = int(connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM events").fetchone()[0])
            authoritative_state = "PROXY_RUNNING" if context.phase == "proxy" else "FULL_RUNNING"
        trace.append({"name": name, "sequence": sequence, "state": authoritative_state, "phase": context.phase})
        run_repo = Path(working_dir)
        run_id = json.loads((agent.context.project_root / "plan/variant.json").read_text(encoding="utf-8"))["variant_id"]
        script = "from pathlib import Path; import json,sys; p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); (p/sys.argv[2]).write_text(sys.argv[3])"
        if name == "proxy_baseline_train":
            target = run_repo / "local/auto_research_runs/proxy_baseline/checkpoints/final"; subprocess.run([sys.executable,"-c",script,str(target),"marker.txt","ok"],check=True)
        elif name == "proxy_baseline_eval_mmlu-redux":
            target = run_repo / "local/auto_research_runs/proxy_baseline/results/mmlu-redux"; payload=json.dumps({"model":"Rosetta","dataset":"mmlu-redux","answer_method":"generate","overall_accuracy":0.50}); subprocess.run([sys.executable,"-c",script,str(target),"Rosetta_mmlu-redux_generate_summary.json",payload],check=True)
        elif name == "proxy_command_0":
            target = run_repo / f"local/auto_research_runs/{run_id}/proxy/checkpoints/final"; subprocess.run([sys.executable,"-c",script,str(target),"marker.txt","ok"],check=True)
        elif name == "proxy_command_1":
            target = run_repo / f"local/auto_research_runs/{run_id}/proxy/results/mmlu-redux"; payload=json.dumps({"model":"Rosetta","dataset":"mmlu-redux","answer_method":"generate","overall_accuracy":proxy_accuracy}); subprocess.run([sys.executable,"-c",script,str(target),"Rosetta_mmlu-redux_generate_summary.json",payload],check=True)
        elif name == "activation_smoke_eval_mmlu-redux":
            target = run_repo / f"local/auto_research_runs/{run_id}/proxy/activation_smoke_disabled/results/mmlu-redux"; payload=json.dumps({"model":"Rosetta","dataset":"mmlu-redux","answer_method":"generate","overall_accuracy":proxy_accuracy}); subprocess.run([sys.executable,"-c",script,str(target),"Rosetta_mmlu-redux_generate_summary.json",payload],check=True)
        elif name == "train":
            target = run_repo / f"local/auto_research_runs/{run_id}/checkpoints/final"; subprocess.run([sys.executable,"-c",script,str(target),"marker.txt","ok"],check=True)
        elif name == "eval_mmlu-redux":
            target = run_repo / f"local/auto_research_runs/{run_id}/results/mmlu-redux"; payload=json.dumps({"model":"Rosetta","dataset":"mmlu-redux","answer_method":"generate","overall_accuracy":0.52}); subprocess.run([sys.executable,"-c",script,str(target),"Rosetta_mmlu-redux_generate_summary.json",payload],check=True)
        return {"step": name, "status": "ok", "attempts": [{"stdout": "", "stderr": "", "returncode": 0}], "returncode": 0}
    agent.runner.run_step = run_step


@pytest.mark.parametrize(("proxy_accuracy", "expected_route", "full_calls"), [(0.49, "PROPOSE_NEXT_VARIANT", 0), (0.51, "PROPOSE_NEXT_VARIANT", 2)])
def test_c2c_non_simulated_physical_proxy_barrier(tmp_path: Path, proxy_accuracy: float, expected_route: str, full_calls: int) -> None:
    repo = _fake_c2c_repo(tmp_path)
    root = tmp_path / ("c2c-pass" if proxy_accuracy > 0.5 else "c2c-reject"); root.mkdir()
    context = _c2c_context(root, repo, profile="standard")
    agent = ExperimentAgent(context); trace: list[dict] = []; _install_local_c2c_runner(agent, proxy_accuracy=proxy_accuracy, trace=trace)
    result = agent.run()
    ledger = ResearchEventLedger(root); event_types = [event["event_type"] for event in ledger.events()]
    assert "route_outcome" in result, result
    assert result["route_outcome"]["next_action"] == expected_route
    assert sum(item["name"] in {"train", "eval_mmlu-redux"} for item in trace) == full_calls
    if full_calls:
        assert event_types.index("ProxyEvidenceCommitted") < event_types.index("FullPhaseStarted")
        assert all(item["state"] == "FULL_RUNNING" for item in trace if item["name"] in {"train", "eval_mmlu-redux"})
    else:
        assert "FullPhaseStarted" not in event_types
        assert ledger.state()["directions"][next(iter(ledger.state()["directions"]))]["budget"] == {"target": 5, "reserved": 0, "consumed": 0}
    assert run_stage_gate("S3_experiment", root, context.config).to_dict()["status"] == "PASS"


def test_c2c_non_simulated_bootstrap_is_proxy_only_and_budget_isolated(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path); root = tmp_path / "c2c-bootstrap"; root.mkdir()
    context = _c2c_context(root, repo, profile="bootstrap")
    agent = ExperimentAgent(context); trace: list[dict] = []; _install_local_c2c_runner(agent, proxy_accuracy=0.51, trace=trace)
    result = agent.run(); ledger = ResearchEventLedger(root); state = ledger.state()
    assert result["route_outcome"]["next_action"] == "FINISH_RUN"
    assert not any(item["name"] in {"train", "eval_mmlu-redux"} for item in trace)
    assert next(iter(state["directions"].values()))["budget"] == {"target": 5, "reserved": 0, "consumed": 0}
    assert state["method_tried_history"] == []
    assert state["latest_direction_aggregate"] is None
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
        "execution": {"mode": "real", "collector": "external_manifest", "commands": [f"{sys.executable} producer.py"], "workdir": str(root), "phase_manifest_path": "runner/phase_manifest.json", "evaluator_id": "fixture-evaluator", "evaluator_source_paths": ["evaluator.py"], "dependency_payloads": [{"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}]},
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
    command_count = len(state["phase_commands"])
    sixth = _generic_variant(direction, 6); _write_projection(root, "plan/variant.json", sixth); _write_projection(root, "plan/trial_spec.json", _generic_trial_spec(root, sixth)); _implementation_projection(root, sixth)
    with pytest.raises(Exception, match="closed|FINISHED|direction|budget"):
        ExperimentAgent(context).run()
    assert len(ResearchEventLedger(root).state()["phase_commands"]) == command_count


def _c2c_variant(direction: dict, index: int) -> dict:
    return build_variant_spec(direction, {
        "variant_id": f"c2c-local-variant-{index}", "variation_coordinates": {"routing": f"utility-{index}"},
        "intervention": {"summary": f"utility route {index}", "algorithm_operations": [f"route-utility-{index}"], "configuration": {"train": {"model": {"soft_alignment_top_k": index + 1}}, "eval": {"model": {"rosetta_config": {"soft_alignment_top_k": index + 1}}}}},
        "hypothesis": f"utility routing {index} improves accuracy", "null_hypothesis": f"routing {index} has no improvement", "alternative_hypothesis": f"routing {index} improves accuracy",
        "controlled_variables": {"dataset": "mmlu-redux"}, "nuisance_variables": ["process noise"],
        "implementation_surface_ids": ["rosetta/model/wrapper.py"], "expected_metric_signature": {"primary": "three_dataset_mean", "direction": "increase"},
        "falsification_conditions": ["no paired improvement"], "ablation": {"switch": f"disable-utility-{index}"},
        "resource_budget": {"max_wall_seconds": 60, "max_retries": 1}, "failure_routing": {"implementation": "REPAIR_IMPLEMENTATION", "method": "PROPOSE_NEXT_VARIANT"},
        "lineage": {"s2_run_id": f"s2-local-{index}", "iteration": index, "direction_spec_hash": direction["direction_spec_hash"], "feedback_from_attempt_ids": []},
    })


def _c2c_trial_spec(root: Path, variant: dict, *, profile: str = "standard") -> dict:
    raw = (root / "samples/mmlu-redux.jsonl").read_bytes()
    return _trial_spec_from_plan({
        "datasets": [{"name": "mmlu-redux", "split": "test", "sample_count": 1, "source_revision": "local-c2c-v1", "ordered_sample_ids": [hashlib.sha256(raw).hexdigest()]}],
        "metrics": [{"name": "three_dataset_mean", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [42], "aggregation": "mean", "require_complete_seed_coverage": True},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 2.0}, "ablation_matrix": [],
        "execution": {"mode": "real", "collector": "c2c_small_loop", "commands": [], "evaluator_id": "c2c-local", "evaluator_source_paths": ["evaluator.py"], "dependency_payloads": [{"name": "python", "version": "local"}]},
    }, variant, profile=profile, project_root=root)


def test_c2c_non_simulated_five_variants_have_physical_proxy_full_order(tmp_path: Path) -> None:
    repo = _fake_c2c_repo(tmp_path); root = tmp_path / "c2c-five"; root.mkdir(); context = _c2c_context(root, repo, profile="standard")
    direction = json.loads((root / "literature/direction.json").read_text()); trace: list[dict] = []; agent = ExperimentAgent(context); _install_local_c2c_runner(agent, proxy_accuracy=0.51, trace=trace, verify_callback_state=False)
    routes = []; semantic_hashes = []
    for index in range(1, 6):
        variant = _c2c_variant(direction, index); semantic_hashes.append(variant["variant_semantic_hash"])
        _write_projection(root, "plan/variant.json", variant); _write_projection(root, "plan/trial_spec.json", _c2c_trial_spec(root, variant)); _implementation_projection(root, variant)
        routes.append(agent.run()["route_outcome"]["next_action"])
    ledger = ResearchEventLedger(root); state = ledger.state(); budget = state["directions"][direction["direction_semantic_hash"]]["budget"]
    assert len(set(semantic_hashes)) == 5 and budget == {"target": 5, "reserved": 0, "consumed": 5}
    assert routes[:4] == ["PROPOSE_NEXT_VARIANT"] * 4 and routes[4] == "FINISH_DIRECTION"
    assert len(state["trial_results"]) == 5
    for attempt in state["attempts"].values():
        assert attempt["phase_executions"]["proxy"]["phase_start_event_id"]
        assert attempt["phase_executions"]["full"]["phase_start_event_id"]
    full_steps = [item for item in trace if item["name"] in {"train", "eval_mmlu-redux"}]
    assert len(full_steps) == 10 and all(item["state"] == "FULL_RUNNING" for item in full_steps)
