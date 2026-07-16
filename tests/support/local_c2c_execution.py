from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

from auto_research.agents.base import AgentContext
from auto_research.agents.plan import _trial_spec_from_plan
from auto_research.artifacts import ArtifactManager
from auto_research.domain_contracts import build_direction_spec, build_variant_spec, canonical_hash
from auto_research.llm import ModelClient
from auto_research.utils import write_json


def create_local_c2c_repo(root: Path, *, proxy_accuracy: float, full_accuracy: float = 0.52) -> Path:
    repo = root / "C2C"
    for relative in (
        "rosetta/model",
        "script/train",
        "script/evaluation",
        "recipe/train_recipe",
        "recipe/eval_recipe",
        "local/final_results/route1_alignment_v22/small_loop_summary",
        "local/final_results/demo/mmlu-redux",
    ):
        (repo / relative).mkdir(parents=True, exist_ok=True)
    for filename in ("README.md", "RUNBOOK.md", "C2C_跨Tokenizer柔性对齐改进方向研究备忘.md"):
        (repo / filename).write_text(f"# {filename}\nlocal production-component fixture\n", encoding="utf-8")
    for filename in ("aligner.py", "projector.py", "wrapper.py"):
        (repo / "rosetta/model" / filename).write_text("VALUE = 'fixture'\n", encoding="utf-8")
    (repo / "recipe/train_recipe/C2C_0.6+0.5.json").write_text(
        json.dumps({"output": {}, "data": {"kwargs": {}}, "training": {}, "model": {}}),
        encoding="utf-8",
    )
    (repo / "recipe/eval_recipe/unified_eval.yaml").write_text(
        yaml.safe_dump({"model": {"rosetta_config": {}}, "output": {}, "eval": {"dataset": "mmlu-redux"}}),
        encoding="utf-8",
    )
    (repo / "local_execution_control.json").write_text(
        json.dumps({"proxy_accuracy": proxy_accuracy, "full_accuracy": full_accuracy}, sort_keys=True),
        encoding="utf-8",
    )
    (repo / "script/train/SFT_train.py").write_text(_TRAIN_SCRIPT, encoding="utf-8")
    (repo / "script/evaluation/unified_evaluator.py").write_text(_EVAL_SCRIPT, encoding="utf-8")
    scores = repo / "local/final_results/route1_alignment_v22/small_loop_summary/route1_v22_small_loop_scores.csv"
    scores.write_text(
        "method,receiver,sharer,alignment_strategy,confidence_gate,train_samples,final_train_loss,mid_eval_loss,final_eval_loss,mmlu_redux,ai2_arc_challenge,openbookqa,mean,delta_mean_vs_v21_entropy050\n"
        "fixture,Qwen,Tiny,fixture,fixture,1,0,0,0,50,50,50,50,0\n",
        encoding="utf-8",
    )
    (repo / "local/final_results/demo/mmlu-redux/Rosetta_mmlu-redux_generate_summary.json").write_text(
        json.dumps({"model": "Rosetta", "dataset": "mmlu-redux", "answer_method": "generate", "overall_accuracy": 0.5}),
        encoding="utf-8",
    )
    return repo


def install_fake_gpu(root: Path, monkeypatch) -> Path:
    bin_dir = root / "fake-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    executable = bin_dir / "nvidia-smi"
    executable.write_text(_NVIDIA_SMI_SCRIPT, encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(bin_dir), os.environ.get("PATH", ""))))
    return executable


def build_c2c_context(project_root: Path, repo: Path, *, profile: str) -> AgentContext:
    hf_home, dataset_cache = _ensure_local_dataset_cache(project_root)
    direction, variant, trial_spec = _contracts(project_root, profile=profile)
    for relative, payload in (
        ("literature/direction.json", direction),
        ("plan/variant.json", variant),
        ("plan/trial_spec.json", trial_spec),
    ):
        write_json(project_root / relative, payload)
    write_json(project_root / "plan/code_patches/implementation_contract.json", {
        "schema_version": "implementation_contract_test_v1",
        "variant_spec_hash": variant["variant_spec_hash"],
    })
    write_json(project_root / "plan/code_patches/patch_manifest.json", {
        "schema_version": "auto_research_patch_manifest_v1",
        "status": "disabled",
        "selected_candidate_id": variant["variant_id"],
        "variant_spec_hash": variant["variant_spec_hash"],
    })
    write_json(project_root / "plan/code_patches/patch_gate_report.json", {
        "schema_version": "auto_research_patch_gate_v1",
        "gate": "pass",
        "variant_id": variant["variant_id"],
        "variant_spec_hash": variant["variant_spec_hash"],
        "checks": {"activation": True},
    })
    config = {
        "project": {"workspace_root": str(project_root.parent), "target_venue": "TestConf", "language": "en"},
        "llm": {"provider": "openai", "use_real_api": False, "model": "mock"},
        "literature": {"download_pdfs": False, "request_timeout_seconds": 1, "max_papers": 0, "arxiv_max_results": 0},
        "experiment": {"simulate": False, "random_seeds": [42], "gpu_policy": {"min_free_mb": 1}},
        "writing": {"claim_verification": {"enabled": True, "min_pass_rate": 0.8}, "require_compile": False},
        "review": {"pass_threshold": 7.0, "max_iterations": 1},
        "orchestration": {"judge_max_retries": 1, "auto_mode": True, "profile": profile},
        "c2c": {
            "enabled": True,
            "snapshot_path": str(repo),
            "env_python": sys.executable,
            "hf_home": str(hf_home),
            "hf_datasets_cache": str(dataset_cache),
            "model_map": {},
            "baseline": {"name": "base", "mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
            "datasets": ["mmlu-redux"],
            "small_loop": {
                "eval_datasets": ["mmlu-redux"],
                "train_samples": 1,
                "gpu_ids": [0],
                "proxy_screen": {
                    "enabled": True,
                    "train_samples": 1,
                    "eval_datasets": ["mmlu-redux"],
                    "min_delta_to_pass": 0.1,
                },
            },
        },
    }
    if profile == "bootstrap":
        config["orchestration"]["bootstrap"] = {"proxy_only": True}
    return AgentContext(project_root, config, ArtifactManager(project_root), ModelClient(config, project_root=project_root))


def _ensure_local_dataset_cache(project_root: Path) -> tuple[Path, Path]:
    hf_home = project_root / ".fixture-hf"
    dataset_cache = hf_home / "datasets"
    for dataset_id in ("mmlu-redux", "ai2-arc", "openbookqa"):
        (dataset_cache / dataset_id).mkdir(parents=True, exist_ok=True)
    return hf_home, dataset_cache


def invocation_records(project_root: Path) -> list[dict]:
    records = []
    for marker in project_root.parent.rglob("local_command_invocations.jsonl"):
        for line in marker.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _contracts(root: Path, *, profile: str) -> tuple[dict, dict, dict]:
    direction = build_direction_spec({
        "direction_id": "c2c-local-direction",
        "research_question": "Does a local routing operation improve paired C2C accuracy?",
        "mechanism_invariants": {"causal_hypothesis": "routing changes cache utility", "target_mediator": "cache_utility", "invariants": ["fixed data", "fixed evaluator"]},
        "falsification_conditions": ["paired accuracy does not improve"],
        "support_claim_ids": ["support-1"],
        "counter_claim_ids": ["counter-1"],
        "implementation_surface_ids": ["rosetta/model/wrapper.py"],
        "metric_signature": {"primary": "three_dataset_mean", "direction": "increase"},
        "benchmark_contract_hash": canonical_hash({"dataset": "mmlu-redux"}),
        "variant_space": {"mutable_axes": ["routing"], "immutable_axes": ["data", "evaluator"], "forbidden_combinations": []},
        "s2_entry_conditions": ["gate"],
        "return_to_s1_conditions": ["five rejects"],
        "lineage": {"s1_run_id": "s1-local", "iteration": 1, "input_manifest_hash": canonical_hash({"input": 1})},
    })
    variant = build_variant_spec(direction, {
        "variant_id": "c2c-local-variant",
        "variation_coordinates": {"routing": "utility"},
        "intervention": {"summary": "utility route", "algorithm_operations": ["route utility"], "configuration": {"train": {"model": {"soft_alignment_top_k": 2}}, "eval": {"model": {"rosetta_config": {"soft_alignment_top_k": 2}}}}},
        "hypothesis": "utility routing improves accuracy",
        "null_hypothesis": "no improvement",
        "alternative_hypothesis": "improvement",
        "controlled_variables": {"dataset": "mmlu-redux"},
        "nuisance_variables": ["process noise"],
        "implementation_surface_ids": ["rosetta/model/wrapper.py"],
        "expected_metric_signature": {"primary": "three_dataset_mean", "direction": "increase"},
        "falsification_conditions": ["no paired improvement"],
        "ablation": {"switch": "disable_utility"},
        "resource_budget": {"max_wall_seconds": 60, "max_retries": 1},
        "failure_routing": {"implementation": "REPAIR_IMPLEMENTATION", "method": "PROPOSE_NEXT_VARIANT"},
        "lineage": {"s2_run_id": "s2-local", "iteration": 1, "direction_spec_hash": direction["direction_spec_hash"], "feedback_from_attempt_ids": []},
    })
    sample = root / "samples/mmlu-redux.jsonl"
    sample.parent.mkdir(parents=True, exist_ok=True)
    raw = b'{"id":"q1","answer":"A"}\n'
    sample.write_bytes(raw)
    (root / "evaluator.py").write_text("def score(value):\n    return value\n", encoding="utf-8")
    trial_spec = _trial_spec_from_plan({
        "datasets": [{"name": "mmlu-redux", "split": "test", "sample_count": 1, "source_revision": "local-c2c-v1", "ordered_sample_ids": [hashlib.sha256(raw).hexdigest()]}],
        "metrics": [{"name": "three_dataset_mean", "primary": True, "higher_is_better": True}],
        "statistical_testing": {"seeds": [42], "aggregation": "mean", "require_complete_seed_coverage": True},
        "acceptance_criteria": {"minimum_mean_delta": 0.1, "maximum_dataset_regression": 2.0},
        "ablation_matrix": [],
        "execution": {"mode": "real", "collector": "c2c_small_loop", "commands": [], "evaluator_id": "c2c-local", "evaluator_source_paths": ["evaluator.py"], "dependency_payloads": [{"name": "python", "version": "local"}]},
    }, variant, profile=profile, project_root=root)
    return direction, variant, trial_spec


_TRAIN_SCRIPT = r'''import json,sys
from pathlib import Path
config=Path(sys.argv[sys.argv.index('--config')+1])
payload=json.loads(config.read_text())
out=Path(payload['output']['output_dir'])/'final'
out.mkdir(parents=True,exist_ok=True)
(out/'model.pth').write_bytes(b'local-checkpoint')
marker=Path('local_command_invocations.jsonl')
with marker.open('a',encoding='utf-8') as handle:
    handle.write(json.dumps({'kind':'train','argv':sys.argv,'config':str(config),'output':str(out)},sort_keys=True)+'\n')
print(json.dumps({'status':'completed','output':str(out)}))
'''


_EVAL_SCRIPT = r'''import json,sys
from pathlib import Path
import yaml
config=Path(sys.argv[sys.argv.index('--config')+1])
payload=yaml.safe_load(config.read_text())
out=Path(payload['output']['output_dir'])
out.mkdir(parents=True,exist_ok=True)
control=json.loads(Path('local_execution_control.json').read_text())
text=str(out)
if 'proxy_baseline' in text:
    accuracy=0.50
elif 'activation_smoke_disabled' in text:
    accuracy=max(0.0,float(control['proxy_accuracy'])-0.02)
elif '/proxy/' in text:
    accuracy=float(control['proxy_accuracy'])
else:
    accuracy=float(control['full_accuracy'])
dataset=str(payload.get('eval',{}).get('dataset') or 'mmlu-redux')
summary={'model':'Rosetta','dataset':dataset,'answer_method':'generate','overall_accuracy':accuracy}
(out/f'Rosetta_{dataset}_generate_summary.json').write_text(json.dumps(summary,sort_keys=True))
(out/'prediction_outputs.jsonl').write_text(
    json.dumps({'prediction':'Answer: A','answer':'A'})+'\n'+
    json.dumps({'prediction':'Answer: B','answer':'B'})+'\n'
)
(out/'raw_metrics.json').write_text(json.dumps({'dataset':dataset,'accuracy':accuracy,'config':str(config)},sort_keys=True))
marker=Path('local_command_invocations.jsonl')
with marker.open('a',encoding='utf-8') as handle:
    handle.write(json.dumps({'kind':'eval','argv':sys.argv,'config':str(config),'output':str(out),'accuracy':accuracy},sort_keys=True)+'\n')
print(json.dumps({'status':'completed','output':str(out),'accuracy':accuracy}))
'''


_NVIDIA_SMI_SCRIPT = r'''#!/usr/bin/env python3
import sys
query=' '.join(sys.argv[1:])
if 'name,memory.total,memory.free' in query:
    print('Local GPU, 24576 MiB, 24576 MiB')
elif 'index,memory.total,memory.free,memory.used,utilization.gpu' in query:
    print('0, 24576, 24576, 0, 0')
elif 'memory.free' in query:
    print('24576')
else:
    print('0, 24576, 24576, 0, 0')
'''
