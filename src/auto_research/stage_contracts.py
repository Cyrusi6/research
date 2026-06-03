"""Explicit stage input/output contracts for orchestration and resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import STAGE_LABELS, STAGE_ORDER
from .utils import ensure_dir, now_utc, read_json, sha256_file, write_json


STAGE_CONTRACT_SCHEMA_VERSION = "stage_contract_v2"


DEFAULT_STAGE_CONTRACTS: dict[str, dict[str, Any]] = {
    "S0_intake": {
        "required_inputs": [
            "meta/project_config.yaml",
            "references/papers/manifest.json",
        ],
        "optional_inputs": [],
        "conditional_inputs": [
            {
                "when": "project.mode == c2c",
                "paths": [
                    "external/c2c_snapshot",
                ],
            },
        ],
        "required_outputs": [
            "intake/gate_report.json",
        ],
        "optional_outputs": [],
        "conditional_outputs": [
            {
                "when": "project.mode == c2c",
                "paths": [
                    "intake/c2c/static_bundle.json",
                    "intake/c2c/repo_manifest.json",
                    "intake/c2c/repo_card.json",
                    "intake/c2c/baseline_evidence.json",
                    "intake/c2c/paper_full_manifest.json",
                    "intake/c2c/paper_chunks.jsonl",
                    "intake/c2c/rebuttal_chunks.jsonl",
                    "intake/c2c/rebuttal_concern_matrix.json",
                    "intake/c2c/code_file_manifest.json",
                    "intake/c2c/code_symbols.jsonl",
                    "intake/c2c/code_chunks.jsonl",
                    "intake/c2c/code_edges.jsonl",
                    "intake/c2c/code_repo_map.json",
                    "intake/c2c/code_repo_map.md",
                    "intake/c2c/code_intake_report.json",
                    "intake/c2c/code_intake_report.md",
                    "intake/c2c/implementation_surface_map.json",
                    "intake/c2c/code_retrieval_index.json",
                    "intake/c2c/cache_summary.json",
                    "intake/c2c/chunk_index.json",
                    "intake/c2c/chunk_index.jsonl",
                    "intake/c2c/evidence_brief.json",
                ],
            },
        ],
        "required_config": [],
        "conditional_config": [{"when": "project.mode == c2c", "keys": ["c2c"]}],
        "gate_validator": "s0_intake_gate_v1",
    },
    "S1_literature": {
        "required_inputs": [
            "meta/project_config.yaml",
            "references/papers/manifest.json",
        ],
        "optional_inputs": [
            "meta/negative_memory.jsonl",
        ],
        "conditional_inputs": [
            {
                "when": "iteration > 1",
                "paths": [
                    "experiment/results/failure_feedback.json",
                    "literature/feedback",
                ],
            },
            {
                "when": "project.mode == c2c",
                "paths": [
                    "intake/c2c/static_bundle.json",
                    "intake/c2c/evidence_brief.json",
                ],
            },
        ],
        "required_outputs": [
            "literature/ideas.json",
            "literature/survey.md",
            "literature/feasibility_check.md",
            "literature/gate_report.json",
        ],
        "optional_outputs": [],
        "conditional_outputs": [
            {
                "when": "project.mode == c2c",
                "paths": [
                    "literature/idea_debate.json",
                    "literature/negative_constraints.json",
                    "literature/c2c/baseline_evidence.json",
                    "literature/c2c/rebuttal_concern_matrix.json",
                    "literature/c2c/code_file_manifest.json",
                    "literature/c2c/code_symbols.jsonl",
                    "literature/c2c/code_edges.jsonl",
                    "literature/c2c/code_repo_map.json",
                    "literature/c2c/code_repo_map.md",
                    "literature/c2c/code_intake_report.json",
                    "literature/c2c/implementation_surface_map.json",
                    "literature/c2c/code_retrieval_index.json",
                    "literature/c2c/cache_summary.json",
                    "literature/c2c/chunk_index.json",
                ],
            }
        ],
        "required_config": ["llm", "literature", "ideation"],
        "conditional_config": [{"when": "project.mode == c2c", "keys": ["c2c"]}],
        "gate_validator": "s1_literature_gate_v1",
    },
    "S2_plan": {
        "required_inputs": [
            "literature/ideas.json",
        ],
        "optional_inputs": [
            "experiment/results/failure_feedback.json",
            "plan/performance_feedback.json",
        ],
        "conditional_inputs": [
            {
                "when": "project.mode == c2c",
                "paths": [
                    "literature/negative_constraints.json",
                    "literature/c2c/baseline_evidence.json",
                ],
            },
            {
                "when": "iteration > 1",
                "paths": [
                    "literature/feedback",
                    "plan/plan_feedback.json",
                ],
            },
        ],
        "required_outputs": [
            "plan/plan.yaml",
            "plan/gate_report.json",
        ],
        "optional_outputs": [
            "plan/code_patches/patch_manifest.json",
            "plan/plan_feedback.json",
            "plan/resource_budget.md",
        ],
        "conditional_outputs": [
            {
                "when": "execution.collector == c2c_small_loop",
                "paths": [
                    "plan/candidate_ideas.json",
                    "plan/short_loop_plan.yaml",
                ],
            }
        ],
        "required_config": ["experiment.gpu_policy"],
        "conditional_config": [
            {"when": "project.mode == c2c", "keys": ["c2c.small_loop", "code_patch"]},
        ],
        "gate_validator": "s2_plan_gate_v1",
    },
    "S3_experiment": {
        "required_inputs": [
            "plan/plan.yaml",
        ],
        "optional_inputs": [
            "plan/code_patches/patch_manifest.json",
        ],
        "conditional_inputs": [
            {
                "when": "execution.collector == c2c_small_loop",
                "paths": [
                    "plan/candidate_ideas.json",
                    "plan/short_loop_plan.yaml",
                    "external/c2c_snapshot",
                ],
            }
        ],
        "required_outputs": [
            "experiment/results/main_results.json",
            "experiment/results/ablation_results.json",
            "experiment/results/hypothesis_verification.md",
            "experiment/gate_report.json",
        ],
        "optional_outputs": [
            "experiment/results/posthoc_review.json",
            "experiment/results/failure_feedback.json",
            "experiment/results/c2c_small_loop_results.json",
            "experiment/code_snapshots",
        ],
        "conditional_outputs": [],
        "required_config": ["experiment"],
        "conditional_config": [{"when": "project.mode == c2c", "keys": ["c2c", "llm.execution_provider"]}],
        "gate_validator": "s3_experiment_gate_v1",
    },
    "S4_writing": {
        "required_inputs": [
            "literature/survey.md",
            "experiment/results/main_results.json",
            "experiment/results/ablation_results.json",
            "experiment/results/hypothesis_verification.md",
        ],
        "optional_inputs": [],
        "conditional_inputs": [],
        "required_outputs": [
            "paper/main.tex",
            "paper/claim_audit.json",
            "paper/compile_report.json",
            "paper/gate_report.json",
        ],
        "optional_outputs": ["paper/sections", "paper/tables", "paper/figures"],
        "conditional_outputs": [],
        "required_config": ["writing"],
        "conditional_config": [],
        "gate_validator": "s4_writing_gate_v1",
    },
    "S5_review": {
        "required_inputs": [
            "paper/main.tex",
            "paper/claim_audit.json",
            "paper/compile_report.json",
        ],
        "optional_inputs": [],
        "conditional_inputs": [],
        "required_outputs": [
            "review/reviewer_A_round_1.md",
            "review/reviewer_B_round_1.md",
            "review/reviewer_C_round_1.md",
            "review/meta_review_round_1.md",
            "review/revision_dispatch.yaml",
            "review/score_history.json",
            "review/gate_report.json",
        ],
        "optional_outputs": ["review/rebuttal.md"],
        "conditional_outputs": [],
        "required_config": ["review"],
        "conditional_config": [],
        "gate_validator": "s5_review_gate_v1",
    },
}


class StageContractManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.contract_dir = project_root / "orchestration" / "stage_contracts"

    def path(self, stage_key: str) -> Path:
        return self.contract_dir / f"{stage_key}.json"

    def initialize_all(self, *, force: bool = False, config: dict[str, Any] | None = None, iteration: int | None = None) -> None:
        for stage_key in STAGE_ORDER:
            self.initialize(stage_key, force=force, config=config, iteration=iteration)

    def initialize(self, stage_key: str, *, force: bool = False, config: dict[str, Any] | None = None, iteration: int | None = None) -> dict[str, Any]:
        if self.path(stage_key).exists() and not force:
            contract = self.load(stage_key)
        else:
            contract = _default_contract(stage_key)
        contract = self._migrate_contract(stage_key, contract)
        contract = self._with_effective_fields(stage_key, contract, config=config, iteration=iteration)
        self.save(stage_key, contract)
        return contract

    def load(self, stage_key: str) -> dict[str, Any]:
        return read_json(self.path(stage_key), default=_default_contract(stage_key)) or _default_contract(stage_key)

    def save(self, stage_key: str, contract: dict[str, Any]) -> None:
        ensure_dir(self.contract_dir)
        contract["schema_version"] = STAGE_CONTRACT_SCHEMA_VERSION
        contract["project_id"] = self.project_root.name
        contract["updated_at"] = now_utc()
        write_json(self.path(stage_key), contract)

    def stage_started(self, stage_key: str, *, iteration: int | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
        contract = self._normalized(stage_key, config=config, iteration=iteration)
        contract["status"] = "running"
        contract["iteration"] = iteration
        contract["started_at"] = now_utc()
        input_paths = _dedupe([*contract.get("required_inputs", []), *contract.get("optional_inputs", [])])
        contract["resolved_inputs"] = self._records(input_paths, required_paths=set(contract.get("required_inputs", [])))
        contract["missing_inputs"] = [item["path"] for item in contract["resolved_inputs"] if not item["exists"] and item["required"]]
        contract["input_hash"] = _combined_hash(contract["resolved_inputs"])
        contract["gate"] = None
        self.save(stage_key, contract)
        return contract

    def gate_recorded(self, stage_key: str, gate_report: dict[str, Any], *, report_path: str | None = None) -> dict[str, Any]:
        contract = self._normalized(stage_key)
        contract["gate"] = {
            "status": gate_report.get("status"),
            "passed": gate_report.get("passed"),
            "reason": gate_report.get("reason"),
            "validator": gate_report.get("validator"),
            "report_path": report_path,
            "check_count": len(gate_report.get("checks") or []),
            "created_at": gate_report.get("created_at") or now_utc(),
        }
        self.save(stage_key, contract)
        return contract

    def stage_completed(
        self,
        stage_key: str,
        *,
        artifacts: list[str] | None = None,
        status: str = "completed",
        reason: str = "",
        config: dict[str, Any] | None = None,
        iteration: int | None = None,
    ) -> dict[str, Any]:
        contract = self._normalized(stage_key, config=config, iteration=iteration)
        contract["status"] = status
        contract["completed_at"] = now_utc()
        contract["reason"] = reason
        required_outputs = list(contract.get("required_outputs", []))
        output_paths = _dedupe([*(artifacts or []), *required_outputs, *contract.get("optional_outputs", [])])
        contract["produced_outputs"] = self._records(output_paths, required_paths=set(required_outputs))
        contract["missing_outputs"] = [item["path"] for item in contract["produced_outputs"] if item["required"] and not item["exists"]]
        contract["output_hash"] = _combined_hash(contract["produced_outputs"])
        self.save(stage_key, contract)
        return contract

    def stage_stopped(
        self,
        stage_key: str,
        *,
        status: str,
        reason: str,
        artifacts: list[str] | None = None,
        config: dict[str, Any] | None = None,
        iteration: int | None = None,
    ) -> dict[str, Any]:
        contract = self.stage_completed(stage_key, artifacts=artifacts, status=status, reason=reason, config=config, iteration=iteration)
        return contract

    def _normalized(self, stage_key: str, *, config: dict[str, Any] | None = None, iteration: int | None = None) -> dict[str, Any]:
        current = self._migrate_contract(stage_key, self.load(stage_key))
        default = _default_contract(stage_key)
        for key in [
            "required_inputs",
            "optional_inputs",
            "conditional_inputs",
            "required_outputs",
            "optional_outputs",
            "conditional_outputs",
            "required_config",
            "conditional_config",
            "gate_validator",
        ]:
            current.setdefault(key, default.get(key))
        current.setdefault("schema_version", STAGE_CONTRACT_SCHEMA_VERSION)
        current.setdefault("project_id", self.project_root.name)
        current.setdefault("stage_key", stage_key)
        current.setdefault("stage", STAGE_LABELS[stage_key])
        current.setdefault("created_at", now_utc())
        return self._with_effective_fields(stage_key, current, config=config, iteration=iteration)

    def _migrate_contract(self, stage_key: str, current: dict[str, Any]) -> dict[str, Any]:
        if current.get("schema_version") == STAGE_CONTRACT_SCHEMA_VERSION:
            return current
        migrated = _default_contract(stage_key)
        for key in ["status", "iteration", "started_at", "completed_at", "gate", "reason", "input_hash", "output_hash"]:
            if key in current:
                migrated[key] = current[key]
        migrated["created_at"] = current.get("created_at") or migrated["created_at"]
        migrated["resolved_inputs"] = current.get("resolved_inputs") or []
        migrated["produced_outputs"] = current.get("produced_outputs") or []
        migrated["missing_inputs"] = current.get("missing_inputs") or []
        migrated["missing_outputs"] = current.get("missing_outputs") or []
        return migrated

    def _with_effective_fields(
        self,
        stage_key: str,
        contract: dict[str, Any],
        *,
        config: dict[str, Any] | None,
        iteration: int | None,
    ) -> dict[str, Any]:
        context = _condition_context(self.project_root, config or {}, iteration)
        required_inputs = list(contract.get("required_inputs") or contract.get("declared_inputs") or [])
        optional_inputs = list(contract.get("optional_inputs") or [])
        required_outputs = list(contract.get("required_outputs") or contract.get("declared_outputs") or [])
        optional_outputs = list(contract.get("optional_outputs") or [])
        required_config = list(contract.get("required_config") or [])
        active_conditions: list[dict[str, Any]] = []
        for item in contract.get("conditional_inputs") or []:
            active = _condition_matches(str(item.get("when") or ""), context)
            active_conditions.append({"kind": "input", "when": item.get("when"), "active": active, "paths": item.get("paths") or []})
            if active:
                required_inputs.extend(item.get("paths") or [])
        for item in contract.get("conditional_outputs") or []:
            active = _condition_matches(str(item.get("when") or ""), context)
            active_conditions.append({"kind": "output", "when": item.get("when"), "active": active, "paths": item.get("paths") or []})
            if active:
                required_outputs.extend(item.get("paths") or [])
        for item in contract.get("conditional_config") or []:
            active = _condition_matches(str(item.get("when") or ""), context)
            active_conditions.append({"kind": "config", "when": item.get("when"), "active": active, "keys": item.get("keys") or []})
            if active:
                required_config.extend(item.get("keys") or [])
        contract["required_inputs"] = _dedupe(required_inputs)
        contract["optional_inputs"] = _dedupe(optional_inputs)
        contract["required_outputs"] = _dedupe(required_outputs)
        contract["optional_outputs"] = _dedupe(optional_outputs)
        contract["required_config"] = _dedupe(required_config)
        contract["active_conditions"] = active_conditions
        contract["declared_inputs"] = _dedupe([*contract["required_inputs"], *contract["optional_inputs"]])
        contract["declared_outputs"] = _dedupe([*contract["required_outputs"], *contract["optional_outputs"]])
        return contract

    def _records(self, paths: list[str], *, required_paths: set[str] | None = None) -> list[dict[str, Any]]:
        required_paths = set(paths) if required_paths is None else required_paths
        return [_path_record(self.project_root, path, required=path in required_paths) for path in paths]


def _default_contract(stage_key: str) -> dict[str, Any]:
    spec = DEFAULT_STAGE_CONTRACTS[stage_key]
    return {
        "schema_version": STAGE_CONTRACT_SCHEMA_VERSION,
        "project_id": "",
        "stage_key": stage_key,
        "stage": STAGE_LABELS[stage_key],
        "status": "pending",
        "iteration": None,
        "created_at": now_utc(),
        "updated_at": now_utc(),
        "required_inputs": list(spec.get("required_inputs", [])),
        "optional_inputs": list(spec.get("optional_inputs", [])),
        "conditional_inputs": list(spec.get("conditional_inputs", [])),
        "required_outputs": list(spec.get("required_outputs", [])),
        "optional_outputs": list(spec.get("optional_outputs", [])),
        "conditional_outputs": list(spec.get("conditional_outputs", [])),
        "required_config": list(spec.get("required_config", [])),
        "conditional_config": list(spec.get("conditional_config", [])),
        "gate_validator": spec["gate_validator"],
        "declared_inputs": _dedupe([*spec.get("required_inputs", []), *spec.get("optional_inputs", [])]),
        "declared_outputs": _dedupe([*spec.get("required_outputs", []), *spec.get("optional_outputs", [])]),
        "active_conditions": [],
        "resolved_inputs": [],
        "produced_outputs": [],
        "missing_inputs": [],
        "missing_outputs": [],
        "input_hash": None,
        "output_hash": None,
        "gate": None,
        "reason": "",
    }


def _path_record(project_root: Path, rel_path: str, *, required: bool) -> dict[str, Any]:
    path = project_root / rel_path
    record: dict[str, Any] = {"path": rel_path, "required": required, "exists": path.exists()}
    if path.exists():
        if path.is_file():
            record.update({"kind": "file", "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        elif path.is_dir():
            record.update({"kind": "directory", "file_count": sum(1 for child in path.rglob("*") if child.is_file())})
    return record


def _combined_hash(records: list[dict[str, Any]]) -> str | None:
    hashes = [f"{item.get('path')}:{item.get('sha256')}" for item in records if item.get("exists") and item.get("sha256")]
    if not hashes:
        return None
    import hashlib

    digest = hashlib.sha256()
    for value in sorted(hashes):
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _condition_context(project_root: Path, config: dict[str, Any], iteration: int | None) -> dict[str, Any]:
    c2c_enabled = bool(config.get("c2c", {}).get("enabled") or (project_root / "external" / "c2c_snapshot").exists())
    collector = _discover_execution_collector(project_root)
    return {
        "project.mode": "c2c" if c2c_enabled else "generic",
        "iteration": int(iteration or 1),
        "execution.collector": collector,
    }


def _discover_execution_collector(project_root: Path) -> str:
    plan_path = project_root / "plan" / "plan.yaml"
    if not plan_path.exists():
        return ""
    try:
        import yaml

        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
        return str((plan.get("execution") or {}).get("collector") or "")
    except Exception:
        return ""


def _condition_matches(condition: str, context: dict[str, Any]) -> bool:
    condition = condition.strip()
    if not condition:
        return False
    if "==" in condition:
        left, right = [part.strip() for part in condition.split("==", 1)]
        right = right.strip("'\"")
        return str(context.get(left, "")) == right
    if ">" in condition:
        left, right = [part.strip() for part in condition.split(">", 1)]
        try:
            return int(context.get(left, 0)) > int(right)
        except ValueError:
            return False
    return False


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
