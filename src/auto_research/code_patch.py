"""Frozen code patch generation, validation, and application."""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.runner import ExperimentRunner, GpuSelection
from .artifacts import ArtifactManager
from .c2c import C2CAdapter, c2c_candidate_config_overrides, c2c_idea_novelty_report, repo_snapshot_manifest
from .llm import codex_subprocess_env
from .s2_planner_contracts import build_s2_5_patch_gate_report
from .utils import ensure_dir, now_utc, read_json, read_yaml, repo_root, sanitize_filename, sha256_file, write_json, write_yaml


DEFAULT_CODE_PATCH_CONFIG = {
    "enabled": False,
    "backend": "codex_persistent_cli",
    "timeout_seconds": 1800,
    "max_candidates": 1,
    "variants_per_candidate": 1,
    "stop_after_first_ok_score": None,
    "materialize_snapshot_baseline": True,
    "codex_json_events": True,
    "no_progress_timeout_seconds": None,
    "worktree_base_ref": "HEAD",
    "worktree_storage_root": None,
    "codex_sandbox": "workspace-write",
    "codex_sandbox_fallback": "danger-full-access",
    "codex_approval_policy": "never",
    "reasoning_effort": "high",
    "implementation_repair_diagnosis": {
        "enabled": True,
        "max_file_reads": 20,
        "allow_lightweight_commands": True,
        "repeated_failure_detection": {
            "enabled": True,
            "min_repeats": 2,
            "changed_files_jaccard_threshold": 0.8,
        },
    },
    "validation": {
        "gate_mode": "discovery",
        "require_py_compile": True,
        "require_targeted_tests": True,
        "require_config_activation": True,
        "runtime_smoke": {
            "enabled": True,
            "train_samples": 8,
            "timeout_seconds": 600,
            "gpu_ids": "auto",
            "min_free_mb": 8192,
            "oom_retry": {"enabled": True, "max_retries": 1},
            "resource_wait": {"enabled": True, "timeout_seconds": 7200, "poll_seconds": 120},
            "skip_if_missing_train_entry": True,
            "mechanism_activation": {
                "enabled": True,
                "hard_gate": True,
                "require_switch_in_disabled_eval_config": True,
                "require_switch_referenced_in_runtime_code": True,
                "forward_probe": {
                    "enabled": True,
                    "hard_gate": True,
                    "script": "script/auto_research/activation_forward_probe.py",
                    "builtin_fallback": True,
                    "timeout_seconds": 180,
                    "min_changed_fields": 1,
                },
                "runtime_code_files": [
                    "rosetta/model/aligner.py",
                    "rosetta/model/projector.py",
                    "rosetta/model/wrapper.py",
                ],
            },
        },
        "max_repair_attempts": 1,
        "max_contract_repair_attempts": 1,
        "max_changed_files": 6,
        "strict_max_changed_files": False,
        "auto_prune_over_scope_files": False,
        "repair_eval_code_changes": True,
        "repair_test_only_changes": True,
        "mechanism_self_review": {
            "enabled": True,
            "mode": "runnable_first",
            "require_mechanism_evidence": True,
            "require_ablation_wired": False,
            "require_coverage_evidence": False,
            "require_matched_coverage_evidence": False,
            "warn_large_diff_lines": 900,
            "block_evaluator_like_files": True,
        },
    },
    "dynamic_whitelist": {
        "include_prefixes": ["rosetta/", "script/", "recipe/", "test/", "tests/"],
        "include_extensions": [".py", ".json", ".yaml", ".yml", ".toml", ".txt"],
        "include_root_globs": ["pyproject.toml", "requirements*.txt"],
        "exclude_prefixes": [
            ".git/",
            "wandb/",
            "__pycache__/",
            "local/checkpoints/",
            "local/snapshots/",
            "local/final_results/",
            "data/",
            "datasets/",
            "models/",
        ],
        "exclude_extensions": [".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".parquet", ".arrow"],
    },
}

CORE_PATCH_BASELINE_FILES = [
    "rosetta/model/aligner.py",
    "rosetta/model/projector.py",
    "rosetta/model/wrapper.py",
    "script/train/SFT_train.py",
    "script/evaluation/unified_evaluator.py",
    "recipe/train_recipe/C2C_0.6+0.5.json",
    "recipe/eval_recipe/unified_eval.yaml",
]


@dataclass
class DynamicEditPolicy:
    include_prefixes: list[str]
    include_extensions: list[str]
    exclude_prefixes: list[str]
    exclude_extensions: list[str]
    include_root_globs: list[str]

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "DynamicEditPolicy":
        merged = dict(DEFAULT_CODE_PATCH_CONFIG["dynamic_whitelist"])
        if config:
            merged.update(config)
        return cls(
            include_prefixes=list(merged.get("include_prefixes") or []),
            include_extensions=list(merged.get("include_extensions") or []),
            exclude_prefixes=list(merged.get("exclude_prefixes") or []),
            exclude_extensions=list(merged.get("exclude_extensions") or []),
            include_root_globs=list(merged.get("include_root_globs") or []),
        )

    def allowed(self, path_value: str, *, repo_root: Path | None = None) -> bool:
        return self.validate_path(path_value, repo_root=repo_root) is None

    def validate_path(self, path_value: str, *, repo_root: Path | None = None) -> str | None:
        if not path_value:
            return "empty path"
        path = Path(path_value)
        if path.is_absolute() or ".." in path.parts:
            return "absolute paths and parent traversal are not allowed"
        normalized = path.as_posix()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if any(normalized == prefix.rstrip("/") or normalized.startswith(prefix.rstrip("/") + "/") for prefix in self.exclude_prefixes):
            return f"path is excluded by prefix: {normalized}"
        if Path(normalized).suffix in self.exclude_extensions:
            return f"path is excluded by extension: {normalized}"
        if repo_root is not None:
            root = repo_root.resolve()
            target = (repo_root / normalized)
            existing = target if target.exists() else _nearest_existing_parent(target)
            try:
                resolved = existing.resolve()
            except OSError:
                return f"path cannot be resolved safely: {normalized}"
            if root != resolved and root not in resolved.parents:
                return f"path escapes repository via symlink: {normalized}"
        if "/" not in normalized and any(Path(normalized).match(pattern) for pattern in self.include_root_globs):
            return None
        if any(normalized.startswith(prefix) for prefix in self.include_prefixes) and Path(normalized).suffix in self.include_extensions:
            return None
        return f"path is outside dynamic edit whitelist: {normalized}"


class FrozenPatchGuard:
    def __init__(self, policy: DynamicEditPolicy):
        self.policy = policy

    def apply(self, repo_root: Path, patch_json: dict[str, Any]) -> dict[str, Any]:
        operations = patch_json.get("operations") or []
        if not isinstance(operations, list):
            return {"status": "rejected", "errors": ["operations must be a list"], "changed_files": [], "restore_state": []}
        preflight_errors: list[str] = []
        for idx, operation in enumerate(operations):
            if not isinstance(operation, dict):
                preflight_errors.append(f"operation {idx}: must be an object")
                continue
            op = operation.get("op")
            if op not in {"replace_file", "add_file"}:
                preflight_errors.append(f"operation {idx}: unsupported op {op}")
            rel_path = str(operation.get("path") or "")
            reason = self.policy.validate_path(rel_path, repo_root=repo_root)
            if reason:
                preflight_errors.append(f"operation {idx}: {reason}")
        if preflight_errors:
            return {"status": "rejected", "errors": preflight_errors, "changed_files": [], "restore_state": []}
        restore_state = self.snapshot(repo_root, operations)
        changed: list[str] = []
        errors: list[str] = []
        try:
            for idx, operation in enumerate(operations):
                if not isinstance(operation, dict):
                    errors.append(f"operation {idx}: must be an object")
                    break
                op = operation.get("op")
                rel_path = str(operation.get("path") or "")
                reason = self.policy.validate_path(rel_path, repo_root=repo_root)
                if reason:
                    errors.append(f"operation {idx}: {reason}")
                    break
                target = repo_root / rel_path
                if op == "replace_file":
                    if not target.exists() or not target.is_file():
                        errors.append(f"operation {idx}: missing target file: {rel_path}")
                        break
                    expected = str(operation.get("old_sha256") or "")
                    actual = sha256_file(target)
                    if not expected:
                        errors.append(f"operation {idx}: missing old_sha256 for {rel_path}")
                        break
                    if expected != actual:
                        errors.append(f"operation {idx}: sha256 mismatch for {rel_path}")
                        break
                    target.write_text(str(operation.get("new") or ""), encoding="utf-8")
                    changed.append(rel_path)
                elif op == "add_file":
                    if target.exists():
                        errors.append(f"operation {idx}: add_file target already exists: {rel_path}")
                        break
                    ensure_dir(target.parent)
                    target.write_text(str(operation.get("new") or ""), encoding="utf-8")
                    changed.append(rel_path)
                else:
                    errors.append(f"operation {idx}: unsupported op {op}")
                    break
            if errors:
                self.restore(repo_root, restore_state)
                return {"status": "rejected", "errors": errors, "changed_files": [], "restore_state": restore_state}
            return {"status": "applied", "errors": [], "changed_files": changed, "restore_state": restore_state}
        except OSError as exc:
            self.restore(repo_root, restore_state)
            return {"status": "rejected", "errors": [str(exc)], "changed_files": [], "restore_state": restore_state}

    def snapshot(self, repo_root: Path, operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        state = []
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            rel_path = str(operation.get("path") or "")
            if not rel_path or rel_path in seen:
                continue
            seen.add(rel_path)
            target = repo_root / rel_path
            if target.exists() and target.is_file():
                state.append({"path": rel_path, "existed": True, "content": target.read_text(encoding="utf-8", errors="ignore")})
            else:
                state.append({"path": rel_path, "existed": False, "content": ""})
        return state

    @staticmethod
    def restore(repo_root: Path, repo_state_snapshot: list[dict[str, Any]]) -> None:
        for item in repo_state_snapshot:
            target = repo_root / str(item.get("path") or "")
            if item.get("existed"):
                ensure_dir(target.parent)
                target.write_text(str(item.get("content") or ""), encoding="utf-8")
            elif target.exists():
                target.unlink()


class CodexPersistentPatchBackend:
    def __init__(self, config: dict[str, Any], project_root: Path):
        self.config = config
        self.project_root = project_root

    def generate(self, implementation_contract: dict[str, Any], temp_repo: Path, edit_policy: DynamicEditPolicy) -> dict[str, Any]:
        if not shutil.which("codex"):
            return {"status": "codex_failed", "reason": "codex executable not found"}
        workspace = _worktree_workspace_from_contract(implementation_contract)
        if not workspace:
            return {"status": "codex_failed", "reason": "persistent Codex backend requires code_worktree metadata"}
        repo = Path(workspace["repo"])
        if not repo.exists():
            return {"status": "codex_failed", "reason": f"persistent Codex worktree missing: {repo}"}
        code_patch_config = _code_patch_config(self.config)
        primary_sandbox = str(code_patch_config.get("codex_sandbox") or "workspace-write")
        llm_codex_config = (
            self.config.get("llm", {}).get("codex_cli", {})
            if isinstance(self.config.get("llm", {}), dict)
            else {}
        )
        preload_sandbox = str(
            code_patch_config.get("codex_preload_sandbox")
            or code_patch_config.get("codex_sandbox")
            or llm_codex_config.get("sandbox")
            or "read-only"
        )
        recovery_actions: list[dict[str, Any]] = []
        if not _load_persistent_codex_session(Path(str(workspace.get("session_path") or ""))):
            preload = self._run_persistent_codex_once_inner(
                implementation_contract,
                repo,
                edit_policy,
                workspace,
                sandbox=preload_sandbox,
                force_new_session=False,
                prompt_kind="preload",
            )
            if preload.get("status") != "ok":
                return preload
            _write_persistent_patch_blueprint(workspace, preload)
            recovery_actions.append(
                {
                    "action": "codex_context_preload",
                    "status": preload.get("status"),
                    "session_id": preload.get("session_id"),
                    "duration_seconds": (preload.get("codex_call") or {}).get("duration_seconds"),
                }
            )
        primary = self._run_persistent_codex_once(implementation_contract, repo, edit_policy, workspace, sandbox=primary_sandbox)
        if recovery_actions:
            primary["recovery_actions"] = [*recovery_actions, *primary.get("recovery_actions", [])]
        fallback_sandbox = str(code_patch_config.get("codex_sandbox_fallback") or "")
        if (
            fallback_sandbox
            and fallback_sandbox != primary_sandbox
            and _codex_sandbox_error(primary)
        ):
            fallback = self._run_persistent_codex_once(implementation_contract, repo, edit_policy, workspace, sandbox=fallback_sandbox)
            action = {
                "action": "retry_codex_with_fallback_sandbox",
                "status": fallback.get("status"),
                "primary_sandbox": primary_sandbox,
                "fallback_sandbox": fallback_sandbox,
                "reason": _codex_sandbox_reason(primary),
            }
            fallback["recovery_actions"] = [*primary.get("recovery_actions", []), action, *fallback.get("recovery_actions", [])]
            fallback["primary_attempt"] = _compact_backend_attempt(primary)
            return fallback
        return primary

    def _run_persistent_codex_once(
        self,
        implementation_contract: dict[str, Any],
        repo: Path,
        edit_policy: DynamicEditPolicy,
        workspace: dict[str, Any],
        *,
        sandbox: str,
        prompt_kind: str = "patch",
    ) -> dict[str, Any]:
        result = self._run_persistent_codex_once_inner(
            implementation_contract,
            repo,
            edit_policy,
            workspace,
            sandbox=sandbox,
            force_new_session=False,
            prompt_kind=prompt_kind,
        )
        if result.get("status") == "codex_failed" and result.get("resume_failed") and result.get("session_id"):
            retry = self._run_persistent_codex_once_inner(
                implementation_contract,
                repo,
                edit_policy,
                workspace,
                sandbox=sandbox,
                force_new_session=True,
                prompt_kind=prompt_kind,
            )
            action = {
                "action": "retry_codex_with_new_persistent_session",
                "status": retry.get("status"),
                "failed_session_id": result.get("session_id"),
                "reason": result.get("reason"),
            }
            retry["recovery_actions"] = [*result.get("recovery_actions", []), action, *retry.get("recovery_actions", [])]
            retry["primary_attempt"] = _compact_backend_attempt(result)
            return retry
        return result

    def _run_persistent_codex_once_inner(
        self,
        implementation_contract: dict[str, Any],
        repo: Path,
        edit_policy: DynamicEditPolicy,
        workspace: dict[str, Any],
        *,
        sandbox: str,
        force_new_session: bool,
        prompt_kind: str,
    ) -> dict[str, Any]:
        code_patch_config = _code_patch_config(self.config)
        session_path = Path(workspace["session_path"])
        events_path = Path(workspace["events_path"])
        prompt = _persistent_codex_prompt(implementation_contract, edit_policy, prompt_kind=prompt_kind)
        session_id = None if force_new_session else _load_persistent_codex_session(session_path)
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as handle:
            output_path = Path(handle.name)
        command = [
            "codex",
            "-s",
            sandbox,
            "-a",
            str(code_patch_config.get("codex_approval_policy") or "never"),
            "exec",
            "--skip-git-repo-check",
            "--output-last-message",
            str(output_path),
        ]
        if code_patch_config.get("codex_json_events", True):
            command.append("--json")
        model = str(self.config.get("llm", {}).get("model") or "")
        if model:
            command.extend(["-m", model])
        reasoning_effort = str(
            code_patch_config.get("reasoning_effort")
            or self.config.get("llm", {}).get("reasoning_effort")
            or ""
        ).strip()
        if reasoning_effort and reasoning_effort != "none":
            command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        command.extend(["-C", str(repo)])
        if session_id:
            command.extend(["resume", session_id, "-"])
        else:
            command.append("-")
        started_at = now_utc()
        start_monotonic = time.monotonic()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                input=prompt,
                cwd=repo,
                timeout=int(code_patch_config.get("timeout_seconds") or 1800),
                env=codex_subprocess_env(self.config),
            )
            message = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        except subprocess.TimeoutExpired as exc:
            output_text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            stdout = _decode_timeout_output(exc.stdout)
            stderr = _decode_timeout_output(exc.stderr) + f"\nTimeout after {int(code_patch_config.get('timeout_seconds') or 1800)}s"
            _append_codex_events(
                events_path,
                stdout,
                _codex_call_record(
                    started_at,
                    start_monotonic,
                    session_id=session_id,
                    parsed_session_id=None,
                    command_kind="resume" if session_id else "new",
                    prompt_kind=prompt_kind,
                    returncode=124,
                ),
            )
            return {
                "status": "codex_failed",
                "reason": f"codex timed out: {exc}",
                "rationale": output_text.strip(),
                "stdout": stdout[-2000:],
                "stderr": stderr[-2000:],
                "sandbox": sandbox,
                "session_id": session_id,
                "failure_category": "codex_cli_timeout",
            }
        finally:
            output_path.unlink(missing_ok=True)
        parsed_session_id = _parse_codex_session_id(result.stderr, result.stdout) or session_id
        call_record = _codex_call_record(
            started_at,
            start_monotonic,
            session_id=session_id,
            parsed_session_id=parsed_session_id,
            command_kind="resume" if session_id else "new",
            prompt_kind=prompt_kind,
            returncode=result.returncode,
        )
        _append_codex_events(events_path, result.stdout, call_record)
        if parsed_session_id:
            _save_persistent_codex_session(
                session_path,
                parsed_session_id,
                self.config,
                workspace=workspace,
                call_record=call_record,
            )
            _sync_project_codex_session(self.project_root, str(workspace.get("session_key") or ""), parsed_session_id, self.config, workspace=workspace)
        if result.returncode != 0:
            reason = result.stderr[-2000:] or result.stdout[-2000:] or f"codex exited {result.returncode}"
            retryable = _codex_retryable_error_text(reason, result.stderr)
            return {
                "status": "retryable_codex_failed" if retryable else "codex_failed",
                "reason": reason,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
                "sandbox": sandbox,
                "retryable": retryable,
                "failure_category": "llm_rate_limit_or_quota" if retryable else "codex_cli_failure",
                "session_id": parsed_session_id or session_id,
                "resume_failed": bool(session_id and not retryable),
                "same_session_reused": bool(session_id and (parsed_session_id or session_id) == session_id),
                "previous_session_id": session_id,
                "codex_call": call_record,
                "code_worktree": _compact_code_worktree(workspace),
            }
        return {
            "status": "ok",
            "rationale": message.strip(),
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
            "sandbox": sandbox,
            "session_id": parsed_session_id or session_id,
            "same_session_reused": bool(session_id and (parsed_session_id or session_id) == session_id),
            "previous_session_id": session_id,
            "codex_call": call_record,
            "code_worktree": _compact_code_worktree(workspace),
        }


class CodePatchAgent:
    stage_key = "S2_plan"

    def __init__(self, project_root: Path, config: dict[str, Any], artifacts: ArtifactManager, backend: Any | None = None):
        self.project_root = project_root
        self.config = config
        self.artifacts = artifacts
        self.backend = backend or _default_code_patch_backend(config, project_root)

    def run_selected_variant(
        self,
        plan: dict[str, Any],
        selected_variant: dict[str, Any],
        implementation_contract: dict[str, Any],
        planner_gate_report: dict[str, Any],
        variant_fingerprint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        code_patch_config = _code_patch_config(self.config)
        if not code_patch_config.get("enabled", False):
            return {"status": "disabled", "candidates": [], "artifacts": []}
        candidate = dict(selected_variant)
        candidate["selected"] = True
        candidate["id"] = str(candidate.get("id") or implementation_contract.get("variant_id") or "selected_variant")
        candidate["variant_fingerprint"] = str(candidate.get("variant_fingerprint") or implementation_contract.get("variant_fingerprint") or "")
        plan = dict(plan)
        plan["selected_idea"] = candidate
        plan["candidate_ideas"] = [candidate]
        plan["next_variant"] = candidate
        plan["selected_variant_candidates"] = [{"id": candidate.get("id"), "variant_fingerprint": candidate.get("variant_fingerprint")}]
        contract_record = self.artifacts.write_json(
            self.stage_key,
            "code_patches/implementation_contract.json",
            implementation_contract,
            artifact_type="c2c_s2_5_implementation_contract",
            summary="S2.5 implementation contract generated after planner gate pass",
            source_paths=["plan/s2_planner/planner_gate_report.json", "plan/s2_planner/next_variant.json", "plan/variant_contract.json"],
        )
        manifest = self.run(plan, [candidate])
        patch_gate_report = build_s2_5_patch_gate_report(
            patch_manifest=manifest,
            implementation_contract=implementation_contract,
            planner_gate_report=planner_gate_report,
            variant_fingerprint=variant_fingerprint,
            config=self.config,
        )
        patch_gate_record = self.artifacts.write_json(
            self.stage_key,
            "code_patches/patch_gate_report.json",
            patch_gate_report,
            artifact_type="c2c_s2_5_patch_gate_report",
            summary="S2.5 patch gate report against selected planner variant",
            source_paths=["plan/code_patches/implementation_contract.json", "plan/code_patches/patch_manifest.json"],
        )
        artifacts = list(manifest.get("artifacts") or [])
        artifacts.extend([contract_record["path"], patch_gate_record["path"]])
        manifest["artifacts"] = artifacts
        manifest["implementation_contract_path"] = contract_record["path"]
        manifest["patch_gate_report_path"] = patch_gate_record["path"]
        manifest["patch_gate"] = patch_gate_report.get("gate")
        return manifest

    def run(self, plan: dict[str, Any], candidate_ideas: list[dict[str, Any]]) -> dict[str, Any]:
        code_patch_config = _code_patch_config(self.config)
        if not code_patch_config.get("enabled", False):
            return {"status": "disabled", "candidates": [], "artifacts": []}
        adapter = C2CAdapter(self.project_root, self.config)
        policy = DynamicEditPolicy.from_config(code_patch_config.get("dynamic_whitelist") or {})
        previous_failures = _load_previous_patch_failures(self.project_root)
        manifest_candidates = []
        artifacts = []
        selection = _select_single_s2_5_candidate(plan, candidate_ideas)
        selected_candidate = selection.get("candidate") if isinstance(selection.get("candidate"), dict) else None
        skipped_candidates: list[dict[str, Any]] = []
        if selected_candidate is None:
            for candidate in candidate_ideas:
                if not isinstance(candidate, dict):
                    continue
                candidate["code_patch"] = {
                    "status": "skipped_no_s2_5_selected_candidate",
                    "reason": "S2.5 could not identify a selected S2 candidate to implement.",
                }
                skipped_candidates.append(_compact_skipped_patch_candidate(candidate, reason="no_selected_candidate"))
        else:
            selected_index = selection.get("selected_index")
            for idx, candidate in enumerate(candidate_ideas):
                if not isinstance(candidate, dict):
                    continue
                if idx == selected_index:
                    continue
                candidate["code_patch"] = {
                    "status": "skipped_s2_5_single_candidate_mode",
                    "reason": "S2.5 implements only the S2-selected variant; alternative variants stay in S2 planner history.",
                    "selection_policy": "single_s2_selected_variant",
                }
                skipped_candidates.append(
                    _compact_skipped_patch_candidate(candidate, reason="single_s2_selected_variant")
                )
            candidate = selected_candidate
            result = self._generate_candidate_patch(adapter, policy, candidate, plan, previous_failures=previous_failures)
            candidate["code_patch"] = result["code_patch"]
            manifest_candidates.append(result["manifest_entry"])
            artifacts.extend(result.get("artifacts", []))
        retryable_patch_count = sum(1 for item in manifest_candidates if _patch_failure_retryable(item))
        valid_patch_count = sum(1 for item in manifest_candidates if item.get("status") == "ok")
        valid_patch_ids = [str(item.get("candidate_id")) for item in manifest_candidates if item.get("status") == "ok" and item.get("candidate_id")]
        selected_patch = _compact_selected_manifest_patch(next((item for item in manifest_candidates if item.get("status") == "ok"), {}))
        if valid_patch_count:
            manifest_status = "ok"
        elif retryable_patch_count:
            manifest_status = "retryable_no_valid_patch"
        else:
            manifest_status = "no_valid_patch"
        manifest = {
            "status": manifest_status,
            "created_at": now_utc(),
            "backend": "codex_persistent_cli",
            "selection_policy": {
                "mode": "single_s2_selected_variant",
                "selected_by": selection.get("selected_by"),
                "selected_index": selection.get("selected_index"),
                "selected_candidate_id": selection.get("selected_candidate_id"),
                "input_candidate_count": len([item for item in candidate_ideas if isinstance(item, dict)]),
                "implementation_candidate_count": len(manifest_candidates),
                "repair_loop": "same_candidate_persistent_session",
                "ignored_legacy_config": {
                    "max_candidates": code_patch_config.get("max_candidates"),
                    "variants_per_candidate": code_patch_config.get("variants_per_candidate"),
                    "stop_after_first_ok_score": code_patch_config.get("stop_after_first_ok_score"),
                },
            },
            "candidate_count": len(manifest_candidates),
            "input_candidate_count": len([item for item in candidate_ideas if isinstance(item, dict)]),
            "skipped_candidate_count": len(skipped_candidates),
            "valid_patch_count": valid_patch_count,
            "valid_patch_ids": valid_patch_ids,
            "failed_patch_count": sum(1 for item in manifest_candidates if item.get("status") not in {"ok", "skipped_not_in_patch_budget"}),
            "retryable_patch_count": retryable_patch_count,
            "retryable": bool(retryable_patch_count and not valid_patch_count),
            "selected_candidate_id": selected_patch.get("candidate_id"),
            "selected_patch": selected_patch,
            "policy": {
                "include_prefixes": policy.include_prefixes,
                "include_extensions": policy.include_extensions,
                "exclude_prefixes": policy.exclude_prefixes,
                "exclude_extensions": policy.exclude_extensions,
                "include_root_globs": policy.include_root_globs,
            },
            "candidates": manifest_candidates,
            "patches": manifest_candidates,
            "skipped_candidates": skipped_candidates,
        }
        manifest_record = self.artifacts.write_json(
            self.stage_key,
            "code_patches/patch_manifest.json",
            manifest,
            artifact_type="c2c_code_patch_manifest",
            summary="Frozen S2.5 code patch manifest",
        )
        artifacts.append(manifest_record["path"])
        manifest["artifacts"] = artifacts
        return manifest

    def _generate_candidate_patch(
        self,
        adapter: C2CAdapter,
        policy: DynamicEditPolicy,
        candidate: dict[str, Any],
        plan: dict[str, Any],
        *,
        previous_failures: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        result = self._generate_selected_candidate_patch_once(
            adapter,
            policy,
            candidate,
            plan,
            previous_failures=previous_failures,
        )
        attempt = _compact_single_patch_attempt(result.get("manifest_entry") or {})
        attempts.append(attempt)
        _annotate_single_patch_attempt_result(
            result,
            attempts=attempts,
            selection_reason="single_s2_selected_variant",
        )
        return result

    def _generate_selected_candidate_patch_once(
        self,
        adapter: C2CAdapter,
        policy: DynamicEditPolicy,
        candidate: dict[str, Any],
        plan: dict[str, Any],
        *,
        previous_failures: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        idea_id = sanitize_filename(str(candidate.get("id") or candidate.get("title") or "candidate"))
        base_rel = f"code_patches/{idea_id}"
        implementation_contract = _build_implementation_contract(
            candidate,
            plan,
            policy,
            previous_failure=_previous_failure_for_candidate(previous_failures, idea_id, candidate),
        )
        repeated_failure_context = _repeated_implementation_failure_context(
            self.project_root,
            implementation_contract,
            self.config,
        )
        if repeated_failure_context.get("is_repeated"):
            implementation_contract = _implementation_contract_with_repeated_failure_context(
                implementation_contract,
                repeated_failure_context,
            )
        worktree_workspace = _prepare_code_worktree_workspace(
            self.project_root,
            self.config,
            idea_id=idea_id,
            variant_index=1,
            enabled=isinstance(self.backend, CodexPersistentPatchBackend),
        )
        if worktree_workspace.get("status") != "ok":
            reason = str(worktree_workspace.get("reason") or "failed to prepare persistent code worktree")
            return self._write_failed_patch_artifacts(
                base_rel,
                idea_id,
                candidate,
                worktree_workspace.get("status", "codex_failed"),
                reason,
                implementation_contract=implementation_contract,
                prompt_text=_codex_patch_prompt(implementation_contract, policy),
                recovery_actions=list(worktree_workspace.get("recovery_actions") or []),
            )
        if worktree_workspace.get("enabled"):
            implementation_contract = _implementation_contract_with_worktree(
                implementation_contract,
                worktree_workspace,
            )
        workspace_recovery_actions = list(worktree_workspace.get("recovery_actions") or [])
        diagnosis_record: dict[str, Any] | None = None
        diagnosis_payload: dict[str, Any] = {}
        repeated_failure_context = _repeated_failure_context_from_contract(implementation_contract)
        if _implementation_repair_diagnosis_enabled(self.config, implementation_contract):
            diagnosis_record, diagnosis_payload = self._run_implementation_repair_diagnosis(
                base_rel,
                implementation_contract,
                worktree_workspace,
                adapter.env_python,
                policy,
                recovery_actions=workspace_recovery_actions,
            )
            if diagnosis_payload:
                implementation_contract = _implementation_contract_with_repair_diagnosis(
                    implementation_contract,
                    diagnosis_payload,
                )
            if diagnosis_record:
                workspace_recovery_actions.append(
                    {
                        "action": "s2_5_implementation_repair_diagnosis",
                        "status": diagnosis_payload.get("status") or "unknown",
                        "path": diagnosis_record.get("path"),
                        "same_session_reused": diagnosis_payload.get("same_session_reused"),
                    }
                )
        prompt_text = _codex_patch_prompt(implementation_contract, policy)
        workspace_context = _patch_workspace_context(adapter.repo_root, worktree_workspace, idea_id=idea_id)
        with workspace_context as temp_repo:
            pre_codex_action = _auto_prune_worktree_scope_before_codex(
                adapter.repo_root,
                temp_repo,
                policy,
                candidate,
                implementation_contract,
                self.config,
            )
            if pre_codex_action:
                workspace_recovery_actions.append(pre_codex_action)
            backend_result = self.backend.generate(implementation_contract, temp_repo, policy)
            if workspace_recovery_actions:
                backend_result["recovery_actions"] = [
                    *workspace_recovery_actions,
                    *backend_result.get("recovery_actions", []),
                ]
            if repeated_failure_context:
                backend_result["repeated_failure_context"] = repeated_failure_context
            contract_repair_attempts = int(
                (_code_patch_config(self.config).get("validation", {}) or {}).get("max_contract_repair_attempts", 1)
                or 0
            )
            contract_repair_attempt = 0
            while (
                backend_result.get("status") != "ok"
                and not _patch_failure_retryable(backend_result)
                and contract_repair_attempt < contract_repair_attempts
            ):
                contract_repair_attempt += 1
                repair = self._retry_backend_after_contract_failure(
                    backend_result,
                    implementation_contract,
                    temp_repo,
                    policy,
                    {
                        "status": backend_result.get("status", "codex_failed"),
                        "reason": backend_result.get("reason", "codex failed"),
                        "rationale": backend_result.get("rationale", ""),
                    },
                    attempt=contract_repair_attempt,
                )
                if repair is None:
                    break
                backend_result = repair
                implementation_contract = repair.get("implementation_contract", implementation_contract)
                prompt_text = _codex_patch_prompt(implementation_contract, policy)
            if backend_result.get("status") != "ok":
                return self._write_failed_patch_artifacts(
                    base_rel,
                    idea_id,
                    candidate,
                    backend_result.get("status", "codex_failed"),
                    backend_result.get("reason", "codex failed"),
                    rationale=backend_result.get("rationale", ""),
                    implementation_contract=implementation_contract,
                    prompt_text=prompt_text,
                    recovery_actions=backend_result.get("recovery_actions", []),
                )
            draft, scope_action = _build_pruned_patch_from_repo_delta(
                adapter.repo_root,
                temp_repo,
                policy,
                candidate,
                implementation_contract,
                self.config,
            )
            if scope_action:
                backend_result["recovery_actions"] = [*backend_result.get("recovery_actions", []), scope_action]
            if draft["status"] == "blocked_no_executable_change" and _codex_sandbox_error(backend_result):
                fallback = self._retry_backend_after_noop_sandbox_error(
                    backend_result,
                    implementation_contract,
                    temp_repo,
                    policy,
                )
                if fallback is not None:
                    backend_result = fallback
                    if backend_result.get("status") != "ok":
                        return self._write_failed_patch_artifacts(
                            base_rel,
                            idea_id,
                            candidate,
                            backend_result.get("status", "codex_failed"),
                            backend_result.get("reason", "codex failed"),
                            rationale=backend_result.get("rationale", ""),
                            implementation_contract=implementation_contract,
                            prompt_text=prompt_text,
                            recovery_actions=backend_result.get("recovery_actions", []),
                        )
                    draft, scope_action = _build_pruned_patch_from_repo_delta(
                        adapter.repo_root,
                        temp_repo,
                        policy,
                        candidate,
                        implementation_contract,
                        self.config,
                    )
                    if scope_action:
                        backend_result["recovery_actions"] = [*backend_result.get("recovery_actions", []), scope_action]
            while draft["status"] != "ok" and contract_repair_attempt < contract_repair_attempts:
                contract_repair_attempt += 1
                repair = self._retry_backend_after_contract_failure(
                    backend_result,
                    implementation_contract,
                    temp_repo,
                    policy,
                    draft,
                    attempt=contract_repair_attempt,
                )
                if repair is None:
                    break
                backend_result = repair
                implementation_contract = repair.get("implementation_contract", implementation_contract)
                prompt_text = _codex_patch_prompt(implementation_contract, policy)
                draft, scope_action = _build_pruned_patch_from_repo_delta(
                    adapter.repo_root,
                    temp_repo,
                    policy,
                    candidate,
                    implementation_contract,
                    self.config,
                )
                if scope_action:
                    backend_result["recovery_actions"] = [*backend_result.get("recovery_actions", []), scope_action]
            if draft["status"] != "ok":
                return self._write_failed_patch_artifacts(
                    base_rel,
                    idea_id,
                    candidate,
                    draft["status"],
                    "; ".join(draft.get("errors", [])),
                    rationale=backend_result.get("rationale", ""),
                    implementation_contract=implementation_contract,
                    prompt_text=prompt_text,
                    recovery_actions=backend_result.get("recovery_actions", []),
                )
            risk_check = _validate_patch_proxy_risk(draft, candidate, self.config)
            while risk_check["status"] != "ok" and contract_repair_attempt < contract_repair_attempts:
                contract_repair_attempt += 1
                restore_action = _restore_proxy_risk_files(adapter.repo_root, temp_repo, risk_check)
                if restore_action:
                    backend_result["recovery_actions"] = [*backend_result.get("recovery_actions", []), restore_action]
                repair = self._retry_backend_after_contract_failure(
                    backend_result,
                    implementation_contract,
                    temp_repo,
                    policy,
                    risk_check,
                    attempt=contract_repair_attempt,
                )
                if repair is None:
                    break
                backend_result = repair
                implementation_contract = repair.get("implementation_contract", implementation_contract)
                prompt_text = _codex_patch_prompt(implementation_contract, policy)
                draft, scope_action = _build_pruned_patch_from_repo_delta(
                    adapter.repo_root,
                    temp_repo,
                    policy,
                    candidate,
                    implementation_contract,
                    self.config,
                )
                if scope_action:
                    backend_result["recovery_actions"] = [*backend_result.get("recovery_actions", []), scope_action]
                if draft["status"] != "ok":
                    return self._write_failed_patch_artifacts(
                        base_rel,
                        idea_id,
                        candidate,
                        draft["status"],
                        "; ".join(draft.get("errors", [])),
                        rationale=backend_result.get("rationale", ""),
                        implementation_contract=implementation_contract,
                        prompt_text=prompt_text,
                        recovery_actions=backend_result.get("recovery_actions", []),
                    )
                risk_check = _validate_patch_proxy_risk(draft, candidate, self.config)
            if risk_check["status"] != "ok":
                return self._write_failed_patch_artifacts(
                    base_rel,
                    idea_id,
                    candidate,
                    risk_check["status"],
                    risk_check.get("reason", "patch risk check failed"),
                    rationale=backend_result.get("rationale", ""),
                    implementation_contract=implementation_contract,
                    prompt_text=prompt_text,
                    recovery_actions=backend_result.get("recovery_actions", []),
                    risk_check=risk_check,
                    changed_files=draft.get("changed_files") or [],
                    diff_text=draft.get("diff", ""),
                )
            mechanism_review = _mechanism_self_review(draft, implementation_contract, self.config)
            validation = _validate_patch_repo(temp_repo, draft["changed_files"], adapter.env_python, self.config, candidate=candidate)
            activation = _validate_config_activation(draft.get("diff", ""), candidate, self.config)
            validation["activation_check"] = activation
            validation["risk_check"] = risk_check
            validation["mechanism_review"] = mechanism_review
            validation["recovery_actions"] = backend_result.get("recovery_actions", [])
            if _validation_has_runtime_resource_retry(validation):
                validation["retryable"] = True
                validation["resource_retry"] = True
                validation["failure_category"] = "runtime_smoke_resource_retry"
            max_repair_attempts = int(
                (_code_patch_config(self.config).get("validation", {}) or {}).get("max_repair_attempts", 1)
                or 0
            )
            repair_attempt = 0
            while _patch_needs_repair(
                validation,
                activation,
                risk_check,
                mechanism_review,
                draft,
                self.config,
            ) and repair_attempt < max_repair_attempts:
                repair_attempt += 1
                restore_action = _restore_proxy_risk_files(adapter.repo_root, temp_repo, risk_check)
                if restore_action:
                    backend_result["recovery_actions"] = [*backend_result.get("recovery_actions", []), restore_action]
                repair = self._retry_backend_after_validation_failure(
                    backend_result,
                    implementation_contract,
                    temp_repo,
                    policy,
                    validation,
                    draft,
                    attempt=repair_attempt,
                )
                if repair is None:
                    break
                if repair.get("status") != "ok":
                    if _patch_failure_retryable(repair):
                        return self._write_failed_patch_artifacts(
                            base_rel,
                            idea_id,
                            candidate,
                            repair.get("status", "retryable_codex_failed"),
                            repair.get("reason", "retryable Codex failure during patch repair"),
                            rationale=repair.get("rationale", ""),
                            implementation_contract=repair.get("implementation_contract", implementation_contract),
                            prompt_text=_codex_patch_prompt(repair.get("implementation_contract", implementation_contract), policy),
                            recovery_actions=repair.get("recovery_actions", []),
                            risk_check=risk_check,
                            changed_files=draft.get("changed_files") or [],
                            diff_text=draft.get("diff", ""),
                        )
                    validation["recovery_actions"] = repair.get("recovery_actions", [])
                    break
                backend_result = repair
                implementation_contract = repair.get("implementation_contract", implementation_contract)
                prompt_text = _codex_patch_prompt(implementation_contract, policy)
                draft, scope_action = _build_pruned_patch_from_repo_delta(
                    adapter.repo_root,
                    temp_repo,
                    policy,
                    candidate,
                    implementation_contract,
                    self.config,
                )
                if scope_action:
                    backend_result["recovery_actions"] = [*backend_result.get("recovery_actions", []), scope_action]
                if draft["status"] != "ok":
                    return self._write_failed_patch_artifacts(
                        base_rel,
                        idea_id,
                        candidate,
                        draft["status"],
                        "; ".join(draft.get("errors", [])),
                        rationale=backend_result.get("rationale", ""),
                        implementation_contract=implementation_contract,
                        prompt_text=prompt_text,
                        recovery_actions=backend_result.get("recovery_actions", []),
                    )
                mechanism_review = _mechanism_self_review(draft, implementation_contract, self.config)
                validation = _validate_patch_repo(temp_repo, draft["changed_files"], adapter.env_python, self.config, candidate=candidate)
                activation = _validate_config_activation(draft.get("diff", ""), candidate, self.config)
                validation["activation_check"] = activation
                risk_check = _validate_patch_proxy_risk(draft, candidate, self.config)
                validation["risk_check"] = risk_check
                validation["mechanism_review"] = mechanism_review
                validation["recovery_actions"] = backend_result.get("recovery_actions", [])
                if _validation_has_runtime_resource_retry(validation):
                    validation["retryable"] = True
                    validation["resource_retry"] = True
                    validation["failure_category"] = "runtime_smoke_resource_retry"
            patch_payload = {
                "schema_version": 1,
                "candidate_id": candidate.get("id"),
                "title": candidate.get("title"),
                "variant_fingerprint": candidate.get("variant_fingerprint") or ((candidate.get("s2_variant") or {}).get("variant_fingerprint") if isinstance(candidate.get("s2_variant"), dict) else None),
                "s2_variant": candidate.get("s2_variant") if isinstance(candidate.get("s2_variant"), dict) else {},
                "created_at": now_utc(),
                "backend": "codex_persistent_cli",
                "backend_sandbox": backend_result.get("sandbox"),
                "code_worktree": backend_result.get("code_worktree") or _compact_code_worktree(worktree_workspace),
                "codex_call": backend_result.get("codex_call") or {},
                "recovery_actions": backend_result.get("recovery_actions", []),
                "operations": draft["operations"],
                "changed_files": draft["changed_files"],
                "implementation_contract": implementation_contract,
                "repair_diagnosis": diagnosis_payload,
                "activation_check": activation,
                "risk_check": risk_check,
                "mechanism_review": mechanism_review,
                "rationale": backend_result.get("rationale", ""),
            }
            status = _patch_blocking_status(
                validation,
                activation,
                risk_check,
                mechanism_review,
                draft,
                self.config,
            )
            quality_score = _patch_quality_score(
                draft,
                validation,
                activation,
                risk_check,
                mechanism_review,
                implementation_contract,
            )
            quality_debt = quality_score.get("quality_debt", [])
            patch_payload["quality_score"] = quality_score
            patch_payload["quality_debt"] = quality_debt
            snapshot = _freeze_patched_repo_snapshot(
                self.project_root,
                temp_repo,
                base_rel,
                idea_id=idea_id,
                status=status,
                changed_files=draft["changed_files"],
            )
            patch_payload["patched_repo_snapshot"] = snapshot
            records = self._write_patch_artifacts(
                base_rel,
                patch_payload,
                draft.get("diff", ""),
                backend_result.get("rationale", ""),
                validation,
                implementation_contract=implementation_contract,
                prompt_text=prompt_text,
            )
            code_patch = {
                "status": status,
                "patch_json": records["patch_json"],
                "diff": records["diff"],
                "rationale": records["rationale"],
                "validation": records["validation"],
                "implementation_contract": records["implementation_contract"],
                "codex_prompt": records["codex_prompt"],
                "changed_files": draft["changed_files"],
                "has_executable_change": status == "ok" and bool(draft["operations"]),
                "patched_repo_snapshot": snapshot,
                "quality_score": quality_score,
                "quality_debt": quality_debt,
                "recovery_actions": backend_result.get("recovery_actions", []),
            }
            if _validation_has_runtime_resource_retry(validation):
                code_patch["retryable"] = True
                code_patch["resource_retry"] = True
                code_patch["failure_category"] = "runtime_smoke_resource_retry"
                code_patch["reason"] = "runtime smoke could not complete because available GPU memory was insufficient after OOM retry"
            if diagnosis_record:
                code_patch["repair_diagnosis"] = diagnosis_record["path"]
            if backend_result.get("code_worktree"):
                code_patch["code_worktree"] = backend_result.get("code_worktree")
            if backend_result.get("session_id"):
                code_patch["codex_session_id"] = backend_result.get("session_id")
            if status != "ok":
                if _validation_has_runtime_resource_retry(validation):
                    code_patch["reason"] = "runtime smoke could not complete because available GPU memory was insufficient after OOM retry"
                else:
                    code_patch["reason"] = (
                        risk_check.get("reason")
                        or activation.get("reason")
                        or mechanism_review.get("reason")
                        or validation.get("reason")
                        or status
                    )
                code_patch["risk_check"] = risk_check
                code_patch["mechanism_review"] = mechanism_review
            entry = dict(code_patch)
            entry.update(
                {
                    "candidate_id": candidate.get("id"),
                    "title": candidate.get("title"),
                    "variant_fingerprint": candidate.get("variant_fingerprint") or ((candidate.get("s2_variant") or {}).get("variant_fingerprint") if isinstance(candidate.get("s2_variant"), dict) else None),
                    "s2_variant": candidate.get("s2_variant") if isinstance(candidate.get("s2_variant"), dict) else {},
                }
            )
            return {"code_patch": code_patch, "manifest_entry": entry, "artifacts": list(records.values())}

    def _retry_backend_after_noop_sandbox_error(
        self,
        primary_result: dict[str, Any],
        implementation_contract: dict[str, Any],
        temp_repo: Path,
        policy: DynamicEditPolicy,
    ) -> dict[str, Any] | None:
        code_patch_config = _code_patch_config(self.config)
        primary_sandbox = str(primary_result.get("sandbox") or code_patch_config.get("codex_sandbox") or "workspace-write")
        fallback_sandbox = str(code_patch_config.get("codex_sandbox_fallback") or "")
        if not fallback_sandbox or fallback_sandbox == primary_sandbox:
            return None
        if isinstance(self.backend, CodexPersistentPatchBackend):
            fallback = self.backend.generate(implementation_contract, temp_repo, policy)
            action = {
                "action": "retry_codex_noop_with_persistent_session",
                "status": fallback.get("status"),
                "primary_sandbox": primary_sandbox,
                "fallback_sandbox": fallback_sandbox,
                "reason": _codex_sandbox_reason(primary_result),
            }
            fallback["recovery_actions"] = [*primary_result.get("recovery_actions", []), action, *fallback.get("recovery_actions", [])]
            fallback["primary_attempt"] = _compact_backend_attempt(primary_result)
            return fallback
        return None

    def _retry_backend_after_contract_failure(
        self,
        primary_result: dict[str, Any],
        implementation_contract: dict[str, Any],
        temp_repo: Path,
        policy: DynamicEditPolicy,
        failure: dict[str, Any],
        *,
        attempt: int,
    ) -> dict[str, Any] | None:
        repair_packet = _contract_repair_packet(failure, primary_result, attempt=attempt)
        repair_contract = _implementation_contract_with_contract_feedback(
            implementation_contract,
            failure,
            repair_packet=repair_packet,
            attempt=attempt,
        )
        repair = self.backend.generate(repair_contract, temp_repo, policy)
        action = {
            "action": "retry_codex_after_contract_failure",
            "status": repair.get("status"),
            "attempt": attempt,
            "failed_status": failure.get("status"),
            "failed_reason": _contract_failure_reason(failure),
            "repair_session": _repair_session_reuse_report(
                self.config,
                primary_result,
                repair,
                backend=self.backend,
            ),
            "repair_packet_summary": _repair_packet_summary(repair_packet),
        }
        repair["recovery_actions"] = [
            *primary_result.get("recovery_actions", []),
            action,
            *repair.get("recovery_actions", []),
        ]
        repair["primary_attempt"] = _compact_backend_attempt(primary_result)
        repair["implementation_contract"] = repair_contract
        return repair

    def _retry_backend_after_validation_failure(
        self,
        primary_result: dict[str, Any],
        implementation_contract: dict[str, Any],
        temp_repo: Path,
        policy: DynamicEditPolicy,
        validation: dict[str, Any],
        draft: dict[str, Any],
        *,
        attempt: int,
    ) -> dict[str, Any] | None:
        repair_packet = _validation_repair_packet(validation, draft, primary_result, attempt=attempt)
        repair_contract = _implementation_contract_with_validation_feedback(
            implementation_contract,
            validation,
            draft,
            repair_packet=repair_packet,
            attempt=attempt,
        )
        repair = self.backend.generate(repair_contract, temp_repo, policy)
        action = {
            "action": "retry_codex_after_validation_failure",
            "status": repair.get("status"),
            "attempt": attempt,
            "failed_checks": _failed_validation_checks(validation, limit=3),
            "repair_session": _repair_session_reuse_report(
                self.config,
                primary_result,
                repair,
                backend=self.backend,
            ),
            "repair_packet_summary": _repair_packet_summary(repair_packet),
        }
        repair["recovery_actions"] = [
            *primary_result.get("recovery_actions", []),
            action,
            *repair.get("recovery_actions", []),
        ]
        repair["primary_attempt"] = _compact_backend_attempt(primary_result)
        repair["implementation_contract"] = repair_contract
        return repair

    def _write_failed_patch_artifacts(
        self,
        base_rel: str,
        idea_id: str,
        candidate: dict[str, Any],
        status: str,
        reason: str,
        *,
        rationale: str = "",
        implementation_contract: dict[str, Any] | None = None,
        prompt_text: str = "",
        recovery_actions: list[dict[str, Any]] | None = None,
        risk_check: dict[str, Any] | None = None,
        changed_files: list[str] | None = None,
        diff_text: str = "",
    ) -> dict[str, Any]:
        normalized = _normalize_failed_patch_status(status, reason, recovery_actions or [])
        status = normalized["status"]
        reason = normalized["reason"]
        implementation_contract = implementation_contract or _build_implementation_contract(candidate, {}, DynamicEditPolicy.from_config())
        prompt_text = prompt_text or _codex_patch_prompt(implementation_contract, DynamicEditPolicy.from_config())
        validation = {
            "status": status,
            "reason": reason,
            "checks": [],
            "recovery_actions": recovery_actions or [],
            "retryable": normalized["retryable"],
            "failure_category": normalized["failure_category"],
        }
        if risk_check:
            validation["risk_check"] = risk_check
        patch_payload = {
            "schema_version": 1,
            "candidate_id": candidate.get("id"),
            "title": candidate.get("title"),
            "variant_fingerprint": candidate.get("variant_fingerprint") or ((candidate.get("s2_variant") or {}).get("variant_fingerprint") if isinstance(candidate.get("s2_variant"), dict) else None),
            "s2_variant": candidate.get("s2_variant") if isinstance(candidate.get("s2_variant"), dict) else {},
            "created_at": now_utc(),
            "operations": [],
            "changed_files": changed_files or [],
            "implementation_contract": implementation_contract,
            "recovery_actions": recovery_actions or [],
            "risk_check": risk_check or {},
            "rationale": rationale,
        }
        records = self._write_patch_artifacts(
            base_rel,
            patch_payload,
            diff_text,
            rationale,
            validation,
            implementation_contract=implementation_contract,
            prompt_text=prompt_text,
        )
        code_patch = {
            "status": status,
            "patch_json": records["patch_json"],
            "diff": records["diff"],
            "rationale": records["rationale"],
            "validation": records["validation"],
            "implementation_contract": records["implementation_contract"],
            "codex_prompt": records["codex_prompt"],
            "changed_files": changed_files or [],
            "has_executable_change": False,
            "reason": reason,
            "recovery_actions": recovery_actions or [],
            "retryable": normalized["retryable"],
            "failure_category": normalized["failure_category"],
        }
        if risk_check:
            code_patch["risk_check"] = risk_check
        entry = dict(code_patch)
        entry.update(
            {
                "candidate_id": candidate.get("id") or idea_id,
                "title": candidate.get("title"),
                "variant_fingerprint": candidate.get("variant_fingerprint") or ((candidate.get("s2_variant") or {}).get("variant_fingerprint") if isinstance(candidate.get("s2_variant"), dict) else None),
                "s2_variant": candidate.get("s2_variant") if isinstance(candidate.get("s2_variant"), dict) else {},
            }
        )
        return {"code_patch": code_patch, "manifest_entry": entry, "artifacts": list(records.values())}

    def _run_implementation_repair_diagnosis(
        self,
        base_rel: str,
        implementation_contract: dict[str, Any],
        worktree_workspace: dict[str, Any],
        env_python: str,
        policy: DynamicEditPolicy,
        *,
        recovery_actions: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if not isinstance(self.backend, CodexPersistentPatchBackend):
            return None, {}
        if not worktree_workspace.get("enabled") or worktree_workspace.get("status") != "ok":
            return None, {}
        repo = Path(str(worktree_workspace.get("repo") or ""))
        if not repo.exists():
            return None, {}
        code_patch_config = _code_patch_config(self.config)
        sandbox = str(
            (code_patch_config.get("implementation_repair_diagnosis") or {}).get("codex_sandbox")
            or code_patch_config.get("codex_preload_sandbox")
            or code_patch_config.get("codex_sandbox")
            or "workspace-write"
        )
        diagnosis_contract = _implementation_repair_diagnosis_contract(
                implementation_contract,
                self.project_root,
                env_python,
                self.config,
                repeated_failure_context=_repeated_failure_context_from_contract(implementation_contract),
            )
        started_session = _load_persistent_codex_session(Path(str(worktree_workspace.get("session_path") or "")))
        result = self.backend._run_persistent_codex_once(
            diagnosis_contract,
            repo,
            policy,
            worktree_workspace,
            sandbox=sandbox,
            prompt_kind="repair_diagnosis",
        )
        payload = _repair_diagnosis_payload_from_codex_result(
            result,
            diagnosis_contract,
            started_session=started_session,
        )
        payload["recovery_actions_before_diagnosis"] = list(recovery_actions)
        record = self.artifacts.write_json(
            self.stage_key,
            f"{base_rel}/repair_diagnosis.json",
            payload,
            artifact_type="c2c_s2_5_repair_diagnosis",
            summary="S2.5 implementation repair root-cause diagnosis",
            source_paths=[
                "plan/s2_5_repair_dispatch.json",
                "plan/code_patches/patch_manifest.json",
                f"{base_rel}/implementation_contract.json",
            ],
        )
        return record, payload

    def _write_patch_artifacts(
        self,
        base_rel: str,
        patch_payload: dict[str, Any],
        diff_text: str,
        rationale: str,
        validation: dict[str, Any],
        *,
        implementation_contract: dict[str, Any],
        prompt_text: str,
    ) -> dict[str, str]:
        contract_record = self.artifacts.write_json(
            self.stage_key,
            f"{base_rel}/implementation_contract.json",
            implementation_contract,
            artifact_type="c2c_patch_implementation_contract",
            summary="Compressed implementation contract passed to Codex",
        )
        prompt_record = self.artifacts.write_text(
            self.stage_key,
            f"{base_rel}/codex_prompt.md",
            prompt_text,
            artifact_type="c2c_patch_codex_prompt",
            summary="Compressed Codex patch prompt",
        )
        patch_record = self.artifacts.write_json(self.stage_key, f"{base_rel}/patch.json", patch_payload, artifact_type="c2c_frozen_patch", summary="Frozen candidate code patch")
        diff_record = self.artifacts.write_text(self.stage_key, f"{base_rel}/patch.diff", diff_text, artifact_type="c2c_patch_diff", summary="Candidate patch diff")
        rationale_record = self.artifacts.write_text(self.stage_key, f"{base_rel}/rationale.md", rationale or "No rationale returned.\n", artifact_type="c2c_patch_rationale", summary="Codex patch rationale")
        validation_record = self.artifacts.write_json(self.stage_key, f"{base_rel}/validation.json", validation, artifact_type="c2c_patch_validation", summary="Patch validation result")
        return {
            "implementation_contract": contract_record["path"],
            "codex_prompt": prompt_record["path"],
            "patch_json": patch_record["path"],
            "diff": diff_record["path"],
            "rationale": rationale_record["path"],
            "validation": validation_record["path"],
        }


def _code_patch_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_CODE_PATCH_CONFIG)
    user = config.get("code_patch") or {}
    merged = _deep_merge_dict(merged, user if isinstance(user, dict) else {})
    return _normalize_code_patch_config_for_project(merged, config)


def _normalize_code_patch_config_for_project(code_patch_config: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Keep S2.5 runtime validation portable across resumed C2C projects."""
    if not isinstance(config.get("c2c"), dict) or not config.get("c2c", {}).get("enabled", False):
        return code_patch_config
    validation = code_patch_config.setdefault("validation", {})
    if not isinstance(validation, dict):
        return code_patch_config
    smoke_cfg = validation.setdefault("runtime_smoke", {})
    if not isinstance(smoke_cfg, dict):
        return code_patch_config
    if not bool(smoke_cfg.get("respect_configured_gpu_ids", False)):
        original_gpu_ids = smoke_cfg.get("gpu_ids", "auto")
        if original_gpu_ids not in (None, "", "auto"):
            smoke_cfg["legacy_configured_gpu_ids"] = original_gpu_ids
        smoke_cfg["gpu_ids"] = "auto"
        smoke_cfg["gpu_selection_policy"] = "auto_free_memory"
    smoke_cfg.setdefault("min_free_mb", DEFAULT_CODE_PATCH_CONFIG["validation"]["runtime_smoke"]["min_free_mb"])
    smoke_cfg.setdefault("oom_retry", copy.deepcopy(DEFAULT_CODE_PATCH_CONFIG["validation"]["runtime_smoke"]["oom_retry"]))
    smoke_cfg.setdefault("resource_wait", copy.deepcopy(DEFAULT_CODE_PATCH_CONFIG["validation"]["runtime_smoke"]["resource_wait"]))
    return code_patch_config


def code_patch_gate_mode(config: dict[str, Any] | None) -> str:
    code_patch = _code_patch_config(config or {})
    validation = code_patch.get("validation") if isinstance(code_patch.get("validation"), dict) else {}
    mode = validation.get("gate_mode", code_patch.get("gate_mode", "discovery"))
    return "strict" if str(mode or "").strip().lower() == "strict" else "discovery"


def _strict_patch_gate(config: dict[str, Any] | None) -> bool:
    return code_patch_gate_mode(config) == "strict"


def _activation_wiring_hard_gate(config: dict[str, Any] | None) -> bool:
    validation = (_code_patch_config(config or {}).get("validation") or {})
    smoke_cfg = validation.get("runtime_smoke") if isinstance(validation.get("runtime_smoke"), dict) else {}
    wiring_cfg = smoke_cfg.get("mechanism_activation") if isinstance(smoke_cfg.get("mechanism_activation"), dict) else {}
    if "hard_gate" in wiring_cfg:
        return bool(wiring_cfg.get("hard_gate"))
    return _strict_patch_gate(config)


def _activation_check_blocks_patch(activation: dict[str, Any], config: dict[str, Any] | None) -> bool:
    if activation.get("status") == "ok":
        return False
    if activation.get("blocking") is not None:
        return bool(activation.get("blocking"))
    return _strict_patch_gate(config)


def _mechanism_review_blocks_patch(mechanism_review: dict[str, Any], config: dict[str, Any] | None) -> bool:
    if mechanism_review.get("status") == "ok":
        return False
    if mechanism_review.get("blocking") is not None:
        return bool(mechanism_review.get("blocking"))
    return _strict_patch_gate(config)


def _validation_check_blocks_patch(validation: dict[str, Any]) -> bool:
    return validation.get("status") != "ok"


def _validation_has_runtime_resource_retry(validation: dict[str, Any] | None) -> bool:
    if not isinstance(validation, dict):
        return False
    if validation.get("resource_retry") is True:
        return True
    if validation.get("failure_category") == "runtime_smoke_resource_retry":
        return True
    for check in validation.get("checks") or []:
        if not isinstance(check, dict):
            continue
        if check.get("resource_retry") is True or check.get("failure_category") == "runtime_smoke_resource_retry":
            return True
    return False


def _patch_blocking_status(
    validation: dict[str, Any],
    activation: dict[str, Any],
    risk_check: dict[str, Any],
    mechanism_review: dict[str, Any],
    draft: dict[str, Any],
    config: dict[str, Any] | None,
) -> str:
    if _validation_check_blocks_patch(validation):
        return "validation_failed"
    if _activation_check_blocks_patch(activation, config):
        return str(activation.get("status") or "config_activation_missing")
    if risk_check.get("status") != "ok":
        return str(risk_check.get("status") or "proxy_risk_repair_required")
    if _mechanism_review_blocks_patch(mechanism_review, config):
        return str(mechanism_review.get("status") or "mechanism_self_review_failed")
    if not draft.get("operations"):
        return "blocked_no_executable_change"
    return "ok"


def _patch_needs_repair(
    validation: dict[str, Any],
    activation: dict[str, Any],
    risk_check: dict[str, Any],
    mechanism_review: dict[str, Any],
    draft: dict[str, Any],
    config: dict[str, Any] | None,
) -> bool:
    if _validation_has_runtime_resource_retry(validation):
        return False
    return _patch_blocking_status(validation, activation, risk_check, mechanism_review, draft, config) != "ok"


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge_dict(copy.deepcopy(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _default_code_patch_backend(config: dict[str, Any], project_root: Path) -> CodexPersistentPatchBackend:
    code_patch_config = _code_patch_config(config)
    backend_name = str(code_patch_config.get("backend") or "codex_persistent_cli")
    if backend_name == "codex_persistent_cli":
        return CodexPersistentPatchBackend(config, project_root)
    raise ValueError("S2.5 code patch generation only supports backend=codex_persistent_cli")


def _prepare_code_worktree_workspace(
    project_root: Path,
    config: dict[str, Any],
    *,
    idea_id: str,
    variant_index: int,
    enabled: bool,
) -> dict[str, Any]:
    code_patch_config = _code_patch_config(config)
    if not enabled:
        return {"enabled": False, "status": "ok"}
    target_repo_value = ((config.get("c2c") or {}).get("target_repo") or "")
    target_repo = Path(str(target_repo_value)).expanduser() if target_repo_value else None
    if target_repo is None or not target_repo.exists():
        return {
            "enabled": True,
            "status": "codex_failed",
            "reason": "Git worktree requires c2c.target_repo to exist",
            "recovery_actions": [],
        }
    git_check = subprocess.run(["git", "-C", str(target_repo), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, timeout=30)
    if git_check.returncode != 0 or git_check.stdout.strip() != "true":
        return {
            "enabled": True,
            "status": "codex_failed",
            "reason": "Git worktree requires c2c.target_repo to be a git repo",
            "stderr": git_check.stderr[-1000:],
            "recovery_actions": [],
        }
    baseline_guard = _validate_worktree_baseline_source(project_root, config, target_repo)
    if baseline_guard.get("status") != "ok":
        return baseline_guard
    base_ref = str(code_patch_config.get("worktree_base_ref") or baseline_guard.get("source_git_commit") or "HEAD")
    commit = subprocess.run(["git", "-C", str(target_repo), "rev-parse", base_ref], capture_output=True, text=True, timeout=30)
    if commit.returncode != 0:
        return {
            "enabled": True,
            "status": "retryable_codex_failed",
            "reason": f"failed to resolve worktree base ref {base_ref}: {commit.stderr[-1000:]}",
            "retryable": True,
            "failure_category": "git_worktree_setup_failed",
            "recovery_actions": [],
        }
    worktree_paths = _code_worktree_paths(project_root, code_patch_config, idea_id, variant_index)
    session_root = worktree_paths["session_root"]
    worktree_root = worktree_paths["worktree_root"]
    repo = worktree_paths["repo"]
    session_path = session_root / "codex_session.json"
    events_path = session_root / "codex_events.jsonl"
    metadata_path = session_root / "worktree_metadata.json"
    branch = _persistent_worktree_branch(project_root.name, idea_id, variant_index)
    session_key = f"s2_5:{idea_id}:v{variant_index}"
    recovery_actions: list[dict[str, Any]] = []
    ensure_dir(session_root)
    ensure_dir(worktree_root)
    created_worktree = False
    if not repo.exists():
        result = subprocess.run(
            ["git", "-C", str(target_repo), "worktree", "add", "-B", branch, str(repo), base_ref],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return {
                "enabled": True,
                "status": "retryable_codex_failed",
                "reason": f"git worktree add failed: {result.stderr[-2000:] or result.stdout[-2000:]}",
                "retryable": True,
                "failure_category": "git_worktree_setup_failed",
                "recovery_actions": [],
            }
        recovery_actions.append({"action": "git_worktree_add", "status": "ok", "branch": branch, "repo": str(repo)})
        created_worktree = True
        if session_path.exists():
            session_path.unlink()
            recovery_actions.append(
                {
                    "action": "discard_stale_codex_session_for_recreated_worktree",
                    "status": "ok",
                    "session_path": str(session_path),
                }
            )
    baseline_sync = _prepare_worktree_snapshot_baseline(
        repo,
        baseline_guard,
        metadata_path,
        session_path,
        code_patch_config,
        created_worktree=created_worktree,
    )
    if baseline_sync.get("status") != "ok":
        baseline_sync["recovery_actions"] = [
            *recovery_actions,
            *list(baseline_sync.get("recovery_actions") or []),
        ]
        return baseline_sync
    if baseline_sync.get("action"):
        recovery_actions.append(baseline_sync)
    metadata = {
        "enabled": True,
        "status": "ok",
        "project_id": project_root.name,
        "idea_id": idea_id,
        "variant_index": variant_index,
        "session_key": session_key,
        "target_repo": str(target_repo.resolve()),
        "repo": str(repo.resolve()),
        "session_root": str(session_root.resolve()),
        "worktree_root": str(worktree_root.resolve()),
        "worktree_storage": worktree_paths["storage"],
        "branch": branch,
        "base_ref": base_ref,
        "base_commit": commit.stdout.strip(),
        "baseline_guard": baseline_guard,
        "baseline_sync": baseline_sync,
        "baseline_materialized_from_snapshot": bool(
            baseline_sync.get("materialized")
            or baseline_sync.get("trusted_existing_materialization")
            or baseline_sync.get("core_files_match_snapshot")
        ),
        "session_path": str(session_path.resolve()),
        "events_path": str(events_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "created_or_updated_at": now_utc(),
        "session_policy": "persistent_resume_required",
        "recovery_actions": recovery_actions,
    }
    write_json(metadata_path, metadata)
    return metadata


def _code_worktree_paths(
    project_root: Path,
    code_patch_config: dict[str, Any],
    idea_id: str,
    variant_index: int,
) -> dict[str, Any]:
    session_root = project_root / "plan" / "code_worktrees" / idea_id / f"v{variant_index}"
    metadata_repo = _code_worktree_repo_from_metadata(session_root / "worktree_metadata.json")
    if metadata_repo is not None:
        return {
            "session_root": session_root,
            "worktree_root": metadata_repo.parent,
            "repo": metadata_repo,
            "storage": {"mode": "metadata_repo", "source": "worktree_metadata.json"},
        }

    legacy_repo = session_root / "repo"
    if legacy_repo.is_dir():
        return {
            "session_root": session_root,
            "worktree_root": session_root,
            "repo": legacy_repo,
            "storage": {"mode": "legacy_project_workspace"},
        }

    storage_root = _code_worktree_storage_root(code_patch_config)
    project_key = _code_worktree_project_key(project_root)
    worktree_root = storage_root / project_key / idea_id / f"v{variant_index}"
    return {
        "session_root": session_root,
        "worktree_root": worktree_root,
        "repo": worktree_root / "repo",
        "storage": {
            "mode": "external_cache",
            "storage_root": str(storage_root),
            "project_key": project_key,
        },
    }


def _code_worktree_repo_from_metadata(metadata_path: Path) -> Path | None:
    if not metadata_path.exists():
        return None
    try:
        metadata = read_json(metadata_path, default={}) or {}
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(metadata, dict):
        return None
    repo_value = metadata.get("repo")
    if not repo_value:
        return None
    repo = Path(str(repo_value)).expanduser()
    if repo.is_dir():
        return repo.resolve()
    return None


def _code_worktree_storage_root(code_patch_config: dict[str, Any]) -> Path:
    configured = os.environ.get("AUTO_RESEARCH_WORKTREE_ROOT")
    if not configured:
        configured = code_patch_config.get("worktree_storage_root")
    if configured:
        root = Path(str(configured)).expanduser()
        if not root.is_absolute():
            root = repo_root() / root
        return root.resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return (base / "auto-research" / "code-worktrees").resolve()


def _code_worktree_project_key(project_root: Path) -> str:
    digest = hashlib.sha256(str(project_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{sanitize_filename(project_root.name, max_length=60)}-{digest}"


def _persistent_worktree_branch(project_id: str, idea_id: str, variant_index: int) -> str:
    return "/".join(
        [
            "auto-research",
            sanitize_filename(project_id, max_length=40),
            sanitize_filename(idea_id, max_length=60),
            f"v{variant_index}",
        ]
    )


def _validate_worktree_baseline_source(project_root: Path, config: dict[str, Any], target_repo: Path) -> dict[str, Any]:
    adapter = C2CAdapter(project_root, config)
    snapshot_root = adapter.repo_root
    if not snapshot_root.exists():
        return {
            "enabled": True,
            "status": "codex_failed",
            "reason": f"Git worktree baseline guard requires snapshot_path to exist: {snapshot_root}",
            "recovery_actions": [],
        }
    manifest = read_json(project_root / "experiment" / "c2c" / "repo_snapshot_manifest.json", default={}) or {}
    source_git_commit = str(manifest.get("source_git_commit") or "")
    mismatches = _baseline_core_file_mismatches(snapshot_root, target_repo)
    if mismatches:
        return {
            "enabled": True,
            "status": "codex_failed",
            "reason": "Git worktree target_repo does not match the baseline snapshot core files",
            "mismatched_files": mismatches[:8],
            "recovery_actions": [],
        }
    current_commit = subprocess.run(["git", "-C", str(target_repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    current_git_commit = current_commit.stdout.strip() if current_commit.returncode == 0 else ""
    return {
        "enabled": True,
        "status": "ok",
        "snapshot_root": str(snapshot_root.resolve()),
        "target_repo": str(target_repo.resolve()),
        "source_git_commit": source_git_commit,
        "current_git_commit": current_git_commit,
        "commit_matches_snapshot": bool(source_git_commit and current_git_commit == source_git_commit),
        "core_file_hash_check": "ok",
    }


def _prepare_worktree_snapshot_baseline(
    repo: Path,
    baseline_guard: dict[str, Any],
    metadata_path: Path,
    session_path: Path,
    code_patch_config: dict[str, Any],
    *,
    created_worktree: bool,
) -> dict[str, Any]:
    if not code_patch_config.get("materialize_snapshot_baseline", True):
        return {
            "status": "ok",
            "action": "snapshot_baseline_materialization_disabled",
            "materialized": False,
        }
    snapshot_root_value = str(baseline_guard.get("snapshot_root") or "")
    snapshot_root = Path(snapshot_root_value) if snapshot_root_value else None
    if snapshot_root is None or not snapshot_root.exists():
        return {
            "enabled": True,
            "status": "codex_failed",
            "reason": "Git worktree baseline materialization requires an existing snapshot_root",
            "recovery_actions": [],
        }
    previous_metadata = read_json(metadata_path, default={}) or {}
    if (
        not created_worktree
        and previous_metadata.get("baseline_materialized_from_snapshot")
        and str(previous_metadata.get("baseline_guard", {}).get("snapshot_root") or "") == str(snapshot_root.resolve())
    ):
        return {
            "status": "ok",
            "action": "reuse_snapshot_materialized_worktree",
            "materialized": False,
            "trusted_existing_materialization": True,
        }
    mismatches = _baseline_core_file_mismatches(snapshot_root, repo)
    if not mismatches and not created_worktree:
        return {
            "status": "ok",
            "action": "existing_worktree_core_matches_snapshot",
            "materialized": False,
            "core_files_match_snapshot": True,
        }
    if not created_worktree and session_path.exists():
        return {
            "enabled": True,
            "status": "codex_failed",
            "reason": (
                "Existing persistent worktree differs from the baseline snapshot and predates snapshot baseline "
                "materialization; remove this idea worktree to recreate it from the correct baseline snapshot."
            ),
            "mismatched_files": mismatches[:8],
            "recovery_actions": [],
        }
    materialized = _materialize_snapshot_into_worktree(snapshot_root, repo)
    post_mismatches = _baseline_core_file_mismatches(snapshot_root, repo)
    if post_mismatches:
        return {
            "enabled": True,
            "status": "codex_failed",
            "reason": "Failed to materialize baseline snapshot into Git worktree",
            "mismatched_files": post_mismatches[:8],
            "recovery_actions": [],
        }
    return {
        "status": "ok",
        "action": "materialize_snapshot_baseline",
        "materialized": True,
        "snapshot_root": str(snapshot_root.resolve()),
        "copied_files": materialized["copied_files"],
        "removed_extra_files": materialized["removed_extra_files"][:20],
    }


def _materialize_snapshot_into_worktree(snapshot_root: Path, repo: Path) -> dict[str, Any]:
    snapshot_hashes = _all_file_hashes(snapshot_root)
    repo_hashes = _all_file_hashes(repo)
    removed: list[str] = []
    for rel_path in sorted(set(repo_hashes) - set(snapshot_hashes)):
        target = repo / rel_path
        if target.exists() and target.is_file():
            target.unlink()
            removed.append(rel_path)
            _remove_empty_parents(target.parent, stop_at=repo)
    shutil.copytree(snapshot_root, repo, ignore=_worktree_snapshot_materialize_ignore, dirs_exist_ok=True)
    copied = sum(1 for path in snapshot_root.rglob("*") if path.is_file() and not _path_ignored_for_patch_delta(path.relative_to(snapshot_root).as_posix()))
    return {"copied_files": copied, "removed_extra_files": removed}


def _freeze_patched_repo_snapshot(
    project_root: Path,
    patched_repo: Path,
    base_rel: str,
    *,
    idea_id: str,
    status: str,
    changed_files: list[str],
) -> dict[str, Any]:
    if status != "ok":
        return {"status": "skipped", "reason": f"patch status is {status}"}
    snapshot_rel = f"plan/{base_rel}/patched_repo_snapshot"
    manifest_rel = f"plan/{base_rel}/patched_repo_snapshot_manifest.json"
    snapshot_root = project_root / snapshot_rel
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    ensure_dir(snapshot_root.parent)
    shutil.copytree(patched_repo, snapshot_root, ignore=_patched_repo_snapshot_ignore)
    manifest = repo_snapshot_manifest(snapshot_root)
    manifest.update(
        {
            "schema_version": "patched_repo_snapshot_v1",
            "candidate_id": idea_id,
            "created_at": now_utc(),
            "snapshot_path": snapshot_rel,
            "changed_files": list(changed_files or []),
            "source": "s2_5_codex_validated_worktree",
            "execution_truth": True,
            "filtered_runtime_artifacts": True,
        }
    )
    write_json(project_root / manifest_rel, manifest)
    return {
        "status": "ok",
        "path": snapshot_rel,
        "manifest": manifest_rel,
        "file_count": manifest.get("file_count"),
        "sha256": manifest.get("sha256"),
        "changed_files": list(changed_files or []),
    }


def _patched_repo_snapshot_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set(_temporary_patch_ignore(directory, names))
    for name in names:
        rel = Path(directory) / name
        rel_name = name
        try:
            rel_name = rel.name
        except OSError:
            pass
        if rel_name in ignored:
            continue
        if _path_ignored_for_patch_delta(rel_name):
            ignored.add(name)
    if Path(directory).name == "local":
        ignored.update({"auto_research_runs", "checkpoints", "snapshots", "final_results"})
    return ignored.intersection(names)


def _remove_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path
    stop = stop_at.resolve()
    while current.exists():
        try:
            if current.resolve() == stop:
                return
        except OSError:
            return
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _worktree_snapshot_materialize_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {".git", "wandb", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov"}
    if Path(directory).name == "local":
        ignored.update({"checkpoints", "snapshots"})
    for name in names:
        if name in ignored or Path(name).suffix in {".pt", ".pth", ".safetensors", ".bin", ".ckpt"}:
            ignored.add(name)
    return ignored.intersection(names)


def _baseline_core_file_mismatches(snapshot_root: Path, target_repo: Path) -> list[dict[str, str]]:
    mismatches: list[dict[str, str]] = []
    for rel_path in sorted(set(CORE_PATCH_BASELINE_FILES)):
        snapshot_path = snapshot_root / rel_path
        target_path = target_repo / rel_path
        if not snapshot_path.exists() and not target_path.exists():
            continue
        if not snapshot_path.exists() or not target_path.exists():
            mismatches.append({"path": rel_path, "reason": "exists_mismatch"})
            continue
        if not snapshot_path.is_file() or not target_path.is_file():
            mismatches.append({"path": rel_path, "reason": "not_file"})
            continue
        if sha256_file(snapshot_path) != sha256_file(target_path):
            mismatches.append({"path": rel_path, "reason": "sha256_mismatch"})
    return mismatches


def _implementation_contract_with_worktree(implementation_contract: dict[str, Any], workspace: dict[str, Any]) -> dict[str, Any]:
    contract = json.loads(json.dumps(implementation_contract, ensure_ascii=False))
    contract["code_worktree"] = _compact_code_worktree(workspace)
    return contract


def _implementation_contract_with_repair_diagnosis(
    implementation_contract: dict[str, Any],
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    contract = json.loads(json.dumps(implementation_contract, ensure_ascii=False))
    contract["repair_diagnosis"] = _compact_value(diagnosis, max_chars=6000)
    requirements = list(contract.get("s2_5_requirements") or [])
    requirements.append(
        "Before editing, use repair_diagnosis.root_cause/evidence/repair_target as the source of truth for this implementation repair; "
        "fix the diagnosed config/constructor/forward/tensor path without changing the candidate or S2 method."
    )
    contract["s2_5_requirements"] = requirements
    return contract


def _repeated_implementation_failure_context(
    project_root: Path,
    implementation_contract: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    repeat_cfg = (((_code_patch_config(config).get("implementation_repair_diagnosis") or {}).get("repeated_failure_detection")) or {})
    if repeat_cfg.get("enabled") is False:
        return {"is_repeated": False, "reason": "disabled"}
    previous_failure = implementation_contract.get("previous_failure") if isinstance(implementation_contract.get("previous_failure"), dict) else {}
    dispatch = previous_failure.get("s2_5_repair_dispatch") if isinstance(previous_failure.get("s2_5_repair_dispatch"), dict) else {}
    current = _implementation_failure_fingerprint(
        diagnostics=dispatch.get("activation_forward_probe_diagnostics") if isinstance(dispatch.get("activation_forward_probe_diagnostics"), dict) else {},
        tensor_checks=dispatch.get("tensor_checks"),
        changed_files=dispatch.get("changed_files") or [],
    )
    manifest_entry = _previous_manifest_entry_for_candidate(project_root, str(implementation_contract.get("candidate_id") or ""))
    previous = _previous_manifest_failure_fingerprint(project_root, manifest_entry) if manifest_entry else {}
    comparison = _compare_implementation_failure_fingerprints(
        current,
        previous,
        changed_files_jaccard_threshold=float(repeat_cfg.get("changed_files_jaccard_threshold", 0.8) or 0.8),
    )
    is_repeated = bool(comparison.get("repeated_signals"))
    return {
        "schema_version": "s2_5_repeated_implementation_failure_context_v1",
        "is_repeated": is_repeated,
        "min_repeats": int(repeat_cfg.get("min_repeats", 2) or 2),
        "current_fingerprint": current,
        "previous_fingerprint": previous,
        "previous_manifest_entry": {
            "candidate_id": manifest_entry.get("candidate_id") if isinstance(manifest_entry, dict) else None,
            "status": manifest_entry.get("status") if isinstance(manifest_entry, dict) else None,
            "validation": manifest_entry.get("validation") if isinstance(manifest_entry, dict) else None,
        },
        "repeated_signals": comparison.get("repeated_signals") or [],
        "changed_files_similarity": comparison.get("changed_files_similarity"),
        "instruction": (
            "If is_repeated=true, stop ordinary same-path patch repair. Diagnose why the same tensor/cache/switch/changed-file fingerprint persisted, "
            "then change the implementation repair target or wiring boundary while preserving the same candidate and variant."
        ),
    }


def _previous_manifest_entry_for_candidate(project_root: Path, candidate_id: str) -> dict[str, Any] | None:
    manifest = read_json(project_root / "plan" / "code_patches" / "patch_manifest.json", default={}) or {}
    entries = [entry for entry in manifest.get("candidates") or manifest.get("patches") or [] if isinstance(entry, dict)]
    if candidate_id:
        matched = next((entry for entry in entries if str(entry.get("candidate_id") or "") == candidate_id), None)
        if matched:
            return matched
    return entries[0] if entries else None


def _previous_manifest_failure_fingerprint(project_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    validation = {}
    validation_path = entry.get("validation") if isinstance(entry, dict) else None
    if validation_path:
        validation = read_json(project_root / str(validation_path), default={}) or {}
    diagnostics = _forward_probe_diagnostics_from_validation(validation) if isinstance(validation, dict) else {}
    return _implementation_failure_fingerprint(
        diagnostics=diagnostics,
        tensor_checks=diagnostics,
        changed_files=(entry.get("changed_files") or []) if isinstance(entry, dict) else [],
    )


def _implementation_failure_fingerprint(
    *,
    diagnostics: dict[str, Any],
    tensor_checks: Any,
    changed_files: list[Any],
) -> dict[str, Any]:
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    tensor_payload = tensor_checks if tensor_checks not in (None, [], {}) else diagnostics
    identical_tensors = _normalize_identical_tensor_names(
        _nested_tensor_value(tensor_payload, "identical_tensors")
        or _nested_tensor_value(diagnostics, "identical_tensors")
    )
    return {
        "identical_tensors": identical_tensors,
        "changed_tensors": _normalize_identical_tensor_names(
            _nested_tensor_value(tensor_payload, "changed_tensors")
            or _nested_tensor_value(diagnostics, "changed_tensors")
        ),
        "cache_key_diff": _normalized_float(diagnostics.get("cache_key_diff")),
        "cache_value_diff": _normalized_float(diagnostics.get("cache_value_diff")),
        "switch_seen_by_forward": diagnostics.get("switch_seen_by_forward"),
        "projector_output_identical": diagnostics.get("projector_output_identical"),
        "wrapper_cache_identical": diagnostics.get("wrapper_cache_identical"),
        "repair_focus": [str(item) for item in diagnostics.get("repair_focus") or [] if item],
        "changed_files": sorted({str(path) for path in changed_files if path}),
    }


def _nested_tensor_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload.get(key)
        tensor_checks = payload.get("tensor_checks")
        if isinstance(tensor_checks, dict) and key in tensor_checks:
            return tensor_checks.get(key)
    return None


def _normalize_identical_tensor_names(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value] if value else []
    names: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("tensor") or item.get("field")
        else:
            name = item
        if name:
            names.append(str(name))
    return sorted(set(names))


def _normalized_float(value: Any) -> float | None:
    try:
        return round(float(value), 8)
    except (TypeError, ValueError):
        return None


def _compare_implementation_failure_fingerprints(
    current: dict[str, Any],
    previous: dict[str, Any],
    *,
    changed_files_jaccard_threshold: float,
) -> dict[str, Any]:
    if not current or not previous:
        return {"repeated_signals": [], "changed_files_similarity": 0.0}
    repeated: list[str] = []
    current_identical = set(current.get("identical_tensors") or [])
    previous_identical = set(previous.get("identical_tensors") or [])
    same_identical = sorted(current_identical.intersection(previous_identical))
    if same_identical:
        repeated.append("same_identical_tensors:" + ",".join(same_identical[:5]))
    for key in ["cache_key_diff", "cache_value_diff"]:
        if current.get(key) is not None and current.get(key) == previous.get(key):
            repeated.append(f"same_{key}:{current.get(key)}")
    if current.get("switch_seen_by_forward") is False and previous.get("switch_seen_by_forward") is False:
        repeated.append("same_switch_seen_by_forward_false")
    if current.get("projector_output_identical") is True and previous.get("projector_output_identical") is True:
        repeated.append("same_projector_output_identical")
    if current.get("wrapper_cache_identical") is True and previous.get("wrapper_cache_identical") is True:
        repeated.append("same_wrapper_cache_identical")
    similarity = _jaccard_similarity(set(current.get("changed_files") or []), set(previous.get("changed_files") or []))
    if similarity >= changed_files_jaccard_threshold and current.get("changed_files") and previous.get("changed_files"):
        repeated.append(f"changed_files_still_same:{similarity:.2f}")
    return {"repeated_signals": repeated, "changed_files_similarity": similarity}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left.union(right)
    if not union:
        return 0.0
    return len(left.intersection(right)) / len(union)


def _implementation_contract_with_repeated_failure_context(
    implementation_contract: dict[str, Any],
    repeated_failure_context: dict[str, Any],
) -> dict[str, Any]:
    contract = json.loads(json.dumps(implementation_contract, ensure_ascii=False))
    context = _compact_value(repeated_failure_context, max_chars=5000)
    contract["repeated_failure_context"] = context
    requirements = list(contract.get("s2_5_requirements") or [])
    requirements.append(
        "Repeated implementation failure detected: do not keep making small edits along the same failing path. "
        "Use repeated_failure_context to identify the unchanged tensor/switch/cache/changed-file fingerprint, then switch integration point or repair a lower-level wiring boundary while preserving the same candidate."
    )
    contract["s2_5_requirements"] = requirements
    return contract


def _implementation_repair_diagnosis_enabled(config: dict[str, Any], implementation_contract: dict[str, Any]) -> bool:
    cfg = (_code_patch_config(config).get("implementation_repair_diagnosis") or {})
    if cfg.get("enabled") is False:
        return False
    previous_failure = implementation_contract.get("previous_failure") if isinstance(implementation_contract.get("previous_failure"), dict) else {}
    repair_contract = (
        previous_failure.get("proxy_effect_repair_contract")
        if isinstance(previous_failure.get("proxy_effect_repair_contract"), dict)
        else implementation_contract.get("proxy_effect_repair_contract")
        if isinstance(implementation_contract.get("proxy_effect_repair_contract"), dict)
        else {}
    )
    dispatch = previous_failure.get("s2_5_repair_dispatch") if isinstance(previous_failure.get("s2_5_repair_dispatch"), dict) else {}
    if str(repair_contract.get("mode") or "") == "s2_5_only_implementation_repair":
        return True
    if str(dispatch.get("mode") or dispatch.get("repair_lane") or "") == "s2_5_only_implementation_repair":
        return True
    if previous_failure.get("failure_class") == "implementation_failure":
        return True
    return False


def _implementation_repair_diagnosis_contract(
    implementation_contract: dict[str, Any],
    project_root: Path,
    env_python: str,
    config: dict[str, Any],
    *,
    repeated_failure_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = json.loads(json.dumps(implementation_contract, ensure_ascii=False))
    previous_failure = contract.get("previous_failure") if isinstance(contract.get("previous_failure"), dict) else {}
    repair_dispatch = previous_failure.get("s2_5_repair_dispatch") if isinstance(previous_failure.get("s2_5_repair_dispatch"), dict) else {}
    patch_manifest = read_json(project_root / "plan" / "code_patches" / "patch_manifest.json", default={}) or {}
    env_path = str(env_python or ((config.get("c2c") or {}).get("env_python") or sys.executable))
    diagnosis_cfg = (_code_patch_config(config).get("implementation_repair_diagnosis") or {})
    contract["implementation_repair_diagnosis"] = {
        "schema_version": "s2_5_implementation_repair_diagnosis_request_v1",
        "mode": "root_cause_pre_pass",
        "do_not_edit_files": True,
        "same_candidate_required": True,
        "same_variant_fingerprint_required": bool(contract.get("variant_fingerprint")),
        "environment": {
            "python_cmd": env_path,
            "must_use_env_python": True,
            "using_c2c_env_python": _using_target_env_python(env_path),
        },
        "artifacts_to_read": [
            "plan/s2_5_repair_dispatch.json",
            "plan/code_patches/patch_manifest.json",
            "plan/code_patches/<candidate>/implementation_contract.json",
        ],
        "changed_files": repair_dispatch.get("changed_files") or [],
        "activation_forward_probe_diagnostics": repair_dispatch.get("activation_forward_probe_diagnostics") or {},
        "tensor_checks": repair_dispatch.get("tensor_checks") or {},
        "repeated_failure_context": repeated_failure_context or {},
        "patch_manifest": _compact_value(patch_manifest, max_chars=5000),
        "max_file_reads": diagnosis_cfg.get("max_file_reads", 20),
        "allowed_lightweight_commands": [
            f"{env_path} -m py_compile <changed python files>",
            f"{env_path} <targeted smoke/forward probe only when available>",
        ],
        "forbidden_commands": [
            "full train",
            "large proxy train/eval",
            "torch.distributed.run",
            "multi-GPU jobs",
            "editing evaluator/probe code to bypass checks",
        ],
        "required_output_schema": {
            "root_cause": "string",
            "evidence": ["string"],
            "repair_target": ["string"],
            "forbidden": ["string"],
            "lightweight_commands_run": ["string"],
            "env_python_used": "string",
            "confidence": "low|medium|high",
        },
    }
    requirements = list(contract.get("s2_5_requirements") or [])
    requirements.append(
        "Diagnosis pre-pass only: inspect files and lightweight command output, then report root_cause/evidence/repair_target; do not edit files in this turn."
    )
    if repeated_failure_context and repeated_failure_context.get("is_repeated"):
        requirements.append(
            "Repeated failure pre-pass: explicitly explain why the same tensor/cache/switch/changed-files fingerprint persisted, and choose a different implementation repair target than the repeated changed-file-only path."
        )
    contract["s2_5_requirements"] = requirements
    return contract


def _repeated_failure_context_from_contract(implementation_contract: dict[str, Any]) -> dict[str, Any]:
    context = implementation_contract.get("repeated_failure_context")
    return context if isinstance(context, dict) else {}


def _repair_diagnosis_payload_from_codex_result(
    result: dict[str, Any],
    diagnosis_contract: dict[str, Any],
    *,
    started_session: str | None,
) -> dict[str, Any]:
    text = str(result.get("rationale") or "").strip()
    parsed = _parse_repair_diagnosis_text(text)
    call = result.get("codex_call") if isinstance(result.get("codex_call"), dict) else {}
    environment = (diagnosis_contract.get("implementation_repair_diagnosis") or {}).get("environment") or {}
    session_after = result.get("session_id") or call.get("session_id")
    previous_session = result.get("previous_session_id") or call.get("previous_session_id")
    payload = {
        "schema_version": "s2_5_implementation_repair_diagnosis_v1",
        "created_at": now_utc(),
        "status": result.get("status"),
        "implementation_repair_diagnosis": _compact_value(
            diagnosis_contract.get("implementation_repair_diagnosis") or {},
            max_chars=6000,
        ),
        "root_cause": parsed.get("root_cause") or "",
        "evidence": parsed.get("evidence") or [],
        "repair_target": parsed.get("repair_target") or [],
        "forbidden": parsed.get("forbidden") or [],
        "lightweight_commands_run": parsed.get("lightweight_commands_run") or [],
        "env_python_used": parsed.get("env_python_used") or environment.get("python_cmd"),
        "confidence": parsed.get("confidence") or "unknown",
        "raw_response": text[-6000:],
        "environment": environment,
        "same_session_reused": bool(
            started_session
            and (
                session_after == started_session
                or previous_session == started_session
                or result.get("same_session_reused") is True
            )
        ),
        "session_id_before": started_session,
        "session_id_after": session_after,
        "previous_session_id": previous_session,
        "codex_call": call,
        "code_worktree": result.get("code_worktree") or diagnosis_contract.get("code_worktree") or {},
        "primary_attempt": _compact_backend_attempt(result),
    }
    if result.get("status") != "ok":
        payload["reason"] = result.get("reason") or result.get("stderr") or "diagnosis Codex call failed"
    return payload


def _parse_repair_diagnosis_text(text: str) -> dict[str, Any]:
    if not text:
        return {}
    parsed = _extract_json_object(text)
    if isinstance(parsed, dict):
        return {
            "root_cause": str(parsed.get("root_cause") or ""),
            "evidence": _coerce_string_list(parsed.get("evidence")),
            "repair_target": _coerce_string_list(parsed.get("repair_target")),
            "forbidden": _coerce_string_list(parsed.get("forbidden")),
            "lightweight_commands_run": _coerce_string_list(parsed.get("lightweight_commands_run")),
            "env_python_used": str(parsed.get("env_python_used") or ""),
            "confidence": str(parsed.get("confidence") or ""),
        }
    fields: dict[str, Any] = {}
    for key in ["root_cause", "env_python_used", "confidence"]:
        match = re.search(rf"(?im)^\s*[-*# ]*{re.escape(key)}\s*[:：]\s*(.+)$", text)
        if match:
            fields[key] = match.group(1).strip()
    for key in ["evidence", "repair_target", "forbidden", "lightweight_commands_run"]:
        fields[key] = _extract_markdown_list_field(text, key)
    return fields


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced + candidates
    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()][:20]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_markdown_list_field(text: str, key: str) -> list[str]:
    pattern = re.compile(rf"(?ims)^\s*[-*# ]*{re.escape(key)}\s*[:：]\s*(.*?)(?=^\s*[-*# ]*[a-zA-Z_]+\s*[:：]|\Z)")
    match = pattern.search(text)
    if not match:
        return []
    block = match.group(1).strip()
    items = []
    for line in block.splitlines():
        cleaned = re.sub(r"^\s*[-*]\s*", "", line).strip()
        if cleaned:
            items.append(cleaned)
    if not items and block:
        items.append(block)
    return items[:20]


def _worktree_workspace_from_contract(implementation_contract: dict[str, Any]) -> dict[str, Any] | None:
    workspace = implementation_contract.get("code_worktree")
    return workspace if isinstance(workspace, dict) and workspace.get("repo") else None


@dataclass
class _PatchWorkspaceContext:
    source_repo: Path
    workspace: dict[str, Any]
    idea_id: str
    _tmp: tempfile.TemporaryDirectory[str] | None = None
    _repo: Path | None = None

    def __enter__(self) -> Path:
        if self.workspace.get("enabled"):
            self._repo = Path(str(self.workspace["repo"]))
            return self._repo
        self._tmp = tempfile.TemporaryDirectory(prefix=f"auto_research_patch_{self.idea_id}_")
        self._repo = Path(self._tmp.name) / "repo"
        shutil.copytree(self.source_repo, self._repo, ignore=_temporary_patch_ignore)
        return self._repo

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


def _patch_workspace_context(source_repo: Path, workspace: dict[str, Any], *, idea_id: str) -> _PatchWorkspaceContext:
    return _PatchWorkspaceContext(source_repo=source_repo, workspace=workspace, idea_id=idea_id)


def _compact_code_worktree(workspace: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(workspace, dict) or not workspace.get("enabled"):
        return {}
    return {
        "session_key": workspace.get("session_key"),
        "repo": workspace.get("repo"),
        "branch": workspace.get("branch"),
        "base_ref": workspace.get("base_ref"),
        "base_commit": workspace.get("base_commit"),
        "session_path": workspace.get("session_path"),
        "events_path": workspace.get("events_path"),
        "metadata_path": workspace.get("metadata_path"),
        "variant_index": workspace.get("variant_index"),
    }


def _persistent_codex_prompt(implementation_contract: dict[str, Any], edit_policy: DynamicEditPolicy, *, prompt_kind: str) -> str:
    prompt = _codex_patch_prompt(implementation_contract, edit_policy)
    repeated_failure_note = ""
    if isinstance(implementation_contract.get("repeated_failure_context"), dict) and implementation_contract["repeated_failure_context"].get("is_repeated"):
        repeated_failure_note = (
            "Repeated failure warning: repeated_failure_context.is_repeated=true. "
            "ordinary same-path repair has already failed; change the implementation repair target or wiring boundary while preserving the candidate. "
            "Do not keep making small edits around the same identical tensor/switch/cache/changed-files signature.\n\n"
        )
    if prompt_kind == "preload":
        return (
            "Persistent S2.5 Codex session bootstrap and context preload.\n"
            "Inspect the repository enough to form a compact patch blueprint for this idea. "
            "Do not edit files in this turn. Return a concise blueprint with intended files, integration path, "
            "runtime-smoke risks, and why the patch should affect cheap proxy.\n\n"
            + repeated_failure_note
            + prompt
        )
    if prompt_kind == "repair_diagnosis":
        return (
            "Persistent S2.5 Codex root-cause diagnosis pre-pass.\n"
            "You are in the same implementation repair context for the locked candidate. "
            "Do not edit files in this turn. Diagnose why the current patch is not eligible for S3.\n\n"
            "Mandatory environment rule: run any lightweight Python command with the supplied C2C conda env python "
            "from implementation_repair_diagnosis.environment.python_cmd. Do not use system python unless that exact path is unavailable.\n\n"
            "Read the listed artifacts when present: plan/s2_5_repair_dispatch.json, plan/code_patches/patch_manifest.json, "
            "the failed patch implementation_contract.json, and changed_files. Use rg/file reads to inspect whether config reaches rosetta_config, "
            "constructors, wrapper/projector/aligner forward, and enabled/disabled tensor changes.\n\n"
            "Allowed lightweight commands only: py_compile, focused/targeted smoke, and forward probe with the configured env python. "
            "Do not run full train, large proxy, distributed jobs, or edit evaluator/validation code.\n\n"
            "Return a concise JSON object with keys root_cause, evidence, repair_target, forbidden, lightweight_commands_run, env_python_used, confidence. "
            "If implementation_repair_diagnosis.repeated_failure_context.is_repeated is true, explicitly name the repeated signals and choose a different repair target or lower-level wiring boundary; do not recommend another small edit to the same unchanged files/tensors.\n"
            "If exact JSON is impossible, return a short markdown section with the same fields.\n\n"
            + repeated_failure_note
            + prompt
        )
    if implementation_contract.get("codex_repair_packet"):
        return (
            "Persistent S2.5 Codex same-session repair turn.\n"
            "You are continuing the same implementation context for the current idea. "
            "Do not restart from high-level planning and do not change the research direction. "
            "Read codex_repair_packet first: it contains failed commands, traces, changed files, config/probe evidence, and the current diff excerpt. "
            "Edit repository files directly so the current patch becomes eligible for S3. "
            "If activation or forward-probe evidence failed, repair the real config -> constructor -> forward -> tensor path; do not edit validation/probe code to bypass the check.\n\n"
            + repeated_failure_note
            + prompt
        )
    return (
        "Persistent S2.5 Codex repair turn. This is an implementation turn, not a planning turn. "
        "You must edit repository files directly before your final answer. Do not return only a blueprint, plan, analysis, or intended-files list. "
        "Use the supplied validation or contract feedback as the primary task, keep the diff narrow, "
        "and avoid re-designing unrelated mechanism pieces. If you conclude no edit is needed, verify the exact failing check is already fixed; otherwise make the minimal code edit that fixes it.\n\n"
        + repeated_failure_note
        + prompt
    )


def _write_persistent_patch_blueprint(workspace: dict[str, Any], preload_result: dict[str, Any]) -> None:
    metadata_path = Path(str(workspace.get("metadata_path") or ""))
    if not metadata_path:
        return
    blueprint_path = metadata_path.parent / "patch_blueprint.json"
    payload = {
        "created_at": now_utc(),
        "session_key": workspace.get("session_key"),
        "session_id": preload_result.get("session_id"),
        "rationale": preload_result.get("rationale") or "",
        "codex_call": preload_result.get("codex_call") or {},
    }
    write_json(blueprint_path, payload)


def _load_persistent_codex_session(session_path: Path) -> str | None:
    if not session_path.exists():
        return None
    try:
        payload = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    session_id = payload.get("session_id")
    return str(session_id) if session_id else None


def _save_persistent_codex_session(
    session_path: Path,
    session_id: str,
    config: dict[str, Any],
    *,
    workspace: dict[str, Any],
    call_record: dict[str, Any],
) -> None:
    payload = {
        "session_id": session_id,
        "provider": "codex_cli",
        "model": (config.get("llm") or {}).get("model"),
        "updated_at": now_utc(),
        "session_key": workspace.get("session_key"),
        "repo": workspace.get("repo"),
        "branch": workspace.get("branch"),
        "last_call": call_record,
    }
    write_json(session_path, payload)


def _sync_project_codex_session(
    project_root: Path,
    session_key: str,
    session_id: str,
    config: dict[str, Any],
    *,
    workspace: dict[str, Any],
) -> None:
    if not session_key:
        return
    path = project_root / "meta" / "codex_sessions.yaml"
    payload = read_yaml(path, default={"sessions": {}}) or {"sessions": {}}
    payload.setdefault("sessions", {})
    payload["sessions"][session_key] = {
        "session_id": session_id,
        "provider": "codex_cli",
        "model": (config.get("llm") or {}).get("model"),
        "updated_at": now_utc(),
        "repo": workspace.get("repo"),
        "branch": workspace.get("branch"),
    }
    write_yaml(path, payload)


def _append_codex_events(events_path: Path, stdout: str, call_record: dict[str, Any]) -> None:
    ensure_dir(events_path.parent)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "auto_research_codex_call", **call_record}, ensure_ascii=False) + "\n")
        for line in (stdout or "").splitlines():
            if line.strip():
                handle.write(line.rstrip() + "\n")


def _codex_call_record(
    started_at: str,
    start_monotonic: float,
    *,
    session_id: str | None,
    parsed_session_id: str | None,
    command_kind: str,
    prompt_kind: str,
    returncode: int,
) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "ended_at": now_utc(),
        "duration_seconds": round(time.monotonic() - start_monotonic, 3),
        "session_id": parsed_session_id or session_id,
        "previous_session_id": session_id,
        "command_kind": command_kind,
        "prompt_kind": prompt_kind,
        "returncode": returncode,
    }


def _parse_codex_session_id(stderr: str, stdout: str = "") -> str | None:
    match = re.search(r"session id:\s*([0-9a-fA-F-]+)", stderr or "")
    if match:
        return match.group(1)
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started" and event.get("thread_id"):
            return str(event["thread_id"])
    return None


def _codex_sandbox_error(result: dict[str, Any]) -> bool:
    text = "\n".join(
        str(result.get(key) or "")
        for key in ["reason", "stderr", "stdout", "rationale"]
    ).lower()
    indicators = [
        "bwrap:",
        "bubblewrap",
        "sandbox error",
        "sandbox before",
        "can't bind mount",
        "cannot bind mount",
        "unable to apply mount flags",
        "no such device",
    ]
    return any(indicator in text for indicator in indicators)


def _codex_retryable_error_text(*parts: Any) -> bool:
    text = "\n".join(str(part or "") for part in parts).lower()
    indicators = [
        "429",
        "too many requests",
        "rate limit",
        "rate_limit",
        "ratelimit",
        "insufficient_quota",
        "exceeded your current quota",
        "quota exceeded",
        "billing hard limit",
        "billing limit",
        "payment required",
        "temporarily unavailable",
        "retry-after",
    ]
    return any(indicator in text for indicator in indicators)


def _patch_failure_retryable(entry: dict[str, Any] | None) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("retryable") is True:
        return True
    if entry.get("resource_retry") is True:
        return True
    if entry.get("validation_status") == "runtime_smoke_resource_retry":
        return True
    if entry.get("status") == "retryable_codex_failed":
        return True
    if entry.get("failure_category") in {"llm_rate_limit_or_quota", "runtime_smoke_resource_retry"}:
        return True
    if _codex_retryable_error_text(entry.get("reason"), entry.get("stderr"), entry.get("stdout")):
        return True
    validation = entry.get("validation") if isinstance(entry.get("validation"), dict) else {}
    if _validation_has_runtime_resource_retry(validation):
        return True
    for action in entry.get("recovery_actions") or []:
        if isinstance(action, dict) and _patch_failure_retryable(action):
            return True
    for attempt in entry.get("variant_attempts") or []:
        if isinstance(attempt, dict) and _patch_failure_retryable(attempt):
            return True
    return False


def is_retryable_patch_manifest(manifest: dict[str, Any] | None) -> bool:
    if not isinstance(manifest, dict):
        return False
    if manifest.get("retryable") is True:
        return True
    if manifest.get("status") == "retryable_no_valid_patch":
        return True
    try:
        if int(manifest.get("retryable_patch_count") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    entries = manifest.get("patches") or manifest.get("candidates") or []
    return any(_patch_failure_retryable(entry) for entry in entries if isinstance(entry, dict))


def _normalize_failed_patch_status(status: str, reason: str, recovery_actions: list[dict[str, Any]]) -> dict[str, Any]:
    retryable = status == "retryable_codex_failed" or _codex_retryable_error_text(reason)
    if not retryable:
        retryable = any(_patch_failure_retryable(action) for action in recovery_actions if isinstance(action, dict))
    normalized_status = "retryable_codex_failed" if retryable and status == "codex_failed" else status
    return {
        "status": normalized_status,
        "reason": reason,
        "retryable": retryable,
        "failure_category": "llm_rate_limit_or_quota" if retryable else normalized_status,
    }


def _codex_sandbox_reason(result: dict[str, Any]) -> str:
    for key in ["reason", "stderr", "rationale", "stdout"]:
        value = str(result.get(key) or "").strip()
        if value:
            return value[-1000:]
    return "Codex sandbox failure detected"


def _compact_backend_attempt(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "sandbox": result.get("sandbox"),
        "session_id": result.get("session_id"),
        "previous_session_id": result.get("previous_session_id"),
        "same_session_reused": result.get("same_session_reused"),
        "reason": str(result.get("reason") or "")[-1000:],
        "stderr": str(result.get("stderr") or "")[-1000:],
        "stdout": str(result.get("stdout") or "")[-1000:],
        "rationale": str(result.get("rationale") or "")[-1000:],
    }


def _repair_session_reuse_report(
    config: dict[str, Any],
    primary_result: dict[str, Any],
    repair: dict[str, Any],
    *,
    backend: Any,
) -> dict[str, Any]:
    primary_call = primary_result.get("codex_call") if isinstance(primary_result.get("codex_call"), dict) else {}
    repair_call = repair.get("codex_call") if isinstance(repair.get("codex_call"), dict) else {}
    before_session = (
        primary_result.get("session_id")
        or primary_call.get("session_id")
        or primary_call.get("previous_session_id")
    )
    after_session = repair.get("session_id") or repair_call.get("session_id")
    repair_previous_session = repair.get("previous_session_id") or repair_call.get("previous_session_id")
    persistent_backend = isinstance(backend, CodexPersistentPatchBackend)
    same_session_reused = bool(
        persistent_backend
        and before_session
        and (
            after_session == before_session
            or repair_previous_session == before_session
            or repair.get("same_session_reused") is True
        )
    )
    del config
    report = {
        "required": True,
        "backend": type(backend).__name__,
        "persistent_backend": persistent_backend,
        "session_id_before": before_session,
        "session_id_after": after_session,
        "repair_previous_session_id": repair_previous_session,
        "same_session_reused": same_session_reused,
        "resume_command_used": bool(repair_previous_session),
    }
    if not persistent_backend:
        report["warning"] = "same-session repair requires codex_persistent_cli"
    elif before_session and not same_session_reused:
        report["warning"] = "repair did not reuse the previous Codex session"
    return report


def _s2_5_repair_session_policy() -> dict[str, Any]:
    return {
        "same_resume_session_required": True,
        "do_not_replan_method": True,
        "repair_scope": "implementation_only",
        "instruction": (
            "Continue from the existing S2.5 Codex context when available. "
            "Use codex_repair_packet as the source of truth for failed commands, traces, changed files, config/probe evidence, and the current diff. "
            "Repair the current implementation until it is eligible for S3; do not send this failure back to S1/S2 as method evidence."
        ),
    }


def _contract_repair_packet(failure: dict[str, Any], primary_result: dict[str, Any], *, attempt: int) -> dict[str, Any]:
    return {
        "schema_version": "s2_5_codex_repair_packet_v1",
        "repair_kind": "contract_failure",
        "attempt": attempt,
        "failed_status": failure.get("status"),
        "failed_reason": _contract_failure_reason(failure),
        "errors": _compact_value(failure.get("errors") or [], max_chars=3000),
        "changed_files": list(failure.get("changed_files") or [])[:30],
        "risk_labels": list(failure.get("risk_labels") or [])[:20],
        "risk_files": _compact_value(failure.get("risk_files") or [], max_chars=4000),
        "diff_excerpt": _validation_output_excerpt(str(failure.get("diff") or ""), max_chars=8000),
        "primary_attempt": _compact_backend_attempt(primary_result),
        "instruction": (
            "Repair the existing patch in the same Codex resume context when available. "
            "Use the failed_status/reason and diff_excerpt to fix the concrete implementation problem; keep the method idea unchanged."
        ),
    }


def _validation_repair_packet(
    validation: dict[str, Any],
    draft: dict[str, Any],
    primary_result: dict[str, Any],
    *,
    attempt: int,
) -> dict[str, Any]:
    return {
        "schema_version": "s2_5_codex_repair_packet_v1",
        "repair_kind": "validation_or_activation_failure",
        "attempt": attempt,
        "failed_status": validation.get("status"),
        "changed_files": list(draft.get("changed_files") or validation.get("changed_files") or [])[:30],
        "diff_excerpt": _validation_output_excerpt(str(draft.get("diff") or ""), max_chars=10000),
        "failed_checks": _failed_validation_checks(validation, limit=8),
        "failed_command_evidence": _failed_validation_command_evidence(validation, limit=8),
        "activation_check": _compact_value(validation.get("activation_check") or {}, max_chars=5000),
        "risk_check": _compact_value(validation.get("risk_check") or {}, max_chars=5000),
        "mechanism_review": _compact_value(validation.get("mechanism_review") or {}, max_chars=5000),
        "activation_probe_evidence": _activation_probe_evidence_from_validation(validation),
        "activation_forward_probe_diagnostics": _forward_probe_diagnostics_from_validation(validation),
        "repeated_failure_context": primary_result.get("repeated_failure_context") if isinstance(primary_result.get("repeated_failure_context"), dict) else {},
        "primary_attempt": _compact_backend_attempt(primary_result),
        "instruction": (
            "Repair the current checkout, not the research direction. "
            "Prioritize failed_command_evidence, activation_check, activation_probe_evidence, and activation_forward_probe_diagnostics. "
            "For forward-probe failures, use the unchanged tensor names, enabled/disabled sha pairs, switch/config status, "
            "and repair_focus to fix constructor params, forward branches, or wrapper parameter passing instead of editing validation code."
            " If repeated_failure_context.is_repeated is true, stop ordinary same-path repair and change the wiring boundary or integration target while preserving the candidate."
        ),
    }


def _failed_validation_command_evidence(validation: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for check in validation.get("checks") or []:
        if not isinstance(check, dict) or check.get("returncode") == 0:
            continue
        evidence.append(
            {
                "name": check.get("name"),
                "status": check.get("status"),
                "returncode": check.get("returncode"),
                "blocking": check.get("blocking"),
                "failure_category": check.get("failure_category"),
                "command": check.get("command"),
                "switch": check.get("switch"),
                "probe_environment": check.get("probe_environment") if str(check.get("name") or "") == "runtime_smoke:mechanism_activation_forward_probe" and isinstance(check.get("probe_environment"), dict) else {},
                "repair_hint": check.get("repair_hint"),
                "stdout_tail": str(check.get("stdout") or "")[-2200:],
                "stderr_tail": str(check.get("stderr") or "")[-2200:],
                "forward_probe_diagnostics": _forward_probe_diagnostics_from_check(check)
                if str(check.get("name") or "") == "runtime_smoke:mechanism_activation_forward_probe"
                else {},
            }
        )
    return evidence[:limit]


def _activation_probe_evidence_from_validation(validation: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for check in validation.get("checks") or []:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or "")
        if name == "runtime_smoke:mechanism_activation_wiring":
            result["wiring"] = {
                "status": check.get("status"),
                "returncode": check.get("returncode"),
                "failure_category": check.get("failure_category"),
                "stderr": check.get("stderr"),
                "switch": check.get("switch"),
                "enabled_eval_configs": check.get("enabled_eval_configs"),
                "disabled_eval_configs": check.get("disabled_eval_configs"),
                "rosetta_config": check.get("rosetta_config"),
                "runtime_code_refs": check.get("runtime_code_refs"),
            }
        elif name == "runtime_smoke:mechanism_activation_forward_probe":
            diagnostics = _forward_probe_diagnostics_from_check(check)
            result["forward_probe"] = {
                "status": check.get("status"),
                "returncode": check.get("returncode"),
                "failure_category": check.get("failure_category"),
                "stderr": check.get("stderr"),
                "command": check.get("command"),
                "switch": check.get("switch"),
                "probe_source": check.get("probe_source"),
                "probe_script": check.get("probe_script"),
                "probe_environment": check.get("probe_environment") if isinstance(check.get("probe_environment"), dict) else {},
                "probe": _compact_value(check.get("probe") or {}, max_chars=5000),
                "diagnostics": diagnostics,
            }
    return result


def _forward_probe_diagnostics_from_validation(validation: dict[str, Any]) -> dict[str, Any]:
    for check in validation.get("checks") or []:
        if isinstance(check, dict) and str(check.get("name") or "") == "runtime_smoke:mechanism_activation_forward_probe":
            return _forward_probe_diagnostics_from_check(check)
    return {}


def _forward_probe_diagnostics_from_check(check: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(check, dict):
        return {}
    probe = check.get("probe") if isinstance(check.get("probe"), dict) else {}
    if not probe:
        return {}
    tensor_checks = [
        _compact_forward_tensor_check(item)
        for item in probe.get("tensor_checks") or []
        if isinstance(item, dict)
    ]
    tensor_checks = [item for item in tensor_checks if item]
    changed_tensors = [item for item in tensor_checks if item.get("changed")]
    identical_tensors = [item for item in tensor_checks if item and not item.get("changed")]
    projector_checks = [item for item in tensor_checks if str(item.get("name") or "").startswith("projector_output.")]
    wrapper_checks = [item for item in tensor_checks if str(item.get("name") or "").startswith("wrapper_cache.")]
    wrapper_probe = probe.get("wrapper_probe") if isinstance(probe.get("wrapper_probe"), dict) else {}
    projector_output_identical = bool(projector_checks) and not any(item.get("changed") for item in projector_checks)
    wrapper_cache_identical = bool(wrapper_checks) and not any(item.get("changed") for item in wrapper_checks)
    projector_called = _first_bool(probe.get("projector_called"), wrapper_probe.get("projector_called"))
    switch_seen_by_forward = _first_bool(probe.get("switch_seen_by_forward"), wrapper_probe.get("switch_seen_by_forward"))
    cache_key_diff = _first_non_none(probe.get("cache_key_diff"), wrapper_probe.get("cache_key_diff"))
    cache_value_diff = _first_non_none(probe.get("cache_value_diff"), wrapper_probe.get("cache_value_diff"))
    enabled = probe.get("enabled") if isinstance(probe.get("enabled"), dict) else {}
    disabled = probe.get("disabled") if isinstance(probe.get("disabled"), dict) else {}
    diagnostics = {
        "status": check.get("status"),
        "returncode": check.get("returncode"),
        "failure_category": check.get("failure_category"),
        "switch": check.get("switch"),
        "probe_source": check.get("probe_source"),
        "probe_type": probe.get("probe_type"),
        "probe_environment": check.get("probe_environment") if isinstance(check.get("probe_environment"), dict) else {},
        "mechanism_observed": bool(probe.get("mechanism_observed")),
        "switch_config": {
            "enabled_value": enabled.get("switch_value"),
            "disabled_value": disabled.get("switch_value"),
            "enabled_rosetta_hash": enabled.get("rosetta_hash"),
            "disabled_rosetta_hash": disabled.get("rosetta_hash"),
        },
        "projector_called": projector_called,
        "switch_seen_by_forward": switch_seen_by_forward,
        "cache_key_diff": cache_key_diff,
        "cache_value_diff": cache_value_diff,
        "projector_output_identical": projector_output_identical,
        "wrapper_cache_identical": wrapper_cache_identical,
        "changed_tensors": changed_tensors[:8],
        "identical_tensors": identical_tensors[:12],
        "repair_focus": _forward_probe_repair_focus(
            disabled_switch=disabled.get("switch_value"),
            projector_called=projector_called,
            switch_seen_by_forward=switch_seen_by_forward,
            projector_output_identical=projector_output_identical,
            wrapper_cache_identical=wrapper_cache_identical,
            changed_tensors=changed_tensors,
        ),
    }
    if wrapper_probe:
        diagnostics["wrapper_probe_status"] = wrapper_probe.get("status")
        diagnostics["wrapper_failures"] = list(wrapper_probe.get("failures") or [])[:8]
    return diagnostics


def _compact_forward_tensor_check(item: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name") or "")
    if not name:
        return {}
    return {
        "name": name,
        "changed": bool(item.get("changed")),
        "max_abs_diff": item.get("max_abs_diff"),
        "mean_abs_diff": item.get("mean_abs_diff"),
        "enabled_sha256": item.get("enabled_sha256"),
        "disabled_sha256": item.get("disabled_sha256"),
        "shape": item.get("shape") if isinstance(item.get("shape"), list) else [],
    }


def _forward_probe_repair_focus(
    *,
    disabled_switch: Any,
    projector_called: bool | None,
    switch_seen_by_forward: bool | None,
    projector_output_identical: bool,
    wrapper_cache_identical: bool,
    changed_tensors: list[dict[str, Any]],
) -> list[str]:
    focus: list[str] = []
    if disabled_switch is not True:
        focus.append("config_materialization_or_ablation_switch_polarity")
    if switch_seen_by_forward is False:
        focus.append("forward_branch_missing_switch_or_rosetta_config_read")
    if projector_called is False:
        focus.append("wrapper_projection_path_not_executed_or_projector_not_called")
    if projector_output_identical:
        focus.append("constructor_params_or_projector_forward_branch_noop")
    if changed_tensors and wrapper_cache_identical:
        focus.append("wrapper_cache_injection_not_using_changed_projector_output")
    if wrapper_cache_identical:
        focus.append("wrapper_cache_key_value_identical_enabled_disabled")
    return list(dict.fromkeys(focus)) or ["inspect_forward_probe_tensor_checks"]


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _repair_packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    failed_checks = []
    for check in packet.get("failed_checks") or packet.get("failed_command_evidence") or []:
        if isinstance(check, dict):
            failed_checks.append(str(check.get("name") or check.get("failure_category") or "")[:120])
    return {
        "schema_version": packet.get("schema_version"),
        "repair_kind": packet.get("repair_kind"),
        "attempt": packet.get("attempt"),
        "failed_status": packet.get("failed_status"),
        "failed_reason": str(packet.get("failed_reason") or "")[-600:],
        "failed_checks": [item for item in failed_checks if item][:8],
        "changed_files": list(packet.get("changed_files") or [])[:12],
    }


def _compact_single_patch_attempt(entry: dict[str, Any]) -> dict[str, Any]:
    quality_score = entry.get("quality_score") if isinstance(entry.get("quality_score"), dict) else {}
    return {
        "variant_index": 1,
        "status": entry.get("status"),
        "reason": str(entry.get("reason") or "")[-1000:],
        "changed_files": list(entry.get("changed_files") or [])[:12],
        "has_executable_change": bool(entry.get("has_executable_change")),
        "risk_check": entry.get("risk_check") or {},
        "quality_score": quality_score,
        "validation": entry.get("validation"),
    }


def _annotate_single_patch_attempt_result(
    result: dict[str, Any],
    *,
    attempts: list[dict[str, Any]],
    selection_reason: str = "single_patch_attempt",
) -> None:
    for key in ("code_patch", "manifest_entry"):
        value = result.get(key)
        if not isinstance(value, dict):
            continue
        value["variant_index"] = 1
        value["variant_count"] = 1
        value["selected_variant"] = 1 if value.get("status") == "ok" else None
        value["variant_attempts"] = list(attempts)
        value["selection_reason"] = selection_reason


def _select_single_s2_5_candidate(plan: dict[str, Any], candidate_ideas: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [(idx, item) for idx, item in enumerate(candidate_ideas) if isinstance(item, dict)]
    if not candidates:
        return {"candidate": None, "selected_index": None, "selected_by": "none", "selected_candidate_id": None}
    for idx, candidate in candidates:
        if candidate.get("selected") is True:
            return {
                "candidate": candidate,
                "selected_index": idx,
                "selected_by": "candidate.selected",
                "selected_candidate_id": candidate.get("id"),
            }
    selected_idea = plan.get("selected_idea") if isinstance(plan, dict) and isinstance(plan.get("selected_idea"), dict) else {}
    matched = _match_s2_5_candidate_selector(candidates, selected_idea)
    if matched:
        idx, candidate = matched
        return {
            "candidate": candidate,
            "selected_index": idx,
            "selected_by": "plan.selected_idea",
            "selected_candidate_id": candidate.get("id"),
        }
    next_variant = plan.get("next_variant") if isinstance(plan, dict) and isinstance(plan.get("next_variant"), dict) else {}
    matched = _match_s2_5_candidate_selector(candidates, next_variant)
    if matched:
        idx, candidate = matched
        return {
            "candidate": candidate,
            "selected_index": idx,
            "selected_by": "plan.next_variant",
            "selected_candidate_id": candidate.get("id"),
        }
    selected_variants = plan.get("selected_variant_candidates") if isinstance(plan, dict) else None
    if isinstance(selected_variants, list):
        for selector in selected_variants:
            if not isinstance(selector, dict):
                continue
            matched = _match_s2_5_candidate_selector(candidates, selector)
            if matched:
                idx, candidate = matched
                return {
                    "candidate": candidate,
                    "selected_index": idx,
                    "selected_by": "plan.selected_variant_candidates",
                    "selected_candidate_id": candidate.get("id"),
                }
    idx, candidate = candidates[0]
    return {
        "candidate": candidate,
        "selected_index": idx,
        "selected_by": "first_candidate_fallback",
        "selected_candidate_id": candidate.get("id"),
    }


def _match_s2_5_candidate_selector(
    candidates: list[tuple[int, dict[str, Any]]],
    selector: dict[str, Any],
) -> tuple[int, dict[str, Any]] | None:
    if not isinstance(selector, dict) or not selector:
        return None
    selector_id = str(selector.get("id") or selector.get("candidate_id") or "").strip()
    selector_fingerprint = str(
        selector.get("variant_fingerprint")
        or ((selector.get("s2_variant") or {}).get("variant_fingerprint") if isinstance(selector.get("s2_variant"), dict) else "")
        or ""
    ).strip()
    for idx, candidate in candidates:
        candidate_id = str(candidate.get("id") or candidate.get("candidate_id") or "").strip()
        candidate_fingerprint = str(
            candidate.get("variant_fingerprint")
            or ((candidate.get("s2_variant") or {}).get("variant_fingerprint") if isinstance(candidate.get("s2_variant"), dict) else "")
            or ""
        ).strip()
        if selector_id and candidate_id and selector_id == candidate_id:
            return idx, candidate
        if selector_fingerprint and candidate_fingerprint and selector_fingerprint == candidate_fingerprint:
            return idx, candidate
    return None


def _compact_skipped_patch_candidate(candidate: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("id") or candidate.get("candidate_id"),
        "title": candidate.get("title"),
        "variant_fingerprint": candidate.get("variant_fingerprint")
        or ((candidate.get("s2_variant") or {}).get("variant_fingerprint") if isinstance(candidate.get("s2_variant"), dict) else None),
        "status": (candidate.get("code_patch") or {}).get("status"),
        "reason": reason,
    }


def _compact_selected_manifest_patch(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict) or not entry:
        return {}
    return {
        "candidate_id": entry.get("candidate_id") or entry.get("id"),
        "title": entry.get("title"),
        "status": entry.get("status"),
        "variant_fingerprint": entry.get("variant_fingerprint"),
        "s2_variant": entry.get("s2_variant") or {},
        "patch_json": entry.get("patch_json"),
        "diff": entry.get("diff"),
        "validation": entry.get("validation"),
        "implementation_contract": entry.get("implementation_contract"),
        "codex_prompt": entry.get("codex_prompt"),
        "patched_repo_snapshot": entry.get("patched_repo_snapshot") or {},
        "changed_files": list(entry.get("changed_files") or []),
        "has_executable_change": bool(entry.get("has_executable_change")),
        "quality_score": entry.get("quality_score") or {},
        "quality_debt": entry.get("quality_debt") or (entry.get("quality_score") or {}).get("quality_debt") or [],
        "selected_variant": entry.get("selected_variant"),
        "selection_reason": entry.get("selection_reason"),
    }


def _patch_quality_score(
    draft: dict[str, Any],
    validation: dict[str, Any],
    activation: dict[str, Any],
    risk_check: dict[str, Any],
    mechanism_review: dict[str, Any],
    implementation_contract: dict[str, Any],
) -> dict[str, Any]:
    changed_files = list(draft.get("changed_files") or [])
    diff_line_count = len(str(draft.get("diff") or "").splitlines())
    score_inputs = mechanism_review.get("score_inputs") if isinstance(mechanism_review.get("score_inputs"), dict) else {}
    quality_debt = _patch_quality_debt(validation, activation, risk_check, mechanism_review)
    score = 0
    reasons: list[str] = []
    if validation.get("status") == "ok":
        score += 20
        reasons.append("validation_ok")
    if activation.get("status") == "ok":
        score += 10
        reasons.append("activation_ok")
    if risk_check.get("status") == "ok":
        score += 15
        reasons.append("risk_ok")
    if mechanism_review.get("status") == "ok":
        score += 20
        reasons.append("mechanism_review_ok")
    if score_inputs.get("core_mechanism_files"):
        score += 12
        reasons.append("core_mechanism_files")
    if score_inputs.get("expected_file_hits"):
        score += 8
        reasons.append("expected_file_hits")
    if score_inputs.get("ablation_wired"):
        score += 8
        reasons.append("ablation_wired")
    if score_inputs.get("coverage_evidence"):
        score += 6
        reasons.append("coverage_evidence")
    if score_inputs.get("matched_coverage_evidence"):
        score += 6
        reasons.append("matched_coverage_evidence")
    soft_issues = list(mechanism_review.get("soft_issues") or [])
    if soft_issues:
        score -= 4 * len(soft_issues)
        reasons.append("soft_quality_gaps")
    if quality_debt:
        score -= 3 * len(quality_debt)
        reasons.append("quality_debt")
    if any(str(check.get("name") or "").startswith("runtime_smoke:") and check.get("returncode") == 0 for check in validation.get("checks") or [] if isinstance(check, dict)):
        score += 12
        reasons.append("runtime_smoke_ok")
    if any(str(check.get("name") or "").startswith("pytest:") and check.get("returncode") == 0 for check in validation.get("checks") or [] if isinstance(check, dict)):
        score += 4
        reasons.append("focused_tests_ok")
    risk_labels = list(risk_check.get("risk_labels") or [])
    evaluator_like_files = list(score_inputs.get("evaluator_like_files") or [])
    penalty = 0
    if risk_labels:
        penalty += 12 * len(risk_labels)
    if evaluator_like_files:
        penalty += 20
    if diff_line_count > 900:
        penalty += min(25, (diff_line_count - 900) // 100 + 1)
    if len(changed_files) > 6:
        penalty += 4 * (len(changed_files) - 6)
    score -= penalty
    return {
        "score": score,
        "reasons": reasons,
        "penalty": penalty,
        "risk_labels": risk_labels,
        "soft_issues": soft_issues,
        "quality_debt": quality_debt,
        "changed_file_count": len(changed_files),
        "diff_line_count": diff_line_count,
        "mechanism_review_status": mechanism_review.get("status"),
        "mechanism_type": (implementation_contract.get("mechanism_contract") or {}).get("mechanism_type")
        if isinstance(implementation_contract.get("mechanism_contract"), dict)
        else None,
    }


def _patch_quality_debt(
    validation: dict[str, Any],
    activation: dict[str, Any],
    risk_check: dict[str, Any],
    mechanism_review: dict[str, Any],
) -> list[dict[str, Any]]:
    debt: list[dict[str, Any]] = []
    for warning in risk_check.get("warnings") or []:
        debt.append({"source": "risk_check", "label": "patch_scope_warning", "message": str(warning)})
    for label in risk_check.get("risk_labels") or []:
        if label in {"patch_too_broad"} and risk_check.get("status") == "ok":
            debt.append({"source": "risk_check", "label": str(label), "message": "Patch is runnable but broader than the soft discovery limit."})
    for label in activation.get("soft_issues") or []:
        debt.append({"source": "activation_check", "label": str(label), "message": activation.get("reason") or str(label)})
    for parameter in activation.get("missing_parameters") or []:
        if activation.get("status") == "ok" and activation.get("soft_issues"):
            debt.append({"source": "activation_check", "label": "unactivated_config_parameter", "parameter": str(parameter)})
    for label in mechanism_review.get("soft_issues") or []:
        debt.append({"source": "mechanism_review", "label": str(label), "message": "Deferred mechanism quality evidence."})
    for warning in mechanism_review.get("warnings") or []:
        if warning not in mechanism_review.get("soft_issues", []):
            debt.append({"source": "mechanism_review", "label": str(warning), "message": "Mechanism review warning."})
    for check in validation.get("checks") or []:
        if not isinstance(check, dict):
            continue
        if check.get("soft_failure") or check.get("blocking") is False:
            debt.append(
                {
                    "source": "validation",
                    "label": str(check.get("failure_category") or check.get("name") or "soft_validation_failure"),
                    "check": check.get("name"),
                    "message": check.get("stderr") or check.get("repair_hint") or "",
                }
            )
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in debt:
        key = (str(item.get("source") or ""), str(item.get("label") or ""), str(item.get("parameter") or item.get("check") or ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _temporary_patch_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {".git", "wandb", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov"}
    if Path(directory).name == "local":
        ignored.update({"checkpoints", "snapshots", "final_results"})
    for name in names:
        if name in ignored or _path_ignored_for_patch_delta(name):
            ignored.add(name)
    return ignored.intersection(names)


def _build_patch_from_repo_delta(source_repo: Path, modified_repo: Path, policy: DynamicEditPolicy) -> dict[str, Any]:
    errors: list[str] = []
    operations: list[dict[str, Any]] = []
    diff_chunks: list[str] = []
    changed_files: list[str] = []
    source_files = _text_file_map(source_repo, policy)
    modified_files = _text_file_map(modified_repo, policy)
    unauthorized = _unauthorized_delta_paths(source_repo, modified_repo, policy)
    if unauthorized:
        return {"status": "patch_rejected", "errors": [f"unauthorized changed paths: {', '.join(unauthorized[:10])}"], "operations": [], "changed_files": [], "diff": ""}
    for rel_path in sorted(set(source_files) | set(modified_files)):
        source_path = source_repo / rel_path
        modified_path = modified_repo / rel_path
        if rel_path in source_files and rel_path not in modified_files:
            errors.append(f"delete_file is not allowed: {rel_path}")
            continue
        if rel_path not in source_files and rel_path in modified_files:
            new_text = modified_path.read_text(encoding="utf-8")
            operations.append({"op": "add_file", "path": rel_path, "new": new_text})
            changed_files.append(rel_path)
            diff_chunks.extend(_unified_diff("", new_text, rel_path))
            continue
        if source_files[rel_path] != modified_files[rel_path]:
            old_text = source_path.read_text(encoding="utf-8")
            new_text = modified_path.read_text(encoding="utf-8")
            operations.append({"op": "replace_file", "path": rel_path, "old_sha256": sha256_file(source_path), "new": new_text})
            changed_files.append(rel_path)
            diff_chunks.extend(_unified_diff(old_text, new_text, rel_path))
    if errors:
        return {"status": "patch_rejected", "errors": errors, "operations": [], "changed_files": [], "diff": ""}
    if not operations:
        return {"status": "blocked_no_executable_change", "errors": ["Codex produced no allowed file changes"], "operations": [], "changed_files": [], "diff": ""}
    return {"status": "ok", "errors": [], "operations": operations, "changed_files": changed_files, "diff": "\n".join(diff_chunks)}


def _auto_prune_worktree_scope_before_build(
    source_repo: Path,
    modified_repo: Path,
    policy: DynamicEditPolicy,
    candidate: dict[str, Any],
    implementation_contract: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    validation_config = _code_patch_config(config).get("validation", {}) or {}
    if not validation_config.get("auto_prune_scope", True):
        return None

    source_files = _text_file_map(source_repo, policy)
    modified_files = _text_file_map(modified_repo, policy)
    changed_files = [
        rel_path
        for rel_path in sorted(set(source_files) | set(modified_files))
        if source_files.get(rel_path) != modified_files.get(rel_path)
    ]
    if not changed_files:
        return None

    restored: list[str] = []
    reasons: dict[str, str] = {}

    for rel_path in changed_files:
        if rel_path in source_files and rel_path not in modified_files:
            if _restore_repo_path(source_repo, modified_repo, rel_path):
                restored.append(rel_path)
                reasons[rel_path] = "delete_file_not_allowed"
        elif _evaluator_like_path(rel_path):
            if _restore_repo_path(source_repo, modified_repo, rel_path):
                restored.append(rel_path)
                reasons[rel_path] = "evaluator_like_file"

    if restored:
        source_files = _text_file_map(source_repo, policy)
        modified_files = _text_file_map(modified_repo, policy)
        changed_files = [
            rel_path
            for rel_path in sorted(set(source_files) | set(modified_files))
            if source_files.get(rel_path) != modified_files.get(rel_path)
        ]

    max_changed_files = validation_config.get("max_changed_files")
    if (
        max_changed_files is not None
        and len(changed_files) > int(max_changed_files)
        and validation_config.get("auto_prune_over_scope_files", False)
    ):
        keep = set(
            _allowed_core_patch_files(
                changed_files,
                candidate,
                implementation_contract,
                max_files=int(max_changed_files),
            )
        )
        for rel_path in changed_files:
            if rel_path in keep:
                continue
            if _restore_repo_path(source_repo, modified_repo, rel_path):
                restored.append(rel_path)
                reasons[rel_path] = "over_scope_low_priority_file"

    if not restored:
        return None
    return {
        "action": "auto_prune_worktree_scope_before_build",
        "status": "ok",
        "restored_files": list(dict.fromkeys(restored)),
        "reasons": reasons,
        "reason": "restored deletions, evaluator-like files, or low-priority over-scope files before patch construction",
    }


def _auto_prune_worktree_scope_before_codex(
    source_repo: Path,
    modified_repo: Path,
    policy: DynamicEditPolicy,
    candidate: dict[str, Any],
    implementation_contract: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    validation_config = _code_patch_config(config).get("validation", {}) or {}
    if not validation_config.get("auto_prune_scope", True):
        return None

    source_files = _text_file_map(source_repo, policy)
    modified_files = _text_file_map(modified_repo, policy)
    changed_files = [
        rel_path
        for rel_path in sorted(set(source_files) | set(modified_files))
        if source_files.get(rel_path) != modified_files.get(rel_path)
    ]
    if not changed_files:
        return None

    keep = _pre_codex_keep_files(changed_files, candidate, implementation_contract)
    restored: list[str] = []
    reasons: dict[str, str] = {}
    for rel_path in changed_files:
        reason = ""
        if rel_path in source_files and rel_path not in modified_files:
            reason = "delete_file_not_allowed"
        elif _evaluator_like_path(rel_path):
            reason = "evaluator_like_file"
        elif rel_path not in keep:
            reason = "stale_unrequested_worktree_change"
        if reason and _restore_repo_path(source_repo, modified_repo, rel_path):
            restored.append(rel_path)
            reasons[rel_path] = reason

    if not restored:
        return None
    return {
        "action": "auto_prune_worktree_scope_before_codex",
        "status": "ok",
        "restored_files": list(dict.fromkeys(restored)),
        "reasons": reasons,
        "reason": "restored stale persistent worktree changes before Codex sees the implementation repair context",
    }


def _pre_codex_keep_files(
    changed_files: list[str],
    candidate: dict[str, Any],
    implementation_contract: dict[str, Any],
) -> set[str]:
    expected = set(_expected_file_list(((implementation_contract.get("implementation_targets") or {}).get("expected_files"))))
    scope = implementation_contract.get("implementation_scope") if isinstance(implementation_contract.get("implementation_scope"), dict) else {}
    required = set(_expected_file_list(scope.get("required_new_files")))
    smoke_tests = set(_expected_file_list(scope.get("smoke_tests")))
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    explicit = set(
        _expected_file_list(candidate.get("expected_files"))
        + _expected_file_list(contract.get("expected_files"))
        + _expected_file_list(candidate.get("required_new_files"))
        + _expected_file_list((candidate.get("implementation_plan") or {}).get("required_new_files") if isinstance(candidate.get("implementation_plan"), dict) else [])
    )
    keep: set[str] = set()
    keep.update(path for path in changed_files if path in expected or path in required or path in smoke_tests or path in explicit)
    keep.update(path for path in changed_files if _model_mechanism_path(path))
    keep.update(path for path in changed_files if _focused_test_path(path))
    keep.update(path for path in changed_files if _recipe_path(path))
    return keep


def _build_pruned_patch_from_repo_delta(
    source_repo: Path,
    modified_repo: Path,
    policy: DynamicEditPolicy,
    candidate: dict[str, Any],
    implementation_contract: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    prebuild_action = _auto_prune_worktree_scope_before_build(
        source_repo,
        modified_repo,
        policy,
        candidate,
        implementation_contract,
        config,
    )
    draft = _build_patch_from_repo_delta(source_repo, modified_repo, policy)
    draft, postbuild_action = _auto_prune_patch_scope_and_rebuild(
        source_repo,
        modified_repo,
        policy,
        draft,
        candidate,
        implementation_contract,
        config,
    )
    actions = [action for action in (prebuild_action, postbuild_action) if action]
    if not actions:
        return draft, None
    if len(actions) == 1:
        return draft, actions[0]
    restored: list[str] = []
    reasons: dict[str, str] = {}
    for action in actions:
        restored.extend(str(path) for path in action.get("restored_files") or [])
        if isinstance(action.get("reasons"), dict):
            reasons.update({str(key): str(value) for key, value in action["reasons"].items()})
    return draft, {
        "action": "auto_prune_worktree_and_patch_scope",
        "status": "ok",
        "restored_files": list(dict.fromkeys(restored)),
        "reasons": reasons,
        "steps": actions,
        "reason": "restored unsafe worktree changes before freezing patch",
    }


def _validate_patch_proxy_risk(draft: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    validation_config = _code_patch_config(config).get("validation", {}) or {}
    changed_files = list(draft.get("changed_files") or [])
    executable_files = [path for path in changed_files if Path(path).suffix in {".py", ".json", ".yaml", ".yml", ".toml"}]
    risk_labels: list[str] = []
    risk_files: list[dict[str, Any]] = []
    warnings: list[str] = []
    status = "ok"
    reason = ""
    max_changed_files = validation_config.get("max_changed_files")
    if max_changed_files is not None and len(changed_files) > int(max_changed_files):
        risk_labels.append("patch_too_broad")
        warning = f"patch changes {len(changed_files)} files, above S2.5 soft risk limit {int(max_changed_files)}"
        warnings.append(warning)
        if validation_config.get("strict_max_changed_files", False):
            status = "patch_too_broad"
            reason = warning.replace("soft risk limit", "strict risk limit")
    if validation_config.get("repair_eval_code_changes", True):
        evaluator_files = [path for path in changed_files if _evaluator_like_path(path)]
        if evaluator_files:
            status = "proxy_risk_repair_required"
            reason = "patch changes evaluator code; move mechanism into model/train/recipe files so S3 metrics are not contaminated"
            risk_labels.append("evaluation_code_changed")
            risk_files.extend({"path": path, "reasons": ["evaluation code changed"]} for path in evaluator_files)
    if validation_config.get("repair_test_only_changes", True) and changed_files:
        test_only = all(path.startswith(("test/", "tests/")) for path in changed_files)
        if test_only and not c2c_candidate_config_overrides(candidate):
            status = "proxy_risk_repair_required"
            reason = "patch only changes tests; implement executable model/train/recipe behavior before S3"
            risk_labels.append("test_only_change")
    return {
        "status": status,
        "reason": reason,
        "changed_files": changed_files,
        "executable_files": executable_files,
        "risk_labels": list(dict.fromkeys(risk_labels)),
        "risk_files": risk_files,
        "warnings": warnings,
        "repair_hint": (
            "Keep the idea, but shrink or relocate the patch so cheap proxy can execute it safely."
            if status != "ok"
            else ""
        ),
    }


def _auto_prune_patch_scope_and_rebuild(
    source_repo: Path,
    modified_repo: Path,
    policy: DynamicEditPolicy,
    draft: dict[str, Any],
    candidate: dict[str, Any],
    implementation_contract: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    validation_config = _code_patch_config(config).get("validation", {}) or {}
    if not validation_config.get("auto_prune_scope", True) or draft.get("status") != "ok":
        return draft, None

    changed_files = list(draft.get("changed_files") or [])
    restored: list[str] = []
    reasons: dict[str, str] = {}

    for rel_path in changed_files:
        if _evaluator_like_path(rel_path):
            if _restore_repo_path(source_repo, modified_repo, rel_path):
                restored.append(rel_path)
                reasons[rel_path] = "evaluator_like_file"

    if restored:
        draft = _build_patch_from_repo_delta(source_repo, modified_repo, policy)
        changed_files = list(draft.get("changed_files") or [])

    max_changed_files = validation_config.get("max_changed_files")
    if (
        max_changed_files is not None
        and len(changed_files) > int(max_changed_files)
        and validation_config.get("auto_prune_over_scope_files", False)
    ):
        keep = set(_allowed_core_patch_files(changed_files, candidate, implementation_contract, max_files=int(max_changed_files)))
        for rel_path in changed_files:
            if rel_path in keep:
                continue
            if _restore_repo_path(source_repo, modified_repo, rel_path):
                restored.append(rel_path)
                reasons[rel_path] = "over_scope_low_priority_file"
        draft = _build_patch_from_repo_delta(source_repo, modified_repo, policy)

    if not restored:
        return draft, None
    return draft, {
        "action": "auto_prune_patch_scope_before_freeze",
        "status": "ok",
        "restored_files": list(dict.fromkeys(restored)),
        "reasons": reasons,
        "reason": "restored evaluator or low-priority over-scope files before freezing patch",
    }


def _restore_repo_path(source_repo: Path, modified_repo: Path, rel_path: str) -> bool:
    source = source_repo / rel_path
    target = modified_repo / rel_path
    if source.exists() and source.is_file():
        ensure_dir(target.parent)
        target.write_text(source.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return True
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return True
    return False


def _allowed_core_patch_files(
    changed_files: list[str],
    candidate: dict[str, Any],
    implementation_contract: dict[str, Any],
    *,
    max_files: int,
) -> list[str]:
    expected = set(_expected_file_list(((implementation_contract.get("implementation_targets") or {}).get("expected_files"))))
    scope = implementation_contract.get("implementation_scope") if isinstance(implementation_contract.get("implementation_scope"), dict) else {}
    required = set(_expected_file_list(scope.get("required_new_files")))
    smoke_tests = set(_expected_file_list(scope.get("smoke_tests")))
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    forbidden = set(_expected_file_list(candidate.get("forbidden_files")) + _expected_file_list(contract.get("forbidden_files")))

    keep = [path for path in changed_files if path in required]
    keep.extend(path for path in changed_files if path in expected and path not in keep)
    keep.extend(path for path in changed_files if _model_mechanism_path(path) and path not in keep)
    keep.extend(path for path in changed_files if path in smoke_tests and path not in keep)
    keep.extend(path for path in changed_files if _focused_test_path(path) and path not in keep)
    keep.extend(path for path in changed_files if _recipe_path(path) and path not in keep)
    if len(keep) < max_files:
        keep.extend(path for path in changed_files if path not in keep and path not in forbidden and _train_integration_path(path))
    if len(keep) < max_files:
        keep.extend(path for path in changed_files if path not in keep and path not in forbidden)
    return keep[:max(1, int(max_files))]


def _model_mechanism_path(path: str) -> bool:
    return path.startswith("rosetta/model/") and Path(path).suffix in {".py", ".json", ".yaml", ".yml", ".toml"}


def _focused_test_path(path: str) -> bool:
    return path.startswith(("test/", "tests/")) and Path(path).suffix == ".py"


def _recipe_path(path: str) -> bool:
    return path.startswith("recipe/") and Path(path).suffix in {".json", ".yaml", ".yml", ".toml"}


def _train_integration_path(path: str) -> bool:
    return path.startswith("script/train/") and Path(path).suffix == ".py"


def _restore_proxy_risk_files(source_repo: Path, modified_repo: Path, risk_check: dict[str, Any]) -> dict[str, Any] | None:
    risk_files = risk_check.get("risk_files") or []
    if not isinstance(risk_files, list) or not risk_files:
        return None
    restored: list[str] = []
    for item in risk_files:
        if not isinstance(item, dict):
            continue
        rel_path = str(item.get("path") or "")
        if not _evaluator_like_path(rel_path):
            continue
        source = source_repo / rel_path
        target = modified_repo / rel_path
        if source.exists() and source.is_file():
            ensure_dir(target.parent)
            target.write_text(source.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            restored.append(rel_path)
        elif target.exists():
            target.unlink()
            restored.append(rel_path)
    if not restored:
        return None
    return {
        "action": "restore_evaluator_files_before_repair",
        "status": "ok",
        "failed_status": risk_check.get("status"),
        "restored_files": restored,
        "reason": "evaluation_code_changed risk must be repaired without carrying evaluator edits forward",
    }


def _mechanism_self_review(draft: dict[str, Any], implementation_contract: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    review_cfg = ((_code_patch_config(config).get("validation", {}) or {}).get("mechanism_self_review") or {})
    if not isinstance(review_cfg, dict) or not review_cfg.get("enabled", True):
        return {"status": "ok", "skipped": True, "reason": "mechanism self-review disabled", "mechanism_evidence_map": []}
    strict = _strict_patch_gate(config)
    changed_files = list(draft.get("changed_files") or [])
    diff_text = str(draft.get("diff") or "")
    mechanism_contract = implementation_contract.get("mechanism_contract") if isinstance(implementation_contract.get("mechanism_contract"), dict) else {}
    implementation_targets = implementation_contract.get("implementation_targets") if isinstance(implementation_contract.get("implementation_targets"), dict) else {}
    expected_files = _expected_file_list(implementation_targets.get("expected_files"))
    ablation_switch = str(mechanism_contract.get("ablation_switch") or "").strip()
    coverage_required = _truthy_required(mechanism_contract.get("coverage_diagnostics"))
    matched_required = _truthy_required(mechanism_contract.get("matched_coverage_ablation"))
    evidence_map = []
    for path in changed_files:
        evidence_map.append(
            {
                "path": path,
                "mechanism_component": _infer_mechanism_component(path, diff_text, ablation_switch=ablation_switch),
                "evidence": _file_diff_evidence(path, diff_text, ablation_switch=ablation_switch),
            }
        )
    core_files = [path for path in changed_files if _core_mechanism_path(path)]
    expected_hits = [path for path in changed_files if path in expected_files]
    ablation_wired = bool(ablation_switch and _diff_mentions(diff_text, ablation_switch))
    coverage_evidence = _diff_mentions_any(diff_text, ["coverage", "diagnostic", "accepted_span", "acceptance_rate", "pathology"])
    matched_coverage_evidence = _diff_mentions_any(diff_text, ["matched_coverage", "coverage_delta", "control_mode"])
    evaluator_like_files = [path for path in changed_files if _evaluator_like_path(path)]
    diff_line_count = len(diff_text.splitlines())
    issues: list[str] = []
    warnings: list[str] = []
    soft_issues: list[str] = []
    if review_cfg.get("require_mechanism_evidence", True) and not (core_files or expected_hits) and strict:
        issues.append("missing_core_mechanism_file")
    elif review_cfg.get("require_mechanism_evidence", True) and not (core_files or expected_hits):
        soft_issues.append("missing_core_mechanism_file")
    if ablation_switch and not ablation_wired:
        if strict and review_cfg.get("require_ablation_wired", False):
            issues.append("ablation_switch_not_wired")
        else:
            soft_issues.append("ablation_switch_not_wired")
    if coverage_required and not coverage_evidence:
        if strict and review_cfg.get("require_coverage_evidence", False):
            issues.append("missing_coverage_diagnostics_evidence")
        else:
            soft_issues.append("missing_coverage_diagnostics_evidence")
    if matched_required and not matched_coverage_evidence:
        if strict and review_cfg.get("require_matched_coverage_evidence", False):
            issues.append("missing_matched_coverage_evidence")
        else:
            soft_issues.append("missing_matched_coverage_evidence")
    if review_cfg.get("block_evaluator_like_files", True) and evaluator_like_files:
        issues.append("evaluator_like_file_touched")
    warn_large = int(review_cfg.get("warn_large_diff_lines") or 0)
    if warn_large and diff_line_count > warn_large:
        warnings.append("large_diff")
    warnings.extend(soft_issues)
    status = "ok" if not issues else "mechanism_self_review_failed"
    quality_repair = _quality_repair_advice(soft_issues, implementation_contract)
    return {
        "status": status,
        "gate_mode": code_patch_gate_mode(config),
        "blocking": bool(issues),
        "reason": "; ".join(issues),
        "issues": issues,
        "soft_issues": soft_issues,
        "warnings": warnings,
        "mode": str(review_cfg.get("mode") or "runnable_first"),
        "quality_repair": quality_repair,
        "mechanism_evidence_map": evidence_map,
        "score_inputs": {
            "core_mechanism_files": core_files,
            "expected_file_hits": expected_hits,
            "ablation_wired": ablation_wired,
            "coverage_evidence": coverage_evidence,
            "matched_coverage_evidence": matched_coverage_evidence,
            "evaluator_like_files": evaluator_like_files,
            "diff_line_count": diff_line_count,
            "changed_file_count": len(changed_files),
        },
        "repair_hint": (
            "Add a real mechanism implementation in model/train/recipe paths and avoid evaluator-like files. "
            "Ablation and diagnostic gaps are tracked as soft quality repair unless explicitly configured as hard gates."
            if issues
            else ""
        ),
    }


def _truthy_required(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("required", True))
    return bool(value)


def _quality_repair_advice(soft_issues: list[str], implementation_contract: dict[str, Any]) -> dict[str, Any]:
    if not soft_issues:
        return {"needed": False, "issues": [], "mode": "none"}
    mechanism_contract = implementation_contract.get("mechanism_contract") if isinstance(implementation_contract.get("mechanism_contract"), dict) else {}
    return {
        "needed": False,
        "deferred": True,
        "mode": "paperization_after_effect",
        "issues": list(dict.fromkeys(soft_issues)),
        "constraints": [
            "Do not spend S2.5 discovery repair budget on these issues before an effect is found.",
            "After a full S3 win, add instrumentation, ablation wiring, diagnostics, or matched-control bookkeeping in a separate paperization stage.",
            "Paperization must not change default enabled scoring, routing, loss weights, data sampling, recipe hyperparameters, evaluator, or metric computation.",
        ],
        "ablation_switch": mechanism_contract.get("ablation_switch"),
    }


def _core_mechanism_path(path: str) -> bool:
    return path.startswith(("rosetta/model/", "script/train/", "recipe/")) and Path(path).suffix in {".py", ".json", ".yaml", ".yml", ".toml"}


def _evaluator_like_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = Path(normalized).name.lower()
    return (
        normalized.startswith("script/evaluation/")
        or normalized == "script/auto_research/activation_forward_probe.py"
        or normalized == "rosetta/utils/evaluate.py"
        or name in {"unified_evaluator.py", "evaluate.py", "evaluator.py", "activation_forward_probe.py"}
    )


def _infer_mechanism_component(path: str, diff_text: str, *, ablation_switch: str) -> str:
    file_diff = _diff_for_file(diff_text, path).lower()
    if _evaluator_like_path(path):
        return "evaluator_risk"
    if path.startswith("test/") or path.startswith("tests/"):
        return "focused_test"
    if path.startswith("recipe/"):
        return "recipe_activation"
    if ablation_switch and ablation_switch.lower() in file_diff:
        return "ablation_wiring"
    if any(token in file_diff for token in ["coverage", "diagnostic", "matched_coverage", "pathology"]):
        return "diagnostics"
    if path.startswith("script/train/"):
        return "train_integration"
    if path.startswith("rosetta/model/"):
        return "model_mechanism"
    return "supporting_change"


def _file_diff_evidence(path: str, diff_text: str, *, ablation_switch: str) -> list[str]:
    file_diff = _diff_for_file(diff_text, path)
    lowered = file_diff.lower()
    evidence = []
    if ablation_switch and ablation_switch.lower() in lowered:
        evidence.append("ablation_switch")
    for label, terms in {
        "coverage_diagnostics": ["coverage", "diagnostic", "accepted_span", "acceptance_rate"],
        "matched_coverage": ["matched_coverage", "coverage_delta", "control_mode"],
        "pathology_stats": ["pathology", "sample_family", "bucket"],
        "train_path": ["loss", "train", "forward"],
    }.items():
        if any(term in lowered for term in terms):
            evidence.append(label)
    if not evidence and _core_mechanism_path(path):
        evidence.append("core_mechanism_path")
    return list(dict.fromkeys(evidence))


def _diff_mentions(diff_text: str, needle: str) -> bool:
    return bool(needle) and needle.lower() in diff_text.lower()


def _diff_mentions_any(diff_text: str, needles: list[str]) -> bool:
    lowered = diff_text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _diff_for_file(diff_text: str, path: str) -> str:
    chunks = []
    current: list[str] = []
    in_target = False
    target_headers = {f"+++ b/{path}", f"--- a/{path}"}
    for line in diff_text.splitlines():
        if line.startswith("--- a/"):
            if in_target and current:
                chunks.extend(current)
            current = [line]
            in_target = line in target_headers
            continue
        if line.startswith("+++ b/"):
            current.append(line)
            in_target = in_target or line in target_headers
            continue
        if current:
            current.append(line)
    if in_target and current:
        chunks.extend(current)
    return "\n".join(chunks)


def _text_file_map(repo_root: Path, policy: DynamicEditPolicy) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root).as_posix()
        if _path_ignored_for_patch_delta(rel):
            continue
        if not policy.allowed(rel, repo_root=repo_root):
            continue
        try:
            result[rel] = sha256_file(path)
        except OSError:
            continue
    return result


def _unauthorized_delta_paths(source_repo: Path, modified_repo: Path, policy: DynamicEditPolicy) -> list[str]:
    source_all = _all_file_hashes(source_repo)
    modified_all = _all_file_hashes(modified_repo)
    changed = [
        rel
        for rel in sorted(set(source_all) | set(modified_all))
        if source_all.get(rel) != modified_all.get(rel) and not _path_ignored_for_patch_delta(rel)
    ]
    unauthorized = []
    for rel in changed:
        if policy.allowed(rel, repo_root=modified_repo):
            continue
        if rel in source_all and rel not in modified_all and _path_ignored_for_patch_delta(rel):
            continue
        unauthorized.append(rel)
    return unauthorized


def _path_ignored_for_patch_delta(rel_path: str) -> bool:
    path = Path(rel_path)
    if not path.parts:
        return False
    ignored_names = {
        ".git",
        "wandb",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
    }
    if any(part in ignored_names for part in path.parts):
        return True
    if path.name in {".coverage", "coverage.xml"}:
        return True
    if len(path.parts) >= 2 and path.parts[0] == "local" and path.parts[1] in {"checkpoints", "snapshots", "final_results"}:
        return True
    if _c2c_generated_run_artifact_path(path):
        return True
    return path.suffix in {".pyc", ".pyo", ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".parquet", ".arrow"}


def _c2c_generated_run_artifact_path(path: Path) -> bool:
    parts = path.parts
    if len(parts) < 3 or parts[0] != "local" or parts[1] != "auto_research_runs":
        return False
    return True


def _all_file_hashes(repo_root: Path) -> dict[str, str]:
    result = {}
    for path in repo_root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(repo_root).as_posix()
            if _path_ignored_for_patch_delta(rel):
                continue
            try:
                result[rel] = sha256_file(path)
            except OSError:
                continue
    return result


def _unified_diff(old_text: str, new_text: str, rel_path: str) -> list[str]:
    return list(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
    )


def _validate_patch_repo(
    repo_root: Path,
    changed_files: list[str],
    env_python: str,
    config: dict[str, Any],
    *,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation_config = _code_patch_config(config).get("validation", {})
    python_cmd = env_python if env_python and Path(env_python).exists() else sys.executable
    checks = []
    status = "ok"
    if validation_config.get("require_py_compile", True):
        for rel_path in changed_files:
            if Path(rel_path).suffix != ".py":
                continue
            result = _run_validation_command(
                [python_cmd, "-m", "py_compile", rel_path],
                cwd=repo_root,
                timeout=120,
            )
            check = {
                "name": f"py_compile:{rel_path}",
                "returncode": result.returncode,
                "stdout": _validation_output_excerpt(result.stdout),
                "stderr": _validation_output_excerpt(result.stderr),
            }
            if result.timed_out:
                check.update(
                    {
                        "failure_category": "py_compile_timeout",
                        "timeout_seconds": result.timeout_seconds,
                        "repair_hint": f"Make {rel_path} import/compile quickly; remove import-time heavyweight model, dataset, or GPU work.",
                    }
                )
            checks.append(check)
            if result.returncode != 0:
                status = "validation_failed"
    if status == "ok" and validation_config.get("require_targeted_tests", True):
        for test_path in _targeted_tests(repo_root, changed_files):
            result = _run_validation_command(
                [python_cmd, "-m", "pytest", "-q", "--tb=short", "-x", test_path],
                cwd=repo_root,
                timeout=180,
            )
            check = {
                "name": f"pytest:{test_path}",
                "returncode": result.returncode,
                "stdout": _validation_output_excerpt(result.stdout),
                "stderr": _validation_output_excerpt(result.stderr),
            }
            if result.timed_out:
                check.update(
                    {
                        "failure_category": "pytest_timeout",
                        "timeout_seconds": result.timeout_seconds,
                        "repair_hint": (
                            f"Make focused test {test_path} finish quickly; remove hangs, import-time heavyweight work, "
                            "network/model downloads, or tests that execute full training/evaluation."
                        ),
                    }
                )
            checks.append(check)
            if result.returncode != 0:
                status = "validation_failed"
    if status == "ok":
        smoke_check = _runtime_smoke_check(repo_root, python_cmd, config, candidate or {})
        if smoke_check:
            checks.append(smoke_check)
            if smoke_check.get("returncode") not in {0, None}:
                status = "validation_failed"
    if status == "ok":
        activation_wiring_check = _mechanism_activation_wiring_smoke_check(repo_root, python_cmd, config, candidate or {})
        if activation_wiring_check:
            checks.append(activation_wiring_check)
            if activation_wiring_check.get("returncode") not in {0, None} and activation_wiring_check.get("blocking", True):
                status = "validation_failed"
    if status == "ok":
        activation_forward_check = _mechanism_activation_forward_probe_check(repo_root, python_cmd, config, candidate or {})
        if activation_forward_check:
            checks.append(activation_forward_check)
            if activation_forward_check.get("returncode") not in {0, None} and activation_forward_check.get("blocking", True):
                status = "validation_failed"
    return {"status": status, "checks": checks, "changed_files": changed_files}


@dataclass
class ValidationCommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    timeout_seconds: int | None = None


def _run_validation_command(command: list[str], *, cwd: Path, timeout: int) -> ValidationCommandResult:
    try:
        env = None
        if command and Path(str(command[1] if len(command) > 1 else "")).name == "c2c_activation_forward_probe.py":
            env = dict(os.environ)
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(cwd) + ((os.pathsep + existing_pythonpath) if existing_pythonpath else "")
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_timeout_output(exc.stdout)
        stderr_parts = [_decode_timeout_output(exc.stderr)]
        stderr_parts.append(f"Command timed out after {timeout}s")
        return ValidationCommandResult(
            returncode=124,
            stdout=stdout,
            stderr="\n".join(part for part in stderr_parts if part).strip(),
            timed_out=True,
            timeout_seconds=timeout,
        )
    return ValidationCommandResult(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def _validation_output_excerpt(output: str, *, max_chars: int = 4000) -> str:
    if len(output) <= max_chars:
        return output
    head_size = max_chars // 2
    tail_size = max_chars - head_size
    return (
        output[:head_size]
        + "\n...[validation output truncated; preserving head and tail]...\n"
        + output[-tail_size:]
    )


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _targeted_tests(repo_root: Path, changed_files: list[str]) -> list[str]:
    candidates = []
    default = repo_root / "test" / "test_aligner_span_overlap.py"
    if default.exists():
        candidates.append(default.relative_to(repo_root).as_posix())
    for rel_path in changed_files:
        path = Path(rel_path)
        if path.parts and path.parts[0] in {"test", "tests"} and path.suffix == ".py":
            candidates.append(rel_path)
    return sorted(set(candidates))


def _runtime_smoke_check(repo_root: Path, python_cmd: str, config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    smoke_cfg = ((_code_patch_config(config).get("validation", {}) or {}).get("runtime_smoke") or {})
    if not isinstance(smoke_cfg, dict) or not smoke_cfg.get("enabled", True):
        return None
    train_entry = repo_root / "script" / "train" / "SFT_train.py"
    if not train_entry.exists():
        if smoke_cfg.get("skip_if_missing_train_entry", True):
            return {
                "name": "runtime_smoke:first_batch_train",
                "returncode": 0,
                "status": "skipped",
                "reason": "script/train/SFT_train.py not found in temporary C2C repo",
            }
        return {
            "name": "runtime_smoke:first_batch_train",
            "returncode": 1,
            "status": "failed",
            "failure_category": "missing_train_entry",
            "stderr": "script/train/SFT_train.py not found in temporary C2C repo",
        }

    with tempfile.TemporaryDirectory(prefix="auto_research_runtime_smoke_") as tmp:
        smoke_repo = Path(tmp) / "repo"
        try:
            shutil.copytree(repo_root, smoke_repo, ignore=_temporary_patch_ignore)
        except Exception as exc:
            return {
                "name": "runtime_smoke:first_batch_train",
                "returncode": 1,
                "status": "failed",
                "failure_category": "runtime_smoke_materialization_failed",
                "stderr": _validation_output_excerpt(f"{type(exc).__name__}: {exc}"),
                "repair_hint": "Fix the patch so C2C candidate train config materialization succeeds before S3.",
            }

        timeout_seconds = max(1, int(smoke_cfg.get("timeout_seconds") or 600))
        attempts: list[dict[str, Any]] = []
        last_check: dict[str, Any] | None = None
        attempt_queue = _runtime_smoke_gpu_attempts(config, smoke_cfg)
        attempt_index = 0
        while attempt_index < len(attempt_queue):
            attempt_plan = attempt_queue[attempt_index]
            attempt_index += 1
            selected_gpu_ids = list(attempt_plan.get("selected_gpu_ids") or [])
            if attempt_plan.get("resource_unavailable"):
                return {
                    "name": "runtime_smoke:first_batch_train",
                    "returncode": 75,
                    "status": "resource_retry",
                    "failure_category": "runtime_smoke_resource_retry",
                    "retryable": True,
                    "resource_retry": True,
                    "stderr": (
                        "Runtime smoke did not start because no allowed GPU reached "
                        f"min_free_mb={attempt_plan.get('min_free_mb')} before resource_wait timeout."
                    ),
                    "attempts": attempts,
                    "selected_gpu_ids": selected_gpu_ids,
                    "gpu_selection": {
                        "selected_gpu_ids": selected_gpu_ids,
                        "reason": attempt_plan.get("reason"),
                        "memory_free_mb": attempt_plan.get("memory_free_mb"),
                        "memory_total_mb": attempt_plan.get("memory_total_mb"),
                        "min_free_mb": attempt_plan.get("min_free_mb"),
                        "snapshot": attempt_plan.get("snapshot") or [],
                        "resource_wait": attempt_plan.get("resource_wait") or {},
                    },
                    "train_samples": _runtime_smoke_train_samples(smoke_cfg),
                    "repair_hint": _runtime_smoke_repair_hint("runtime_smoke_resource_retry"),
                }
            try:
                adapter = C2CAdapter(
                    smoke_repo,
                    _runtime_smoke_adapter_config(
                        config,
                        smoke_repo,
                        python_cmd,
                        smoke_cfg,
                        selected_gpu_ids=selected_gpu_ids,
                    ),
                )
                run_spec = adapter.materialize_candidate_configs(
                    candidate or {},
                    _runtime_smoke_gpu_selection(smoke_cfg, selected_gpu_ids=selected_gpu_ids, snapshot=attempt_plan.get("snapshot") or []),
                )
                _harden_runtime_smoke_train_config(run_spec.get("train_config"), smoke_cfg)
            except Exception as exc:
                return {
                    "name": "runtime_smoke:first_batch_train",
                    "returncode": 1,
                    "status": "failed",
                    "failure_category": "runtime_smoke_materialization_failed",
                    "stderr": _validation_output_excerpt(f"{type(exc).__name__}: {exc}"),
                    "attempts": attempts,
                    "selected_gpu_ids": selected_gpu_ids,
                    "repair_hint": "Fix the patch so C2C candidate train config materialization succeeds before S3.",
                }

            commands = run_spec.get("commands") or {}
            command = commands.get("train")
            if not command:
                return {
                    "name": "runtime_smoke:first_batch_train",
                    "returncode": 1,
                    "status": "failed",
                    "failure_category": "missing_train_command",
                    "stderr": "C2C runtime smoke could not build a train command.",
                    "attempts": attempts,
                    "selected_gpu_ids": selected_gpu_ids,
                }

            command = _disable_wandb_for_command(command)
            try:
                result = subprocess.run(
                    command,
                    cwd=smoke_repo,
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = _validation_output_excerpt(_decode_timeout_output(exc.stdout))
                stderr = _validation_output_excerpt((_decode_timeout_output(exc.stderr) + f"\nCommand timed out after {timeout_seconds}s").strip())
                return {
                    "name": "runtime_smoke:first_batch_train",
                    "returncode": 124,
                    "status": "failed",
                    "failure_category": "runtime_smoke_timeout",
                    "stdout": stdout,
                    "stderr": stderr,
                    "command": _validation_output_excerpt(command, max_chars=1200),
                    "attempts": attempts,
                    "selected_gpu_ids": selected_gpu_ids,
                    "timeout_seconds": timeout_seconds,
                    "repair_hint": _runtime_smoke_repair_hint("runtime_smoke_timeout"),
                }

            stdout = _validation_output_excerpt(result.stdout)
            stderr = _validation_output_excerpt(result.stderr)
            category = _runtime_smoke_failure_category(stdout, stderr, result.returncode)
            attempt_record = {
                "attempt": len(attempts) + 1,
                "selected_gpu_ids": selected_gpu_ids,
                "gpu_free_mb": attempt_plan.get("memory_free_mb"),
                "gpu_total_mb": attempt_plan.get("memory_total_mb"),
                "returncode": result.returncode,
                "failure_category": category,
                "command": _validation_output_excerpt(command, max_chars=800),
                "stdout_tail": str(stdout or "")[-1200:],
                "stderr_tail": str(stderr or "")[-1200:],
            }
            attempts.append(attempt_record)
            last_check = {
                "name": "runtime_smoke:first_batch_train",
                "returncode": result.returncode,
                "status": "ok" if result.returncode == 0 else "failed",
                "failure_category": category,
                "stdout": stdout,
                "stderr": stderr,
                "command": _validation_output_excerpt(command, max_chars=1200),
                "train_samples": _runtime_smoke_train_samples(smoke_cfg),
                "selected_gpu_ids": selected_gpu_ids,
                "gpu_selection": {
                    "selected_gpu_ids": selected_gpu_ids,
                    "reason": attempt_plan.get("reason"),
                    "memory_free_mb": attempt_plan.get("memory_free_mb"),
                    "memory_total_mb": attempt_plan.get("memory_total_mb"),
                    "min_free_mb": attempt_plan.get("min_free_mb"),
                    "snapshot": attempt_plan.get("snapshot") or [],
                    "resource_wait": attempt_plan.get("resource_wait") or {},
                },
                "attempts": attempts,
                "repair_hint": _runtime_smoke_repair_hint(category),
            }
            if result.returncode == 0:
                return last_check
            if category != "runtime_smoke_oom":
                return last_check
            if _runtime_smoke_oom_retry_enabled(smoke_cfg) and _runtime_smoke_oom_count(attempts) <= _runtime_smoke_oom_max_retries(smoke_cfg):
                retry_plan = _runtime_smoke_oom_retry_attempt(
                    config,
                    smoke_cfg,
                    tried_gpu_ids={gpu for attempt in attempts for gpu in attempt.get("selected_gpu_ids") or []},
                )
            else:
                retry_plan = None
            if retry_plan:
                attempt_queue.append(retry_plan)
        if last_check:
            if last_check.get("failure_category") == "runtime_smoke_oom":
                last_check["failure_category"] = "runtime_smoke_resource_retry"
                last_check["retryable"] = True
                last_check["resource_retry"] = True
                last_check["status"] = "resource_retry"
                last_check["repair_hint"] = _runtime_smoke_repair_hint("runtime_smoke_resource_retry")
            return last_check
    return None


def _mechanism_activation_wiring_smoke_check(repo_root: Path, python_cmd: str, config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    smoke_cfg = ((_code_patch_config(config).get("validation", {}) or {}).get("runtime_smoke") or {})
    if not isinstance(smoke_cfg, dict) or not smoke_cfg.get("enabled", True):
        return None
    wiring_cfg = smoke_cfg.get("mechanism_activation")
    if wiring_cfg is None:
        wiring_cfg = {}
    if not isinstance(wiring_cfg, dict) or not wiring_cfg.get("enabled", True):
        return None
    hard_gate = _activation_wiring_hard_gate(config)
    switch = _candidate_ablation_switch_for_patch(candidate)
    if not switch:
        return {
            "name": "runtime_smoke:mechanism_activation_wiring",
            "returncode": 1,
            "status": "failed",
            "blocking": hard_gate,
            "soft_failure": not hard_gate,
            "failure_category": "missing_ablation_switch",
            "stderr": "candidate experiment_contract does not declare an ablation_switch",
            "repair_hint": "Add experiment_contract.ablation_switch and wire it to disable only the proposed mechanism.",
        }

    with tempfile.TemporaryDirectory(prefix="auto_research_activation_wiring_smoke_") as tmp:
        smoke_repo = Path(tmp) / "repo"
        try:
            shutil.copytree(repo_root, smoke_repo, ignore=_temporary_patch_ignore)
            adapter = C2CAdapter(smoke_repo, _runtime_smoke_adapter_config(config, smoke_repo, python_cmd, smoke_cfg))
            run_spec = adapter.materialize_candidate_configs(candidate or {}, _runtime_smoke_gpu_selection(smoke_cfg))
            activation_spec = _materialize_runtime_activation_wiring_configs(
                run_spec,
                switch=switch,
                wiring_cfg=wiring_cfg,
            )
        except Exception as exc:
            return {
                "name": "runtime_smoke:mechanism_activation_wiring",
                "returncode": 1,
                "status": "failed",
                "blocking": hard_gate,
                "soft_failure": not hard_gate,
                "failure_category": "mechanism_activation_materialization_failed",
                "stderr": _validation_output_excerpt(f"{type(exc).__name__}: {exc}"),
                "repair_hint": "Fix candidate config materialization so enabled and ablation-disabled eval configs can be built before S3.",
            }

        evidence = _mechanism_activation_wiring_evidence(
            smoke_repo,
            candidate,
            run_spec,
            activation_spec,
            switch=switch,
            wiring_cfg=wiring_cfg,
        )
    failures = list(evidence.get("failures") or [])
    return {
        "name": "runtime_smoke:mechanism_activation_wiring",
        "returncode": 0 if not failures else 1,
        "status": "ok" if not failures else "failed",
        "blocking": bool(failures and hard_gate),
        "soft_failure": bool(failures and not hard_gate),
        "gate_mode": code_patch_gate_mode(config),
        "failure_category": "" if not failures else "mechanism_activation_wiring_failed",
        "stdout": _validation_output_excerpt(json.dumps(evidence, ensure_ascii=False, indent=2)),
        "stderr": "" if not failures else "; ".join(failures),
        "switch": switch,
        "enabled_eval_configs": evidence.get("enabled_eval_configs") or {},
        "disabled_eval_configs": evidence.get("disabled_eval_configs") or {},
        "rosetta_config": evidence.get("rosetta_config") or {},
        "runtime_code_refs": evidence.get("runtime_code_refs") or {},
        "repair_hint": _mechanism_activation_wiring_repair_hint(failures),
    }


def _mechanism_activation_forward_probe_check(repo_root: Path, python_cmd: str, config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    smoke_cfg = ((_code_patch_config(config).get("validation", {}) or {}).get("runtime_smoke") or {})
    if not isinstance(smoke_cfg, dict) or not smoke_cfg.get("enabled", True):
        return None
    activation_cfg = smoke_cfg.get("mechanism_activation")
    if activation_cfg is None:
        activation_cfg = {}
    if not isinstance(activation_cfg, dict) or not activation_cfg.get("enabled", True):
        return None
    probe_cfg = activation_cfg.get("forward_probe")
    if probe_cfg is None:
        probe_cfg = {}
    if not isinstance(probe_cfg, dict) or not probe_cfg.get("enabled", True):
        return None
    hard_gate = bool(probe_cfg.get("hard_gate", activation_cfg.get("hard_gate", False)))
    switch = _candidate_ablation_switch_for_patch(candidate)
    if not switch:
        return {
            "name": "runtime_smoke:mechanism_activation_forward_probe",
            "returncode": 1,
            "status": "failed",
            "blocking": hard_gate,
            "soft_failure": not hard_gate,
            "failure_category": "missing_ablation_switch",
            "stderr": "candidate experiment_contract does not declare an ablation_switch",
            "repair_hint": _mechanism_activation_forward_probe_repair_hint(["missing_ablation_switch"]),
        }
    script_rel = str(probe_cfg.get("script") or "script/auto_research/activation_forward_probe.py")
    script_path = repo_root / script_rel
    builtin_probe = Path(__file__).resolve().parent / "probes" / "c2c_activation_forward_probe.py"
    use_builtin_probe = bool(not script_path.exists() and probe_cfg.get("builtin_fallback", True) and builtin_probe.exists())
    if not script_path.exists() and not use_builtin_probe:
        return {
            "name": "runtime_smoke:mechanism_activation_forward_probe",
            "returncode": 0,
            "status": "skipped",
            "blocking": False,
            "soft_failure": False,
            "reason": f"forward activation probe script not found: {script_rel}",
            "expected_script": script_rel,
            "repair_hint": "Add a lightweight activation_forward_probe.py to compare enabled/disabled mechanism tensors before S3.",
        }

    with tempfile.TemporaryDirectory(prefix="auto_research_activation_forward_probe_") as tmp:
        smoke_repo = Path(tmp) / "repo"
        output_path = Path(tmp) / "activation_forward_probe.json"
        probe_environment: dict[str, Any] = {}
        try:
            shutil.copytree(repo_root, smoke_repo, ignore=_temporary_patch_ignore)
            adapter = C2CAdapter(smoke_repo, _runtime_smoke_adapter_config(config, smoke_repo, python_cmd, smoke_cfg))
            run_spec = adapter.materialize_candidate_configs(candidate or {}, _runtime_smoke_gpu_selection(smoke_cfg))
            activation_spec = _materialize_runtime_activation_wiring_configs(
                run_spec,
                switch=switch,
                wiring_cfg=activation_cfg,
            )
            enabled_eval = next(iter(((activation_spec.get("enabled_eval_configs") or {}) or {}).values()), "")
            disabled_eval = next(iter((activation_spec.get("eval_configs") or {}).values()), "")
        except Exception as exc:
            return {
                "name": "runtime_smoke:mechanism_activation_forward_probe",
                "returncode": 1,
                "status": "failed",
                "blocking": hard_gate,
                "soft_failure": not hard_gate,
                "failure_category": "mechanism_activation_forward_probe_materialization_failed",
                "stderr": _validation_output_excerpt(f"{type(exc).__name__}: {exc}"),
                "repair_hint": _mechanism_activation_forward_probe_repair_hint(["materialization_failed"]),
            }

        probe_script = smoke_repo / script_rel if not use_builtin_probe else builtin_probe
        probe_environment = _forward_probe_environment_preflight(
            smoke_repo,
            python_cmd,
            use_builtin_probe=use_builtin_probe,
        )
        command = [
            python_cmd,
            _probe_script_command_arg(probe_script, smoke_repo),
            "--enabled-config",
            str(Path(enabled_eval).relative_to(smoke_repo) if Path(enabled_eval).is_absolute() and _is_relative_to(Path(enabled_eval), smoke_repo) else enabled_eval),
            "--disabled-config",
            str(Path(disabled_eval).relative_to(smoke_repo) if Path(disabled_eval).is_absolute() and _is_relative_to(Path(disabled_eval), smoke_repo) else disabled_eval),
            "--switch",
            switch,
            "--output",
            str(output_path),
        ]
        timeout_seconds = max(1, int(probe_cfg.get("timeout_seconds") or 180))
        result = _run_validation_command(command, cwd=smoke_repo, timeout=timeout_seconds)
        probe_payload = read_json(output_path, default={}) if output_path.exists() else {}

    evidence = _normalize_activation_forward_probe_payload(probe_payload, min_changed_fields=int(probe_cfg.get("min_changed_fields") or 1))
    failures = list(evidence.get("failures") or [])
    if result.returncode != 0:
        failures.append("forward_probe_command_failed")
    return {
        "name": "runtime_smoke:mechanism_activation_forward_probe",
        "returncode": 0 if not failures else 1,
        "status": "ok" if not failures else "failed",
        "blocking": bool(failures and hard_gate),
        "soft_failure": bool(failures and not hard_gate),
        "failure_category": "" if not failures else "mechanism_activation_forward_probe_failed",
        "stdout": _validation_output_excerpt(result.stdout),
        "stderr": _validation_output_excerpt("; ".join(failures) or result.stderr),
        "command": _validation_output_excerpt(" ".join(str(part) for part in command), max_chars=1200),
        "switch": switch,
        "probe_source": "builtin" if use_builtin_probe else "repo",
        "probe_script": str(builtin_probe if use_builtin_probe else script_rel),
        "probe_environment": probe_environment,
        "probe": evidence,
        "repair_hint": _mechanism_activation_forward_probe_repair_hint(failures),
    }


def _forward_probe_environment_preflight(repo_root: Path, python_cmd: str, *, use_builtin_probe: bool) -> dict[str, Any]:
    payload = {
        "probe_python": str(python_cmd),
        "using_c2c_env_python": _using_target_env_python(python_cmd),
        "torch_available": False,
        "torch_version": None,
        "repo_import_ok": False,
        "repo_import_error": "",
        "returncode": None,
    }
    script = (
        "import importlib, json, sys\n"
        "payload = {\n"
        "  'executable': sys.executable,\n"
        "  'torch_available': False,\n"
        "  'torch_version': None,\n"
        "  'repo_import_ok': False,\n"
        "  'repo_import_error': '',\n"
        "}\n"
        "try:\n"
        "    torch = importlib.import_module('torch')\n"
        "    payload['torch_available'] = True\n"
        "    payload['torch_version'] = getattr(torch, '__version__', None)\n"
        "except Exception as exc:\n"
        "    payload['torch_error'] = type(exc).__name__ + ': ' + str(exc)\n"
        "try:\n"
        "    importlib.import_module('rosetta.model.projector')\n"
        "    payload['repo_import_ok'] = True\n"
        "except Exception as exc:\n"
        "    payload['repo_import_error'] = type(exc).__name__ + ': ' + str(exc)\n"
        "print(json.dumps(payload, ensure_ascii=True))\n"
    )
    try:
        result = subprocess.run(
            [python_cmd, "-c", script],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            env=_pythonpath_env_for_repo(repo_root),
        )
    except Exception as exc:
        payload["repo_import_error"] = f"{type(exc).__name__}: {exc}"
        return payload
    payload["returncode"] = result.returncode
    stdout = (result.stdout or "").strip().splitlines()
    if stdout:
        try:
            observed = json.loads(stdout[-1])
        except json.JSONDecodeError:
            observed = {}
        if isinstance(observed, dict):
            payload["probe_python"] = str(observed.get("executable") or payload["probe_python"])
            payload["torch_available"] = bool(observed.get("torch_available"))
            payload["torch_version"] = observed.get("torch_version")
            payload["repo_import_ok"] = bool(observed.get("repo_import_ok"))
            payload["repo_import_error"] = str(observed.get("repo_import_error") or observed.get("torch_error") or "")
    if result.stderr and not payload.get("repo_import_error"):
        payload["stderr_tail"] = result.stderr[-1000:]
    payload["using_builtin_probe"] = bool(use_builtin_probe)
    return payload


def _pythonpath_env_for_repo(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + ((os.pathsep + existing_pythonpath) if existing_pythonpath else "")
    return env


def _using_target_env_python(python_cmd: str) -> bool:
    try:
        resolved = Path(python_cmd).resolve()
    except OSError:
        return str(python_cmd) != sys.executable
    try:
        current = Path(sys.executable).resolve()
    except OSError:
        current = Path(sys.executable)
    return resolved != current


def _normalize_activation_forward_probe_payload(payload: Any, *, min_changed_fields: int) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {
            "status": "missing_output",
            "mechanism_observed": False,
            "changed_fields": [],
            "failures": ["missing_forward_probe_output"],
        }
    changed_fields = [str(item) for item in payload.get("changed_fields") or payload.get("changed_tensors") or [] if item]
    compared_fields = [str(item) for item in payload.get("compared_fields") or payload.get("compared_tensors") or [] if item]
    unchanged_fields = [str(item) for item in payload.get("unchanged_fields") or payload.get("unchanged_tensors") or [] if item]
    if "mechanism_observed" in payload:
        mechanism_observed = bool(payload.get("mechanism_observed"))
    else:
        mechanism_observed = bool(len(changed_fields) >= max(1, min_changed_fields))
    failures = [str(item) for item in payload.get("failures") or [] if item]
    if not compared_fields and not changed_fields:
        failures.append("no_forward_probe_fields_compared")
    if not mechanism_observed:
        failures.append("enabled_disabled_forward_outputs_identical")
    return {
        "status": "changed" if mechanism_observed else "unchanged",
        "mechanism_observed": mechanism_observed,
        "changed_fields": changed_fields[:20],
        "unchanged_fields": unchanged_fields[:20],
        "compared_fields": compared_fields[:40],
        "tensor_checks": payload.get("tensor_checks") if isinstance(payload.get("tensor_checks"), list) else [],
        "projector_tensor_checks": payload.get("projector_tensor_checks") if isinstance(payload.get("projector_tensor_checks"), list) else [],
        "wrapper_probe": payload.get("wrapper_probe") if isinstance(payload.get("wrapper_probe"), dict) else {},
        "cache_key_diff": payload.get("cache_key_diff"),
        "cache_value_diff": payload.get("cache_value_diff"),
        "projector_called": payload.get("projector_called"),
        "switch_seen_by_forward": payload.get("switch_seen_by_forward"),
        "enabled": payload.get("enabled") if isinstance(payload.get("enabled"), dict) else {},
        "disabled": payload.get("disabled") if isinstance(payload.get("disabled"), dict) else {},
        "probe_type": payload.get("probe_type"),
        "fallback_reason": payload.get("fallback_reason"),
        "forward_probe_error": payload.get("forward_probe_error") if isinstance(payload.get("forward_probe_error"), dict) else {},
        "projector_spec": payload.get("projector_spec") if isinstance(payload.get("projector_spec"), dict) else {},
        "forward_probe_projector_spec": payload.get("forward_probe_projector_spec") if isinstance(payload.get("forward_probe_projector_spec"), dict) else {},
        "static_trace": payload.get("static_trace") if isinstance(payload.get("static_trace"), dict) else {},
        "trace": payload.get("trace") if isinstance(payload.get("trace"), dict) else {},
        "raw_status": payload.get("status"),
        "failures": list(dict.fromkeys(failures)),
    }


def _mechanism_activation_wiring_evidence(
    repo_root: Path,
    candidate: dict[str, Any],
    run_spec: dict[str, Any],
    activation_spec: dict[str, Any],
    *,
    switch: str,
    wiring_cfg: dict[str, Any],
) -> dict[str, Any]:
    enabled_eval_configs = {
        str(dataset): str(path)
        for dataset, path in (
            activation_spec.get("enabled_eval_configs")
            or (run_spec.get("proxy_screen") or {}).get("eval_configs")
            or run_spec.get("eval_configs")
            or {}
        ).items()
    }
    disabled_eval_configs = {
        str(dataset): str(path)
        for dataset, path in (activation_spec.get("eval_configs") or {}).items()
    }
    enabled_rosetta = _first_rosetta_config(enabled_eval_configs.values())
    disabled_rosetta = _first_rosetta_config(disabled_eval_configs.values())
    config_keys = sorted(
        key
        for key in _candidate_rosetta_activation_keys(candidate)
        if key and key != "checkpoints_dir"
    )
    runtime_refs = _runtime_code_reference_evidence(
        repo_root,
        switch=switch,
        config_keys=config_keys,
        files=wiring_cfg.get("runtime_code_files"),
    )
    failures: list[str] = []
    if activation_spec.get("status") != "materialized":
        failures.append(f"activation smoke disabled config was not materialized: {activation_spec.get('reason') or activation_spec.get('status')}")
    if wiring_cfg.get("require_switch_in_disabled_eval_config", True):
        if disabled_rosetta.get(switch) is not True:
            failures.append(f"disabled eval rosetta_config does not set {switch}=true")
    if enabled_rosetta.get(switch) is True:
        failures.append(f"enabled proxy eval rosetta_config already sets {switch}=true")
    missing_enabled_keys = [
        key
        for key in config_keys
        if key not in enabled_rosetta and key not in disabled_rosetta
    ]
    if config_keys and len(missing_enabled_keys) == len(config_keys):
        failures.append("candidate config_overrides did not appear in enabled/disabled eval rosetta_config")
    if wiring_cfg.get("require_switch_referenced_in_runtime_code", True) and not runtime_refs.get("switch_refs"):
        if not runtime_refs.get("config_key_refs"):
            failures.append(f"no runtime model file references ablation switch {switch} or activated config keys")
        else:
            failures.append(f"runtime model files reference config keys but not ablation switch {switch}")
    if (
        wiring_cfg.get("require_switch_referenced_in_runtime_code", True)
        and runtime_refs.get("switch_refs")
        and not runtime_refs.get("forward_switch_refs")
    ):
        failures.append(f"runtime model files mention {switch} but no forward function reads it")
    return {
        "switch": switch,
        "enabled_eval_configs": enabled_eval_configs,
        "disabled_eval_configs": disabled_eval_configs,
        "rosetta_config": {
            "enabled_keys": sorted(enabled_rosetta.keys()),
            "disabled_keys": sorted(disabled_rosetta.keys()),
            "disabled_switch_value": disabled_rosetta.get(switch),
            "enabled_switch_value": enabled_rosetta.get(switch),
            "activated_config_keys": config_keys,
            "missing_enabled_or_disabled_config_keys": missing_enabled_keys,
        },
        "runtime_code_refs": runtime_refs,
        "failures": failures,
    }


def _materialize_runtime_activation_wiring_configs(
    run_spec: dict[str, Any],
    *,
    switch: str,
    wiring_cfg: dict[str, Any],
) -> dict[str, Any]:
    enabled_eval_configs = (run_spec.get("proxy_screen") or {}).get("eval_configs") or run_spec.get("eval_configs") or {}
    if not isinstance(enabled_eval_configs, dict) or not enabled_eval_configs:
        return {"enabled": True, "status": "failed", "reason": "enabled eval configs missing for mechanism activation wiring smoke"}
    run_root = Path(run_spec["run_root"])
    smoke_root = run_root / "runtime_activation_wiring_disabled"
    ensure_dir(smoke_root)
    run_root_rel = f"local/auto_research_runs/{run_spec['run_id']}"
    max_datasets = max(1, int(wiring_cfg.get("max_datasets") or 1))
    selected = list(enabled_eval_configs.items())[:max_datasets]
    disabled_eval_configs: dict[str, Path] = {}
    for dataset, enabled_path_value in selected:
        enabled_path = Path(enabled_path_value)
        payload = read_yaml(enabled_path, default={}) if enabled_path.suffix in {".yaml", ".yml"} else read_json(enabled_path, default={})
        if not isinstance(payload, dict):
            payload = {}
        if not isinstance(payload.get("model"), dict):
            payload["model"] = {}
        payload["model"].setdefault("rosetta_config", {})
        if not isinstance(payload["model"]["rosetta_config"], dict):
            payload["model"]["rosetta_config"] = {}
        payload["model"]["rosetta_config"][switch] = True
        payload.setdefault("output", {})
        if isinstance(payload["output"], dict):
            payload["output"]["output_dir"] = f"{run_root_rel}/runtime_activation_wiring_disabled/results/{dataset}"
        eval_path = smoke_root / f"eval_{dataset}.yaml"
        write_yaml(eval_path, payload)
        disabled_eval_configs[str(dataset)] = eval_path
    return {
        "enabled": True,
        "status": "materialized",
        "switch": switch,
        "enabled_eval_configs": {str(dataset): Path(path) for dataset, path in enabled_eval_configs.items()},
        "eval_configs": disabled_eval_configs,
        "run_root": smoke_root,
    }


def _first_rosetta_config(paths: Any) -> dict[str, Any]:
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            continue
        payload = read_yaml(path, default={}) if path.suffix in {".yaml", ".yml"} else read_json(path, default={})
        if not isinstance(payload, dict):
            continue
        rosetta = ((payload.get("model") or {}).get("rosetta_config") or {})
        return dict(rosetta) if isinstance(rosetta, dict) else {}
    return {}


def _runtime_code_reference_evidence(repo_root: Path, *, switch: str, config_keys: list[str], files: Any) -> dict[str, Any]:
    if not files:
        files = ["rosetta/model/aligner.py", "rosetta/model/projector.py", "rosetta/model/wrapper.py"]
    file_refs: dict[str, dict[str, Any]] = {}
    switch_refs: list[str] = []
    forward_functions: list[str] = []
    forward_switch_refs: list[str] = []
    config_key_refs: list[dict[str, str]] = []
    forward_config_key_refs: list[dict[str, str]] = []
    for rel_path in _expected_file_list(files):
        path = repo_root / rel_path
        if not path.exists() or not path.is_file():
            file_refs[rel_path] = {"exists": False, "switch_ref": False, "forward_switch_ref": False, "config_key_refs": []}
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        file_switch_ref = bool(switch and switch in text)
        file_config_refs = [key for key in config_keys if key and key in text]
        forward_refs = _forward_reference_evidence(text, switch=switch, config_keys=config_keys)
        file_refs[rel_path] = {
            "exists": True,
            "switch_ref": file_switch_ref,
            "forward_functions": forward_refs.get("forward_functions") or [],
            "forward_switch_ref": bool(forward_refs.get("switch_ref")),
            "forward_config_key_refs": forward_refs.get("config_key_refs") or [],
            "config_key_refs": file_config_refs,
        }
        if file_switch_ref:
            switch_refs.append(rel_path)
        if forward_refs.get("forward_functions"):
            forward_functions.extend(f"{rel_path}:{name}" for name in forward_refs.get("forward_functions") or [])
        if forward_refs.get("switch_ref"):
            forward_switch_refs.append(rel_path)
        for key in file_config_refs:
            config_key_refs.append({"path": rel_path, "key": key})
        for key in forward_refs.get("config_key_refs") or []:
            forward_config_key_refs.append({"path": rel_path, "key": str(key)})
    return {
        "switch_refs": switch_refs,
        "forward_functions": forward_functions,
        "forward_switch_refs": forward_switch_refs,
        "config_key_refs": config_key_refs,
        "forward_config_key_refs": forward_config_key_refs,
        "files": file_refs,
    }


def _forward_reference_evidence(text: str, *, switch: str, config_keys: list[str]) -> dict[str, Any]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {"forward_functions": [], "switch_ref": False, "config_key_refs": []}
    forward_functions: list[str] = []
    switch_ref = False
    config_refs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "forward":
            continue
        forward_functions.append(node.name)
        segment = ast.get_source_segment(text, node) or ""
        if switch and switch in segment:
            switch_ref = True
        config_refs.update(key for key in config_keys if key and key in segment)
    return {
        "forward_functions": forward_functions,
        "switch_ref": switch_ref,
        "config_key_refs": sorted(config_refs),
    }


def _candidate_rosetta_activation_keys(candidate: dict[str, Any]) -> set[str]:
    overrides = c2c_candidate_config_overrides(candidate)
    eval_override = overrides.get("eval") if isinstance(overrides, dict) else {}
    rosetta = (((eval_override or {}).get("model") or {}).get("rosetta_config") or {}) if isinstance(eval_override, dict) else {}
    if not isinstance(rosetta, dict):
        return set()
    keys = _flatten_config_keys(rosetta)
    return {key for key in keys if key not in {"model", "eval", "train", "rosetta_config", "checkpoints_dir"}}


def _candidate_ablation_switch_for_patch(candidate: dict[str, Any]) -> str:
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    ablation_plan = candidate.get("ablation_plan") if isinstance(candidate.get("ablation_plan"), dict) else {}
    switch = contract.get("ablation_switch") or ablation_plan.get("switch")
    return str(switch).strip() if switch not in (None, "") else ""


def _mechanism_activation_wiring_repair_hint(failures: list[str]) -> str:
    if not failures:
        return ""
    return (
        "Repair eval-path activation before S3: ensure experiment_contract.config_overrides reaches eval model.rosetta_config, "
        "ensure the disabled activation-smoke eval config sets the ablation switch, and make wrapper/projector/aligner forward code "
        "read that switch to bypass only the proposed mechanism."
    )


def _mechanism_activation_forward_probe_repair_hint(failures: list[str]) -> str:
    if not failures:
        return ""
    return (
        "Repair forward-level mechanism activation before S3: use the ablation switch inside the actual wrapper/projector/aligner forward path "
        "so enabled and disabled configs change a concrete tensor, routing score, cache weight, projector output, or wrapper projected cache on the same small batch."
    )


def _probe_script_command_arg(probe_script: Path, smoke_repo: Path) -> str:
    if _is_relative_to(probe_script, smoke_repo):
        return str(probe_script.relative_to(smoke_repo))
    return str(probe_script)


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _disable_wandb_for_command(command: str) -> str:
    prefix = (
        "WANDB_DISABLED=true "
        "WANDB_MODE=disabled "
        "WANDB_START_METHOD=thread "
        "WANDB_REQUIRE_SERVICE=false "
    )
    return f"{prefix}{command}"


def _runtime_smoke_adapter_config(
    config: dict[str, Any],
    repo_root: Path,
    python_cmd: str,
    smoke_cfg: dict[str, Any],
    *,
    selected_gpu_ids: list[int] | None = None,
) -> dict[str, Any]:
    adapter_config = copy.deepcopy(config)
    c2c_cfg = adapter_config.setdefault("c2c", {})
    c2c_cfg["snapshot_path"] = str(repo_root)
    c2c_cfg["env_python"] = python_cmd
    small_loop = c2c_cfg.setdefault("small_loop", {})
    small_loop["train_samples"] = _runtime_smoke_train_samples(smoke_cfg)
    small_loop["eval_limit"] = int(smoke_cfg.get("eval_limit") or 1)
    small_loop["eval_datasets"] = list(smoke_cfg.get("eval_datasets") or small_loop.get("eval_datasets") or ["mmlu-redux"])[:1]
    if selected_gpu_ids is not None:
        small_loop["gpu_ids"] = list(selected_gpu_ids)
    else:
        small_loop["gpu_ids"] = smoke_cfg.get("gpu_ids", small_loop.get("gpu_ids", "auto"))
    small_loop["num_train_processes"] = int(smoke_cfg.get("num_train_processes") or 1)
    small_loop["strict_dataset_cache"] = bool(smoke_cfg.get("strict_dataset_cache", False))
    small_loop.setdefault("proxy_screen", {"enabled": False})
    if isinstance(small_loop.get("proxy_screen"), dict):
        small_loop["proxy_screen"]["enabled"] = False
    return adapter_config


def _runtime_smoke_train_samples(smoke_cfg: dict[str, Any]) -> int:
    return max(2, int(smoke_cfg.get("train_samples") or 8))


def _harden_runtime_smoke_train_config(train_config_path: Any, smoke_cfg: dict[str, Any]) -> None:
    if not train_config_path:
        return
    path = Path(train_config_path)
    if not path.exists():
        return
    payload = read_json(path, default={}) or {}
    if not isinstance(payload, dict):
        return
    payload.setdefault("data", {})
    if isinstance(payload["data"], dict):
        payload["data"]["train_ratio"] = float(smoke_cfg.get("train_ratio") or 0.75)
        payload["data"].setdefault("kwargs", {})
        if isinstance(payload["data"]["kwargs"], dict):
            payload["data"]["kwargs"]["num_samples"] = _runtime_smoke_train_samples(smoke_cfg)
    training = payload.setdefault("training", {})
    if isinstance(training, dict):
        training["num_epochs"] = int(smoke_cfg.get("num_epochs") or 1)
        training["per_device_train_batch_size"] = int(smoke_cfg.get("per_device_train_batch_size") or 1)
        training["gradient_accumulation_steps"] = int(smoke_cfg.get("gradient_accumulation_steps") or 1)
    output = payload.setdefault("output", {})
    if isinstance(output, dict):
        output["save_steps"] = int(smoke_cfg.get("save_steps") or 1_000_000)
        output["eval_steps"] = int(smoke_cfg.get("eval_steps") or 1_000_000)
        wandb_config = output.setdefault("wandb_config", {})
        if isinstance(wandb_config, dict):
            wandb_config["mode"] = "disabled"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _runtime_smoke_gpu_selection(
    smoke_cfg: dict[str, Any],
    *,
    selected_gpu_ids: list[int] | None = None,
    snapshot: list[dict[str, Any]] | None = None,
) -> GpuSelection:
    selected = list(selected_gpu_ids if selected_gpu_ids is not None else _coerce_runtime_smoke_gpu_ids(smoke_cfg.get("gpu_ids", "auto")))
    policy = {
        "source": "s2_5_runtime_smoke",
        "gpu_ids": smoke_cfg.get("gpu_ids", "auto"),
        "min_free_mb": _runtime_smoke_min_free_mb(smoke_cfg),
        "max_gpus": 1,
    }
    reason = "runtime_smoke_auto_selected_by_free_memory" if str(smoke_cfg.get("gpu_ids", "auto")) == "auto" else "configured_runtime_smoke_gpu_ids"
    return GpuSelection(selected_ids=selected, policy=policy, snapshot=list(snapshot or []), reason=reason)


def _runtime_smoke_gpu_attempts(config: dict[str, Any], smoke_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    gpu_policy = {
        "gpu_ids": smoke_cfg.get("gpu_ids", "auto"),
        "min_free_mb": _runtime_smoke_min_free_mb(smoke_cfg),
        "max_gpus": 1,
    }
    selection = ExperimentRunner(config).select_gpus(gpu_policy)
    snapshot = selection.snapshot or []
    allowed_ids = _runtime_smoke_allowed_gpu_ids(smoke_cfg, snapshot)
    candidates = [
        item
        for item in snapshot
        if int(item.get("index", -1)) in allowed_ids and int(item.get("memory_free_mb") or 0) >= _runtime_smoke_min_free_mb(smoke_cfg)
    ]
    candidates.sort(key=lambda item: (-int(item.get("memory_free_mb") or 0), int(item.get("utilization_gpu") or 100), int(item.get("index") or 999)))
    if not candidates and _runtime_smoke_resource_wait_enabled(smoke_cfg):
        wait = _wait_for_runtime_smoke_gpu(config, smoke_cfg, allowed_ids=allowed_ids)
        snapshot = wait.get("snapshot") or snapshot
        candidates = [
            item
            for item in snapshot
            if int(item.get("index", -1)) in allowed_ids and int(item.get("memory_free_mb") or 0) >= _runtime_smoke_min_free_mb(smoke_cfg)
        ]
        candidates.sort(key=lambda item: (-int(item.get("memory_free_mb") or 0), int(item.get("utilization_gpu") or 100), int(item.get("index") or 999)))
        for item in candidates:
            item["resource_wait"] = wait
        if not candidates and snapshot:
            fallback = sorted(
                [item for item in snapshot if int(item.get("index", -1)) in allowed_ids],
                key=lambda item: (-int(item.get("memory_free_mb") or 0), int(item.get("index") or 999)),
            )
            selected_snapshot = fallback[0] if fallback else {}
            selected = int(selected_snapshot.get("index", -1)) if selected_snapshot else None
            return [
                {
                    "selected_gpu_ids": [selected] if selected is not None and selected >= 0 else [],
                    "reason": "runtime_smoke_resource_wait_timeout",
                    "memory_free_mb": selected_snapshot.get("memory_free_mb"),
                    "memory_total_mb": selected_snapshot.get("memory_total_mb"),
                    "min_free_mb": _runtime_smoke_min_free_mb(smoke_cfg),
                    "snapshot": snapshot,
                    "resource_wait": wait,
                    "resource_unavailable": True,
                }
            ]
    if not candidates and snapshot:
        fallback = sorted(
            [item for item in snapshot if int(item.get("index", -1)) in allowed_ids],
            key=lambda item: (-int(item.get("memory_free_mb") or 0), int(item.get("index") or 999)),
        )
        selected_snapshot = fallback[0] if fallback else {}
        selected = int(selected_snapshot.get("index", -1)) if selected_snapshot else None
        return [
            {
                "selected_gpu_ids": [selected] if selected is not None and selected >= 0 else [],
                "reason": "runtime_smoke_resource_unavailable",
                "memory_free_mb": selected_snapshot.get("memory_free_mb"),
                "memory_total_mb": selected_snapshot.get("memory_total_mb"),
                "min_free_mb": _runtime_smoke_min_free_mb(smoke_cfg),
                "snapshot": snapshot,
                "resource_wait": {
                    "status": "timeout" if _runtime_smoke_resource_wait_enabled(smoke_cfg) else "disabled",
                    "enabled": _runtime_smoke_resource_wait_enabled(smoke_cfg),
                    "min_free_mb": _runtime_smoke_min_free_mb(smoke_cfg),
                },
                "resource_unavailable": True,
            }
        ]
    if not candidates:
        return [
            {
                "selected_gpu_ids": _coerce_runtime_smoke_gpu_ids(smoke_cfg.get("gpu_ids", "auto"))[:1],
                "reason": "runtime_smoke_no_gpu_snapshot",
                "memory_free_mb": None,
                "memory_total_mb": None,
                "min_free_mb": _runtime_smoke_min_free_mb(smoke_cfg),
                "snapshot": snapshot,
                "resource_wait": {"status": "not_ready", "enabled": _runtime_smoke_resource_wait_enabled(smoke_cfg)},
            }
        ]
    return [
        {
            "selected_gpu_ids": [int(item["index"])],
            "reason": "runtime_smoke_auto_selected_by_free_memory",
            "memory_free_mb": item.get("memory_free_mb"),
            "memory_total_mb": item.get("memory_total_mb"),
            "min_free_mb": _runtime_smoke_min_free_mb(smoke_cfg),
            "snapshot": snapshot,
            "resource_wait": item.get("resource_wait") or {"status": "ready", "enabled": _runtime_smoke_resource_wait_enabled(smoke_cfg)},
        }
        for item in candidates[:1]
    ]


def _runtime_smoke_allowed_gpu_ids(smoke_cfg: dict[str, Any], snapshot: list[dict[str, Any]]) -> set[int]:
    explicit = _coerce_runtime_smoke_gpu_ids(smoke_cfg.get("gpu_ids", "auto"))
    if explicit:
        return {int(item) for item in explicit}
    return {int(item.get("index")) for item in snapshot if "index" in item}


def _runtime_smoke_min_free_mb(smoke_cfg: dict[str, Any]) -> int:
    return max(0, int(smoke_cfg.get("min_free_mb") or 8192))


def _runtime_smoke_resource_wait_enabled(smoke_cfg: dict[str, Any]) -> bool:
    wait_cfg = smoke_cfg.get("resource_wait") if isinstance(smoke_cfg.get("resource_wait"), dict) else {}
    return bool(wait_cfg.get("enabled", True))


def _runtime_smoke_oom_retry_enabled(smoke_cfg: dict[str, Any]) -> bool:
    retry_cfg = smoke_cfg.get("oom_retry") if isinstance(smoke_cfg.get("oom_retry"), dict) else {}
    return bool(retry_cfg.get("enabled", True))


def _runtime_smoke_oom_max_retries(smoke_cfg: dict[str, Any]) -> int:
    retry_cfg = smoke_cfg.get("oom_retry") if isinstance(smoke_cfg.get("oom_retry"), dict) else {}
    return max(0, int(retry_cfg.get("max_retries") or 1))


def _runtime_smoke_oom_count(attempts: list[dict[str, Any]]) -> int:
    return sum(1 for attempt in attempts if attempt.get("failure_category") == "runtime_smoke_oom")


def _runtime_smoke_oom_retry_attempt(config: dict[str, Any], smoke_cfg: dict[str, Any], *, tried_gpu_ids: set[int]) -> dict[str, Any] | None:
    min_free_mb = _runtime_smoke_min_free_mb(smoke_cfg)
    selection = ExperimentRunner(config).select_gpus({"gpu_ids": smoke_cfg.get("gpu_ids", "auto"), "min_free_mb": min_free_mb, "max_gpus": 1})
    snapshot = selection.snapshot or []
    allowed_ids = _runtime_smoke_allowed_gpu_ids(smoke_cfg, snapshot)
    candidates = [
        item
        for item in snapshot
        if int(item.get("index", -1)) in allowed_ids
        and int(item.get("index", -1)) not in tried_gpu_ids
        and int(item.get("memory_free_mb") or 0) >= min_free_mb
    ]
    candidates.sort(key=lambda item: (-int(item.get("memory_free_mb") or 0), int(item.get("utilization_gpu") or 100), int(item.get("index") or 999)))
    wait: dict[str, Any] | None = None
    if not candidates and _runtime_smoke_resource_wait_enabled(smoke_cfg):
        wait = _wait_for_runtime_smoke_gpu(config, smoke_cfg, allowed_ids=allowed_ids - set(tried_gpu_ids))
        snapshot = wait.get("snapshot") or snapshot
        candidates = [
            item
            for item in snapshot
            if int(item.get("index", -1)) in allowed_ids
            and int(item.get("index", -1)) not in tried_gpu_ids
            and int(item.get("memory_free_mb") or 0) >= min_free_mb
        ]
        candidates.sort(key=lambda item: (-int(item.get("memory_free_mb") or 0), int(item.get("utilization_gpu") or 100), int(item.get("index") or 999)))
    if not candidates:
        return None
    item = candidates[0]
    return {
        "selected_gpu_ids": [int(item["index"])],
        "reason": "runtime_smoke_oom_retry_auto_selected_by_free_memory",
        "memory_free_mb": item.get("memory_free_mb"),
        "memory_total_mb": item.get("memory_total_mb"),
        "min_free_mb": min_free_mb,
        "snapshot": snapshot,
        "resource_wait": wait or {"status": "ready", "enabled": _runtime_smoke_resource_wait_enabled(smoke_cfg)},
    }


def _wait_for_runtime_smoke_gpu(config: dict[str, Any], smoke_cfg: dict[str, Any], *, allowed_ids: set[int]) -> dict[str, Any]:
    if not allowed_ids:
        return {
            "status": "no_allowed_gpu",
            "enabled": True,
            "polls": 0,
            "waited_seconds": 0,
            "min_free_mb": _runtime_smoke_min_free_mb(smoke_cfg),
            "snapshot": [],
        }
    wait_cfg = smoke_cfg.get("resource_wait") if isinstance(smoke_cfg.get("resource_wait"), dict) else {}
    timeout_seconds = max(0, int(wait_cfg.get("timeout_seconds") or 0))
    poll_seconds = max(1, int(wait_cfg.get("poll_seconds") or 60))
    min_free_mb = _runtime_smoke_min_free_mb(smoke_cfg)
    started = time.monotonic()
    polls = 0
    last_snapshot: list[dict[str, Any]] = []
    while time.monotonic() - started <= timeout_seconds:
        polls += 1
        snapshot = ExperimentRunner._gpu_snapshot()
        last_snapshot = snapshot
        ready = [
            item
            for item in snapshot
            if int(item.get("index", -1)) in allowed_ids and int(item.get("memory_free_mb") or 0) >= min_free_mb
        ]
        if ready:
            return {
                "status": "ready",
                "enabled": True,
                "polls": polls,
                "waited_seconds": round(time.monotonic() - started, 3),
                "min_free_mb": min_free_mb,
                "snapshot": snapshot,
            }
        if timeout_seconds == 0:
            break
        time.sleep(min(poll_seconds, max(0.0, timeout_seconds - (time.monotonic() - started))))
    return {
        "status": "timeout",
        "enabled": True,
        "polls": polls,
        "waited_seconds": round(time.monotonic() - started, 3),
        "min_free_mb": min_free_mb,
        "snapshot": last_snapshot,
    }


def _coerce_runtime_smoke_gpu_ids(value: Any) -> list[int]:
    if value in (None, "", "auto"):
        return []
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    try:
        return [int(item) for item in value]
    except TypeError:
        return []


def _runtime_smoke_failure_category(stdout: str, stderr: str, returncode: int) -> str:
    if returncode == 0:
        return ""
    text = f"{stdout}\n{stderr}".lower()
    if "valid_mask" in text:
        return "valid_mask_runtime_error"
    if "expected scalar type" in text or "bfloat16" in text or "float16" in text or "dtype" in text:
        return "dtype_mismatch"
    if "cuda" in text and ("device" in text or "cpu" in text or "same device" in text):
        return "device_mismatch"
    if "out of memory" in text or "cuda error: out of memory" in text:
        return "runtime_smoke_oom"
    if "no such file" in text or "not found" in text:
        return "runtime_dependency_missing"
    if "traceback" in text or "runtimeerror" in text:
        return "first_batch_runtime_error"
    return "first_batch_failed"


def _runtime_smoke_repair_hint(category: str) -> str:
    hints = {
        "valid_mask_runtime_error": "Define and thread valid_mask through the new mechanism for both train and eval paths; add a fallback when masks are absent.",
        "dtype_mismatch": "Cast new tensors/modules to the hidden-state dtype before arithmetic, especially Float/BFloat16 paths.",
        "device_mismatch": "Create new tensors on the same device as the hidden states or source tensors; avoid default CPU tensors in forward.",
        "runtime_smoke_oom": (
            "Treat this as resource-sensitive first: check whether the configured runtime-smoke GPU is already occupied "
            "and retry on a freer GPU when possible. If memory is still insufficient on an idle GPU, shrink temporary "
            "tensors and avoid full-cache materialization in the first-batch path."
        ),
        "runtime_smoke_resource_retry": (
            "No code repair should be attempted yet: runtime smoke could not obtain a GPU with enough free memory. "
            "Wait for a GPU that satisfies runtime_smoke.min_free_mb, then resume S2.5 validation."
        ),
        "runtime_dependency_missing": "Use existing C2C dependencies and paths; do not introduce unavailable runtime files.",
        "runtime_smoke_timeout": "Reduce first-batch work and remove blocking full-dataset/full-cache work from the patched train path.",
        "first_batch_runtime_error": "Fix the patched forward/train path so a one-sample first batch can complete before proxy train.",
    }
    return hints.get(category, "Fix the patched C2C runtime path so S2.5 first-batch smoke passes before S3.")


def _build_implementation_contract(
    candidate: dict[str, Any],
    plan: dict[str, Any],
    edit_policy: DynamicEditPolicy,
    *,
    previous_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    config_overrides = c2c_candidate_config_overrides(candidate)
    novelty_report = c2c_idea_novelty_report(candidate)
    selected_plan = {
        "acceptance_criteria": plan.get("acceptance_criteria", {}),
        "metrics": plan.get("metrics", []),
        "datasets": plan.get("datasets", []),
        "reviewer_risk_controls": plan.get("reviewer_risk_controls", {}),
    }
    expected_files = contract.get("expected_files") or candidate.get("expected_files")
    editable_expected_files = _editable_expected_files(expected_files, edit_policy)
    blocked_expected_files = _blocked_expected_files(expected_files, edit_policy)
    ablation_plan = candidate.get("ablation_plan") if isinstance(candidate.get("ablation_plan"), dict) else {}
    ablation_switch = contract.get("ablation_switch") or ablation_plan.get("switch")
    coverage_diagnostics = candidate.get("coverage_diagnostics") or contract.get("coverage_diagnostics") or {}
    matched_coverage_ablation = candidate.get("matched_coverage_ablation") or contract.get("matched_coverage_ablation") or {}
    implementation_plan = candidate.get("implementation_plan") if isinstance(candidate.get("implementation_plan"), dict) else contract.get("implementation_plan")
    if not isinstance(implementation_plan, dict):
        implementation_plan = {}
    s2_variant = candidate.get("s2_variant") if isinstance(candidate.get("s2_variant"), dict) else {}
    implementation_scope = candidate.get("implementation_scope") or contract.get("implementation_scope") or implementation_plan.get("scope") or "bounded"
    scope_requirements = _scope_requirements(str(implementation_scope), implementation_plan)
    proxy_effect_repair_contract = (
        previous_failure.get("proxy_effect_repair_contract")
        if isinstance(previous_failure, dict) and isinstance(previous_failure.get("proxy_effect_repair_contract"), dict)
        else {}
    )
    return {
        "candidate_id": candidate.get("id"),
        "title": candidate.get("title"),
        "variant_fingerprint": candidate.get("variant_fingerprint") or s2_variant.get("variant_fingerprint"),
        "s2_variant": _compact_value(s2_variant, max_chars=1600),
        "hypothesis": candidate.get("hypothesis"),
        "mechanism": _first_present(
            candidate,
            ["mechanism", "method", "implementation", "one_line_conclusion", "decision_rationale"],
        ),
        "mechanism_contract": {
            "mechanism_type": candidate.get("mechanism_type") or contract.get("mechanism_type") or novelty_report.get("mechanism_type"),
            "mechanism_summary": _compact_value(candidate.get("mechanism_summary") or candidate.get("description"), max_chars=600),
            "paper_claim": _compact_value(candidate.get("paper_claim"), max_chars=500),
            "why_baseline_fails": _compact_value(candidate.get("why_baseline_fails"), max_chars=500),
            "expected_signature": _compact_value(candidate.get("expected_signature"), max_chars=900),
            "ablation_plan": _compact_value(ablation_plan, max_chars=900),
            "ablation_switch": ablation_switch,
            "coverage_diagnostics": _compact_value(coverage_diagnostics, max_chars=900),
            "matched_coverage_ablation": _compact_value(matched_coverage_ablation, max_chars=900),
            "novelty_gate": novelty_report,
        },
        "implementation_scope": {
            "scope": implementation_scope,
            "required_new_files": _compact_value(candidate.get("required_new_files") or implementation_plan.get("required_new_files") or [], max_chars=1000),
            "integration_points": _compact_value(candidate.get("integration_points") or implementation_plan.get("integration_points") or [], max_chars=1400),
            "smoke_tests": _compact_value(candidate.get("smoke_tests") or implementation_plan.get("smoke_tests") or [], max_chars=600),
            "decomposition_plan": _compact_value(candidate.get("decomposition_plan") or implementation_plan.get("decomposition_plan") or [], max_chars=1600),
            "mvp_slice": implementation_plan.get("mvp_slice") or candidate.get("mvp_slice") or "",
            "allowed_first_patch_files": _compact_value(implementation_plan.get("allowed_first_patch_files") or [], max_chars=1200),
        },
        "decision_chain": _brief_decision_chain(candidate.get("decision_chain")),
        "evidence_summary": {
            "supporting": _brief_refs(candidate.get("evidence_refs") or candidate.get("evidence"), limit=3),
            "counterevidence": _brief_refs(candidate.get("counterevidence") or candidate.get("risks"), limit=3),
            "failure_feedback_refs": _brief_refs(candidate.get("failure_feedback_refs"), limit=2),
        },
        "implementation_targets": {
            "expected_files": _compact_value(editable_expected_files, max_chars=1000),
            "blocked_expected_files": _compact_value(blocked_expected_files, max_chars=1000),
            "code_refs": _brief_refs(candidate.get("code_refs") or candidate.get("code_targets"), limit=5),
            "allowed_prefixes": edit_policy.include_prefixes,
            "allowed_extensions": edit_policy.include_extensions,
            "forbidden_prefixes": edit_policy.exclude_prefixes,
            "forbidden_extensions": edit_policy.exclude_extensions,
        },
        "experiment_contract": {
            "primary_metric": contract.get("primary_metric"),
            "baseline": contract.get("baseline"),
            "config_overrides": config_overrides,
            "verification_commands": _compact_value(contract.get("verification_commands") or candidate.get("verification_commands"), max_chars=1000),
            "kill_criteria": _compact_value(candidate.get("kill_criteria") or contract.get("kill_criteria"), max_chars=1000),
            "ablation_switch": contract.get("ablation_switch"),
            "coverage_diagnostics": _compact_value(coverage_diagnostics, max_chars=900),
            "matched_coverage_ablation": _compact_value(matched_coverage_ablation, max_chars=900),
        },
        "plan_context": _compact_value(selected_plan, max_chars=2000),
        "s2_5_requirements": [
            "Implement only the selected idea, with enough coherent code to make the mechanism executable and effect-bearing in S3.",
            "Codex implementation freedom is intentional: multi-file or helper-module patches are acceptable when the mechanism genuinely requires them.",
            "Implement the mechanism-level claim in mechanism_contract; do not satisfy the task with only threshold, top-k, confidence-floor, fallback, or recipe tuning.",
            "Do not implement the idea as an added hard accept/reject gate on top of the baseline. If gating is part of the mechanism, it must be trained/estimated and paired with coverage diagnostics.",
            "Effect-first discovery: prioritize runnable code and cheap-proxy/full-S3 effect over paperization-only diagnostics.",
            "S3 eligibility is decided by execution truth: py_compile/tests/runtime smoke, config activation, ablation wiring, forward activation, evaluator safety, and cheap proxy readiness.",
            "Keep ablation_switch, coverage diagnostics, and matched-coverage support lightweight when natural; missing paperization evidence can be completed after an effect is found.",
            "Emit or preserve lightweight mechanism evidence where natural, such as accepted-span counts, utility/verifier/pathology stats, or bridge-memory stats.",
            *scope_requirements,
            "Prefer existing local abstractions and configuration style over new framework code.",
            "If you add a new configurable constructor or recipe parameter, it must be explicitly represented in the supplied config_overrides or by editing an allowed recipe file.",
            "Do not repeat prior failed patch behavior; if previous_patch_failure is present, fix the reported validation or activation issue.",
            "Honor s2_variant.variant_fingerprint and implement that selected mechanism variant, not a generic or previously tried same-direction patch.",
            "If s2_variant.integration_point or s2_variant.control_signal is provided, wire the patch around those choices unless validation proves they are impossible.",
            *_previous_proxy_effect_repair_requirements(proxy_effect_repair_contract),
            "Add focused tests for the new behavior. Do not run GPU training.",
            "Do not create run-result artifacts such as local/auto_research_runs files during S2.5; S3 materializes those outputs.",
            "Do not edit data, model weights, checkpoints, historical results, caches, or logs.",
            *_previous_quality_repair_requirements(previous_failure),
        ],
        "previous_patch_failure": previous_failure or {},
        "previous_failure": previous_failure or {},
        "proxy_effect_repair_contract": _compact_value(proxy_effect_repair_contract, max_chars=2800) if proxy_effect_repair_contract else {},
    }


def _scope_requirements(scope: str, implementation_plan: dict[str, Any]) -> list[str]:
    normalized = scope.strip().lower()
    if normalized == "large":
        mvp = implementation_plan.get("mvp_slice") or "Use decomposition_plan to choose a coherent executable slice when the full mechanism is too broad for one reliable patch."
        return [
            "This is a large-scope idea: broad cross-file implementation is allowed when needed, but do not rewrite evaluator/metric code or unrelated training infrastructure.",
            f"Use the decomposition guidance to keep the patch coherent and executable, not artificially tiny: {mvp}",
            "Keep integration points explicit and leave TODO-free, runnable baseline fallback behavior behind the ablation switch.",
        ]
    if normalized == "medium":
        return [
            "This is a medium-scope idea: new helper modules and multiple integration files are allowed when needed for a fully wired mechanism.",
            "Wire every new module through the listed integration_points and add or update the listed smoke_tests.",
        ]
    return [
        "This is a bounded-scope idea: prefer existing C2C model surfaces, but add files when needed to make the mechanism real and testable.",
    ]


def _previous_proxy_effect_repair_requirements(contract: dict[str, Any]) -> list[str]:
    if not isinstance(contract, dict) or not contract:
        return []
    if str(contract.get("mode") or "") == "s2_5_only_implementation_repair":
        signals = [str(signal) for signal in contract.get("implementation_failure_signals") or [] if signal][:6]
        selected_candidate_id = str(contract.get("selected_candidate_id") or "").strip()
        variant_fingerprint = str(contract.get("variant_fingerprint") or "").strip()
        requirements = [
            "This is S2.5-only implementation repair, not S1/S2 method planning: keep the same candidate and repair patch eligibility for S3.",
            "Do not invent a new mechanism, switch variants, rerun S2 planning, or weaken validation/proxy/readiness gates.",
            "Reuse the same persistent Codex session/worktree for this candidate when available; preserve the implementation context and fix the current patch.",
            "Use previous_failure.s2_5_repair_dispatch, activation_forward_probe_diagnostics, tensor_checks, patch_manifest, and changed_files as primary evidence.",
            "Repair the real config -> rosetta_config -> constructor/wrapper/projector/aligner forward -> tensor/output activation path before changing effect logic.",
            "If a tensor is identical enabled-vs-disabled, make the ablation switch reach the forward path and change the relevant tensor; do not edit probe/evaluator code to bypass this.",
            "The repair target is patch_eligible_for_s3=true. If this cannot be achieved after implementation repair, leave an implementation_blocked rationale instead of changing the method.",
        ]
        if selected_candidate_id:
            requirements.append(f"Same-candidate lock: continue candidate `{selected_candidate_id}`.")
        if variant_fingerprint:
            requirements.append(f"Same-variant lock: preserve variant_fingerprint `{variant_fingerprint}`.")
        if signals:
            requirements.append("Implementation failure signals to address: " + ", ".join(signals) + ".")
        return requirements
    dragging = [
        str(item.get("dataset"))
        for item in contract.get("dragging_datasets") or []
        if isinstance(item, dict) and item.get("dataset")
    ][:3]
    risk_labels = [str(label) for label in contract.get("patch_risk_labels") or []][:4]
    requirements = [
        "This is an effect-first cheap-proxy repair: keep the same idea and repair the patch so cheap proxy can pass before full S3.",
        "Use proxy_effect_repair_contract.proxy_dataset_deltas, proxy_dataset_regressions, soft_flags, and command_failure as the primary repair evidence.",
        "Do not spend this cheap-proxy repair on ablation, coverage, matched-coverage, or paperization-only diagnostics unless required to make the effect runnable.",
        "Do not edit evaluator or metric computation files, and do not weaken proxy thresholds or baseline metrics.",
    ]
    activation_smoke = contract.get("activation_smoke") if isinstance(contract.get("activation_smoke"), dict) else {}
    if activation_smoke:
        switch = activation_smoke.get("switch") or "ablation_switch"
        disabled_configs = activation_smoke.get("eval_configs") if isinstance(activation_smoke.get("eval_configs"), dict) else {}
        disabled_config_paths = ", ".join(str(path) for path in disabled_configs.values())[:500]
        requirements.extend(
            [
                f"Activation smoke failed for `{switch}`: repair eval-path activation/wiring before any mechanism retuning.",
                "Use proxy_effect_repair_contract.activation_smoke.enabled_metrics, disabled_metrics, metric_comparison, and prediction_comparison as primary evidence.",
                "Focus on rosetta_config loading, wrapper/projector/aligner forward-path wiring, train-vs-eval config parity, and ablation switch polarity.",
                "Do not respond by inventing a new idea, weakening the activation smoke threshold, or editing evaluator/metric code.",
            ]
        )
        if disabled_config_paths:
            requirements.append("Inspect the disabled activation-smoke eval config path(s): " + disabled_config_paths + ".")
    if dragging:
        requirements.append("Prioritize dragging proxy datasets before broad tuning: " + ", ".join(dragging) + ".")
    if risk_labels:
        requirements.append("Reduce patch-risk labels that hurt proxy score: " + ", ".join(risk_labels) + ".")
    return requirements


def _previous_quality_repair_requirements(previous_failure: dict[str, Any] | None) -> list[str]:
    if not isinstance(previous_failure, dict):
        return []
    quality_repair = previous_failure.get("quality_repair") or {}
    if not isinstance(quality_repair, dict) or not quality_repair.get("force_s2_5_quality_repair"):
        return []
    issues = ", ".join(str(item) for item in quality_repair.get("issues") or []) or "soft mechanism diagnostics"
    return [
        f"This is an instrumentation-only quality repair after a promising cheap proxy; address only: {issues}.",
        "Do not change the default enabled mechanism behavior, scoring/routing formula, loss weights, data sampling, or recipe hyperparameters.",
        "Only add ablation wiring, coverage diagnostics, matched-control bookkeeping, or focused tests around the existing mechanism.",
        "Preserve the previously validated enabled path so the same cheap proxy subset can be rerun without material regression.",
    ]


def _first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return ""


def _brief_decision_chain(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"summary": _compact_value(value, max_chars=1200)}
    return {
        "conclusion": _compact_value(value.get("conclusion") or value.get("decision") or value.get("summary"), max_chars=500),
        "top_evidence": _brief_refs(value.get("evidence"), limit=2),
        "top_counterevidence": _brief_refs(value.get("counterevidence"), limit=2),
    }


def _brief_refs(value: Any, *, limit: int) -> list[dict[str, Any]] | str:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return _compact_value(value, max_chars=1200)
    if not isinstance(value, list):
        return _compact_value(value, max_chars=1200)
    refs = []
    for item in value[:limit]:
        if isinstance(item, str):
            refs.append({"snippet": _compact_value(item, max_chars=260)})
            continue
        if not isinstance(item, dict):
            refs.append({"snippet": _compact_value(item, max_chars=260)})
            continue
        refs.append(
            {
                "source_type": item.get("source_type"),
                "source_path": item.get("source_path"),
                "chunk_id": item.get("chunk_id"),
                "symbol": item.get("symbol"),
                "start_line": item.get("start_line"),
                "snippet": _compact_value(item.get("snippet"), max_chars=260),
                "why_relevant": _compact_value(item.get("why_relevant"), max_chars=180),
                "failure_mode": item.get("failure_mode"),
                "dataset_regressions": item.get("dataset_regressions") or {},
                "avoid_repeat_rule": _compact_value(item.get("avoid_repeat_rule"), max_chars=180),
            }
        )
    return refs


def _compact_value(value: Any, *, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value
    if len(text) <= max_chars:
        return value
    return text[: max_chars - 32] + "\n...[truncated for S2.5 prompt]"


def _editable_expected_files(value: Any, edit_policy: DynamicEditPolicy) -> list[str]:
    return [
        path
        for path in _expected_file_list(value)
        if edit_policy.allowed(path)
    ]


def _blocked_expected_files(value: Any, edit_policy: DynamicEditPolicy) -> list[dict[str, str]]:
    blocked = []
    for path in _expected_file_list(value):
        reason = edit_policy.validate_path(path)
        if reason:
            blocked.append({"path": path, "reason": reason})
    return blocked


def _expected_file_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and item.get("path"):
                result.append(str(item["path"]))
        return result
    if isinstance(value, dict):
        result = []
        for key in ("files", "expected_files", "paths"):
            result.extend(_expected_file_list(value.get(key)))
        if value.get("path"):
            result.append(str(value["path"]))
        return result
    return []


def _implementation_contract_with_validation_feedback(
    implementation_contract: dict[str, Any],
    validation: dict[str, Any],
    draft: dict[str, Any],
    *,
    repair_packet: dict[str, Any] | None = None,
    attempt: int,
) -> dict[str, Any]:
    repair_contract = json.loads(json.dumps(implementation_contract, ensure_ascii=False))
    activation = validation.get("activation_check") or {}
    mechanism_review = validation.get("mechanism_review") or {}
    feedback_status = validation.get("status")
    if activation.get("status") not in {None, "ok"}:
        feedback_status = activation.get("status")
    elif mechanism_review.get("status") not in {None, "ok"}:
        feedback_status = mechanism_review.get("status")
    repair_contract["validation_failure_feedback"] = {
        "attempt": attempt,
        "status": feedback_status,
        "failed_checks": _failed_validation_checks(validation, limit=3),
        "activation_forward_probe_diagnostics": _forward_probe_diagnostics_from_validation(validation),
        "activation_check": activation,
        "mechanism_review": mechanism_review,
        "changed_files": draft.get("changed_files") or [],
        "instruction": (
            "Repair the existing patch in this checkout so validation passes. "
            "If activation_check reports missing config parameters, either remove/inline the unapproved public parameter, "
            "rename it to an existing activated config key, or update the patch so the parameter is activated by supplied config_overrides. "
            "Core mechanism config parameters must close the idea -> config -> constructor -> forward path before S3; do not leave them as defaults. "
            "If mechanism_review reports issues, repair the patch quality directly: add mechanism_evidence_map-worthy model/train/recipe changes, "
            "wire the ablation switch, emit coverage and matched-coverage diagnostics, and remove evaluator-like edits. "
            "If failed_checks contains runtime_smoke:first_batch_train, repair the actual C2C train/forward path before S3: "
            "fix dtype/device mismatches, valid_mask handling, and first-batch runtime exceptions instead of bypassing the smoke check. "
            "If failed_checks contains runtime_smoke:mechanism_activation_wiring, repair eval-path activation specifically: "
            "make experiment_contract.config_overrides reach eval model.rosetta_config, make the ablation-disabled eval config set the switch, "
            "and make wrapper/projector/aligner forward code read the switch to bypass only the proposed mechanism. "
            "If failed_checks contains runtime_smoke:mechanism_activation_forward_probe, repair forward-level causal activation specifically: "
            "the enabled and disabled configs must change at least one concrete tensor, routing score, cache weight, or projector output on the same small batch. "
            "Read codex_repair_packet.activation_forward_probe_diagnostics: unchanged tensor names and enabled/disabled sha pairs tell you exactly which path is no-op; "
            "repair_focus tells whether to fix config materialization, constructor parameter flow, projector/aligner forward branching, or wrapper cache injection/parameter passing. "
            "Do not edit script/auto_research/activation_forward_probe.py; it is validation code, not the mechanism. "
            "Keep the same idea, preserve allowed changes, and do not add generated run artifacts."
        ),
    }
    repair_contract["s2_5_repair_session_policy"] = _s2_5_repair_session_policy()
    if repair_packet:
        repair_contract["codex_repair_packet"] = repair_packet
    requirements = list(repair_contract.get("s2_5_requirements") or [])
    requirements.append("This is a repair retry after validation or activation failure; fix the failed checks before returning.")
    if (validation.get("mechanism_review") or {}).get("status") not in {None, "ok"}:
        requirements.append("Mechanism self-review is mandatory: the repaired patch must provide concrete mechanism evidence, ablation wiring, and required diagnostics without evaluator-like edits.")
    if (validation.get("activation_check") or {}).get("status") == "config_activation_missing":
        missing = ", ".join(str(item) for item in (validation.get("activation_check") or {}).get("blocking_missing_parameters") or (validation.get("activation_check") or {}).get("missing_parameters") or [])
        requirements.append("Config activation is mandatory: activate these core mechanism parameter(s) through experiment_contract.config_overrides or an allowed recipe, or inline/remove them for the current implementation: " + missing + ".")
    if any(str(check.get("name") or "").startswith("runtime_smoke:") for check in validation.get("checks") or [] if isinstance(check, dict)):
        requirements.append("Runtime smoke is mandatory: the repaired patch must complete a one-sample first-batch train smoke before proxy train.")
    if any(str(check.get("name") or "") == "runtime_smoke:mechanism_activation_wiring" for check in validation.get("checks") or [] if isinstance(check, dict)):
        requirements.append("Mechanism activation wiring is mandatory: enabled/disabled eval configs and model forward code must prove the ablation switch reaches the active C2C path.")
    if any(str(check.get("name") or "") == "runtime_smoke:mechanism_activation_forward_probe" for check in validation.get("checks") or [] if isinstance(check, dict)):
        requirements.append("Forward activation probe is mandatory: repair the actual wrapper/projector/aligner forward path so enabled/disabled configs change at least one reported tensor or routing field; use activation_forward_probe_diagnostics.identical_tensors sha pairs and repair_focus to choose the fix; do not edit script/auto_research/activation_forward_probe.py.")
    repair_contract["s2_5_requirements"] = requirements
    return repair_contract


def _implementation_contract_with_contract_feedback(
    implementation_contract: dict[str, Any],
    failure: dict[str, Any],
    *,
    repair_packet: dict[str, Any] | None = None,
    attempt: int,
) -> dict[str, Any]:
    repair_contract = json.loads(json.dumps(implementation_contract, ensure_ascii=False))
    repair_contract["contract_failure_feedback"] = {
        "attempt": attempt,
        "status": failure.get("status"),
        "reason": _contract_failure_reason(failure),
        "errors": failure.get("errors") or [],
        "changed_files": failure.get("changed_files") or [],
        "risk_labels": failure.get("risk_labels") or [],
        "risk_files": failure.get("risk_files") or [],
        "instruction": (
            "Repair the implementation attempt so it produces an allowed executable diff. "
            "Do not touch forbidden paths or generated run artifacts. "
            "If the prior attempt made no change, edit the declared integration point directly. "
            "If it touched unauthorized files, move the implementation into allowed model/script/recipe/test files."
            " If contract_failure_feedback.status is patch_too_broad, remove unrelated files while keeping every file required for the real mechanism and focused validation."
            " If it is proxy_risk_repair_required or risk_labels include evaluation_code_changed, fully revert evaluator edits and do not edit script/evaluation/*."
            " Move evidence emission into model/train outputs, recipe config, or focused tests instead."
        ),
    }
    repair_contract["s2_5_repair_session_policy"] = _s2_5_repair_session_policy()
    if repair_packet:
        repair_contract["codex_repair_packet"] = repair_packet
    requirements = list(repair_contract.get("s2_5_requirements") or [])
    requirements.append("This is a contract-aware repair retry; directly address contract_failure_feedback before returning.")
    if "evaluation_code_changed" in set(failure.get("risk_labels") or []) or any(
        str(item.get("path") or "").startswith("script/evaluation/") for item in failure.get("risk_files") or [] if isinstance(item, dict)
    ):
        repair_contract["forbidden_repair_files"] = ["script/evaluation/"]
        requirements.append(
            "Evaluator-touched repair is strict: restore every script/evaluation/* file to its original behavior and produce a non-empty model/train/recipe/test diff instead."
        )
    repair_contract["s2_5_requirements"] = requirements
    return repair_contract


def _contract_failure_reason(failure: dict[str, Any]) -> str:
    if failure.get("reason"):
        return str(failure.get("reason"))[-2000:]
    errors = failure.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(str(item) for item in errors[:5])[-2000:]
    if failure.get("rationale"):
        return str(failure.get("rationale"))[-2000:]
    return str(failure.get("status") or "unknown patch contract failure")


def _failed_validation_checks(validation: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    failed = []
    for check in validation.get("checks") or []:
        if check.get("returncode") == 0:
            continue
        failed.append(
            {
                "name": check.get("name"),
                "returncode": check.get("returncode"),
                "failure_category": check.get("failure_category"),
                "repair_hint": check.get("repair_hint"),
                "stdout_tail": str(check.get("stdout") or "")[-1600:],
                "stderr_tail": str(check.get("stderr") or "")[-1600:],
                "forward_probe_diagnostics": _forward_probe_diagnostics_from_check(check)
                if str(check.get("name") or "") == "runtime_smoke:mechanism_activation_forward_probe"
                else {},
            }
        )
    return failed[:limit]


def _validate_config_activation(diff_text: str, candidate: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    introduced = _introduced_python_config_parameters(diff_text)
    if not introduced:
        return {
            "status": "ok",
            "gate_mode": code_patch_gate_mode(config or {}),
            "blocking": False,
            "introduced_config_parameters": [],
            "activated_parameters": [],
            "missing_parameters": [],
            "blocking_missing_parameters": [],
            "soft_missing_parameters": [],
        }
    configured_keys = _flatten_config_keys(c2c_candidate_config_overrides(candidate))
    recipe_keys = _added_recipe_keys(diff_text)
    activated = sorted(key for key in introduced if key in configured_keys or key in recipe_keys)
    missing = sorted(key for key in introduced if key not in set(activated))
    if missing:
        blocking_missing = sorted(key for key in missing if _config_activation_blocks_parameter(key, config))
        soft_missing = sorted(key for key in missing if key not in set(blocking_missing))
        blocking = bool(blocking_missing)
        return {
            "status": "config_activation_missing" if blocking else "ok",
            "gate_mode": code_patch_gate_mode(config or {}),
            "blocking": blocking,
            "soft_issues": [] if blocking else ["unactivated_config_parameter"],
            "reason": (
                "Patch introduced configurable parameter(s) that are not explicitly activated "
                f"by experiment_contract.config_overrides or an allowed recipe edit: {', '.join(blocking_missing or missing)}"
            ),
            "introduced_config_parameters": sorted(introduced),
            "activated_parameters": activated,
            "missing_parameters": missing,
            "blocking_missing_parameters": blocking_missing,
            "soft_missing_parameters": soft_missing,
            "configured_keys": sorted(configured_keys),
            "recipe_keys": sorted(recipe_keys),
            "repair_hint": (
                "Wire the core mechanism config parameter through experiment_contract.config_overrides or an allowed recipe, "
                "or inline/remove it for the current implementation so S3 cannot silently run the default path."
                if blocking
                else ""
            ),
        }
    return {
        "status": "ok",
        "gate_mode": code_patch_gate_mode(config or {}),
        "blocking": False,
        "introduced_config_parameters": sorted(introduced),
        "activated_parameters": activated,
        "missing_parameters": [],
        "blocking_missing_parameters": [],
        "soft_missing_parameters": [],
        "configured_keys": sorted(configured_keys),
        "recipe_keys": sorted(recipe_keys),
    }


def _config_activation_blocks_parameter(parameter: str, config: dict[str, Any] | None) -> bool:
    validation = (_code_patch_config(config or {}).get("validation") or {})
    setting = validation.get("require_config_activation", True)
    if setting is False:
        return _strict_patch_gate(config)
    if str(setting).strip().lower() in {"soft", "warn", "debt"}:
        return _strict_patch_gate(config)
    mode = str(setting).strip().lower()
    if mode in {"all", "true", "1", "yes"}:
        return True
    return _core_mechanism_config_parameter(parameter) or _strict_patch_gate(config)


def _core_mechanism_config_parameter(parameter: str) -> bool:
    return _looks_like_public_config_key(parameter)


def _introduced_python_config_parameters(diff_text: str) -> set[str]:
    introduced: set[str] = set()
    current_file = ""
    in_signature = False
    signature_depth = 0
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/")
            in_signature = False
            signature_depth = 0
            continue
        if not current_file.endswith(".py"):
            continue
        if not line.startswith(("+", " ")):
            continue
        if line.startswith("+++"):
            continue
        is_added = line.startswith("+")
        text = line[1:]
        stripped = text.strip()
        if stripped.startswith(("def ", "async def ")):
            in_signature = True
            signature_depth = stripped.count("(") - stripped.count(")")
            if is_added:
                introduced.update(_public_config_parameters_from_signature_line(stripped))
            if signature_depth <= 0:
                in_signature = False
            continue
        if not in_signature:
            continue
        if is_added:
            introduced.update(_public_config_parameters_from_signature_line(stripped))
        signature_depth += stripped.count("(") - stripped.count(")")
        if signature_depth <= 0:
            in_signature = False
    return introduced


def _public_config_parameters_from_signature_line(stripped: str) -> set[str]:
    introduced: set[str] = set()
    if stripped.startswith(("#", "*", "**")):
        return introduced
    for match in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*[^,)=]+=\s*[^,)]*", stripped):
        name = match.group(1)
        if name in {"self", "cls"}:
            continue
        if _looks_like_public_config_key(name):
            introduced.add(name)
    return introduced


def _looks_like_public_config_key(name: str) -> bool:
    if name in {
        "self",
        "cls",
        "dtype",
        "device",
        "model_list",
        "projector",
        "projector_list",
        "tokenizer",
    }:
        return False
    return any(marker in name for marker in ("alignment", "confidence", "gate", "router", "projector", "soft_", "learned_", "span", "cache"))


def _flatten_config_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_flatten_config_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_flatten_config_keys(child))
    return keys


def _added_recipe_keys(diff_text: str) -> set[str]:
    keys: set[str] = set()
    current_file = ""
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/")
            continue
        if not current_file.startswith("recipe/") or Path(current_file).suffix not in {".json", ".yaml", ".yml"}:
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        text = line[1:].strip()
        for match in re.finditer(r"[\"']?([a-zA-Z_][a-zA-Z0-9_]*)[\"']?\s*[:=]", text):
            keys.add(match.group(1))
    return keys


def _load_previous_patch_failures(project_root: Path) -> dict[str, Any]:
    failures: dict[str, Any] = {}
    manifest_path = project_root / "plan" / "code_patches" / "patch_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        for entry in manifest.get("candidates") or []:
            if not isinstance(entry, dict) or entry.get("status") == "ok":
                continue
            key = sanitize_filename(str(entry.get("candidate_id") or entry.get("title") or "candidate"))
            failures[key] = _compact_previous_patch_failure(project_root, entry)
    failures.update(_load_previous_proxy_patch_failures(project_root))
    return failures


def _previous_failure_for_candidate(previous_failures: dict[str, Any] | None, idea_id: str, candidate: dict[str, Any]) -> dict[str, Any] | None:
    candidate_failure = candidate.get("previous_patch_failure") if isinstance(candidate.get("previous_patch_failure"), dict) else {}
    if candidate_failure:
        return candidate_failure
    if not previous_failures:
        return None
    keys = [
        idea_id,
        sanitize_filename(str(candidate.get("id") or "")),
        sanitize_filename(str(candidate.get("title") or "")),
    ]
    for key in keys:
        if key and key in previous_failures:
            return previous_failures[key]
    return None


def _compact_previous_patch_failure(project_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "status": entry.get("status"),
        "reason": entry.get("reason"),
        "changed_files": entry.get("changed_files") or [],
        "failed_checks": [],
        "activation_check": {},
    }
    validation_path = entry.get("validation")
    if validation_path:
        path = project_root / str(validation_path)
        if path.exists():
            try:
                validation = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                validation = {}
            payload["activation_check"] = validation.get("activation_check") or {}
            failed_checks = []
            for check in validation.get("checks") or []:
                if check.get("returncode") == 0:
                    continue
                failed_checks.append(
                    {
                        "name": check.get("name"),
                        "returncode": check.get("returncode"),
                        "stdout_tail": str(check.get("stdout") or "")[-1200:],
                        "stderr_tail": str(check.get("stderr") or "")[-1200:],
                    }
                )
            payload["failed_checks"] = failed_checks[:3]
    return payload


def _load_previous_proxy_patch_failures(project_root: Path) -> dict[str, Any]:
    result_path = project_root / "experiment" / "results" / "main_results.json"
    if not result_path.exists():
        result_path = project_root / "experiment" / "results" / "c2c_main_results.json"
    if not result_path.exists():
        return {}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    failures: dict[str, Any] = {}
    for candidate in payload.get("candidate_results") or []:
        if not isinstance(candidate, dict) or candidate.get("decision") not in {"proxy_rejected", "proxy_repairable"}:
            continue
        key = sanitize_filename(str(candidate.get("id") or candidate.get("title") or "candidate"))
        proxy_screen = candidate.get("proxy_screen") or {}
        attribution = candidate.get("failure_attribution") or {}
        proxy_effect_repair_contract = (
            attribution.get("proxy_effect_repair_contract")
            or proxy_screen.get("proxy_effect_repair_contract")
            or {}
        )
        failures[key] = {
            "status": candidate.get("decision"),
            "reason": proxy_screen.get("reason") or attribution.get("primary_failure") or candidate.get("decision"),
            "changed_files": ((candidate.get("patch_result") or {}).get("changed_files") or []),
            "proxy_screen": proxy_screen,
            "patch_risk": attribution.get("patch_risk") or {},
            "proxy_effect_repair_contract": proxy_effect_repair_contract,
            "quality_repair": attribution.get("quality_repair") or proxy_screen.get("quality_repair") or {},
            "repair_hint": proxy_screen.get("repair_hint") or "Repair the S2.5 patch so it clears cheap proxy before full S3.",
        }
    return failures


def _codex_patch_prompt(implementation_contract: dict[str, Any], edit_policy: DynamicEditPolicy) -> str:
    return (
        "You are the S2.5 code patch agent for a C2C research workflow.\n"
        "Edit the repository files directly in this temporary checkout. Do not run GPU training.\n"
        "This is not a proposal stage: do not stop after a blueprint, patch plan, or file list. The filesystem must contain the implemented code changes when you finish.\n"
        "Implement the supplied implementation contract, not the whole research transcript.\n"
        "Use as much coherent code as the selected mechanism needs to be executable and effect-bearing, while avoiding unrelated rewrites.\n"
        "This must be a mechanism-level implementation. Do not return a patch that only changes thresholds, top-k values, confidence floors, fallback values, or recipes.\n"
        "Do not implement an extra hard gate that merely lowers transfer coverage; any gating mechanism must be supported by coverage diagnostics and a matched-coverage ablation path.\n"
        "Honor the mechanism_contract ablation_switch so the new component can be disabled for S3 ablation.\n"
        "Honor mechanism_contract.coverage_diagnostics and mechanism_contract.matched_coverage_ablation.\n"
        "Do not edit script/evaluation/* or evaluator code; emit diagnostics through model/train/recipe outputs or focused tests.\n"
        "Use implementation_scope as guidance, not a hard size cap: bounded/medium/large describe expected integration surfaces; implement the coherent mechanism slice needed to pass truth gates.\n"
        "Patch size is not the default hard gate; execution truth is: config activation, ablation wiring, forward activation, runtime smoke, evaluator safety, and cheap proxy readiness.\n"
        "You may edit multiple files when the idea requires it, but all edits must respect the dynamic edit policy and avoid evaluator contamination.\n"
        "If you introduce a new configurable parameter, ensure it is explicitly activated by the supplied config_overrides or by an allowed recipe edit.\n"
        "Add or update focused tests for the implemented behavior. Do not edit generated caches or coverage outputs.\n\n"
        f"Implementation contract:\n{json.dumps(implementation_contract, ensure_ascii=False, indent=2)}\n\n"
        f"Allowed prefixes: {edit_policy.include_prefixes}\n"
        f"Allowed extensions: {edit_policy.include_extensions}\n"
        f"Forbidden prefixes: {edit_policy.exclude_prefixes}\n"
        f"Forbidden extensions: {edit_policy.exclude_extensions}\n\n"
        "Return a concise final rationale after editing and local validation. The framework will inspect the filesystem and freeze the diff."
    )


def archive_patched_code_snapshot(artifacts: ArtifactManager, adapter: C2CAdapter, candidate: dict[str, Any], patch_result: dict[str, Any]) -> dict[str, Any]:
    changed_files = list(patch_result.get("changed_files") or [])
    if not changed_files:
        return {"status": "skipped", "reason": "no changed files", "files": []}
    idea_id = sanitize_filename(str(candidate.get("id") or candidate.get("title") or "candidate"))
    records = []
    files = []
    for rel_path in changed_files:
        source = adapter.repo_root / rel_path
        if not source.exists() or not source.is_file():
            continue
        target_rel = f"code_snapshots/{idea_id}/{rel_path}"
        record = artifacts.write_text("S3_experiment", target_rel, source.read_text(encoding="utf-8", errors="ignore"), artifact_type="c2c_patched_code", summary=f"Patched code for {idea_id}")
        records.append(record["path"])
        files.append({"path": rel_path, "sha256": sha256_file(source), "artifact": record["path"]})
    manifest = {
        "status": "ok",
        "candidate_id": candidate.get("id"),
        "title": candidate.get("title"),
        "created_at": now_utc(),
        "changed_files": files,
        "repo_manifest": repo_snapshot_manifest(adapter.repo_root),
    }
    manifest_record = artifacts.write_json("S3_experiment", f"code_snapshots/{idea_id}/manifest.json", manifest, artifact_type="c2c_patched_code_manifest", summary=f"Patched code manifest for {idea_id}")
    return {"status": "ok", "manifest": manifest_record["path"], "files": records}
