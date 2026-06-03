"""Frozen code patch generation, validation, and application."""

from __future__ import annotations

import difflib
import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactManager
from .c2c import C2CAdapter, c2c_candidate_config_overrides, c2c_idea_novelty_report, repo_snapshot_manifest
from .utils import ensure_dir, now_utc, read_json, read_yaml, sanitize_filename, sha256_file, write_json, write_yaml


DEFAULT_CODE_PATCH_CONFIG = {
    "enabled": False,
    "backend": "codex_cli",
    "timeout_seconds": 1800,
    "max_candidates": 3,
    "variants_per_candidate": 2,
    "persistent_session": False,
    "use_git_worktree": False,
    "materialize_snapshot_baseline": True,
    "codex_json_events": True,
    "no_progress_timeout_seconds": None,
    "worktree_base_ref": "HEAD",
    "codex_sandbox": "workspace-write",
    "codex_sandbox_fallback": "danger-full-access",
    "codex_approval_policy": "never",
    "reasoning_effort": "high",
    "validation": {
        "require_py_compile": True,
        "require_targeted_tests": True,
        "runtime_smoke": {
            "enabled": True,
            "train_samples": 8,
            "timeout_seconds": 600,
            "gpu_ids": [0],
            "skip_if_missing_train_entry": True,
        },
        "max_repair_attempts": 1,
        "max_contract_repair_attempts": 1,
        "max_changed_files": 6,
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


class CodexPatchBackend:
    def __init__(self, config: dict[str, Any], project_root: Path):
        self.config = config
        self.project_root = project_root

    def generate(self, implementation_contract: dict[str, Any], temp_repo: Path, edit_policy: DynamicEditPolicy) -> dict[str, Any]:
        if not shutil.which("codex"):
            return {"status": "codex_failed", "reason": "codex executable not found"}
        code_patch_config = _code_patch_config(self.config)
        primary_sandbox = str(code_patch_config.get("codex_sandbox") or "workspace-write")
        primary = self._run_codex_once(implementation_contract, temp_repo, edit_policy, sandbox=primary_sandbox)
        fallback_sandbox = str(code_patch_config.get("codex_sandbox_fallback") or "")
        if (
            fallback_sandbox
            and fallback_sandbox != primary_sandbox
            and _codex_sandbox_error(primary)
        ):
            fallback = self._run_codex_once(implementation_contract, temp_repo, edit_policy, sandbox=fallback_sandbox)
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

    def _run_codex_once(self, implementation_contract: dict[str, Any], temp_repo: Path, edit_policy: DynamicEditPolicy, *, sandbox: str) -> dict[str, Any]:
        code_patch_config = _code_patch_config(self.config)
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as handle:
            output_path = Path(handle.name)
        prompt = _codex_patch_prompt(implementation_contract, edit_policy)
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
        command.extend(["-C", str(temp_repo), "-"])
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                input=prompt,
                cwd=temp_repo,
                timeout=int(code_patch_config.get("timeout_seconds") or 1800),
            )
            message = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        except subprocess.TimeoutExpired as exc:
            return {"status": "codex_failed", "reason": f"codex timed out: {exc}"}
        finally:
            output_path.unlink(missing_ok=True)
        if result.returncode != 0:
            reason = result.stderr[-2000:] or result.stdout[-2000:] or f"codex exited {result.returncode}"
            retryable = _codex_retryable_error_text(reason, result.stdout, result.stderr)
            return {
                "status": "retryable_codex_failed" if retryable else "codex_failed",
                "reason": reason,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
                "sandbox": sandbox,
                "retryable": retryable,
                "failure_category": "llm_rate_limit_or_quota" if retryable else "codex_cli_failure",
            }
        return {"status": "ok", "rationale": message.strip(), "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:], "sandbox": sandbox}


class CodexPersistentPatchBackend(CodexPatchBackend):
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
        recovery_actions: list[dict[str, Any]] = []
        if not _load_persistent_codex_session(Path(str(workspace.get("session_path") or ""))):
            preload = self._run_persistent_codex_once_inner(
                implementation_contract,
                repo,
                edit_policy,
                workspace,
                sandbox="read-only",
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
    ) -> dict[str, Any]:
        result = self._run_persistent_codex_once_inner(
            implementation_contract,
            repo,
            edit_policy,
            workspace,
            sandbox=sandbox,
            force_new_session=False,
            prompt_kind="patch",
        )
        if result.get("status") == "codex_failed" and result.get("resume_failed") and result.get("session_id"):
            retry = self._run_persistent_codex_once_inner(
                implementation_contract,
                repo,
                edit_policy,
                workspace,
                sandbox=sandbox,
                force_new_session=True,
                prompt_kind="patch",
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
            retryable = _codex_retryable_error_text(reason, result.stdout, result.stderr)
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

    def run(self, plan: dict[str, Any], candidate_ideas: list[dict[str, Any]]) -> dict[str, Any]:
        code_patch_config = _code_patch_config(self.config)
        if not code_patch_config.get("enabled", False):
            return {"status": "disabled", "candidates": [], "artifacts": []}
        adapter = C2CAdapter(self.project_root, self.config)
        policy = DynamicEditPolicy.from_config(code_patch_config.get("dynamic_whitelist") or {})
        max_candidates = int(code_patch_config.get("max_candidates") or 3)
        previous_failures = _load_previous_patch_failures(self.project_root)
        manifest_candidates = []
        artifacts = []
        for idx, candidate in enumerate(candidate_ideas):
            if idx >= max_candidates:
                candidate["code_patch"] = {"status": "skipped_not_in_patch_budget"}
                continue
            result = self._generate_candidate_patch(adapter, policy, candidate, plan, previous_failures=previous_failures)
            candidate["code_patch"] = result["code_patch"]
            manifest_candidates.append(result["manifest_entry"])
            artifacts.extend(result.get("artifacts", []))
        retryable_patch_count = sum(1 for item in manifest_candidates if _patch_failure_retryable(item))
        valid_patch_count = sum(1 for item in manifest_candidates if item.get("status") == "ok")
        if valid_patch_count:
            manifest_status = "ok"
        elif retryable_patch_count:
            manifest_status = "retryable_no_valid_patch"
        else:
            manifest_status = "no_valid_patch"
        manifest = {
            "status": manifest_status,
            "created_at": now_utc(),
            "backend": code_patch_config.get("backend", "codex_cli"),
            "candidate_count": len(manifest_candidates),
            "valid_patch_count": valid_patch_count,
            "failed_patch_count": sum(1 for item in manifest_candidates if item.get("status") not in {"ok", "skipped_not_in_patch_budget"}),
            "retryable_patch_count": retryable_patch_count,
            "retryable": bool(retryable_patch_count and not valid_patch_count),
            "policy": {
                "include_prefixes": policy.include_prefixes,
                "include_extensions": policy.include_extensions,
                "exclude_prefixes": policy.exclude_prefixes,
                "exclude_extensions": policy.exclude_extensions,
                "include_root_globs": policy.include_root_globs,
            },
            "candidates": manifest_candidates,
            "patches": manifest_candidates,
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
        code_patch_config = _code_patch_config(self.config)
        variant_count = max(1, min(3, int(code_patch_config.get("variants_per_candidate") or 1)))
        attempts: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        last_result: dict[str, Any] | None = None
        for variant_index in range(1, variant_count + 1):
            result = self._generate_candidate_patch_variant(
                adapter,
                policy,
                candidate,
                plan,
                previous_failures=previous_failures,
                variant_index=variant_index,
                variant_count=variant_count,
                previous_variant_attempts=attempts,
            )
            attempt = _compact_patch_variant_attempt(result.get("manifest_entry") or {}, variant_index=variant_index)
            attempts.append(attempt)
            _annotate_patch_variant_result(
                result,
                variant_index=variant_index,
                variant_count=variant_count,
                attempts=attempts,
            )
            last_result = result
            results.append(result)
        ok_results = [result for result in results if (result.get("manifest_entry") or {}).get("status") == "ok"]
        if ok_results:
            selected = _select_best_patch_variant(ok_results)
            _annotate_selected_patch_variant(selected, attempts=attempts, selected_reason="quality_score")
            return selected
        if last_result is None:
            raise RuntimeError("variant_count must be at least one")
        return last_result

    def _generate_candidate_patch_variant(
        self,
        adapter: C2CAdapter,
        policy: DynamicEditPolicy,
        candidate: dict[str, Any],
        plan: dict[str, Any],
        *,
        previous_failures: dict[str, Any] | None = None,
        variant_index: int = 1,
        variant_count: int = 1,
        previous_variant_attempts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        idea_id = sanitize_filename(str(candidate.get("id") or candidate.get("title") or "candidate"))
        base_rel = f"code_patches/{idea_id}" if variant_index == 1 else f"code_patches/{idea_id}/variants/v{variant_index}"
        implementation_contract = _build_implementation_contract(
            candidate,
            plan,
            policy,
            previous_failure=_previous_failure_for_candidate(previous_failures, idea_id, candidate),
        )
        if variant_count > 1:
            implementation_contract = _implementation_contract_with_variant_feedback(
                implementation_contract,
                previous_variant_attempts or [],
                variant_index=variant_index,
                variant_count=variant_count,
            )
        worktree_workspace = _prepare_code_worktree_workspace(
            self.project_root,
            self.config,
            idea_id=idea_id,
            variant_index=variant_index,
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
        prompt_text = _codex_patch_prompt(implementation_contract, policy)
        workspace_context = _patch_workspace_context(adapter.repo_root, worktree_workspace, idea_id=idea_id)
        with workspace_context as temp_repo:
            backend_result = self.backend.generate(implementation_contract, temp_repo, policy)
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
            draft = _build_patch_from_repo_delta(adapter.repo_root, temp_repo, policy)
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
                    draft = _build_patch_from_repo_delta(adapter.repo_root, temp_repo, policy)
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
                draft = _build_patch_from_repo_delta(adapter.repo_root, temp_repo, policy)
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
                draft = _build_patch_from_repo_delta(adapter.repo_root, temp_repo, policy)
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
            activation = _validate_config_activation(draft.get("diff", ""), candidate)
            validation["activation_check"] = activation
            validation["risk_check"] = risk_check
            validation["mechanism_review"] = mechanism_review
            validation["recovery_actions"] = backend_result.get("recovery_actions", [])
            max_repair_attempts = int(
                (_code_patch_config(self.config).get("validation", {}) or {}).get("max_repair_attempts", 1)
                or 0
            )
            repair_attempt = 0
            while (
                validation["status"] != "ok"
                or activation["status"] != "ok"
                or risk_check["status"] != "ok"
                or mechanism_review["status"] != "ok"
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
                draft = _build_patch_from_repo_delta(adapter.repo_root, temp_repo, policy)
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
                activation = _validate_config_activation(draft.get("diff", ""), candidate)
                validation["activation_check"] = activation
                risk_check = _validate_patch_proxy_risk(draft, candidate, self.config)
                validation["risk_check"] = risk_check
                validation["mechanism_review"] = mechanism_review
                validation["recovery_actions"] = backend_result.get("recovery_actions", [])
            patch_payload = {
                "schema_version": 1,
                "candidate_id": candidate.get("id"),
                "title": candidate.get("title"),
                "created_at": now_utc(),
                "backend": _code_patch_config(self.config).get("backend", "codex_cli"),
                "backend_sandbox": backend_result.get("sandbox"),
                "code_worktree": backend_result.get("code_worktree") or _compact_code_worktree(worktree_workspace),
                "codex_call": backend_result.get("codex_call") or {},
                "recovery_actions": backend_result.get("recovery_actions", []),
                "operations": draft["operations"],
                "changed_files": draft["changed_files"],
                "implementation_contract": implementation_contract,
                "activation_check": activation,
                "risk_check": risk_check,
                "mechanism_review": mechanism_review,
                "rationale": backend_result.get("rationale", ""),
            }
            if validation["status"] != "ok":
                status = "validation_failed"
            elif activation["status"] != "ok":
                status = activation["status"]
            elif risk_check["status"] != "ok":
                status = risk_check["status"]
            elif mechanism_review["status"] != "ok":
                status = mechanism_review["status"]
            elif not draft["operations"]:
                status = "blocked_no_executable_change"
            else:
                status = "ok"
            quality_score = _patch_quality_score(
                draft,
                validation,
                activation,
                risk_check,
                mechanism_review,
                implementation_contract,
            )
            patch_payload["quality_score"] = quality_score
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
                "quality_score": quality_score,
                "recovery_actions": backend_result.get("recovery_actions", []),
            }
            if backend_result.get("code_worktree"):
                code_patch["code_worktree"] = backend_result.get("code_worktree")
            if backend_result.get("session_id"):
                code_patch["codex_session_id"] = backend_result.get("session_id")
            if status != "ok":
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
            entry.update({"candidate_id": candidate.get("id"), "title": candidate.get("title")})
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
        if not isinstance(self.backend, CodexPatchBackend):
            return None
        fallback = self.backend._run_codex_once(implementation_contract, temp_repo, policy, sandbox=fallback_sandbox)
        action = {
            "action": "retry_codex_noop_with_fallback_sandbox",
            "status": fallback.get("status"),
            "primary_sandbox": primary_sandbox,
            "fallback_sandbox": fallback_sandbox,
            "reason": _codex_sandbox_reason(primary_result),
        }
        fallback["recovery_actions"] = [*primary_result.get("recovery_actions", []), action, *fallback.get("recovery_actions", [])]
        fallback["primary_attempt"] = _compact_backend_attempt(primary_result)
        return fallback

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
        repair_contract = _implementation_contract_with_contract_feedback(
            implementation_contract,
            failure,
            attempt=attempt,
        )
        repair = self.backend.generate(repair_contract, temp_repo, policy)
        action = {
            "action": "retry_codex_after_contract_failure",
            "status": repair.get("status"),
            "attempt": attempt,
            "failed_status": failure.get("status"),
            "failed_reason": _contract_failure_reason(failure),
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
        repair_contract = _implementation_contract_with_validation_feedback(
            implementation_contract,
            validation,
            draft,
            attempt=attempt,
        )
        repair = self.backend.generate(repair_contract, temp_repo, policy)
        action = {
            "action": "retry_codex_after_validation_failure",
            "status": repair.get("status"),
            "attempt": attempt,
            "failed_checks": _failed_validation_checks(validation, limit=3),
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
        entry.update({"candidate_id": candidate.get("id") or idea_id, "title": candidate.get("title")})
        return {"code_patch": code_patch, "manifest_entry": entry, "artifacts": list(records.values())}

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
    merged = dict(DEFAULT_CODE_PATCH_CONFIG)
    user = config.get("code_patch") or {}
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _default_code_patch_backend(config: dict[str, Any], project_root: Path) -> CodexPatchBackend:
    code_patch_config = _code_patch_config(config)
    backend_name = str(code_patch_config.get("backend") or "codex_cli")
    if backend_name == "codex_persistent_cli" or code_patch_config.get("persistent_session"):
        return CodexPersistentPatchBackend(config, project_root)
    return CodexPatchBackend(config, project_root)


def _prepare_code_worktree_workspace(
    project_root: Path,
    config: dict[str, Any],
    *,
    idea_id: str,
    variant_index: int,
) -> dict[str, Any]:
    code_patch_config = _code_patch_config(config)
    enabled = (
        str(code_patch_config.get("backend") or "") == "codex_persistent_cli"
        or bool(code_patch_config.get("persistent_session"))
        or bool(code_patch_config.get("use_git_worktree"))
    )
    if not enabled:
        return {"enabled": False, "status": "ok"}
    if not code_patch_config.get("use_git_worktree", True):
        return {
            "enabled": True,
            "status": "codex_failed",
            "reason": "persistent Codex backend requires use_git_worktree=true",
            "recovery_actions": [],
        }
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
    worktree_root = project_root / "plan" / "code_worktrees" / idea_id / f"v{variant_index}"
    repo = worktree_root / "repo"
    session_path = worktree_root / "codex_session.json"
    events_path = worktree_root / "codex_events.jsonl"
    metadata_path = worktree_root / "worktree_metadata.json"
    branch = _persistent_worktree_branch(project_root.name, idea_id, variant_index)
    session_key = f"s2_5:{idea_id}:v{variant_index}"
    recovery_actions: list[dict[str, Any]] = []
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
        "worktree_root": str(worktree_root.resolve()),
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
        "recovery_actions": recovery_actions,
    }
    write_json(metadata_path, metadata)
    return metadata


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
    if prompt_kind == "preload":
        return (
            "Persistent S2.5 Codex session bootstrap and context preload.\n"
            "Inspect the repository enough to form a compact patch blueprint for this idea. "
            "Do not edit files in this turn. Return a concise blueprint with intended files, integration path, "
            "runtime-smoke risks, and why the patch should affect cheap proxy.\n\n"
            + prompt
        )
    return (
        "Persistent S2.5 Codex repair turn. Continue from the same idea/session/worktree. "
        "Use the supplied validation or contract feedback as the primary task, keep the diff narrow, "
        "and avoid re-designing unrelated mechanism pieces.\n\n"
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
        "quota",
        "billing",
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
    if entry.get("status") == "retryable_codex_failed":
        return True
    if entry.get("failure_category") == "llm_rate_limit_or_quota":
        return True
    if _codex_retryable_error_text(entry.get("reason"), entry.get("stderr"), entry.get("stdout")):
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
        "reason": str(result.get("reason") or "")[-1000:],
        "stderr": str(result.get("stderr") or "")[-1000:],
        "stdout": str(result.get("stdout") or "")[-1000:],
        "rationale": str(result.get("rationale") or "")[-1000:],
    }


def _compact_patch_variant_attempt(entry: dict[str, Any], *, variant_index: int) -> dict[str, Any]:
    quality_score = entry.get("quality_score") if isinstance(entry.get("quality_score"), dict) else {}
    return {
        "variant_index": variant_index,
        "status": entry.get("status"),
        "reason": str(entry.get("reason") or "")[-1000:],
        "changed_files": list(entry.get("changed_files") or [])[:12],
        "has_executable_change": bool(entry.get("has_executable_change")),
        "risk_check": entry.get("risk_check") or {},
        "quality_score": quality_score,
        "validation": entry.get("validation"),
    }


def _annotate_patch_variant_result(
    result: dict[str, Any],
    *,
    variant_index: int,
    variant_count: int,
    attempts: list[dict[str, Any]],
) -> None:
    for key in ("code_patch", "manifest_entry"):
        value = result.get(key)
        if not isinstance(value, dict):
            continue
        value["variant_index"] = variant_index
        value["variant_count"] = variant_count
        value["selected_variant"] = variant_index if value.get("status") == "ok" else None
        value["variant_attempts"] = list(attempts)


def _annotate_selected_patch_variant(result: dict[str, Any], *, attempts: list[dict[str, Any]], selected_reason: str) -> None:
    entry = result.get("manifest_entry") if isinstance(result.get("manifest_entry"), dict) else {}
    selected_variant = entry.get("variant_index")
    selected_score = entry.get("quality_score") or {}
    updated_attempts = []
    for attempt in attempts:
        copied = dict(attempt)
        copied["selected"] = copied.get("variant_index") == selected_variant
        updated_attempts.append(copied)
    for key in ("code_patch", "manifest_entry"):
        value = result.get(key)
        if not isinstance(value, dict):
            continue
        value["selected_variant"] = selected_variant
        value["variant_attempts"] = updated_attempts
        value["selection_reason"] = selected_reason
        value["selected_quality_score"] = selected_score


def _select_best_patch_variant(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda result: (
            ((result.get("manifest_entry") or {}).get("quality_score") or {}).get("score", 0),
            -int(((result.get("manifest_entry") or {}).get("quality_score") or {}).get("diff_line_count", 0) or 0),
            -int(((result.get("manifest_entry") or {}).get("quality_score") or {}).get("changed_file_count", 0) or 0),
            -int((result.get("manifest_entry") or {}).get("variant_index") or 0),
        ),
    )


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
        "changed_file_count": len(changed_files),
        "diff_line_count": diff_line_count,
        "mechanism_review_status": mechanism_review.get("status"),
        "mechanism_type": (implementation_contract.get("mechanism_contract") or {}).get("mechanism_type")
        if isinstance(implementation_contract.get("mechanism_contract"), dict)
        else None,
    }


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


def _validate_patch_proxy_risk(draft: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    validation_config = _code_patch_config(config).get("validation", {}) or {}
    changed_files = list(draft.get("changed_files") or [])
    executable_files = [path for path in changed_files if Path(path).suffix in {".py", ".json", ".yaml", ".yml", ".toml"}]
    risk_labels: list[str] = []
    risk_files: list[dict[str, Any]] = []
    status = "ok"
    reason = ""
    max_changed_files = validation_config.get("max_changed_files")
    if max_changed_files is not None and len(changed_files) > int(max_changed_files):
        status = "patch_too_broad"
        reason = f"patch changes {len(changed_files)} files, above S2.5 risk limit {int(max_changed_files)}"
        risk_labels.append("patch_too_broad")
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
        "repair_hint": (
            "Keep the idea, but shrink or relocate the patch so cheap proxy can execute it safely."
            if status != "ok"
            else ""
        ),
    }


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
    if review_cfg.get("require_mechanism_evidence", True) and not (core_files or expected_hits):
        issues.append("missing_core_mechanism_file")
    if ablation_switch and not ablation_wired:
        if review_cfg.get("require_ablation_wired", False):
            issues.append("ablation_switch_not_wired")
        else:
            soft_issues.append("ablation_switch_not_wired")
    if coverage_required and not coverage_evidence:
        if review_cfg.get("require_coverage_evidence", False):
            issues.append("missing_coverage_diagnostics_evidence")
        else:
            soft_issues.append("missing_coverage_diagnostics_evidence")
    if matched_required and not matched_coverage_evidence:
        if review_cfg.get("require_matched_coverage_evidence", False):
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
        or normalized == "rosetta/utils/evaluate.py"
        or name in {"unified_evaluator.py", "evaluate.py", "evaluator.py"}
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
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
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
            adapter = C2CAdapter(smoke_repo, _runtime_smoke_adapter_config(config, smoke_repo, python_cmd, smoke_cfg))
            run_spec = adapter.materialize_candidate_configs(candidate or {}, _runtime_smoke_gpu_selection(smoke_cfg))
            _harden_runtime_smoke_train_config(run_spec.get("train_config"), smoke_cfg)
        except Exception as exc:
            return {
                "name": "runtime_smoke:first_batch_train",
                "returncode": 1,
                "status": "failed",
                "failure_category": "runtime_smoke_materialization_failed",
                "stderr": _validation_output_excerpt(f"{type(exc).__name__}: {exc}"),
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
            }

        command = _disable_wandb_for_command(command)
        timeout_seconds = max(1, int(smoke_cfg.get("timeout_seconds") or 600))
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
                "timeout_seconds": timeout_seconds,
                "repair_hint": _runtime_smoke_repair_hint("runtime_smoke_timeout"),
            }
    stdout = _validation_output_excerpt(result.stdout)
    stderr = _validation_output_excerpt(result.stderr)
    category = _runtime_smoke_failure_category(stdout, stderr, result.returncode)
    check = {
        "name": "runtime_smoke:first_batch_train",
        "returncode": result.returncode,
        "status": "ok" if result.returncode == 0 else "failed",
        "failure_category": category,
        "stdout": stdout,
        "stderr": stderr,
        "command": _validation_output_excerpt(command, max_chars=1200),
        "train_samples": _runtime_smoke_train_samples(smoke_cfg),
        "repair_hint": _runtime_smoke_repair_hint(category),
    }
    return check


def _disable_wandb_for_command(command: str) -> str:
    prefix = (
        "WANDB_DISABLED=true "
        "WANDB_MODE=disabled "
        "WANDB_START_METHOD=thread "
        "WANDB_REQUIRE_SERVICE=false "
    )
    return f"{prefix}{command}"


def _runtime_smoke_adapter_config(config: dict[str, Any], repo_root: Path, python_cmd: str, smoke_cfg: dict[str, Any]) -> dict[str, Any]:
    adapter_config = copy.deepcopy(config)
    c2c_cfg = adapter_config.setdefault("c2c", {})
    c2c_cfg["snapshot_path"] = str(repo_root)
    c2c_cfg["env_python"] = python_cmd
    small_loop = c2c_cfg.setdefault("small_loop", {})
    small_loop["train_samples"] = _runtime_smoke_train_samples(smoke_cfg)
    small_loop["eval_limit"] = int(smoke_cfg.get("eval_limit") or 1)
    small_loop["eval_datasets"] = list(smoke_cfg.get("eval_datasets") or small_loop.get("eval_datasets") or ["mmlu-redux"])[:1]
    small_loop["gpu_ids"] = smoke_cfg.get("gpu_ids", small_loop.get("gpu_ids", [0]))
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


def _runtime_smoke_gpu_selection(smoke_cfg: dict[str, Any]) -> Any:
    class _SmokeGpuSelection:
        pass

    selection = _SmokeGpuSelection()
    selection.selected_ids = _coerce_runtime_smoke_gpu_ids(smoke_cfg.get("gpu_ids", [0]))
    selection.policy = {"source": "s2_5_runtime_smoke"}
    selection.reason = "configured_runtime_smoke_gpu_ids"
    return selection


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
        "runtime_smoke_oom": "Shrink temporary tensors and avoid full-cache materialization in the first-batch path.",
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
            "Implement only the selected idea, with the smallest coherent code change that can affect S3 training/evaluation.",
            "Implement the mechanism-level claim in mechanism_contract; do not satisfy the task with only threshold, top-k, confidence-floor, fallback, or recipe tuning.",
            "Do not implement the idea as an added hard accept/reject gate on top of the baseline. If gating is part of the mechanism, it must be trained/estimated and paired with coverage diagnostics.",
            "Effect-first discovery: prioritize runnable code and cheap-proxy/full-S3 effect over paperization-only diagnostics.",
            "Keep ablation_switch, coverage diagnostics, and matched-coverage support lightweight when natural; missing paperization evidence can be completed after an effect is found.",
            "Emit or preserve lightweight mechanism evidence where natural, such as accepted-span counts, utility/verifier/pathology stats, or bridge-memory stats.",
            *scope_requirements,
            "Prefer existing local abstractions and configuration style over new framework code.",
            "If you add a new configurable constructor or recipe parameter, it must be explicitly represented in the supplied config_overrides or by editing an allowed recipe file.",
            "Do not repeat prior failed patch behavior; if previous_patch_failure is present, fix the reported validation or activation issue.",
            *_previous_proxy_effect_repair_requirements(proxy_effect_repair_contract),
            "Add focused tests for the new behavior. Do not run GPU training.",
            "Do not create run-result artifacts such as local/auto_research_runs files during S2.5; S3 materializes those outputs.",
            "Do not edit data, model weights, checkpoints, historical results, caches, or logs.",
            *_previous_quality_repair_requirements(previous_failure),
        ],
        "previous_patch_failure": previous_failure or {},
        "proxy_effect_repair_contract": _compact_value(proxy_effect_repair_contract, max_chars=1800) if proxy_effect_repair_contract else {},
    }


def _scope_requirements(scope: str, implementation_plan: dict[str, Any]) -> list[str]:
    normalized = scope.strip().lower()
    if normalized == "large":
        mvp = implementation_plan.get("mvp_slice") or "Implement only the first executable MVP slice from decomposition_plan."
        return [
            "This is a large-scope idea: do not rewrite the full training/evaluation stack in one patch.",
            f"Implement the first decomposed MVP slice only: {mvp}",
            "Keep integration points explicit and leave TODO-free, runnable baseline fallback behavior behind the ablation switch.",
        ]
    if normalized == "medium":
        return [
            "This is a medium-scope idea: new helper modules are allowed when listed in implementation_scope.required_new_files.",
            "Wire every new module through the listed integration_points and add or update the listed smoke_tests.",
        ]
    return [
        "This is a bounded-scope idea: prefer in-place changes to existing C2C model files unless a listed required_new_file is essential.",
    ]


def _previous_proxy_effect_repair_requirements(contract: dict[str, Any]) -> list[str]:
    if not isinstance(contract, dict) or not contract:
        return []
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
        "activation_check": activation,
        "mechanism_review": mechanism_review,
        "changed_files": draft.get("changed_files") or [],
        "instruction": (
            "Repair the existing patch in this checkout so validation passes. "
            "If activation_check reports missing config parameters, either remove/inline the unapproved public parameter, "
            "rename it to an existing activated config key, or update the patch so the parameter is activated by supplied config_overrides. "
            "If mechanism_review reports issues, repair the patch quality directly: add mechanism_evidence_map-worthy model/train/recipe changes, "
            "wire the ablation switch, emit coverage and matched-coverage diagnostics, and remove evaluator-like edits. "
            "If failed_checks contains runtime_smoke:first_batch_train, repair the actual C2C train/forward path before S3: "
            "fix dtype/device mismatches, valid_mask handling, and first-batch runtime exceptions instead of bypassing the smoke check. "
            "Keep the same idea, preserve allowed changes, and do not add generated run artifacts."
        ),
    }
    requirements = list(repair_contract.get("s2_5_requirements") or [])
    requirements.append("This is a repair retry after validation or activation failure; fix the failed checks before returning.")
    if (validation.get("mechanism_review") or {}).get("status") not in {None, "ok"}:
        requirements.append("Mechanism self-review is mandatory: the repaired patch must provide concrete mechanism evidence, ablation wiring, and required diagnostics without evaluator-like edits.")
    if any(str(check.get("name") or "").startswith("runtime_smoke:") for check in validation.get("checks") or [] if isinstance(check, dict)):
        requirements.append("Runtime smoke is mandatory: the repaired patch must complete a one-sample first-batch train smoke before proxy train.")
    repair_contract["s2_5_requirements"] = requirements
    return repair_contract


def _implementation_contract_with_variant_feedback(
    implementation_contract: dict[str, Any],
    previous_variant_attempts: list[dict[str, Any]],
    *,
    variant_index: int,
    variant_count: int,
) -> dict[str, Any]:
    variant_contract = json.loads(json.dumps(implementation_contract, ensure_ascii=False))
    variant_contract["patch_variant"] = {
        "variant_index": variant_index,
        "variant_count": variant_count,
        "previous_variant_attempts": previous_variant_attempts[-3:],
        "instruction": (
            "Generate a distinct implementation variant for the same idea. "
            "If previous variants failed, avoid repeating their changed files, failure mode, or risk pattern unless that file is the only viable integration point. "
            "Prefer the smallest executable mechanism that can pass S2.5 validation and S3 cheap proxy."
        ),
    }
    requirements = list(variant_contract.get("s2_5_requirements") or [])
    if variant_index > 1:
        requirements.append("This is a best-of-N retry after a failed patch variant; produce a materially different implementation, not a cosmetic retry.")
    variant_contract["s2_5_requirements"] = requirements
    return variant_contract


def _implementation_contract_with_contract_feedback(
    implementation_contract: dict[str, Any],
    failure: dict[str, Any],
    *,
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
            " If contract_failure_feedback.status is patch_too_broad, reduce changed files to the core mechanism and one focused test."
            " If it is proxy_risk_repair_required or risk_labels include evaluation_code_changed, fully revert evaluator edits and do not edit script/evaluation/*."
            " Move evidence emission into model/train outputs, recipe config, or focused tests instead."
        ),
    }
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
            }
        )
    return failed[:limit]


def _validate_config_activation(diff_text: str, candidate: dict[str, Any]) -> dict[str, Any]:
    introduced = _introduced_python_config_parameters(diff_text)
    if not introduced:
        return {"status": "ok", "introduced_config_parameters": [], "activated_parameters": [], "missing_parameters": []}
    configured_keys = _flatten_config_keys(c2c_candidate_config_overrides(candidate))
    recipe_keys = _added_recipe_keys(diff_text)
    activated = sorted(key for key in introduced if key in configured_keys or key in recipe_keys)
    missing = sorted(key for key in introduced if key not in set(activated))
    if missing:
        return {
            "status": "config_activation_missing",
            "reason": (
                "Patch introduced configurable parameter(s) that are not explicitly activated "
                f"by experiment_contract.config_overrides or an allowed recipe edit: {', '.join(missing)}"
            ),
            "introduced_config_parameters": sorted(introduced),
            "activated_parameters": activated,
            "missing_parameters": missing,
            "configured_keys": sorted(configured_keys),
            "recipe_keys": sorted(recipe_keys),
        }
    return {
        "status": "ok",
        "introduced_config_parameters": sorted(introduced),
        "activated_parameters": activated,
        "missing_parameters": [],
        "configured_keys": sorted(configured_keys),
        "recipe_keys": sorted(recipe_keys),
    }


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
        "Implement the supplied implementation contract, not the whole research transcript.\n"
        "Use the smallest coherent code change that can affect the deterministic S3 experiment.\n"
        "This must be a mechanism-level implementation. Do not return a patch that only changes thresholds, top-k values, confidence floors, fallback values, or recipes.\n"
        "Do not implement an extra hard gate that merely lowers transfer coverage; any gating mechanism must be supported by coverage diagnostics and a matched-coverage ablation path.\n"
        "Honor the mechanism_contract ablation_switch so the new component can be disabled for S3 ablation.\n"
        "Honor mechanism_contract.coverage_diagnostics and mechanism_contract.matched_coverage_ablation.\n"
        "Do not edit script/evaluation/* or evaluator code; emit diagnostics through model/train/recipe outputs or focused tests.\n"
        "Use implementation_scope to decide patch size: bounded means in-place change, medium may add listed helper files, large means implement only the first MVP slice from the decomposition plan.\n"
        "You may edit multiple files when the idea requires it, but all edits must respect the dynamic edit policy.\n"
        "If you introduce a new configurable parameter, ensure it is explicitly activated by the supplied config_overrides or by an allowed recipe edit.\n"
        "Add or update focused tests for the implemented behavior. Do not edit generated caches or coverage outputs.\n\n"
        f"Implementation contract:\n{json.dumps(implementation_contract, ensure_ascii=False, indent=2)}\n\n"
        f"Allowed prefixes: {edit_policy.include_prefixes}\n"
        f"Allowed extensions: {edit_policy.include_extensions}\n"
        f"Forbidden prefixes: {edit_policy.exclude_prefixes}\n"
        f"Forbidden extensions: {edit_policy.exclude_extensions}\n\n"
        "Return a concise final rationale. The framework will inspect the filesystem and freeze the diff."
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
