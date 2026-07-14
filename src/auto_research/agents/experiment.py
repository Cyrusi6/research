"""Experiment stage."""

from __future__ import annotations

import json
import hashlib
import os
import shlex
import copy
import math
import shutil
import re
import time
from pathlib import Path
from typing import Any
from copy import deepcopy


from ..c2c import C2CAdapter, DEFAULT_BASELINE, c2c_candidate_config_overrides, c2c_proxy_screen_config, default_c2c_ideas
from ..code_patch import DynamicEditPolicy, FrozenPatchGuard, archive_patched_code_snapshot
from ..config import bootstrap_proxy_only_enabled
from ..adapters.runner import ExperimentRunner
from ..failure_log import FailureLogManager, build_c2c_feedback_bundle, is_retryable_c2c_candidate
from ..itr_ideas import screening_summary_markdown
from ..s3_proxy_contracts import (
    build_c2c_effective_proxy_policy,
    build_c2c_full_s3_worthiness_score,
    build_c2c_proxy_baseline_fingerprint,
    build_c2c_proxy_cache_report,
    build_c2c_proxy_calibration_policy,
    build_c2c_proxy_decision_report,
)
from ..utils import compact_markdown, ensure_dir, now_utc, read_json, read_yaml, sanitize_filename, sha256_file, write_json
from ..domain_contracts import (
    canonical_hash,
    implementation_hash,
    trial_spec_hash,
    validate_contract,
    validate_direction_identity,
    validate_variant_identity,
)
from ..evidence import (
    EVIDENCE_SCHEMA_VERSIONS,
    EVIDENCE_MANIFEST_SCHEMA_VERSION as STAGED_EVIDENCE_MANIFEST_SCHEMA_VERSION,
    content_addressed_evidence_path,
    decode_evidence_inventory,
    encode_canonical_evidence,
)
from ..research_state import IntegrityError, ResearchEventLedger
from ..s3_validation import S3ValidationError, validate_failure_precommit, validate_trial_precommit
from .base import AgentContext


class ExperimentAgent:
    stage_key = "S3_experiment"

    def __init__(self, context: AgentContext):
        self.context = context
        self.runner = ExperimentRunner(context.config)

    def run(self, *, mode: str = "full", revisions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        env = self.runner.env_report()
        env_record = self.context.artifacts.write_text(
            self.stage_key,
            "env_report.md",
            compact_markdown(self._env_report_md(env)),
            artifact_type="env_report",
            summary="Environment inspection",
        )
        self._scaffold_code()
        if mode == "env_check":
            return {"artifacts": [env_record["path"]], "status": "ok"}

        direction = read_json(self.context.project_root / "literature" / "direction.json", default={}) or {}
        variant = read_json(self.context.project_root / "plan" / "variant.json", default={}) or {}
        trial_spec = read_json(self.context.project_root / "plan" / "trial_spec.json", default={}) or {}
        validate_direction_identity(direction)
        ledger = ResearchEventLedger(self.context.project_root)
        tried = [item for item in ledger.state().get("method_tried_history") or [] if isinstance(item, dict)]
        validate_variant_identity(direction, variant, tried_variants=tried)
        ledger.select_direction(direction, event_id=f"direction:{direction['direction_spec_hash']}")
        ledger.plan_variant(
            variant,
            feedback_from_attempt_ids=list((variant.get("lineage") or {}).get("feedback_from_attempt_ids") or []),
            event_id=f"variant:{variant['variant_spec_hash']}",
        )
        patch_manifest = read_json(self.context.project_root / "plan" / "code_patches" / "patch_manifest.json", default={}) or {}
        implementation_contract = read_json(self.context.project_root / "plan" / "code_patches" / "implementation_contract.json", default={}) or {}
        implementation_hash_value = implementation_hash(
            frozen_patch=patch_manifest,
            files=_implementation_file_hashes(self.context.project_root, patch_manifest),
            manifest=implementation_contract,
        )
        profile = str(((self.context.config.get("orchestration") or {}).get("profile") or "standard")).lower()
        required_phases = list((trial_spec.get("protocol") or {}).get("required_phases") or [])
        if profile == "bootstrap":
            attempt_kind = "bootstrap_proxy"
        elif required_phases == ["proxy", "full"]:
            attempt_kind = "proxy_full"
        elif required_phases == ["proxy"]:
            attempt_kind = "proxy"
        else:
            attempt_kind = "full"
        attempt = ledger.reserve_attempt(
            profile=profile,
            direction=direction,
            variant=variant,
            implementation_hash=implementation_hash_value,
            attempt_kind=attempt_kind,
            trial_spec=deepcopy(trial_spec),
        )
        plan = _trial_execution_view(trial_spec)
        execution = plan["execution"]
        simulate = bool(self.context.config.get("experiment", {}).get("simulate"))
        if revisions:
            revision_record = self.context.artifacts.write_text(
                self.stage_key,
                "revision_notes.md",
                compact_markdown(self._revision_notes_md(revisions)),
                artifact_type="revision_notes",
                summary="Applied revision instructions",
            )
        else:
            revision_record = None
        if attempt["state"] == "IMPLEMENTATION_REPAIR":
            state = ledger.state()
            route = state.get("last_route_outcome") if isinstance(state.get("last_route_outcome"), dict) else {}
            return {
                "artifacts": [env_record["path"]],
                "status": "blocked",
                "blocked_reason": "Implementation repair produced no new frozen implementation revision.",
                "attempt": attempt,
                "route_outcome": route,
            }
        if attempt["state"] in {"READY", "RESOURCE_PAUSED"}:
            attempt = self._prepare_attempt_execution(
                ledger,
                attempt,
                resource_probe=env.get("resource_probe") if attempt["state"] == "RESOURCE_PAUSED" else None,
                project_root=self.context.project_root,
            )
        elif attempt["state"] != "PROXY_RUNNING":
            raise IntegrityError(f"attempt {attempt['attempt_id']} cannot resume execution from {attempt['state']}")

        if execution.get("collector") == "c2c_small_loop":
            result = self._run_c2c_small_loop(
                plan,
                execution,
                env_record["path"],
                revision_record["path"] if revision_record else None,
                attempt=attempt,
                trial_spec=trial_spec,
            )
        elif simulate or execution.get("mode") == "simulate":
            result = self._run_simulated(
                plan,
                env_record["path"],
                revision_record["path"] if revision_record else None,
                attempt=attempt,
                trial_spec=trial_spec,
            )
        elif execution.get("mode") == "reuse" and execution.get("collector") == "reused_runs":
            result = self._collect_reused_run_results(execution, env_record["path"], revision_record["path"] if revision_record else None)
        elif execution.get("collector") == "itr_quick_screen":
            result = self._run_itr_quick_screen(execution, env_record["path"], revision_record["path"] if revision_record else None)
        else:
            commands = execution.get("commands") or []
            if not commands:
                blocked_reason = execution.get("blocked_reason") or "No execution commands defined."
                self.context.artifacts.write_text(
                    self.stage_key,
                    "self_heal_log.jsonl",
                    json.dumps({"result": "blocked", "reason": blocked_reason}) + "\n",
                    artifact_type="self_heal_log",
                    summary="Blocked run",
                )
                result = {"artifacts": [env_record["path"]], "status": "blocked", "blocked_reason": blocked_reason}
            else:
                execution_workdir = Path(execution.get("workdir") or self.context.project_root)
                log_path = self.context.project_root / "experiment" / "logs" / "command_runs.json"
                run_result = self.runner.run_plan_commands(commands, execution_workdir, log_path)
                log_record = self.context.artifacts.copy_into_stage(
                    self.stage_key,
                    log_path,
                    "logs/command_runs.json",
                    artifact_type="run_log",
                    summary="Executed experiment commands",
                )
                if run_result["status"] != "ok":
                    self.context.artifacts.write_text(
                        self.stage_key,
                        "self_heal_log.jsonl",
                        json.dumps({"result": "failed", "runs": run_result["runs"]}) + "\n",
                        artifact_type="self_heal_log",
                        summary="Self-heal trace",
                    )
                    result = {
                        "artifacts": [env_record["path"], log_record["path"]],
                        "status": "failed",
                        "failure_class": "implementation_failure",
                        "failure_evidence": {
                            "command_status": "failed",
                            "exit_code": next((run.get("returncode") for run in run_result.get("runs") or [] if run.get("returncode") not in {None, 0}), 1),
                            "artifact_path": log_record["path"],
                            "reason": "Experiment command returned a non-zero exit code.",
                        },
                    }
                elif execution.get("collector") == "laps_eval":
                    result = self._collect_laps_eval_results(log_path, env_record["path"], revision_record["path"] if revision_record else None)
                else:
                    result = {"artifacts": [env_record["path"]], "status": "blocked", "blocked_reason": "No supported result collector was available."}
        return self._finalize_trial(result, attempt=attempt, trial_spec=trial_spec, ledger=ledger)

    def _finalize_trial(
        self,
        result: dict[str, Any],
        *,
        attempt: dict[str, Any],
        trial_spec: dict[str, Any],
        ledger: ResearchEventLedger,
    ) -> dict[str, Any]:
        raw_artifacts: dict[str, str] = {}
        for rel_path in result.get("artifacts") or []:
            artifact_path = self.context.project_root / rel_path
            if artifact_path.exists() and artifact_path.is_file():
                raw_artifacts[str(rel_path)] = sha256_file(artifact_path)
        inventory = result.get("evidence_inventory")
        if not isinstance(inventory, list) or not inventory:
            failure_class = _structured_failure_class(result, {})
            evidence = _failure_evidence_from_result(
                project_root=self.context.project_root,
                attempt=attempt,
                result=result,
                failure_class=failure_class,
                raw_artifacts=raw_artifacts,
            )
            if evidence is None:
                return self._quarantine_invalid_s3(
                    result,
                    attempt,
                    S3ValidationError("successful execution requires a non-empty explicit staged evidence inventory"),
                    [],
                    raw_artifacts,
                )
            try:
                validate_failure_precommit(
                    project_root=self.context.project_root,
                    attempt=attempt,
                    failure_class=str(failure_class),
                    result=result,
                    artifact_hashes=raw_artifacts,
                    state=ledger.state(),
                )
            except S3ValidationError as exc:
                return self._quarantine_invalid_s3(result, attempt, exc, [], raw_artifacts)
            completed_attempt, route = ledger.disposition_failure(evidence)
            result["attempt"] = completed_attempt
            result["route_outcome"] = route
            result["committed_event_id"] = route["source"]["event_id"]
            result["committed_event_sequence"] = route["source"]["sequence"]
            result["committed_attempt_id"] = completed_attempt["attempt_id"]
            return result
        try:
            completion_evidence = _stage_evidence_inventory(
                project_root=self.context.project_root,
                attempt=attempt,
                trial_spec=trial_spec,
                inventory=inventory,
            )
            trial = ledger.validate_trial_precommit(completion_evidence)
            validate_trial_precommit(
                project_root=self.context.project_root,
                direction=read_json(self.context.project_root / "literature" / "direction.json", default={}) or {},
                variant=read_json(self.context.project_root / "plan" / "variant.json", default={}) or {},
                attempt=attempt,
                trial_spec=trial_spec,
                trial_result=trial,
                state=ledger.state(),
                allow_pending_full_transition=True,
            )
        except (ValueError, IntegrityError, S3ValidationError) as exc:
            return self._quarantine_invalid_s3(result, attempt, exc, [], raw_artifacts)
        if trial["completeness"] == "full":
            current = ledger.state()["attempts"][attempt["attempt_id"]]
            if current["state"] == "PROXY_RUNNING":
                ledger.transition_attempt(attempt["attempt_id"], "PROXY_COMPLETED", phase="proxy", phase_state="COMPLETED")
                ledger.transition_attempt(attempt["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
        completed_attempt, route = ledger.complete_attempt(completion_evidence)
        result.setdefault("artifacts", []).append("experiment/results/trial_result.json")
        result["attempt"] = completed_attempt
        result["route_outcome"] = route
        result["committed_event_id"] = route["source"]["event_id"]
        result["committed_event_sequence"] = route["source"]["sequence"]
        result["committed_attempt_id"] = completed_attempt["attempt_id"]
        return result

    @staticmethod
    def _prepare_attempt_execution(
        ledger: ResearchEventLedger,
        attempt: dict[str, Any],
        *,
        resource_probe: dict[str, Any] | None,
        project_root: Path,
    ) -> dict[str, Any]:
        current = attempt
        if hasattr(ledger, "state"):
            state = ledger.state()
            current = (state.get("attempts") or {}).get(attempt["attempt_id"], attempt)
        if current["state"] == "RESOURCE_PAUSED":
            if not isinstance(resource_probe, dict):
                raise IntegrityError("resource resume requires a current resource probe artifact")
            producer_run_id = f"resume-{current['attempt_id'][:12]}-g{current['lifecycle_generation']}"
            probe_payload = _identity_evidence_payload(
                attempt=current,
                producer_run_id=producer_run_id,
                evidence_kind="resource_probe",
                fields={
                    "resource_type": resource_probe.get("resource_type"),
                    "resource_id": resource_probe.get("resource_id"),
                    "required_capacity": resource_probe.get("required_capacity"),
                    "observed_capacity": resource_probe.get("observed_capacity"),
                    "unit": resource_probe.get("unit"),
                    "probe_status": resource_probe.get("probe_status"),
                    "observed_at": resource_probe.get("observed_at") or now_utc(),
                },
            )
            validate_contract(probe_payload, "resource_probe_v1.schema.json")
            probe_bytes = encode_canonical_evidence(probe_payload)
            probe_hash = hashlib.sha256(probe_bytes).hexdigest()
            probe_path = project_root / content_addressed_evidence_path(
                attempt_id=current["attempt_id"],
                producer_run_id=producer_run_id,
                evidence_kind="resource_probe",
                content_hash=probe_hash,
            )
            ensure_dir(probe_path.parent)
            if probe_path.exists() and probe_path.read_bytes() != probe_bytes:
                raise IntegrityError("resource probe content-addressed collision")
            if not probe_path.exists():
                temporary = probe_path.with_name(f".{probe_path.name}.{os.getpid()}.tmp")
                temporary.write_bytes(probe_bytes)
                os.replace(temporary, probe_path)
            pause_event = next(
                (
                    event for event in reversed(ledger.events())
                    if event["event_type"] == "AttemptDispositioned"
                    and (event.get("payload") or {}).get("failure_evidence", {}).get("attempt_id") == current["attempt_id"]
                    and (event.get("payload") or {}).get("failure_evidence", {}).get("failure_class") in {"resource_pause", "oom_retry"}
                ),
                None,
            )
            if pause_event is None:
                raise IntegrityError("resource resume requires a committed pause event")
            pause_evidence = pause_event["payload"]["failure_evidence"]
            resume_evidence = {
                "schema_version": "auto_research_resume_evidence_v2",
                "evidence_kind": "resume_evidence",
                "evidence_id": f"resume-evidence-{current['attempt_id'][:12]}-g{current['lifecycle_generation']}",
                "attempt_id": current["attempt_id"],
                "producer_run_id": producer_run_id,
                "direction_semantic_hash": current["direction_semantic_hash"],
                "direction_spec_hash": current["direction_spec_hash"],
                "variant_semantic_hash": current["variant_semantic_hash"],
                "variant_spec_hash": current["variant_spec_hash"],
                "trial_spec_hash": current["trial_spec_hash"],
                "protocol_hash": current["protocol_hash"],
                "sample_manifest_hash": current["sample_manifest_hash"],
                "evaluator_hash": current["evaluator_hash"],
                "cross_references": {"resource_probe_hash": probe_hash},
                "lifecycle_generation": current["lifecycle_generation"],
                "pause_event_id": pause_event["event_id"],
                "pause_evidence_hash": canonical_hash(pause_evidence),
                "resource_type": probe_payload["resource_type"],
                "resource_id": probe_payload["resource_id"],
                "required_capacity": probe_payload["required_capacity"],
                "observed_capacity": probe_payload["observed_capacity"],
                "unit": probe_payload["unit"],
                "probe_status": probe_payload["probe_status"],
                "observed_at": probe_payload["observed_at"],
            }
            current = ledger.resume_attempt(resume_evidence)
        if current["state"] != "READY":
            raise IntegrityError(f"attempt {current['attempt_id']} cannot enter execution from {current['state']}")
        proxy_state = (current.get("phases") or {}).get("proxy")
        if proxy_state == "COMPLETED" and current["attempt_kind"] == "proxy_full":
            return ledger.transition_attempt(current["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
        if current["attempt_kind"] == "full":
            return ledger.transition_attempt(current["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
        return ledger.transition_attempt(current["attempt_id"], "PROXY_RUNNING", phase="proxy", phase_state="RUNNING")

    def _quarantine_invalid_s3(
        self,
        result: dict[str, Any],
        attempt: dict[str, Any],
        error: Exception,
        observations: list[dict[str, Any]],
        raw_artifacts: dict[str, str],
    ) -> dict[str, Any]:
        quarantine = self.context.artifacts.write_json(
            self.stage_key,
            f"quarantine/{attempt['attempt_id']}.json",
            {"attempt_id": attempt["attempt_id"], "error": str(error), "observations": observations, "raw_artifacts": raw_artifacts},
            artifact_type="invalid_trial_draft",
            summary="Rejected S3 draft; not canonical",
            source_paths=list(raw_artifacts),
        )
        result.setdefault("artifacts", []).append(quarantine["path"])
        result["status"] = "blocked"
        result["blocked_reason"] = f"S3 pre-commit validation failed: {error}"
        return result

    def _scaffold_code(self) -> None:
        self.context.artifacts.write_text(
            self.stage_key,
            "code/README.md",
            "# Experiment Code\n\nThis directory stores executable experiment code and tests.\n",
            artifact_type="code_readme",
            summary="Experiment code scaffold",
        )
        self.context.artifacts.write_text(
            self.stage_key,
            "code/tests/test_placeholder.py",
            "def test_placeholder():\n    assert True\n",
            artifact_type="test_stub",
            summary="Placeholder experiment test",
        )
        self.context.artifacts.write_text(
            self.stage_key,
            "configs/default.yaml",
            "seed: 42\n",
            artifact_type="config",
            summary="Default experiment config",
        )
        self.context.artifacts.write_text(
            self.stage_key,
            "run_all.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\necho 'Populate commands via trial_spec.json execution.commands'\n",
            artifact_type="script",
            summary="Run scaffold",
        )

    def _run_simulated(
        self,
        plan: dict[str, Any],
        env_source: str,
        revision_source: str | None,
        *,
        attempt: dict[str, Any],
        trial_spec: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_name = str((((plan.get("execution_contract") or {}).get("runtime_config") or {}).get("baseline") or {}).get("name") or "registered_baseline")
        variant = read_json(self.context.project_root / "plan" / "variant.json", default={}) or {}
        results = {
            "schema_version": "auto_research_main_results_diagnostic_v1",
            "baseline": {"name": baseline_name, "primary_metric": {"mean": 78.4, "std": 0.5}},
            "proposed_method": {"name": variant.get("variant_id"), "primary_metric": {"mean": 80.1, "std": 0.4}},
            "acceptance": {"passed": True, "baseline_mean": 78.4, "candidate_mean": 80.1},
        }
        ablation = {
            "schema_version": "auto_research_ablation_results_diagnostic_v1",
            "full_model": 80.1,
            "without_core_module": 78.9,
        }
        verification = compact_markdown(
            "\n".join(
                [
                    "# Hypothesis Verification",
                    "",
                    "- H1: supported. The proposed method improves the primary metric over the baseline.",
                    "- H2: supported. Removing the core module reduces the metric.",
                ]
            )
        )
        sources = [env_source]
        if revision_source:
            sources.append(revision_source)
        main_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/main_results.json",
            results,
            artifact_type="results",
            summary="Main experiment results",
            source_paths=sources,
        )
        ablation_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/ablation_results.json",
            ablation,
            artifact_type="ablation",
            summary="Ablation results",
            source_paths=[main_record["path"]],
        )
        verification_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/hypothesis_verification.md",
            verification,
            artifact_type="verification",
            summary="Hypothesis verification",
            source_paths=[main_record["path"], ablation_record["path"]],
        )
        summary_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/summary.md",
            compact_markdown(
                "\n".join(
                    [
                        "# Experiment Summary",
                        "",
                        f"- Baseline {baseline_name}: 78.4 ± 0.5",
                        f"- Proposed: 80.1 ± 0.4",
                        "- Ablation confirms the added module contributes most of the gain.",
                    ]
                )
            ),
            artifact_type="summary",
            summary="Experiment summary",
            source_paths=[main_record["path"], ablation_record["path"]],
        )
        table_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/tables/main_table.tex",
            "\\begin{tabular}{lc}\n\\toprule\nMethod & Primary \\\\\n\\midrule\nBaseline & 78.4 \\\\\nProposed & 80.1 \\\\\n\\bottomrule\n\\end{tabular}\n",
            artifact_type="table",
            summary="Main table",
            source_paths=[main_record["path"]],
        )
        ablation_table = self.context.artifacts.write_text(
            self.stage_key,
            "results/tables/ablation_table.tex",
            "\\begin{tabular}{lc}\n\\toprule\nVariant & Primary \\\\\n\\midrule\nFull & 80.1 \\\\\nwo module & 78.9 \\\\\n\\bottomrule\n\\end{tabular}\n",
            artifact_type="table",
            summary="Ablation table",
            source_paths=[ablation_record["path"]],
        )
        figure_record = self.context.artifacts.write_text(
            self.stage_key,
            "figures/main_comparison.txt",
            "Figure placeholder: proposed method improves over baseline.\n",
            artifact_type="figure_placeholder",
            summary="Comparison figure placeholder",
            source_paths=[main_record["path"]],
        )
        self.context.artifacts.write_text(
            self.stage_key,
            "self_heal_log.jsonl",
            "",
            artifact_type="self_heal_log",
            summary="No self-heal actions needed",
        )
        artifacts = [
            main_record["path"],
            ablation_record["path"],
            verification_record["path"],
            summary_record["path"],
            table_record["path"],
            ablation_table["path"],
            figure_record["path"],
        ]
        producer_run_id = f"simulate-{attempt['attempt_id'][:12]}-g{attempt['lifecycle_generation']}"
        phase = "proxy" if attempt["attempt_kind"] == "bootstrap_proxy" else "full"
        inventory = []
        main_payload = _quantitative_evidence_payload(
            attempt=attempt,
            trial_spec=trial_spec,
            producer_run_id=producer_run_id,
            evidence_kind="main_results",
            phase=phase,
            role_values={"baseline": 78.4, "candidate": 80.1},
        )
        inventory.append(
            _write_staged_evidence_source(
                self.context.project_root,
                producer_run_id=producer_run_id,
                evidence_kind="main_results",
                payload=main_payload,
            )
        )
        if "ablation" in trial_spec["required_roles"]:
            inventory.append(
                _write_staged_evidence_source(
                    self.context.project_root,
                    producer_run_id=producer_run_id,
                    evidence_kind="ablation_results",
                    payload=_quantitative_evidence_payload(
                        attempt=attempt,
                        trial_spec=trial_spec,
                        producer_run_id=producer_run_id,
                        evidence_kind="ablation_results",
                        phase="ablation",
                        role_values={"ablation": 78.9},
                    ),
                )
            )
        inventory.append(
            _write_staged_evidence_source(
                self.context.project_root,
                producer_run_id=producer_run_id,
                evidence_kind="activation_evidence",
                payload=_identity_evidence_payload(
                    attempt=attempt,
                    producer_run_id=producer_run_id,
                    evidence_kind="activation_evidence",
                    fields={
                        "probe_id": "synthetic-forward-probe",
                        "status": "passed",
                        "command_status": "completed",
                        "exit_code": 0,
                        "implementation_surface_ids": list(variant.get("implementation_surface_ids") or ["synthetic-surface"]),
                    },
                ),
            )
        )
        return {"artifacts": artifacts, "evidence_inventory": inventory, "status": "ok"}

    @staticmethod
    def _env_report_md(report: dict[str, Any]) -> str:
        lines = ["# Environment Report", ""]
        lines.append(f"- Python executable: {report.get('python')}")
        lines.append(f"- tmux available: {report.get('tmux')}")
        gpu = report.get("gpu") or []
        if gpu:
            lines.append("- GPUs:")
            for item in gpu:
                lines.append(f"  - {item}")
        else:
            lines.append("- GPUs: none detected")
        return "\n".join(lines)

    @staticmethod
    def _revision_notes_md(revisions: list[dict[str, Any]]) -> str:
        lines = ["# Revision Notes", ""]
        for revision in revisions:
            lines.append(f"- {revision['id']}: {revision['action']}")
        return "\n".join(lines)

    def _collect_laps_eval_results(self, log_path: Path, env_source: str, revision_source: str | None) -> dict[str, Any] | None:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        text = "\n".join((run.get("stdout") or "") + "\n" + (run.get("stderr") or "") for run in payload.get("runs", []))
        metrics = self._parse_laps_metrics(text)
        if not metrics:
            return None
        sources = [env_source, "experiment/logs/command_runs.json"]
        if revision_source:
            sources.append(revision_source)
        main_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/main_results.json",
            metrics,
            artifact_type="results",
            summary="Parsed LAPS evaluation results",
            source_paths=sources,
        )
        ablation_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/ablation_results.json",
            {"note": "Baseline-only evaluation; ablation pending."},
            artifact_type="ablation",
            summary="Ablation placeholder",
            source_paths=[main_record["path"]],
        )
        verification_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/hypothesis_verification.md",
            compact_markdown(
                "\n".join(
                    [
                        "# Hypothesis Verification",
                        "",
                        "- H1: not yet tested against a proposed method; baseline evaluation completed.",
                        "- H2: not yet tested; ablation pending.",
                    ]
                )
            ),
            artifact_type="verification",
            summary="Baseline-only verification status",
            source_paths=[main_record["path"]],
        )
        summary_lines = ["# Experiment Summary", ""]
        for model_name, scores in metrics.items():
            summary_lines.append(
                f"- {model_name}: i2t R@1={scores['i2t']['R@1']}, t2i R@1={scores['t2i']['R@1']}, rsum={scores['rsum']}"
            )
        summary_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/summary.md",
            compact_markdown("\n".join(summary_lines)),
            artifact_type="summary",
            summary="Parsed evaluation summary",
            source_paths=[main_record["path"]],
        )
        return {
            "artifacts": [main_record["path"], ablation_record["path"], verification_record["path"], summary_record["path"]],
            "status": "blocked",
            "blocked_reason": "Baseline evaluation completed from local LAPS checkpoints; add proposed-method or fine-tuning commands before continuing.",
        }

    def _collect_reused_run_results(self, execution: dict[str, Any], env_source: str, revision_source: str | None) -> dict[str, Any]:
        selected_runs = execution.get("selected_runs", [])
        copied_artifacts = []
        main_results = {}
        summary_lines = ["# Reused Baseline Summary", ""]
        for idx, run in enumerate(selected_runs, start=1):
            label = f"run_{idx}_{run.get('repo_family','unknown')}_{run.get('dataset','unknown')}_{run.get('encoder','unknown')}"
            log_path = Path(run["log_path"])
            if log_path.exists():
                copied = self.context.artifacts.copy_into_stage(
                    self.stage_key,
                    log_path,
                    f"reused/{label}/eval.log",
                    artifact_type="reused_eval_log",
                    summary="Imported existing evaluation log",
                )
                copied_artifacts.append(copied["path"])
            model_path = run.get("model_best_path")
            if model_path and Path(model_path).exists():
                copied = self.context.artifacts.copy_into_stage(
                    self.stage_key,
                    Path(model_path),
                    f"reused/{label}/model_best.pth",
                    artifact_type="reused_checkpoint",
                    summary="Imported existing checkpoint",
                )
                copied_artifacts.append(copied["path"])
            for results_path in run.get("results_paths", []):
                path = Path(results_path)
                if path.exists():
                    copied = self.context.artifacts.copy_into_stage(
                        self.stage_key,
                        path,
                        f"reused/{label}/{path.name}",
                        artifact_type="reused_similarity",
                        summary="Imported existing similarity file",
                    )
                    copied_artifacts.append(copied["path"])
            key = f"{run.get('repo_family')}:{run.get('dataset')}:{run.get('encoder')}"
            main_results[key] = {
                "rsum": run.get("rsum"),
                "i2t": run.get("i2t"),
                "t2i": run.get("t2i"),
                "source_log": run.get("log_path"),
                "source_repo": run.get("repo_family"),
            }
            summary_lines.append(
                f"- {key}: rsum={run.get('rsum')}, i2t R@1={run.get('i2t',{}).get('R@1')}, t2i R@1={run.get('t2i',{}).get('R@1')}"
            )

        sources = [env_source, *copied_artifacts]
        if revision_source:
            sources.append(revision_source)
        main_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/main_results.json",
            main_results,
            artifact_type="results",
            summary="Reused baseline results from local MM runs",
            source_paths=sources,
        )
        ablation_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/ablation_results.json",
            {"status": "pending", "note": "Existing reusable runs provide baselines; ablation not yet generated for the new project."},
            artifact_type="ablation",
            summary="Ablation placeholder",
            source_paths=[main_record["path"]],
        )
        verification_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/hypothesis_verification.md",
            compact_markdown(
                "\n".join(
                    [
                        "# Hypothesis Verification",
                        "",
                        "- H1: baseline references imported from existing local runs; proposed method not yet evaluated.",
                        "- H2: ablation remains pending for the new project-specific method.",
                    ]
                )
            ),
            artifact_type="verification",
            summary="Reused baseline verification status",
            source_paths=[main_record["path"]],
        )
        summary_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/summary.md",
            compact_markdown("\n".join(summary_lines)),
            artifact_type="summary",
            summary="Summary of reused baseline runs",
            source_paths=[main_record["path"]],
        )
        return {
            "artifacts": [main_record["path"], ablation_record["path"], verification_record["path"], summary_record["path"], *copied_artifacts],
            "status": "blocked",
            "blocked_reason": "Reusable local baseline runs were imported successfully; add project-specific fine-tuning or new-method commands before continuing.",
        }

    def _run_itr_quick_screen(self, execution: dict[str, Any], env_source: str, revision_source: str | None) -> dict[str, Any]:
        workdir = Path(execution["workdir"])
        project_id = self.context.project_root.name
        control = execution["control"]
        candidates = execution.get("candidates", [])
        data_path = execution["data_path"]
        image_root = execution["image_root"]
        python_cmd = execution["python"]

        screen_specs = [{"id": "control", "title": "LAPS quick control baseline", "direction": "control", "screening_recipe": control}]
        screen_specs.extend(candidates)

        run_results = []
        copied_artifacts = []
        gpu_ids = [0, 1, 2, 3]

        for idx, idea in enumerate(screen_specs):
            recipe = idea["screening_recipe"]
            logger_rel = recipe["logger_name"]
            if not logger_rel.startswith("artifacts/"):
                logger_rel = f"artifacts/runs/{project_id}_{logger_rel}"
            recipe["logger_name"] = logger_rel
            run_dir = workdir / logger_rel
            train_log = run_dir / "train.log"
            eval_log = run_dir / "eval.log"

            metrics = self._parse_metrics_from_log(train_log) or self._parse_metrics_from_log(eval_log)
            status = "reused" if metrics else "pending"
            if not metrics:
                command = self._build_laps_train_command(
                    python_cmd=python_cmd,
                    data_path=data_path,
                    image_root=image_root,
                    logger_name=logger_rel,
                    gpu_id=gpu_ids[idx % len(gpu_ids)],
                    resume_path=recipe.get("resume"),
                    resume_strict=int(recipe.get("resume_strict", 1)),
                    learning_rate=float(recipe.get("learning_rate", 2e-4)),
                    batch_size=int(recipe.get("batch_size", 32)),
                    num_epochs=int(recipe.get("num_epochs", 1)),
                    changes=recipe.get("changes", {}),
                )
                command_result = self.runner.run_plan_commands(
                    [command],
                    workdir,
                    self.context.project_root / "experiment" / "logs" / f"{Path(logger_rel).name}_command.json",
                )
                status = command_result["status"]
                metrics = self._parse_metrics_from_log(train_log) or self._parse_metrics_from_log(eval_log)
            run_results.append(
                {
                    "id": idea["id"],
                    "title": idea["title"],
                    "direction": idea.get("direction", idea["id"]),
                    "run_dir": str(run_dir),
                    "train_log": str(train_log),
                    "metrics": metrics,
                    "status": status if metrics else "failed",
                }
            )

            if run_dir.exists():
                for rel_name in ["train.log", "Parameters.txt", "model_best.pth", "results_f30k.npy"]:
                    source = run_dir / rel_name
                    if source.exists():
                        copied = self.context.artifacts.copy_into_stage(
                            self.stage_key,
                            source,
                            f"screening/{Path(logger_rel).name}/{rel_name}",
                            artifact_type="itr_screen_artifact",
                            summary="Imported quick-screen artifact",
                        )
                        copied_artifacts.append(copied["path"])

        baseline = next((item for item in run_results if item["id"] == "control"), None)
        baseline_metrics = baseline.get("metrics") if baseline else None
        candidate_results = []
        for item in run_results:
            if item["id"] == "control":
                continue
            decision = self._screening_decision(item.get("metrics"), baseline_metrics)
            candidate_results.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "direction": item["direction"],
                    "status": item["status"],
                    "metrics": item.get("metrics"),
                    "decision": decision,
                    "train_log": item["train_log"],
                }
            )

        candidate_results.sort(
            key=lambda item: (
                0 if item["decision"] == "viable" else 1,
                -(item.get("metrics", {}) or {}).get("rsum", 0),
            )
        )

        if self.context.config.get("experiment", {}).get("cleanup_failed_idea_runs", True):
            failure_manager = FailureLogManager(self.context.config, external_root=workdir / "artifacts" / "runs")
            failure_manager.record_not_viable_ideas(
                project_id=project_id,
                baseline_metrics=baseline_metrics,
                candidate_results=candidate_results,
                cleanup=True,
            )

        summary_payload = {"baseline": baseline_metrics, "candidates": candidate_results}
        sources = [env_source, *copied_artifacts]
        if revision_source:
            sources.append(revision_source)
        main_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/main_results.json",
            {
                "baseline_control": baseline_metrics,
                "candidate_results": candidate_results,
            },
            artifact_type="results",
            summary="Quick-screen results for RIS-driven image-text retrieval ideas",
            source_paths=sources,
        )
        ablation_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/ablation_results.json",
            {"status": "pending", "note": "Quick-screen only; no full ablation package yet."},
            artifact_type="ablation",
            summary="Quick-screen ablation placeholder",
            source_paths=[main_record["path"]],
        )
        verification_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/hypothesis_verification.md",
            compact_markdown(
                "\n".join(
                    [
                        "# Hypothesis Verification",
                        "",
                        "- H1: quick-screen completed. Use the viable candidate list to decide the follow-up 3-epoch confirmation run.",
                        "- H2: ablation remains pending until one candidate is selected for confirmation.",
                    ]
                )
            ),
            artifact_type="verification",
            summary="Quick-screen hypothesis status",
            source_paths=[main_record["path"]],
        )
        summary_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/summary.md",
            screening_summary_markdown(summary_payload),
            artifact_type="summary",
            summary="Idea screening summary",
            source_paths=[main_record["path"]],
        )
        ideas_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/idea_screening_summary.json",
            summary_payload,
            artifact_type="idea_screening",
            summary="Structured idea screening results",
            source_paths=[main_record["path"]],
        )
        viable = [item for item in candidate_results if item["decision"] == "viable"]
        blocked_reason = (
            f"Quick-screen completed; {len(viable)} viable idea(s) found. Review results and select one for a 3-epoch confirmation run."
            if viable
            else "Quick-screen completed but no candidate cleared the viability threshold."
        )
        return {
            "artifacts": [main_record["path"], ablation_record["path"], verification_record["path"], summary_record["path"], ideas_record["path"], *copied_artifacts],
            "status": "blocked",
            "blocked_reason": blocked_reason,
        }

    def _run_c2c_small_loop(
        self,
        plan: dict[str, Any],
        execution: dict[str, Any],
        env_source: str,
        revision_source: str | None,
        *,
        attempt: dict[str, Any],
        trial_spec: dict[str, Any],
    ) -> dict[str, Any]:
        bootstrap_proxy_only = bootstrap_proxy_only_enabled(self.context.config)
        adapter = C2CAdapter(self.context.project_root, self.context.config)
        variant_spec = read_json(self.context.project_root / "plan" / "variant.json", default={}) or {}
        raw_candidates = [_c2c_candidate_from_variant_spec(variant_spec)]
        candidate_selection = self._c2c_s3_candidate_selection(
            raw_candidates,
            max_candidates=1,
        )
        candidates = candidate_selection["candidates"]
        run_results = []
        command_logs = []
        copied_sources = [env_source]
        if revision_source:
            copied_sources.append(revision_source)

        simulate = bool(self.context.config.get("experiment", {}).get("simulate"))
        mock_results = bool(self.context.config.get("c2c", {}).get("small_loop", {}).get("mock_results"))
        baseline_mean = float((execution.get("baseline") or adapter.baseline).get("mean") or DEFAULT_BASELINE["mean"])
        min_delta = float(execution.get("min_delta_to_pass", self.context.config.get("c2c", {}).get("small_loop", {}).get("min_delta_to_pass", 0.1)))
        max_regression = float(execution.get("max_dataset_regression", self.context.config.get("c2c", {}).get("small_loop", {}).get("max_dataset_regression", 2.0)))
        gpu_policy = dict(execution.get("gpu_policy") or self.context.config.get("experiment", {}).get("gpu_policy", {}))
        c2c_small_loop = self.context.config.get("c2c", {}).get("small_loop", {})
        configured_gpus = c2c_small_loop.get("gpu_ids", "auto")
        if configured_gpus not in (None, "", "auto"):
            gpu_policy["gpu_ids"] = configured_gpus
        elif execution.get("selected_gpu_ids") and gpu_policy.get("reuse_plan_selected_gpu_ids", False):
            gpu_policy["gpu_ids"] = execution["selected_gpu_ids"]
        else:
            gpu_policy["gpu_ids"] = "auto"
        gpu_policy.setdefault("min_free_mb", 8192)
        gpu_policy.setdefault("max_utilization_gpu", 40)
        gpu_policy.setdefault("respect_resource_filters", True)
        gpu_selection = self.runner.select_gpus(gpu_policy)
        proxy_gpu_policy = self._c2c_proxy_gpu_policy(execution=execution)
        proxy_gpu_selection = self._select_c2c_proxy_gpus(proxy_gpu_policy)

        selection_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/s3_candidate_selection.json",
            {
                **candidate_selection["report"],
                "execution_max_candidates": execution.get("max_candidates"),
                "effective_candidate_count": len(candidates),
            },
            artifact_type="c2c_s3_candidate_selection",
            summary="Locked C2C S3 candidate selection from S2.5 patch manifest",
            source_paths=[path for path in ["plan/code_patches/patch_manifest.json", "plan/variant.json", "plan/trial_spec.json"] if (self.context.project_root / path).exists()],
        )
        copied_sources.append(selection_record["path"])

        for idx, candidate in enumerate(candidates):
            candidate_result = self._run_single_c2c_candidate(
                adapter=adapter,
                candidate=candidate,
                index=idx,
                simulate=simulate or mock_results,
                baseline_mean=baseline_mean,
                min_delta=min_delta,
                max_regression=max_regression,
                gpu_selection=gpu_selection,
                proxy_gpu_selection=proxy_gpu_selection,
            )
            run_results.append(candidate_result)
            command_logs.extend(candidate_result.get("command_logs", []))
            if candidate_result.get("decision") in {"failed_no_metrics", "patch_rejected", "proxy_rejected", "proxy_repairable"} and candidate_result.get("command_status") == "failed":
                break
            if candidate_result.get("decision") == "candidate_win":
                break

        best_proxy = self._best_c2c_proxy_candidate(run_results)
        best = self._best_c2c_candidate(run_results)
        comparison_candidate = best or best_proxy
        comparison = self._c2c_acceptance_comparison(comparison_candidate, execution.get("baseline") or adapter.baseline, min_delta, max_regression)
        main_payload = {
            "schema_version": "c2c_main_results_diagnostic_v1",
            "baseline": execution.get("baseline") or adapter.baseline,
            "best_candidate": best,
            "best_proxy_candidate": best_proxy,
            "candidate_results": run_results,
            "s3_candidate_selection": candidate_selection["report"],
            "acceptance": comparison,
            "workflow_goal": self.context.config.get("c2c", {}).get("workflow_goal", "effect_first_discovery"),
            "bootstrap": {
                "enabled": bootstrap_proxy_only,
                "profile": "bootstrap" if bootstrap_proxy_only else "standard",
                "proxy_only": bootstrap_proxy_only,
                "proxy_reached": bool(best_proxy),
            },
            "gpu_selection": {
                "selected_gpu_ids": gpu_selection.selected_ids,
                "cuda_visible_devices": gpu_selection.cuda_visible_devices,
                "policy": gpu_selection.policy,
                "snapshot": gpu_selection.snapshot,
                "reason": gpu_selection.reason,
                "resource_aware": True,
                "plan_selected_gpu_ids": execution.get("selected_gpu_ids"),
            },
            "proxy_gpu_selection": {
                "selected_gpu_ids": proxy_gpu_selection.selected_ids,
                "cuda_visible_devices": proxy_gpu_selection.cuda_visible_devices,
                "policy": proxy_gpu_selection.policy,
                "snapshot": proxy_gpu_selection.snapshot,
                "reason": proxy_gpu_selection.reason,
                "resource_aware": True,
            },
        }
        main_payload["strong_reference_comparisons"] = self._c2c_strong_reference_comparisons(best, adapter)
        main_payload["paperization_readiness"] = _c2c_paperization_readiness(best, comparison)
        proxy_calibration = self._append_c2c_proxy_calibration(main_payload)
        main_payload["proxy_calibration"] = proxy_calibration.get("current_iteration")
        main_payload["proxy_calibration_summary"] = proxy_calibration.get("summary")
        self._write_c2c_proxy_policy_contracts(plan=plan, execution=execution, main_payload=main_payload, include_baseline=False)
        history = self._append_c2c_iteration_history(main_payload)
        main_payload["iteration_history"] = {
            "path": "experiment/results/c2c_iteration_history.json",
            "entry_count": len(history.get("iterations", [])),
            "best_mean_so_far": history.get("best_mean_so_far"),
            "best_delta_so_far": history.get("best_delta_so_far"),
            "best_proxy_mean_so_far": history.get("best_proxy_mean_so_far"),
            "best_proxy_delta_so_far": history.get("best_proxy_delta_so_far"),
            "consecutive_not_viable": history.get("consecutive_not_viable"),
        }
        posthoc = None if comparison.get("passed") or bootstrap_proxy_only else self._c2c_posthoc_review(main_payload)
        main_payload["posthoc_review"] = posthoc or {
            "status": "skipped",
            "reason": "bootstrap proxy-only profile" if bootstrap_proxy_only else "candidate accepted",
        }
        ablation_payload = self._c2c_ablation_payload(main_payload, adapter)
        main_payload["ablation_summary"] = {
            "status": ablation_payload.get("status"),
            "best_candidate_id": ablation_payload.get("best_candidate_id"),
            "best_supported": ablation_payload.get("best_supported"),
            "best_delta_enabled_vs_disabled": ablation_payload.get("best_delta_enabled_vs_disabled"),
        }
        main_payload = _compact_c2c_result_payload(main_payload)
        ablation_payload = _compact_c2c_result_payload(ablation_payload)
        sources = copied_sources
        matched_control_record = None
        coverage_record = None
        full_readiness_record = None
        if isinstance(best, dict) and isinstance(best.get("matched_control_metrics"), dict):
            matched_control_record = self.context.artifacts.write_json(
                self.stage_key,
                "results/matched_control_results.json",
                {"schema_version": "auto_research_matched_control_results_diagnostic_v1", **best["matched_control_metrics"]},
                artifact_type="matched_control_results",
                summary="Pre-registered matched-control execution results",
                source_paths=sources,
            )
        if isinstance(best, dict) and isinstance(best.get("coverage_metrics"), dict):
            coverage_record = self.context.artifacts.write_json(
                self.stage_key,
                "results/coverage_results.json",
                {"schema_version": "auto_research_coverage_results_diagnostic_v1", **best["coverage_metrics"]},
                artifact_type="coverage_results",
                summary="Pre-registered execution coverage results",
                source_paths=sources,
            )
        readiness_path = self.context.project_root / "experiment" / "results" / "c2c_full_s3_worthiness.json"
        if isinstance(best, dict) and not readiness_path.is_file():
            full_readiness_record = self.context.artifacts.write_json(
                self.stage_key,
                "results/c2c_full_s3_worthiness.json",
                {
                    "schema_version": "c2c_proxy_to_full_readiness_v1",
                    "candidate_id": best.get("id"),
                    "run_id": best.get("run_id"),
                    "status": "ready",
                    "full_train_allowed": True,
                    "source": "verified_direct_full_execution",
                },
                artifact_type="c2c_full_s3_readiness",
                summary="Direct full-execution readiness evidence",
                source_paths=sources,
            )
        main_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/main_results.json",
            main_payload,
            artifact_type="results",
            summary="C2C small-loop candidate results",
            source_paths=sources,
        )
        bootstrap_record = None
        if bootstrap_proxy_only:
            bootstrap_record = self.context.artifacts.write_json(
                self.stage_key,
                "results/bootstrap_proxy_completion.json",
                {
                    "schema_version": "bootstrap_proxy_completion_v1",
                    "profile": "bootstrap",
                    "proxy_only": True,
                    "bootstrap_proxy_complete": bool(best_proxy),
                    "status": "proxy_reached" if best_proxy else "proxy_missing",
                    "candidate_id": best_proxy.get("id") if isinstance(best_proxy, dict) else None,
                    "proxy_screen": best_proxy.get("proxy_screen") if isinstance(best_proxy, dict) else None,
                    "full_train_executed": False,
                    "full_eval_executed": False,
                },
                artifact_type="bootstrap_proxy_completion",
                summary="Bootstrap profile stopped after the first cheap proxy metric",
                source_paths=[main_record["path"]],
            )
        ablation_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/ablation_results.json",
            ablation_payload,
            artifact_type="ablation",
            summary="C2C automatic ablation results",
            source_paths=[main_record["path"]],
        )
        verification_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/hypothesis_verification.md",
            compact_markdown(self._c2c_verification_md(best, baseline_mean, run_results, min_delta, max_regression)),
            artifact_type="verification",
            summary="C2C hypothesis verification",
            source_paths=[main_record["path"]],
        )
        summary_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/summary.md",
            compact_markdown(self._c2c_summary_md(main_payload)),
            artifact_type="summary",
            summary="C2C small-loop summary",
            source_paths=[main_record["path"]],
        )
        loop_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/c2c_small_loop_results.json",
            main_payload,
            artifact_type="c2c_small_loop",
            summary="Structured C2C small-loop results",
            source_paths=[main_record["path"]],
        )
        posthoc_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/posthoc_review.json",
            main_payload["posthoc_review"],
            artifact_type="c2c_posthoc_review",
            summary="GPT posthoc review of C2C training and evaluation outcome",
            source_paths=[main_record["path"]],
        )
        failure_analysis_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/failure_analysis.md",
            compact_markdown(self._c2c_failure_analysis_md(main_payload, posthoc)),
            artifact_type="c2c_failure_analysis",
            summary="Failure analysis and next-round recommendations",
            source_paths=[main_record["path"], posthoc_record["path"]],
        )
        feedback_record = None
        if not comparison.get("passed") and not bootstrap_proxy_only:
            feedback_record = self._write_c2c_failure_feedback(main_payload, artifacts=[main_record["path"], posthoc_record["path"]])
        self.context.artifacts.write_text(
            self.stage_key,
            "self_heal_log.jsonl",
            "\n".join(json.dumps(item, ensure_ascii=False) for item in command_logs) + ("\n" if command_logs else ""),
            artifact_type="self_heal_log",
            summary="C2C command and patch trace",
        )
        if not best:
            blocked_reason = self._c2c_blocked_reason(run_results)
            if blocked_reason:
                return {
                    "artifacts": [
                        main_record["path"],
                        ablation_record["path"],
                        verification_record["path"],
                        summary_record["path"],
                        loop_record["path"],
                        posthoc_record["path"],
                        failure_analysis_record["path"],
                        *([bootstrap_record["path"]] if bootstrap_record else []),
                        *([feedback_record["path"]] if feedback_record else []),
                        *([matched_control_record["path"]] if matched_control_record else []),
                        *([coverage_record["path"]] if coverage_record else []),
                        *([full_readiness_record["path"]] if full_readiness_record else []),
                    ],
                    "status": "blocked",
                    "blocked_reason": blocked_reason,
                }
        evidence_inventory = _c2c_strict_evidence_inventory(
            project_root=self.context.project_root,
            attempt=attempt,
            trial_spec=trial_spec,
            comparison_candidate=comparison_candidate,
            baseline=execution.get("baseline") or adapter.baseline,
            simulate=simulate or mock_results,
        )
        return {
            "artifacts": [
                main_record["path"],
                ablation_record["path"],
                verification_record["path"],
                summary_record["path"],
                loop_record["path"],
                posthoc_record["path"],
                failure_analysis_record["path"],
                *([bootstrap_record["path"]] if bootstrap_record else []),
                *([feedback_record["path"]] if feedback_record else []),
                *([matched_control_record["path"]] if matched_control_record else []),
                *([coverage_record["path"]] if coverage_record else []),
                *([full_readiness_record["path"]] if full_readiness_record else []),
            ],
            "evidence_inventory": evidence_inventory,
            "status": "ok" if comparison.get("passed") or (bootstrap_proxy_only and best_proxy) else "not_viable",
        }

    def _c2c_proxy_gpu_policy(self, *, execution: dict[str, Any]) -> dict[str, Any]:
        experiment_policy = self.context.config.get("experiment", {}).get("gpu_policy", {})
        policy = dict(experiment_policy if isinstance(experiment_policy, dict) else {})
        proxy_cfg = c2c_proxy_screen_config(self.context.config)
        proxy_policy = proxy_cfg.get("gpu_policy") if isinstance(proxy_cfg.get("gpu_policy"), dict) else {}
        policy.update(proxy_policy)
        configured = policy.get("gpu_ids", "auto")
        if configured in (None, ""):
            configured = "auto"
        policy["gpu_ids"] = configured
        policy.setdefault("max_gpus", 1)
        policy.setdefault("min_free_mb", 8192)
        policy.setdefault("max_utilization_gpu", 40)
        policy.setdefault("respect_resource_filters", True)
        policy.setdefault("disable_resource_fallback", True)
        policy["selection_scope"] = "c2c_proxy_screen"
        if execution.get("selected_gpu_ids") and policy.get("reuse_plan_selected_gpu_ids", False):
            policy["gpu_ids"] = execution["selected_gpu_ids"]
        return policy

    def _select_c2c_proxy_gpus(self, policy: dict[str, Any]) -> Any:
        selection = self.runner.select_gpus(policy)
        if selection.selected_ids or not _c2c_gpu_resource_wait_enabled(policy) or not selection.snapshot:
            return selection
        wait_cfg = policy.get("resource_wait") if isinstance(policy.get("resource_wait"), dict) else {}
        timeout_seconds = max(0, int(wait_cfg.get("timeout_seconds") or 0))
        poll_seconds = max(1, int(wait_cfg.get("poll_seconds") or 60))
        started = time.monotonic()
        while time.monotonic() - started < timeout_seconds:
            time.sleep(min(poll_seconds, max(0.0, timeout_seconds - (time.monotonic() - started))))
            selection = self.runner.select_gpus(policy)
            if selection.selected_ids:
                selection.reason = "proxy_resource_wait_then_auto_selected_by_free_memory"
                return selection
        selection.reason = "proxy_resource_wait_timeout"
        return selection

    def _c2c_s3_candidate_selection(self, candidates: list[dict[str, Any]], *, max_candidates: int) -> dict[str, Any]:
        candidates = [copy.deepcopy(candidate) for candidate in candidates if isinstance(candidate, dict)]
        manifest_path = self.context.project_root / "plan" / "code_patches" / "patch_manifest.json"
        manifest = read_json(manifest_path, default={}) if manifest_path.exists() else {}
        selected_candidate_id = str(manifest.get("selected_candidate_id") or "").strip() if isinstance(manifest, dict) else ""
        report = {
            "schema_version": "c2c_s3_candidate_selection_v1",
            "created_at": now_utc(),
            "lock_source": "plan/code_patches/patch_manifest.json" if manifest_path.exists() else "plan.candidate_ideas",
            "lock_policy": "s2_5_artifacts_are_execution_truth; trial_spec.json supplies experiment protocol only",
            "patch_manifest": _c2c_selection_artifact_lock(self.context.project_root, "plan/code_patches/patch_manifest.json"),
            "manifest_status": manifest.get("status") if isinstance(manifest, dict) else None,
            "selected_candidate_id": selected_candidate_id or None,
            "candidate_ids_before_lock": [candidate.get("id") for candidate in candidates if candidate.get("id")],
            "skipped_candidate_ids": [],
            "selected_candidate_found_in_plan": False,
            "selected_candidate_recovered_from_manifest": False,
        }
        if not selected_candidate_id:
            selected = candidates[: max(0, max_candidates)]
            report.update(
                {
                    "mode": "plan_candidate_budget",
                    "executed_candidate_ids": [candidate.get("id") for candidate in selected if candidate.get("id")],
                    "reason": "No selected_candidate_id in patch_manifest; using existing plan candidate budget.",
                }
            )
            return {"candidates": selected, "report": report}

        selected_entry = _c2c_manifest_selected_patch_entry(manifest, selected_candidate_id)
        selected_candidate = next((candidate for candidate in candidates if str(candidate.get("id") or "") == selected_candidate_id), None)
        if selected_candidate is not None:
            report["selected_candidate_found_in_plan"] = True
            selected = self._c2c_candidate_with_locked_patch(selected_candidate, selected_entry)
        else:
            selected = self._c2c_candidate_from_locked_patch(selected_entry, selected_candidate_id)
            report["selected_candidate_recovered_from_manifest"] = bool(selected)

        if not selected:
            report.update(
                {
                    "mode": "manifest_lock_failed",
                    "executed_candidate_ids": [],
                    "reason": f"patch_manifest selected_candidate_id={selected_candidate_id} but no matching plan candidate or selected patch entry was found.",
                }
            )
            return {"candidates": [], "report": report}

        report.update(
            {
                "mode": "patch_manifest_selected",
                "executed_candidate_ids": [selected.get("id")],
                "skipped_candidate_ids": [
                    candidate.get("id")
                    for candidate in candidates
                    if candidate.get("id") and str(candidate.get("id")) != selected_candidate_id
                ],
                "selected_patch_json": (selected.get("code_patch") or {}).get("patch_json") if isinstance(selected.get("code_patch"), dict) else None,
                "selected_patch_validation": (selected.get("code_patch") or {}).get("validation") if isinstance(selected.get("code_patch"), dict) else None,
                "selected_patched_repo_snapshot": (
                    ((selected.get("code_patch") or {}).get("patched_repo_snapshot") or {}).get("path")
                    if isinstance((selected.get("code_patch") or {}).get("patched_repo_snapshot"), dict)
                    else None
                ),
                "selected_patch": _c2c_selection_artifact_lock(
                    self.context.project_root,
                    ((selected.get("code_patch") or {}).get("patch_json") if isinstance(selected.get("code_patch"), dict) else None),
                ),
                "selected_patched_repo_snapshot_lock": _c2c_selection_artifact_lock(
                    self.context.project_root,
                    (
                        ((selected.get("code_patch") or {}).get("patched_repo_snapshot") or {}).get("manifest")
                        if isinstance((selected.get("code_patch") or {}).get("patched_repo_snapshot"), dict)
                        else None
                    ),
                ),
                "selected_implementation_contract": _c2c_selection_artifact_lock(
                    self.context.project_root,
                    ((selected.get("code_patch") or {}).get("implementation_contract") if isinstance(selected.get("code_patch"), dict) else None),
                ),
                "selected_patch_gate_report": _c2c_selection_artifact_lock(
                    self.context.project_root,
                    "plan/code_patches/patch_gate_report.json",
                ),
                "selected_planner_gate_report": _c2c_selection_artifact_lock(
                    self.context.project_root,
                    "plan/s2_planner/planner_gate_report.json",
                ),
                "selected_variant_scorecard": _c2c_selection_artifact_lock(
                    self.context.project_root,
                    "plan/s2_planner/variant_scorecard.json",
                ),
                "selected_patch_status": (selected.get("code_patch") or {}).get("status") if isinstance(selected.get("code_patch"), dict) else None,
                "reason": "S3 is locked to the S2.5 patch_manifest selected candidate.",
            }
        )
        return {"candidates": [selected], "report": report}

    @staticmethod
    def _c2c_candidate_with_locked_patch(candidate: dict[str, Any], selected_entry: dict[str, Any]) -> dict[str, Any]:
        selected = copy.deepcopy(candidate)
        if selected_entry:
            selected["code_patch"] = _c2c_code_patch_from_manifest_entry(selected_entry)
            selected["id"] = selected.get("id") or selected_entry.get("candidate_id") or selected_entry.get("id")
            selected["title"] = selected.get("title") or selected_entry.get("title")
            if selected_entry.get("variant_fingerprint"):
                selected["variant_fingerprint"] = selected_entry.get("variant_fingerprint")
            if isinstance(selected_entry.get("s2_variant"), dict) and selected_entry.get("s2_variant"):
                selected["s2_variant"] = selected_entry.get("s2_variant")
        selected["selected"] = True
        return selected

    def _c2c_candidate_from_locked_patch(self, selected_entry: dict[str, Any], selected_candidate_id: str) -> dict[str, Any] | None:
        if not selected_entry:
            return None
        candidate = {
            "id": selected_candidate_id,
            "title": selected_entry.get("title") or selected_candidate_id,
            "selected": True,
            "code_patch": _c2c_code_patch_from_manifest_entry(selected_entry),
        }
        if selected_entry.get("variant_fingerprint"):
            candidate["variant_fingerprint"] = selected_entry.get("variant_fingerprint")
        if isinstance(selected_entry.get("s2_variant"), dict) and selected_entry.get("s2_variant"):
            candidate["s2_variant"] = selected_entry.get("s2_variant")

        contract: dict[str, Any] = {}
        contract_path = selected_entry.get("implementation_contract") or candidate["code_patch"].get("implementation_contract")
        if contract_path:
            contract = read_json(self.context.project_root / str(contract_path), default={}) if isinstance(contract_path, str) else {}
        if not contract:
            patch_json_path = selected_entry.get("patch_json") or candidate["code_patch"].get("patch_json")
            if patch_json_path:
                patch_payload = read_json(self.context.project_root / str(patch_json_path), default={}) or {}
                if isinstance(patch_payload, dict):
                    contract = patch_payload.get("implementation_contract") if isinstance(patch_payload.get("implementation_contract"), dict) else {}
        if isinstance(contract, dict) and contract:
            mechanism = contract.get("mechanism_contract") if isinstance(contract.get("mechanism_contract"), dict) else {}
            experiment_contract = contract.get("experiment_contract") if isinstance(contract.get("experiment_contract"), dict) else {}
            candidate["hypothesis"] = contract.get("hypothesis") or candidate.get("hypothesis")
            candidate["mechanism_type"] = mechanism.get("mechanism_type") or candidate.get("mechanism_type")
            ablation_switch = experiment_contract.get("ablation_switch") or mechanism.get("ablation_switch")
            candidate["experiment_contract"] = {
                "primary_metric": experiment_contract.get("primary_metric"),
                "baseline": experiment_contract.get("baseline"),
                "config_overrides": experiment_contract.get("config_overrides") or {},
                "verification_commands": experiment_contract.get("verification_commands"),
                "ablation_switch": ablation_switch,
                "coverage_diagnostics": experiment_contract.get("coverage_diagnostics") or mechanism.get("coverage_diagnostics") or {},
                "matched_coverage_ablation": experiment_contract.get("matched_coverage_ablation") or mechanism.get("matched_coverage_ablation") or {},
            }
            if ablation_switch:
                candidate["ablation_plan"] = {"switch": ablation_switch}
            if mechanism.get("expected_signature"):
                candidate["expected_signature"] = mechanism.get("expected_signature")
        return candidate

    def _append_c2c_iteration_history(self, payload: dict[str, Any]) -> dict[str, Any]:
        history_path = self.context.project_root / "experiment" / "results" / "c2c_iteration_history.json"
        history = read_json(
            history_path,
            default={
                "schema_version": "c2c_iteration_history_v1",
                "project_id": self.context.project_root.name,
                "iterations": [],
            },
        )
        iterations = [
            item
            for item in history.get("iterations", [])
            if isinstance(item, dict) and int(item.get("iteration") or -1) != self._registry_iteration()
        ]
        acceptance = payload.get("acceptance") or {}
        best = payload.get("best_candidate") or {}
        best_proxy = payload.get("best_proxy_candidate") or {}
        best_metrics = best.get("metrics") or {}
        best_proxy_screen = best_proxy.get("proxy_screen") or {}
        best_proxy_metrics = best_proxy_screen.get("metrics") or {}
        entry = {
            "timestamp": now_utc(),
            "iteration": self._registry_iteration(),
            "accepted": bool(acceptance.get("passed")),
            "acceptance": {
                "passed": acceptance.get("passed"),
                "reason": acceptance.get("reason"),
                "baseline_mean": acceptance.get("baseline_mean"),
                "best_mean": acceptance.get("best_mean"),
                "delta": acceptance.get("delta"),
                "proxy_best_mean": acceptance.get("proxy_best_mean"),
                "proxy_delta": acceptance.get("proxy_delta"),
                "proxy_score": acceptance.get("proxy_score"),
                "proxy_worst_dataset_regression": acceptance.get("proxy_worst_dataset_regression"),
                "worst_dataset_regression": acceptance.get("worst_dataset_regression"),
                "min_delta_to_pass": acceptance.get("min_delta_to_pass"),
                "max_dataset_regression": acceptance.get("max_dataset_regression"),
            },
            "best_candidate": {
                "id": best.get("id"),
                "title": best.get("title"),
                "decision": best.get("decision"),
                "metrics": best_metrics,
                "delta_vs_baseline": best.get("delta_vs_baseline"),
                "dataset_regressions": best.get("dataset_regressions") or {},
                "worst_dataset_regression": best.get("worst_dataset_regression"),
            },
            "best_proxy_candidate": {
                "id": best_proxy.get("id"),
                "title": best_proxy.get("title"),
                "decision": best_proxy.get("decision"),
                "proxy_metrics": best_proxy_metrics,
                "proxy_mean": best_proxy_metrics.get("mean"),
                "proxy_delta_vs_baseline": best_proxy_screen.get("proxy_delta_vs_baseline"),
                "proxy_score": best_proxy_screen.get("proxy_score"),
                "proxy_worst_dataset_regression": best_proxy_screen.get("proxy_worst_dataset_regression"),
                "proxy_dataset_deltas": best_proxy_screen.get("proxy_dataset_deltas") or {},
            },
            "candidate_count": len(payload.get("candidate_results") or []),
            "candidate_ids": [
                item.get("id")
                for item in payload.get("candidate_results") or []
                if isinstance(item, dict) and item.get("id")
            ],
        }
        iterations.append(entry)
        iterations.sort(key=lambda item: int(item.get("iteration") or 0))
        best_entries = [
            item
            for item in iterations
            if ((item.get("best_candidate") or {}).get("metrics") or {}).get("mean") is not None
        ]
        best_proxy_entries = [
            item
            for item in iterations
            if ((item.get("best_proxy_candidate") or {}).get("proxy_metrics") or {}).get("mean") is not None
        ]
        best_entry = max(
            best_entries,
            key=lambda item: float(((item.get("best_candidate") or {}).get("metrics") or {}).get("mean")),
            default=None,
        )
        best_proxy_entry = max(
            best_proxy_entries,
            key=lambda item: float(((item.get("best_proxy_candidate") or {}).get("proxy_metrics") or {}).get("mean")),
            default=None,
        )
        consecutive_not_viable = 0
        for item in reversed(iterations):
            if item.get("accepted"):
                break
            consecutive_not_viable += 1
        history = {
            "schema_version": "c2c_iteration_history_v1",
            "project_id": self.context.project_root.name,
            "updated_at": now_utc(),
            "iterations": iterations,
            "iteration_count": len(iterations),
            "best_iteration": best_entry.get("iteration") if best_entry else None,
            "best_candidate_id": (best_entry.get("best_candidate") or {}).get("id") if best_entry else None,
            "best_mean_so_far": ((best_entry.get("best_candidate") or {}).get("metrics") or {}).get("mean") if best_entry else None,
            "best_delta_so_far": (best_entry.get("best_candidate") or {}).get("delta_vs_baseline") if best_entry else None,
            "best_proxy_iteration": best_proxy_entry.get("iteration") if best_proxy_entry else None,
            "best_proxy_candidate_id": (best_proxy_entry.get("best_proxy_candidate") or {}).get("id") if best_proxy_entry else None,
            "best_proxy_mean_so_far": ((best_proxy_entry.get("best_proxy_candidate") or {}).get("proxy_metrics") or {}).get("mean") if best_proxy_entry else None,
            "best_proxy_delta_so_far": (best_proxy_entry.get("best_proxy_candidate") or {}).get("proxy_delta_vs_baseline") if best_proxy_entry else None,
            "consecutive_not_viable": consecutive_not_viable,
        }
        write_json(history_path, history)
        return history

    def _append_c2c_proxy_calibration(self, payload: dict[str, Any]) -> dict[str, Any]:
        calibration_path = self.context.project_root / "experiment" / "results" / "proxy_calibration.json"
        calibration = read_json(
            calibration_path,
            default={
                "schema_version": "c2c_proxy_calibration_v1",
                "project_id": self.context.project_root.name,
                "iterations": [],
            },
        )
        if not isinstance(calibration, dict):
            calibration = {"schema_version": "c2c_proxy_calibration_v1", "project_id": self.context.project_root.name, "iterations": []}
        iterations = [
            item
            for item in calibration.get("iterations", [])
            if isinstance(item, dict) and int(item.get("iteration") or -1) != self._registry_iteration()
        ]
        current = _c2c_proxy_calibration_iteration(payload, iteration=self._registry_iteration())
        if current.get("candidate_count"):
            iterations.append(current)
        iterations.sort(key=lambda item: int(item.get("iteration") or 0))
        calibration = {
            "schema_version": "c2c_proxy_calibration_v1",
            "project_id": self.context.project_root.name,
            "updated_at": now_utc(),
            "iterations": iterations[-50:],
            "summary": _c2c_proxy_calibration_summary(iterations[-50:]),
            "current_iteration": current,
        }
        write_json(calibration_path, calibration)
        return calibration

    def _write_c2c_proxy_policy_contracts(
        self,
        *,
        plan: dict[str, Any] | None = None,
        execution: dict[str, Any] | None = None,
        candidate: dict[str, Any] | None = None,
        run_spec: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        main_payload: dict[str, Any] | None = None,
        include_baseline: bool = True,
    ) -> dict[str, Any]:
        cfg = config or self.context.config
        proxy_cfg = c2c_proxy_screen_config(cfg)
        if not proxy_cfg.get("enabled", False):
            return {}
        results_dir = self.context.project_root / "experiment" / "results"
        direction_fingerprint = read_json(self.context.project_root / "literature" / "c2c" / "direction_fingerprint.json", default={}) or {}
        variant_scorecard = read_json(self.context.project_root / "plan" / "s2_planner" / "variant_scorecard.json", default={}) or {}
        next_variant = read_json(self.context.project_root / "plan" / "s2_planner" / "variant.json", default={}) or {}
        patch_gate_report = read_json(self.context.project_root / "plan" / "code_patches" / "patch_gate_report.json", default={}) or {}
        calibration_policy = build_c2c_proxy_calibration_policy(
            project_root=self.context.project_root,
            config=cfg,
            direction_fingerprint=direction_fingerprint,
        )
        effective_policy = build_c2c_effective_proxy_policy(
            static_proxy_config=proxy_cfg,
            calibration_policy=calibration_policy,
            variant_scorecard=variant_scorecard,
            next_variant=next_variant,
            patch_gate_report=patch_gate_report,
            direction_fingerprint=direction_fingerprint,
        )
        policy_sources = [
            rel
            for rel in [
                "literature/c2c/direction_fingerprint.json",
                "plan/s2_planner/variant_scorecard.json",
                "plan/variant.json",
                "plan/code_patches/patch_gate_report.json",
            ]
            if (self.context.project_root / rel).exists()
        ]
        self._write_c2c_experiment_json_artifact(
            "results/c2c_proxy_calibration_policy.json",
            calibration_policy,
            artifact_type="c2c_proxy_calibration_policy",
            summary="Deterministic C2C proxy calibration policy for S3 routing",
            source_paths=policy_sources,
        )
        self._write_c2c_experiment_json_artifact(
            "results/c2c_effective_proxy_policy.json",
            effective_policy,
            artifact_type="c2c_effective_proxy_policy",
            summary="Effective C2C proxy policy after calibration adjustments",
            source_paths=policy_sources + ["experiment/results/c2c_proxy_calibration_policy.json"],
        )
        baseline_contracts = (
            self._write_c2c_proxy_baseline_contracts(
                plan=plan,
                execution=execution,
                candidate=candidate,
                run_spec=run_spec,
                config=cfg,
                proxy_cfg=proxy_cfg,
            )
            if include_baseline
            else {}
        )
        if main_payload is not None:
            main_payload["proxy_calibration_policy"] = {
                "path": "experiment/results/c2c_proxy_calibration_policy.json",
                "policy_hash": calibration_policy.get("policy_hash"),
                "adjustments_active": calibration_policy.get("adjustments_active"),
            }
            main_payload["effective_proxy_policy"] = {
                "path": "experiment/results/c2c_effective_proxy_policy.json",
                "policy_hash": effective_policy.get("policy_hash"),
                "effective_policy": effective_policy.get("effective_policy"),
            }
        return {
            "calibration_policy": calibration_policy,
            "effective_proxy_policy": effective_policy,
            **baseline_contracts,
        }

    def _write_c2c_proxy_baseline_contracts(
        self,
        *,
        plan: dict[str, Any] | None = None,
        execution: dict[str, Any] | None = None,
        candidate: dict[str, Any] | None = None,
        run_spec: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        proxy_cfg: dict[str, Any] | None = None,
        existing_fingerprint: dict[str, Any] | None = None,
        baseline_cache_exists: bool | None = None,
    ) -> dict[str, Any]:
        cfg = config or self.context.config
        proxy_cfg = proxy_cfg or c2c_proxy_screen_config(cfg)
        fingerprint_cfg = proxy_cfg.get("baseline_fingerprint") if isinstance(proxy_cfg.get("baseline_fingerprint"), dict) else {}
        if not fingerprint_cfg.get("enabled", True):
            return {}
        results_dir = self.context.project_root / "experiment" / "results"
        fingerprint_path = results_dir / "c2c_proxy_baseline_fingerprint.json"
        previous = existing_fingerprint
        if previous is None and fingerprint_path.exists():
            previous = read_json(fingerprint_path, default={}) or {}
        baseline_path = self.context.project_root / str(proxy_cfg.get("baseline_cache_path") or "experiment/results/c2c_proxy_baseline.json")
        fingerprint = build_c2c_proxy_baseline_fingerprint(
            project_root=self.context.project_root,
            config=cfg,
            plan=plan,
            execution=execution,
            proxy_config=proxy_cfg,
            run_spec=run_spec,
            candidate=candidate,
        )
        cache_report = build_c2c_proxy_cache_report(
            expected_fingerprint=fingerprint,
            actual_fingerprint=previous if isinstance(previous, dict) else {},
            baseline_cache_path=baseline_path,
            baseline_cache_exists=baseline_cache_exists,
            require_cache_fingerprint_match=bool(fingerprint_cfg.get("require_cache_fingerprint_match", True)),
        )
        baseline_sources = [
            rel
            for rel in [
                "plan/trial_spec.json",
                "plan/variant.json",
                "plan/code_patches/patch_manifest.json",
                "plan/code_patches/implementation_contract.json",
                "plan/code_patches/selected_patch.json",
            ]
            if (self.context.project_root / rel).exists()
        ]
        self._write_c2c_experiment_json_artifact(
            "results/c2c_proxy_baseline_fingerprint.json",
            fingerprint,
            artifact_type="c2c_proxy_baseline_fingerprint",
            summary="Fingerprint used to decide whether the C2C paired proxy baseline cache is reusable",
            source_paths=baseline_sources,
        )
        self._write_c2c_experiment_json_artifact(
            "results/c2c_proxy_cache_report.json",
            cache_report,
            artifact_type="c2c_proxy_cache_report",
            summary="C2C proxy baseline cache reuse, rerun, or invalidation decision",
            source_paths=baseline_sources + ["experiment/results/c2c_proxy_baseline_fingerprint.json"],
        )
        return {"baseline_fingerprint": fingerprint, "cache_report": cache_report}

    def _write_c2c_proxy_decision_contracts(
        self,
        *,
        candidate: dict[str, Any],
        proxy_screen: dict[str, Any],
        run_state: dict[str, Any],
        run_spec: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(config)
        results_dir = self.context.project_root / "experiment" / "results"
        contracts = self._write_c2c_proxy_policy_contracts(candidate=candidate, run_spec=run_spec, config=config, include_baseline=False)
        baseline_fingerprint = contracts.get("baseline_fingerprint") or read_json(results_dir / "c2c_proxy_baseline_fingerprint.json", default={}) or {}
        effective_policy = contracts.get("effective_proxy_policy") or read_json(results_dir / "c2c_effective_proxy_policy.json", default={}) or {}
        calibration_policy = contracts.get("calibration_policy") or read_json(results_dir / "c2c_proxy_calibration_policy.json", default={}) or {}
        variant_scorecard = read_json(self.context.project_root / "plan" / "s2_planner" / "variant_scorecard.json", default={}) or {}
        patch_gate_report = read_json(self.context.project_root / "plan" / "code_patches" / "patch_gate_report.json", default={}) or {}
        planner_gate_report = read_json(self.context.project_root / "plan" / "s2_planner" / "planner_gate_report.json", default={}) or {}
        novelty_audit = read_json(self.context.project_root / "literature" / "novelty_audit.json", default={}) or {}
        if isinstance(run_state.get("activation_smoke"), dict):
            proxy_screen["activation_smoke"] = run_state["activation_smoke"]
        worthiness = None
        worth_cfg = proxy_cfg.get("full_s3_worthiness") if isinstance(proxy_cfg.get("full_s3_worthiness"), dict) else {}
        if worth_cfg.get("enabled", True) and proxy_screen.get("status") == "passed":
            worthiness = build_c2c_full_s3_worthiness_score(
                candidate=candidate,
                proxy_screen=proxy_screen,
                effective_proxy_policy=effective_policy,
                variant_scorecard=variant_scorecard,
                patch_gate_report=patch_gate_report,
                novelty_audit=novelty_audit,
                calibration_policy=calibration_policy,
                config=config,
                neutral_proxy_budget_remaining=True,
            )
            self._write_c2c_experiment_json_artifact(
                "results/c2c_full_s3_worthiness.json",
                worthiness,
                artifact_type="c2c_full_s3_worthiness",
                summary="Deterministic worthiness score for spending full S3 budget on a neutral proxy",
                source_paths=[
                    rel
                    for rel in [
                        "experiment/results/c2c_effective_proxy_policy.json",
                        "plan/s2_planner/variant_scorecard.json",
                        "plan/code_patches/patch_gate_report.json",
                        "literature/novelty_audit.json",
                    ]
                    if (self.context.project_root / rel).exists()
                ],
            )
        decision_report = build_c2c_proxy_decision_report(
            candidate=candidate,
            proxy_screen=proxy_screen,
            baseline_fingerprint=baseline_fingerprint,
            effective_proxy_policy=effective_policy,
            patch_gate_report=patch_gate_report,
            planner_gate_report=planner_gate_report,
            variant_scorecard=variant_scorecard,
            full_s3_worthiness=worthiness,
        )
        self._write_c2c_experiment_json_artifact(
            "results/c2c_proxy_decision_report.json",
            decision_report,
            artifact_type="c2c_proxy_decision_report",
            summary="Source-of-truth C2C proxy routing decision for S3",
            source_paths=[
                rel
                for rel in [
                    "experiment/results/c2c_proxy_baseline_fingerprint.json",
                    "experiment/results/c2c_effective_proxy_policy.json",
                    "experiment/results/c2c_full_s3_worthiness.json",
                    "plan/s2_planner/planner_gate_report.json",
                    "plan/s2_planner/variant_scorecard.json",
                    "plan/code_patches/patch_gate_report.json",
                ]
                if (self.context.project_root / rel).exists()
            ],
        )
        proxy_screen.setdefault("artifact_paths", {})
        proxy_screen["artifact_paths"].update(
            {
                "proxy_baseline_fingerprint": "experiment/results/c2c_proxy_baseline_fingerprint.json",
                "proxy_cache_report": "experiment/results/c2c_proxy_cache_report.json",
                "effective_proxy_policy": "experiment/results/c2c_effective_proxy_policy.json",
                "proxy_decision_report": "experiment/results/c2c_proxy_decision_report.json",
                "proxy_calibration_policy": "experiment/results/c2c_proxy_calibration_policy.json",
            }
        )
        if worthiness:
            proxy_screen["artifact_paths"]["full_s3_worthiness"] = "experiment/results/c2c_full_s3_worthiness.json"
            proxy_screen["full_s3_worthiness"] = {
                "score": worthiness.get("score"),
                "threshold": worthiness.get("threshold"),
                "decision": worthiness.get("decision"),
            }
        proxy_screen["proxy_decision_report"] = {
            "decision": decision_report.get("decision"),
            "route_hint": decision_report.get("route_hint"),
            "failure_class": decision_report.get("failure_class"),
            "path": "experiment/results/c2c_proxy_decision_report.json",
        }
        run_state["proxy_screen"] = proxy_screen
        run_state["proxy_decision_report"] = decision_report
        return decision_report

    def _write_c2c_experiment_json_artifact(
        self,
        relative_path: str,
        payload: Any,
        *,
        artifact_type: str,
        summary: str,
        source_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.context.artifacts.write_json(
            self.stage_key,
            relative_path,
            payload,
            artifact_type=artifact_type,
            summary=summary,
            source_paths=source_paths or [],
        )

    @staticmethod
    def _snapshot_c2c_core_files(adapter: C2CAdapter) -> dict[str, str]:
        return {key: value["content"] for key, value in ExperimentAgent._snapshot_c2c_repo_state(adapter).items() if value.get("existed")}

    @staticmethod
    def _restore_c2c_core_files(adapter: C2CAdapter, originals: dict[str, str]) -> None:
        for rel_path, content in originals.items():
            path = adapter.repo_root / rel_path
            path.write_text(content, encoding="utf-8")

    @staticmethod
    def _snapshot_c2c_repo_state(adapter: C2CAdapter) -> dict[str, dict[str, Any]]:
        policy = DynamicEditPolicy.from_config(adapter.config.get("code_patch", {}).get("dynamic_whitelist") or {})
        originals: dict[str, dict[str, Any]] = {}
        for path in adapter.repo_root.rglob("*"):
            if not path.is_file():
                continue
            rel_path = path.relative_to(adapter.repo_root).as_posix()
            if not policy.allowed(rel_path, repo_root=adapter.repo_root):
                continue
            originals[rel_path] = {"existed": True, "content": path.read_text(encoding="utf-8", errors="ignore")}
        return originals

    @staticmethod
    def _restore_c2c_repo_state(adapter: C2CAdapter, originals: dict[str, dict[str, Any]]) -> None:
        policy = DynamicEditPolicy.from_config(adapter.config.get("code_patch", {}).get("dynamic_whitelist") or {})
        for path in sorted(adapter.repo_root.rglob("*"), reverse=True):
            if not path.is_file():
                continue
            rel_path = path.relative_to(adapter.repo_root).as_posix()
            if policy.allowed(rel_path, repo_root=adapter.repo_root) and rel_path not in originals:
                path.unlink()
        for rel_path, item in originals.items():
            target = adapter.repo_root / rel_path
            ensure_dir(target.parent)
            target.write_text(str(item.get("content") or ""), encoding="utf-8")

    def _run_single_c2c_candidate(
        self,
        *,
        adapter: C2CAdapter,
        candidate: dict[str, Any],
        index: int,
        simulate: bool,
        baseline_mean: float,
        min_delta: float,
        max_regression: float,
        gpu_selection: Any,
        proxy_gpu_selection: Any,
    ) -> dict[str, Any]:
        bootstrap_proxy_only = bootstrap_proxy_only_enabled(self.context.config)
        patch = self._load_frozen_c2c_patch(candidate)
        original_repo_root = adapter.repo_root
        execution_repo = self._prepare_c2c_execution_repo(candidate, adapter, patch)
        if execution_repo.get("status") == "ok":
            adapter = C2CAdapter(
                self.context.project_root,
                _config_with_c2c_snapshot_path(self.context.config, execution_repo["repo_root"]),
            )
        patch_result = self._apply_frozen_c2c_patch(candidate, adapter, patch, execution_repo=execution_repo)
        code_snapshot = (
            archive_patched_code_snapshot(self.context.artifacts, adapter, candidate, patch_result)
            if patch_result.get("status") in {"applied", "snapshot_applied"}
            else {"status": "skipped"}
        )
        run_spec = adapter.materialize_candidate_configs(candidate, gpu_selection, proxy_gpu_selection=proxy_gpu_selection)
        self._write_c2c_proxy_policy_contracts(candidate=candidate, run_spec=run_spec, config=adapter.config)
        output_pollution_before = _c2c_original_output_state(original_repo_root, run_spec)
        execution_repo_audit = _c2c_execution_repo_path_audit(
            original_repo_root=original_repo_root,
            execution_repo=execution_repo,
            run_spec=run_spec,
        )
        has_executable_change = bool(patch_result.get("changed_files") or run_spec.get("has_executable_change"))
        patch_fingerprint = self._c2c_patch_fingerprint(adapter, patch_result, run_spec)
        reusable_state = self._load_reusable_c2c_proxy_state(run_spec, patch_fingerprint)
        cached_proxy_rejudge_action = None
        if reusable_state:
            cached_proxy_screen = _normalize_c2c_proxy_screen_artifacts(
                reusable_state.get("proxy_screen") or {},
                full_baseline=adapter.baseline,
                run_spec=run_spec,
            )
            current_decision = self._c2c_rejudge_cached_proxy_screen(
                cached_proxy_screen,
                baseline=adapter.baseline,
                proxy_cfg=c2c_proxy_screen_config(self.context.config),
            )
            if current_decision and current_decision.get("status") == "passed":
                cached_proxy_rejudge_action = {
                    "action": "ignore_cached_proxy_block_after_threshold_rejudge",
                    "status": "ok",
                    "previous_status": cached_proxy_screen.get("status"),
                    "previous_reason": cached_proxy_screen.get("reason"),
                    "current_status": current_decision.get("status"),
                    "current_reason": current_decision.get("reason"),
                    "soft_flags": current_decision.get("soft_flags") or [],
                }
                reusable_state = None
            else:
                if current_decision:
                    cached_proxy_screen.update(current_decision)
                    cached_proxy_screen["cached_proxy_rejudged"] = True
                reusable_state["proxy_screen"] = cached_proxy_screen
        run_state = {
            "candidate_id": candidate.get("id"),
            "run_id": run_spec["run_id"],
            "created_at": now_utc(),
            "preflight": None,
            "proxy_screen": None,
            "activation_smoke": None,
            "full_s3_readiness": None,
            "ablation": None,
            "train": None,
            "eval_by_dataset": {},
            "ablation_eval_by_dataset": {},
            "metrics": None,
            "ablation_metrics": None,
            "attempts": [],
            "recovery_actions": [],
            "frozen_hashes": run_spec.get("frozen_hashes", {}),
            "config_overrides": run_spec.get("config_overrides", {}),
            "has_executable_change": has_executable_change,
            "patch_fingerprint": patch_fingerprint,
            "code_snapshot": code_snapshot,
            "execution_repo": execution_repo,
            "execution_repo_audit": execution_repo_audit,
            "runtime_localization": run_spec.get("runtime_localization", {}),
        }
        logs = [
            {
                "candidate_id": candidate.get("id"),
                "event": "patch",
                "patch_summary": patch.get("summary", ""),
                "patch_result": patch_result,
                "code_snapshot": code_snapshot,
                "execution_repo": execution_repo,
                "execution_repo_audit": execution_repo_audit,
                "config_overrides": run_spec.get("config_overrides", {}),
                "has_executable_change": has_executable_change,
                "patch_fingerprint": patch_fingerprint,
                "frozen_hashes": run_spec.get("frozen_hashes", {}),
                "runtime_localization": run_spec.get("runtime_localization", {}),
            }
        ]
        def refresh_execution_repo_audit(phase: str) -> dict[str, Any]:
            audit = _c2c_execution_repo_output_audit(
                original_repo_root=original_repo_root,
                run_spec=run_spec,
                before_state=output_pollution_before,
                path_audit=execution_repo_audit,
                phase=phase,
            )
            run_state["execution_repo_audit"] = audit
            return audit

        def block_for_execution_repo_audit(phase: str) -> bool:
            nonlocal command_status
            audit = refresh_execution_repo_audit(phase)
            if audit.get("status") != "failed":
                return False
            reason = audit.get("reason") or "S3 output was written outside execution repo"
            command_status = "blocked"
            action = {
                "action": "block_original_snapshot_output_pollution",
                "status": "blocked",
                "phase": phase,
                "reason": reason,
                "polluted_files": (audit.get("output_pollution") or {}).get("added_files", [])[:20],
                "modified_files": (audit.get("output_pollution") or {}).get("modified_files", [])[:20],
            }
            run_state["recovery_actions"].append(action)
            logs.append({"candidate_id": candidate.get("id"), "event": "execution_repo_audit_failed", **action, "audit": audit})
            return True

        if cached_proxy_rejudge_action:
            run_state["recovery_actions"].append(cached_proxy_rejudge_action)
            logs.append({"candidate_id": candidate.get("id"), "event": "proxy_cache_rejudge", **cached_proxy_rejudge_action})
        command_status = "skipped"
        preflight = None
        if execution_repo_audit.get("status") == "failed":
            metrics = None
            command_status = "blocked"
            reason = execution_repo_audit.get("reason") or "S3 execution repo path audit failed"
            run_state["preflight"] = {"status": "blocked", "reason": reason}
            run_state["recovery_actions"].append({"action": "block_execution_repo_path_audit", "status": "blocked", "reason": reason})
            logs.append({"candidate_id": candidate.get("id"), "event": "blocked", "reason": reason, "execution_repo_audit": execution_repo_audit})
            self._save_c2c_run_state(run_spec, run_state)
        elif patch_result["status"] == "rejected":
            metrics = None
            command_status = "patch_rejected"
            self._save_c2c_run_state(run_spec, run_state)
        elif not simulate and not has_executable_change:
            metrics = None
            command_status = "blocked"
            reason = "candidate lacks frozen executable patch or config_overrides; refusing deterministic S3 no-op run"
            run_state["preflight"] = {"status": "blocked", "reason": reason}
            run_state["recovery_actions"].append({"action": "block_noop_candidate", "status": "blocked", "reason": reason})
            logs.append({"candidate_id": candidate.get("id"), "event": "blocked", "reason": reason})
            self._save_c2c_run_state(run_spec, run_state)
        elif not simulate and bool(c2c_proxy_screen_config(self.context.config).get("enabled", False)) and not getattr(proxy_gpu_selection, "selected_ids", []):
            metrics = None
            command_status = "blocked"
            reason = "cheap proxy resource wait timed out without an available GPU; resume after GPU resources are available"
            proxy_screen = {
                "enabled": True,
                "status": "resource_retry",
                "reason": reason,
                "repair_hint": "do not repair the S2.5 patch for this failure; rerun S3 when proxy GPU resources are available",
                "resource_retry": True,
                "failure_category": "s3_proxy_gpu_resource_retry",
                "gpu_selection": {
                    "selected_gpu_ids": list(getattr(proxy_gpu_selection, "selected_ids", []) or []),
                    "policy": getattr(proxy_gpu_selection, "policy", {}),
                    "snapshot": getattr(proxy_gpu_selection, "snapshot", []),
                    "reason": getattr(proxy_gpu_selection, "reason", ""),
                },
            }
            run_state["proxy_screen"] = proxy_screen
            run_state["preflight"] = {"status": "blocked", "reason": reason}
            run_state["recovery_actions"].append(
                {
                    "action": "pause_for_proxy_gpu_resources",
                    "status": "resource_retry",
                    "reason": reason,
                    "gpu_selection": proxy_screen["gpu_selection"],
                }
            )
            logs.append({"candidate_id": candidate.get("id"), "event": "proxy_gpu_resource_retry", "proxy_screen": proxy_screen})
            self._save_c2c_run_state(run_spec, run_state)
        elif simulate:
            metrics = adapter.write_mock_candidate_results(run_spec["run_id"], offset=max(min_delta + 0.15, 0.25) if index == 0 else -0.25)
            ablation = self._run_c2c_ablation_eval(
                adapter=adapter,
                candidate=candidate,
                run_spec=run_spec,
                gpu_selection=gpu_selection,
                retry_policy={"max_attempts": 1},
                simulate=True,
            )
            logs.append({"candidate_id": candidate.get("id"), "event": "mock_results", "metrics": metrics})
            logs.append({"candidate_id": candidate.get("id"), "event": "ablation", "status": ablation.get("status"), "ablation": ablation})
            command_status = "mocked"
            run_state["preflight"] = {"status": "skipped", "simulate": True}
            run_state["proxy_screen"] = {"enabled": False, "status": "skipped", "reason": "simulate mode"}
            run_state["metrics"] = metrics
            run_state["ablation"] = ablation
            run_state["ablation_metrics"] = ablation.get("metrics")
            self._save_c2c_run_state(run_spec, run_state)
        elif reusable_state:
            metrics = reusable_state.get("metrics")
            proxy_screen = _normalize_c2c_proxy_screen_artifacts(
                reusable_state.get("proxy_screen") or {},
                full_baseline=adapter.baseline,
                run_spec=run_spec,
            )
            reusable_state["proxy_screen"] = proxy_screen
            preflight = reusable_state.get("preflight")
            run_state.update(reusable_state)
            run_state["patch_fingerprint"] = patch_fingerprint
            run_state["code_snapshot"] = code_snapshot
            logs.append(
                {
                    "candidate_id": candidate.get("id"),
                    "event": "reuse_proxy_screen",
                    "status": proxy_screen.get("status"),
                    "reason": "existing run_state has complete proxy_screen for matching frozen hashes and patch fingerprint",
                }
            )
            if proxy_screen.get("status") in {"resource_retry", "blocked"}:
                command_status = "blocked"
            elif proxy_screen.get("status") == "rejected":
                command_status = "proxy_rejected"
            elif proxy_screen.get("status") == "repairable_proxy_risk":
                command_status = "proxy_repairable"
            else:
                command_status = str(reusable_state.get("command_status") or "partial")
            self._save_c2c_run_state(run_spec, run_state)
        else:
            metrics = None
            preflight = adapter.preflight(run_spec, gpu_selection)
            run_state["preflight"] = preflight
            run_state["recovery_actions"].extend(preflight.get("recovery_actions", []))
            logs.append({"candidate_id": candidate.get("id"), "event": "preflight", "status": preflight.get("status"), "path": str(run_spec["preflight_path"])})
            if preflight.get("status") == "blocked":
                command_status = "blocked"
                self._save_c2c_run_state(run_spec, run_state)
            else:
                log_path = self.context.project_root / "experiment" / "logs" / f"c2c_{run_spec['run_id']}_commands.json"
                ensure_dir(log_path.parent)
                retry_policy = {
                    "max_attempts": int(self.context.config.get("experiment", {}).get("self_heal", {}).get("max_attempts", 1) or 1)
                }
                step_runs = []
                proxy_enabled = bool(c2c_proxy_screen_config(self.context.config).get("enabled", False))
                for step_idx, command in enumerate(run_spec["commands"]["preflight"]):
                    result = self.runner.run_step(
                        name=f"preflight_command_{step_idx}",
                        command=command,
                        working_dir=adapter.repo_root,
                        retry_policy={"max_attempts": 1},
                    )
                    step_runs.append(result)
                    run_state["attempts"].append(result)
                    if result["status"] != "ok":
                        command_status = "failed"
                        break
                if command_status != "failed" and proxy_enabled:
                    baseline_proxy = self._ensure_c2c_proxy_baseline(
                        adapter,
                        run_spec,
                        proxy_gpu_selection,
                        retry_policy,
                        baseline_repo_state=None,
                    )
                    baseline_attempts = baseline_proxy.get("attempts") or []
                    step_runs.extend(baseline_attempts)
                    run_state["attempts"].extend(baseline_attempts)
                    if baseline_proxy.get("status") in {"failed", "blocked"}:
                        run_state["proxy_baseline"] = baseline_proxy
                        command_status = "blocked"
                        run_state["proxy_screen"] = {
                            "enabled": True,
                            "status": "baseline_blocked",
                            "reason": baseline_proxy.get("reason") or "proxy baseline could not be established",
                            "baseline_status": baseline_proxy.get("status"),
                            "baseline_failure": baseline_proxy.get("command_failure") or {},
                            "baseline_attempt_count": len(baseline_proxy.get("attempts") or []),
                            "patch_fingerprint": patch_fingerprint,
                        }
                        logs.append({"candidate_id": candidate.get("id"), "event": "proxy_baseline", "status": baseline_proxy.get("status"), "proxy_baseline": baseline_proxy})
                    elif baseline_proxy.get("status") in {"fallback", "cached", "ok"}:
                        run_state["proxy_baseline"] = baseline_proxy
                        logs.append({"candidate_id": candidate.get("id"), "event": "proxy_baseline", "status": baseline_proxy.get("status"), "proxy_baseline": baseline_proxy})
                    if command_status not in {"failed", "blocked"} and block_for_execution_repo_audit("proxy_baseline"):
                        metrics = None
                    if command_status == "blocked" and isinstance(run_state.get("proxy_screen"), dict):
                        decision_report = self._write_c2c_proxy_decision_contracts(
                            candidate=candidate,
                            proxy_screen=run_state["proxy_screen"],
                            run_state=run_state,
                            run_spec=run_spec,
                            config=adapter.config,
                        )
                        logs.append(
                            {
                                "candidate_id": candidate.get("id"),
                                "event": "proxy_decision",
                                "decision": decision_report.get("decision"),
                                "route_hint": decision_report.get("route_hint"),
                            }
                        )
                if command_status not in {"failed", "blocked"}:
                    if proxy_enabled:
                        proxy_screen = self._run_c2c_proxy_screen(
                            adapter=adapter,
                            candidate=candidate,
                            run_spec=run_spec,
                            patch_result=patch_result,
                            has_executable_change=has_executable_change,
                            baseline=adapter.baseline,
                        )
                    else:
                        proxy_screen = {"enabled": False, "status": "skipped", "reason": "proxy_screen disabled"}
                    proxy_screen["patch_fingerprint"] = patch_fingerprint
                    run_state["proxy_screen"] = proxy_screen
                    logs.append({"candidate_id": candidate.get("id"), "event": "proxy_screen", "status": proxy_screen.get("status"), "proxy_screen": proxy_screen})
                    proxy_attempts = proxy_screen.get("attempts") or []
                    step_runs.extend(proxy_attempts)
                    run_state["attempts"].extend(proxy_attempts)
                    if proxy_screen.get("status") in {"resource_retry", "blocked"}:
                        command_status = "blocked"
                    elif proxy_screen.get("status") == "failed":
                        command_status = "failed"
                    elif proxy_screen.get("status") == "rejected":
                        command_status = "proxy_rejected"
                    elif proxy_screen.get("status") == "repairable_proxy_risk":
                        command_status = "proxy_repairable"
                    if command_status not in {"failed", "proxy_rejected", "proxy_repairable", "blocked"} and block_for_execution_repo_audit("proxy_screen"):
                        metrics = None
                    if command_status in {"failed", "proxy_rejected", "proxy_repairable", "blocked"}:
                        decision_report = self._write_c2c_proxy_decision_contracts(
                            candidate=candidate,
                            proxy_screen=run_state.get("proxy_screen") if isinstance(run_state.get("proxy_screen"), dict) else proxy_screen,
                            run_state=run_state,
                            run_spec=run_spec,
                            config=adapter.config,
                        )
                        logs.append(
                            {
                                "candidate_id": candidate.get("id"),
                                "event": "proxy_decision",
                                "decision": decision_report.get("decision"),
                                "route_hint": decision_report.get("route_hint"),
                            }
                        )
                if proxy_enabled and command_status not in {"failed", "blocked", "proxy_rejected", "proxy_repairable"}:
                    activation_smoke = self._run_c2c_proxy_activation_smoke(
                        adapter=adapter,
                        candidate=candidate,
                            run_spec=run_spec,
                            gpu_selection=proxy_gpu_selection,
                        )
                    run_state["activation_smoke"] = activation_smoke
                    activation_attempts = activation_smoke.get("attempts") or []
                    step_runs.extend(activation_attempts)
                    run_state["attempts"].extend(activation_attempts)
                    logs.append(
                        {
                            "candidate_id": candidate.get("id"),
                            "event": "activation_smoke",
                            "status": activation_smoke.get("status"),
                            "activation_smoke": activation_smoke,
                        }
                    )
                    if activation_smoke.get("status") == "failed" and activation_smoke.get("hard_gate", True):
                        proxy_screen = run_state.get("proxy_screen") if isinstance(run_state.get("proxy_screen"), dict) else {}
                        reason = activation_smoke.get("reason") or "proxy activation smoke failed"
                        repair_hint = activation_smoke.get("repair_hint") or "repair S2.5 patch activation before full S3"
                        trace = activation_smoke.get("mechanism_trace") if isinstance(activation_smoke.get("mechanism_trace"), dict) else {}
                        if trace.get("status") == "wired":
                            if _c2c_neutral_proxy_full_s3_allowed(
                                proxy_screen,
                                c2c_proxy_screen_config(self.context.config),
                            ):
                                reason = "metric-neutral activation smoke allowed for exploratory full S3 because proxy is near-neutral and mechanism wiring is present"
                                comparison = activation_smoke.setdefault("comparison", {})
                                if isinstance(comparison, dict):
                                    comparison["mechanism_wired_metric_neutral"] = True
                                activation_smoke.update(
                                    {
                                        "status": "passed",
                                        "hard_gate_overridden": True,
                                        "reason": reason,
                                        "full_s3_allowed_reason": reason,
                                    }
                                )
                                proxy_screen.update(
                                    {
                                        "activation_smoke": activation_smoke,
                                        "activation_metric_neutral_allowed_for_full_s3": True,
                                        "neutral_proxy_policy": _c2c_neutral_proxy_policy_summary(
                                            proxy_screen,
                                            c2c_proxy_screen_config(self.context.config),
                                        ),
                                    }
                                )
                                run_state["activation_smoke"] = activation_smoke
                                run_state["proxy_screen"] = proxy_screen
                                run_state["recovery_actions"].append(
                                    {
                                        "action": "allow_metric_neutral_activation_for_full_s3",
                                        "status": "warning",
                                        "reason": reason,
                                        "neutral_proxy_policy": proxy_screen.get("neutral_proxy_policy"),
                                    }
                                )
                            else:
                                command_status = "proxy_repairable"
                                reason = "mechanism is wired into eval path but produced metric-neutral proxy activation smoke"
                                repair_hint = "repair proxy effect or dataset tradeoff; eval-path wiring is present, so do not spend this repair on ablation switch plumbing"
                                proxy_screen.update(
                                    {
                                        "status": "repairable_proxy_risk",
                                        "reason": reason,
                                        "repair_hint": repair_hint,
                                        "repair_route": "S2_plan",
                                        "repair_mode": "effect_first_proxy_repair",
                                        "activation_smoke": activation_smoke,
                                        "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                                            reason=reason,
                                            repair_hint=repair_hint,
                                            evidence={"activation_smoke": activation_smoke, "mechanism_trace": trace},
                                            patch_risk=(proxy_screen.get("patch_risk") or {}),
                                            source="proxy_activation_smoke",
                                        ),
                                    }
                                )
                                run_state["proxy_screen"] = proxy_screen
                        else:
                            command_status = "proxy_repairable"
                            proxy_screen.update(
                                {
                                    "status": "repairable_proxy_risk",
                                    "reason": reason,
                                    "repair_hint": repair_hint,
                                    "repair_route": "S2_plan",
                                    "repair_mode": "effect_first_proxy_repair",
                                    "activation_smoke": activation_smoke,
                                    "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                                        reason=reason,
                                        repair_hint=repair_hint,
                                        evidence={"activation_smoke": activation_smoke, "mechanism_trace": trace},
                                        patch_risk=(proxy_screen.get("patch_risk") or {}),
                                        source="proxy_activation_smoke",
                                    ),
                                }
                            )
                            run_state["proxy_screen"] = proxy_screen
                    if command_status not in {"failed", "proxy_rejected", "proxy_repairable", "blocked"} and block_for_execution_repo_audit("activation_smoke"):
                        metrics = None
                if proxy_enabled and command_status not in {"failed", "blocked"} and isinstance(run_state.get("proxy_screen"), dict) and not run_state.get("proxy_decision_report"):
                    proxy_screen = run_state["proxy_screen"]
                    decision_report = self._write_c2c_proxy_decision_contracts(
                        candidate=candidate,
                        proxy_screen=proxy_screen,
                        run_state=run_state,
                        run_spec=run_spec,
                        config=adapter.config,
                    )
                    logs.append(
                        {
                            "candidate_id": candidate.get("id"),
                            "event": "proxy_decision",
                            "decision": decision_report.get("decision"),
                            "route_hint": decision_report.get("route_hint"),
                            "failure_class": decision_report.get("failure_class"),
                        }
                    )
                    if decision_report.get("decision") == "blocked":
                        command_status = "blocked"
                        proxy_screen.update(
                            {
                                "status": "blocked",
                                "reason": decision_report.get("failure_class") or "proxy decision blocked full S3",
                                "repair_route": decision_report.get("route_hint"),
                            }
                        )
                    elif decision_report.get("decision") == "proxy_repairable":
                        command_status = "proxy_repairable"
                        proxy_screen.update(
                            {
                                "status": "repairable_proxy_risk",
                                "reason": decision_report.get("failure_class") or "proxy decision requires repair before full S3",
                                "repair_route": decision_report.get("route_hint"),
                                "repair_mode": decision_report.get("failure_class") or "proxy_decision_repair",
                            }
                        )
                    elif decision_report.get("decision") == "proxy_rejected":
                        command_status = "proxy_rejected"
                        proxy_screen.update(
                            {
                                "status": "rejected",
                                "reason": decision_report.get("failure_class") or "proxy decision rejected full S3",
                                "repair_route": decision_report.get("route_hint"),
                            }
                        )
                    run_state["proxy_screen"] = proxy_screen
                proxy_screen = run_state.get("proxy_screen") if isinstance(run_state.get("proxy_screen"), dict) else {}
                proxy_mean = _finite_proxy_mean(proxy_screen)
                if bootstrap_proxy_only and proxy_enabled and proxy_mean is not None and command_status not in {"failed", "blocked"}:
                    original_command_status = command_status
                    command_status = "bootstrap_proxy_complete"
                    run_state["bootstrap"] = {
                        "profile": "bootstrap",
                        "proxy_only": True,
                        "status": "proxy_reached",
                        "original_command_status": original_command_status,
                        "original_proxy_status": proxy_screen.get("status"),
                    }
                    logs.append(
                        {
                            "candidate_id": candidate.get("id"),
                            "event": "bootstrap_proxy_complete",
                            "proxy_mean": proxy_mean,
                            "original_command_status": original_command_status,
                        }
                    )
                if not bootstrap_proxy_only and proxy_enabled and command_status not in {"failed", "blocked", "proxy_rejected", "proxy_repairable"}:
                    readiness = self._record_c2c_full_s3_readiness(
                        candidate=candidate,
                        run_spec=run_spec,
                        patch_result=patch_result,
                        run_state=run_state,
                        baseline=adapter.baseline,
                        min_delta=min_delta,
                        max_regression=max_regression,
                    )
                    run_state["full_s3_readiness"] = readiness
                    logs.append(
                        {
                            "candidate_id": candidate.get("id"),
                            "event": "full_s3_readiness",
                            "status": readiness.get("status"),
                            "worth_full_train": (readiness.get("worth_full_train") or {}).get("decision"),
                            "full_train_allowed": readiness.get("full_train_allowed"),
                            "readiness_report_path": (readiness.get("artifact_paths") or {}).get("project_readiness_report"),
                        }
                    )
                    if readiness.get("full_train_allowed") is not True:
                        command_status = "proxy_repairable"
                        proxy_screen = run_state.get("proxy_screen") if isinstance(run_state.get("proxy_screen"), dict) else {}
                        reason = _c2c_full_s3_readiness_block_reason(readiness)
                        repair_hint = "repair S2.5 patch or proxy/eval wiring until full_s3_readiness.full_train_allowed=true before full S3"
                        proxy_screen.update(
                            {
                                "status": "repairable_proxy_risk",
                                "reason": reason,
                                "repair_hint": repair_hint,
                                "repair_route": "S2_plan",
                                "repair_mode": "full_s3_readiness_repair",
                                "full_s3_readiness": readiness,
                                "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                                    reason=reason,
                                    repair_hint=repair_hint,
                                    evidence={
                                        "full_s3_readiness": readiness,
                                        "eval_smoke": readiness.get("eval_smoke") if isinstance(readiness.get("eval_smoke"), dict) else {},
                                        "activation_smoke": readiness.get("activation_smoke") if isinstance(readiness.get("activation_smoke"), dict) else {},
                                    },
                                    patch_risk=((readiness.get("static_risk") or {}) if isinstance(readiness.get("static_risk"), dict) else {}),
                                    source="full_s3_readiness",
                                ),
                            }
                        )
                        run_state["proxy_screen"] = proxy_screen
                        run_state["recovery_actions"].append(
                            {
                                "action": "block_full_train_until_readiness",
                                "status": "blocked",
                                "reason": reason,
                                "readiness_report_path": (readiness.get("artifact_paths") or {}).get("project_readiness_report"),
                            }
                        )
                        logs.append(
                            {
                                "candidate_id": candidate.get("id"),
                                "event": "full_s3_readiness_blocked_train",
                                "status": "proxy_repairable",
                                "reason": reason,
                            }
                        )
                    if command_status == "proxy_repairable":
                        metrics = None
                if not bootstrap_proxy_only and command_status not in {"failed", "blocked", "proxy_rejected", "proxy_repairable"}:
                    if proxy_enabled:
                        full_s3_decision = {
                            "schema_version": "c2c_full_s3_decision_v1",
                            "created_at": now_utc(),
                            "candidate_id": candidate.get("id"),
                            "run_id": run_spec.get("run_id"),
                            "decision": "run_full_s3",
                            "proxy_decision": (run_state.get("proxy_decision_report") or {}).get("decision") if isinstance(run_state.get("proxy_decision_report"), dict) else None,
                            "proxy_route_hint": (run_state.get("proxy_decision_report") or {}).get("route_hint") if isinstance(run_state.get("proxy_decision_report"), dict) else None,
                            "full_s3_readiness": {
                                "status": (run_state.get("full_s3_readiness") or {}).get("status") if isinstance(run_state.get("full_s3_readiness"), dict) else None,
                                "full_train_allowed": (run_state.get("full_s3_readiness") or {}).get("full_train_allowed") if isinstance(run_state.get("full_s3_readiness"), dict) else None,
                            },
                            "source_artifacts": {
                                "proxy_decision_report": "experiment/results/c2c_proxy_decision_report.json",
                                "effective_proxy_policy": "experiment/results/c2c_effective_proxy_policy.json",
                                "full_s3_worthiness": "experiment/results/c2c_full_s3_worthiness.json",
                                "full_s3_readiness_report": "experiment/results/full_s3_readiness_report.json",
                            },
                        }
                        self._write_c2c_experiment_json_artifact(
                            "results/c2c_full_s3_decision.json",
                            full_s3_decision,
                            artifact_type="c2c_full_s3_decision",
                            summary="C2C decision to spend full S3 budget after proxy screening",
                            source_paths=[
                                rel
                                for rel in [
                                    "experiment/results/c2c_proxy_decision_report.json",
                                    "experiment/results/c2c_effective_proxy_policy.json",
                                    "experiment/results/c2c_full_s3_worthiness.json",
                                    "experiment/results/full_s3_readiness_report.json",
                                ]
                                if (self.context.project_root / rel).exists()
                            ],
                        )
                        run_state["full_s3_decision"] = full_s3_decision
                    train_result = self.runner.run_step(
                        name="train",
                        command=run_spec["commands"]["train"],
                        working_dir=adapter.repo_root,
                        retry_policy=retry_policy,
                    )
                    step_runs.append(train_result)
                    run_state["train"] = train_result
                    run_state["attempts"].append(train_result)
                    if train_result["status"] != "ok":
                        oom_action = self._c2c_oom_recovery_hint(train_result)
                        if oom_action and len(getattr(gpu_selection, "selected_ids", []) or []) > 1:
                            recovery_selection = self.runner.select_gpus(
                                {
                                    "gpu_ids": self.context.config.get("c2c", {}).get("small_loop", {}).get("gpu_ids", "auto"),
                                    "max_gpus": 1,
                                    "min_free_mb": self.context.config.get("experiment", {}).get("gpu_policy", {}).get("min_free_mb", 8192),
                                    "max_utilization_gpu": self.context.config.get("experiment", {}).get("gpu_policy", {}).get("max_utilization_gpu", 40),
                                    "respect_resource_filters": True,
                                }
                            )
                            recovery_gpu_ids = list(getattr(recovery_selection, "selected_ids", []) or []) or [gpu_selection.selected_ids[0]]
                            recovery_command = adapter._candidate_commands(run_spec["train_config"], run_spec["eval_configs"], recovery_gpu_ids)["train"]
                            recovery_result = self.runner.run_step(
                                name="train_recovery_reduced_concurrency",
                                command=recovery_command,
                                working_dir=adapter.repo_root,
                                retry_policy={"max_attempts": 1},
                            )
                            step_runs.append(recovery_result)
                            run_state["attempts"].append(recovery_result)
                            oom_action.update(
                                {
                                    "recovery_gpu_ids": recovery_gpu_ids,
                                    "recovery_status": recovery_result["status"],
                                    "recovery_gpu_selection": {
                                        "selected_gpu_ids": list(getattr(recovery_selection, "selected_ids", []) or []),
                                        "reason": getattr(recovery_selection, "reason", ""),
                                        "snapshot": getattr(recovery_selection, "snapshot", []),
                                    },
                                }
                            )
                            run_state["recovery_actions"].append(oom_action)
                            if recovery_result["status"] == "ok":
                                train_result = recovery_result
                                run_state["train"] = recovery_result
                        if oom_action and train_result["status"] != "ok":
                            recovery_gpu_ids = list(getattr(gpu_selection, "selected_ids", []) or [])
                            memory_safe = adapter.materialize_train_oom_recovery_config(run_spec, gpu_ids=recovery_gpu_ids)
                            action = {
                                "action": "retry_train_memory_safe_recipe",
                                "status": memory_safe.get("status"),
                                "reason": "detected CUDA OOM signature; retry with a memory-safe full S3 train recipe",
                                "recovery_gpu_ids": recovery_gpu_ids,
                                "memory_safe_train_config": str(memory_safe.get("train_config") or ""),
                                "config_changes": memory_safe.get("config_changes") or {},
                            }
                            if memory_safe.get("status") == "materialized":
                                recovery_result = self.runner.run_step(
                                    name="train_recovery_memory_safe",
                                    command=str(memory_safe["command"]),
                                    working_dir=adapter.repo_root,
                                    retry_policy={"max_attempts": 1},
                                )
                                step_runs.append(recovery_result)
                                run_state["attempts"].append(recovery_result)
                                action["recovery_status"] = recovery_result["status"]
                                if recovery_result["status"] == "ok":
                                    train_result = recovery_result
                                    run_state["train"] = recovery_result
                            else:
                                action["recovery_status"] = "skipped"
                            run_state["recovery_actions"].append(self._jsonable(action))
                        if train_result["status"] != "ok" and self._c2c_checkpoint_final_exists(run_spec):
                            command_status = "partial"
                            action = {"action": "skip_failed_train_with_existing_final_checkpoint", "status": "ok", "run_id": run_spec["run_id"]}
                            run_state["recovery_actions"].append(action)
                            logs.append({"candidate_id": candidate.get("id"), "event": "recovery", **action})
                        elif train_result["status"] != "ok":
                            command_status = "failed"
                    if command_status != "failed":
                        eval_commands = run_spec["commands"]["eval"]
                        eval_items = list(run_spec.get("eval_configs", {}).items())
                        for eval_idx, command in enumerate(eval_commands):
                            dataset = eval_items[eval_idx][0] if eval_idx < len(eval_items) else f"dataset_{eval_idx}"
                            result = self.runner.run_step(
                                name=f"eval_{dataset}",
                                command=command,
                                working_dir=adapter.repo_root,
                                retry_policy=retry_policy,
                            )
                            step_runs.append(result)
                            run_state["eval_by_dataset"][dataset] = result
                            run_state["attempts"].append(result)
                            if result["status"] != "ok":
                                command_status = "partial"
                    metrics = adapter.collect_candidate_metrics(run_spec["run_id"])
                    run_state["metrics"] = metrics
                    if metrics is not None and command_status not in {"failed", "blocked", "proxy_rejected", "proxy_repairable"}:
                        ablation = self._run_c2c_ablation_eval(
                            adapter=adapter,
                            candidate=candidate,
                            run_spec=run_spec,
                            gpu_selection=gpu_selection,
                            retry_policy=retry_policy,
                            simulate=False,
                        )
                        run_state["ablation"] = ablation
                        run_state["ablation_metrics"] = ablation.get("metrics")
                        run_state["ablation_eval_by_dataset"] = ablation.get("eval_by_dataset") or {}
                        run_state["attempts"].extend(ablation.get("attempts") or [])
                        step_runs.extend(ablation.get("attempts") or [])
                        logs.append({"candidate_id": candidate.get("id"), "event": "ablation", "status": ablation.get("status"), "ablation": ablation})
                        if ablation.get("status") == "partial" and command_status == "ok":
                            command_status = "partial"
                    if command_status not in {"failed", "blocked", "proxy_rejected", "proxy_repairable"} and block_for_execution_repo_audit("full_train_eval"):
                        metrics = None
                    if command_status == "skipped":
                        command_status = "ok"
                    if metrics is None and command_status not in {"failed", "blocked", "proxy_rejected", "proxy_repairable"}:
                        command_status = "partial"
                if command_status not in {"blocked"}:
                    refresh_execution_repo_audit("final")
                write_json(
                    log_path,
                    {
                        "status": command_status,
                        "runs": _compact_attempts(step_runs, stdout_chars=4000, stderr_chars=4000),
                        "full_log_note": "stdout/stderr are stored as bounded tails to keep artifacts parseable",
                    },
                )
                logs.append({"candidate_id": candidate.get("id"), "event": "commands", "log_path": str(log_path), "status": command_status})
                self._save_c2c_run_state(run_spec, run_state)
        final_proxy = run_state.get("proxy_screen") if isinstance(run_state.get("proxy_screen"), dict) else {}
        final_proxy_mean = _finite_proxy_mean(final_proxy)
        if bootstrap_proxy_only and final_proxy_mean is not None and command_status not in {"failed", "blocked"}:
            existing_bootstrap = run_state.get("bootstrap") if isinstance(run_state.get("bootstrap"), dict) else {}
            original_command_status = existing_bootstrap.get("original_command_status") or command_status
            command_status = "bootstrap_proxy_complete"
            run_state["command_status"] = command_status
            run_state["bootstrap"] = {
                "profile": "bootstrap",
                "proxy_only": True,
                "status": "proxy_reached",
                "original_command_status": original_command_status,
                "original_proxy_status": final_proxy.get("status"),
            }
            self._save_c2c_run_state(run_spec, run_state)
        mean = (metrics or {}).get("mean")
        ablation_result = run_state.get("ablation") or {"enabled": False, "status": "skipped", "reason": "not run"}
        dataset_regressions = self._c2c_dataset_regressions(metrics, adapter.baseline)
        worst_regression = max(dataset_regressions.values()) if dataset_regressions else 0.0
        ablation_comparison = (ablation_result.get("comparison") or {}) if isinstance(ablation_result, dict) else {}
        require_ablation_support = bool(
            self.context.config.get("c2c", {}).get("small_loop", {}).get("require_ablation_support", False)
        )
        mechanism_supported = bool(ablation_comparison.get("mechanism_supported"))
        decision = (
            "candidate_win"
            if mean is not None and float(mean) >= baseline_mean + min_delta and worst_regression <= max_regression
            and (not require_ablation_support or mechanism_supported)
            else "not_viable"
        )
        if patch_result["status"] == "rejected":
            decision = "patch_rejected"
        elif preflight and preflight.get("status") == "blocked":
            decision = "blocked"
        elif command_status == "blocked":
            decision = "blocked"
        elif command_status == "proxy_rejected":
            decision = "proxy_rejected"
        elif command_status == "proxy_repairable":
            decision = "proxy_repairable"
        elif command_status == "bootstrap_proxy_complete":
            decision = "bootstrap_proxy_complete"
        elif metrics is None:
            decision = "failed_no_metrics" if command_status == "failed" else "partial"
        result = {
            "id": candidate.get("id"),
            "title": candidate.get("title"),
            "hypothesis": candidate.get("hypothesis"),
            "mechanism_type": candidate.get("mechanism_type"),
            "run_id": run_spec["run_id"],
            "run_root": str(run_spec["run_root"]),
            "patch_result": _compact_patch_result_for_payload(patch_result),
            "code_snapshot": code_snapshot,
            "execution_repo": execution_repo,
            "execution_repo_audit": _compact_execution_repo_audit(run_state.get("execution_repo_audit")),
            "commands": _compact_command_plan(run_spec.get("commands") or {}),
            "command_status": command_status,
            "preflight": preflight,
            "proxy_screen": _compact_proxy_screen(run_state.get("proxy_screen")),
            "proxy_decision_report": _compact_proxy_decision_report(run_state.get("proxy_decision_report")),
            "activation_smoke": _compact_activation_smoke(run_state.get("activation_smoke")),
            "full_s3_readiness": _compact_full_s3_readiness(run_state.get("full_s3_readiness")),
            "run_state_path": str(run_spec["run_state_path"]),
            "preflight_path": str(run_spec["preflight_path"]),
            "config_overrides": run_spec.get("config_overrides", {}),
            "runtime_localization": run_spec.get("runtime_localization", {}),
            "has_executable_change": has_executable_change,
            "patch_fingerprint": patch_fingerprint,
            "frozen_hashes": run_spec.get("frozen_hashes", {}),
            "metrics": metrics,
            "ablation": ablation_result,
            "delta_vs_baseline": round(float(mean) - baseline_mean, 4) if mean is not None else None,
            "dataset_regressions": dataset_regressions,
            "worst_dataset_regression": worst_regression,
            "acceptance_rule": {
                "min_delta_to_pass": min_delta,
                "max_dataset_regression": max_regression,
                "baseline_mean": baseline_mean,
                "require_ablation_support": require_ablation_support,
            },
            "mechanism_supported": mechanism_supported,
            "matched_control_metrics": (
                {"mean": adapter.baseline.get("mean"), "datasets": deepcopy(adapter.baseline.get("datasets") or {})}
                if simulate and metrics is not None
                else deepcopy(run_state.get("matched_control_metrics"))
            ),
            "coverage_metrics": (
                {"mean": 1.0, "datasets": {dataset_id: 1.0 for dataset_id in (metrics or {}).get("datasets", {})}}
                if simulate and metrics is not None
                else deepcopy(run_state.get("coverage_metrics"))
            ),
            "decision": decision,
            "command_logs": _compact_event_logs(logs),
        }
        result["failure_attribution"] = self._c2c_failure_attribution(result, adapter.baseline)
        return result

    @staticmethod
    def _c2c_patch_fingerprint(adapter: C2CAdapter, patch_result: dict[str, Any], run_spec: dict[str, Any]) -> str:
        changed_files = sorted(set(str(item) for item in (patch_result or {}).get("changed_files") or [] if item))
        file_hashes: dict[str, str] = {}
        for rel_path in changed_files:
            path = adapter.repo_root / rel_path
            file_hashes[rel_path] = sha256_file(path) if path.exists() and path.is_file() else "<missing>"
        payload = {
            "frozen_hashes": run_spec.get("frozen_hashes") or {},
            "patch_status": (patch_result or {}).get("status"),
            "patch_changed_files": changed_files,
            "patch_file_hashes": file_hashes,
            "execution_repo_source": ((patch_result or {}).get("execution_repo") or {}).get("source")
            if isinstance((patch_result or {}).get("execution_repo"), dict)
            else None,
            "patched_repo_snapshot_sha256": ((patch_result or {}).get("execution_repo") or {}).get("snapshot_sha256")
            if isinstance((patch_result or {}).get("execution_repo"), dict)
            else None,
        }
        return _sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=True))

    @staticmethod
    def _load_reusable_c2c_proxy_state(run_spec: dict[str, Any], patch_fingerprint: str | None = None) -> dict[str, Any] | None:
        state_path = Path(run_spec.get("run_state_path") or "")
        if not state_path.exists():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(state, dict):
            return None
        if state.get("frozen_hashes") != run_spec.get("frozen_hashes"):
            return None
        if not patch_fingerprint or state.get("patch_fingerprint") != patch_fingerprint:
            return None
        proxy_screen = state.get("proxy_screen") if isinstance(state.get("proxy_screen"), dict) else {}
        if proxy_screen.get("patch_fingerprint") != patch_fingerprint:
            return None
        if proxy_screen.get("status") not in {"rejected", "repairable_proxy_risk"}:
            return None
        if ((proxy_screen.get("metrics") or {}).get("mean")) is None:
            return None
        return state

    @staticmethod
    def _c2c_rejudge_cached_proxy_screen(
        proxy_screen: dict[str, Any],
        *,
        baseline: dict[str, Any],
        proxy_cfg: dict[str, Any],
    ) -> dict[str, Any] | None:
        metrics = proxy_screen.get("metrics") if isinstance(proxy_screen.get("metrics"), dict) else None
        if not metrics:
            return None
        proxy_baseline = proxy_screen.get("proxy_baseline")
        if not isinstance(proxy_baseline, dict) or not proxy_baseline:
            proxy_baseline = proxy_screen.get("baseline_metrics")
        if not isinstance(proxy_baseline, dict) or not proxy_baseline:
            proxy_baseline = None
        patch_risk = proxy_screen.get("patch_risk") if isinstance(proxy_screen.get("patch_risk"), dict) else {}
        eval_smoke = proxy_screen.get("eval_smoke") if isinstance(proxy_screen.get("eval_smoke"), dict) else None
        return ExperimentAgent._c2c_proxy_metric_decision(
            metrics=metrics,
            baseline=baseline,
            proxy_cfg=proxy_cfg,
            proxy_baseline=proxy_baseline,
            patch_risk=patch_risk,
            eval_smoke=eval_smoke,
        )

    def _generate_c2c_patch(self, candidate: dict[str, Any], adapter: C2CAdapter) -> dict[str, Any]:
        del adapter
        return self._load_frozen_c2c_patch(candidate)

    def _load_frozen_c2c_patch(self, candidate: dict[str, Any]) -> dict[str, Any]:
        code_patch = candidate.get("code_patch") if isinstance(candidate.get("code_patch"), dict) else {}
        if code_patch and code_patch.get("status") != "ok":
            status = code_patch.get("status", "unknown")
            reason = code_patch.get("reason") or f"S2.5 code patch status is {status}; candidate is not eligible for deterministic S3."
            return {
                "summary": reason,
                "operations": [],
                "changed_files": [],
                "status": status,
                "fatal_patch_status": True,
            }
        if code_patch.get("status") != "ok" or not code_patch.get("patch_json"):
            return {"summary": "No valid frozen S2.5 patch attached to candidate.", "operations": [], "changed_files": [], "status": code_patch.get("status", "missing")}
        patch_path = self.context.project_root / str(code_patch["patch_json"])
        if not patch_path.exists():
            return {"summary": "Frozen S2.5 patch file is missing.", "operations": [], "changed_files": [], "status": "missing"}
        patch = json.loads(patch_path.read_text(encoding="utf-8"))
        snapshot = code_patch.get("patched_repo_snapshot")
        if isinstance(snapshot, dict) and snapshot:
            patch["patched_repo_snapshot"] = snapshot
        validation_path = code_patch.get("validation")
        if validation_path:
            validation_file = self.context.project_root / str(validation_path)
            if validation_file.exists():
                try:
                    patch["validation"] = json.loads(validation_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    patch["validation"] = {"status": "invalid", "reason": f"Could not parse {validation_path}"}
        patch.setdefault("summary", patch.get("rationale", "Frozen S2.5 patch."))
        return patch

    def _prepare_c2c_execution_repo(self, candidate: dict[str, Any], adapter: C2CAdapter, patch: dict[str, Any]) -> dict[str, Any]:
        snapshot = patch.get("patched_repo_snapshot") if isinstance(patch.get("patched_repo_snapshot"), dict) else {}
        candidate_id = sanitize_filename(str(candidate.get("id") or patch.get("candidate_id") or "candidate"))
        execution_rel = f"execution_repos/{candidate_id}"
        execution_repo = self.context.project_root / "experiment" / execution_rel
        ensure_dir(execution_repo.parent)
        snapshot_rel = str(snapshot.get("path") or "")
        snapshot_path = self.context.project_root / snapshot_rel if snapshot_rel else None
        if snapshot.get("status") == "ok" and snapshot_path and snapshot_path.exists() and snapshot_path.is_dir():
            source_lock = {
                "source": "patched_repo_snapshot",
                "snapshot_path": snapshot_rel,
                "snapshot_manifest": snapshot.get("manifest"),
                "snapshot_sha256": snapshot.get("sha256"),
            }
            reused = _materialize_c2c_execution_repo(execution_repo, snapshot_path, source_lock)
            return {
                "status": "ok",
                "source": "patched_repo_snapshot",
                "repo_root": str(execution_repo),
                "repo_root_rel": f"experiment/{execution_rel}",
                "snapshot_path": snapshot_rel,
                "snapshot_manifest": snapshot.get("manifest"),
                "snapshot_sha256": snapshot.get("sha256"),
                "reused": reused,
            }
        source_lock = {"source": "baseline_snapshot_patch_json_fallback", "snapshot_path": str(adapter.repo_root)}
        reused = _materialize_c2c_execution_repo(execution_repo, adapter.repo_root, source_lock)
        return {
            "status": "ok",
            "source": "baseline_snapshot_patch_json_fallback",
            "repo_root": str(execution_repo),
            "repo_root_rel": f"experiment/{execution_rel}",
            "reused": reused,
            "reason": "patched_repo_snapshot missing; S3 will apply patch.json fallback to isolated execution repo",
        }

    def _apply_frozen_c2c_patch(
        self,
        candidate: dict[str, Any],
        adapter: C2CAdapter,
        patch: dict[str, Any],
        *,
        execution_repo: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if patch.get("fatal_patch_status"):
            return {
                "status": "rejected",
                "errors": [patch.get("summary", "S2.5 patch is not eligible for S3.")],
                "changed_files": [],
                "reason": patch.get("summary", "S2.5 patch is not eligible for S3."),
                "candidate_id": candidate.get("id"),
                "patch_status": patch.get("status", "unknown"),
            }
        if isinstance(execution_repo, dict) and execution_repo.get("source") == "patched_repo_snapshot":
            result = {
                "status": "snapshot_applied",
                "errors": [],
                "changed_files": list(patch.get("changed_files") or []),
                "reason": "S3 execution repo was materialized directly from S2.5 patched_repo_snapshot",
                "candidate_id": candidate.get("id"),
                "patch_status": patch.get("status", "ok"),
                "execution_repo": execution_repo,
                "restore_state": [],
            }
            for key in ["risk_check", "activation_check", "mechanism_review", "quality_score", "validation"]:
                if isinstance(patch.get(key), dict):
                    result[key] = patch[key]
            return result
        if not patch.get("operations"):
            return {
                "status": "skipped",
                "errors": [],
                "changed_files": [],
                "reason": patch.get("summary", "no frozen patch operations"),
                "candidate_id": candidate.get("id"),
                "patch_status": patch.get("status", "ok"),
                "execution_repo": execution_repo or {},
            }
        policy = DynamicEditPolicy.from_config(self.context.config.get("code_patch", {}).get("dynamic_whitelist") or {})
        result = FrozenPatchGuard(policy).apply(adapter.repo_root, patch)
        result["candidate_id"] = candidate.get("id")
        result["patch_status"] = patch.get("status", "ok")
        if isinstance(execution_repo, dict):
            result["execution_repo"] = execution_repo
        for key in ["risk_check", "activation_check", "mechanism_review", "quality_score", "validation"]:
            if isinstance(patch.get(key), dict):
                result[key] = patch[key]
        return result

    @staticmethod
    def _c2c_allowed_file_snippets(adapter: C2CAdapter, *, max_chars: int = 6000) -> list[dict[str, str]]:
        snippets = []
        for rel_path in adapter.allowed_files:
            path = adapter.repo_root / rel_path
            if path.exists():
                snippets.append({"path": rel_path, "text": path.read_text(encoding="utf-8", errors="ignore")[:max_chars]})
        return snippets

    @staticmethod
    def _save_c2c_run_state(run_spec: dict[str, Any], run_state: dict[str, Any]) -> None:
        run_state["updated_at"] = now_utc()
        write_json(Path(run_spec["run_state_path"]), run_state)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): ExperimentAgent._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [ExperimentAgent._jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [ExperimentAgent._jsonable(item) for item in value]
        return value

    @staticmethod
    def _c2c_checkpoint_final_exists(run_spec: dict[str, Any]) -> bool:
        final_path = Path(run_spec["run_root"]) / "checkpoints" / "final"
        return final_path.exists() and any(final_path.iterdir()) if final_path.is_dir() else final_path.exists()

    def _run_c2c_proxy_screen(
        self,
        *,
        adapter: C2CAdapter,
        candidate: dict[str, Any],
        run_spec: dict[str, Any],
        patch_result: dict[str, Any],
        has_executable_change: bool,
        baseline: dict[str, Any],
        effective_proxy_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(self.context.config)
        if not proxy_cfg.get("enabled", False):
            return {"enabled": False, "status": "skipped", "reason": "proxy_screen disabled"}
        if effective_proxy_policy is None:
            effective_proxy_policy = read_json(self.context.project_root / "experiment" / "results" / "c2c_effective_proxy_policy.json", default={}) or {}
        threshold_proxy_cfg = dict(proxy_cfg)
        if isinstance(effective_proxy_policy, dict) and isinstance(effective_proxy_policy.get("effective_policy"), dict):
            threshold_proxy_cfg.update(effective_proxy_policy["effective_policy"])

        proxy = self._c2c_static_proxy_screen(
            candidate=candidate,
            run_spec=run_spec,
            patch_result=patch_result,
            has_executable_change=has_executable_change,
        )
        if proxy["status"] in {"rejected", "repairable_proxy_risk"}:
            return proxy

        attempts: list[dict[str, Any]] = []
        rendered_commands = self._c2c_proxy_commands(adapter, run_spec, proxy_cfg)
        if rendered_commands:
            for idx, command in enumerate(rendered_commands):
                result = self.runner.run_step(
                    name=f"proxy_command_{idx}",
                    command=command,
                    working_dir=adapter.repo_root,
                    retry_policy={"max_attempts": 1, "timeout_seconds": self._c2c_proxy_command_timeout(proxy_cfg, idx, rendered_commands)},
                )
                attempts.append(result)
                if result.get("status") != "ok" and proxy_cfg.get("reject_on_command_failure", True):
                    command_failure = self._c2c_proxy_command_failure(result)
                    reason = f"proxy command {idx} failed: {command_failure.get('summary')}"
                    repair_hint = command_failure.get("repair_hint")
                    if command_failure.get("category") == "resource_oom":
                        proxy.update(
                            {
                                "status": "resource_retry",
                                "reason": reason,
                                "repair_hint": repair_hint
                                or "do not repair the S2.5 patch for this failure; rerun cheap proxy when GPU memory is available",
                                "resource_retry": True,
                                "failure_category": "s3_proxy_resource_oom",
                                "repair_route": "resource_retry",
                                "repair_mode": "resource_retry",
                                "command_failure": command_failure,
                                "attempts": attempts,
                                "commands": rendered_commands,
                            }
                        )
                        return proxy
                    proxy.update(
                        {
                            "status": "repairable_proxy_risk",
                            "reason": reason,
                            "repair_hint": repair_hint,
                            "repair_route": "S2_plan",
                            "repair_mode": "effect_first_proxy_repair",
                            "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                                reason=reason,
                                repair_hint=repair_hint,
                                evidence={"command_failure": command_failure},
                                patch_risk=proxy.get("patch_risk") or {},
                                source="proxy_command",
                            ),
                            "command_failure": command_failure,
                            "attempts": attempts,
                            "commands": rendered_commands,
                        }
                    )
                    return proxy

        metrics = adapter.collect_proxy_metrics(run_spec)
        eval_smoke = adapter.collect_proxy_eval_smoke(run_spec)
        baseline_metrics = adapter.proxy_baseline_metrics(run_spec)
        proxy["require_proxy_metrics"] = bool(threshold_proxy_cfg.get("require_proxy_metrics", proxy_cfg.get("require_proxy_metrics", False)))
        proxy["attempts"] = attempts
        proxy["commands"] = rendered_commands
        proxy["metrics"] = metrics
        proxy["eval_smoke"] = eval_smoke
        proxy["baseline_metrics"] = baseline_metrics
        threshold_decision = self._c2c_proxy_metric_decision(
            metrics=metrics,
            baseline=baseline,
            proxy_cfg=threshold_proxy_cfg,
            proxy_baseline=baseline_metrics,
            patch_risk=proxy.get("patch_risk") or {},
            eval_smoke=eval_smoke,
        )
        if threshold_decision:
            proxy.update(threshold_decision)
            return proxy
        proxy["status"] = "passed"
        proxy["reason"] = proxy.get("reason") or "static and optional command proxy checks passed"
        return proxy

    @staticmethod
    def _c2c_proxy_command_timeout(proxy_cfg: dict[str, Any], idx: int, commands: list[str]) -> int | None:
        def coerce(value: Any) -> int | None:
            try:
                timeout = int(value)
            except (TypeError, ValueError):
                return None
            return timeout if timeout > 0 else None

        if idx < 0:
            return coerce(proxy_cfg.get("preflight_timeout_seconds")) or coerce(proxy_cfg.get("command_timeout_seconds"))
        command = commands[idx] if 0 <= idx < len(commands) else ""
        lowered = str(command).lower()
        if "script/evaluation/" in lowered or "unified_evaluator.py" in lowered or "eval_" in lowered:
            return coerce(proxy_cfg.get("eval_timeout_seconds")) or coerce(proxy_cfg.get("command_timeout_seconds"))
        if "script/train/" in lowered or "sft_train.py" in lowered or "torch.distributed" in lowered:
            return coerce(proxy_cfg.get("train_timeout_seconds")) or coerce(proxy_cfg.get("command_timeout_seconds"))
        return coerce(proxy_cfg.get("command_timeout_seconds"))

    def _run_c2c_proxy_activation_smoke(
        self,
        *,
        adapter: C2CAdapter,
        candidate: dict[str, Any],
        run_spec: dict[str, Any],
        gpu_selection: Any,
    ) -> dict[str, Any]:
        activation_spec = adapter.materialize_proxy_activation_smoke_configs(candidate, run_spec, gpu_selection)
        if not activation_spec.get("enabled") or activation_spec.get("status") != "materialized":
            return self._jsonable({**activation_spec, "attempts": [], "eval_by_dataset": {}})
        attempts: list[dict[str, Any]] = []
        eval_by_dataset: dict[str, Any] = {}
        eval_items = list((activation_spec.get("eval_configs") or {}).items())
        timeout_seconds = ((activation_spec.get("config") or {}).get("timeout_seconds"))
        command_failed = False
        for eval_idx, command in enumerate((activation_spec.get("commands") or {}).get("eval") or []):
            dataset = eval_items[eval_idx][0] if eval_idx < len(eval_items) else f"dataset_{eval_idx}"
            result = self.runner.run_step(
                name=f"activation_smoke_eval_{dataset}",
                command=command,
                working_dir=adapter.repo_root,
                retry_policy={"max_attempts": 1, "timeout_seconds": timeout_seconds},
            )
            attempts.append(result)
            eval_by_dataset[str(dataset)] = result
            if result.get("status") != "ok":
                command_failed = True
        collected = adapter.collect_proxy_activation_smoke(run_spec, activation_spec)
        if command_failed:
            collected.update(
                {
                    "status": "failed",
                    "reason": collected.get("reason") or "proxy activation smoke eval command failed",
                    "repair_hint": collected.get("repair_hint") or "repair eval-path/runtime activation before full S3",
                }
            )
        return self._jsonable({**collected, "attempts": attempts, "eval_by_dataset": eval_by_dataset})

    def _run_c2c_ablation_eval(
        self,
        *,
        adapter: C2CAdapter,
        candidate: dict[str, Any],
        run_spec: dict[str, Any],
        gpu_selection: Any,
        retry_policy: dict[str, Any],
        simulate: bool,
    ) -> dict[str, Any]:
        ablation_spec = adapter.materialize_ablation_eval_configs(candidate, run_spec, gpu_selection)
        if not ablation_spec.get("enabled"):
            return ablation_spec
        if simulate:
            metrics = adapter.write_mock_ablation_results(run_spec, offset=-0.05)
            enabled_metrics = adapter.collect_candidate_metrics(run_spec["run_id"])
            comparison = self._c2c_ablation_comparison(enabled_metrics, metrics)
            return self._jsonable({
                **ablation_spec,
                "status": "mocked",
                "metrics": metrics,
                "comparison": comparison,
                "attempts": [],
                "eval_by_dataset": {},
            })
        attempts: list[dict[str, Any]] = []
        eval_by_dataset: dict[str, Any] = {}
        eval_items = list((ablation_spec.get("eval_configs") or {}).items())
        status = "ok"
        for eval_idx, command in enumerate((ablation_spec.get("commands") or {}).get("eval") or []):
            dataset = eval_items[eval_idx][0] if eval_idx < len(eval_items) else f"dataset_{eval_idx}"
            result = self.runner.run_step(
                name=f"ablation_eval_{dataset}",
                command=command,
                working_dir=adapter.repo_root,
                retry_policy=retry_policy,
            )
            attempts.append(result)
            eval_by_dataset[dataset] = result
            if result.get("status") != "ok":
                status = "partial"
        metrics = adapter.collect_ablation_metrics(run_spec, ablation_spec)
        if metrics:
            write_json(Path(ablation_spec["metrics_path"]), metrics)
        else:
            status = "partial"
        comparison = self._c2c_ablation_comparison(adapter.collect_candidate_metrics(run_spec["run_id"]), metrics)
        return self._jsonable({
            **ablation_spec,
            "status": status,
            "metrics": metrics,
            "comparison": comparison,
            "attempts": attempts,
            "eval_by_dataset": eval_by_dataset,
        })

    @staticmethod
    def _c2c_ablation_comparison(enabled_metrics: dict[str, Any] | None, disabled_metrics: dict[str, Any] | None) -> dict[str, Any]:
        if not enabled_metrics or not disabled_metrics:
            return {"status": "insufficient_metrics", "enabled_mean": (enabled_metrics or {}).get("mean"), "disabled_mean": (disabled_metrics or {}).get("mean")}
        enabled_mean = enabled_metrics.get("mean")
        disabled_mean = disabled_metrics.get("mean")
        mean_delta = round(float(enabled_mean) - float(disabled_mean), 4) if enabled_mean is not None and disabled_mean is not None else None
        dataset_deltas = ExperimentAgent._c2c_dataset_deltas(enabled_metrics, disabled_metrics)
        return {
            "status": "ok",
            "enabled_mean": enabled_mean,
            "disabled_mean": disabled_mean,
            "enabled_minus_disabled_mean": mean_delta,
            "dataset_enabled_minus_disabled": dataset_deltas,
            "mechanism_supported": mean_delta is not None and mean_delta > 0,
        }

    @staticmethod
    def _c2c_ablation_payload(payload: dict[str, Any], adapter: C2CAdapter) -> dict[str, Any]:
        candidate_entries: list[dict[str, Any]] = []
        for candidate in payload.get("candidate_results") or []:
            if not isinstance(candidate, dict):
                continue
            ablation = candidate.get("ablation") or {}
            comparison = ablation.get("comparison") or {}
            contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
            ablation_plan = candidate.get("ablation_plan") if isinstance(candidate.get("ablation_plan"), dict) else {}
            declared_switch = ablation.get("switch") or contract.get("ablation_switch") or ablation_plan.get("switch")
            candidate_entries.append(
                {
                    "candidate_id": candidate.get("id"),
                    "title": candidate.get("title"),
                    "decision": candidate.get("decision"),
                    "command_status": candidate.get("command_status"),
                    "enabled": bool(ablation.get("enabled")),
                    "status": ablation.get("status", "missing"),
                    "switch": declared_switch,
                    "declared_switch": declared_switch,
                    "reached_ablation_stage": bool(ablation.get("comparison") or ablation.get("metrics") or ablation.get("attempts")),
                    "enabled_metrics": candidate.get("metrics"),
                    "disabled_metrics": ablation.get("metrics"),
                    "comparison": comparison,
                    "supported": comparison.get("mechanism_supported"),
                    "delta_enabled_vs_disabled": comparison.get("enabled_minus_disabled_mean"),
                    "dataset_enabled_minus_disabled": comparison.get("dataset_enabled_minus_disabled") or {},
                    "eval_configs": ablation.get("eval_configs") or {},
                    "metrics_path": ablation.get("metrics_path"),
                    "attempt_statuses": [
                        {
                            "step": attempt.get("step"),
                            "status": attempt.get("status"),
                            "returncode": attempt.get("returncode"),
                        }
                        for attempt in ablation.get("attempts") or []
                        if isinstance(attempt, dict)
                    ],
                    "reason": ablation.get("reason"),
                }
            )

        best = payload.get("best_candidate") or {}
        best_id = best.get("id")
        best_entry = next((item for item in candidate_entries if item.get("candidate_id") == best_id), None)
        completed = [item for item in candidate_entries if item.get("status") in {"ok", "mocked"} and item.get("disabled_metrics")]
        partial = [item for item in candidate_entries if item.get("enabled") and item.get("status") not in {"ok", "mocked", "skipped"}]
        eligible = [item for item in candidate_entries if item.get("enabled")]
        declared = [item for item in candidate_entries if item.get("declared_switch")]
        if completed:
            status = "ok"
            reason = "automatic ablation completed for at least one candidate"
        elif partial:
            status = "partial"
            reason = "ablation was materialized but did not produce complete disabled metrics"
        elif eligible:
            status = "pending"
            reason = "ablation was eligible but no disabled metrics were parsed"
        elif declared:
            status = "skipped"
            reason = "candidate ablation switches were declared, but no candidate reached full eval before ablation"
        else:
            status = "skipped"
            reason = "no candidate exposed an ablation_switch"

        best_comparison = (best_entry or {}).get("comparison") or {}
        proxy_baseline = None
        if best:
            proxy_baseline = ((best.get("proxy_screen") or {}).get("baseline_metrics") or {}).copy()
        return ExperimentAgent._jsonable(
            {
                "schema_version": "c2c_ablation_results_diagnostic_v1",
                "status": status,
                "reason": reason,
                "baseline": payload.get("baseline") or adapter.baseline,
                "proxy_baseline": proxy_baseline or None,
                "best_candidate_id": best_id,
                "best_supported": best_comparison.get("mechanism_supported"),
                "best_delta_enabled_vs_disabled": best_comparison.get("enabled_minus_disabled_mean"),
                "best_dataset_enabled_minus_disabled": best_comparison.get("dataset_enabled_minus_disabled") or {},
                "candidate_ablations": candidate_entries,
                "allowed_files": adapter.allowed_files,
            }
        )

    def _ensure_c2c_proxy_baseline(
        self,
        adapter: C2CAdapter,
        candidate_run_spec: dict[str, Any],
        gpu_selection: Any,
        retry_policy: dict[str, Any],
        baseline_repo_state: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(self.context.config)
        if not proxy_cfg.get("enabled", False) or not proxy_cfg.get("require_paired_baseline", True):
            return {"enabled": bool(proxy_cfg.get("enabled", False)), "status": "skipped", "reason": "paired proxy baseline disabled"}
        cached = adapter.proxy_baseline_metrics(candidate_run_spec)
        baseline_path = Path((candidate_run_spec.get("proxy_screen") or {}).get("baseline_metrics_path") or self.context.project_root / str(proxy_cfg.get("baseline_cache_path") or "experiment/results/c2c_proxy_baseline.json"))
        has_real_cache = bool(cached and cached.get("source") != "configured_full_baseline_subset_fallback" and baseline_path.exists())
        baseline_contracts = self._write_c2c_proxy_baseline_contracts(
            candidate=(candidate_run_spec.get("candidate") if isinstance(candidate_run_spec.get("candidate"), dict) else None),
            run_spec=candidate_run_spec,
            config=adapter.config,
            proxy_cfg=proxy_cfg,
            baseline_cache_exists=has_real_cache,
        )
        cache_report = baseline_contracts.get("cache_report") if isinstance(baseline_contracts.get("cache_report"), dict) else {}
        if has_real_cache and cache_report.get("action") == "reuse":
            return {
                "enabled": True,
                "status": "cached",
                "metrics": cached,
                "path": str(baseline_path),
                "cache_report": cache_report,
                "baseline_fingerprint": baseline_contracts.get("baseline_fingerprint"),
            }
        if not proxy_cfg.get("run_baseline_if_missing", True):
            if cached:
                if has_real_cache:
                    return {
                        "enabled": True,
                        "status": "blocked",
                        "reason": "proxy baseline cache fingerprint mismatch and run_baseline_if_missing is false",
                        "cache_report": cache_report,
                        "baseline_fingerprint": baseline_contracts.get("baseline_fingerprint"),
                    }
                return {"enabled": True, "status": "fallback", "metrics": cached, "reason": "using configured baseline subset fallback", "cache_report": cache_report}
            return {"enabled": True, "status": "blocked", "reason": "proxy baseline cache missing and run_baseline_if_missing is false", "cache_report": cache_report}

        patched_state = self._snapshot_c2c_repo_state(adapter) if baseline_repo_state is not None else None
        if baseline_repo_state is not None:
            self._restore_c2c_repo_state(adapter, baseline_repo_state)
        try:
            baseline_spec = adapter.materialize_proxy_baseline_configs(gpu_selection)
            attempts: list[dict[str, Any]] = []
            for step_idx, command in enumerate(baseline_spec["commands"]["preflight"]):
                result = self.runner.run_step(
                    name=f"proxy_baseline_preflight_{step_idx}",
                    command=command,
                    working_dir=adapter.repo_root,
                    retry_policy={"max_attempts": 1, "timeout_seconds": self._c2c_proxy_command_timeout(proxy_cfg, -1, [])},
                )
                attempts.append(result)
                if result.get("status") != "ok":
                    return self._c2c_proxy_baseline_failure(
                        adapter,
                        candidate_run_spec,
                        attempts=attempts,
                        reason=f"proxy baseline preflight {step_idx} failed",
                        failed_step=result,
                    )
            train_result = self.runner.run_step(
                name="proxy_baseline_train",
                command=baseline_spec["commands"]["train"],
                working_dir=adapter.repo_root,
                retry_policy={**retry_policy, "timeout_seconds": self._c2c_proxy_command_timeout(proxy_cfg, 0, [baseline_spec["commands"]["train"]])},
            )
            attempts.append(train_result)
            if train_result.get("status") != "ok" and not self._c2c_checkpoint_final_exists(baseline_spec):
                return self._c2c_proxy_baseline_failure(
                    adapter,
                    candidate_run_spec,
                    attempts=attempts,
                    reason="proxy baseline train failed",
                    failed_step=train_result,
                )
            eval_items = list(baseline_spec.get("eval_configs", {}).items())
            for eval_idx, command in enumerate(baseline_spec["commands"]["eval"]):
                dataset = eval_items[eval_idx][0] if eval_idx < len(eval_items) else f"dataset_{eval_idx}"
                result = self.runner.run_step(
                    name=f"proxy_baseline_eval_{dataset}",
                    command=command,
                    working_dir=adapter.repo_root,
                    retry_policy={**retry_policy, "timeout_seconds": self._c2c_proxy_command_timeout(proxy_cfg, 1 + eval_idx, baseline_spec["commands"]["eval"])},
                )
                attempts.append(result)
                if result.get("status") != "ok":
                    return self._c2c_proxy_baseline_failure(
                        adapter,
                        candidate_run_spec,
                        attempts=attempts,
                        reason=f"proxy baseline eval {dataset} failed",
                        failed_step=result,
                    )
            metrics = adapter.collect_proxy_baseline_run_metrics(baseline_spec)
            if not metrics:
                if proxy_cfg.get("allow_configured_baseline_fallback", True):
                    fallback = adapter.proxy_baseline_metrics(candidate_run_spec)
                    if fallback:
                        return {
                            "enabled": True,
                            "status": "fallback",
                            "reason": "proxy baseline run produced no metrics; using configured baseline subset fallback",
                            "metrics": fallback,
                            "attempts": attempts,
                        }
                return self._c2c_proxy_baseline_failure(
                    adapter,
                    candidate_run_spec,
                    attempts=attempts,
                    reason="proxy baseline run produced no metrics",
                    failed_step=attempts[-1] if attempts else {},
                )
            metrics = dict(metrics)
            metrics.setdefault("source", "proxy_baseline_run")
            write_json(Path(baseline_spec["metrics_path"]), metrics)
            post_run_contracts = self._write_c2c_proxy_baseline_contracts(
                candidate=(candidate_run_spec.get("candidate") if isinstance(candidate_run_spec.get("candidate"), dict) else None),
                run_spec=candidate_run_spec,
                config=adapter.config,
                proxy_cfg=proxy_cfg,
                baseline_cache_exists=True,
            )
            final_cache_report = post_run_contracts.get("cache_report")
            if isinstance(cache_report, dict) and cache_report.get("action") == "rerun_baseline":
                final_cache_report = {
                    **cache_report,
                    "rerun_status": "ok",
                    "post_rerun_fingerprint_hash": (post_run_contracts.get("baseline_fingerprint") or {}).get("fingerprint_hash"),
                }
                self._write_c2c_experiment_json_artifact(
                    "results/c2c_proxy_cache_report.json",
                    final_cache_report,
                    artifact_type="c2c_proxy_cache_report",
                    summary="C2C proxy baseline cache report after successful rerun",
                    source_paths=[
                        rel
                        for rel in [
                            "experiment/results/c2c_proxy_baseline_fingerprint.json",
                            "experiment/results/c2c_proxy_baseline.json",
                        ]
                        if (self.context.project_root / rel).exists()
                    ],
                )
            return {
                "enabled": True,
                "status": "ok",
                "metrics": metrics,
                "path": str(baseline_spec["metrics_path"]),
                "attempts": attempts,
                "cache_report": final_cache_report,
                "baseline_fingerprint": post_run_contracts.get("baseline_fingerprint"),
            }
        finally:
            if patched_state is not None:
                self._restore_c2c_repo_state(adapter, patched_state)

    def _c2c_proxy_baseline_failure(
        self,
        adapter: C2CAdapter,
        candidate_run_spec: dict[str, Any],
        *,
        attempts: list[dict[str, Any]],
        reason: str,
        failed_step: dict[str, Any],
    ) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(self.context.config)
        command_failure = self._c2c_proxy_command_failure(failed_step) if failed_step else {}
        payload = {
            "enabled": True,
            "status": "blocked",
            "reason": reason,
            "attempts": attempts,
            "command_failure": command_failure,
            "failure_scope": "paired_proxy_baseline",
        }
        if proxy_cfg.get("allow_configured_baseline_fallback", True):
            fallback = adapter.proxy_baseline_metrics(candidate_run_spec)
            if fallback:
                return {
                    **payload,
                    "status": "fallback",
                    "reason": f"{reason}; using configured baseline subset fallback",
                    "metrics": fallback,
                    "fallback_reason": reason,
                    "fallback_source": fallback.get("source"),
                }
        return payload

    def _c2c_static_proxy_screen(
        self,
        *,
        candidate: dict[str, Any],
        run_spec: dict[str, Any],
        patch_result: dict[str, Any],
        has_executable_change: bool,
    ) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(self.context.config)
        patch_risk = self._c2c_patch_risk(
            patch_result=patch_result,
            config_overrides=run_spec.get("config_overrides", {}),
            candidate=candidate,
        )
        signals = {
            "has_executable_change": has_executable_change,
            "changed_file_count": len((patch_result or {}).get("changed_files") or []),
            "config_override_keys": patch_risk.get("config_override_keys", []),
            "risk_labels": patch_risk.get("risk_labels", []),
            "mechanism_soft_issues": ((patch_result or {}).get("mechanism_review") or {}).get("soft_issues") or [],
            "quality_repair_needed": bool((((patch_result or {}).get("mechanism_review") or {}).get("quality_repair") or {}).get("needed")),
        }
        base = {
            "enabled": True,
            "status": "passed",
            "mode": proxy_cfg.get("mode", "static"),
            "signals": signals,
            "patch_risk": patch_risk,
            "quality_repair": _instrumentation_quality_repair_request(patch_result),
            "artifact_paths": _c2c_proxy_artifact_paths(run_spec),
        }
        static_hard_gate = bool(proxy_cfg.get("static_hard_gate", True))
        if static_hard_gate and proxy_cfg.get("reject_if_no_executable_change", True) and not has_executable_change:
            base.update({"status": "rejected", "reason": "no executable patch or config override"})
            return base
        if static_hard_gate and proxy_cfg.get("reject_eval_code_changes", True) and "evaluation_code_changed" in set(patch_risk.get("risk_labels") or []):
            base.update(
                _repairable_proxy_risk(
                    "candidate changes evaluator code; repair S2.5 patch before full S3",
                    "move mechanism evidence out of script/evaluation and into model/train artifacts",
                    patch_risk=patch_risk,
                    source="static_proxy",
                )
            )
            return base
        if static_hard_gate and proxy_cfg.get("reject_test_only_changes", True):
            labels = set(patch_risk.get("risk_labels") or [])
            changed_files = patch_risk.get("changed_files") or []
            config_keys = patch_risk.get("config_override_keys") or []
            if changed_files and labels and labels <= {"test_change"} and not config_keys:
                base.update(
                    _repairable_proxy_risk(
                        "candidate only changes tests; repair S2.5 patch before full S3",
                        "add an executable model/train/recipe mechanism change",
                        patch_risk=patch_risk,
                        source="static_proxy",
                    )
                )
                return base
        max_risk_files = proxy_cfg.get("max_risk_files")
        if static_hard_gate and max_risk_files is not None and len(patch_risk.get("risk_files") or []) > int(max_risk_files):
            base.update(
                _repairable_proxy_risk(
                    f"patch risk file count exceeds proxy threshold {max_risk_files}",
                    "shrink patch to the core mechanism and one focused validation hook",
                    patch_risk=patch_risk,
                    source="static_proxy",
                )
            )
            return base
        return base

    @staticmethod
    def _c2c_proxy_commands(adapter: C2CAdapter, run_spec: dict[str, Any], proxy_cfg: dict[str, Any]) -> list[str]:
        commands = list(proxy_cfg.get("commands") or [])
        if not commands and proxy_cfg.get("mode") in {"command", "commands", "replay", "validation"}:
            proxy_commands = (run_spec.get("proxy_screen") or {}).get("commands") or {}
            commands.extend(proxy_commands.get("train") if isinstance(proxy_commands.get("train"), list) else [proxy_commands.get("train")] if proxy_commands.get("train") else [])
            commands.extend(proxy_commands.get("eval") or [])
        fields = {
            "repo_root": str(adapter.repo_root),
            "run_id": str(run_spec.get("run_id") or ""),
            "run_root": str(run_spec.get("run_root") or ""),
            "train_config": str(run_spec.get("train_config") or ""),
            "proxy_root": str((run_spec.get("proxy_screen") or {}).get("run_root") or ""),
            "proxy_train_config": str((run_spec.get("proxy_screen") or {}).get("train_config") or ""),
            "proxy_metrics": str((run_spec.get("proxy_screen") or {}).get("metrics_path") or ""),
        }
        rendered: list[str] = []
        for command in commands:
            if not command:
                continue
            try:
                rendered.append(str(command).format(**fields))
            except KeyError:
                rendered.append(str(command))
        return rendered

    @staticmethod
    def _c2c_proxy_metric_decision(
        *,
        metrics: dict[str, Any] | None,
        baseline: dict[str, Any],
        proxy_cfg: dict[str, Any],
        proxy_baseline: dict[str, Any] | None,
        patch_risk: dict[str, Any] | None = None,
        eval_smoke: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not metrics:
            if proxy_cfg.get("require_proxy_metrics"):
                return {"status": "rejected", "reason": "proxy metrics required but not found"}
            return None
        has_paired_proxy_baseline = bool(proxy_baseline)
        comparison_baseline = proxy_baseline or baseline
        if proxy_cfg.get("require_paired_baseline", True) and not proxy_baseline:
            return {"status": "rejected", "reason": "paired proxy baseline required but not found"}
        full_baseline_mean = _coerce_float(baseline.get("mean"), default=0.0)
        proxy_baseline_mean = _coerce_float(comparison_baseline.get("mean"), default=full_baseline_mean)
        proxy_deltas = ExperimentAgent._c2c_dataset_deltas(metrics, comparison_baseline)
        regressions = ExperimentAgent._c2c_dataset_regressions(metrics, comparison_baseline)
        worst_regression = max(regressions.values()) if regressions else 0.0
        mean_delta = round(float(metrics["mean"]) - proxy_baseline_mean, 4) if metrics.get("mean") is not None else None
        proxy_score = ExperimentAgent._c2c_proxy_score(mean_delta, worst_regression, patch_risk or {}, proxy_cfg)
        evidence = {
            "full_baseline": baseline,
            "full_baseline_mean": full_baseline_mean,
            "proxy_baseline": proxy_baseline,
            "proxy_baseline_mean": _coerce_float(proxy_baseline.get("mean"), default=None) if isinstance(proxy_baseline, dict) else None,
            "comparison_baseline": comparison_baseline,
            "comparison_baseline_mean": proxy_baseline_mean,
            "proxy_delta_vs_baseline": mean_delta,
            "proxy_delta_vs_comparison_baseline": mean_delta,
            "proxy_delta_vs_proxy_baseline": mean_delta if has_paired_proxy_baseline else None,
            "proxy_dataset_deltas": proxy_deltas,
            "proxy_dataset_regressions": regressions,
            "proxy_worst_dataset_regression": round(worst_regression, 4),
            "proxy_score": proxy_score,
            "proxy_decision_mode": "paired_baseline" if has_paired_proxy_baseline else "configured_full_baseline",
        }
        if isinstance(eval_smoke, dict) and eval_smoke:
            evidence["eval_smoke"] = eval_smoke
            red_flags = set(eval_smoke.get("red_flags") or [])
            if "all_summary_scores_zero" in red_flags and (
                "low_nonempty_prediction_rate" in red_flags
                or "low_answer_parse_rate" in red_flags
                or "answer_distribution_collapsed" in red_flags
                or "all_zero_without_prediction_artifacts" in red_flags
            ):
                evidence["proxy_eval_health_failure"] = {
                    "status": "suspected_output_or_parser_failure",
                    "red_flags": sorted(red_flags),
                    "repair_hint": "inspect proxy eval outputs, answer parsing, eval recipe output paths, and mechanism activation before treating all-zero proxy as a pure method failure",
                }
        threshold = proxy_cfg.get("min_proxy_mean_delta")
        if threshold is not None and mean_delta is not None and mean_delta < float(threshold):
            repairable_margin = float(proxy_cfg.get("repairable_proxy_mean_margin", 0.0) or 0.0)
            if repairable_margin > 0 and mean_delta >= float(threshold) - repairable_margin:
                reason = f"proxy mean delta {mean_delta} below threshold {float(threshold)} but within repairable margin {repairable_margin}"
                repair_hint = "repair S2.5 patch using proxy dataset deltas before full S3"
                return {
                    "status": "repairable_proxy_risk",
                    "reason": reason,
                    "repair_hint": repair_hint,
                    "repair_route": "S2_plan",
                    "repair_mode": "effect_first_proxy_repair",
                    "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                        reason=reason,
                        repair_hint=repair_hint,
                        evidence=evidence,
                        patch_risk=patch_risk or {},
                    ),
                    **evidence,
                }
            return {
                "status": "rejected",
                "reason": f"proxy mean delta {mean_delta} below hard threshold {float(threshold)}",
                **evidence,
            }
        max_regression = proxy_cfg.get("max_proxy_dataset_regression")
        if max_regression is not None and regressions:
            worst_dataset = max(regressions, key=lambda key: regressions[key])
            worst = float(regressions[worst_dataset])
            if worst > float(max_regression):
                repairable_margin = float(proxy_cfg.get("repairable_proxy_regression_margin", 0.0) or 0.0)
                if repairable_margin > 0 and worst <= float(max_regression) + repairable_margin:
                    reason = f"proxy dataset regression {worst_dataset}={round(worst, 4)} exceeds threshold {float(max_regression)} but within repairable margin {repairable_margin}"
                    repair_hint = f"repair S2.5 patch to bound {worst_dataset} regression before full S3"
                    return {
                        "status": "repairable_proxy_risk",
                        "reason": reason,
                        "repair_hint": repair_hint,
                        "repair_route": "S2_plan",
                        "repair_mode": "effect_first_proxy_repair",
                        "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                            reason=reason,
                            repair_hint=repair_hint,
                            evidence=evidence,
                            patch_risk=patch_risk or {},
                        ),
                        **evidence,
                    }
                return {
                    "status": "rejected",
                    "reason": f"proxy dataset regression {worst_dataset}={round(worst, 4)} exceeds hard threshold {float(max_regression)}",
                    **evidence,
                }
        min_proxy_score = proxy_cfg.get("min_proxy_score")
        if min_proxy_score is not None and proxy_score is not None and proxy_score < float(min_proxy_score):
            repairable_margin = float(proxy_cfg.get("repairable_proxy_score_margin", 0.0) or 0.0)
            if repairable_margin > 0 and proxy_score >= float(min_proxy_score) - repairable_margin:
                reason = f"proxy score {proxy_score} below threshold {float(min_proxy_score)} but within repairable margin {repairable_margin}"
                repair_hint = "repair S2.5 patch to reduce patch risk or dataset regression before full S3"
                return {
                    "status": "repairable_proxy_risk",
                    "reason": reason,
                    "repair_hint": repair_hint,
                    "repair_route": "S2_plan",
                    "repair_mode": "effect_first_proxy_repair",
                    "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                        reason=reason,
                        repair_hint=repair_hint,
                        evidence=evidence,
                        patch_risk=patch_risk or {},
                    ),
                    **evidence,
                }
            return {
                "status": "rejected",
                "reason": f"proxy score {proxy_score} below hard threshold {float(min_proxy_score)}",
                **evidence,
            }

        soft_flags = []
        soft_delta = proxy_cfg.get("soft_proxy_mean_delta")
        if soft_delta is not None and mean_delta is not None and mean_delta < float(soft_delta):
            soft_flags.append(f"proxy mean delta {mean_delta} below soft threshold {float(soft_delta)}")
        soft_regression = proxy_cfg.get("soft_max_proxy_dataset_regression")
        if soft_regression is not None and worst_regression > float(soft_regression):
            soft_flags.append(f"proxy worst dataset regression {round(worst_regression, 4)} above soft threshold {float(soft_regression)}")
        soft_score = proxy_cfg.get("soft_min_proxy_score")
        if soft_score is not None and proxy_score is not None and proxy_score < float(soft_score):
            soft_flags.append(f"proxy score {proxy_score} below soft threshold {float(soft_score)}")
        if soft_flags:
            if proxy_cfg.get("repair_soft_proxy_fail", False):
                reason = "; ".join(soft_flags)
                repair_hint = "effect repair only: improve cheap-proxy mean/regression/runtime behavior before full S3; do not spend this repair on ablation, coverage, matched-coverage, or paperization diagnostics"
                return {
                    "status": "repairable_proxy_risk",
                    "reason": reason,
                    "repair_hint": repair_hint,
                    "repair_route": "S2_plan",
                    "repair_mode": "effect_first_proxy_repair",
                    "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                        reason=reason,
                        repair_hint=repair_hint,
                        evidence=evidence,
                        patch_risk=patch_risk or {},
                        soft_flags=soft_flags,
                    ),
                    "soft_fail": True,
                    "soft_flags": soft_flags,
                    **evidence,
                }
            return {
                "status": "passed",
                "reason": "proxy passed hard thresholds with soft warnings",
                "soft_fail": True,
                "soft_flags": soft_flags,
                **evidence,
            }
        return {"status": "passed", "reason": "proxy passed hard and soft thresholds", **evidence}

    @staticmethod
    def _c2c_proxy_command_failure(step_result: dict[str, Any]) -> dict[str, Any]:
        attempts = [attempt for attempt in step_result.get("attempts", []) if isinstance(attempt, dict)]
        text = "\n".join((attempt.get("stdout") or "") + "\n" + (attempt.get("stderr") or "") for attempt in attempts)
        lower = text.lower()
        timed_out = any(bool(attempt.get("timed_out")) for attempt in attempts)
        distributed_failure = (
            "childfailederror" in lower
            or "torch.distributed.elastic" in lower
            or "torch.distributed.run" in lower and "local_rank" in lower and "exitcode" in lower
        )
        if timed_out or step_result.get("returncode") == 124 or "timed out" in lower:
            category = "proxy_timeout"
            repair_hint = "reduce mechanism inference/training cost or add bounded early-exit guards before rerunning cheap proxy"
        elif distributed_failure:
            category = "distributed_child_failed"
            repair_hint = "rerun cheap proxy with a single auto-selected GPU or capture the child rank traceback before treating this as a patch repair"
        elif "mat1 and mat2 must have the same dtype" in lower or "same dtype" in lower:
            category = "dtype_mismatch"
            repair_hint = "cast new mechanism tensors/modules to the active model dtype/device before matmul or linear layers"
        elif "must be real number, not list" in lower:
            category = "schema_shape_mismatch"
            repair_hint = "normalize dataset adapter fields to scalar/tensor shapes expected by the existing training path"
        elif "out of memory" in lower or "cuda oom" in lower or "cublas_status_alloc_failed" in lower:
            category = "resource_oom"
            repair_hint = "do not repair the S2.5 patch for this failure; rerun cheap proxy after GPU memory is available or with a different auto-selected GPU"
        elif "traceback" in lower:
            category = "runtime_exception"
            repair_hint = "fix the proxy training runtime exception in the S2.5 patch before full S3"
        else:
            category = "command_failed"
            repair_hint = "inspect proxy command stdout/stderr and repair the S2.5 patch before full S3"
        summary = category
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line and (
                "error" in line.lower()
                or "exception" in line.lower()
                or "runtimeerror" in line.lower()
                or "typeerror" in line.lower()
                or "childfailederror" in line.lower()
                or "local_rank" in line.lower()
            ):
                summary = f"{category}: {line[-240:]}"
                break
        rank_failure = _parse_torch_distributed_rank_failure(text)
        latest_attempt = attempts[-1] if attempts else {}
        return {
            "category": category,
            "summary": summary,
            "repair_hint": repair_hint,
            "returncode": step_result.get("returncode"),
            "step": step_result.get("step"),
            "rank_failure": rank_failure,
            "stdout_tail": str(latest_attempt.get("stdout") or "")[-1200:],
            "stderr_tail": str(latest_attempt.get("stderr") or "")[-2000:],
            "elapsed_seconds": max([float(attempt.get("elapsed_seconds") or 0.0) for attempt in attempts] or [0.0]),
            "timeout_seconds": next(
                (
                    attempt.get("timeout_seconds")
                    for attempt in attempts
                    if attempt.get("timeout_seconds") is not None
                ),
                None,
            ),
        }

    def _record_c2c_full_s3_readiness(
        self,
        *,
        candidate: dict[str, Any],
        run_spec: dict[str, Any],
        patch_result: dict[str, Any],
        run_state: dict[str, Any],
        baseline: dict[str, Any],
        min_delta: float,
        max_regression: float,
    ) -> dict[str, Any]:
        report = _c2c_full_s3_readiness_report(
            candidate=candidate,
            run_spec=run_spec,
            patch_result=patch_result,
            run_state=run_state,
            baseline=baseline,
            min_delta=min_delta,
            max_regression=max_regression,
            iteration=self._registry_iteration(),
            project_id=self.context.project_root.name,
        )
        run_report_path = Path(run_spec["run_root"]) / "full_s3_readiness_report.json"
        stage_rel = self.context.artifacts.stage_dir(self.stage_key).relative_to(self.context.project_root).as_posix()
        report.setdefault("artifact_paths", {})
        report["artifact_paths"]["candidate_readiness_report"] = str(run_report_path)
        report["artifact_paths"]["project_readiness_report"] = f"{stage_rel}/results/full_s3_readiness_report.json"
        write_json(run_report_path, report)
        self.context.artifacts.write_json(
            self.stage_key,
            "results/full_s3_readiness_report.json",
            report,
            artifact_type="c2c_full_s3_readiness",
            summary="Proxy-to-full readiness report before C2C full S3 training",
        )
        return report

    @staticmethod
    def _c2c_oom_recovery_hint(step_result: dict[str, Any]) -> dict[str, Any] | None:
        text = "\n".join(
            (attempt.get("stdout") or "") + "\n" + (attempt.get("stderr") or "")
            for attempt in step_result.get("attempts", [])
        ).lower()
        if "out of memory" not in text and "cuda oom" not in text and "cublas_status_alloc_failed" not in text:
            return None
        return {
            "action": "retry_train_reduced_concurrency",
            "status": "attempted",
            "reason": "detected CUDA OOM signature; retry with fewer visible GPUs/processes without changing hyperparameters",
        }

    def _c2c_posthoc_review(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        fallback = self._c2c_deterministic_posthoc_review(payload)
        if self._c2c_proxy_only_failure(payload):
            fallback["status"] = "deterministic_proxy_feedback"
            fallback["reason"] = "cheap proxy rejected or repair-routed all candidates; skipped GPT posthoc to keep effect-first loop cheap"
            return fallback
        if self.context.config.get("experiment", {}).get("disable_llm_during_execution"):
            fallback["status"] = "deterministic_execution_feedback"
            fallback["reason"] = "experiment.disable_llm_during_execution=true; deterministic posthoc feedback generated"
            return fallback
        if not self.context.llm.use_real_api:
            fallback["status"] = "deterministic_no_llm"
            fallback["reason"] = "real GPT API unavailable; deterministic posthoc feedback generated"
            return fallback
        prompt = {
            "baseline": payload.get("baseline"),
            "acceptance": payload.get("acceptance"),
            "candidate_results": _compact_for_review(payload.get("candidate_results", []), 12000),
            "constraints": [
                "Do not modify metrics or acceptance.",
                "Only analyze causes and propose next-round S1/S2 constraints.",
                "Training and evaluation execution already completed deterministically.",
            ],
        }
        schema = {"type": "object", "required": ["status", "failure_modes", "next_round_suggestions", "avoid_repeat_rules"]}
        try:
            review = self.context.llm.generate_json_with_schema(
                instructions=(
                    "You are a posthoc experiment reviewer for C2C. Analyze failures, regressions, and likely next steps. "
                    "Return JSON only; do not alter reported results."
                ),
                prompt=json.dumps(prompt, ensure_ascii=False),
                default=fallback,
                schema=schema,
                agent_name="c2c-posthoc-reviewer",
            )
        except Exception as exc:
            fallback["status"] = "degraded"
            fallback["reason"] = f"GPT posthoc review unavailable: {_short_error(exc)}"
            fallback["gpt_error_type"] = exc.__class__.__name__
            return fallback
        return review if isinstance(review, dict) else None

    @staticmethod
    def _c2c_deterministic_posthoc_review(payload: dict[str, Any]) -> dict[str, Any]:
        acceptance = payload.get("acceptance") or {}
        baseline = payload.get("baseline") or {}
        best = payload.get("best_candidate") or {}
        candidate_results = payload.get("candidate_results") or []
        baseline_datasets = baseline.get("datasets") or {}
        failure_modes: list[str] = []
        suggestions: list[str] = []
        avoid_rules: list[str] = []
        feedback_entries: list[dict[str, Any]] = []
        proxy_failed = [
            item
            for item in candidate_results
            if isinstance(item, dict) and item.get("decision") in {"proxy_rejected", "proxy_repairable"}
        ]
        if proxy_failed and len(proxy_failed) == len(candidate_results):
            repairable_count = sum(1 for item in proxy_failed if item.get("decision") == "proxy_repairable")
            failure_modes.append(f"cheap proxy blocked all candidates before full S3: repairable={repairable_count}/{len(proxy_failed)}")
            suggestions.append("Route back to S2.5 and repair only the failing patch mechanism before any full S3 run.")
            suggestions.append("Require next S2.5 to cite proxy_dataset_deltas, command_failure category, and patch_risk_labels for each repaired candidate.")
            avoid_rules.append("Do not enter full S3 for a candidate until cheap proxy passes paired-baseline mean, worst-dataset regression, and proxy score checks.")

        best_metrics = best.get("metrics") or {}
        delta = acceptance.get("delta")
        if delta is None and best_metrics.get("mean") is not None and baseline.get("mean") is not None:
            try:
                delta = float(best_metrics["mean"]) - float(baseline["mean"])
            except (TypeError, ValueError):
                delta = None
        if delta is not None and float(delta) < float(acceptance.get("min_delta_to_pass", 0.1)):
            failure_modes.append(f"best mean did not clear baseline margin: delta={round(float(delta), 4)}")
            suggestions.append("Prioritize mechanisms with an explicit expected mean gain, not only regression protection.")
            avoid_rules.append("Do not repeat below-baseline configuration-only variants without a new mechanism and ablation.")

        best_ablation_evidence = ExperimentAgent._c2c_ablation_evidence(best)
        if best_ablation_evidence.get("status") == "no_effect":
            failure_modes.append(
                f"ablation switch did not change metrics: enabled_minus_disabled={best_ablation_evidence.get('enabled_minus_disabled_mean')}"
            )
            suggestions.append("Require S2.5 to prove the ablation switch changes the active inference path before full S3.")
            avoid_rules.append("Do not accept a mechanism whose ablation-disabled eval is identical to enabled eval.")

        regressions = best.get("dataset_regressions") or {}
        if regressions:
            worst_dataset = max(regressions, key=lambda key: regressions[key])
            worst_value = regressions.get(worst_dataset)
            failure_modes.append(f"worst dataset regression: {worst_dataset}={worst_value}")
            suggestions.append(f"Add a {worst_dataset}-specific guard or fallback before rerunning similar alignment gates.")
            avoid_rules.append(f"Do not increase alignment transfer strength without bounding {worst_dataset} regression.")

        for candidate in candidate_results:
            if not isinstance(candidate, dict):
                continue
            decision = candidate.get("decision")
            if decision not in {"not_viable", "failed_no_metrics", "partial", "blocked", "patch_rejected", "proxy_rejected", "proxy_repairable"}:
                continue
            candidate_metrics = candidate.get("metrics") or {}
            candidate_regressions = candidate.get("dataset_regressions") or {}
            attribution = candidate.get("failure_attribution") or {}
            proxy_screen = candidate.get("proxy_screen") or {}
            reason = (
                proxy_screen.get("reason")
                or candidate.get("blocked_reason")
                or acceptance.get("reason")
                or "candidate did not pass acceptance"
            )
            feedback_entries.append(
                {
                    "kind": "c2c_posthoc_feedback",
                    "idea_id": candidate.get("id"),
                    "title": candidate.get("title"),
                    "decision": decision,
                    "failure_mode": decision,
                    "reason": reason,
                    "metrics": candidate_metrics,
                    "dataset_regressions": candidate_regressions,
                    "failure_attribution": attribution,
                    "proxy_screen": proxy_screen,
                    "dragging_datasets": attribution.get("dragging_datasets") or [],
                    "sample_type_failures": attribution.get("sample_type_failures") or [],
                    "patch_risk": attribution.get("patch_risk") or {},
                    "mixed_gain_patterns": attribution.get("mixed_gain_patterns") or [],
                    "avoid_repeat_rule": _candidate_avoid_rule(candidate, baseline_datasets),
                }
            )

        return {
            "status": "deterministic_fallback",
            "reason": acceptance.get("reason") or "candidate did not pass acceptance",
            "failure_modes": failure_modes or [acceptance.get("reason") or "candidate did not pass acceptance"],
            "next_round_suggestions": suggestions or ["Generate a materially different mechanism before rerunning S3."],
            "avoid_repeat_rules": list(dict.fromkeys(avoid_rules or ["Do not repeat failed ideas without a new mechanism and explicit ablation."])),
            "feedback_entries": feedback_entries,
        }

    @staticmethod
    def _c2c_failure_analysis_md(payload: dict[str, Any], posthoc: dict[str, Any] | None) -> str:
        acceptance = payload.get("acceptance") or {}
        lines = ["# C2C Failure Analysis", ""]
        lines.append(f"- Acceptance passed: {acceptance.get('passed')}")
        lines.append(f"- Reason: {acceptance.get('reason')}")
        best = payload.get("best_candidate") or {}
        if best:
            lines.append(f"- Best candidate: {best.get('id')} decision={best.get('decision')} mean={(best.get('metrics') or {}).get('mean')}")
            lines.append(f"- Dataset regressions: {json.dumps(best.get('dataset_regressions') or {}, ensure_ascii=False)}")
            attribution = best.get("failure_attribution") or {}
            if attribution:
                lines.append(f"- Primary failure: {attribution.get('primary_failure')}")
                lines.append(f"- Dragging datasets: {json.dumps(attribution.get('dragging_datasets') or [], ensure_ascii=False)}")
                lines.append(f"- Sample families failed: {json.dumps(attribution.get('sample_type_failures') or [], ensure_ascii=False)}")
                lines.append(f"- Patch risk: {json.dumps((attribution.get('patch_risk') or {}).get('risk_files') or [], ensure_ascii=False)}")
                lines.append(f"- Mixed gain patterns: {json.dumps(attribution.get('mixed_gain_patterns') or [], ensure_ascii=False)}")
                lines.append(f"- Ablation evidence: {json.dumps(attribution.get('ablation_evidence') or {}, ensure_ascii=False)}")
        proxy_candidates = [
            item
            for item in payload.get("candidate_results", [])
            if isinstance(item, dict) and item.get("decision") in {"proxy_rejected", "proxy_repairable"}
        ]
        if proxy_candidates:
            lines.append("")
            lines.append("## Cheap Proxy Evidence")
            for item in proxy_candidates:
                proxy_screen = item.get("proxy_screen") or {}
                attribution = item.get("failure_attribution") or {}
                command_failure = proxy_screen.get("command_failure") or {}
                lines.append(f"- {item.get('id')}: decision={item.get('decision')} proxy_status={proxy_screen.get('status')}")
                if proxy_screen.get("reason"):
                    lines.append(f"  - Reason: {proxy_screen.get('reason')}")
                if command_failure:
                    lines.append(f"  - Command failure: {json.dumps(command_failure, ensure_ascii=False)}")
                if proxy_screen.get("proxy_dataset_deltas"):
                    lines.append(f"  - Proxy dataset deltas: {json.dumps(proxy_screen.get('proxy_dataset_deltas'), ensure_ascii=False)}")
                    lines.append(f"  - Proxy score: {proxy_screen.get('proxy_score')}")
                if attribution.get("dragging_datasets"):
                    lines.append(f"  - Dragging datasets: {json.dumps(attribution.get('dragging_datasets'), ensure_ascii=False)}")
                patch_labels = ((attribution.get("patch_risk") or {}).get("risk_labels") or [])
                if patch_labels:
                    lines.append(f"  - Patch risk labels: {json.dumps(patch_labels, ensure_ascii=False)}")
        lines.append("")
        lines.append("## Posthoc Review")
        if posthoc:
            for item in _posthoc_items(posthoc.get("failure_modes"), limit=8):
                lines.append(f"- Failure mode: {item}")
            for item in _posthoc_items(posthoc.get("next_round_suggestions"), limit=8):
                lines.append(f"- Next round: {item}")
            for item in _posthoc_items(posthoc.get("avoid_repeat_rules"), limit=8):
                lines.append(f"- Avoid: {item}")
        else:
            lines.append("- Posthoc review unavailable.")
        return "\n".join(lines)

    def _write_c2c_failure_feedback(self, payload: dict[str, Any], *, artifacts: list[str]) -> dict[str, Any]:
        if _c2c_payload_is_retryable_pause(payload):
            return self.context.artifacts.write_json(
                self.stage_key,
                "results/retryable_pause_feedback.json",
                {
                    "created_at": now_utc(),
                    "kind": "c2c_retryable_pause_feedback",
                    "reason": ((payload.get("acceptance") or {}).get("reason") or "C2C S3 paused for retryable external resources"),
                    "candidate_results": [
                        _compact_candidate_result(item)
                        for item in payload.get("candidate_results", [])
                        if isinstance(item, dict)
                    ],
                    "acceptance": payload.get("acceptance") or {},
                    "posthoc_review": _compact_c2c_result_payload(payload.get("posthoc_review")),
                    "excluded_from_failure_memory": True,
                    "excluded_from_failure_memory_reason": "retryable_resource_or_quota_pause_is_not_method_failure",
                },
                artifact_type="c2c_retryable_pause_feedback",
                summary="Retryable C2C S3 pause metadata excluded from failure memory",
                source_paths=artifacts,
            )
        best = payload.get("best_candidate")
        feedback_candidate = best
        if (not feedback_candidate or not feedback_candidate.get("metrics")) and isinstance(payload.get("best_proxy_candidate"), dict):
            feedback_candidate = payload.get("best_proxy_candidate")
        acceptance = payload.get("acceptance") or {}
        reason = acceptance.get("reason") or "C2C candidate did not clear acceptance"
        failure_mode = "not_viable"
        if feedback_candidate and feedback_candidate.get("decision") == "proxy_rejected":
            failure_mode = "proxy_rejected"
            reason = ((feedback_candidate.get("proxy_screen") or {}).get("reason") or reason)
        elif feedback_candidate and feedback_candidate.get("decision") == "proxy_repairable":
            failure_mode = "proxy_repairable"
            reason = ((feedback_candidate.get("proxy_screen") or {}).get("reason") or reason)
        elif not feedback_candidate or not feedback_candidate.get("metrics"):
            failure_mode = "no_metrics"
        elif feedback_candidate.get("decision") == "blocked":
            failure_mode = "blocked"
        manager = FailureLogManager(self.context.config, external_root=self.context.project_root / "meta")
        entry = manager.append_c2c_feedback(
            project_id=self.context.project_root.name,
            iteration=self._registry_iteration(),
            candidate=feedback_candidate,
            acceptance=acceptance,
            failure_mode=failure_mode,
            reason=reason,
            artifacts=artifacts,
        )
        feedback_bundle = build_c2c_feedback_bundle(
            [entry, *((payload.get("posthoc_review") or {}).get("feedback_entries") or [])],
            project_id=self.context.project_root.name,
            iteration=self._registry_iteration(),
            traces=self._load_iteration_traces(),
            sources=artifacts,
        )
        method_feedback_bundle = build_c2c_feedback_bundle(
            [entry, *((payload.get("posthoc_review") or {}).get("feedback_entries") or [])],
            project_id=self.context.project_root.name,
            iteration=self._registry_iteration(),
            traces=self._load_iteration_traces(),
            sources=artifacts,
            view="method",
        )
        implementation_feedback_bundle = build_c2c_feedback_bundle(
            [entry, *((payload.get("posthoc_review") or {}).get("feedback_entries") or [])],
            project_id=self.context.project_root.name,
            iteration=self._registry_iteration(),
            traces=self._load_iteration_traces(),
            sources=artifacts,
            view="implementation",
        )
        meta_memory = self.context.project_root / "meta" / "negative_memory.jsonl"
        ensure_dir(meta_memory.parent)
        with meta_memory.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(feedback_bundle["summary_entry"], ensure_ascii=False) + "\n")
        feedback_round_dir = self.context.project_root / "literature" / "feedback"
        ensure_dir(feedback_round_dir)
        round_path = feedback_round_dir / f"failed_ideas_round_{self._registry_iteration():03d}.json"
        write_json(round_path, feedback_bundle)
        failed_candidates = [
            item
            for item in payload.get("candidate_results", [])
            if item.get("decision") in {"not_viable", "failed_no_metrics", "partial", "blocked", "patch_rejected", "proxy_rejected", "proxy_repairable"}
        ]
        feedback_payload = {
            "created_at": now_utc(),
            "entry": entry,
            "summary": feedback_bundle["summary"],
            "method_feedback": method_feedback_bundle,
            "implementation_feedback": implementation_feedback_bundle,
            "candidate": _compact_candidate_result(feedback_candidate) if isinstance(feedback_candidate, dict) else feedback_candidate,
            "best_candidate": _compact_candidate_result(best) if isinstance(best, dict) else best,
            "best_proxy_candidate": _compact_candidate_result(payload.get("best_proxy_candidate")) if isinstance(payload.get("best_proxy_candidate"), dict) else payload.get("best_proxy_candidate"),
            "candidate_results": [
                _compact_candidate_result(item)
                for item in payload.get("candidate_results", [])
                if isinstance(item, dict)
            ],
            "failed_idea_ids": [item.get("id") for item in failed_candidates if item.get("id")],
            "failed_titles": [item.get("title") for item in failed_candidates if item.get("title")],
            "acceptance": acceptance,
            "posthoc_review": _compact_c2c_result_payload(payload.get("posthoc_review")),
            "avoid_repeat_rules": [
                item
                for item in [entry.get("avoid_repeat_rule"), *_posthoc_items((payload.get("posthoc_review") or {}).get("avoid_repeat_rules"))]
                if item
            ],
            "feedback_round_path": round_path.relative_to(self.context.project_root).as_posix(),
        }
        return self.context.artifacts.write_json(
            self.stage_key,
            "results/failure_feedback.json",
            feedback_payload,
            artifact_type="c2c_failure_feedback",
            summary="Failure feedback routed to next S1/S2 iteration",
            source_paths=artifacts,
        )

    def _load_iteration_traces(self) -> list[dict[str, Any]]:
        trace_path = self.context.project_root / "meta" / "iteration_trace.jsonl"
        if not trace_path.exists():
            return []
        traces: list[dict[str, Any]] = []
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                traces.append(item)
        return traces

    def _registry_iteration(self) -> int:
        registry_path = self.context.project_root / "meta" / "registry.yaml"
        payload = read_yaml(registry_path, default={}) or {}
        return int(payload.get("iteration", 1) or 1)

    @staticmethod
    def _best_c2c_candidate(run_results: list[dict[str, Any]]) -> dict[str, Any] | None:
        scored = [
            item
            for item in run_results
            if (item.get("metrics") or {}).get("mean") is not None
        ]
        if not scored:
            return None
        return max(scored, key=lambda item: item["metrics"]["mean"])

    @staticmethod
    def _best_c2c_proxy_candidate(run_results: list[dict[str, Any]]) -> dict[str, Any] | None:
        scored = [
            item
            for item in run_results
            if ((item.get("proxy_screen") or {}).get("metrics") or {}).get("mean") is not None
        ]
        if not scored:
            return None

        def key(item: dict[str, Any]) -> tuple[float, float]:
            proxy = item.get("proxy_screen") or {}
            metrics = proxy.get("metrics") or {}
            score = proxy.get("proxy_score")
            try:
                proxy_score = float(score)
            except (TypeError, ValueError):
                proxy_score = float("-inf")
            return (float(metrics.get("mean")), proxy_score)

        return max(scored, key=key)

    @staticmethod
    def _c2c_proxy_only_failure(payload: dict[str, Any]) -> bool:
        candidates = [item for item in payload.get("candidate_results") or [] if isinstance(item, dict)]
        if not candidates:
            return False
        return all(item.get("decision") in {"proxy_rejected", "proxy_repairable"} for item in candidates)

    @staticmethod
    def _c2c_verification_md(
        best: dict[str, Any] | None,
        baseline_mean: float,
        run_results: list[dict[str, Any]],
        min_delta: float,
        max_regression: float,
    ) -> str:
        lines = ["# Hypothesis Verification", ""]
        threshold = baseline_mean + min_delta
        worst_regression = (best or {}).get("worst_dataset_regression", 0.0)
        if best and best.get("metrics", {}).get("mean", 0) >= threshold and worst_regression <= max_regression:
            lines.append(f"- H1: supported. Best candidate `{best['id']}` reached mean {best['metrics']['mean']} >= threshold {threshold:.4f}.")
        else:
            lines.append(f"- H1: not supported in this loop. No candidate cleared baseline {baseline_mean} + min_delta {min_delta}.")
        ablation = (best or {}).get("ablation") or {}
        comparison = ablation.get("comparison") or {}
        if ablation.get("status") in {"ok", "mocked"} and comparison.get("status") == "ok":
            delta = comparison.get("enabled_minus_disabled_mean")
            switch = ablation.get("switch") or "ablation_switch"
            if comparison.get("mechanism_supported"):
                lines.append(f"- H2: supported. Disabling `{switch}` reduced mean by {delta}, so the measured gain depends on the proposed mechanism.")
            else:
                lines.append(f"- H2: not supported. Disabling `{switch}` did not reduce mean; enabled-minus-disabled mean delta={delta}.")
        elif ablation.get("enabled"):
            lines.append(f"- H2: inconclusive. Ablation status={ablation.get('status')} reason={ablation.get('reason') or 'disabled metrics unavailable'}.")
        else:
            lines.append(f"- H2: skipped. Candidate did not expose an ablation switch ({ablation.get('reason') or 'not run'}).")
        lines.append("")
        lines.append("## Candidate Decisions")
        for item in run_results:
            mean = (item.get("metrics") or {}).get("mean")
            ablation_comparison = ((item.get("ablation") or {}).get("comparison") or {})
            lines.append(
                f"- {item.get('id')}: decision={item.get('decision')}, mean={mean}, "
                f"delta={item.get('delta_vs_baseline')}, ablation_delta={ablation_comparison.get('enabled_minus_disabled_mean')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _c2c_dataset_regressions(metrics: dict[str, Any] | None, baseline: dict[str, Any]) -> dict[str, float]:
        if not metrics:
            return {}
        baseline_scores = baseline.get("datasets") or {}
        candidate_scores = metrics.get("datasets") or {}
        regressions = {}
        for dataset, base_score in baseline_scores.items():
            candidate_score = candidate_scores.get(dataset)
            if candidate_score is None:
                continue
            regressions[dataset] = round(max(0.0, float(base_score) - float(candidate_score)), 4)
        return regressions

    @staticmethod
    def _c2c_dataset_deltas(metrics: dict[str, Any] | None, baseline: dict[str, Any]) -> dict[str, float]:
        if not metrics:
            return {}
        baseline_scores = baseline.get("datasets") or {}
        candidate_scores = metrics.get("datasets") or {}
        deltas = {}
        for dataset, base_score in baseline_scores.items():
            candidate_score = candidate_scores.get(dataset)
            if candidate_score is None:
                continue
            deltas[str(dataset)] = round(float(candidate_score) - float(base_score), 4)
        return deltas

    @staticmethod
    def _c2c_proxy_score(
        mean_delta: float | None,
        worst_regression: float,
        patch_risk: dict[str, Any],
        proxy_cfg: dict[str, Any],
    ) -> float | None:
        if mean_delta is None:
            return None
        regression_weight = float(proxy_cfg.get("proxy_score_regression_weight", 0.5) or 0.0)
        risk_penalty = float(proxy_cfg.get("risk_penalty_per_label", 0.05) or 0.0) * len(patch_risk.get("risk_labels") or [])
        return round(float(mean_delta) - regression_weight * float(worst_regression) - risk_penalty, 4)

    @staticmethod
    def _c2c_failure_attribution(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        metrics = candidate.get("metrics") or {}
        baseline_scores = baseline.get("datasets") or {}
        candidate_scores = metrics.get("datasets") or {}
        proxy_screen = candidate.get("proxy_screen") or {}
        proxy_baseline = proxy_screen.get("proxy_baseline") or proxy_screen.get("baseline_metrics") or {}
        proxy_scores = (proxy_screen.get("metrics") or {}).get("datasets") or {}
        proxy_baseline_scores = proxy_baseline.get("datasets") or {}
        if not candidate_scores and proxy_scores:
            candidate_scores = proxy_scores
            baseline_scores = proxy_baseline_scores or baseline_scores
        proxy_deltas = proxy_screen.get("proxy_dataset_deltas") or {}
        dragging: list[dict[str, Any]] = []
        improved: list[dict[str, Any]] = []
        sample_type_failures: list[dict[str, Any]] = []
        for dataset, base_score in baseline_scores.items():
            if dataset not in candidate_scores:
                continue
            try:
                delta = round(float(candidate_scores[dataset]) - float(base_score), 4)
                item = {
                    "dataset": str(dataset),
                    "sample_family": ExperimentAgent._c2c_dataset_sample_family(str(dataset)),
                    "baseline": float(base_score),
                    "score": float(candidate_scores[dataset]),
                    "delta": delta,
                }
                if proxy_scores:
                    item["source"] = "proxy_screen"
            except (TypeError, ValueError):
                continue
            if delta < 0:
                failed = dict(item)
                failed["regression"] = round(abs(delta), 4)
                dragging.append(failed)
                sample_type_failures.append(
                    {
                        "sample_family": failed["sample_family"],
                        "dataset": failed["dataset"],
                        "evidence": f"dataset delta {delta}",
                    }
                )
            elif delta > 0:
                improved.append(item)
        if not dragging and proxy_deltas:
            for dataset, delta_value in proxy_deltas.items():
                try:
                    delta = round(float(delta_value), 4)
                except (TypeError, ValueError):
                    continue
                baseline_score = proxy_baseline_scores.get(dataset)
                score = proxy_scores.get(dataset)
                item = {
                    "dataset": str(dataset),
                    "sample_family": ExperimentAgent._c2c_dataset_sample_family(str(dataset)),
                    "baseline": baseline_score,
                    "score": score,
                    "delta": delta,
                    "source": "proxy_screen",
                }
                if delta < 0:
                    failed = dict(item)
                    failed["regression"] = round(abs(delta), 4)
                    dragging.append(failed)
                    sample_type_failures.append(
                        {
                            "sample_family": failed["sample_family"],
                            "dataset": failed["dataset"],
                            "evidence": f"proxy dataset delta {delta}",
                        }
                    )
                elif delta > 0:
                    improved.append(item)
        dragging.sort(key=lambda item: item.get("regression", 0.0), reverse=True)
        improved.sort(key=lambda item: item.get("delta", 0.0), reverse=True)

        mixed_patterns = []
        dragging_names = {item["dataset"] for item in dragging}
        improved_names = {item["dataset"] for item in improved}
        if "openbookqa" in improved_names and "mmlu-redux" in dragging_names:
            mixed_patterns.append("openbookqa_gain_mmlu_redux_regression")
        if improved_names and dragging_names:
            mixed_patterns.append("cross_dataset_tradeoff")

        patch_risk = ExperimentAgent._c2c_patch_risk(
            patch_result=candidate.get("patch_result") or {},
            config_overrides=candidate.get("config_overrides") or {},
            candidate=candidate,
        )
        quality_repair = ((candidate.get("proxy_screen") or {}).get("quality_repair") or {})
        eval_smoke = ((candidate.get("proxy_screen") or {}).get("eval_smoke") or {})
        proxy_eval_health_failure = ((candidate.get("proxy_screen") or {}).get("proxy_eval_health_failure") or {})
        activation_smoke = candidate.get("activation_smoke") or ((candidate.get("proxy_screen") or {}).get("activation_smoke") or {})
        full_s3_readiness = candidate.get("full_s3_readiness") or ((candidate.get("proxy_screen") or {}).get("full_s3_readiness") or {})
        execution_repo_audit = candidate.get("execution_repo_audit") if isinstance(candidate.get("execution_repo_audit"), dict) else {}
        ablation_evidence = ExperimentAgent._c2c_ablation_evidence(candidate)
        primary_failure = "none"
        if execution_repo_audit.get("status") == "failed":
            primary_failure = "execution_repo_output_pollution"
        elif proxy_eval_health_failure:
            primary_failure = "proxy_eval_output_health_failure"
        elif isinstance(full_s3_readiness, dict) and full_s3_readiness.get("full_train_allowed") is False:
            primary_failure = "full_s3_readiness_not_ready"
        elif isinstance(activation_smoke, dict) and activation_smoke.get("status") == "failed":
            trace = activation_smoke.get("mechanism_trace") if isinstance(activation_smoke.get("mechanism_trace"), dict) else {}
            if trace.get("status") == "wired":
                primary_failure = "proxy_activation_metric_neutral"
            else:
                primary_failure = "proxy_activation_smoke_no_effect"
        elif ablation_evidence.get("status") == "no_effect":
            primary_failure = "ablation_no_effect"
        elif candidate.get("decision") == "proxy_rejected":
            primary_failure = "cheap_proxy_rejected_before_full_training"
        elif candidate.get("decision") == "proxy_repairable":
            primary_failure = "repairable_proxy_risk_before_full_training"
        elif dragging:
            primary_failure = f"{dragging[0]['dataset']}_regression"
        elif candidate.get("delta_vs_baseline") is not None and float(candidate.get("delta_vs_baseline") or 0.0) < 0:
            primary_failure = "mean_below_baseline"
        elif candidate.get("decision") in {"failed_no_metrics", "partial", "blocked", "patch_rejected"}:
            primary_failure = str(candidate.get("decision"))

        return {
            "primary_failure": primary_failure,
            "dragging_datasets": dragging,
            "improved_datasets": improved,
            "sample_type_failures": sample_type_failures,
            "mixed_gain_patterns": list(dict.fromkeys(mixed_patterns)),
            "patch_risk": patch_risk,
            "proxy_screen": _proxy_screen_for_failure_attribution(proxy_screen),
            "eval_smoke": eval_smoke,
            "proxy_eval_health_failure": proxy_eval_health_failure,
            "activation_smoke": activation_smoke,
            "full_s3_readiness": full_s3_readiness,
            "execution_repo_audit": execution_repo_audit,
            "proxy_effect_repair_contract": proxy_screen.get("proxy_effect_repair_contract") or {},
            "ablation_evidence": ablation_evidence,
            "quality_repair": quality_repair,
        }

    @staticmethod
    def _c2c_ablation_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
        ablation = candidate.get("ablation") or {}
        comparison = ablation.get("comparison") or {}
        if not comparison:
            return {"status": "missing", "reason": (ablation or {}).get("reason") or "no ablation comparison"}
        delta = comparison.get("enabled_minus_disabled_mean")
        supported = bool(comparison.get("mechanism_supported"))
        dataset_deltas = comparison.get("dataset_enabled_minus_disabled") or {}
        no_effect = delta is not None and abs(float(delta)) <= 1e-4 and all(abs(float(v)) <= 1e-4 for v in dataset_deltas.values())
        if supported:
            status = "supported"
        elif no_effect:
            status = "no_effect"
        else:
            status = "unsupported"
        return {
            "status": status,
            "enabled_minus_disabled_mean": delta,
            "dataset_enabled_minus_disabled": dataset_deltas,
            "mechanism_supported": supported,
            "switch": ablation.get("switch"),
            "ablation_status": ablation.get("status"),
        }

    @staticmethod
    def _c2c_dataset_sample_family(dataset: str) -> str:
        mapping = {
            "mmlu-redux": "multi_domain_knowledge_reasoning",
            "ai2-arc": "science_reasoning_challenge",
            "openbookqa": "openbook_science_qa",
        }
        return mapping.get(dataset, dataset.replace("-", "_"))

    @staticmethod
    def _c2c_patch_risk(
        *,
        patch_result: dict[str, Any],
        config_overrides: dict[str, Any] | None = None,
        candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        changed_files = list((patch_result or {}).get("changed_files") or [])
        risk_files = []
        labels: set[str] = set()
        for rel_path in changed_files:
            reasons = []
            if str(rel_path).startswith("script/evaluation/"):
                reasons.append("evaluation code changed")
                labels.add("evaluation_code_changed")
            if str(rel_path).startswith("script/train/"):
                reasons.append("training loop changed")
                labels.add("training_loop_changed")
            if str(rel_path) == "rosetta/model/projector.py":
                reasons.append("projector mechanism changed")
                labels.add("projector_mechanism_changed")
            if str(rel_path) == "rosetta/model/aligner.py":
                reasons.append("alignment mechanism changed")
                labels.add("alignment_mechanism_changed")
            if str(rel_path).startswith("recipe/"):
                reasons.append("recipe or hyperparameter file changed")
                labels.add("recipe_changed")
            if str(rel_path).startswith("test/") or str(rel_path).startswith("tests/"):
                reasons.append("test-only change")
                labels.add("test_change")
            if reasons:
                risk_files.append({"path": str(rel_path), "reasons": reasons})
        override_keys = ExperimentAgent._flatten_config_keys(config_overrides or {})
        if override_keys:
            labels.add("config_override_changed")
        if (patch_result or {}).get("errors"):
            labels.add("patch_error")
        return {
            "changed_files": changed_files,
            "risk_files": risk_files,
            "risk_labels": sorted(labels),
            "config_override_keys": override_keys,
            "patch_errors": list((patch_result or {}).get("errors") or []),
            "candidate_id": (candidate or {}).get("id"),
        }

    @staticmethod
    def _flatten_config_keys(value: Any, *, prefix: str = "", limit: int = 40) -> list[str]:
        keys: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                keys.extend(ExperimentAgent._flatten_config_keys(item, prefix=next_prefix, limit=limit))
                if len(keys) >= limit:
                    return keys[:limit]
        elif prefix:
            keys.append(prefix)
        return keys[:limit]

    @staticmethod
    def _c2c_acceptance_comparison(
        best: dict[str, Any] | None,
        baseline: dict[str, Any],
        min_delta: float,
        max_regression: float,
    ) -> dict[str, Any]:
        baseline_mean = float(baseline.get("mean") or DEFAULT_BASELINE["mean"])
        if not best or not best.get("metrics"):
            proxy = (best or {}).get("proxy_screen") or {}
            if (proxy.get("metrics") or {}).get("mean") is not None:
                return {
                    "passed": False,
                    "baseline_mean": baseline_mean,
                    "best_mean": None,
                    "delta": None,
                    "proxy_best_mean": (proxy.get("metrics") or {}).get("mean"),
                    "proxy_delta": proxy.get("proxy_delta_vs_baseline"),
                    "proxy_score": proxy.get("proxy_score"),
                    "proxy_worst_dataset_regression": proxy.get("proxy_worst_dataset_regression"),
                    "min_delta_to_pass": min_delta,
                    "max_dataset_regression": max_regression,
                    "reason": proxy.get("reason") or "cheap proxy blocked candidate before full S3",
                }
            return {
                "passed": False,
                "baseline_mean": baseline_mean,
                "best_mean": None,
                "delta": None,
                "min_delta_to_pass": min_delta,
                "max_dataset_regression": max_regression,
                "reason": "no candidate metrics",
            }
        best_mean = float(best["metrics"]["mean"])
        delta = round(best_mean - baseline_mean, 4)
        worst_regression = float(best.get("worst_dataset_regression") or 0.0)
        passed = delta >= min_delta and worst_regression <= max_regression
        if passed and best.get("acceptance_rule", {}).get("require_ablation_support", False) and best.get("decision") != "candidate_win":
            passed = False
            reason = "mechanism ablation support not met"
        else:
            reason = "accepted" if passed else "mean delta or dataset regression threshold not met"
        return {
            "passed": passed,
            "baseline_mean": baseline_mean,
            "best_mean": best_mean,
            "delta": delta,
            "min_delta_to_pass": min_delta,
            "max_dataset_regression": max_regression,
            "worst_dataset_regression": worst_regression,
            "require_ablation_support": best.get("acceptance_rule", {}).get("require_ablation_support", False),
            "mechanism_supported": best.get("mechanism_supported"),
            "reason": reason,
        }

    @staticmethod
    def _c2c_strong_reference_comparisons(best: dict[str, Any] | None, adapter: C2CAdapter) -> list[dict[str, Any]]:
        metrics = (best or {}).get("metrics") or {}
        if not metrics:
            return []
        best_mean = metrics.get("mean")
        comparisons = []
        for reference in adapter.strong_references:
            if reference.get("enabled") is False:
                continue
            if reference.get("visible_to_ideation") is not False:
                continue
            ref_mean = reference.get("mean")
            delta = None
            if best_mean is not None and ref_mean is not None:
                delta = round(float(best_mean) - float(ref_mean), 4)
            comparisons.append(
                {
                    "name": reference.get("name"),
                    "reference_role": reference.get("reference_role", "s3_strong_reference_only"),
                    "visible_to_ideation": False,
                    "used_for_acceptance": False,
                    "candidate_id": (best or {}).get("id"),
                    "candidate_mean": best_mean,
                    "reference_mean": ref_mean,
                    "delta_vs_reference": delta,
                    "dataset_deltas": ExperimentAgent._c2c_dataset_deltas(metrics, reference),
                    "dataset_regressions": ExperimentAgent._c2c_dataset_regressions(metrics, reference),
                    "source": reference.get("source"),
                }
            )
        return comparisons

    @staticmethod
    def _c2c_summary_md(payload: dict[str, Any]) -> str:
        baseline = payload.get("baseline") or {}
        acceptance = payload.get("acceptance") or {}
        lines = [
            "# C2C Small-Loop Summary",
            "",
            f"- Baseline: {baseline.get('name')} mean={baseline.get('mean')}",
            f"- Acceptance: passed={acceptance.get('passed')} delta={acceptance.get('delta')} min_delta={acceptance.get('min_delta_to_pass')}",
        ]
        best = payload.get("best_candidate")
        if best:
            lines.append(f"- Best candidate: {best.get('id')} mean={best.get('metrics', {}).get('mean')} decision={best.get('decision')}")
            ablation_comparison = ((best.get("ablation") or {}).get("comparison") or {})
            if ablation_comparison:
                lines.append(
                    f"- Best ablation: status={(best.get('ablation') or {}).get('status')} "
                    f"enabled_minus_disabled={ablation_comparison.get('enabled_minus_disabled_mean')} "
                    f"supported={ablation_comparison.get('mechanism_supported')}"
                )
        else:
            lines.append("- Best candidate: none with parsed metrics")
        strong_refs = payload.get("strong_reference_comparisons") or []
        if strong_refs:
            lines.append("")
            lines.append("## S3-Only Strong References")
            for ref in strong_refs:
                lines.append(
                    f"- {ref.get('name')}: delta_vs_reference={ref.get('delta_vs_reference')} "
                    f"used_for_acceptance={ref.get('used_for_acceptance')}"
                )
        lines.append("")
        lines.append("## Runs")
        for item in payload.get("candidate_results", []):
            lines.append(
                f"- {item.get('id')}: status={item.get('command_status')}, decision={item.get('decision')}, metrics={item.get('metrics')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _c2c_blocked_reason(run_results: list[dict[str, Any]]) -> str | None:
        if not run_results:
            return "C2C small-loop produced no candidate runs."
        resource_blocked = [
            item
            for item in run_results
            if item.get("decision") == "blocked"
            and ((item.get("proxy_screen") or {}).get("status") in {"resource_retry", "blocked"})
        ]
        if resource_blocked and len(resource_blocked) == len(run_results):
            return "C2C cheap proxy is waiting for GPU resources; resume after an available proxy GPU is detected."
        blocked = [item for item in run_results if item.get("decision") == "blocked"]
        if blocked and len(blocked) == len(run_results):
            return "C2C small-loop blocked before metrics; inspect experiment/results/main_results.json and experiment/logs for the blocking reason."
        failed = [item for item in run_results if item.get("decision") in {"failed_no_metrics", "patch_rejected"}]
        if failed and len(failed) == len(run_results):
            return "C2C small-loop did not produce metrics; inspect experiment/logs/c2c_*_commands.json for the training or evaluation failure."
        proxy_rejected = [item for item in run_results if item.get("decision") == "proxy_rejected"]
        if proxy_rejected and len(proxy_rejected) == len(run_results):
            return "C2C cheap proxy rejected all candidates before full S3; inspect proxy_screen and failure_attribution fields."
        proxy_repairable = [item for item in run_results if item.get("decision") == "proxy_repairable"]
        if proxy_repairable and len(proxy_repairable) == len(run_results):
            return "C2C cheap proxy found repairable S2.5 patch risk for all candidates; reroute to S2.5 patch repair before full S3."
        return None

    @staticmethod
    def _parse_laps_metrics(text: str) -> dict[str, Any]:
        import re

        model_blocks = []
        pattern = re.compile(
            r"Evaluating\s+(runs/\S+).*?rsum:\s*([0-9.]+).*?i2t:\s*\[([0-9., ]+)\].*?t2i:\s*\[([0-9., ]+)\]",
            re.DOTALL,
        )
        for match in pattern.finditer(text):
            model_blocks.append(match.groups())
        metrics = {}
        for name, rsum, i2t_raw, t2i_raw in model_blocks:
            i2t = [float(x.strip()) for x in i2t_raw.split(",")]
            t2i = [float(x.strip()) for x in t2i_raw.split(",")]
            metrics[name] = {
                "rsum": float(rsum),
                "i2t": {"R@1": i2t[0], "R@5": i2t[1], "R@10": i2t[2]},
                "t2i": {"R@1": t2i[0], "R@5": t2i[1], "R@10": t2i[2]},
            }
        return metrics

    @staticmethod
    def _parse_metrics_from_log(log_path: Path) -> dict[str, Any] | None:
        if not log_path.exists():
            return None
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        import re

        rsum = re.search(r"(?:rsum:\s*|Current rsum is\s*)([0-9.]+)", text)
        i2t = re.search(r"Image to text \(R@1, R@5, R@10\):\s*([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)", text)
        t2i = re.search(r"Text to image \(R@1, R@5, R@10\):\s*([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)", text)
        sim_t = re.search(r"calculate similarity time:\s*([0-9.]+)", text)
        if not (rsum and i2t and t2i):
            return None
        return {
            "rsum": float(rsum.group(1)),
            "i2t": {"R@1": float(i2t.group(1)), "R@5": float(i2t.group(2)), "R@10": float(i2t.group(3))},
            "t2i": {"R@1": float(t2i.group(1)), "R@5": float(t2i.group(2)), "R@10": float(t2i.group(3))},
            "similarity_time": float(sim_t.group(1)) if sim_t else None,
        }

    @staticmethod
    def _screening_decision(metrics: dict[str, Any] | None, baseline_metrics: dict[str, Any] | None) -> str:
        if not metrics or not baseline_metrics:
            return "failed"
        baseline_rsum = baseline_metrics.get("rsum", 0)
        delta_i2t = metrics["i2t"]["R@1"] - baseline_metrics["i2t"]["R@1"]
        delta_t2i = metrics["t2i"]["R@1"] - baseline_metrics["t2i"]["R@1"]
        time_ok = True
        if metrics.get("similarity_time") and baseline_metrics.get("similarity_time"):
            time_ok = metrics["similarity_time"] <= baseline_metrics["similarity_time"] * 1.15
        if time_ok and (metrics["rsum"] >= baseline_rsum or delta_i2t >= 0.5 or delta_t2i >= 0.5):
            return "viable"
        return "not_viable"

    @staticmethod
    def _build_laps_train_command(
        *,
        python_cmd: str,
        data_path: str,
        image_root: str,
        logger_name: str,
        gpu_id: int,
        resume_path: str | None,
        resume_strict: int,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        changes: dict[str, Any],
    ) -> str:
        args = {
            "dataset": "f30k",
            "data_path": data_path,
            "f30k_img_path": image_root,
            "gpu-id": 0,
            "logger_name": logger_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "workers": 8,
            "vit_type": "vit",
            "embed_size": 512,
            "loss": "trip",
            "size_augment": 0,
            "route_mode": "sum",
            "route_conflict_weight": 0.25,
            "attention_weight": 0.8,
            "sparse_ratio": 0.5,
            "aggr_ratio": 0.4,
            "use_cupr": 0,
            "seed": 0,
            "eval": 1,
            "save_results": 1,
            "learning_rate": learning_rate,
        }
        if resume_path:
            args["resume"] = resume_path
            args["resume_strict"] = resume_strict
        args.update(changes)
        rendered = []
        for key, value in args.items():
            rendered.append(f"--{key} {shlex.quote(str(value))}")
        return f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES={gpu_id} {python_cmd} train.py {' '.join(rendered)}"


def _compact_for_review(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return value
    return {"truncated_json": text[:max_chars]}


def _repairable_proxy_risk(
    reason: str,
    repair_hint: str,
    *,
    patch_risk: dict[str, Any] | None = None,
    source: str = "static_proxy",
) -> dict[str, Any]:
    return {
        "status": "repairable_proxy_risk",
        "reason": reason,
        "repair_hint": repair_hint,
        "repair_route": "S2_plan",
        "repair_mode": "effect_first_proxy_repair",
        "proxy_effect_repair_contract": _proxy_effect_repair_contract(
            reason=reason,
            repair_hint=repair_hint,
            evidence={},
            patch_risk=patch_risk or {},
            source=source,
        ),
    }


def _c2c_full_s3_readiness_report(
    *,
    candidate: dict[str, Any],
    run_spec: dict[str, Any],
    patch_result: dict[str, Any],
    run_state: dict[str, Any],
    baseline: dict[str, Any],
    min_delta: float,
    max_regression: float,
    iteration: int,
    project_id: str,
) -> dict[str, Any]:
    proxy_screen = run_state.get("proxy_screen") if isinstance(run_state.get("proxy_screen"), dict) else {}
    activation_smoke = run_state.get("activation_smoke") if isinstance(run_state.get("activation_smoke"), dict) else {}
    patch_risk = proxy_screen.get("patch_risk") if isinstance(proxy_screen.get("patch_risk"), dict) else {}
    if not patch_risk:
        patch_risk = ExperimentAgent._c2c_patch_risk(
            patch_result=patch_result,
            config_overrides=run_spec.get("config_overrides") or {},
            candidate=candidate,
        )
    eval_smoke = proxy_screen.get("eval_smoke") if isinstance(proxy_screen.get("eval_smoke"), dict) else {}
    proxy_delta = proxy_screen.get("proxy_delta_vs_comparison_baseline", proxy_screen.get("proxy_delta_vs_proxy_baseline", proxy_screen.get("proxy_delta_vs_baseline")))
    hard_risk_labels = sorted(
        label
        for label in set(patch_risk.get("risk_labels") or [])
        if label in {"evaluation_code_changed", "test_change", "patch_error"}
    )
    static_clean = proxy_screen.get("status") == "passed" and not hard_risk_labels
    activation_comparison = activation_smoke.get("comparison") if isinstance(activation_smoke.get("comparison"), dict) else {}
    mechanism_trace = activation_smoke.get("mechanism_trace") if isinstance(activation_smoke.get("mechanism_trace"), dict) else {}
    ablation_switch = _readiness_ablation_switch(candidate, activation_smoke)
    ablation_effective = None
    if activation_smoke.get("status") == "passed":
        ablation_effective = bool(activation_comparison.get("mechanism_observed", True))
    elif activation_smoke.get("status") == "failed":
        ablation_effective = mechanism_trace.get("status") == "wired"
    readiness_reasons = []
    warnings = []
    blockers = []
    if static_clean:
        readiness_reasons.append("static patch risk is clean enough for effect-first full S3")
    else:
        warnings.append("static risk has labels: " + ", ".join(hard_risk_labels or patch_risk.get("risk_labels") or ["unknown"]))
    if proxy_screen.get("status") == "passed":
        readiness_reasons.append(f"cheap proxy passed with delta={_fmt_readiness_number(proxy_delta)}")
    else:
        blockers.append(f"cheap proxy status is {proxy_screen.get('status') or 'missing'}")
    if eval_smoke:
        if _eval_smoke_healthy(eval_smoke):
            readiness_reasons.append("proxy eval smoke output health is acceptable")
        else:
            blockers.append("proxy eval smoke has red flags: " + ", ".join(eval_smoke.get("red_flags") or []))
    else:
        warnings.append("proxy eval smoke missing")
    if activation_smoke:
        if activation_smoke.get("status") == "passed":
            if activation_comparison.get("mechanism_wired_metric_neutral"):
                readiness_reasons.append("activation smoke found eval-path wiring but metric-neutral proxy outputs")
                warnings.append("activation smoke was metric-neutral; full train is allowed under the neutral-proxy exploration policy if proxy regression is bounded")
            else:
                readiness_reasons.append("activation smoke observed enabled-vs-disabled change")
        elif activation_smoke.get("status") == "skipped":
            warnings.append("activation smoke skipped")
        else:
            if mechanism_trace.get("status") == "wired":
                warnings.append("activation smoke metric/prediction output is neutral but mechanism wiring is present")
            else:
                blockers.append(f"activation smoke status is {activation_smoke.get('status')}")
    else:
        warnings.append("activation smoke missing")
    if ablation_switch:
        readiness_reasons.append(f"ablation switch declared: {ablation_switch}")
    else:
        warnings.append("ablation switch missing")
    if proxy_screen.get("soft_fail"):
        warnings.extend(str(item) for item in proxy_screen.get("soft_flags") or [])

    allowed = proxy_screen.get("status") == "passed" and not blockers
    return {
        "schema_version": "c2c_proxy_to_full_readiness_v1",
        "project_id": project_id,
        "iteration": iteration,
        "created_at": now_utc(),
        "candidate_id": candidate.get("id"),
        "candidate_title": candidate.get("title"),
        "run_id": run_spec.get("run_id"),
        "run_root": str(run_spec.get("run_root")),
        "full_train_allowed": allowed,
        "status": "ready" if allowed else "not_ready",
        "static_risk": {
            "status": "clean" if static_clean else "warning",
            "hard_risk_labels": hard_risk_labels,
            "risk_labels": patch_risk.get("risk_labels") or [],
            "risk_files": patch_risk.get("risk_files") or [],
            "changed_files": patch_risk.get("changed_files") or [],
            "config_override_keys": patch_risk.get("config_override_keys") or [],
        },
        "proxy": {
            "status": proxy_screen.get("status"),
            "reason": proxy_screen.get("reason"),
            "delta": proxy_delta,
            "score": proxy_screen.get("proxy_score"),
            "soft_fail": bool(proxy_screen.get("soft_fail")),
            "soft_flags": proxy_screen.get("soft_flags") or [],
            "dataset_deltas": proxy_screen.get("proxy_dataset_deltas") or {},
            "worst_dataset_regression": proxy_screen.get("proxy_worst_dataset_regression"),
            "full_baseline_mean": proxy_screen.get("full_baseline_mean"),
            "proxy_baseline_mean": proxy_screen.get("proxy_baseline_mean"),
            "comparison_baseline_mean": proxy_screen.get("comparison_baseline_mean"),
        },
        "eval_smoke": {
            "status": eval_smoke.get("status") or "missing",
            "healthy": _eval_smoke_healthy(eval_smoke),
            "red_flags": eval_smoke.get("red_flags") or [],
            "nonempty_prediction_rate": eval_smoke.get("nonempty_prediction_rate"),
            "answer_parse_rate": eval_smoke.get("answer_parse_rate"),
            "mean_output_length": eval_smoke.get("mean_output_length"),
            "answer_distribution": eval_smoke.get("answer_distribution") or {},
        },
        "activation_smoke": {
            "status": activation_smoke.get("status") or "missing",
            "reason": activation_smoke.get("reason"),
            "switch": activation_smoke.get("switch") or ablation_switch,
            "mechanism_observed": activation_comparison.get("mechanism_observed"),
            "mechanism_wired_metric_neutral": bool(activation_comparison.get("mechanism_wired_metric_neutral")),
            "mechanism_trace_status": mechanism_trace.get("status"),
            "no_op": (
                activation_smoke.get("status") == "failed"
                and not bool(activation_comparison.get("mechanism_observed"))
                and mechanism_trace.get("status") != "wired"
            ),
            "enabled_minus_disabled_mean": activation_comparison.get("enabled_minus_disabled_mean"),
            "prediction_diff_rate": ((activation_comparison.get("prediction_comparison") or {}).get("prediction_diff_rate") if isinstance(activation_comparison.get("prediction_comparison"), dict) else None),
            "answer_diff_rate": ((activation_comparison.get("prediction_comparison") or {}).get("answer_diff_rate") if isinstance(activation_comparison.get("prediction_comparison"), dict) else None),
            "output_smoke_red_flags": ((activation_smoke.get("output_smoke") or {}).get("red_flags") if isinstance(activation_smoke.get("output_smoke"), dict) else []),
            "mechanism_trace": mechanism_trace,
        },
        "ablation_switch": {
            "declared": bool(ablation_switch),
            "switch": ablation_switch,
            "effective_in_activation_smoke": ablation_effective,
        },
        "acceptance_targets": {
            "full_baseline_mean": baseline.get("mean"),
            "min_delta_to_pass": min_delta,
            "max_dataset_regression": max_regression,
        },
        "worth_full_train": {
            "decision": "yes" if allowed else "no",
            "reason": _readiness_worth_reason(proxy_delta, proxy_screen, activation_smoke, eval_smoke, blockers),
            "evidence": readiness_reasons[:8],
            "warnings": list(dict.fromkeys(warnings))[:8],
            "blockers": list(dict.fromkeys(blockers))[:8],
        },
        "artifact_paths": {
            "run_state": str(run_spec.get("run_state_path")),
            "proxy_metrics": str(((run_spec.get("proxy_screen") or {}).get("metrics_path") or "")),
            "proxy_baseline_metrics": str(((run_spec.get("proxy_screen") or {}).get("baseline_metrics_path") or "")),
            "proxy_train_config": str(((run_spec.get("proxy_screen") or {}).get("train_config") or "")),
        },
    }


def _c2c_full_s3_readiness_block_reason(readiness: dict[str, Any]) -> str:
    worth = readiness.get("worth_full_train") if isinstance(readiness.get("worth_full_train"), dict) else {}
    blockers = [str(item) for item in (worth.get("blockers") or []) if item]
    reason = str(worth.get("reason") or "").strip()
    if blockers:
        return "full S3 readiness failed: " + "; ".join(blockers[:4])
    if reason:
        return "full S3 readiness failed: " + _shorten_text(reason, 400)
    return f"full S3 readiness failed with status={readiness.get('status') or 'unknown'}"


def _c2c_neutral_proxy_full_s3_allowed(proxy_screen: dict[str, Any], proxy_cfg: dict[str, Any]) -> bool:
    if not bool(proxy_cfg.get("allow_neutral_proxy_full_s3", True)):
        return False
    if not isinstance(proxy_screen, dict) or proxy_screen.get("status") != "passed":
        return False
    if proxy_screen.get("proxy_eval_health_failure"):
        return False
    delta = _coerce_float(
        proxy_screen.get(
            "proxy_delta_vs_comparison_baseline",
            proxy_screen.get("proxy_delta_vs_proxy_baseline", proxy_screen.get("proxy_delta_vs_baseline")),
        ),
        default=None,
    )
    if delta is None:
        return False
    worst_regression = _coerce_float(proxy_screen.get("proxy_worst_dataset_regression"), default=0.0) or 0.0
    min_delta = float(proxy_cfg.get("neutral_proxy_min_delta", -0.1) or 0.0)
    max_regression = float(proxy_cfg.get("neutral_proxy_max_dataset_regression", 0.25) or 0.0)
    return delta >= min_delta and worst_regression <= max_regression


def _c2c_neutral_proxy_policy_summary(proxy_screen: dict[str, Any], proxy_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "allow_neutral_proxy_full_s3": bool(proxy_cfg.get("allow_neutral_proxy_full_s3", True)),
        "neutral_proxy_min_delta": proxy_cfg.get("neutral_proxy_min_delta", -0.1),
        "neutral_proxy_max_dataset_regression": proxy_cfg.get("neutral_proxy_max_dataset_regression", 0.25),
        "proxy_delta": proxy_screen.get(
            "proxy_delta_vs_comparison_baseline",
            proxy_screen.get("proxy_delta_vs_proxy_baseline", proxy_screen.get("proxy_delta_vs_baseline")),
        ),
        "proxy_worst_dataset_regression": proxy_screen.get("proxy_worst_dataset_regression"),
        "proxy_score": proxy_screen.get("proxy_score"),
        "soft_fail": bool(proxy_screen.get("soft_fail")),
        "soft_flags": proxy_screen.get("soft_flags") or [],
    }


def _readiness_ablation_switch(candidate: dict[str, Any], activation_smoke: dict[str, Any]) -> str | None:
    if activation_smoke.get("switch"):
        return str(activation_smoke["switch"])
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    ablation_plan = candidate.get("ablation_plan") if isinstance(candidate.get("ablation_plan"), dict) else {}
    switch = contract.get("ablation_switch") or ablation_plan.get("switch")
    return str(switch) if switch else None


def _eval_smoke_healthy(smoke: dict[str, Any]) -> bool:
    if not isinstance(smoke, dict) or not smoke:
        return False
    return smoke.get("status") in {"ok", "skipped"} and not smoke.get("red_flags")


def _fmt_readiness_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _readiness_worth_reason(
    proxy_delta: Any,
    proxy_screen: dict[str, Any],
    activation_smoke: dict[str, Any],
    eval_smoke: dict[str, Any],
    blockers: list[str],
) -> str:
    if blockers:
        return "; ".join(blockers[:3])
    pieces = [
        f"cheap proxy {proxy_screen.get('status')} delta={_fmt_readiness_number(proxy_delta)}",
        f"activation_smoke={activation_smoke.get('status') or 'missing'}",
        f"eval_smoke={eval_smoke.get('status') or 'missing'}",
    ]
    if proxy_screen.get("soft_fail"):
        pieces.append("soft warnings recorded but hard gates passed")
    return "; ".join(pieces)


def _proxy_effect_repair_contract(
    *,
    reason: str,
    repair_hint: str | None,
    evidence: dict[str, Any],
    patch_risk: dict[str, Any],
    source: str = "cheap_proxy_metrics",
    soft_flags: list[str] | None = None,
) -> dict[str, Any]:
    dataset_deltas = evidence.get("proxy_dataset_deltas") or {}
    dataset_regressions = evidence.get("proxy_dataset_regressions") or {}
    dragging: list[dict[str, Any]] = []
    improved: list[dict[str, Any]] = []
    for dataset, delta_value in dataset_deltas.items():
        try:
            delta = round(float(delta_value), 4)
        except (TypeError, ValueError):
            continue
        regression = dataset_regressions.get(dataset, max(0.0, -delta))
        item = {"dataset": str(dataset), "delta": delta, "regression": round(float(regression or 0.0), 4)}
        if delta <= 0:
            dragging.append(item)
        elif delta > 0:
            improved.append(item)
    dragging.sort(key=lambda item: (item.get("regression", 0.0), -item.get("delta", 0.0)), reverse=True)
    improved.sort(key=lambda item: item.get("delta", 0.0), reverse=True)

    risk_labels = list(patch_risk.get("risk_labels") or [])
    changed_files = list(patch_risk.get("changed_files") or [])
    config_keys = list(patch_risk.get("config_override_keys") or [])
    command_failure = evidence.get("command_failure") if isinstance(evidence.get("command_failure"), dict) else {}
    eval_smoke = evidence.get("eval_smoke") if isinstance(evidence.get("eval_smoke"), dict) else {}
    proxy_eval_health_failure = evidence.get("proxy_eval_health_failure") if isinstance(evidence.get("proxy_eval_health_failure"), dict) else {}
    activation_smoke = _activation_smoke_repair_evidence(evidence.get("activation_smoke") if isinstance(evidence.get("activation_smoke"), dict) else {})
    full_s3_readiness = evidence.get("full_s3_readiness") if isinstance(evidence.get("full_s3_readiness"), dict) else {}
    readiness_worth = full_s3_readiness.get("worth_full_train") if isinstance(full_s3_readiness.get("worth_full_train"), dict) else {}
    readiness_blockers = [str(item) for item in (readiness_worth.get("blockers") or []) if item]
    readiness_warnings = [str(item) for item in (readiness_worth.get("warnings") or []) if item]
    priorities = []
    if readiness_blockers:
        priorities.append("Clear full S3 readiness blockers before full training: " + "; ".join(readiness_blockers[:3]))
    if command_failure:
        priorities.append(f"Fix proxy runtime/smoke failure first: {_shorten_text(str(command_failure.get('category') or command_failure.get('summary') or 'command_failure'), 180)}")
    if proxy_eval_health_failure:
        priorities.append("Fix proxy eval output health before method tuning: " + ", ".join(proxy_eval_health_failure.get("red_flags") or []))
    if activation_smoke:
        priorities.append("Fix eval-path activation before full S3: " + _shorten_text(str(activation_smoke.get("reason") or "activation_smoke_failed"), 180))
    if dragging:
        priorities.append("Target dragging proxy datasets: " + ", ".join(item["dataset"] for item in dragging[:3]))
    if evidence.get("proxy_delta_vs_comparison_baseline") is not None or evidence.get("proxy_delta_vs_baseline") is not None:
        priorities.append("Raise proxy_delta_vs_comparison_baseline without increasing proxy_worst_dataset_regression.")
    if risk_labels:
        priorities.append("Reduce patch-risk labels that hurt proxy score: " + ", ".join(str(label) for label in risk_labels[:4]))
    priorities.append("Keep the same idea and produce an executable effect-first patch; do not switch to a new idea.")

    contract = {
        "mode": "effect_first_proxy_repair",
        "source": source,
        "goal": "Find a runnable patch with positive cheap-proxy signal and no evaluation pollution before spending full S3.",
        "reason": _shorten_text(str(reason), 500),
        "repair_hint": _shorten_text(str(repair_hint or ""), 500),
        "soft_flags": list(soft_flags or []),
        "full_baseline_mean": evidence.get("full_baseline_mean"),
        "proxy_baseline_mean": evidence.get("proxy_baseline_mean"),
        "comparison_baseline_mean": evidence.get("comparison_baseline_mean"),
        "proxy_delta_vs_baseline": evidence.get("proxy_delta_vs_baseline"),
        "proxy_delta_vs_comparison_baseline": evidence.get("proxy_delta_vs_comparison_baseline"),
        "proxy_delta_vs_proxy_baseline": evidence.get("proxy_delta_vs_proxy_baseline"),
        "proxy_score": evidence.get("proxy_score"),
        "proxy_worst_dataset_regression": evidence.get("proxy_worst_dataset_regression"),
        "proxy_dataset_deltas": dataset_deltas,
        "proxy_dataset_regressions": dataset_regressions,
        "eval_smoke": eval_smoke,
        "proxy_eval_health_failure": proxy_eval_health_failure,
        "activation_smoke": activation_smoke,
        "full_s3_readiness": full_s3_readiness,
        "readiness_blockers": readiness_blockers[:8],
        "readiness_warnings": readiness_warnings[:8],
        "dragging_datasets": dragging[:5],
        "improved_datasets": improved[:5],
        "patch_risk_labels": risk_labels,
        "changed_files": changed_files[:12],
        "config_override_keys": config_keys[:12],
        "repair_priorities": priorities[:6],
        "forbidden": [
            "Do not edit evaluator or metric computation files.",
            "Do not spend this repair on ablation, coverage, matched-coverage, or paperization-only diagnostics.",
            "Do not hide failure by weakening proxy thresholds, changing baseline metrics, or changing evaluation data.",
        ],
    }
    if command_failure:
        contract["command_failure"] = {
            "category": command_failure.get("category"),
            "summary": _shorten_text(str(command_failure.get("summary") or ""), 500),
            "repair_hint": _shorten_text(str(command_failure.get("repair_hint") or ""), 500),
        }
    return {key: value for key, value in contract.items() if value not in (None, "", [], {})}


def _activation_smoke_repair_evidence(smoke: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(smoke, dict) or not smoke:
        return {}
    comparison = smoke.get("comparison") if isinstance(smoke.get("comparison"), dict) else {}
    output_smoke = smoke.get("output_smoke") if isinstance(smoke.get("output_smoke"), dict) else {}
    eval_configs = smoke.get("eval_configs") if isinstance(smoke.get("eval_configs"), dict) else {}
    return {
        "status": smoke.get("status"),
        "reason": smoke.get("reason"),
        "repair_hint": smoke.get("repair_hint"),
        "switch": smoke.get("switch"),
        "datasets": smoke.get("datasets") or [],
        "eval_configs": {str(dataset): str(path) for dataset, path in eval_configs.items()},
        "enabled_metrics": smoke.get("enabled_metrics") or {},
        "disabled_metrics": smoke.get("disabled_metrics") or {},
        "metric_comparison": comparison.get("metric_comparison") or {
            key: comparison.get(key)
            for key in [
                "enabled_mean",
                "disabled_mean",
                "enabled_minus_disabled_mean",
                "dataset_enabled_minus_disabled",
                "max_abs_dataset_delta",
                "min_abs_metric_delta",
            ]
            if key in comparison
        },
        "prediction_comparison": comparison.get("prediction_comparison") or {},
        "mechanism_observed": comparison.get("mechanism_observed"),
        "mechanism_wired_metric_neutral": comparison.get("mechanism_wired_metric_neutral"),
        "mechanism_trace": smoke.get("mechanism_trace") or {},
        "output_smoke_red_flags": output_smoke.get("red_flags") or [],
    }


def _instrumentation_quality_repair_request(patch_result: dict[str, Any] | None) -> dict[str, Any]:
    mechanism_review = (patch_result or {}).get("mechanism_review") or {}
    quality_repair = mechanism_review.get("quality_repair") or {}
    if not isinstance(quality_repair, dict):
        return {"needed": False}
    return {
        "needed": bool(quality_repair.get("needed")),
        "deferred": bool(quality_repair.get("deferred", True)),
        "repair_route": "paperization",
        "trigger": "after_effect_found",
        "mode": quality_repair.get("mode") or "paperization_after_effect",
        "issues": list(quality_repair.get("issues") or mechanism_review.get("soft_issues") or []),
        "constraints": list(quality_repair.get("constraints") or []),
        "ablation_switch": quality_repair.get("ablation_switch"),
        "acceptance_guard": {
            "rerun_same_proxy_subset": True,
            "reject_if_enabled_proxy_regresses": True,
            "max_enabled_mean_delta_drop": 0.2,
            "default_behavior_must_remain_enabled": True,
        },
    }


def _c2c_paperization_readiness(best: dict[str, Any] | None, acceptance: dict[str, Any] | None) -> dict[str, Any]:
    if not best or not (acceptance or {}).get("passed"):
        return {
            "status": "waiting_for_effect",
            "reason": "effect-first discovery has not found a full-S3 accepted patch yet",
            "next_stage": "",
            "tasks": [],
        }
    proxy_quality = ((best.get("proxy_screen") or {}).get("quality_repair") or {})
    patch_review = ((best.get("patch_result") or {}).get("mechanism_review") or {})
    issues = list(dict.fromkeys((proxy_quality.get("issues") or []) + (patch_review.get("soft_issues") or [])))
    tasks = []
    if "ablation_switch_not_wired" in issues:
        tasks.append("Add an ablation switch that disables the discovered mechanism without changing enabled behavior.")
    if "missing_coverage_diagnostics_evidence" in issues:
        tasks.append("Add coverage diagnostics for accepted spans/routes/pathology buckets.")
    if "missing_matched_coverage_evidence" in issues:
        tasks.append("Add matched-coverage control bookkeeping for paper analysis.")
    if not tasks:
        tasks.append("Audit ablation, coverage diagnostics, and reviewer-facing evidence before paper writing.")
    return {
        "status": "ready",
        "reason": "effect-first discovery found a full-S3 accepted patch",
        "next_stage": "paperization",
        "candidate_id": best.get("id"),
        "tasks": tasks,
        "constraints": [
            "Do not change default enabled scoring/routing/loss/data sampling.",
            "Do not edit evaluator or metric computation.",
            "Rerun the same cheap proxy and full S3 acceptance checks after paperization.",
        ],
    }


def _candidate_avoid_rule(candidate: dict[str, Any], baseline_datasets: dict[str, Any]) -> str:
    regressions = candidate.get("dataset_regressions") or {}
    if regressions:
        worst_dataset = max(regressions, key=lambda key: regressions[key])
        return f"Do not repeat {candidate.get('id') or 'this candidate'} without addressing {worst_dataset} regression."
    attribution = candidate.get("failure_attribution") or {}
    if attribution.get("primary_failure") == "cheap_proxy_rejected_before_full_training":
        return f"Do not send {candidate.get('id') or 'this candidate'} to full S3 until cheap proxy risk is cleared."
    if attribution.get("primary_failure") == "repairable_proxy_risk_before_full_training":
        return f"Repair the S2.5 patch for {candidate.get('id') or 'this candidate'} before discarding the idea."
    if attribution.get("primary_failure") == "ablation_no_effect":
        return f"Do not repeat {candidate.get('id') or 'this candidate'} until its ablation switch changes enabled-vs-disabled behavior."
    dragging = attribution.get("dragging_datasets") or []
    if dragging and isinstance(dragging[0], dict):
        return f"Do not repeat {candidate.get('id') or 'this candidate'} without addressing {dragging[0].get('dataset')} regression evidence."
    proxy_screen = candidate.get("proxy_screen") or {}
    proxy_deltas = proxy_screen.get("proxy_dataset_deltas") or {}
    if proxy_deltas:
        try:
            worst_dataset = min(proxy_deltas, key=lambda key: float(proxy_deltas[key]))
            if float(proxy_deltas[worst_dataset]) < 0:
                return f"Do not repeat {candidate.get('id') or 'this candidate'} without repairing proxy regression on {worst_dataset}."
        except (TypeError, ValueError):
            pass
    command_failure = proxy_screen.get("command_failure") or {}
    if command_failure.get("category"):
        return f"Do not rerun {candidate.get('id') or 'this candidate'} until proxy command failure {command_failure.get('category')} is fixed."
    metrics = candidate.get("metrics") or {}
    datasets = metrics.get("datasets") or {}
    if datasets and baseline_datasets:
        deltas: dict[str, float] = {}
        for dataset, value in datasets.items():
            try:
                deltas[str(dataset)] = float(value) - float(baseline_datasets.get(dataset, value))
            except (TypeError, ValueError):
                continue
        if deltas:
            worst_dataset = min(deltas, key=lambda key: deltas[key])
            if deltas[worst_dataset] < 0:
                return f"Do not repeat {candidate.get('id') or 'this candidate'} without a guard for {worst_dataset}."
    if not metrics:
        return "Do not rerun without fixing preflight, checkpoint, or evaluator failures."
        return "Do not repeat this candidate without a new mechanism and explicit ablation."


def _c2c_proxy_calibration_iteration(payload: dict[str, Any], *, iteration: int) -> dict[str, Any]:
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    baseline_datasets = baseline.get("datasets") if isinstance(baseline.get("datasets"), dict) else {}
    acceptance = payload.get("acceptance") if isinstance(payload.get("acceptance"), dict) else {}
    candidates = [item for item in payload.get("candidate_results") or [] if isinstance(item, dict)]
    entries = [
        entry
        for entry in (_c2c_proxy_calibration_candidate(candidate, baseline_datasets, acceptance) for candidate in candidates)
        if entry
    ]
    false_positive_entries = [entry for entry in entries if entry.get("proxy_false_positive")]
    dataset_errors: dict[str, list[float]] = {}
    dataset_pairs: dict[str, list[tuple[float, float]]] = {}
    for entry in entries:
        for dataset, comparison in (entry.get("dataset_calibration") or {}).items():
            error = comparison.get("proxy_full_delta_error")
            if isinstance(error, (int, float)):
                dataset_errors.setdefault(dataset, []).append(float(error))
            if _is_number(comparison.get("proxy_delta")) and _is_number(comparison.get("full_delta")):
                dataset_pairs.setdefault(dataset, []).append((float(comparison["proxy_delta"]), float(comparison["full_delta"])))
    return {
        "timestamp": now_utc(),
        "iteration": iteration,
        "acceptance_passed": bool(acceptance.get("passed")),
        "full_s3_completed_candidate_count": len(entries),
        "candidate_count": len(entries),
        "proxy_false_positive_count": len(false_positive_entries),
        "proxy_false_positive_rate": round(len(false_positive_entries) / len(entries), 4) if entries else 0.0,
        "dataset_error_summary": _proxy_dataset_error_summary(dataset_errors, dataset_pairs=dataset_pairs),
        "proxy_full_delta_correlation": _proxy_delta_correlation(entries),
        "method_feedback": _proxy_calibration_method_feedback(entries),
        "candidates": entries,
    }


def _c2c_proxy_calibration_candidate(candidate: dict[str, Any], baseline_datasets: dict[str, Any], acceptance: dict[str, Any]) -> dict[str, Any] | None:
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    if proxy.get("status") != "passed":
        return None
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    full_datasets = metrics.get("datasets") if isinstance(metrics.get("datasets"), dict) else {}
    if not full_datasets:
        return None
    proxy_deltas = proxy.get("proxy_dataset_deltas") if isinstance(proxy.get("proxy_dataset_deltas"), dict) else {}
    if not proxy_deltas:
        proxy_baseline = proxy.get("proxy_baseline") or proxy.get("baseline_metrics") or {}
        proxy_scores = (proxy.get("metrics") or {}).get("datasets") or {}
        proxy_baseline_scores = proxy_baseline.get("datasets") if isinstance(proxy_baseline.get("datasets"), dict) else baseline_datasets
        proxy_deltas = {
            dataset: round(float(score) - float(proxy_baseline_scores[dataset]), 4)
            for dataset, score in proxy_scores.items()
            if dataset in proxy_baseline_scores and _is_number(score) and _is_number(proxy_baseline_scores[dataset])
        }
    full_deltas = {
        dataset: round(float(score) - float(baseline_datasets[dataset]), 4)
        for dataset, score in full_datasets.items()
        if dataset in baseline_datasets and _is_number(score) and _is_number(baseline_datasets[dataset])
    }
    dataset_calibration = {}
    for dataset in sorted(set(proxy_deltas) & set(full_deltas)):
        proxy_delta = float(proxy_deltas[dataset])
        full_delta = float(full_deltas[dataset])
        dataset_calibration[dataset] = {
            "proxy_delta": round(proxy_delta, 4),
            "full_delta": round(full_delta, 4),
            "proxy_full_delta_error": round(proxy_delta - full_delta, 4),
            "proxy_predicted_improvement": proxy_delta > 0,
            "full_improved": full_delta > 0,
            "proxy_mispredicted": (proxy_delta > 0) != (full_delta > 0),
        }
    full_passed = _candidate_full_passed(candidate, acceptance)
    proxy_mean_delta = proxy.get("proxy_delta_vs_baseline")
    full_mean_delta = candidate.get("delta_vs_baseline")
    mechanism_type = candidate.get("mechanism_type")
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    return {
        "id": candidate.get("id"),
        "title": candidate.get("title"),
        "mechanism_type": mechanism_type or contract.get("mechanism_type"),
        "mechanism_axis": candidate.get("mechanism_axis") or ((candidate.get("s2_variant") or {}).get("mechanism_axis") if isinstance(candidate.get("s2_variant"), dict) else None),
        "integration_point": candidate.get("integration_point") or ((candidate.get("s2_variant") or {}).get("integration_point") if isinstance(candidate.get("s2_variant"), dict) else None),
        "control_signal": candidate.get("control_signal") or ((candidate.get("s2_variant") or {}).get("control_signal") if isinstance(candidate.get("s2_variant"), dict) else None),
        "decision": candidate.get("decision"),
        "proxy_status": proxy.get("status"),
        "proxy_mean_delta": proxy_mean_delta,
        "full_mean_delta": full_mean_delta,
        "proxy_score": proxy.get("proxy_score"),
        "proxy_false_positive": not full_passed,
        "false_positive_reason": _proxy_false_positive_reason(
            candidate=candidate,
            acceptance=acceptance,
            full_passed=full_passed,
            proxy_mean_delta=proxy_mean_delta,
            full_mean_delta=full_mean_delta,
            dataset_calibration=dataset_calibration,
        ),
        "proxy_predicted_mean_improvement": _is_number(proxy_mean_delta) and float(proxy_mean_delta) > 0,
        "full_mean_improved": _is_number(full_mean_delta) and float(full_mean_delta) > 0,
        "dataset_calibration": dataset_calibration,
        "mispredicted_datasets": [
            dataset
            for dataset, item in dataset_calibration.items()
            if item.get("proxy_mispredicted")
        ],
    }


def _c2c_proxy_calibration_summary(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_entries = [
        candidate
        for iteration in iterations
        for candidate in iteration.get("candidates") or []
        if isinstance(candidate, dict)
    ]
    false_positive_entries = [entry for entry in candidate_entries if entry.get("proxy_false_positive")]
    dataset_errors: dict[str, list[float]] = {}
    dataset_pairs: dict[str, list[tuple[float, float]]] = {}
    dataset_mispredictions: dict[str, int] = {}
    mechanism_counts: dict[str, dict[str, int]] = {}
    mechanism_dataset_mispredictions: dict[str, dict[str, int]] = {}
    false_positive_reasons: dict[str, int] = {}
    for entry in candidate_entries:
        mechanism = str(entry.get("mechanism_type") or "unknown")
        stats = mechanism_counts.setdefault(mechanism, {"count": 0, "false_positive_count": 0, "proxy_positive_full_nonpositive_count": 0})
        stats["count"] += 1
        if entry.get("proxy_false_positive"):
            stats["false_positive_count"] += 1
            reason = str(entry.get("false_positive_reason") or "unknown")
            false_positive_reasons[reason] = false_positive_reasons.get(reason, 0) + 1
        if entry.get("proxy_predicted_mean_improvement") and not entry.get("full_mean_improved"):
            stats["proxy_positive_full_nonpositive_count"] += 1
        for dataset, comparison in (entry.get("dataset_calibration") or {}).items():
            error = comparison.get("proxy_full_delta_error")
            if isinstance(error, (int, float)):
                dataset_errors.setdefault(dataset, []).append(float(error))
            if _is_number(comparison.get("proxy_delta")) and _is_number(comparison.get("full_delta")):
                dataset_pairs.setdefault(dataset, []).append((float(comparison["proxy_delta"]), float(comparison["full_delta"])))
            if comparison.get("proxy_mispredicted"):
                dataset_mispredictions[dataset] = dataset_mispredictions.get(dataset, 0) + 1
                per_mechanism = mechanism_dataset_mispredictions.setdefault(mechanism, {})
                per_mechanism[dataset] = per_mechanism.get(dataset, 0) + 1
    summary = {
        "candidate_count": len(candidate_entries),
        "proxy_false_positive_count": len(false_positive_entries),
        "proxy_false_positive_rate": round(len(false_positive_entries) / len(candidate_entries), 4) if candidate_entries else 0.0,
        "false_positive_reasons": dict(sorted(false_positive_reasons.items())),
        "proxy_full_delta_correlation": _proxy_delta_correlation(candidate_entries),
        "dataset_error_summary": _proxy_dataset_error_summary(dataset_errors, mispredictions=dataset_mispredictions, dataset_pairs=dataset_pairs),
        "mechanism_false_positive_summary": {
            mechanism: {
                **stats,
                "false_positive_rate": round(stats["false_positive_count"] / stats["count"], 4) if stats["count"] else 0.0,
                "proxy_positive_full_nonpositive_rate": round(stats["proxy_positive_full_nonpositive_count"] / stats["count"], 4) if stats["count"] else 0.0,
                "mispredicted_datasets": dict(sorted(mechanism_dataset_mispredictions.get(mechanism, {}).items())),
            }
            for mechanism, stats in sorted(mechanism_counts.items())
        },
    }
    summary["method_feedback"] = _proxy_calibration_method_feedback(candidate_entries)
    return summary


def _proxy_calibration_method_feedback(entries: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_stats: dict[str, dict[str, int]] = {}
    mechanism_stats: dict[str, dict[str, Any]] = {}
    integration_stats: dict[str, dict[str, Any]] = {}
    for entry in entries:
        mechanism = str(entry.get("mechanism_type") or "unknown")
        integration = str(entry.get("integration_point") or "unknown")
        mech = mechanism_stats.setdefault(mechanism, {"count": 0, "false_positive_count": 0, "mispredicted_datasets": {}})
        integ = integration_stats.setdefault(integration, {"count": 0, "false_positive_count": 0, "mechanisms": {}})
        mech["count"] += 1
        integ["count"] += 1
        integ["mechanisms"][mechanism] = int(integ["mechanisms"].get(mechanism, 0)) + 1
        if entry.get("proxy_false_positive"):
            mech["false_positive_count"] += 1
            integ["false_positive_count"] += 1
        for dataset, comparison in (entry.get("dataset_calibration") or {}).items():
            stats = dataset_stats.setdefault(str(dataset), {"count": 0, "misprediction_count": 0, "proxy_positive_full_negative_count": 0})
            stats["count"] += 1
            if comparison.get("proxy_mispredicted"):
                stats["misprediction_count"] += 1
                mech["mispredicted_datasets"][str(dataset)] = int(mech["mispredicted_datasets"].get(str(dataset), 0)) + 1
            if comparison.get("proxy_delta", 0) > 0 and comparison.get("full_delta", 0) <= 0:
                stats["proxy_positive_full_negative_count"] += 1

    risky_datasets = [
        {
            "dataset": dataset,
            **stats,
            "misprediction_rate": round(stats["misprediction_count"] / stats["count"], 4) if stats["count"] else 0.0,
            "proxy_positive_full_negative_rate": round(stats["proxy_positive_full_negative_count"] / stats["count"], 4) if stats["count"] else 0.0,
        }
        for dataset, stats in sorted(dataset_stats.items())
        if stats.get("misprediction_count") or stats.get("proxy_positive_full_negative_count")
    ]
    risky_mechanisms = [
        {
            "mechanism_type": mechanism,
            "count": stats["count"],
            "false_positive_count": stats["false_positive_count"],
            "false_positive_rate": round(stats["false_positive_count"] / stats["count"], 4) if stats["count"] else 0.0,
            "mispredicted_datasets": dict(sorted(stats["mispredicted_datasets"].items())),
        }
        for mechanism, stats in sorted(mechanism_stats.items())
        if stats.get("false_positive_count") or stats.get("mispredicted_datasets")
    ]
    risky_integration_points = [
        {
            "integration_point": point,
            "count": stats["count"],
            "false_positive_count": stats["false_positive_count"],
            "false_positive_rate": round(stats["false_positive_count"] / stats["count"], 4) if stats["count"] else 0.0,
            "mechanisms": dict(sorted(stats["mechanisms"].items())),
        }
        for point, stats in sorted(integration_stats.items())
        if stats.get("false_positive_count")
    ]
    risky_datasets.sort(key=lambda item: (item["misprediction_rate"], item["proxy_positive_full_negative_rate"], item["dataset"]), reverse=True)
    risky_mechanisms.sort(key=lambda item: (item["false_positive_rate"], item["false_positive_count"], item["mechanism_type"]), reverse=True)
    risky_integration_points.sort(key=lambda item: (item["false_positive_rate"], item["false_positive_count"], item["integration_point"]), reverse=True)
    recommendations = []
    if risky_datasets:
        recommendations.append("Calibrate cheap proxy around risky datasets: " + ", ".join(item["dataset"] for item in risky_datasets[:3]) + ".")
    if risky_mechanisms:
        recommendations.append("Downweight or demand stronger activation/full-readiness evidence for proxy false-positive mechanisms: " + ", ".join(item["mechanism_type"] for item in risky_mechanisms[:3]) + ".")
    if risky_integration_points:
        recommendations.append("Penalize integration points with repeated proxy false positives: " + ", ".join(item["integration_point"] for item in risky_integration_points[:3]) + ".")
    return {
        "status": "ok" if entries else "no_full_s3_proxy_pairs",
        "risky_datasets": risky_datasets[:8],
        "risky_mechanisms": risky_mechanisms[:8],
        "risky_integration_points": risky_integration_points[:8],
        "recommendations": recommendations,
    }


def _proxy_dataset_error_summary(
    dataset_errors: dict[str, list[float]],
    *,
    mispredictions: dict[str, int] | None = None,
    dataset_pairs: dict[str, list[tuple[float, float]]] | None = None,
) -> dict[str, Any]:
    mispredictions = mispredictions or {}
    dataset_pairs = dataset_pairs or {}
    return {
        dataset: {
            "mean_abs_proxy_full_delta_error": round(sum(abs(item) for item in values) / len(values), 4),
            "max_abs_proxy_full_delta_error": round(max(abs(item) for item in values), 4),
            "misprediction_count": mispredictions.get(dataset, 0),
            "proxy_full_delta_correlation": _pearson_correlation(dataset_pairs.get(dataset) or []),
            "count": len(values),
        }
        for dataset, values in sorted(dataset_errors.items())
        if values
    }


def _proxy_delta_correlation(entries: list[dict[str, Any]]) -> float | None:
    pairs = [
        (float(entry["proxy_mean_delta"]), float(entry["full_mean_delta"]))
        for entry in entries
        if _is_number(entry.get("proxy_mean_delta")) and _is_number(entry.get("full_mean_delta"))
    ]
    return _pearson_correlation(pairs)


def _pearson_correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    if denom_x <= 0 or denom_y <= 0:
        return None
    return round(numerator / ((denom_x * denom_y) ** 0.5), 4)


def _proxy_false_positive_reason(
    *,
    candidate: dict[str, Any],
    acceptance: dict[str, Any],
    full_passed: bool,
    proxy_mean_delta: Any,
    full_mean_delta: Any,
    dataset_calibration: dict[str, Any],
) -> str:
    if full_passed:
        return ""
    if _is_number(proxy_mean_delta) and float(proxy_mean_delta) > 0 and _is_number(full_mean_delta) and float(full_mean_delta) <= 0:
        return "proxy_mean_positive_full_mean_nonpositive"
    if any(item.get("proxy_mispredicted") for item in dataset_calibration.values()):
        return "dataset_direction_misprediction"
    if candidate.get("decision") == "not_viable":
        return "full_acceptance_not_viable"
    return str(acceptance.get("reason") or candidate.get("decision") or "full_train_not_accepted")


def _candidate_full_passed(candidate: dict[str, Any], acceptance: dict[str, Any]) -> bool:
    if candidate.get("decision") != "candidate_win":
        return False
    if bool(acceptance.get("passed")):
        return True
    return _is_number(candidate.get("delta_vs_baseline")) and float(candidate.get("delta_vs_baseline")) > 0


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _coerce_float(value: Any, *, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _c2c_proxy_artifact_paths(run_spec: dict[str, Any]) -> dict[str, str]:
    proxy_spec = run_spec.get("proxy_screen") if isinstance(run_spec.get("proxy_screen"), dict) else {}
    paths = {
        "run_state": run_spec.get("run_state_path"),
        "candidate_run_root": run_spec.get("run_root"),
        "proxy_run_root": proxy_spec.get("run_root"),
        "proxy_metrics": proxy_spec.get("metrics_path"),
        "proxy_train_config": proxy_spec.get("train_config"),
        "proxy_baseline_metrics": proxy_spec.get("baseline_metrics_path"),
    }
    return {key: str(value) for key, value in paths.items() if value}


def _normalize_c2c_proxy_screen_artifacts(
    proxy_screen: dict[str, Any],
    *,
    full_baseline: dict[str, Any],
    run_spec: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(proxy_screen, dict):
        return proxy_screen
    normalized = dict(proxy_screen)
    proxy_baseline = normalized.get("proxy_baseline")
    if not isinstance(proxy_baseline, dict) or not proxy_baseline:
        proxy_baseline = normalized.get("baseline_metrics") if isinstance(normalized.get("baseline_metrics"), dict) else None
    comparison_baseline = normalized.get("comparison_baseline")
    if not isinstance(comparison_baseline, dict) or not comparison_baseline:
        comparison_baseline = proxy_baseline or full_baseline
    normalized.setdefault("full_baseline_mean", _coerce_float((full_baseline or {}).get("mean"), default=None))
    normalized.setdefault("proxy_baseline", proxy_baseline)
    normalized.setdefault("proxy_baseline_mean", _coerce_float((proxy_baseline or {}).get("mean"), default=None) if isinstance(proxy_baseline, dict) else None)
    normalized.setdefault("comparison_baseline", comparison_baseline)
    normalized.setdefault("comparison_baseline_mean", _coerce_float((comparison_baseline or {}).get("mean"), default=normalized.get("full_baseline_mean")) if isinstance(comparison_baseline, dict) else normalized.get("full_baseline_mean"))
    if normalized.get("proxy_delta_vs_comparison_baseline") is None and normalized.get("proxy_delta_vs_baseline") is not None:
        normalized["proxy_delta_vs_comparison_baseline"] = normalized.get("proxy_delta_vs_baseline")
    if normalized.get("proxy_delta_vs_proxy_baseline") is None and proxy_baseline and normalized.get("proxy_delta_vs_baseline") is not None:
        normalized["proxy_delta_vs_proxy_baseline"] = normalized.get("proxy_delta_vs_baseline")
    command_failure = normalized.get("command_failure") if isinstance(normalized.get("command_failure"), dict) else {}
    resource_retry = (
        normalized.get("resource_retry") is True
        or normalized.get("status") == "resource_retry"
        or normalized.get("failure_category") in {"s3_proxy_resource_oom", "s3_proxy_gpu_resource_retry"}
        or command_failure.get("category") == "resource_oom"
    )
    if resource_retry:
        normalized["status"] = "resource_retry"
        normalized["resource_retry"] = True
        normalized["failure_category"] = (
            normalized.get("failure_category")
            if normalized.get("failure_category") in {"s3_proxy_resource_oom", "s3_proxy_gpu_resource_retry"}
            else "s3_proxy_resource_oom"
        )
        normalized["repair_route"] = "resource_retry"
        normalized["repair_mode"] = "resource_retry"
        normalized.setdefault(
            "reason",
            "cheap proxy command hit a GPU resource failure; resume S3 when resources are available",
        )
        if not normalized.get("repair_hint"):
            normalized["repair_hint"] = (
                "do not repair the S2.5 patch for this failure; rerun cheap proxy after GPU memory is available"
            )
    if not isinstance(normalized.get("artifact_paths"), dict):
        normalized["artifact_paths"] = _c2c_proxy_artifact_paths(run_spec)
    return normalized


def _compact_c2c_result_payload(value: Any) -> Any:
    if isinstance(value, dict):
        if _looks_like_candidate_result(value):
            return _compact_candidate_result(value)
        return {str(key): _compact_c2c_result_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_c2c_result_payload(item) for item in value]
    return value


def _c2c_manifest_selected_patch_entry(manifest: dict[str, Any], selected_candidate_id: str) -> dict[str, Any]:
    if not isinstance(manifest, dict) or not selected_candidate_id:
        return {}
    for entry in manifest.get("candidates") or manifest.get("patches") or []:
        if isinstance(entry, dict) and str(entry.get("candidate_id") or entry.get("id") or "") == selected_candidate_id:
            return dict(entry)
    selected_patch = manifest.get("selected_patch") if isinstance(manifest.get("selected_patch"), dict) else {}
    if str(selected_patch.get("candidate_id") or selected_patch.get("id") or "") == selected_candidate_id:
        return dict(selected_patch)
    return dict(selected_patch) if selected_patch else {}


def _c2c_code_patch_from_manifest_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    fields = [
        "status",
        "patch_json",
        "diff",
        "rationale",
        "validation",
        "implementation_contract",
        "codex_prompt",
        "changed_files",
        "has_executable_change",
        "quality_score",
        "quality_debt",
        "recovery_actions",
        "reason",
        "risk_check",
        "mechanism_review",
        "codex_session_id",
        "code_worktree",
        "patched_repo_snapshot",
    ]
    code_patch = {key: copy.deepcopy(entry[key]) for key in fields if key in entry}
    code_patch.setdefault("status", entry.get("status") or "missing")
    if "has_executable_change" not in code_patch:
        code_patch["has_executable_change"] = bool(entry.get("changed_files") or entry.get("patch_json") or entry.get("patched_repo_snapshot"))
    return code_patch


def _c2c_selection_artifact_lock(project_root: Path, rel_path: Any) -> dict[str, Any]:
    if not rel_path:
        return {"rel_path": None, "exists": False, "sha256": None}
    rel = str(rel_path)
    path = project_root / rel
    exists = path.exists() and path.is_file()
    return {
        "rel_path": rel,
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
    }


def _finite_proxy_mean(proxy_screen: dict[str, Any]) -> float | None:
    if str(proxy_screen.get("status") or "").strip().lower() in {"failed", "blocked", "resource_retry", "baseline_blocked"}:
        return None
    metrics = proxy_screen.get("metrics") if isinstance(proxy_screen.get("metrics"), dict) else {}
    try:
        mean = float(metrics.get("mean"))
    except (TypeError, ValueError):
        return None
    return mean if math.isfinite(mean) else None


def _config_with_c2c_snapshot_path(config: dict[str, Any], repo_root: str) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    cloned.setdefault("c2c", {})
    cloned["c2c"]["snapshot_path"] = repo_root
    return cloned


def _c2c_execution_repo_path_audit(
    *,
    original_repo_root: Path,
    execution_repo: dict[str, Any],
    run_spec: dict[str, Any],
) -> dict[str, Any]:
    execution_root = Path(str(execution_repo.get("repo_root") or ""))
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def check_path(name: str, path_value: Any) -> None:
        if not path_value:
            return
        path = Path(path_value).resolve()
        inside_execution = _path_is_relative_to(path, execution_root.resolve())
        inside_original = _path_is_relative_to(path, original_repo_root.resolve())
        item = {
            "name": name,
            "path": str(path),
            "inside_execution_repo": inside_execution,
            "inside_original_snapshot": inside_original,
        }
        checks.append(item)
        if not inside_execution:
            errors.append(f"{name} is outside execution repo: {path}")
        if inside_original:
            errors.append(f"{name} points into original snapshot: {path}")

    check_path("run_root", run_spec.get("run_root"))
    check_path("train_config", run_spec.get("train_config"))
    check_path("run_state_path", run_spec.get("run_state_path"))
    check_path("preflight_path", run_spec.get("preflight_path"))
    for dataset, path in (run_spec.get("eval_configs") or {}).items():
        check_path(f"eval_config:{dataset}", path)
    proxy_spec = run_spec.get("proxy_screen") if isinstance(run_spec.get("proxy_screen"), dict) else {}
    check_path("proxy_run_root", proxy_spec.get("run_root"))
    check_path("proxy_train_config", proxy_spec.get("train_config"))
    for dataset, path in (proxy_spec.get("eval_configs") or {}).items():
        check_path(f"proxy_eval_config:{dataset}", path)

    return {
        "schema_version": "c2c_execution_repo_audit_v1",
        "status": "failed" if errors else "ok",
        "reason": "; ".join(errors[:4]) if errors else "all C2C run/config paths are inside execution repo",
        "original_repo_root": str(original_repo_root),
        "execution_repo_root": str(execution_root),
        "checks": checks,
        "errors": errors,
    }


def _c2c_execution_repo_output_audit(
    *,
    original_repo_root: Path,
    run_spec: dict[str, Any],
    before_state: dict[str, dict[str, Any]],
    path_audit: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    pollution = _c2c_original_output_pollution(original_repo_root, run_spec, before_state)
    failed = path_audit.get("status") == "failed" or bool(pollution.get("added_files") or pollution.get("modified_files"))
    reason = ""
    if path_audit.get("status") == "failed":
        reason = str(path_audit.get("reason") or "run paths are outside execution repo")
    elif failed:
        reason = "original C2C snapshot received new or modified run-output files during S3"
    return {
        "schema_version": "c2c_execution_repo_audit_v1",
        "status": "failed" if failed else "ok",
        "phase": phase,
        "reason": reason or "execution repo output audit passed",
        "path_audit": path_audit,
        "output_pollution": pollution,
    }


def _c2c_original_output_state(original_repo_root: Path, run_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tracked: dict[str, dict[str, Any]] = {}
    for root in _c2c_original_output_roots(original_repo_root, run_spec):
        if not root.exists():
            continue
        if root.is_file():
            rel = root.relative_to(original_repo_root).as_posix()
            tracked[rel] = _c2c_output_file_state(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(original_repo_root).as_posix()
            tracked[rel] = _c2c_output_file_state(path)
    return tracked


def _c2c_original_output_pollution(
    original_repo_root: Path,
    run_spec: dict[str, Any],
    before_state: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    after_state = _c2c_original_output_state(original_repo_root, run_spec)
    added = sorted(rel for rel in after_state if rel not in before_state)
    modified = sorted(
        rel
        for rel, state in after_state.items()
        if rel in before_state and state.get("sha256") != before_state[rel].get("sha256")
    )
    return {
        "status": "failed" if added or modified else "ok",
        "tracked_roots": [
            _relpath_or_abs(path, original_repo_root)
            for path in _c2c_original_output_roots(original_repo_root, run_spec)
        ],
        "added_files": added[:100],
        "modified_files": modified[:100],
        "added_count": len(added),
        "modified_count": len(modified),
    }


def _c2c_original_output_roots(original_repo_root: Path, run_spec: dict[str, Any]) -> list[Path]:
    run_id = sanitize_filename(str(run_spec.get("run_id") or ""))
    roots = [
        original_repo_root / "local" / "auto_research_runs" / run_id,
        original_repo_root / "local" / "auto_research_runs" / "proxy_baseline",
    ]
    return list(dict.fromkeys(roots))


def _c2c_output_file_state(path: Path) -> dict[str, Any]:
    try:
        return {"sha256": sha256_file(path), "size": path.stat().st_size}
    except OSError:
        return {"sha256": None, "size": None}


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _relpath_or_abs(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _materialize_c2c_execution_repo(execution_repo: Path, source_repo: Path, source_lock: dict[str, Any]) -> bool:
    metadata_path = execution_repo / ".auto_research_execution_source.json"
    existing_lock = read_json(metadata_path, default={}) if metadata_path.exists() else {}
    if execution_repo.exists() and existing_lock == source_lock:
        return True
    if execution_repo.exists():
        shutil.rmtree(execution_repo)
    shutil.copytree(source_repo, execution_repo, ignore=_c2c_execution_repo_ignore)
    write_json(metadata_path, source_lock)
    return False


def _c2c_execution_repo_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {".git", "wandb", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov"}
    if Path(directory).name == "local":
        ignored.update({"auto_research_runs", "checkpoints", "snapshots", "final_results"})
    for name in names:
        if name in ignored:
            continue
        if Path(name).suffix in {".pyc", ".pyo", ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".parquet", ".arrow"}:
            ignored.add(name)
    return ignored.intersection(names)


def _looks_like_candidate_result(value: dict[str, Any]) -> bool:
    return any(key in value for key in ["patch_result", "proxy_screen", "command_status", "run_state_path"]) and any(
        key in value for key in ["id", "candidate_id", "run_id"]
    )


def _compact_candidate_result(candidate: dict[str, Any]) -> dict[str, Any]:
    compact = dict(candidate)
    if isinstance(compact.get("patch_result"), dict):
        compact["patch_result"] = _compact_patch_result_for_payload(compact["patch_result"])
        if isinstance(compact.get("proxy_screen"), dict):
            compact["proxy_screen"] = _compact_proxy_screen(compact["proxy_screen"])
        if isinstance(compact.get("activation_smoke"), dict):
            compact["activation_smoke"] = _compact_activation_smoke(compact["activation_smoke"])
        if isinstance(compact.get("commands"), dict):
            compact["commands"] = _compact_command_plan(compact["commands"])
    if isinstance(compact.get("command_logs"), list):
        compact["command_logs"] = _compact_event_logs(compact["command_logs"])
    if isinstance(compact.get("ablation"), dict):
        compact["ablation"] = _compact_ablation_payload(compact["ablation"])
    if isinstance(compact.get("failure_attribution"), dict):
        compact["failure_attribution"] = _compact_failure_attribution(compact["failure_attribution"])
    if isinstance(compact.get("execution_repo_audit"), dict):
        compact["execution_repo_audit"] = _compact_execution_repo_audit(compact["execution_repo_audit"])
    return compact


def _compact_patch_result_for_payload(patch_result: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "status",
        "reason",
        "candidate_id",
        "patch_status",
        "changed_files",
        "errors",
        "retryable",
        "failure_category",
    ]
    compact = {key: patch_result.get(key) for key in keep if key in patch_result}
    restore_state = patch_result.get("restore_state") or []
    if isinstance(restore_state, list) and restore_state:
        compact["restore_state_summary"] = [
            {
                "path": item.get("path"),
                "existed": item.get("existed"),
                "content_sha256": _sha256_text(str(item.get("content") or "")) if item.get("existed") else None,
                "content_chars": len(str(item.get("content") or "")) if item.get("existed") else 0,
            }
            for item in restore_state
            if isinstance(item, dict)
        ][:40]
        compact["restore_state_omitted"] = True
    for key in ["risk_check", "activation_check", "mechanism_review", "quality_score"]:
        if isinstance(patch_result.get(key), dict):
            compact[key] = _compact_c2c_result_payload(patch_result[key])
    return {key: value for key, value in compact.items() if value not in (None, [], {})}


def _compact_proxy_screen(proxy_screen: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(proxy_screen, dict):
        return proxy_screen
    compact: dict[str, Any] = {}
    for key in [
        "enabled",
        "status",
        "mode",
        "reason",
        "repair_hint",
        "repair_route",
        "repair_mode",
        "proxy_effect_repair_contract",
        "baseline_status",
        "baseline_failure",
        "baseline_attempt_count",
        "metrics",
        "baseline_metrics",
        "full_baseline_mean",
        "proxy_baseline",
        "proxy_baseline_mean",
        "comparison_baseline_mean",
        "proxy_delta_vs_baseline",
        "proxy_delta_vs_comparison_baseline",
        "proxy_delta_vs_proxy_baseline",
        "proxy_dataset_deltas",
        "proxy_dataset_regressions",
        "proxy_worst_dataset_regression",
        "proxy_score",
        "proxy_decision_mode",
        "eval_smoke",
        "proxy_eval_health_failure",
        "activation_smoke",
        "soft_fail",
        "soft_flags",
        "signals",
        "patch_risk",
        "quality_repair",
        "proxy_decision_report",
        "full_s3_worthiness",
        "artifact_paths",
        "patch_fingerprint",
    ]:
        if key in proxy_screen:
            compact[key] = proxy_screen[key]
    if isinstance(proxy_screen.get("command_failure"), dict):
        compact["command_failure"] = _compact_attempt(proxy_screen["command_failure"])
    if isinstance(proxy_screen.get("attempts"), list):
        compact["attempts"] = _compact_attempts(proxy_screen["attempts"])
    if isinstance(proxy_screen.get("commands"), list):
        compact["commands"] = [_shorten_text(str(command), 240) for command in proxy_screen["commands"]]
        compact["command_count"] = len(proxy_screen["commands"])
    if isinstance(compact.get("activation_smoke"), dict):
        compact["activation_smoke"] = _compact_activation_smoke(compact["activation_smoke"])
    if isinstance(proxy_screen.get("full_s3_readiness"), dict):
        compact["full_s3_readiness"] = _compact_full_s3_readiness(proxy_screen["full_s3_readiness"])
    if isinstance(compact.get("proxy_effect_repair_contract"), dict):
        compact["proxy_effect_repair_contract"] = _compact_proxy_effect_repair_contract(compact["proxy_effect_repair_contract"])
    return _compact_c2c_result_payload(compact)


def _compact_proxy_decision_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return report
    compact = {
        "candidate_id": report.get("candidate_id"),
        "decision": report.get("decision"),
        "failure_class": report.get("failure_class"),
        "route_hint": report.get("route_hint"),
        "reason_codes": list(report.get("reason_codes") or [])[:8],
        "path": "experiment/results/c2c_proxy_decision_report.json",
    }
    if isinstance(report.get("full_s3_worthiness"), dict):
        compact["full_s3_worthiness"] = report["full_s3_worthiness"]
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _proxy_screen_for_failure_attribution(proxy_screen: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "status",
        "reason",
        "repair_hint",
        "repair_route",
        "repair_mode",
        "full_baseline_mean",
        "proxy_baseline_mean",
        "comparison_baseline_mean",
        "proxy_delta_vs_baseline",
        "proxy_delta_vs_comparison_baseline",
        "proxy_delta_vs_proxy_baseline",
        "proxy_dataset_deltas",
        "proxy_dataset_regressions",
        "proxy_worst_dataset_regression",
        "proxy_score",
        "eval_smoke",
        "proxy_eval_health_failure",
        "activation_smoke",
        "full_s3_readiness",
        "soft_fail",
        "soft_flags",
        "artifact_paths",
        "proxy_decision_report",
        "full_s3_worthiness",
    ]
    compact = {key: proxy_screen.get(key) for key in keep if key in proxy_screen}
    if isinstance(proxy_screen.get("command_failure"), dict):
        compact["command_failure"] = _compact_attempt(proxy_screen["command_failure"], stdout_chars=0, stderr_chars=0)
    if isinstance(proxy_screen.get("proxy_effect_repair_contract"), dict):
        compact["proxy_effect_repair_contract"] = _compact_proxy_effect_repair_contract(proxy_screen["proxy_effect_repair_contract"])
    if isinstance(compact.get("activation_smoke"), dict):
        compact["activation_smoke"] = _compact_activation_smoke(compact["activation_smoke"])
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_proxy_effect_repair_contract(contract: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "mode",
        "source",
        "goal",
        "reason",
        "repair_hint",
        "soft_flags",
        "full_baseline_mean",
        "proxy_baseline_mean",
        "comparison_baseline_mean",
        "proxy_delta_vs_baseline",
        "proxy_delta_vs_comparison_baseline",
        "proxy_delta_vs_proxy_baseline",
        "proxy_score",
        "proxy_worst_dataset_regression",
        "proxy_dataset_deltas",
        "proxy_dataset_regressions",
        "eval_smoke",
        "proxy_eval_health_failure",
        "activation_smoke",
        "readiness_blockers",
        "readiness_warnings",
        "dragging_datasets",
        "improved_datasets",
        "patch_risk_labels",
        "changed_files",
        "config_override_keys",
        "repair_priorities",
        "forbidden",
    ]
    compact = {key: contract.get(key) for key in keep if key in contract}
    command_failure = contract.get("command_failure") or {}
    if isinstance(command_failure, dict):
        compact["command_failure"] = {
            key: _shorten_text(str(command_failure.get(key) or ""), 500)
            for key in ["category", "summary", "repair_hint"]
            if command_failure.get(key) not in (None, "")
        }
    if isinstance(compact.get("activation_smoke"), dict):
        compact["activation_smoke"] = _compact_activation_smoke(compact["activation_smoke"])
    if isinstance(contract.get("full_s3_readiness"), dict):
        readiness = contract["full_s3_readiness"]
        compact["full_s3_readiness"] = {
            key: readiness.get(key)
            for key in ["candidate_id", "run_id", "full_train_allowed", "status"]
            if key in readiness
        }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_attempts(attempts: list[Any], *, stdout_chars: int = 1000, stderr_chars: int = 1600) -> list[Any]:
    return [
        _compact_attempt(attempt, stdout_chars=stdout_chars, stderr_chars=stderr_chars)
        if isinstance(attempt, dict)
        else attempt
        for attempt in attempts
    ]


def _compact_attempt(attempt: dict[str, Any], *, stdout_chars: int = 1000, stderr_chars: int = 1600) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in [
        "step",
        "name",
        "status",
        "returncode",
        "category",
        "summary",
        "repair_hint",
        "elapsed_seconds",
        "timeout_seconds",
        "started_at",
        "completed_at",
    ]:
        if key in attempt:
            compact[key] = attempt[key]
    if "command" in attempt:
        compact["command"] = _shorten_text(str(attempt.get("command") or ""), 600)
    if "stdout" in attempt:
        compact["stdout_tail"] = str(attempt.get("stdout") or "")[-stdout_chars:]
        compact["stdout_chars"] = len(str(attempt.get("stdout") or ""))
    if "stderr" in attempt:
        compact["stderr_tail"] = str(attempt.get("stderr") or "")[-stderr_chars:]
        compact["stderr_chars"] = len(str(attempt.get("stderr") or ""))
    if isinstance(attempt.get("attempts"), list):
        compact["attempts"] = _compact_attempts(attempt["attempts"], stdout_chars=stdout_chars, stderr_chars=stderr_chars)
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _parse_torch_distributed_rank_failure(text: str) -> dict[str, Any] | None:
    local_rank = re.search(r"local_rank:\s*(\d+)", text)
    exitcode = re.search(r"exitcode\s*:\s*(-?\d+)", text)
    pid = re.search(r"pid\s*:\s*(\d+)", text)
    if not any([local_rank, exitcode, pid]):
        return None
    return {
        "local_rank": int(local_rank.group(1)) if local_rank else None,
        "exitcode": int(exitcode.group(1)) if exitcode else None,
        "pid": int(pid.group(1)) if pid else None,
    }


def _c2c_gpu_resource_wait_enabled(policy: dict[str, Any]) -> bool:
    wait_cfg = policy.get("resource_wait") if isinstance(policy.get("resource_wait"), dict) else {}
    return bool(wait_cfg.get("enabled", False)) and int(wait_cfg.get("timeout_seconds") or 0) > 0


def _c2c_payload_is_retryable_pause(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    candidates = [item for item in payload.get("candidate_results") or [] if isinstance(item, dict)]
    if candidates and all(is_retryable_c2c_candidate(item) for item in candidates):
        return True
    for key in ["best_candidate", "best_proxy_candidate"]:
        candidate = payload.get(key)
        if isinstance(candidate, dict) and is_retryable_c2c_candidate(candidate):
            return True
    acceptance = payload.get("acceptance") if isinstance(payload.get("acceptance"), dict) else {}
    reason = str(acceptance.get("reason") or payload.get("blocked_reason") or "").lower()
    return any(
        marker in reason
        for marker in [
            "resource retry",
            "waiting for gpu resources",
            "gpu resources",
            "cuda out of memory",
            "quota",
            "rate limit",
        ]
    )


def _compact_command_plan(commands: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in commands.items():
        values = value if isinstance(value, list) else [value]
        compact[key] = [_shorten_text(str(item), 600) for item in values]
    return compact


def _compact_event_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_logs = []
    for item in logs:
        if not isinstance(item, dict):
            continue
        compact = dict(item)
        if isinstance(compact.get("patch_result"), dict):
            compact["patch_result"] = _compact_patch_result_for_payload(compact["patch_result"])
        if isinstance(compact.get("proxy_screen"), dict):
            proxy_screen = compact["proxy_screen"]
            compact["proxy_screen"] = {
                key: proxy_screen.get(key)
                for key in [
                    "enabled",
                    "status",
                    "mode",
                    "reason",
                    "repair_mode",
                    "proxy_delta_vs_baseline",
                    "proxy_delta_vs_comparison_baseline",
                    "proxy_score",
                ]
                if key in proxy_screen
            }
            if isinstance(proxy_screen.get("command_failure"), dict):
                compact["proxy_screen"]["command_failure"] = _compact_attempt(
                    proxy_screen["command_failure"],
                    stdout_chars=0,
                    stderr_chars=0,
                )
        if isinstance(compact.get("activation_smoke"), dict):
            compact["activation_smoke"] = _compact_activation_smoke(compact["activation_smoke"])
        if isinstance(compact.get("execution_repo_audit"), dict):
            compact["execution_repo_audit"] = _compact_execution_repo_audit(compact["execution_repo_audit"])
        if isinstance(compact.get("audit"), dict):
            compact["audit"] = _compact_execution_repo_audit(compact["audit"])
        if isinstance(compact.get("ablation"), dict):
            compact["ablation"] = _compact_ablation_payload(compact["ablation"])
        compact_logs.append(compact)
    return compact_logs


def _compact_ablation_payload(ablation: dict[str, Any]) -> dict[str, Any]:
    compact = dict(ablation)
    if isinstance(compact.get("attempts"), list):
        compact["attempts"] = _compact_attempts(compact["attempts"])
    if isinstance(compact.get("eval_by_dataset"), dict):
        compact["eval_by_dataset"] = {
            dataset: _compact_attempt(attempt) if isinstance(attempt, dict) else attempt
            for dataset, attempt in compact["eval_by_dataset"].items()
        }
    if isinstance(compact.get("commands"), dict):
        compact["commands"] = _compact_command_plan(compact["commands"])
    return compact


def _compact_activation_smoke(smoke: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(smoke, dict):
        return smoke
    compact = _compact_c2c_result_payload(dict(smoke))
    if isinstance(compact.get("attempts"), list):
        compact["attempts"] = _compact_attempts(compact["attempts"])
    if isinstance(compact.get("eval_by_dataset"), dict):
        compact["eval_by_dataset"] = {
            dataset: _compact_attempt(attempt) if isinstance(attempt, dict) else attempt
            for dataset, attempt in compact["eval_by_dataset"].items()
        }
    if isinstance(compact.get("commands"), dict):
        compact["commands"] = _compact_command_plan(compact["commands"])
    for key in ["run_root", "result_root", "metrics_path"]:
        if key in compact:
            compact[key] = str(compact[key])
    if isinstance(compact.get("eval_configs"), dict):
            compact["eval_configs"] = {str(dataset): str(path) for dataset, path in compact["eval_configs"].items()}
    return compact


def _compact_full_s3_readiness(readiness: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(readiness, dict):
        return readiness
    compact = {
        key: readiness.get(key)
        for key in [
            "schema_version",
            "project_id",
            "iteration",
            "created_at",
            "candidate_id",
            "candidate_title",
            "run_id",
            "full_train_allowed",
            "status",
            "static_risk",
            "proxy",
            "eval_smoke",
            "activation_smoke",
            "ablation_switch",
            "acceptance_targets",
            "worth_full_train",
            "artifact_paths",
        ]
        if key in readiness
    }
    return _compact_c2c_result_payload(compact)


def _compact_failure_attribution(attribution: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "primary_failure",
        "dragging_datasets",
        "improved_datasets",
        "sample_type_failures",
        "mixed_gain_patterns",
        "patch_risk",
        "proxy_screen",
        "ablation_evidence",
        "quality_repair",
        "eval_smoke",
        "proxy_eval_health_failure",
        "activation_smoke",
        "execution_repo_audit",
        "proxy_effect_repair_contract",
    ]
    compact = {key: attribution.get(key) for key in keep if key in attribution}
    if isinstance(compact.get("proxy_screen"), dict):
        compact["proxy_screen"] = _compact_proxy_screen(compact["proxy_screen"])
    if isinstance(compact.get("proxy_effect_repair_contract"), dict):
        compact["proxy_effect_repair_contract"] = _compact_proxy_effect_repair_contract(compact["proxy_effect_repair_contract"])
    if isinstance(compact.get("activation_smoke"), dict):
        compact["activation_smoke"] = _compact_activation_smoke(compact["activation_smoke"])
    if isinstance(compact.get("execution_repo_audit"), dict):
        compact["execution_repo_audit"] = _compact_execution_repo_audit(compact["execution_repo_audit"])
    return compact


def _compact_execution_repo_audit(audit: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(audit, dict):
        return audit
    pollution = audit.get("output_pollution") if isinstance(audit.get("output_pollution"), dict) else {}
    path_audit = audit.get("path_audit") if isinstance(audit.get("path_audit"), dict) else {}
    errors = list(audit.get("errors") or path_audit.get("errors") or [])
    compact = {
        "schema_version": audit.get("schema_version"),
        "status": audit.get("status"),
        "phase": audit.get("phase"),
        "reason": audit.get("reason"),
        "path_status": path_audit.get("status") or audit.get("status"),
        "path_error_count": len(errors),
        "path_errors": [_shorten_text(str(item), 240) for item in errors[:4]],
        "output_pollution": {
            "status": pollution.get("status"),
            "added_count": pollution.get("added_count", 0),
            "modified_count": pollution.get("modified_count", 0),
            "added_files": list(pollution.get("added_files") or [])[:12],
            "modified_files": list(pollution.get("modified_files") or [])[:12],
        },
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _shorten_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32] + "\n...[truncated]"


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _posthoc_items(value: Any, *, limit: int | None = None) -> list[str]:
    def render(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("failure_mode", "constraint", "action", "rule", "observed", "likely_root_cause", "reason"):
                if item.get(key):
                    return str(item[key])
            return json.dumps(item, ensure_ascii=False, sort_keys=True)
        return str(item)

    items: list[str] = []
    if isinstance(value, list):
        items.extend(render(item) for item in value)
    elif isinstance(value, dict):
        for group, group_value in value.items():
            if isinstance(group_value, list):
                items.extend(f"{group}: {render(item)}" for item in group_value)
            else:
                items.append(f"{group}: {render(group_value)}")
    elif value:
        items.append(render(value))
    items = [item for item in items if item]
    return items[:limit] if limit is not None else items



def _c2c_candidate_from_variant_spec(variant: dict[str, Any]) -> dict[str, Any]:
    intervention = variant.get("intervention") if isinstance(variant.get("intervention"), dict) else {}
    ablation = variant.get("ablation") if isinstance(variant.get("ablation"), dict) else {}
    return {
        "id": variant.get("variant_id"),
        "variant_id": variant.get("variant_id"),
        "direction_id": variant.get("direction_id"),
        "s1_direction_id": variant.get("direction_id"),
        "title": intervention.get("summary") or variant.get("variant_id"),
        "hypothesis": variant.get("hypothesis"),
        "null_hypothesis": variant.get("null_hypothesis"),
        "alternative_hypothesis": variant.get("alternative_hypothesis"),
        "expected_files": list(variant.get("implementation_surface_ids") or []),
        "expected_signature": variant.get("expected_metric_signature") or {},
        "variant_fingerprint": variant.get("variant_spec_hash"),
        "variation_coordinates": variant.get("variation_coordinates") or {},
        "experiment_contract": {
            "expected_files": list(variant.get("implementation_surface_ids") or []),
            "config_overrides": intervention.get("configuration") or {},
            "ablation_switch": ablation.get("switch") or "disable_selected_intervention",
        },
        "ablation_plan": ablation,
    }

def _implementation_file_hashes(project_root: Path, patch_manifest: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    candidates = patch_manifest.get("candidates") if isinstance(patch_manifest.get("candidates"), list) else []
    selected_id = str(patch_manifest.get("selected_candidate_id") or "")
    selected = next((item for item in candidates if isinstance(item, dict) and str(item.get("candidate_id") or item.get("id") or "") == selected_id), None)
    if not isinstance(selected, dict):
        selected = patch_manifest.get("selected_patch") if isinstance(patch_manifest.get("selected_patch"), dict) else {}
    for key in ["patch_json", "diff_path", "manifest"]:
        value = selected.get(key) if isinstance(selected, dict) else None
        if isinstance(value, dict):
            value = value.get("path") or value.get("rel_path")
        if not value:
            continue
        path = project_root / str(value)
        if path.exists() and path.is_file():
            hashes[str(value)] = sha256_file(path)
    return hashes


def _short_error(exc: Exception, *, limit: int = 500) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _structured_failure_class(result: dict[str, Any], main_results: dict[str, Any]) -> str | None:
    explicit = result.get("failure_class") or result.get("failure_classification")
    aliases = {
        "resource_paused": "resource_pause",
        "resource_unavailable": "resource_pause",
        "activation_failed": "activation_failure",
        "implementation_failed": "implementation_failure",
        "integrity": "integrity_failure",
        "identity_mismatch": "integrity_failure",
        "artifact_hash_mismatch": "integrity_failure",
        "safety": "safety_failure",
    }
    if explicit in {"resource_pause", "oom_retry", "activation_failure", "implementation_failure", "integrity_failure", "safety_failure"}:
        return str(explicit)
    if explicit in aliases:
        return aliases[str(explicit)]
    candidates = [item for item in main_results.get("candidate_results") or [] if isinstance(item, dict)]
    if any(item.get("resource_retry") is True or item.get("command_status") == "resource_paused" for item in candidates):
        return "resource_pause"
    if any(item.get("decision") in {"patch_rejected", "proxy_repairable"} for item in candidates):
        return "activation_failure"
    if result.get("status") == "integrity_blocked":
        return "integrity_failure"
    if result.get("status") == "resource_paused":
        return "resource_pause"
    return None


def _failure_evidence_from_result(
    *,
    project_root: Path,
    attempt: dict[str, Any],
    result: dict[str, Any],
    failure_class: str | None,
    raw_artifacts: dict[str, str],
) -> dict[str, Any] | None:
    if failure_class is None:
        return None
    supplied = result.get("failure_evidence") if isinstance(result.get("failure_evidence"), dict) else {}
    artifact_path = str(supplied.get("artifact_path") or "")
    if not artifact_path or artifact_path not in raw_artifacts:
        return None
    absolute_path = project_root / artifact_path
    if not absolute_path.is_file() or sha256_file(absolute_path) != raw_artifacts[artifact_path]:
        return None
    raw_bytes = absolute_path.read_bytes()
    producer_run_id = str(supplied.get("producer_run_id") or f"failure-{attempt['attempt_id'][:12]}-g{attempt['lifecycle_generation']}")
    evidence_kind = "resource_probe" if failure_class in {"resource_pause", "oom_retry"} else "failure_evidence"
    if evidence_kind == "resource_probe":
        try:
            probe_payload = json.loads(raw_bytes.decode("utf-8"))
            validate_contract(probe_payload, "resource_probe_v1.schema.json")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        for key in [
            "attempt_id", "direction_semantic_hash", "direction_spec_hash", "variant_semantic_hash",
            "variant_spec_hash", "trial_spec_hash", "protocol_hash", "sample_manifest_hash", "evaluator_hash",
        ]:
            if probe_payload.get(key) != attempt.get(key):
                return None
        if probe_payload.get("producer_run_id") != producer_run_id or probe_payload.get("probe_status") != "insufficient":
            return None
    log_hash = hashlib.sha256(raw_bytes).hexdigest()
    scoped_path = project_root / content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=producer_run_id,
        evidence_kind=evidence_kind,
        content_hash=log_hash,
    )
    ensure_dir(scoped_path.parent)
    if scoped_path.exists() and scoped_path.read_bytes() != raw_bytes:
        return None
    if not scoped_path.exists():
        temporary = scoped_path.with_name(f".{scoped_path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(raw_bytes)
        os.replace(temporary, scoped_path)
    source_phase = "full" if (attempt.get("phases") or {}).get("full") == "RUNNING" else "proxy"
    if failure_class == "activation_failure":
        source_phase = "activation"
    elif failure_class == "implementation_failure":
        source_phase = "implementation"
    evidence = {
        "schema_version": "auto_research_failure_evidence_v2",
        "evidence_kind": "failure_evidence",
        "evidence_id": f"failure-evidence-{attempt['attempt_id'][:12]}-g{attempt['lifecycle_generation']}",
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
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "source_state": attempt["state"],
        "source_phase": source_phase,
        "failure_class": failure_class,
        "command_status": supplied.get("command_status"),
        "exit_code": supplied.get("exit_code"),
        "reason": str(supplied.get("reason") or ""),
        "observed_at": str(supplied.get("observed_at") or now_utc()),
        "log_hash": log_hash,
    }
    if not evidence["reason"]:
        return None
    if not evidence["command_status"]:
        return None
    return evidence


def _stage_evidence_inventory(
    *,
    project_root: Path,
    attempt: dict[str, Any],
    trial_spec: dict[str, Any],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and atomically stage only artifacts explicitly emitted by this execution."""

    if not isinstance(inventory, list):
        raise S3ValidationError("staged evidence inventory must be an array")
    if not inventory:
        return {
            "schema_version": "auto_research_completion_evidence_v1",
            "attempt_id": str(attempt["attempt_id"]),
            "trial_spec_hash": str(attempt["trial_spec_hash"]),
            "entries": [],
        }
    root = project_root.resolve()
    entries: list[dict[str, Any]] = []
    evidence_bytes: dict[str, bytes] = {}
    seen_kinds: set[str] = set()
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"kind", "source_path", "producer_run_id"}:
            raise S3ValidationError("staged evidence inventory entries require only kind, source_path, and producer_run_id")
        kind = str(item["kind"])
        producer_run_id = str(item["producer_run_id"])
        source_path = str(item["source_path"])
        if kind in seen_kinds:
            raise S3ValidationError(f"duplicate staged evidence kind: {kind}")
        requirement = next(
            (item for item in trial_spec.get("evidence_requirements") or [] if item.get("kind") == kind),
            None,
        )
        if not isinstance(requirement, dict):
            raise S3ValidationError(f"evidence kind was not preregistered: {kind}")
        source = project_root / source_path
        if source.is_symlink():
            raise S3ValidationError(f"staged evidence source cannot be a symlink: {source_path}")
        resolved = source.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise S3ValidationError(f"staged evidence source escapes project root: {source_path}") from exc
        if not resolved.is_file():
            raise S3ValidationError(f"staged evidence source is missing: {source_path}")
        raw_bytes = resolved.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise S3ValidationError(f"staged evidence is not canonical JSON: {source_path}") from exc
        if not isinstance(payload, dict):
            raise S3ValidationError(f"staged evidence must be a JSON object: {source_path}")
        evidence_id = str(payload.get("evidence_id") or "")
        relative_path = content_addressed_evidence_path(
            attempt_id=str(attempt["attempt_id"]),
            producer_run_id=producer_run_id,
            evidence_kind=kind,
            content_hash=content_hash,
        )
        entry = {
            "evidence_id": evidence_id,
            "kind": kind,
            "relative_path": relative_path,
            "content_hash": content_hash,
            "schema_version": str(payload.get("schema_version") or requirement["schema_version"]),
            "attempt_id": str(attempt["attempt_id"]),
            "producer_run_id": producer_run_id,
            "direction_semantic_hash": str(attempt["direction_semantic_hash"]),
            "direction_spec_hash": str(attempt["direction_spec_hash"]),
            "variant_semantic_hash": str(attempt["variant_semantic_hash"]),
            "variant_spec_hash": str(attempt["variant_spec_hash"]),
            "trial_spec_hash": str(attempt["trial_spec_hash"]),
            "protocol_hash": str(attempt["protocol_hash"]),
            "sample_manifest_hash": str(attempt["sample_manifest_hash"]),
            "evaluator_hash": str(attempt["evaluator_hash"]),
            "cross_references": dict(payload.get("cross_references") or {}),
        }
        entries.append(entry)
        evidence_bytes[evidence_id] = raw_bytes
        seen_kinds.add(kind)
    manifest = {
        "schema_version": STAGED_EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "attempt_id": str(attempt["attempt_id"]),
        "direction_semantic_hash": str(attempt["direction_semantic_hash"]),
        "direction_spec_hash": str(attempt["direction_spec_hash"]),
        "variant_semantic_hash": str(attempt["variant_semantic_hash"]),
        "variant_spec_hash": str(attempt["variant_spec_hash"]),
        "trial_spec_hash": str(attempt["trial_spec_hash"]),
        "protocol_hash": str(attempt["protocol_hash"]),
        "sample_manifest_hash": str(attempt["sample_manifest_hash"]),
        "evaluator_hash": str(attempt["evaluator_hash"]),
        "entries": entries,
    }
    try:
        decode_evidence_inventory(
            attempt=attempt,
            trial_spec=trial_spec,
            manifest=manifest,
            evidence_bytes=evidence_bytes,
        )
    except ValueError as exc:
        raise S3ValidationError(str(exc)) from exc
    for entry in entries:
        target = project_root / entry["relative_path"]
        ensure_dir(target.parent)
        raw_bytes = evidence_bytes[entry["evidence_id"]]
        if target.exists():
            if target.is_symlink() or target.read_bytes() != raw_bytes:
                raise S3ValidationError(f"content-addressed evidence collision: {entry['relative_path']}")
            continue
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_bytes(raw_bytes)
        os.replace(temporary, target)
    return {
        "schema_version": "auto_research_completion_evidence_v1",
        "attempt_id": str(attempt["attempt_id"]),
        "trial_spec_hash": str(attempt["trial_spec_hash"]),
        "entries": [
            {key: deepcopy(value) for key, value in entry.items() if key != "cross_references"}
            for entry in entries
        ],
    }


def _identity_evidence_payload(
    *,
    attempt: dict[str, Any],
    producer_run_id: str,
    evidence_kind: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    evidence_id = f"evidence:{evidence_kind}:{producer_run_id}"
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSIONS[evidence_kind],
        "evidence_kind": evidence_kind,
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
        **fields,
    }


def _quantitative_evidence_payload(
    *,
    attempt: dict[str, Any],
    trial_spec: dict[str, Any],
    producer_run_id: str,
    evidence_kind: str,
    phase: str,
    role_values: dict[str, float],
) -> dict[str, Any]:
    payload = _identity_evidence_payload(
        attempt=attempt,
        producer_run_id=producer_run_id,
        evidence_kind=evidence_kind,
        fields={},
    )
    rows = []
    for dataset in trial_spec["datasets"]:
        for seed in trial_spec["statistical_testing"]["seeds"]:
            for role, value in role_values.items():
                rows.append(
                    {
                        "phase": phase,
                        "role": role,
                        "dataset_id": dataset["dataset_id"],
                        "metric_id": trial_spec["primary_metric_id"],
                        "seed": seed,
                        "metric_value": value,
                        "command_status": "completed",
                        "attempt_id": attempt["attempt_id"],
                        "variant_semantic_hash": attempt["variant_semantic_hash"],
                        "variant_spec_hash": attempt["variant_spec_hash"],
                        "trial_spec_hash": attempt["trial_spec_hash"],
                        "sample_manifest_hash": attempt["sample_manifest_hash"],
                        "evaluator_hash": attempt["evaluator_hash"],
                        "producer_run_id": producer_run_id,
                    }
                )
    payload["rows"] = rows
    return payload


def _c2c_strict_evidence_inventory(
    *,
    project_root: Path,
    attempt: dict[str, Any],
    trial_spec: dict[str, Any],
    comparison_candidate: dict[str, Any] | None,
    baseline: dict[str, Any],
    simulate: bool,
) -> list[dict[str, str]]:
    if not simulate or not isinstance(comparison_candidate, dict):
        return []
    producer_run_id = f"c2c-simulate-{attempt['attempt_id'][:12]}-g{attempt['lifecycle_generation']}"
    seeds = trial_spec["statistical_testing"]["seeds"]
    metric_id = trial_spec["primary_metric_id"]
    candidate_datasets = ((comparison_candidate.get("metrics") or {}).get("datasets") or {})
    baseline_datasets = baseline.get("datasets") if isinstance(baseline.get("datasets"), dict) else {}
    ablation_datasets = ((((comparison_candidate.get("ablation") or {}).get("metrics") or {}).get("datasets")) or {})
    matched_datasets = ((comparison_candidate.get("matched_control_metrics") or {}).get("datasets") or {})
    coverage_datasets = ((comparison_candidate.get("coverage_metrics") or {}).get("datasets") or {})
    dataset_ids = [dataset["dataset_id"] for dataset in trial_spec["datasets"]]
    sources: list[dict[str, str]] = []

    def quantitative(kind: str, phase: str, values: dict[str, dict[str, float]]) -> dict[str, Any]:
        payload = _identity_evidence_payload(attempt=attempt, producer_run_id=producer_run_id, evidence_kind=kind, fields={})
        rows = []
        for dataset_id in dataset_ids:
            for seed in seeds:
                for role, role_values in values.items():
                    value = role_values.get(dataset_id)
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        raise S3ValidationError(f"C2C simulated evidence lacks {kind} row for {role}/{dataset_id}/{seed}")
                    rows.append(
                        {
                            "phase": phase,
                            "role": role,
                            "dataset_id": dataset_id,
                            "metric_id": metric_id,
                            "seed": seed,
                            "metric_value": float(value),
                            "command_status": "completed",
                            "attempt_id": attempt["attempt_id"],
                            "variant_semantic_hash": attempt["variant_semantic_hash"],
                            "variant_spec_hash": attempt["variant_spec_hash"],
                            "trial_spec_hash": attempt["trial_spec_hash"],
                            "sample_manifest_hash": attempt["sample_manifest_hash"],
                            "evaluator_hash": attempt["evaluator_hash"],
                            "producer_run_id": producer_run_id,
                        }
                    )
        payload["rows"] = rows
        return payload

    payloads: dict[str, dict[str, Any]] = {
        "main_results": quantitative("main_results", "full", {"baseline": baseline_datasets, "candidate": candidate_datasets}),
        "ablation_results": quantitative("ablation_results", "ablation", {"ablation": ablation_datasets}),
        "matched_control_results": quantitative("matched_control_results", "full", {"matched_control": matched_datasets}),
        "coverage_results": quantitative("coverage_results", "full", {"coverage": coverage_datasets}),
    }
    hashes = {kind: hashlib.sha256(encode_canonical_evidence(payload)).hexdigest() for kind, payload in payloads.items()}
    fingerprint_inputs = {"sample_manifest_hash": attempt["sample_manifest_hash"], "evaluator_hash": attempt["evaluator_hash"], "protocol_hash": attempt["protocol_hash"]}
    baseline_hash = canonical_hash(fingerprint_inputs)
    payloads["activation_evidence"] = _identity_evidence_payload(
        attempt=attempt,
        producer_run_id=producer_run_id,
        evidence_kind="activation_evidence",
        fields={"probe_id": "c2c-simulated-forward-probe", "status": "passed", "command_status": "completed", "exit_code": 0, "implementation_surface_ids": list((read_json(project_root / "plan/variant.json", default={}) or {}).get("implementation_surface_ids") or ["c2c-simulated-surface"])},
    )
    hashes["activation_evidence"] = hashlib.sha256(encode_canonical_evidence(payloads["activation_evidence"])).hexdigest()
    payloads["proxy_baseline_fingerprint"] = _identity_evidence_payload(
        attempt=attempt, producer_run_id=producer_run_id, evidence_kind="proxy_baseline_fingerprint",
        fields={"baseline_hash": baseline_hash, "dataset_ids": dataset_ids, "seeds": seeds, "fingerprint_inputs": fingerprint_inputs},
    )
    hashes["proxy_baseline_fingerprint"] = hashlib.sha256(encode_canonical_evidence(payloads["proxy_baseline_fingerprint"])).hexdigest()
    payloads["proxy_cache_report"] = _identity_evidence_payload(
        attempt=attempt, producer_run_id=producer_run_id, evidence_kind="proxy_cache_report",
        fields={"cross_references": {"proxy_baseline_fingerprint_hash": hashes["proxy_baseline_fingerprint"]}, "cache_key": canonical_hash({"attempt": attempt["attempt_id"], "producer": producer_run_id}), "baseline_hash": baseline_hash, "cache_entry_hash": baseline_hash, "status": "created"},
    )
    payloads["effective_proxy_policy"] = _identity_evidence_payload(
        attempt=attempt, producer_run_id=producer_run_id, evidence_kind="effective_proxy_policy",
        fields={"policy_hash": canonical_hash({"required_phases": trial_spec["protocol"]["required_phases"], "proxy_terminal_allowed": trial_spec["protocol"]["proxy_terminal_allowed"], "decision_threshold": 0.0}), "required_phases": trial_spec["protocol"]["required_phases"], "proxy_terminal_allowed": trial_spec["protocol"]["proxy_terminal_allowed"], "decision_threshold": 0.0},
    )
    hashes.update({kind: hashlib.sha256(encode_canonical_evidence(payloads[kind])).hexdigest() for kind in ("proxy_cache_report", "effective_proxy_policy")})
    payloads["proxy_calibration_policy"] = _identity_evidence_payload(
        attempt=attempt, producer_run_id=producer_run_id, evidence_kind="proxy_calibration_policy",
        fields={"cross_references": {"proxy_baseline_fingerprint_hash": hashes["proxy_baseline_fingerprint"], "effective_proxy_policy_hash": hashes["effective_proxy_policy"]}, "calibration_hash": canonical_hash({"status": "calibrated", "calibration_metric": metric_id, "calibration_value": 1.0, "cross_references": {"proxy_baseline_fingerprint_hash": hashes["proxy_baseline_fingerprint"], "effective_proxy_policy_hash": hashes["effective_proxy_policy"]}}), "status": "calibrated", "calibration_metric": metric_id, "calibration_value": 1.0},
    )
    hashes["proxy_calibration_policy"] = hashlib.sha256(encode_canonical_evidence(payloads["proxy_calibration_policy"])).hexdigest()
    mean_delta = float((comparison_candidate.get("metrics") or {}).get("mean", 0.0)) - float(baseline.get("mean", 0.0))
    payloads["proxy_decision_report"] = _identity_evidence_payload(
        attempt=attempt, producer_run_id=producer_run_id, evidence_kind="proxy_decision_report",
        fields={"cross_references": {"proxy_baseline_fingerprint_hash": hashes["proxy_baseline_fingerprint"], "proxy_cache_report_hash": hashes["proxy_cache_report"], "effective_proxy_policy_hash": hashes["effective_proxy_policy"], "proxy_calibration_policy_hash": hashes["proxy_calibration_policy"], "main_results_hash": hashes["main_results"]}, "decision": "run_full", "reason_codes": ["simulated_proxy_verified"], "observed_proxy_delta": mean_delta},
    )
    hashes["proxy_decision_report"] = hashlib.sha256(encode_canonical_evidence(payloads["proxy_decision_report"])).hexdigest()
    payloads["full_s3_readiness"] = _identity_evidence_payload(
        attempt=attempt, producer_run_id=producer_run_id, evidence_kind="full_s3_readiness",
        fields={"cross_references": {"activation_evidence_hash": hashes["activation_evidence"], "proxy_decision_report_hash": hashes["proxy_decision_report"]}, "ready": True, "checks": [{"check_id": "simulated-c2c-proxy-full-ready", "status": "PASS"}]},
    )
    for kind, payload in payloads.items():
        sources.append(_write_staged_evidence_source(project_root, producer_run_id=producer_run_id, evidence_kind=kind, payload=payload))
    return sources


def _write_staged_evidence_source(
    project_root: Path,
    *,
    producer_run_id: str,
    evidence_kind: str,
    payload: dict[str, Any],
) -> dict[str, str]:
    relative_path = f"experiment/staging/{producer_run_id}/{evidence_kind}.json"
    path = project_root / relative_path
    ensure_dir(path.parent)
    path.write_bytes(encode_canonical_evidence(payload))
    return {"kind": evidence_kind, "source_path": relative_path, "producer_run_id": producer_run_id}


def _decode_staged_execution_observations(
    project_root: Path,
    attempt: dict[str, Any],
    trial_spec: dict[str, Any],
    completion_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Decode canonical observations only from immutable staged measurement bytes."""

    evidence_bytes: dict[str, bytes] = {}
    manifest_entries: list[dict[str, Any]] = []
    for entry in completion_evidence.get("entries") or []:
        path = project_root / str(entry["relative_path"])
        raw_bytes = path.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != entry["content_hash"]:
            raise S3ValidationError(f"staged evidence hash drift: {entry['relative_path']}")
        payload = json.loads(raw_bytes.decode("utf-8"))
        manifest_entries.append({**deepcopy(entry), "cross_references": deepcopy(payload.get("cross_references") or {})})
        evidence_bytes[entry["evidence_id"]] = raw_bytes
    manifest = {
        "schema_version": STAGED_EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "attempt_id": str(attempt["attempt_id"]),
        "direction_semantic_hash": str(attempt["direction_semantic_hash"]),
        "direction_spec_hash": str(attempt["direction_spec_hash"]),
        "variant_semantic_hash": str(attempt["variant_semantic_hash"]),
        "variant_spec_hash": str(attempt["variant_spec_hash"]),
        "trial_spec_hash": str(attempt["trial_spec_hash"]),
        "protocol_hash": str(attempt["protocol_hash"]),
        "sample_manifest_hash": str(attempt["sample_manifest_hash"]),
        "evaluator_hash": str(attempt["evaluator_hash"]),
        "entries": manifest_entries,
    }
    try:
        observations, _ = decode_evidence_inventory(
            attempt=attempt,
            trial_spec=trial_spec,
            manifest=manifest,
            evidence_bytes=evidence_bytes,
        )
    except ValueError as exc:
        raise S3ValidationError(str(exc)) from exc
    if not observations:
        raise S3ValidationError("quantitative measurement rows are missing")
    return observations


def _trial_execution_view(trial_spec: dict[str, Any]) -> dict[str, Any]:
    view = deepcopy(trial_spec)
    runtime = deepcopy((trial_spec.get("execution_contract") or {}).get("runtime_config") or {})
    view["execution"] = runtime
    view["datasets"] = [
        {"name": item["dataset_id"], "split": item["split"], "sample_count": item["sample_count"], "sample_hash": item["sample_hash"]}
        for item in trial_spec.get("datasets") or []
    ]
    view["metrics"] = [
        {
            "name": item["metric_id"],
            "primary": item["role"] == "primary",
            "higher_is_better": item["objective"] == "maximize",
        }
        for item in trial_spec.get("metrics") or []
    ]
    criteria: dict[str, Any] = {}
    for constraint in trial_spec.get("acceptance_constraints") or []:
        if constraint["kind"] == "minimum_mean_delta":
            criteria["minimum_mean_delta"] = constraint["threshold"]
        elif constraint["kind"] == "per_dataset_maximum_regression":
            criteria["max_dataset_regression"] = constraint["threshold"]
        elif constraint["kind"] == "matched_control_constraint":
            criteria["matched_coverage_ablation_required"] = True
    view["acceptance_criteria"] = criteria
    return view


def _event_bound_evidence_manifest(
    *,
    project_root: Path,
    trial_spec: dict[str, Any],
    attempt: dict[str, Any],
    raw_artifacts: dict[str, str],
) -> dict[str, Any]:
    del project_root, trial_spec, attempt, raw_artifacts
    raise S3ValidationError("fixed-path evidence collection was removed; execution must submit an explicit staged inventory")
