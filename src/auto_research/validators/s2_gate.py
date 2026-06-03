"""S2 plan and patch gate."""

from __future__ import annotations

from ..code_patch import is_retryable_patch_manifest
from ..c2c import c2c_idea_novelty_report, c2c_implementation_scope_report
from .base import PASS, StageGateValidator, load_schema, validate_min_schema


class S2GateValidator(StageGateValidator):
    stage_key = "S2_plan"
    validator_name = "s2_plan_gate_v1"

    def validate(self):
        plan_path = self.require_file("plan/plan.yaml", check_name="plan_yaml_exists")
        if not plan_path:
            return self.finalize()
        plan = self.read_yaml_artifact("plan/plan.yaml")
        if not isinstance(plan, dict):
            return self.finalize()

        schema_errors = validate_min_schema(plan, load_schema("plan.schema.json"))
        if schema_errors:
            self.retry_check("plan_schema", "plan.yaml does not satisfy plan contract", artifact="plan/plan.yaml", details={"errors": schema_errors[:10]})
        else:
            self.pass_check("plan_schema", artifact="plan/plan.yaml")

        if plan.get("execution", {}).get("collector") == "c2c_small_loop":
            self._validate_c2c_plan(plan)
        return self.finalize()

    def _validate_c2c_plan(self, plan: dict) -> None:
        self.require_file("plan/short_loop_plan.yaml", check_name="c2c_short_loop_plan_exists")
        candidate_path = self.require_file("plan/candidate_ideas.json", check_name="c2c_candidate_ideas_exists")

        execution = plan.get("execution") or {}
        missing_thresholds = [field for field in ["min_delta_to_pass", "max_dataset_regression"] if field not in execution]
        if missing_thresholds:
            self.retry_check("c2c_acceptance_thresholds", "C2C execution missing acceptance thresholds", artifact="plan/plan.yaml", details={"missing": missing_thresholds})
        else:
            self.pass_check("c2c_acceptance_thresholds", artifact="plan/plan.yaml")

        if plan.get("acceptance_criteria"):
            self.pass_check("c2c_acceptance_criteria", artifact="plan/plan.yaml")
        else:
            self.retry_check("c2c_acceptance_criteria", "C2C acceptance criteria missing", artifact="plan/plan.yaml")
        acceptance = plan.get("acceptance_criteria") if isinstance(plan.get("acceptance_criteria"), dict) else {}
        missing_controls = [
            key
            for key in ["coverage_diagnostics_required", "matched_coverage_ablation_required"]
            if acceptance.get(key) is not True
        ]
        if missing_controls:
            self.retry_check(
                "c2c_coverage_control_requirements",
                "C2C plan must require coverage diagnostics and matched-coverage ablation",
                artifact="plan/plan.yaml",
                details={"missing_or_false": missing_controls},
            )
        else:
            self.pass_check("c2c_coverage_control_requirements", artifact="plan/plan.yaml")

        ablation_matrix = plan.get("ablation_matrix") if isinstance(plan.get("ablation_matrix"), list) else []
        has_matched_coverage = any(
            isinstance(item, dict) and (item.get("matched_coverage_ablation") or "matched" in str(item.get("experiment", "")).lower())
            for item in ablation_matrix
        )
        if has_matched_coverage:
            self.pass_check("c2c_matched_coverage_ablation", artifact="plan/plan.yaml")
        else:
            self.retry_check(
                "c2c_matched_coverage_ablation",
                "C2C ablation matrix must include a matched-transfer-coverage control",
                artifact="plan/plan.yaml",
            )

        if plan.get("reviewer_risk_controls"):
            self.pass_check("c2c_reviewer_risk_controls", artifact="plan/plan.yaml")
        else:
            self.retry_check("c2c_reviewer_risk_controls", "C2C reviewer risk controls missing", artifact="plan/plan.yaml")

        if candidate_path:
            self._validate_c2c_idea_novelty()
            self._validate_c2c_implementation_scope()

        selected = execution.get("selected_gpu_ids")
        if selected is None:
            self.retry_check("c2c_selected_gpu_ids", "C2C execution missing selected_gpu_ids", artifact="plan/plan.yaml")
        else:
            self.pass_check("c2c_selected_gpu_ids", artifact="plan/plan.yaml", details={"selected_gpu_ids": selected})
            peak = plan.get("resource_budget", {}).get("peak_concurrent_gpus")
            if peak != len(selected):
                self.fail_check(
                    "c2c_gpu_budget_consistency",
                    "C2C resource budget does not match selected GPU count",
                    artifact="plan/plan.yaml",
                    details={"peak_concurrent_gpus": peak, "selected_gpu_count": len(selected)},
                )
            else:
                self.pass_check("c2c_gpu_budget_consistency", artifact="plan/plan.yaml")

        patch_manifest_path = self.project_root / "plan" / "code_patches" / "patch_manifest.json"
        if patch_manifest_path.exists():
            patch_manifest = self.read_json_artifact("plan/code_patches/patch_manifest.json") or {}
            status = patch_manifest.get("status")
            if status in {"ok", "partial", "disabled"}:
                self.pass_check("s2_5_patch_manifest_status", artifact="plan/code_patches/patch_manifest.json", details={"status": status})
            elif is_retryable_patch_manifest(patch_manifest):
                self.retry_check(
                    "s2_5_patch_manifest_status",
                    "S2.5 patch generation hit a retryable Codex/backend limit; rerun S2.5 when quota is available",
                    artifact="plan/code_patches/patch_manifest.json",
                    details={
                        "status": status,
                        "retryable_patch_count": patch_manifest.get("retryable_patch_count"),
                    },
                )
            else:
                self.fail_check("s2_5_patch_manifest_status", f"S2.5 patch manifest status is {status}", artifact="plan/code_patches/patch_manifest.json")
            entries = patch_manifest.get("patches") or patch_manifest.get("candidates") or []
            executable = any(_patch_has_executable_change(entry) for entry in entries if isinstance(entry, dict))
            if status != "disabled" and entries and not executable and not is_retryable_patch_manifest(patch_manifest):
                self.fail_check("s2_5_executable_patch", "S2.5 produced no executable code change", artifact="plan/code_patches/patch_manifest.json")
            elif status != "disabled" and entries and not executable:
                self.retry_check(
                    "s2_5_executable_patch",
                    "S2.5 produced no executable code change because patch generation is retryable",
                    artifact="plan/code_patches/patch_manifest.json",
                )
            elif entries:
                self.pass_check("s2_5_executable_patch", artifact="plan/code_patches/patch_manifest.json")

    def _validate_c2c_idea_novelty(self) -> None:
        ideas = self.read_json_artifact("plan/candidate_ideas.json")
        if not isinstance(ideas, list) or not ideas:
            self.retry_check(
                "c2c_mechanism_novelty_gate",
                "C2C candidate_ideas.json must contain mechanism-level ideas",
                artifact="plan/candidate_ideas.json",
            )
            return
        selected = [idea for idea in ideas if isinstance(idea, dict) and idea.get("selected")]
        candidates = selected or [idea for idea in ideas if isinstance(idea, dict)]
        reports = []
        rejected = []
        for idea in candidates:
            report = c2c_idea_novelty_report(idea)
            row = {"id": idea.get("id"), "title": idea.get("title"), **report}
            reports.append(row)
            if report.get("status") != PASS.lower():
                rejected.append(row)
        if rejected or not reports:
            self.retry_check(
                "c2c_mechanism_novelty_gate",
                "C2C selected ideas look like local tuning or lack mechanism-level fields",
                artifact="plan/candidate_ideas.json",
                details={"rejected": rejected[:5], "reports": reports[:5]},
            )
        else:
            self.pass_check(
                "c2c_mechanism_novelty_gate",
                artifact="plan/candidate_ideas.json",
                details={"checked": reports[:5]},
            )

    def _validate_c2c_implementation_scope(self) -> None:
        ideas = self.read_json_artifact("plan/candidate_ideas.json")
        if not isinstance(ideas, list) or not ideas:
            return
        selected = [idea for idea in ideas if isinstance(idea, dict) and idea.get("selected")]
        candidates = selected or [idea for idea in ideas if isinstance(idea, dict)]
        reports = []
        blocked = []
        for idea in candidates:
            report = c2c_implementation_scope_report(idea)
            row = {"id": idea.get("id"), "title": idea.get("title"), **report}
            reports.append(row)
            if report.get("status") != PASS.lower():
                blocked.append(row)
        if blocked:
            self.retry_check(
                "c2c_implementation_scope_gate",
                "C2C selected ideas need implementation decomposition before S2.5 patching",
                artifact="plan/candidate_ideas.json",
                details={"blocked": blocked[:5], "reports": reports[:5]},
            )
        else:
            self.pass_check(
                "c2c_implementation_scope_gate",
                artifact="plan/candidate_ideas.json",
                details={"checked": reports[:5]},
            )


def _patch_has_executable_change(entry: dict) -> bool:
    if entry.get("has_executable_change") is True:
        return True
    code_patch = entry.get("code_patch") if isinstance(entry.get("code_patch"), dict) else {}
    if code_patch.get("has_executable_change") is True:
        return True
    validation = entry.get("validation") if isinstance(entry.get("validation"), dict) else {}
    if validation.get("status") == PASS and validation.get("has_executable_change") is True:
        return True
    changed = entry.get("changed_files") or code_patch.get("changed_files") or []
    return any(str(path).endswith(".py") for path in changed)
