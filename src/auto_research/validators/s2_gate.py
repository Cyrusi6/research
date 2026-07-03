"""S2 plan and patch gate."""

from __future__ import annotations

from ..code_patch import code_patch_gate_mode, is_retryable_patch_manifest
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

        planner = self._read_required_json_with_schema(
            "plan/planner_decision.json",
            "planner_decision_json_exists",
            "planner_decision_schema",
            "planner_decision.schema.json",
        )
        variant_contract = self._read_required_json_with_schema(
            "plan/variant_contract.json",
            "variant_contract_json_exists",
            "variant_contract_schema",
            "variant_contract.schema.json",
        )
        variant_fingerprint = self._read_required_json_with_schema(
            "plan/variant_fingerprint.json",
            "variant_fingerprint_json_exists",
            "variant_fingerprint_schema",
            "variant_fingerprint.schema.json",
        )
        if isinstance(planner, dict) and isinstance(variant_contract, dict) and isinstance(variant_fingerprint, dict):
            self._validate_variant_handoff(plan, planner, variant_contract, variant_fingerprint)

        if plan.get("execution", {}).get("collector") == "c2c_small_loop":
            self._validate_c2c_plan(plan)
        return self.finalize()

    def _read_required_json_with_schema(self, rel_path: str, exists_check: str, schema_check: str, schema_name: str):
        path = self.require_file(rel_path, check_name=exists_check)
        if not path:
            return None
        payload = self.read_json_artifact(rel_path)
        if payload is None:
            return None
        schema_errors = validate_min_schema(payload, load_schema(schema_name))
        if schema_errors:
            self.retry_check(schema_check, f"{rel_path} does not satisfy contract", artifact=rel_path, details={"errors": schema_errors[:10]})
        else:
            self.pass_check(schema_check, artifact=rel_path)
        return payload

    def _validate_variant_handoff(
        self,
        plan: dict,
        planner: dict,
        variant_contract: dict,
        variant_fingerprint: dict,
    ) -> None:
        direction = self._safe_json("literature/direction.json")
        errors: list[str] = []
        direction_id = direction.get("direction_id") if isinstance(direction, dict) else None
        ids = {
            "literature.direction": direction_id,
            "planner_decision": planner.get("direction_id"),
            "variant_contract": variant_contract.get("direction_id"),
            "variant_fingerprint": variant_fingerprint.get("direction_id"),
        }
        present_ids = {str(value) for value in ids.values() if value}
        if len(present_ids) > 1:
            errors.append(f"direction_id mismatch: {ids}")
        for key in ["mechanism_axis", "integration_point", "control_signal"]:
            contract_value = str(variant_contract.get(key) or "").strip()
            fingerprint_value = str(variant_fingerprint.get(key) or "").strip()
            if not contract_value:
                errors.append(f"variant_contract.{key} must be non-empty")
            if not fingerprint_value:
                errors.append(f"variant_fingerprint.{key} must be non-empty")
            if contract_value and fingerprint_value and contract_value != fingerprint_value:
                errors.append(f"{key} mismatch between variant_contract and variant_fingerprint")
        if not isinstance(variant_contract.get("resource_budget"), dict):
            errors.append("variant_contract.resource_budget must be an object")
        if not isinstance(variant_contract.get("expected_metric_signature"), dict) or not variant_contract.get("expected_metric_signature"):
            errors.append("variant_contract.expected_metric_signature must be a non-empty object")
        if not _non_empty_list(variant_contract.get("expected_files")):
            errors.append("variant_contract.expected_files must be non-empty")
        if not _non_empty_list(variant_contract.get("implementation_surface_refs")):
            errors.append("variant_contract.implementation_surface_refs must be non-empty")
        ablation = variant_contract.get("ablation") if isinstance(variant_contract.get("ablation"), dict) else {}
        if not ablation.get("switch"):
            errors.append("variant_contract.ablation.switch must be present")
        if not ablation.get("control"):
            errors.append("variant_contract.ablation.control must be present")
        routing = variant_contract.get("failure_routing") if isinstance(variant_contract.get("failure_routing"), dict) else {}
        for key in ["go_to_s3_conditions", "return_to_s2_conditions", "return_to_s1_conditions"]:
            if not _non_empty_list(routing.get(key)):
                errors.append(f"variant_contract.failure_routing.{key} must be non-empty")
        fingerprint = str(variant_fingerprint.get("variant_fingerprint") or "").strip()
        contract_fingerprint = str(variant_contract.get("variant_fingerprint") or "").strip()
        if fingerprint and contract_fingerprint and fingerprint != contract_fingerprint:
            errors.append("variant_fingerprint does not match variant_contract.variant_fingerprint")
        next_variant = planner.get("next_variant") if isinstance(planner.get("next_variant"), dict) else {}
        next_fingerprint = str(next_variant.get("variant_fingerprint") or "").strip()
        if next_fingerprint and fingerprint and next_fingerprint != fingerprint:
            errors.append("planner_decision.next_variant fingerprint does not match variant_fingerprint")
        mode = str(variant_fingerprint.get("mode") or variant_contract.get("mode") or "")
        if variant_fingerprint.get("is_repeat") is True and not mode.startswith("implementation_repair"):
            errors.append("variant_fingerprint repeats a previous same-direction variant")
        if plan.get("execution", {}).get("collector") == "c2c_small_loop" and not mode.startswith("implementation_repair"):
            disallowed = self._disallowed_c2c_expected_files(variant_contract)
            if disallowed:
                errors.append(f"variant_contract expected_files outside allowed C2C edit surface: {disallowed[:5]}")
        if errors:
            self.retry_check(
                "s2_variant_handoff_contract",
                "S2 variant contract is not ready for S2.5/S3",
                artifact="plan/variant_contract.json",
                details={"errors": errors[:12]},
            )
        else:
            self.pass_check("s2_variant_handoff_contract", artifact="plan/variant_contract.json")

    def _safe_json(self, rel_path: str):
        path = self.project_root / rel_path
        if not path.exists():
            return None
        return self.read_json_artifact(rel_path)

    def _disallowed_c2c_expected_files(self, variant_contract: dict) -> list[str]:
        files = [str(item) for item in variant_contract.get("expected_files") or [] if item]
        c2c_cfg = self.config.get("c2c", {}) if isinstance(self.config.get("c2c"), dict) else {}
        allowed_files = {str(item).strip("/") for item in c2c_cfg.get("allowed_files") or [] if item}
        allowed_prefixes = [str(item).strip("/") for item in c2c_cfg.get("allowed_prefixes") or [] if item]
        if not allowed_files and not allowed_prefixes:
            return []
        disallowed = []
        for file_path in files:
            normalized = file_path.strip("/")
            if normalized in allowed_files or any(normalized.startswith(prefix.rstrip("/") + "/") or normalized == prefix for prefix in allowed_prefixes):
                continue
            disallowed.append(file_path)
        return disallowed

    def _validate_c2c_plan(self, plan: dict) -> None:
        discovery_mode = code_patch_gate_mode(self.config) == "discovery"
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
            self._quality_or_retry(
                "c2c_coverage_control_requirements",
                "C2C plan should require coverage diagnostics and matched-coverage ablation",
                discovery_mode=discovery_mode,
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
            self._quality_or_retry(
                "c2c_matched_coverage_ablation",
                "C2C ablation matrix should include a matched-transfer-coverage control",
                discovery_mode=discovery_mode,
                artifact="plan/plan.yaml",
            )

        if plan.get("reviewer_risk_controls"):
            self.pass_check("c2c_reviewer_risk_controls", artifact="plan/plan.yaml")
        else:
            self._quality_or_retry(
                "c2c_reviewer_risk_controls",
                "C2C reviewer risk controls missing",
                discovery_mode=discovery_mode,
                artifact="plan/plan.yaml",
            )

        if candidate_path:
            self._validate_c2c_idea_novelty(discovery_mode=discovery_mode)
            self._validate_c2c_implementation_scope(discovery_mode=discovery_mode)

        self.pass_check(
            "c2c_runtime_resource_selection_deferred",
            artifact="plan/plan.yaml",
            details={
                "stage": "S2.5/S3",
                "reason": "Volatile runtime resources are selected immediately before smoke/proxy/full execution, not in S2.",
            },
        )

        patch_manifest_path = self.project_root / "plan" / "code_patches" / "patch_manifest.json"
        if patch_manifest_path.exists():
            patch_manifest = self.read_json_artifact("plan/code_patches/patch_manifest.json") or {}
            status = patch_manifest.get("status")
            if status in {"ok", "partial", "disabled"}:
                self.pass_check("s2_5_patch_manifest_status", artifact="plan/code_patches/patch_manifest.json", details={"status": status})
            elif is_retryable_patch_manifest(patch_manifest):
                retry_reason = "S2.5 patch generation hit a retryable Codex/backend limit; rerun S2.5 when quota is available"
                if _patch_manifest_has_resource_retry(patch_manifest):
                    retry_reason = "S2.5 runtime smoke is waiting for enough free GPU memory; resume when a GPU satisfies runtime_smoke.min_free_mb"
                self.retry_check(
                    "s2_5_patch_manifest_status",
                    retry_reason,
                    artifact="plan/code_patches/patch_manifest.json",
                    details={
                        "status": status,
                        "retryable_patch_count": patch_manifest.get("retryable_patch_count"),
                        "resource_retry": _patch_manifest_has_resource_retry(patch_manifest),
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

    def _quality_or_retry(
        self,
        name: str,
        message: str,
        *,
        discovery_mode: bool,
        artifact: str | None = None,
        details: dict | None = None,
    ) -> None:
        if discovery_mode:
            payload = {"quality_debt": True, "gate_mode": "discovery"}
            if details:
                payload.update(details)
            self.pass_check(name, artifact=artifact, message=message, details=payload)
        else:
            self.retry_check(name, message, artifact=artifact, details=details)

    def _validate_c2c_idea_novelty(self, *, discovery_mode: bool) -> None:
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
            self._quality_or_retry(
                "c2c_mechanism_novelty_gate",
                "C2C selected ideas look like local tuning or lack mechanism-level fields",
                discovery_mode=discovery_mode,
                artifact="plan/candidate_ideas.json",
                details={"rejected": rejected[:5], "reports": reports[:5]},
            )
        else:
            self.pass_check(
                "c2c_mechanism_novelty_gate",
                artifact="plan/candidate_ideas.json",
                details={"checked": reports[:5]},
            )

    def _validate_c2c_implementation_scope(self, *, discovery_mode: bool) -> None:
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
            self._quality_or_retry(
                "c2c_implementation_scope_gate",
                "C2C selected ideas need implementation decomposition before S2.5 patching",
                discovery_mode=discovery_mode,
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


def _patch_manifest_has_resource_retry(patch_manifest: dict) -> bool:
    if not isinstance(patch_manifest, dict):
        return False
    if patch_manifest.get("resource_retry") is True or patch_manifest.get("failure_category") == "runtime_smoke_resource_retry":
        return True
    entries = patch_manifest.get("patches") or patch_manifest.get("candidates") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("resource_retry") is True or entry.get("failure_category") == "runtime_smoke_resource_retry":
            return True
        validation = entry.get("validation") if isinstance(entry.get("validation"), dict) else {}
        if validation.get("resource_retry") is True or validation.get("failure_category") == "runtime_smoke_resource_retry":
            return True
        for check in validation.get("checks") or []:
            if isinstance(check, dict) and (check.get("resource_retry") is True or check.get("failure_category") == "runtime_smoke_resource_retry"):
                return True
    return False


def _non_empty_list(value) -> bool:
    return isinstance(value, list) and any(item not in (None, "", [], {}) for item in value)
