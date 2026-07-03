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
            self._validate_s2_planner_gate_artifacts(variant_fingerprint if isinstance(variant_fingerprint, dict) else {})
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
            self._validate_s2_5_patch_gate()
            patch_manifest = self.read_json_artifact("plan/code_patches/patch_manifest.json") or {}
            patch_gate = self.read_json_artifact("plan/code_patches/patch_gate_report.json") or {}
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
            elif isinstance(patch_gate, dict) and patch_gate.get("gate") == "fail" and patch_gate.get("repairable") is True:
                self.retry_check(
                    "s2_5_patch_manifest_status",
                    "S2.5 patch failed its implementation contract and should route to patch-only repair",
                    artifact="plan/code_patches/patch_manifest.json",
                    details={
                        "status": status,
                        "patch_gate": patch_gate.get("gate"),
                        "failure_class": patch_gate.get("failure_class"),
                    },
                )
            else:
                self.fail_check("s2_5_patch_manifest_status", f"S2.5 patch manifest status is {status}", artifact="plan/code_patches/patch_manifest.json")
            entries = patch_manifest.get("patches") or patch_manifest.get("candidates") or []
            executable = any(_patch_has_executable_change(entry) for entry in entries if isinstance(entry, dict))
            patch_gate_repairable = isinstance(patch_gate, dict) and patch_gate.get("gate") == "fail" and patch_gate.get("repairable") is True
            if status != "disabled" and entries and not executable and not is_retryable_patch_manifest(patch_manifest) and not patch_gate_repairable:
                self.fail_check("s2_5_executable_patch", "S2.5 produced no executable code change", artifact="plan/code_patches/patch_manifest.json")
            elif status != "disabled" and entries and not executable:
                self.retry_check(
                    "s2_5_executable_patch",
                    "S2.5 produced no executable code change and should route to retry/patch-only repair",
                    artifact="plan/code_patches/patch_manifest.json",
                )
            elif entries:
                self.pass_check("s2_5_executable_patch", artifact="plan/code_patches/patch_manifest.json")

    def _validate_s2_planner_gate_artifacts(self, variant_fingerprint: dict) -> None:
        candidate_pool = self._read_required_json_with_schema(
            "plan/s2_planner/candidate_pool.json",
            "s2_candidate_pool_json_exists",
            "s2_candidate_pool_schema",
            "s2_candidate_pool.schema.json",
        )
        scorecard = self._read_required_json_with_schema(
            "plan/s2_planner/variant_scorecard.json",
            "s2_variant_scorecard_json_exists",
            "s2_variant_scorecard_schema",
            "s2_variant_scorecard.schema.json",
        )
        feedback_context = self._read_required_json_with_schema(
            "plan/s2_planner/feedback_context.json",
            "s2_feedback_context_json_exists",
            "s2_feedback_context_schema",
            "s2_feedback_context.schema.json",
        )
        adaptive_policy = self._read_required_json_with_schema(
            "plan/s2_planner/adaptive_policy.json",
            "s2_adaptive_policy_json_exists",
            "s2_adaptive_policy_schema",
            "s2_adaptive_policy.schema.json",
        )
        score_adjustment_report = self._read_required_json_with_schema(
            "plan/s2_planner/score_adjustment_report.json",
            "s2_score_adjustment_report_json_exists",
            "s2_score_adjustment_report_schema",
            "s2_score_adjustment_report.schema.json",
        )
        next_variant_path = self.require_file("plan/s2_planner/next_variant.json", check_name="s2_next_variant_json_exists")
        next_variant = self.read_json_artifact("plan/s2_planner/next_variant.json") if next_variant_path else None
        if isinstance(next_variant, dict):
            self.pass_check("s2_next_variant_schema", artifact="plan/s2_planner/next_variant.json")
        elif next_variant_path:
            self.retry_check("s2_next_variant_schema", "plan/s2_planner/next_variant.json must be an object", artifact="plan/s2_planner/next_variant.json")
        planner_gate = self._read_required_json_with_schema(
            "plan/s2_planner/planner_gate_report.json",
            "s2_planner_gate_report_json_exists",
            "s2_planner_gate_report_schema",
            "s2_planner_gate_report.schema.json",
        )
        if not all(isinstance(item, dict) for item in [candidate_pool, scorecard, feedback_context, adaptive_policy, score_adjustment_report, next_variant, planner_gate]):
            return
        errors: list[str] = []
        direction = self._safe_json("literature/direction.json")
        direction_id = str(direction.get("direction_id") or "") if isinstance(direction, dict) else ""
        candidates = [item for item in candidate_pool.get("candidates") or [] if isinstance(item, dict)]
        candidate_ids = {str(item.get("id") or "") for item in candidates}
        candidate_fps = {str(item.get("variant_fingerprint") or "") for item in candidates}
        selected_id = str(planner_gate.get("selected_variant_id") or next_variant.get("id") or "")
        selected_fp = str(planner_gate.get("selected_variant_fingerprint") or next_variant.get("variant_fingerprint") or "")
        expected_fp = str(variant_fingerprint.get("variant_fingerprint") or "")
        route_constraints = adaptive_policy.get("route_constraints") if isinstance(adaptive_policy.get("route_constraints"), dict) else {}
        force_new_direction = route_constraints.get("force_new_direction") is True
        if planner_gate.get("gate") != "pass" and not force_new_direction:
            errors.append("planner_gate_report.gate must be pass")
        if force_new_direction:
            if planner_gate.get("gate") != "fail" or planner_gate.get("return_to") != "S1_literature":
                errors.append("adaptive_policy.force_new_direction requires planner_gate fail with return_to=S1_literature")
        next_direction = str(next_variant.get("direction_id") or next_variant.get("s1_direction_id") or "")
        if direction_id and next_direction != direction_id:
            errors.append("next_variant.direction_id must match literature/direction.json direction_id")
        if direction_id and str(planner_gate.get("direction_id") or "") != direction_id:
            errors.append("planner_gate_report.direction_id must match literature/direction.json direction_id")
        if selected_id not in candidate_ids:
            errors.append("selected next_variant.id must exist in candidate_pool")
        if selected_fp and selected_fp not in candidate_fps:
            errors.append("selected next_variant.variant_fingerprint must exist in candidate_pool")
        if selected_fp and expected_fp and selected_fp != expected_fp:
            errors.append("selected next_variant.variant_fingerprint must match variant_fingerprint.json")
        if str(scorecard.get("selected_variant_id") or "") != selected_id:
            errors.append("variant_scorecard.selected_variant_id must match planner_gate_report.selected_variant_id")
        policy_hash = str(adaptive_policy.get("policy_hash") or "")
        if not policy_hash:
            errors.append("adaptive_policy.policy_hash must be non-empty")
        if policy_hash and str(scorecard.get("policy_hash") or "") != policy_hash:
            errors.append("variant_scorecard.policy_hash must match adaptive_policy.policy_hash")
        if policy_hash and str(planner_gate.get("policy_hash") or "") != policy_hash:
            errors.append("planner_gate_report.policy_hash must match adaptive_policy.policy_hash")
        if policy_hash and str(score_adjustment_report.get("policy_hash") or "") != policy_hash:
            errors.append("score_adjustment_report.policy_hash must match adaptive_policy.policy_hash")
        if str(score_adjustment_report.get("selected_variant_id") or "") != selected_id:
            errors.append("score_adjustment_report.selected_variant_id must match planner_gate_report.selected_variant_id")
        ranking = [item for item in scorecard.get("ranking") or [] if isinstance(item, dict)]
        selected_rows = [item for item in ranking if item.get("decision") == "selected"]
        if len(selected_rows) != 1:
            errors.append("variant_scorecard.ranking must contain exactly one selected row")
        required_components = {
            "proxy_calibration_prior",
            "route_history_prior",
            "dataset_risk_prior",
            "patch_surface_prior",
            "budget_prior",
        }
        missing_components = [
            {"variant_id": item.get("variant_id"), "missing": sorted(required_components - set((item.get("components") or {}).keys()))}
            for item in ranking
            if isinstance(item.get("components"), dict) and required_components - set(item["components"].keys())
        ]
        if missing_components:
            errors.append(f"variant_scorecard adaptive components missing: {missing_components[:3]}")
        adjustment_ids = {str(item.get("variant_id") or "") for item in score_adjustment_report.get("adjustments") or [] if isinstance(item, dict)}
        if candidate_ids - adjustment_ids:
            errors.append(f"score_adjustment_report.adjustments must cover every candidate: {sorted(candidate_ids - adjustment_ids)[:5]}")
        failed_points = {str(item) for item in route_constraints.get("failed_integration_points") or [] if item}
        if route_constraints.get("force_new_integration_point") and str(next_variant.get("integration_point") or "") in failed_points:
            errors.append("adaptive_policy.force_new_integration_point requires selected variant to use a new integration_point")
        expected_files = [str(item) for item in next_variant.get("expected_files") or [] if item]
        if not expected_files:
            errors.append("next_variant.expected_files must be non-empty")
        disallowed = self._disallowed_c2c_expected_files({"expected_files": expected_files})
        if disallowed:
            errors.append(f"next_variant expected_files outside allowed C2C edit surface: {disallowed[:5]}")
        ablation_switch = str(next_variant.get("ablation_switch") or ((next_variant.get("experiment_contract") or {}).get("ablation_switch") if isinstance(next_variant.get("experiment_contract"), dict) else "") or "")
        if not ablation_switch:
            errors.append("next_variant.ablation_switch must be present")
        rejected = scorecard.get("rejected_variants") if isinstance(scorecard.get("rejected_variants"), list) else []
        if len(candidates) > 1 and not all(isinstance(item, dict) and item.get("reasons") for item in rejected):
            errors.append("variant_scorecard.rejected_variants must include structured rejection reasons")
        if errors:
            self.retry_check(
                "s2_planner_gate_contract",
                "S2 planner gate did not produce a patch-ready selected variant",
                artifact="plan/s2_planner/planner_gate_report.json",
                details={"errors": errors[:12], "planner_gate_errors": planner_gate.get("errors") or []},
            )
        else:
            self.pass_check("s2_planner_gate_contract", artifact="plan/s2_planner/planner_gate_report.json")

    def _validate_s2_5_patch_gate(self) -> None:
        implementation_contract = self._read_required_json_with_schema(
            "plan/code_patches/implementation_contract.json",
            "s2_5_implementation_contract_json_exists",
            "s2_5_implementation_contract_schema",
            "s2_5_implementation_contract.schema.json",
        )
        patch_gate = self._read_required_json_with_schema(
            "plan/code_patches/patch_gate_report.json",
            "s2_5_patch_gate_report_json_exists",
            "s2_5_patch_gate_report_schema",
            "s2_5_patch_gate_report.schema.json",
        )
        patch_manifest = self.read_json_artifact("plan/code_patches/patch_manifest.json") or {}
        planner_gate = self.read_json_artifact("plan/s2_planner/planner_gate_report.json") or {}
        variant_fingerprint = self.read_json_artifact("plan/variant_fingerprint.json") or {}
        if not isinstance(implementation_contract, dict) or not isinstance(patch_gate, dict):
            return
        errors: list[str] = []
        if str(patch_gate.get("variant_id") or "") != str(planner_gate.get("selected_variant_id") or implementation_contract.get("variant_id") or ""):
            errors.append("patch_gate_report.variant_id must match planner_gate selected_variant_id")
        expected_fp = str(variant_fingerprint.get("variant_fingerprint") or implementation_contract.get("variant_fingerprint") or "")
        if str(patch_gate.get("variant_fingerprint") or "") != expected_fp:
            errors.append("patch_gate_report.variant_fingerprint must match variant_fingerprint.json")
        if patch_gate.get("gate") == "pass" and patch_manifest.get("status") != "ok":
            errors.append("patch_gate_report.gate pass requires patch_manifest.status ok")
        checks = patch_gate.get("checks") if isinstance(patch_gate.get("checks"), dict) else {}
        required_true = [
            "has_executable_change",
            "forbidden_files_untouched",
            "ablation_switch_present",
        ]
        for key in required_true:
            if checks.get(key) is not True:
                errors.append(f"patch_gate_report.checks.{key} must be true")
        if checks.get("selected_variant_matches_planner") is not True:
            errors.append("patch_gate_report.checks.selected_variant_matches_planner must be true")
        if errors:
            self.retry_check(
                "s2_5_patch_gate_contract",
                "S2.5 patch gate did not satisfy selected variant implementation contract",
                artifact="plan/code_patches/patch_gate_report.json",
                details={"errors": errors[:12], "failure_class": patch_gate.get("failure_class"), "repairable": patch_gate.get("repairable")},
            )
        else:
            self.pass_check("s2_5_patch_gate_contract", artifact="plan/code_patches/patch_gate_report.json")

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
