"""Strict S2 VariantSpec v3 and implementation gate."""

from __future__ import annotations

from auto_research.domain_contracts import contract_errors, validate_direction_identity, validate_trial_spec, validate_variant_identity
from auto_research.research_state import ResearchEventLedger
from auto_research.utils import read_json

from .base import StageGateValidator


class S2GateValidator(StageGateValidator):
    stage_key = "S2_plan"
    validator_name = "s2_plan_gate_v2"

    def validate(self):
        required = [
            "literature/direction.json",
            "plan/variant.json",
            "plan/trial_spec.json",
            "plan/planner_decision.json",
        ]
        if not all(self.require_file(path, retry=True) for path in required):
            return self.finalize()
        direction = self.read_json_artifact("literature/direction.json")
        variant = self.read_json_artifact("plan/variant.json")
        trial_spec = self.read_json_artifact("plan/trial_spec.json")
        if not isinstance(direction, dict) or not isinstance(variant, dict) or not isinstance(trial_spec, dict):
            self.retry_check("s2_authoritative_json", "DirectionSpec, VariantSpec, and TrialSpec must be JSON objects")
            return self.finalize()
        state = ResearchEventLedger(self.project_root).state()
        tried = [item for item in state.get("method_tried_history") or [] if isinstance(item, dict)]
        errors = contract_errors(variant, "variant_v4.schema.json")
        try:
            validate_direction_identity(direction)
            validate_variant_identity(direction, variant, tried_variants=tried)
        except ValueError as exc:
            errors.append(str(exc))
        if errors:
            self.retry_check("variant_v4", "VariantSpec v4 validation failed", artifact="plan/variant.json", details={"errors": errors[:20]})
        else:
            self.pass_check("variant_v4", artifact="plan/variant.json", details={"variant_id": variant["variant_id"], "variant_semantic_hash": variant["variant_semantic_hash"], "variant_spec_hash": variant["variant_spec_hash"]})
        trial_errors = contract_errors(trial_spec, "trial_spec_v5.schema.json")
        try:
            validate_trial_spec(trial_spec)
        except ValueError as exc:
            trial_errors.append(str(exc))
        if trial_errors:
            self.retry_check("trial_spec", "TrialSpec v5 validation failed", artifact="plan/trial_spec.json", details={"errors": trial_errors[:20]})
        else:
            self.pass_check("trial_spec", artifact="plan/trial_spec.json")

        planner_gate_path = self.project_root / "plan" / "s2_planner" / "planner_gate_report.json"
        if planner_gate_path.exists():
            planner_gate = self.read_json_artifact("plan/s2_planner/planner_gate_report.json") or {}
            selected_id = planner_gate.get("selected_variant_id")
            selected_hash = planner_gate.get("selected_variant_spec_hash") or planner_gate.get("selected_variant_fingerprint")
            planner_errors = []
            if planner_gate.get("gate") != "pass":
                planner_errors.append("planner gate did not pass")
            if selected_id not in {None, variant["variant_id"]}:
                planner_errors.append("selected_variant_id mismatch")
            if selected_hash not in {None, variant["variant_spec_hash"]}:
                planner_errors.append("selected variant hash mismatch")
            if planner_errors:
                self.retry_check("planner_gate", "planner diagnostics disagree with VariantSpec", artifact="plan/s2_planner/planner_gate_report.json", details={"errors": planner_errors})
            else:
                self.pass_check("planner_gate", artifact="plan/s2_planner/planner_gate_report.json")

        code_patch_enabled = bool((self.config.get("code_patch") or {}).get("enabled"))
        manifest_path = self.project_root / "plan" / "code_patches" / "patch_manifest.json"
        if code_patch_enabled or manifest_path.exists():
            for path in ["plan/code_patches/implementation_contract.json", "plan/code_patches/patch_gate_report.json", "plan/code_patches/patch_manifest.json"]:
                self.require_file(path, retry=True)
            patch_gate = read_json(self.project_root / "plan" / "code_patches" / "patch_gate_report.json", default={}) or {}
            manifest = read_json(manifest_path, default={}) or {}
            patch_errors = []
            if patch_gate.get("gate") != "pass":
                patch_errors.append("patch gate did not pass")
            gate_hash = patch_gate.get("variant_spec_hash") or patch_gate.get("variant_fingerprint")
            if gate_hash not in {None, variant["variant_spec_hash"]}:
                patch_errors.append("patch gate variant hash mismatch")
            if manifest.get("status") in {"failed", "blocked", "planner_gate_failed"}:
                patch_errors.append("patch manifest is not executable")
            if patch_errors:
                self.retry_check("s2_5_patch_gate", "S2.5 implementation gate failed", details={"errors": patch_errors})
            else:
                self.pass_check("s2_5_patch_gate")
        return self.finalize()
