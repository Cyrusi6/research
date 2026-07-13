"""Strict S3 TrialResult v1 gate."""

from __future__ import annotations

from auto_research.config import bootstrap_profile_enabled
from auto_research.domain_contracts import contract_errors, validate_direction_identity, validate_variant_identity
from auto_research.utils import read_json

from .base import StageGateValidator, load_schema, validate_min_schema


class S3GateValidator(StageGateValidator):
    stage_key = "S3_experiment"
    validator_name = "s3_experiment_gate_v2"

    def validate(self):
        required = [
            "literature/direction.json",
            "plan/variant.json",
            "experiment/results/trial_result.json",
            "experiment/results/main_results.json",
            "experiment/results/ablation_results.json",
            "experiment/results/hypothesis_verification.md",
        ]
        if not all(self.require_file(path, retry=True) for path in required):
            return self.finalize()
        direction = self.read_json_artifact("literature/direction.json")
        variant = self.read_json_artifact("plan/variant.json")
        trial = self.read_json_artifact("experiment/results/trial_result.json")
        if not all(isinstance(item, dict) for item in [direction, variant, trial]):
            self.retry_check("s3_authoritative_json", "DirectionSpec, VariantSpec, and TrialResult must be JSON objects")
            return self.finalize()
        state = read_json(self.project_root / "meta" / "research_state.json", default={}) or {}
        errors = contract_errors(trial, "trial_result_v1.schema.json")
        try:
            validate_direction_identity(direction)
            validate_variant_identity(direction, variant)
        except ValueError as exc:
            errors.append(str(exc))
        for key in ["direction_id", "direction_hash", "variant_id", "variant_spec_hash"]:
            expected = direction.get(key) if key.startswith("direction") else variant.get(key)
            if trial.get(key) != expected:
                errors.append(f"TrialResult {key} mismatch")
        attempt = (state.get("attempts") or {}).get(trial.get("attempt_id"))
        if not isinstance(attempt, dict):
            errors.append("TrialResult attempt_id is not present in the event-derived state")
        else:
            if attempt.get("attempt_input_hash") != trial.get("attempt_input_hash"):
                errors.append("TrialResult attempt_input_hash mismatch")
            if bool(attempt.get("method_evaluable")) != bool(trial.get("method_evaluable")):
                errors.append("TrialResult method_evaluable disagrees with reducer state")
            budget = ((state.get("directions") or {}).get(direction["direction_hash"]) or {}).get("budget") or {}
            if int(budget.get("consumed", 0)) > 5 or int(budget.get("reserved", 0)) + int(budget.get("consumed", 0)) > 5:
                errors.append("direction budget exceeds five slots")
        if errors:
            self.retry_check("trial_result_v1", "TrialResult identity, hash, or budget validation failed", artifact="experiment/results/trial_result.json", details={"errors": errors[:20]})
        else:
            self.pass_check("trial_result_v1", artifact="experiment/results/trial_result.json")

        if bootstrap_profile_enabled(self.config):
            completion = read_json(self.project_root / "experiment" / "results" / "bootstrap_proxy_completion.json", default={}) or {}
            if trial.get("method_evaluable") and completion.get("bootstrap_proxy_complete") is True and not attempt.get("consumes_direction_budget"):
                self.pass_check("bootstrap_proxy_complete", artifact="experiment/results/bootstrap_proxy_completion.json")
            else:
                self.retry_check("bootstrap_proxy_complete", "bootstrap requires one evaluable proxy that does not consume the standard budget")
        self._validate_proxy_contracts_if_present()
        return self.finalize()

    def _validate_proxy_contracts_if_present(self):
        artifacts = {
            "baseline_fingerprint": ("experiment/results/c2c_proxy_baseline_fingerprint.json", "c2c_proxy_baseline_fingerprint.schema.json"),
            "cache_report": ("experiment/results/c2c_proxy_cache_report.json", "c2c_proxy_cache_report.schema.json"),
            "effective_policy": ("experiment/results/c2c_effective_proxy_policy.json", "c2c_effective_proxy_policy.schema.json"),
            "decision_report": ("experiment/results/c2c_proxy_decision_report.json", "c2c_proxy_decision_report.schema.json"),
            "calibration_policy": ("experiment/results/c2c_proxy_calibration_policy.json", "c2c_proxy_calibration_policy.schema.json"),
        }
        present = any((self.project_root / path).exists() for path, _ in artifacts.values())
        if not present:
            return
        for name, (path, schema_name) in artifacts.items():
            if not (self.project_root / path).exists():
                continue
            self.pass_check(f"{name}_exists", artifact=path)
            payload = self.read_json_artifact(path)
            errors = validate_min_schema(payload, load_schema(schema_name)) if isinstance(payload, dict) else ["expected object"]
            if errors:
                self.retry_check(name, f"{path} strict schema validation failed", artifact=path, details={"errors": errors[:20]})
            else:
                self.pass_check(name, artifact=path)
