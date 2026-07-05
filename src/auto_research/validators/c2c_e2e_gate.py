"""Validators for C2C E2E orchestration artifacts."""

from __future__ import annotations

from .base import StageGateValidator, load_schema, validate_min_schema


class C2CE2EGateValidator(StageGateValidator):
    stage_key = "c2c_e2e"
    validator_name = "c2c_e2e_gate_v1"

    def validate(self):
        self.validate_readiness_report(required=False)
        self.validate_execution_hooks_report(required=False)
        self.validate_artifact_audit_report(required=False)
        self.validate_replay_result(required=False)
        self.validate_real_smoke_record(required=False)
        return self.finalize(default_reason="C2C E2E orchestration artifacts are valid.")

    def validate_readiness_report(self, *, required: bool = True) -> None:
        self._validate_json_artifact(
            "meta/c2c_e2e_readiness_report.json",
            "c2c_e2e_readiness_report.schema.json",
            "c2c_e2e_readiness_report",
            required=required,
        )

    def validate_execution_hooks_report(self, *, required: bool = True) -> None:
        self._validate_json_artifact(
            "meta/c2c_execution_hooks_report.json",
            "c2c_execution_hooks_report.schema.json",
            "c2c_execution_hooks_report",
            required=required,
        )

    def validate_artifact_audit_report(self, *, required: bool = True) -> None:
        self._validate_json_artifact(
            "meta/c2c_artifact_audit_report.json",
            "c2c_artifact_audit_report.schema.json",
            "c2c_artifact_audit_report",
            required=required,
        )

    def validate_replay_result(self, *, required: bool = True) -> None:
        self._validate_json_artifact(
            "meta/c2c_replay_result.json",
            "c2c_replay_result.schema.json",
            "c2c_replay_result",
            required=required,
        )

    def validate_real_smoke_record(self, *, required: bool = True) -> None:
        self._validate_json_artifact(
            "meta/c2c_real_smoke_record.json",
            "c2c_real_smoke_record.schema.json",
            "c2c_real_smoke_record",
            required=required,
        )

    def _validate_json_artifact(self, rel_path: str, schema_name: str, check_prefix: str, *, required: bool) -> None:
        path = self.project_root / rel_path
        if not path.exists():
            if required:
                self.retry_check(f"{check_prefix}_exists", f"{rel_path} missing", artifact=rel_path)
            return
        self.pass_check(f"{check_prefix}_exists", artifact=rel_path)
        payload = self.read_json_artifact(rel_path)
        errors = validate_min_schema(payload, load_schema(schema_name)) if payload is not None else []
        if errors:
            self.retry_check(
                f"{check_prefix}_schema",
                f"{rel_path} failed schema: " + "; ".join(errors[:5]),
                artifact=rel_path,
                details={"errors": errors},
            )
        else:
            self.pass_check(f"{check_prefix}_schema", artifact=rel_path)


def validate_readiness_report(project_root, config=None):
    validator = C2CE2EGateValidator(project_root, config)
    validator.validate_readiness_report(required=True)
    return validator.finalize(default_reason="C2C E2E readiness report is valid.")


def validate_artifact_audit_report(project_root, config=None):
    validator = C2CE2EGateValidator(project_root, config)
    validator.validate_artifact_audit_report(required=True)
    return validator.finalize(default_reason="C2C artifact audit report is valid.")


def validate_replay_result(project_root, config=None):
    validator = C2CE2EGateValidator(project_root, config)
    validator.validate_replay_result(required=True)
    return validator.finalize(default_reason="C2C replay result is valid.")


def validate_real_smoke_record(project_root, config=None):
    validator = C2CE2EGateValidator(project_root, config)
    validator.validate_real_smoke_record(required=True)
    return validator.finalize(default_reason="C2C real smoke record is valid.")
