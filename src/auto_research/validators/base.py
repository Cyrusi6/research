"""Executable stage gate validators."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..utils import now_utc, read_json


PASS = "PASS"
FAIL = "FAIL"
NEEDS_RETRY = "NEEDS_RETRY"
GATE_SCHEMA_VERSION = "stage_gate_v1"


@dataclass
class GateCheck:
    name: str
    status: str
    message: str = ""
    artifact: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "artifact": self.artifact,
            "details": self.details,
        }
        return {key: value for key, value in payload.items() if value not in (None, "", {})}


@dataclass
class GateReport:
    stage: str
    status: str
    reason: str = ""
    checks: list[GateCheck] = field(default_factory=list)
    artifacts_checked: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_utc)
    validator: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "stage": self.stage,
            "status": self.status,
            "passed": self.passed,
            "reason": self.reason,
            "validator": self.validator or f"{self.stage.lower()}_gate_v1",
            "created_at": self.created_at,
            "artifacts_checked": sorted(set(self.artifacts_checked)),
            "checks": [check.to_dict() for check in self.checks],
        }

    def legacy_tuple(self) -> tuple[bool, str]:
        return self.passed, self.reason


class StageGateValidator:
    stage_key: str = ""
    validator_name: str = ""

    def __init__(self, project_root: Path, config: dict[str, Any] | None = None):
        self.project_root = project_root
        self.config = config or {}
        self.checks: list[GateCheck] = []
        self.artifacts_checked: list[str] = []

    def validate(self) -> GateReport:
        raise NotImplementedError

    def pass_check(self, name: str, *, artifact: str | None = None, message: str = "", details: dict[str, Any] | None = None) -> None:
        self._add_check(name, PASS, message=message, artifact=artifact, details=details)

    def retry_check(self, name: str, message: str, *, artifact: str | None = None, details: dict[str, Any] | None = None) -> None:
        self._add_check(name, NEEDS_RETRY, message=message, artifact=artifact, details=details)

    def fail_check(self, name: str, message: str, *, artifact: str | None = None, details: dict[str, Any] | None = None) -> None:
        self._add_check(name, FAIL, message=message, artifact=artifact, details=details)

    def _add_check(self, name: str, status: str, *, message: str = "", artifact: str | None = None, details: dict[str, Any] | None = None) -> None:
        self.checks.append(GateCheck(name=name, status=status, message=message, artifact=artifact, details=details or {}))
        if artifact:
            self.artifacts_checked.append(artifact)

    def finalize(self, *, default_reason: str = "") -> GateReport:
        retry = next((check for check in self.checks if check.status == NEEDS_RETRY), None)
        fail = next((check for check in self.checks if check.status == FAIL), None)
        if retry:
            status = NEEDS_RETRY
            reason = retry.message
        elif fail:
            status = FAIL
            reason = fail.message
        else:
            status = PASS
            reason = default_reason
        return GateReport(
            stage=self.stage_key,
            status=status,
            reason=reason,
            checks=list(self.checks),
            artifacts_checked=list(self.artifacts_checked),
            validator=self.validator_name or f"{self.stage_key.lower()}_gate_v1",
        )

    def require_file(self, rel_path: str, *, check_name: str | None = None, retry: bool = True) -> Path | None:
        path = self.project_root / rel_path
        name = check_name or f"{rel_path}_exists"
        if path.exists():
            self.pass_check(name, artifact=rel_path)
            return path
        message = f"{rel_path} missing"
        if retry:
            self.retry_check(name, message, artifact=rel_path)
        else:
            self.fail_check(name, message, artifact=rel_path)
        return None

    def read_json_artifact(self, rel_path: str) -> Any | None:
        path = self.project_root / rel_path
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.retry_check(f"{rel_path}_valid_json", f"{rel_path} is not valid JSON: {exc}", artifact=rel_path)
            return None

    def read_yaml_artifact(self, rel_path: str) -> Any | None:
        path = self.project_root / rel_path
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.retry_check(f"{rel_path}_valid_yaml", f"{rel_path} is not valid YAML: {exc}", artifact=rel_path)
            return None


def validate_min_schema(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Small JSON-schema subset for required contracts without extra dependency."""

    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _matches_type(payload, expected_type):
        return [f"expected {expected_type}"]
    if isinstance(payload, dict):
        for key in schema.get("required", []):
            if key not in payload:
                errors.append(f"missing required key: {key}")
        properties = schema.get("properties") or {}
        for key, subschema in properties.items():
            if key in payload:
                errors.extend(f"{key}.{error}" for error in validate_min_schema(payload[key], subschema))
    if isinstance(payload, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(payload) < int(min_items):
            errors.append(f"expected at least {min_items} items")
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(payload):
                errors.extend(f"[{idx}].{error}" for error in validate_min_schema(item, item_schema))
    return errors


def load_schema(name: str) -> dict[str, Any]:
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / name
    return read_json(schema_path, default={}) or {}


def _matches_type(payload: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(payload, dict)
    if expected_type == "array":
        return isinstance(payload, list)
    if expected_type == "string":
        return isinstance(payload, str)
    if expected_type == "number":
        return isinstance(payload, (int, float)) and not isinstance(payload, bool)
    if expected_type == "integer":
        return isinstance(payload, int) and not isinstance(payload, bool)
    if expected_type == "boolean":
        return isinstance(payload, bool)
    return True
