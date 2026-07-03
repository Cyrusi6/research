"""Validator for auditable route policy artifacts."""

from __future__ import annotations

from .base import StageGateValidator, load_schema, validate_min_schema


class RoutePolicyGateValidator(StageGateValidator):
    stage_key = "route_policy"
    validator_name = "route_policy_gate_v1"

    def validate(self):
        decision_path = self.require_file("meta/route_decision.json", check_name="route_decision_exists")
        if decision_path:
            decision = self.read_json_artifact("meta/route_decision.json")
            errors = validate_min_schema(decision, load_schema("route_decision.schema.json")) if decision is not None else []
            if errors:
                self.retry_check(
                    "route_decision_schema",
                    "meta/route_decision.json failed schema: " + "; ".join(errors[:5]),
                    artifact="meta/route_decision.json",
                    details={"errors": errors},
                )
            else:
                self.pass_check("route_decision_schema", artifact="meta/route_decision.json")

        ledger_path = self.require_file("meta/attempt_ledger.json", check_name="attempt_ledger_exists")
        if ledger_path:
            ledger = self.read_json_artifact("meta/attempt_ledger.json")
            errors = validate_min_schema(ledger, load_schema("attempt_ledger.schema.json")) if ledger is not None else []
            if errors:
                self.retry_check(
                    "attempt_ledger_schema",
                    "meta/attempt_ledger.json failed schema: " + "; ".join(errors[:5]),
                    artifact="meta/attempt_ledger.json",
                    details={"errors": errors},
                )
            else:
                self.pass_check("attempt_ledger_schema", artifact="meta/attempt_ledger.json")

        return self.finalize(default_reason="Route policy artifacts are valid.")
