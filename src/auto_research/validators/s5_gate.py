"""S5 review gate."""

from __future__ import annotations

from .base import StageGateValidator, load_schema, validate_min_schema


class S5GateValidator(StageGateValidator):
    stage_key = "S5_review"
    validator_name = "s5_review_gate_v1"

    def validate(self):
        required = [
            "review/reviewer_A_round_1.md",
            "review/reviewer_B_round_1.md",
            "review/reviewer_C_round_1.md",
            "review/meta_review_round_1.md",
            "review/revision_dispatch.yaml",
            "review/score_history.json",
        ]
        for rel in required:
            self.require_file(rel, check_name=f"{rel}_exists")

        dispatch_path = self.project_root / "review" / "revision_dispatch.yaml"
        if dispatch_path.exists():
            dispatch = self.read_yaml_artifact("review/revision_dispatch.yaml")
            if isinstance(dispatch, dict):
                schema_errors = validate_min_schema(dispatch, load_schema("revision_dispatch.schema.json"))
                if schema_errors:
                    self.retry_check("revision_dispatch_schema", "revision_dispatch.yaml does not satisfy dispatch contract", artifact="review/revision_dispatch.yaml", details={"errors": schema_errors[:10]})
                else:
                    self.pass_check("revision_dispatch_schema", artifact="review/revision_dispatch.yaml")
        return self.finalize()
