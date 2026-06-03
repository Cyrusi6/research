"""Stage gate validator registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import FAIL, NEEDS_RETRY, PASS, GateReport
from .s0_gate import S0GateValidator
from .s1_gate import S1GateValidator
from .s2_gate import S2GateValidator
from .s3_gate import S3GateValidator
from .s4_gate import S4GateValidator
from .s5_gate import S5GateValidator


VALIDATORS = {
    "S0_intake": S0GateValidator,
    "S1_literature": S1GateValidator,
    "S2_plan": S2GateValidator,
    "S3_experiment": S3GateValidator,
    "S4_writing": S4GateValidator,
    "S5_review": S5GateValidator,
}


def run_stage_gate(stage_key: str, project_root: Path, config: dict[str, Any] | None = None) -> GateReport:
    validator_cls = VALIDATORS[stage_key]
    return validator_cls(project_root, config).validate()


__all__ = ["FAIL", "NEEDS_RETRY", "PASS", "GateReport", "run_stage_gate"]
