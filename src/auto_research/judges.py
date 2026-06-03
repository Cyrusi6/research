"""Compatibility wrappers for executable stage gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .validators import GateReport, run_stage_gate


def gate_s0(project_root: Path, config: dict[str, Any] | None = None) -> GateReport:
    return run_stage_gate("S0_intake", project_root, config)


def gate_s1(project_root: Path, config: dict[str, Any] | None = None) -> GateReport:
    return run_stage_gate("S1_literature", project_root, config)


def gate_s2(project_root: Path, config: dict[str, Any] | None = None) -> GateReport:
    return run_stage_gate("S2_plan", project_root, config)


def gate_s3(project_root: Path, config: dict[str, Any] | None = None) -> GateReport:
    return run_stage_gate("S3_experiment", project_root, config)


def gate_s4(project_root: Path, config: dict[str, Any]) -> GateReport:
    return run_stage_gate("S4_writing", project_root, config)


def gate_s5(project_root: Path, config: dict[str, Any] | None = None) -> GateReport:
    return run_stage_gate("S5_review", project_root, config)


def judge_s0(project_root: Path) -> tuple[bool, str]:
    return gate_s0(project_root).legacy_tuple()


def judge_s1(project_root: Path) -> tuple[bool, str]:
    return gate_s1(project_root).legacy_tuple()


def judge_s2(project_root: Path) -> tuple[bool, str]:
    return gate_s2(project_root).legacy_tuple()


def judge_s3(project_root: Path) -> tuple[bool, str]:
    return gate_s3(project_root).legacy_tuple()


def judge_s4(project_root: Path, config: dict[str, Any]) -> tuple[bool, str]:
    return gate_s4(project_root, config).legacy_tuple()


def judge_s5(project_root: Path) -> tuple[bool, str]:
    return gate_s5(project_root).legacy_tuple()
