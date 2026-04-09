"""Stage gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def judge_s1(project_root: Path) -> tuple[bool, str]:
    ideas_path = project_root / "literature" / "ideas.json"
    manifest_path = project_root / "references" / "papers" / "manifest.json"
    if not ideas_path.exists():
        return False, "ideas.json missing"
    ideas = json.loads(ideas_path.read_text(encoding="utf-8"))
    if len(ideas) < 3:
        return False, "fewer than 3 ideas"
    for idea in ideas:
        if idea.get("novelty_score", 0) < 4 or idea.get("feasibility_score", 0) < 4:
            return False, "idea score below threshold"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("papers") and not (project_root / "literature" / "papers" / "metadata.json").exists():
        return False, "no reference papers registered"
    return True, ""


def judge_s2(project_root: Path) -> tuple[bool, str]:
    path = project_root / "plan" / "plan.yaml"
    if not path.exists():
        return False, "plan.yaml missing"
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    checks = [
        len(plan.get("hypotheses", [])) >= 1,
        len(plan.get("baselines", [])) >= 2,
        len(plan.get("datasets", [])) >= 1,
        "task_graph" in plan,
        "resource_budget" in plan,
    ]
    return (all(checks), "plan incomplete" if not all(checks) else "")


def judge_s3(project_root: Path) -> tuple[bool, str]:
    results_dir = project_root / "experiment" / "results"
    required = [
        results_dir / "main_results.json",
        results_dir / "ablation_results.json",
        results_dir / "hypothesis_verification.md",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        return False, f"missing experiment outputs: {', '.join(missing)}"
    return True, ""


def judge_s4(project_root: Path, config: dict[str, Any]) -> tuple[bool, str]:
    paper_dir = project_root / "paper"
    if not (paper_dir / "main.tex").exists():
        return False, "main.tex missing"
    audit_path = paper_dir / "claim_audit.json"
    if not audit_path.exists():
        return False, "claim_audit.json missing"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    threshold = config.get("writing", {}).get("claim_verification", {}).get("min_pass_rate", 0.8)
    if audit.get("pass_rate", 0) < threshold:
        return False, "claim audit below threshold"
    compile_report_path = paper_dir / "compile_report.json"
    if compile_report_path.exists():
        compile_report = json.loads(compile_report_path.read_text(encoding="utf-8"))
        require_compile = config.get("writing", {}).get("require_compile", False)
        if require_compile and compile_report.get("status") != "ok":
            return False, "latex compile failed"
    return True, ""


def judge_s5(project_root: Path) -> tuple[bool, str]:
    review_dir = project_root / "review"
    required = [
        review_dir / "reviewer_A_round_1.md",
        review_dir / "reviewer_B_round_1.md",
        review_dir / "reviewer_C_round_1.md",
        review_dir / "meta_review_round_1.md",
        review_dir / "revision_dispatch.yaml",
    ]
    if not (review_dir / "revision_dispatch.yaml").exists():
        return False, "revision dispatch missing"
    if not (review_dir / "score_history.json").exists():
        return False, "score history missing"
    return True, ""
