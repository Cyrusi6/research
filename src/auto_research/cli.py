"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from .c2c import DEFAULT_C2C_ENV_PYTHON, DEFAULT_C2C_REPO
from .config import load_project_config, load_root_config
from .orchestrator import Orchestrator
from .reporting import build_memory_report, format_memory_report, format_project_report
from .utils import deep_merge, read_json, read_yaml, sha256_file, write_json, write_yaml


DEFAULT_C2C_RUN_REPO = "/home/lijunsi/projects/C2C"
DEFAULT_C2C_REF_PAPER = "/home/lijunsi/projects/ref_paper"
DEFAULT_C2C_REF_REBUTTAL = "/home/lijunsi/projects/ref_rebuttal"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-research", description="Auto research pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a project workspace")
    init_parser.add_argument("--topic", required=True)
    init_parser.add_argument("--project-id")
    init_parser.add_argument("--simulate", action="store_true")

    init_c2c_parser = subparsers.add_parser("init-c2c", help="Initialize a C2C auto-research MVP workspace")
    init_c2c_parser.add_argument("--topic", required=True)
    init_c2c_parser.add_argument("--target-repo", default=DEFAULT_C2C_REPO)
    init_c2c_parser.add_argument("--ref-paper", required=True)
    init_c2c_parser.add_argument("--ref-rebuttal", required=True)
    init_c2c_parser.add_argument("--env-python", default=DEFAULT_C2C_ENV_PYTHON)
    init_c2c_parser.add_argument("--project-id")
    init_c2c_parser.add_argument("--simulate", action="store_true")

    start_parser = subparsers.add_parser("start", help="Run a project pipeline")
    start_parser.add_argument("--project-id", required=True)

    run_c2c_parser = subparsers.add_parser(
        "run-c2c",
        help="Initialize/configure a C2C project and run the S0-S3 effect-first loop",
    )
    run_c2c_parser.add_argument("--topic", default="cross tokenizer cache communication")
    run_c2c_parser.add_argument("--project-id")
    run_c2c_parser.add_argument("--target-repo", default=None)
    run_c2c_parser.add_argument("--ref-paper", default=None)
    run_c2c_parser.add_argument("--ref-rebuttal", default=None)
    run_c2c_parser.add_argument("--env-python", default=DEFAULT_C2C_ENV_PYTHON)
    run_c2c_parser.add_argument("--max-iterations", type=int, default=3)
    run_c2c_parser.add_argument("--stop-after-stage", default="S3_experiment")
    run_c2c_parser.add_argument("--simulate", action="store_true")
    run_c2c_parser.add_argument("--hitl", action="store_true", help="Keep HITL approvals enabled instead of unattended auto mode")
    run_c2c_parser.add_argument("--no-s0-cache", action="store_true", help="Do not restore a compatible S0 static bundle from previous C2C projects")
    run_c2c_parser.add_argument("--s0-cache-project", help="Restore S0 static bundle from a specific previous project id")
    run_c2c_parser.add_argument("--s0-cache-path", help="Restore S0 static bundle from an explicit JSON path")
    run_c2c_parser.add_argument("--s0-force-refresh", action="store_true", help="Force S0 to regenerate instead of using any local or restored bundle")
    run_c2c_parser.add_argument("--prepare-only", action="store_true", help="Only initialize/configure/cache the project; do not start the pipeline")

    resume_parser = subparsers.add_parser("resume", help="Resume a project pipeline")
    resume_parser.add_argument("--project-id", required=True)

    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.add_argument("--project-id", required=True)

    report_parser = subparsers.add_parser("report", help="Show a concise project monitoring report")
    report_parser.add_argument("--project-id", required=True)
    report_parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of human-readable text")

    doctor_c2c_parser = subparsers.add_parser("doctor-c2c", help="Run C2C real-run readiness and runtime health checks")
    doctor_c2c_parser.add_argument("--project-id", required=True)

    audit_c2c_parser = subparsers.add_parser("audit-c2c", help="Audit C2C E2E artifacts after a run")
    audit_c2c_parser.add_argument("--project-id", required=True)

    replay_c2c_parser = subparsers.add_parser("replay-c2c", help="Replay deterministic C2C route decisions from frozen artifacts")
    replay_c2c_parser.add_argument("--project-id", required=True)
    replay_c2c_parser.add_argument("--from-stage", default="S3_experiment")

    memory_parser = subparsers.add_parser("memory", help="Inspect shared method failure memory")
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_report_parser = memory_subparsers.add_parser("report", help="Show shared method memory report")
    memory_report_parser.add_argument("--project-id", help="Show memory as retrieved by this project's config")
    memory_report_parser.add_argument("--limit", type=int, default=None, help="Override project prompt retrieval limit")
    memory_report_parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of human-readable text")

    enrich_s0_parser = subparsers.add_parser("enrich-s0", help="Run a small DeepSeek semantic enrichment sample for S0 chunks")
    enrich_s0_parser.add_argument("--project-id", required=True)
    enrich_s0_parser.add_argument("--limit", default="8", help="Number of chunks to enrich, or 'all'")
    enrich_s0_parser.add_argument("--source-types", default="paper,rebuttal,code", help="Comma-separated source types to sample")
    enrich_s0_parser.add_argument("--workers", type=int, default=1, help="Concurrent API workers for non-dry-run enrichment")
    enrich_s0_parser.add_argument("--dry-run", action="store_true", help="Estimate token/cost without calling the API")
    enrich_s0_parser.add_argument("--refresh", action="store_true", help="Ignore existing enrichment cache")
    enrich_s0_parser.add_argument("--json", action="store_true", help="Emit full structured JSON")

    review_parser = subparsers.add_parser("review", help="Run review stage")
    review_parser.add_argument("--project-id", required=True)

    catchup_parser = subparsers.add_parser("catchup", help="Show collaborator briefing")
    catchup_parser.add_argument("--project-id", required=True)

    import_consensus_parser = subparsers.add_parser("import-consensus", help="Import a Consensus dialogue export into literature inputs")
    import_consensus_parser.add_argument("--project-id", required=True)
    import_consensus_parser.add_argument("--file", required=True)
    import_consensus_parser.add_argument("--label")

    decide_parser = subparsers.add_parser("decide", help="Submit a human decision for a paused HITL stage")
    decide_parser.add_argument("--project-id", required=True)
    decide_parser.add_argument("--action", required=True, choices=["approve", "reject", "guide"])
    decide_parser.add_argument("--guidance", default="", help="Guidance for the pipeline (required for guide action)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrator = Orchestrator(repo_root=Path.cwd())

    if args.command == "init":
        project_id = orchestrator.init_project(args.topic, project_id=args.project_id, simulate=args.simulate)
        print(project_id)
        return
    if args.command == "init-c2c":
        project_id = orchestrator.init_c2c_project(
            args.topic,
            target_repo=Path(args.target_repo),
            ref_paper=Path(args.ref_paper),
            ref_rebuttal=Path(args.ref_rebuttal),
            env_python=Path(args.env_python),
            project_id=args.project_id,
            simulate=args.simulate,
        )
        print(project_id)
        return
    if args.command == "start":
        print(json.dumps(orchestrator.start(args.project_id), indent=2, ensure_ascii=False))
        return
    if args.command == "run-c2c":
        result = _run_c2c_command(args, orchestrator)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.command == "resume":
        print(json.dumps(orchestrator.resume(args.project_id), indent=2, ensure_ascii=False))
        return
    if args.command == "status":
        print(json.dumps(orchestrator.status(args.project_id), indent=2, ensure_ascii=False))
        return
    if args.command == "report":
        report = orchestrator.report(args.project_id)
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(format_project_report(report))
        return
    if args.command == "doctor-c2c":
        print(json.dumps(orchestrator.doctor_c2c(args.project_id), indent=2, ensure_ascii=False))
        return
    if args.command == "audit-c2c":
        print(json.dumps(orchestrator.audit_c2c(args.project_id), indent=2, ensure_ascii=False))
        return
    if args.command == "replay-c2c":
        print(json.dumps(orchestrator.replay_c2c(args.project_id, from_stage=args.from_stage), indent=2, ensure_ascii=False))
        return
    if args.command == "memory":
        project_root = None
        if args.project_id:
            project_root = orchestrator._project_root(args.project_id)
            config = load_project_config(project_root)
        else:
            config = load_root_config()
        report = build_memory_report(config=config, project_root=project_root, prompt_limit=args.limit)
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(format_memory_report(report))
        return
    if args.command == "enrich-s0":
        source_types = [item.strip() for item in args.source_types.split(",") if item.strip()]
        result = orchestrator.enrich_s0(
            args.project_id,
            limit=args.limit,
            source_types=source_types,
            dry_run=args.dry_run,
            refresh=args.refresh,
            workers=args.workers,
        )
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            report = result.get("report", {})
            cost = report.get("cost_summary", {})
            print(
                "\n".join(
                    [
                        f"S0 semantic enrichment: {result.get('status')}",
                        f"model: {report.get('model')}",
                        f"mode: {report.get('mode')}",
                        f"records: {report.get('success_count')}/{report.get('selected_count')} selected, failures={report.get('failure_count')}",
                        f"sample cost: ${cost.get('actual_sample_cost_usd', 0)} / ¥{cost.get('actual_sample_cost_cny', 0)}",
                        f"projected full cost: ${cost.get('projected_full_cost_usd', 0)} / ¥{cost.get('projected_full_cost_cny', 0)}",
                        "artifacts: " + ", ".join(result.get("artifacts", [])),
                    ]
                )
            )
        return
    if args.command == "review":
        print(json.dumps(orchestrator.run_review(args.project_id), indent=2, ensure_ascii=False))
        return
    if args.command == "catchup":
        print(orchestrator.catchup(args.project_id))
        return
    if args.command == "import-consensus":
        print(json.dumps(orchestrator.import_consensus(args.project_id, args.file, label=args.label), indent=2, ensure_ascii=False))
        return
    if args.command == "decide":
        from .hitl import HITLManager
        config = load_project_config(orchestrator._project_root(args.project_id))
        hitl = HITLManager(orchestrator._project_root(args.project_id), config)
        decision = hitl.submit_decision(args.action, guidance=args.guidance)
        print(json.dumps({
            "status": "decision_submitted",
            "action": decision.action,
            "guidance": decision.guidance,
            "responded_at": decision.responded_at,
            "hint": "Run 'auto-research resume --project-id {}' to continue the pipeline.".format(args.project_id),
        }, indent=2, ensure_ascii=False))

def _run_c2c_command(args: argparse.Namespace, orchestrator: Orchestrator) -> dict[str, object]:
    root_config = load_root_config()
    workspace_root = Path(root_config["project"]["workspace_root"])
    project_id = args.project_id
    project_root = workspace_root / project_id if project_id else None
    created = False

    if project_root is not None and project_root.exists():
        config = load_project_config(project_root)
        if not (config.get("c2c") or {}).get("enabled"):
            raise SystemExit(f"Existing project is not a C2C project: {project_id}")
    else:
        target_repo = _default_existing_path(args.target_repo, DEFAULT_C2C_RUN_REPO, DEFAULT_C2C_REPO)
        ref_paper = _default_existing_path(args.ref_paper, DEFAULT_C2C_REF_PAPER)
        ref_rebuttal = _default_existing_path(args.ref_rebuttal, DEFAULT_C2C_REF_REBUTTAL)
        project_id = orchestrator.init_c2c_project(
            args.topic,
            target_repo=target_repo,
            ref_paper=ref_paper,
            ref_rebuttal=ref_rebuttal,
            env_python=Path(args.env_python),
            project_id=project_id,
            simulate=args.simulate,
        )
        project_root = workspace_root / project_id
        created = True

    assert project_root is not None
    override_result = _apply_c2c_run_overrides(
        project_root,
        max_iterations=args.max_iterations,
        stop_after_stage=args.stop_after_stage,
        auto_mode=not args.hitl,
        s0_force_refresh=args.s0_force_refresh,
    )
    cache_result = {"status": "disabled"}
    if args.s0_force_refresh:
        cache_result = {"status": "disabled", "reason": "s0_force_refresh"}
    elif not args.no_s0_cache:
        cache_result = _restore_c2c_static_bundle_cache(
            project_root,
            workspace_root,
            source_project_id=args.s0_cache_project,
            source_path=Path(args.s0_cache_path) if args.s0_cache_path else None,
        )

    if args.prepare_only:
        return {
            "status": "prepared",
            "project_id": project_id,
            "created": created,
            "project_root": str(project_root),
            "run_overrides": override_result,
            "s0_cache": cache_result,
            "hint": f"Run 'auto-research resume --project-id {project_id}' to start.",
        }

    run_result = orchestrator.start(project_id) if created else orchestrator.resume(project_id)
    return {
        "status": run_result.get("status"),
        "project_id": project_id,
        "created": created,
        "project_root": str(project_root),
        "run_overrides": override_result,
        "s0_cache": cache_result,
        "run": run_result,
    }


def _default_existing_path(value: str | None, *defaults: str) -> Path:
    if value:
        return Path(value)
    for candidate in defaults:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(defaults[-1])


def _apply_c2c_run_overrides(
    project_root: Path,
    *,
    max_iterations: int,
    stop_after_stage: str,
    auto_mode: bool,
    s0_force_refresh: bool = False,
) -> dict[str, object]:
    if max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")
    config_path = project_root / "meta" / "project_config.yaml"
    project_config = read_yaml(config_path, default={}) or {}
    overrides = {
        "review": {"max_iterations": int(max_iterations)},
        "orchestration": {
            "auto_mode": bool(auto_mode),
            "stop_after_stage": stop_after_stage,
        },
    }
    if s0_force_refresh:
        overrides["c2c"] = {"s0_force_refresh": True}
    write_yaml(config_path, deep_merge(project_config, overrides))

    registry_path = project_root / "meta" / "registry.yaml"
    registry = read_yaml(registry_path, default={}) or {}
    registry["max_iterations"] = int(max_iterations)
    write_yaml(registry_path, registry)
    return {
        "max_iterations": int(max_iterations),
        "stop_after_stage": stop_after_stage,
        "auto_mode": bool(auto_mode),
        "s0_force_refresh": bool(s0_force_refresh),
        "project_config": "meta/project_config.yaml",
        "registry": "meta/registry.yaml",
    }


def _restore_c2c_static_bundle_cache(
    project_root: Path,
    workspace_root: Path,
    *,
    source_project_id: str | None = None,
    source_path: Path | None = None,
) -> dict[str, object]:
    current_path = project_root / "intake" / "c2c" / "static_bundle.json"
    current = _load_valid_c2c_static_bundle(current_path)
    if current is not None:
        return {
            "status": "current_project_cache_hit",
            "path": "intake/c2c/static_bundle.json",
            "chunk_count": len((current.get("chunk_index") or {}).get("entries") or []),
        }

    if source_path is not None:
        candidates = [source_path]
    elif source_project_id:
        candidates = [workspace_root / source_project_id / "intake" / "c2c" / "static_bundle.json"]
    else:
        candidates = sorted(
            workspace_root.glob("*/intake/c2c/static_bundle.json"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )
    rejected: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate.resolve() == current_path.resolve():
            continue
        bundle = _load_valid_c2c_static_bundle(candidate)
        if bundle is None:
            rejected.append({"path": str(candidate), "reason": "invalid_bundle"})
            continue
        ok, reason = _c2c_static_bundle_matches_project(project_root, bundle)
        if not ok:
            rejected.append({"path": str(candidate), "reason": reason})
            continue
        source_project = candidate.parents[2].name
        restored = dict(bundle)
        restored["project_id"] = project_root.name
        restored["cache_reuse"] = {
            "status": "restored_from_cross_project_static_bundle",
            "source_project_id": source_project,
            "source_path": str(candidate.relative_to(workspace_root.parent) if candidate.is_relative_to(workspace_root.parent) else candidate),
            "restored_at_unix": int(time.time()),
        }
        write_json(current_path, restored)
        sidecar_result = _restore_c2c_static_bundle_sidecars(
            source_project_root=candidate.parents[2],
            target_project_root=project_root,
            bundle=restored,
        )
        write_json(
            project_root / "intake" / "c2c" / "static_bundle_cache_source.json",
            {
                "status": "restored",
                "source_project_id": source_project,
                "source_path": str(candidate),
                "target_path": str(current_path),
                "chunk_count": len((restored.get("chunk_index") or {}).get("entries") or []),
                "sidecars": sidecar_result,
            },
        )
        return {
            "status": "restored",
            "source_project_id": source_project,
            "source_path": str(candidate),
            "path": "intake/c2c/static_bundle.json",
            "chunk_count": len((restored.get("chunk_index") or {}).get("entries") or []),
            "sidecars": sidecar_result,
        }
    return {
        "status": "miss",
        "reason": "no compatible C2C static bundle found",
        "rejected_candidates": rejected[:5],
    }


def _load_valid_c2c_static_bundle(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    bundle = read_json(path, default={})
    if not isinstance(bundle, dict):
        return None
    if bundle.get("schema_version") != "c2c_static_intake_bundle_v1":
        return None
    chunk_index = bundle.get("chunk_index") or {}
    entries = chunk_index.get("entries") if isinstance(chunk_index, dict) else None
    counts = chunk_index.get("counts") if isinstance(chunk_index, dict) else {}
    if not isinstance(entries, list) or not entries:
        return None
    if int((counts or {}).get("paper") or 0) <= 0:
        return None
    if int((counts or {}).get("rebuttal") or 0) <= 0:
        return None
    if int((counts or {}).get("code") or 0) <= 0:
        return None
    required = {
        "paper_chunks",
        "rebuttal_chunks",
        "code_file_manifest",
        "code_symbols",
        "code_chunks",
        "code_edges",
        "code_repo_map",
        "code_intake_report",
        "implementation_surface_map",
        "code_retrieval_index",
        "cache_summary",
        "paper_full_manifest",
        "evidence_brief",
    }
    if any(key not in bundle for key in required):
        return None
    return bundle


def _restore_c2c_static_bundle_sidecars(
    *,
    source_project_root: Path,
    target_project_root: Path,
    bundle: dict[str, object],
) -> dict[str, object]:
    copied: list[str] = []
    missing: list[str] = []
    skipped: list[str] = []
    for rel_path in _c2c_static_bundle_sidecar_paths(bundle):
        source = source_project_root / rel_path
        target = target_project_root / rel_path
        if target.exists():
            skipped.append(rel_path)
            continue
        if not source.exists() or not source.is_file():
            missing.append(rel_path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(rel_path)
    return {
        "copied_count": len(copied),
        "missing_count": len(missing),
        "skipped_count": len(skipped),
        "copied": copied[:10],
        "missing": missing[:10],
    }


def _c2c_static_bundle_sidecar_paths(bundle: dict[str, object]) -> list[str]:
    paths: list[str] = []
    for item in bundle.get("paper_full_manifest") or []:
        if not isinstance(item, dict):
            continue
        for key in ["local_path", "paper_full_md_path"]:
            _append_safe_relative_path(paths, item.get(key))
        for artifact in item.get("parser_artifacts") or []:
            _append_safe_relative_path(paths, artifact)
    return sorted(set(paths))


def _append_safe_relative_path(paths: list[str], raw: object) -> None:
    if not isinstance(raw, str) or not raw:
        return
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        return
    paths.append(str(path))


def _c2c_static_bundle_matches_project(project_root: Path, bundle: dict[str, object]) -> tuple[bool, str]:
    config = load_project_config(project_root)
    c2c_config = config.get("c2c") or {}
    snapshot_root = project_root / str(c2c_config.get("snapshot_path") or "external/c2c_snapshot")
    repo_manifest = bundle.get("repo_manifest") if isinstance(bundle.get("repo_manifest"), dict) else {}
    core_files = repo_manifest.get("core_files") if isinstance(repo_manifest, dict) else []
    if not isinstance(core_files, list) or not core_files:
        return True, "no_core_file_fingerprint"
    checked = 0
    mismatches: list[str] = []
    missing: list[str] = []
    for item in core_files:
        if not isinstance(item, dict):
            continue
        rel_path = item.get("path")
        expected_sha = item.get("sha256")
        if not rel_path or not expected_sha:
            continue
        current_file = snapshot_root / str(rel_path)
        if not current_file.exists():
            missing.append(str(rel_path))
            continue
        checked += 1
        if sha256_file(current_file) != expected_sha:
            mismatches.append(str(rel_path))
    if missing:
        return False, f"missing_core_files:{','.join(missing[:3])}"
    if mismatches:
        return False, f"core_file_sha_mismatch:{','.join(mismatches[:3])}"
    if checked == 0:
        return True, "no_checked_core_files"
    return True, f"matched_core_files:{checked}"


if __name__ == "__main__":
    main()
