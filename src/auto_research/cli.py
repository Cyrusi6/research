"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestrator import Orchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-research", description="Auto research pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a project workspace")
    init_parser.add_argument("--topic", required=True)
    init_parser.add_argument("--project-id")
    init_parser.add_argument("--simulate", action="store_true")

    start_parser = subparsers.add_parser("start", help="Run a project pipeline")
    start_parser.add_argument("--project-id", required=True)

    resume_parser = subparsers.add_parser("resume", help="Resume a project pipeline")
    resume_parser.add_argument("--project-id", required=True)

    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.add_argument("--project-id", required=True)

    review_parser = subparsers.add_parser("review", help="Run review stage")
    review_parser.add_argument("--project-id", required=True)

    catchup_parser = subparsers.add_parser("catchup", help="Show collaborator briefing")
    catchup_parser.add_argument("--project-id", required=True)

    import_consensus_parser = subparsers.add_parser("import-consensus", help="Import a Consensus dialogue export into literature inputs")
    import_consensus_parser.add_argument("--project-id", required=True)
    import_consensus_parser.add_argument("--file", required=True)
    import_consensus_parser.add_argument("--label")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrator = Orchestrator(repo_root=Path.cwd())

    if args.command == "init":
        project_id = orchestrator.init_project(args.topic, project_id=args.project_id, simulate=args.simulate)
        print(project_id)
        return
    if args.command == "start":
        print(json.dumps(orchestrator.start(args.project_id), indent=2, ensure_ascii=False))
        return
    if args.command == "resume":
        print(json.dumps(orchestrator.resume(args.project_id), indent=2, ensure_ascii=False))
        return
    if args.command == "status":
        print(json.dumps(orchestrator.status(args.project_id), indent=2, ensure_ascii=False))
        return
    if args.command == "review":
        print(json.dumps(orchestrator.run_review(args.project_id), indent=2, ensure_ascii=False))
        return
    if args.command == "catchup":
        print(orchestrator.catchup(args.project_id))
        return
    if args.command == "import-consensus":
        print(json.dumps(orchestrator.import_consensus(args.project_id, args.file, label=args.label), indent=2, ensure_ascii=False))
