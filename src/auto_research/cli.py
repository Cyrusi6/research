"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .c2c import DEFAULT_C2C_ENV_PYTHON, DEFAULT_C2C_REPO
from .orchestrator import Orchestrator


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
        return
    if args.command == "decide":
        from .hitl import HITLManager
        from .config import load_project_config
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


if __name__ == "__main__":
    main()
