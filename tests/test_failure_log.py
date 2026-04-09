import json
from pathlib import Path

from auto_research.failure_log import FailureLogManager


def test_failure_log_manager_records_and_cleans_runs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    failed_run = runs_root / "bad_run"
    failed_run.mkdir(parents=True)
    (failed_run / "train.log").write_text("dummy\n", encoding="utf-8")

    manager = FailureLogManager(
        {"experiment": {"failure_log_filename": "failure.md"}},
        external_root=runs_root,
    )
    removed = manager.record_not_viable_ideas(
        project_id="proj_x",
        baseline_metrics={"rsum": 10, "t2i": {"R@1": 5}, "similarity_time": 1.0},
        candidate_results=[
            {
                "id": "idea_bad",
                "title": "Bad Idea",
                "direction": "test",
                "decision": "not_viable",
                "metrics": {"rsum": 8, "t2i": {"R@1": 4}, "similarity_time": 1.3},
                "train_log": str(failed_run / "train.log"),
            }
        ],
        cleanup=True,
    )

    assert removed == [str(failed_run)]
    assert not failed_run.exists()
    assert (runs_root / "failure.md").exists()
    assert (runs_root / "failure.jsonl").exists()
    payload = [json.loads(line) for line in (runs_root / "failure.jsonl").read_text(encoding="utf-8").splitlines()]
    assert payload[0]["idea_id"] == "idea_bad"

