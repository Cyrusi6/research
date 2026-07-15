import os
from pathlib import Path

from auto_research.resources import _scan_reusable_runs, best_itr_execution_plan, discover_local_mm_resources


def test_discover_local_mm_resources_handles_missing_root() -> None:
    resources = discover_local_mm_resources({"project": {"external_resource_roots": {"mm_root": "/path/does/not/exist"}}})
    assert resources["available"] is False


def test_best_itr_execution_plan_prefers_laps_eval_when_local_assets_exist() -> None:
    resources = {
        "available": True,
        "codebases": {"laps_change": {"available": True, "root": "/tmp/laps", "python": "python", "python_ready": True}},
        "datasets": {"coco": {"available": True, "data_path": "/tmp/data"}},
        "checkpoints": {"laps_coco_vit": "/tmp/vit.pth", "laps_coco_swin": "/tmp/swin.pth"},
    }
    plan = best_itr_execution_plan(resources)
    assert plan["mode"] == "scripted"
    assert plan["collector"] == "laps_eval"
    assert any("eval.py" in command for command in plan["commands"])


def test_scan_reusable_runs_skips_directories_that_disappear_during_walk(tmp_path: Path, monkeypatch) -> None:
    valid_run = tmp_path / "runs" / "valid"
    valid_run.mkdir(parents=True)
    (valid_run / "eval.log").write_text("rsum: 42.0\n", encoding="utf-8")
    vanishing = tmp_path / "runs" / "vanishing"
    vanishing.mkdir()

    real_scandir = os.scandir

    def disappearing_scandir(path):
        if Path(path) == vanishing:
            vanishing.rmdir()
            raise FileNotFoundError(path)
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", disappearing_scandir)

    records = _scan_reusable_runs(tmp_path)

    assert [record["rsum"] for record in records] == [42.0]


def test_scan_reusable_runs_ignores_project_temp_directory(tmp_path: Path) -> None:
    temp_run = tmp_path / ".tmp" / "concurrent-test"
    temp_run.mkdir(parents=True)
    (temp_run / "eval.log").write_text("rsum: 999.0\n", encoding="utf-8")
    valid_run = tmp_path / "runs" / "valid"
    valid_run.mkdir(parents=True)
    (valid_run / "eval.log").write_text("rsum: 42.0\n", encoding="utf-8")

    records = _scan_reusable_runs(tmp_path)

    assert [record["rsum"] for record in records] == [42.0]
