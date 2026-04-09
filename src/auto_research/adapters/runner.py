"""Execution and environment helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRunner:
    config: dict[str, Any]

    def env_report(self) -> dict[str, Any]:
        report = {"python": shutil.which("python3") or shutil.which("python"), "gpu": None, "tmux": bool(shutil.which("tmux"))}
        if shutil.which("nvidia-smi"):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            report["gpu"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return report

    def run_plan_commands(self, commands: list[str], working_dir: Path, log_path: Path) -> dict[str, Any]:
        outputs = []
        for command in commands:
            result = subprocess.run(command, cwd=working_dir, shell=True, capture_output=True, text=True)
            outputs.append(
                {
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-2000:],
                }
            )
            if result.returncode != 0:
                log_path.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
                return {"status": "failed", "runs": outputs}
        log_path.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
        return {"status": "ok", "runs": outputs}
