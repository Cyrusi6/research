"""LaTeX compile adapter."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LatexCompiler:
    config: dict[str, Any]

    def compile(self, paper_dir: Path) -> dict[str, Any]:
        engine = self.config.get("writing", {}).get("compile_engine", "pdflatex")
        if shutil.which(engine) is None:
            return {"status": "unavailable", "engine": engine, "message": f"{engine} not found"}
        commands = [
            [engine, "-interaction=nonstopmode", "main.tex"],
            ["bibtex", "main"] if shutil.which("bibtex") else None,
            [engine, "-interaction=nonstopmode", "main.tex"],
            [engine, "-interaction=nonstopmode", "main.tex"],
        ]
        outputs = []
        for command in [item for item in commands if item]:
            result = subprocess.run(command, cwd=paper_dir, capture_output=True, text=True)
            outputs.append({"command": command, "returncode": result.returncode, "stdout": result.stdout[-1000:], "stderr": result.stderr[-1000:]})
            if result.returncode != 0:
                return {"status": "failed", "engine": engine, "runs": outputs}
        return {"status": "ok", "engine": engine, "runs": outputs}
