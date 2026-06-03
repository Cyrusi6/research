"""Execution and environment helpers."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils import now_utc


@dataclass
class GpuSelection:
    selected_ids: list[int]
    policy: dict[str, Any]
    snapshot: list[dict[str, Any]]
    reason: str

    @property
    def cuda_visible_devices(self) -> str:
        return ",".join(str(item) for item in self.selected_ids)


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

    def select_gpus(self, policy: dict[str, Any] | None = None) -> GpuSelection:
        policy = dict(policy or {})
        max_gpus = int(policy.get("max_gpus") or self.config.get("experiment", {}).get("gpu_policy", {}).get("max_gpus", 6))
        explicit = policy.get("gpu_ids")
        if explicit is None:
            explicit = self.config.get("c2c", {}).get("small_loop", {}).get("gpu_ids")
        if explicit not in (None, "auto"):
            selected = _coerce_gpu_ids(explicit)[:max_gpus]
            return GpuSelection(selected, policy, self._gpu_snapshot(), "explicit_gpu_ids")

        min_free_mb = int(policy.get("min_free_mb") or self.config.get("experiment", {}).get("gpu_policy", {}).get("min_free_mb", 0))
        snapshot = self._gpu_snapshot()
        candidates = [
            item
            for item in snapshot
            if item.get("memory_free_mb", 0) >= min_free_mb
        ]
        candidates.sort(key=lambda item: (-item.get("memory_free_mb", 0), item.get("utilization_gpu", 100), item.get("index", 999)))
        selected = [int(item["index"]) for item in candidates[:max_gpus]]
        if not selected and snapshot:
            fallback = sorted(snapshot, key=lambda item: (-item.get("memory_free_mb", 0), item.get("index", 999)))
            selected = [int(fallback[0]["index"])]
        return GpuSelection(selected, policy, snapshot, "auto_selected_by_free_memory")

    @staticmethod
    def _gpu_snapshot() -> list[dict[str, Any]]:
        if not shutil.which("nvidia-smi"):
            return []
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []
        snapshot = []
        for line in result.stdout.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                snapshot.append(
                    {
                        "index": int(parts[0]),
                        "memory_total_mb": int(parts[1]),
                        "memory_free_mb": int(parts[2]),
                        "memory_used_mb": int(parts[3]),
                        "utilization_gpu": int(parts[4]),
                    }
                )
            except ValueError:
                continue
        return snapshot

    def run_step(
        self,
        *,
        name: str,
        command: str,
        working_dir: Path,
        retry_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retry_policy = retry_policy or {}
        max_attempts = max(1, int(retry_policy.get("max_attempts", 1) or 1))
        timeout_seconds = _coerce_timeout_seconds(retry_policy.get("timeout_seconds"))
        attempts = []
        status = "failed"
        for attempt in range(1, max_attempts + 1):
            started_at = now_utc()
            started_monotonic = time.monotonic()
            timed_out = False
            try:
                process = subprocess.Popen(
                    command,
                    cwd=working_dir,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                returncode = process.returncode
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                _terminate_process_group(process.pid)
                stdout, stderr = process.communicate()
                returncode = 124
                stdout = stdout or _decode_timeout_output(exc.stdout)
                stderr = stderr or _decode_timeout_output(exc.stderr)
                timeout_note = f"Command timed out after {timeout_seconds}s"
                stderr = (stderr + "\n" if stderr else "") + timeout_note
            entry = {
                "step": name,
                "attempt": attempt,
                "command": command,
                "returncode": returncode,
                "stdout": _output_excerpt(stdout),
                "stderr": _output_excerpt(stderr),
                "started_at": started_at,
                "completed_at": now_utc(),
                "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
            }
            if timeout_seconds is not None:
                entry["timeout_seconds"] = timeout_seconds
            if timed_out:
                entry["timed_out"] = True
            attempts.append(entry)
            if returncode == 0:
                status = "ok"
                break
        return {"step": name, "status": status, "attempts": attempts, "returncode": attempts[-1]["returncode"]}

    def run_plan_commands(self, commands: list[str], working_dir: Path, log_path: Path) -> dict[str, Any]:
        outputs = []
        for command in commands:
            step_result = self.run_step(name=f"command_{len(outputs)}", command=command, working_dir=working_dir)
            last = step_result["attempts"][-1]
            outputs.append(
                {
                    "command": command,
                    "returncode": last["returncode"],
                    "stdout": last["stdout"][-2000:],
                    "stderr": last["stderr"][-2000:],
                }
            )
            if step_result["status"] != "ok":
                log_path.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
                return {"status": "failed", "runs": outputs}
        log_path.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
        return {"status": "ok", "runs": outputs}


def _coerce_gpu_ids(value: Any) -> list[int]:
    if value in (None, "", "auto"):
        return []
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _output_excerpt(text: str, *, limit: int = 12000, edge: int = 5500) -> str:
    if len(text) <= limit:
        return text
    return (
        text[:edge]
        + f"\n\n... <truncated {len(text) - 2 * edge} chars; preserving head and tail> ...\n\n"
        + text[-edge:]
    )


def _coerce_timeout_seconds(value: Any) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        return
    time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        return
