"""LLM integration."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from .utils import now_utc, read_yaml, write_yaml


@dataclass
class GenerationResult:
    text: str
    raw: Any = None


class ModelClient:
    def __init__(self, config: dict[str, Any], project_root: Path | None = None):
        llm_config = config.get("llm", {})
        self.provider = llm_config.get("provider", "openai")
        self.model = llm_config.get("model", "gpt-5-mini")
        self.temperature = llm_config.get("temperature", 0.2)
        self.timeout_seconds = llm_config.get("timeout_seconds", 60)
        self.project_root = project_root
        self.codex_cli_config = llm_config.get("codex_cli", {})
        if self.provider == "codex_cli":
            self.use_real_api = bool(shutil.which("codex"))
        else:
            self.use_real_api = bool(llm_config.get("use_real_api", True) and os.environ.get("OPENAI_API_KEY"))
        self._client = (
            OpenAI(timeout=self.timeout_seconds)
            if self.provider == "openai" and self.use_real_api
            else None
        )

    def generate(self, *, instructions: str, prompt: str, agent_name: str | None = None) -> GenerationResult:
        if self.provider == "codex_cli":
            return self._generate_via_codex_cli(instructions=instructions, prompt=prompt, agent_name=agent_name)
        if self.provider != "openai" or not self.use_real_api:
            return GenerationResult(text=self._mock_text(prompt))
        response = self._client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
            temperature=self.temperature,
        )
        return GenerationResult(text=response.output_text, raw=response)

    def generate_json(self, *, instructions: str, prompt: str, default: Any, agent_name: str | None = None) -> Any:
        wrapped_prompt = (
            f"{prompt}\n\nReturn only valid JSON. Do not wrap the response in markdown fences."
        )
        result = self.generate(instructions=instructions, prompt=wrapped_prompt, agent_name=agent_name).text.strip()
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return default

    def _generate_via_codex_cli(self, *, instructions: str, prompt: str, agent_name: str | None = None) -> GenerationResult:
        agent_name = agent_name or "default-agent"
        session_id = self._load_codex_session(agent_name) if self.codex_cli_config.get("use_resume", True) else None
        merged_prompt = self._merge_instructions(instructions, prompt)
        result = self._run_codex_command(merged_prompt=merged_prompt, session_id=session_id)
        if result.returncode != 0 and session_id:
            result = self._run_codex_command(merged_prompt=merged_prompt, session_id=None)
        if result.returncode != 0:
            raise RuntimeError(
                f"codex exec failed for {agent_name} with code {result.returncode}: {result.stderr[-1000:]}"
            )
        if result.parsed_session_id:
            self._save_codex_session(agent_name, result.parsed_session_id)
        return GenerationResult(
            text=result.text.strip(),
            raw={
                "stderr": result.stderr,
                "stdout": result.stdout,
                "session_id": result.parsed_session_id or session_id,
            },
        )

    def _run_codex_command(self, *, merged_prompt: str, session_id: str | None) -> "_CodexRunResult":
        working_root = self.project_root.resolve() if self.project_root else Path.cwd()
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as handle:
            output_path = Path(handle.name)
        command = ["codex"]
        sandbox = self.codex_cli_config.get("sandbox")
        approval_policy = self.codex_cli_config.get("approval_policy")
        if sandbox:
            command.extend(["-s", sandbox])
        if approval_policy:
            command.extend(["-a", approval_policy])
        command.extend(["exec", "--skip-git-repo-check", "--output-last-message", str(output_path)])
        if self.model:
            command.extend(["-m", self.model])
        if self.project_root:
            command.extend(["-C", str(working_root)])
        if session_id:
            command.extend(["resume", session_id, merged_prompt])
        else:
            command.append(merged_prompt)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=working_root,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            output_text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            output_path.unlink(missing_ok=True)
            return _CodexRunResult(
                returncode=124,
                text=output_text,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + f"\nTimeout after {self.timeout_seconds}s",
                parsed_session_id=None,
            )
        try:
            text = output_path.read_text(encoding="utf-8")
        finally:
            output_path.unlink(missing_ok=True)
        parsed_session_id = self._parse_session_id(result.stderr)
        return _CodexRunResult(
            returncode=result.returncode,
            text=text,
            stdout=result.stdout,
            stderr=result.stderr,
            parsed_session_id=parsed_session_id,
        )

    @staticmethod
    def _merge_instructions(instructions: str, prompt: str) -> str:
        return (
            "Follow the instructions exactly.\n\n"
            "<instructions>\n"
            f"{instructions.strip()}\n"
            "</instructions>\n\n"
            "<task>\n"
            f"{prompt.strip()}\n"
            "</task>"
        )

    @staticmethod
    def _parse_session_id(stderr: str) -> str | None:
        match = re.search(r"session id:\s*([0-9a-fA-F-]+)", stderr)
        return match.group(1) if match else None

    def _session_file(self) -> Path | None:
        if not self.project_root:
            return None
        return self.project_root / "meta" / "codex_sessions.yaml"

    def _load_codex_session(self, agent_name: str) -> str | None:
        path = self._session_file()
        if not path or not path.exists():
            return None
        payload = read_yaml(path, default={"sessions": {}}) or {"sessions": {}}
        return payload.get("sessions", {}).get(agent_name, {}).get("session_id")

    def _save_codex_session(self, agent_name: str, session_id: str) -> None:
        path = self._session_file()
        if not path:
            return
        payload = read_yaml(path, default={"sessions": {}}) or {"sessions": {}}
        payload.setdefault("sessions", {})
        payload["sessions"][agent_name] = {
            "session_id": session_id,
            "provider": "codex_cli",
            "model": self.model,
            "updated_at": now_utc(),
        }
        write_yaml(path, payload)

    @staticmethod
    def _mock_text(prompt: str) -> str:
        snippet = " ".join(prompt.split())[:500]
        return f"Mock generation based on: {snippet}"


@dataclass
class _CodexRunResult:
    returncode: int
    text: str
    stdout: str
    stderr: str
    parsed_session_id: str | None = None
