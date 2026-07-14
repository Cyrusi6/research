"""LLM integration."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised in minimal test environments
    OpenAI = None

from .utils import now_utc, read_yaml, write_yaml


@dataclass
class GenerationResult:
    text: str
    raw: Any = None


class ProviderUnavailableError(RuntimeError):
    def __init__(self, *, provider: str, purpose: str, reason: str):
        self.provider = provider
        self.purpose = purpose
        self.reason = reason
        self.status = "provider_unavailable"
        super().__init__(f"{provider} provider unavailable for {purpose}: {reason}")

    def as_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "provider": self.provider,
            "purpose": self.purpose,
            "reason": self.reason,
        }


class ModelClient:
    def __init__(self, config: dict[str, Any], project_root: Path | None = None):
        llm_config = config.get("llm", {})
        self.llm_config = dict(llm_config)
        self.provider = llm_config.get("provider", "openai")
        self.reasoning_provider = llm_config.get("reasoning_provider", self.provider)
        self.execution_provider = llm_config.get("execution_provider", self.provider)
        raw_model = str(llm_config.get("model", "gpt-5-mini"))
        self.reasoning_effort = _normalize_optional_str(
            llm_config.get("reasoning_effort")
            or os.environ.get("OPENAI_REASONING_EFFORT")
        )
        self.model, inferred_reasoning_effort = _split_model_and_effort(raw_model)
        if not self.reasoning_effort:
            self.reasoning_effort = inferred_reasoning_effort
        self.temperature = llm_config.get("temperature", 0.2)
        self.timeout_seconds = llm_config.get("timeout_seconds", 60)
        self.request_retries = max(1, int(llm_config.get("request_retries", 2) or 1))
        self.request_retry_backoff_seconds = float(llm_config.get("request_retry_backoff_seconds", 0) or 0)
        self.project_root = project_root
        self.codex_cli_config = llm_config.get("codex_cli", {})
        self.agent_config = config.get("agents", {})
        self.json_retries = int(llm_config.get("json_retries", 1) or 1)
        self._openai_api_keys = _collect_openai_api_keys(llm_config)
        self.openai_api_key = self._openai_api_keys[0] if self._openai_api_keys else None
        self.openai_base_url = _normalize_openai_base_url(
            llm_config.get("base_url")
            or os.environ.get("OPENAI_BASE_URL")
        )
        self.openai_organization = _normalize_optional_str(
            llm_config.get("organization")
            or os.environ.get("OPENAI_ORGANIZATION")
        )
        self.openai_project = _normalize_optional_str(
            llm_config.get("project")
            or os.environ.get("OPENAI_PROJECT")
        )
        self.openai_default_headers = _parse_headers(
            llm_config.get("default_headers")
            or os.environ.get("OPENAI_DEFAULT_HEADERS")
        )
        self._openai_client_kwargs = self._build_openai_client_kwargs()
        self.simulate = bool(config.get("experiment", {}).get("simulate"))
        self._codex_executable = shutil.which("codex")
        self._codex_available = bool(self._codex_executable)
        self._openai_clients = self._build_openai_clients()
        self._openai_client = self._openai_clients[0] if self._openai_clients else None
        self._openai_client_cursor = 0
        self.use_real_api = self._provider_can_use(self.reasoning_provider)
        self.use_real_execution_api = self._provider_can_use(self.execution_provider)
        if self.simulate:
            self.use_real_api = False
            self.use_real_execution_api = False
        if not llm_config.get("use_real_api", True):
            self.use_real_api = False
            self.use_real_execution_api = False

    def generate(
        self,
        *,
        instructions: str,
        prompt: str,
        agent_name: str | None = None,
        temperature_override: float | None = None,
        purpose: str = "reasoning",
    ) -> GenerationResult:
        agent_settings = self.agent_config.get(agent_name or "", {})
        model, agent_reasoning_effort = _split_model_and_effort(str(agent_settings.get("model", self.model)))
        temperature = temperature_override
        if temperature is None:
            temperature = agent_settings.get("temperature", self.temperature)
        provider = self._provider_for_purpose(purpose)
        if provider == "codex_cli" and self.simulate:
            return GenerationResult(
                text=self._mock_text(prompt),
                raw={"provider": "mock", "policy": "simulate"},
            )
        if provider == "codex_cli" and not self._codex_executable:
            raise ProviderUnavailableError(
                provider=provider,
                purpose=purpose,
                reason="executable_not_found",
            )
        if provider == "codex_cli":
            return self._generate_via_codex_cli(instructions=instructions, prompt=prompt, agent_name=agent_name)
        if provider != "openai" or not self.use_real_api or not self._provider_can_use(provider):
            return GenerationResult(text=self._mock_text(prompt))
        request_kwargs: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": prompt,
            "temperature": temperature,
        }
        reasoning_effort = self.reasoning_effort or agent_reasoning_effort
        if purpose == "reasoning" and reasoning_effort and reasoning_effort != "none":
            request_kwargs["reasoning"] = {"effort": reasoning_effort}
        if not self._openai_clients:
            return GenerationResult(text=self._mock_text(prompt))
        client_count = len(self._openai_clients)
        start_index = self._openai_client_cursor % client_count
        last_error: Exception | None = None
        total_attempts = max(client_count, client_count * self.request_retries)
        for attempt in range(total_attempts):
            client_index = (start_index + attempt) % client_count
            client = self._openai_clients[client_index]
            response, error = self._try_openai_response(client, request_kwargs)
            if response is not None:
                self._openai_client_cursor = client_index
                return GenerationResult(text=response.output_text, raw=response)
            if error is None:
                continue
            last_error = error
            if not _is_openai_key_fallback_error(error):
                raise error
            self._openai_client_cursor = (client_index + 1) % client_count
            if attempt + 1 < total_attempts:
                self._sleep_before_retry(error, attempt=attempt)
        if last_error is not None:
            raise last_error
        return GenerationResult(text=self._mock_text(prompt))

    def generate_json(self, *, instructions: str, prompt: str, default: Any, agent_name: str | None = None, purpose: str = "reasoning") -> Any:
        return self.generate_json_with_schema(
            instructions=instructions,
            prompt=prompt,
            default=default,
            agent_name=agent_name,
            purpose=purpose,
        )

    def generate_json_with_schema(
        self,
        *,
        instructions: str,
        prompt: str,
        default: Any,
        schema: dict[str, Any] | None = None,
        retries: int | None = None,
        agent_name: str | None = None,
        temperature_override: float | None = None,
        purpose: str = "reasoning",
    ) -> Any:
        attempts = retries if retries is not None else self.json_retries
        attempts = max(1, attempts)
        schema_text = f"\n\nJSON schema hints:\n{json.dumps(schema, ensure_ascii=False)}" if schema else ""
        wrapped_prompt = (
            f"{prompt}{schema_text}\n\nReturn only valid JSON. Do not wrap the response in markdown fences."
        )
        last_text = ""
        for attempt in range(attempts):
            repair_suffix = ""
            if attempt:
                repair_suffix = (
                    "\n\nThe previous response was not valid JSON or did not match the expected shape. "
                    "Return corrected JSON only."
                )
            result = self.generate(
                instructions=instructions,
                prompt=wrapped_prompt + repair_suffix,
                agent_name=agent_name,
                temperature_override=temperature_override,
                purpose=purpose,
            ).text.strip()
            last_text = result
            parsed = self._parse_json_text(result)
            if parsed is not None and self._matches_schema_hint(parsed, schema):
                return parsed
        parsed = self._parse_json_text(last_text)
        return parsed if parsed is not None and not schema else default

    @staticmethod
    def _parse_json_text(text: str) -> Any | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                return None
        start_candidates = [idx for idx in [text.find("{"), text.find("[")] if idx >= 0]
        if not start_candidates:
            return None
        start = min(start_candidates)
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _matches_schema_hint(payload: Any, schema: dict[str, Any] | None) -> bool:
        if not schema:
            return True
        expected_type = schema.get("type")
        if expected_type == "object" and not isinstance(payload, dict):
            return False
        if expected_type == "array" and not isinstance(payload, list):
            return False
        required = schema.get("required") or []
        if required and isinstance(payload, dict):
            return all(key in payload for key in required)
        return True

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
        if not self._codex_executable:
            raise ProviderUnavailableError(
                provider="codex_cli",
                purpose="execution",
                reason="executable_not_found",
            )
        command = [self._codex_executable]
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
                env=codex_subprocess_env({"llm": self.llm_config}),
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

    def _provider_for_purpose(self, purpose: str) -> str:
        if purpose == "execution":
            return self.execution_provider
        return self.reasoning_provider

    def _provider_can_use(self, provider: str) -> bool:
        if provider == "openai":
            return bool(self.openai_api_key and OpenAI is not None)
        if provider == "codex_cli":
            return self._codex_available
        if provider in {"mock", "none"}:
            return False
        return False

    def _build_openai_client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url
        if self.openai_organization:
            kwargs["organization"] = self.openai_organization
        if self.openai_project:
            kwargs["project"] = self.openai_project
        if self.openai_default_headers:
            kwargs["default_headers"] = self.openai_default_headers
        return kwargs

    def _build_openai_clients(self) -> list[Any]:
        if OpenAI is None or not self._openai_api_keys:
            return []
        clients: list[Any] = []
        for api_key in self._openai_api_keys:
            kwargs = dict(self._openai_client_kwargs)
            kwargs["api_key"] = api_key
            clients.append(OpenAI(**kwargs))
        return clients

    @staticmethod
    def _try_openai_response(client: Any, request_kwargs: dict[str, Any]) -> tuple[Any | None, Exception | None]:
        try:
            return client.responses.create(**request_kwargs), None
        except Exception as exc:  # pragma: no cover - exercised with live API failures
            return None, exc

    def _sleep_before_retry(self, exc: Exception, *, attempt: int) -> None:
        retry_after = _extract_retry_after(exc)
        if retry_after is None:
            retry_after = self.request_retry_backoff_seconds
        if retry_after <= 0:
            return
        time.sleep(min(float(retry_after), 60.0))


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    text = value.strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return None
    return text


def _collect_openai_api_keys(llm_config: dict[str, Any]) -> list[str]:
    keys: list[str] = []

    def add(value: Any) -> None:
        text = _normalize_optional_str(value)
        if text and text not in keys:
            keys.append(text)

    api_key_env = _normalize_optional_str(
        llm_config.get("api_key_env") or llm_config.get("openai_api_key_env")
    )
    strict_env = bool(
        llm_config.get("disable_api_key_fallback")
        or llm_config.get("api_key_env_strict")
    )
    if api_key_env:
        add(os.environ.get(api_key_env))
        if strict_env:
            return keys

    add(llm_config.get("api_key"))
    add(os.environ.get("OPENAI_API_KEY"))
    numbered: list[tuple[int, str]] = []
    for env_key, env_value in os.environ.items():
        match = re.fullmatch(r"OPENAI_API_KEY_(\d+)", env_key)
        if not match:
            continue
        text = _normalize_optional_str(env_value)
        if not text:
            continue
        numbered.append((int(match.group(1)), text))
    for _, value in sorted(numbered, key=lambda item: item[0]):
        add(value)
    add(os.environ.get("OPENAI_API_TOKEN"))
    return keys


def codex_subprocess_env(config: dict[str, Any]) -> dict[str, str]:
    """Build an OpenAI-key-scoped environment for Codex CLI subprocesses."""
    env = os.environ.copy()
    llm_config = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
    api_key_env = _normalize_optional_str(
        llm_config.get("api_key_env") or llm_config.get("openai_api_key_env")
    )
    strict_env = bool(
        llm_config.get("disable_api_key_fallback")
        or llm_config.get("api_key_env_strict")
    )
    selected_key = _normalize_optional_str(env.get(api_key_env)) if api_key_env else None

    if strict_env:
        for key in list(env):
            if re.fullmatch(r"OPENAI_API_KEY(_\d+)?", key) and key != api_key_env:
                env.pop(key, None)
        env.pop("OPENAI_API_TOKEN", None)
        env.pop("CODEX_API_KEY", None)
        env.pop("CODEX_API_TOKEN", None)

    if selected_key:
        env["OPENAI_API_KEY"] = selected_key
        if api_key_env:
            env[api_key_env] = selected_key

    return env


def _is_openai_key_fallback_error(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code in {401, 402, 403, 429}:
        return True
    text = f"{exc.__class__.__name__} {exc}".lower()
    return any(marker in text for marker in ["insufficient_quota", "quota", "rate limit", "rate_limit", "billing", "payment required"])


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _extract_retry_after(exc: Exception) -> float | None:
    candidates: list[Any] = []
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
            try:
                candidates.append(headers.get(key))
            except AttributeError:
                pass
    for value in candidates:
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return None


def _normalize_openai_base_url(value: Any) -> str | None:
    text = _normalize_optional_str(value)
    if not text:
        return None
    endpoint_was_explicit = False
    if "/v1/responses" in text:
        text = text.replace("/v1/responses", "/v1")
    elif text.rstrip("/").endswith("/responses"):
        text = text.rstrip("/")[: -len("/responses")]
        endpoint_was_explicit = True
    text = text.rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc and not parsed.path and not endpoint_was_explicit:
        text = urlunparse(parsed._replace(path="/v1"))
    return text.rstrip("/")


def _parse_headers(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): str(val) for key, val in value.items() if key and val is not None}
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {str(key): str(val) for key, val in parsed.items() if key and val is not None}
    except json.JSONDecodeError:
        pass
    headers: dict[str, str] = {}
    for raw_item in re.split(r"[;\n,]+", text):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            key, val = item.split(":", 1)
        elif "=" in item:
            key, val = item.split("=", 1)
        else:
            continue
        key = key.strip()
        val = val.strip()
        if key and val:
            headers[key] = val
    return headers


def _split_model_and_effort(raw_model: str) -> tuple[str, str | None]:
    model = raw_model.strip()
    if not model:
        return "gpt-5-mini", None
    efforts = ("xhigh", "high", "medium", "low", "none")
    lowered = model.lower()
    for effort in efforts:
        for suffix in (effort, f"-{effort}", f"_{effort}"):
            if lowered.endswith(suffix):
                base = model[: -len(suffix)].rstrip("-_")
                if base:
                    return base, effort
    return model, None


@dataclass
class _CodexRunResult:
    returncode: int
    text: str
    stdout: str
    stderr: str
    parsed_session_id: str | None = None
