import os
from pathlib import Path
from types import SimpleNamespace

from auto_research.llm import ModelClient, codex_subprocess_env


def test_openai_api_key_rotation_prefers_base_key_then_numbered(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "base-key")
    monkeypatch.setenv("OPENAI_API_KEY_1", "first-key")
    monkeypatch.setenv("OPENAI_API_KEY_2", "second-key")
    client = ModelClient(
        {
            "llm": {
                "provider": "openai",
                "reasoning_provider": "openai",
                "base_url": "https://api-cdn.owlai.tech/v1/responses",
                "model": "gpt-5.4",
                "timeout_seconds": 10,
                "use_real_api": False,
            }
        }
    )

    assert client._openai_api_keys[:3] == ["base-key", "first-key", "second-key"]


def test_openai_api_key_env_strict_uses_only_configured_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "base-key")
    monkeypatch.setenv("OPENAI_API_KEY_1", "first-key")
    monkeypatch.setenv("OPENAI_API_KEY_2", "second-key")
    monkeypatch.setenv("OPENAI_API_TOKEN", "token-key")
    client = ModelClient(
        {
            "llm": {
                "provider": "openai",
                "reasoning_provider": "openai",
                "api_key_env": "OPENAI_API_KEY_2",
                "disable_api_key_fallback": True,
                "model": "gpt-5.4",
                "timeout_seconds": 10,
                "use_real_api": False,
            }
        }
    )

    assert client._openai_api_keys == ["second-key"]


def test_codex_subprocess_env_scopes_only_openai_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "base-key")
    monkeypatch.setenv("OPENAI_API_KEY_1", "first-key")
    monkeypatch.setenv("OPENAI_API_KEY_2", "second-key")
    monkeypatch.setenv("OPENAI_API_TOKEN", "token-key")
    monkeypatch.setenv("CODEX_API_KEY", "stale-codex-key")
    monkeypatch.setenv("CODEX_API_TOKEN", "stale-codex-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("MINERU_API_KEY", "mineru-key")

    env = codex_subprocess_env(
        {
            "llm": {
                "api_key_env": "OPENAI_API_KEY_2",
                "disable_api_key_fallback": True,
            }
        }
    )

    assert env["OPENAI_API_KEY"] == "second-key"
    assert env["OPENAI_API_KEY_2"] == "second-key"
    assert "OPENAI_API_KEY_1" not in env
    assert "OPENAI_API_TOKEN" not in env
    assert "CODEX_API_KEY" not in env
    assert "CODEX_API_TOKEN" not in env
    assert env["DEEPSEEK_API_KEY"] == "deepseek-key"
    assert env["MINERU_API_KEY"] == "mineru-key"


def test_openai_api_key_rotation_falls_back_on_quota(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "base-key")
    monkeypatch.setenv("OPENAI_API_KEY_1", "first-key")

    class FakeQuotaError(Exception):
        status_code = 429

    calls = []

    class FakeResponses:
        def __init__(self, api_key: str):
            self.api_key = api_key

        def create(self, **kwargs):
            calls.append(self.api_key)
            if self.api_key == "first-key":
                raise FakeQuotaError("quota exceeded")
            return SimpleNamespace(output_text='{"ok": true}')

    class FakeClient:
        def __init__(self, api_key: str, **kwargs):
            self.responses = FakeResponses(api_key)

    monkeypatch.setattr("auto_research.llm.OpenAI", FakeClient)

    client = ModelClient(
        {
            "llm": {
                "provider": "openai",
                "reasoning_provider": "openai",
                "base_url": "https://api-cdn.owlai.tech/v1/responses",
                "model": "gpt-5.4",
                "timeout_seconds": 10,
                "use_real_api": True,
            }
        }
    )

    result = client.generate(
        instructions="Return JSON.",
        prompt="{}",
    )

    assert result.text == '{"ok": true}'
    assert calls == ["base-key"]


def test_openai_api_key_rotation_falls_back_from_base_to_numbered(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "base-key")
    monkeypatch.setenv("OPENAI_API_KEY_1", "first-key")

    class FakeQuotaError(Exception):
        status_code = 429

    calls = []

    class FakeResponses:
        def __init__(self, api_key: str):
            self.api_key = api_key

        def create(self, **kwargs):
            calls.append(self.api_key)
            if self.api_key == "base-key":
                raise FakeQuotaError("quota exceeded")
            return SimpleNamespace(output_text='{"ok": true}')

    class FakeClient:
        def __init__(self, api_key: str, **kwargs):
            self.responses = FakeResponses(api_key)

    monkeypatch.setattr("auto_research.llm.OpenAI", FakeClient)

    client = ModelClient(
        {
            "llm": {
                "provider": "openai",
                "reasoning_provider": "openai",
                "base_url": "https://api-cdn.owlai.tech/v1/responses",
                "model": "gpt-5.4",
                "timeout_seconds": 10,
                "use_real_api": True,
            }
        }
    )

    result = client.generate(
        instructions="Return JSON.",
        prompt="{}",
    )

    assert result.text == '{"ok": true}'
    assert calls == ["base-key", "first-key"]


def test_openai_api_key_rotation_retries_all_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "base-key")
    monkeypatch.setenv("OPENAI_API_KEY_1", "first-key")

    class FakeRateLimitError(Exception):
        status_code = 429

    calls = []

    class FakeResponses:
        def __init__(self, api_key: str):
            self.api_key = api_key

        def create(self, **kwargs):
            calls.append(self.api_key)
            if len(calls) < 3:
                raise FakeRateLimitError("Too Many Requests")
            return SimpleNamespace(output_text='{"ok": true}')

    class FakeClient:
        def __init__(self, api_key: str, **kwargs):
            self.responses = FakeResponses(api_key)

    monkeypatch.setattr("auto_research.llm.OpenAI", FakeClient)

    client = ModelClient(
        {
            "llm": {
                "provider": "openai",
                "reasoning_provider": "openai",
                "base_url": "https://api-cdn.owlai.tech/v1/responses",
                "model": "gpt-5.4",
                "timeout_seconds": 10,
                "request_retries": 2,
                "use_real_api": True,
            }
        }
    )

    result = client.generate(instructions="Return JSON.", prompt="{}")

    assert result.text == '{"ok": true}'
    assert calls == ["base-key", "first-key", "base-key"]


def test_codex_cli_provider_persists_session(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    meta_dir = project_root / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "codex_sessions.yaml").write_text("sessions: {}\n", encoding="utf-8")

    def fake_run(command, capture_output, text, cwd, timeout, env=None):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"value": 1}\n', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="session id: 123e4567-e89b-12d3-a456-426614174000\n")

    monkeypatch.setattr("auto_research.llm.subprocess.run", fake_run)

    client = ModelClient(
        {
            "llm": {
                "provider": "codex_cli",
                "model": "gpt-5.4",
                "timeout_seconds": 10,
                "codex_cli": {"use_resume": True, "sandbox": "read-only", "approval_policy": "never"},
            }
        },
        project_root=project_root,
    )

    payload = client.generate_json(
        instructions="Return JSON.",
        prompt="Return {\"value\": 1}",
        default={},
        agent_name="literature-agent",
    )

    sessions = (meta_dir / "codex_sessions.yaml").read_text(encoding="utf-8")
    assert payload == {"value": 1}
    assert "literature-agent" in sessions
    assert "123e4567-e89b-12d3-a456-426614174000" in sessions
