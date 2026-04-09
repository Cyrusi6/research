from pathlib import Path
from types import SimpleNamespace

from auto_research.llm import ModelClient


def test_codex_cli_provider_persists_session(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    meta_dir = project_root / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "codex_sessions.yaml").write_text("sessions: {}\n", encoding="utf-8")

    def fake_run(command, capture_output, text, cwd, timeout):
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
