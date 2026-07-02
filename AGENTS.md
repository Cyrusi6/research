# AGENTS.md
auto-research is a uv-managed Python CLI package requiring Python >=3.10. Fully automated AI research loop: read papers/code/rebuttals → generate ideas → patch code → run experiments → evaluate → iterate.

## Commands

| Intent | Command |
|--------|---------|
| Full tests | `uv run pytest -q` |
| Core pipeline tests | `uv run pytest -q tests/test_stage_contracts.py tests/test_pipeline.py` |
| C2C tests | `uv run pytest -q tests/test_c2c.py` |
| Syntax check | `python -m py_compile src/auto_research/<file>.py` |
| Init project | `uv run auto-research init --topic "demo" --simulate` |
| Start pipeline | `uv run auto-research start --project-id <id> --simulate` |
| Check status | `uv run auto-research status --project-id <id>` |

C2C init:

```bash
uv run auto-research init-c2c \
  --topic "cross tokenizer cache communication" \
  --target-repo /home/lijunsi/projects//C2C \
  --ref-paper /home/lijunsi/projects//ref_paper \
  --ref-rebuttal /home/lijunsi/projects/ref_rebuttal \
  --env-python /home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python \
  --project-id <id>
```

## Architecture Map

| Path | Responsibility |
|------|---------------|
| `src/auto_research/orchestrator.py` | Stage transitions, retries, failure routing |
| `src/auto_research/stage_contracts.py` | Stage input/output contracts |
| `src/auto_research/artifacts.py` | Artifact writes + `stage_manifest.json` |
| `src/auto_research/validators/` | Executable stage gates (`gate_report.json`) |
| `src/auto_research/code_patch.py` | S2.5 Codex patch generation, validation, archive |
| `workspace/` | Generated runs and artifacts — read-only unless asked |

## Documentation
- `README.md` — user-facing, concise.
- Keep design history in `FRAMEWORK_STAGE_UPDATES.md`.
- `/docs/agent_context/project_brief.md` contains the project context and user-provided requirements summarized from previous conversations.

## Commit 

不使用Pull Request, 通过 Git commit 管理变更, 使用 git commit -m "提交信息"; git push origin main去提交 
每次改进后按照之前的结构更新到`FRAMEWORK_UPDATE.md`

完成后需要commit / push，并回复以下内容，除非明确要求：
1. Commit hash 和 Push range
2. 修改文件列表与摘要，按文件或目录列出变化
3. 测试命令和结果
4. 端到端命令结果摘要，如果本任务不涉及端到端命令，写“不涉及”
