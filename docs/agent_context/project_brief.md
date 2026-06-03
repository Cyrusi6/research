# AutoResearch C2C 项目上下文

本文档记录用户在之前对话中提供的项目信息、研究目标、环境路径和关键设计决策，供后续 agent 快速理解项目背景。不要把本文档当作实现细节的唯一来源；实现状态以源码、测试和 workspace artifact 为准。

## 项目目标
构建一个全自动 AI 科研工作流：

```text
已有同领域代码 repo + 论文 + rebuttal
-> 自动读取 repo、文献、审稿意见
-> 多 agent 调研、讨论、产生 idea
-> 自动改代码、跑实验、评测
-> 失败后自愈和反馈
-> 循环迭代直到超过 baseline
-> 后续扩展到论文写作
```

当前重点是 S1-S3：

```text
S1 文献 / Repo / Rebuttal / Idea
S2 实验计划
S2.5 Codex 代码修改
S3 训练 / 评测 / recovery / failure feedback
```

论文写作 S4/S5 暂时不是重点。

## 研究方向

目标科研方向：

```text
cross-tokenizer cache communication
跨 tokenizer 的 cache communication
```

目标是围绕 C2C 做出可发表顶会级别的改进。

## 核心路径

```text
/home/lijunsi/projects/C2C
/home/lijunsi/projects/ref_paper
/home/lijunsi/projects/ref_rebuttal
```

说明：

- `C2C` 是目标 AI repo，可作为同领域已有项目代码和论文实现。
- `ref_paper` 是相关领域论文文件夹，包含 PDF / md。
- `ref_rebuttal` 是 OpenReview 审稿意见文件夹。
- C2C repo 包含顶会论文源代码和用户之前做过的多版改进。
- 之前跑过的数据不用全部复现。

当前 auto-research 项目路径：

```text
/home/lijunsi/paper_writting/auto-research
```

重要 workspace：

```text
/home/lijunsi/paper_writting/auto-research/workspace/c2c_auto_20260518_162715
```

## C2C 运行环境

C2C 真实训练 / 评测环境：

```text
/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python
```

用途：

- C2C GPU 训练
- C2C eval
- C2C preflight
- S2.5 patch validation 中的 C2C targeted tests

区分：

```text
auto-research 框架自身测试:
  /usr/bin/pytest -q

C2C 实验执行:
  /home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python
```

## GPU 要求

- S2 根据服务器实时 GPU 资源选择训练卡。
- 如果 `gpu_ids` 是显式列表，优先使用显式列表。
- 如果 `gpu_ids: auto`，按空闲显存和利用率选择。
- 最多选择 6 张 GPU。
- `plan.yaml` 必须记录真实 `selected_gpu_ids`、`resource_snapshot` 和 `gpu_policy`。
- `resource_budget.peak_concurrent_gpus` 必须等于实际训练 GPU 数。
- 避免出现“计划 1 卡，实际 4 卡”的不一致。

## HuggingFace / 模型缓存

```text
/home/lijunsi/.cache/huggingface/
```

因此 C2C preflight / recovery 需要支持：

- model symlink 检查
- broken symlink 自动从 HF cache snapshot 修复
- tokenizer / config 离线加载检查
- model path 错误时给明确 blocked reason

## OpenAI / GPT 配置

用户希望 S1/S2 使用真实 GPT API。

用户提供 endpoint：

```text
https://api-cdn.owlai.tech/v1/responses
```

模型和推理强度：

```text
gpt-5.4
reasoning effort: XHigh
```

用户说明一次请求可能很慢：

```text
约 480s - 595s
```

因此需要：

- 长 timeout
- S1 多 agent 不要过早 fallback
- 支持 `OPENAI_API_KEY` 和 `OPENAI_API_KEY_1` fallback
- 一个 key 无额度时自动换下一个

## S1 质量要求

用户要求 S1 不只是给出可执行配置，而要像真正研究决策：

```text
证据 -> 反证 -> 结论链
```

用户关注：

- 可分析的结构化材料有哪些
- 是否会损失信息
- PDF 为什么不能全看
- 是否需要去掉参考文献
- S1 能不能看代码
- idea 生成质量如何

当前 S1 应该使用：

- repo cards / code cards / code chunks
- paper cards / paper chunks
- bibliography 单独保留
- rebuttal concern matrix / rebuttal chunks
- negative result memory
- idea debate
- negative constraints

## S1 多 Agent 角色

用户指定 S1 多 agent：

```text
literature_scout
rebuttal_analyst
method_inventor
skeptic_reviewer
systems_feasibility
experiment_designer
meta_judge
```

每个 agent 输出结构化 JSON，至少包括：

```text
claims
evidence_refs
reviewer_concerns
risks
missing_evidence
proposed_ideas
kill_criteria
score
```

S1 两轮讨论：

1. 第一轮独立分析
2. 第二轮读取其他 agent 反驳后 revise
3. `meta_judge` 选择 3-5 个候选 idea

## S1 Timeout / 上下文策略

用户关注过 timeout fallback 问题，并希望兼顾质量和性能。

当前策略：

- 适度提高 timeout
- 优化每个 agent 的 `paper_chunks` / `code_chunks`
- `meta_judge` 不读完整 transcript，而读压缩 summaries
- fallback 保留，但明确标记 GPT 未完成
- fallback 不作为高质量 idea 直接放行
- S1 gate 会把 GPT fallback 标为 `NEEDS_RETRY`

## S2 要求

S2 需要：

- 根据 S1 idea 生成实验计划
- 自动选择 GPU
- 写入真实资源策略
- 读取 failure feedback
- 生成可执行实验合同

重要产物：

```text
plan/plan.yaml
plan/candidate_ideas.json
plan/short_loop_plan.yaml
plan/plan_feedback.json
plan/resource_budget.md
```

## S2.5 Codex 代码修改

用户明确选择：

```text
框架负责流程、边界、验证、冻结、归档；
Codex 作为 S2.5 的代码生成后端。
```

流程：

```text
S1 idea
-> S2 plan
-> S2.5 CodexCodePatchAgent 生成并验证 patch
-> S3 deterministic apply patch + train/eval
```

要求：

- Codex 在临时 C2C repo 副本中工作。
- 不污染主 snapshot。
- 产出 frozen patch。
- validation 通过后才进入 S3。
- 支持 Codex sandbox fallback。
- 忽略 `.pytest_cache`、`__pycache__`、`.coverage`、`htmlcov` 等噪声。

Patch schema：

```text
replace_file
add_file
默认不允许 delete_file
replace_file 必须 old_sha256 匹配
```

允许编辑范围：

```text
rosetta/**/*.py
script/**/*.py
recipe/**/*.json
recipe/**/*.yaml
recipe/**/*.yml
test/**/*.py
tests/**/*.py
pyproject.toml
requirements*.txt
```

禁止范围：

```text
.git/
wandb/
__pycache__/
local/checkpoints/
local/snapshots/
local/final_results/
data/
datasets/
models/
*.pt
*.pth
*.safetensors
*.bin
*.ckpt
*.parquet
*.arrow
```

## S3 要求

S3 需要：

- 训练前 preflight
- 模型 symlink 检查
- tokenizer 检查
- dataset cache 检查
- checkpoint output path 检查
- train failed 后自动 retry
- 成功后自动继续 eval 和 collect metrics
- eval 单数据集失败不丢弃其他结果
- metrics collect 最后总是执行
- partial metrics 可记录但不通过 acceptance

S3 状态需要区分：

```text
ok
partial
blocked
failed
not_viable
```

S3 执行边界：

- 训练 / 评测期间禁止 GPT 改代码或改配置。
- S3 只读取 S2.5 frozen patch。
- 训练结束后才允许 GPT 做 posthoc analysis。
- posthoc GPT 只能分析原因和建议下一轮，不能改变 acceptance。

## Acceptance 规则

用户指定当前 C2C acceptance：

```text
mean 至少超过 baseline +0.1
任一数据集回退不超过 2.0
```

## Failure Feedback / 自动迭代

用户要求：

- S3 acceptance false 或 not_viable 不直接死停。
- 写 failure feedback。
- candidate、metrics、dataset regressions、stderr/posthoc reason 都进入反馈。
- avoid-repeat rules 回传到 S1。
- `iteration += 1`
- invalidate 回 S1。
- 达到 `max_iterations` 或连续无新 idea 才 blocked。

相关产物：

```text
meta/negative_memory.jsonl
meta/iteration_trace.jsonl
experiment/results/failure_feedback.json
literature/feedback/failed_ideas_round_<N>.json
plan/plan_feedback.json
```

## 全局状态 / Manifest / Gate / Contract

用户要求并已纳入框架设计：

### 全局状态

```text
orchestration/state.json
```

记录：

- project_id
- 当前阶段
- 每阶段状态
- attempt 次数
- 最后 gate 结果
- 失败原因
- artifact hash
- revision loop 次数

### Artifact Manifest

每个 `stage_manifest.json` 条目至少包含：

```json
{
  "path": "...",
  "type": "...",
  "sha256": "...",
  "created_by": "...",
  "created_at": "...",
  "source_paths": [],
  "status": "committed",
  "validator": "..."
}
```

用途：

- 去重
- 溯源
- 缓存
- 增量重跑
- 结果审计

### Executable Stage Gates

用户要求 gate 是可执行 validator，而非自然语言：

```text
validators/
  s1_gate.py
  s2_gate.py
  s3_gate.py
  s4_gate.py
  s5_gate.py
```

gate 返回：

```text
PASS
FAIL
NEEDS_RETRY
```

### Stage Contracts

用户要求每阶段有明确输入输出：

```text
orchestration/stage_contracts/
  S1_literature.json
  S2_plan.json
  S3_experiment.json
  S4_writing.json
  S5_review.json
```

当前 contract 需要区分：

```text
required_inputs
optional_inputs
conditional_inputs
required_outputs
optional_outputs
conditional_outputs
```

条件包括：

```text
project.mode == c2c
iteration > 1
execution.collector == c2c_small_loop
```

目的：

- 避免普通非 C2C 项目被 C2C 文件误判缺输入。
- 为后续 contract preflight、skip unchanged stage 和增量重跑打基础。

## 监控和执行习惯

用户希望：

- 长训练时低频监控，不要一直刷。
- 估算时间去检查即可。
- 全自动执行，除非遇到不能解决的问题。
- 每轮完成后分析每阶段产出是否符合预期。

## 文档

已生成阶段改造说明：

```text
FRAMEWORK_STAGE_UPDATES.md
```

已更新 agent 指令文件：

```text
AGENTS.md
```

本文件是面向后续 agent 的项目上下文摘要：

```text
docs/agent_context/project_brief.md
```

## 当前关键原则

- S1/S2 可以使用 GPT 做研究推理。
- S2.5 可以使用 Codex 做代码生成。
- S3 训练 / 评测必须 deterministic。
- S3 不允许临时调用 LLM 改代码或改配置。
- 所有阶段必须有 artifact manifest、gate report、stage contract。
- C2C 路径是当前最完整路径。
- 普通非 C2C 路径仍较轻量，不应被 C2C contract 误伤。
## Git Workflow
git add .
git commit -m "Update docs"
git push