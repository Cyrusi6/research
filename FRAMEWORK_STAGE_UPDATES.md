# AutoResearch C2C 阶段改造说明

本文档说明当前 `auto-research` 框架相对最初版本，在每个阶段上做过的主要更新。重点覆盖目前已经实现和测试过的 S1-S3 全自动科研闭环，S4/S5 暂时只保留结构化接口和 gate 能力。

## 总体变化

最初框架更接近“半自动流水线”：agent 会生成一些阶段产物，orchestrator 依据简单 judge 判断是否继续，但阶段输入输出、失败恢复、artifact 审计、GPU 资源选择、S2.5 代码修改、失败回传都不够明确。

现在框架增加了四层基础设施：

1. 全局运行状态：`orchestration/state.json`
2. 阶段输入输出合同：`orchestration/stage_contracts/<stage>.json`
3. 阶段 artifact 审计：`<stage>/stage_manifest.json`
4. 可执行阶段 gate：`<stage>/gate_report.json`

这四层把“agent 说完成了”改成“产物存在、hash 可追踪、gate 可执行、状态可恢复”。

最新全局 S1 语义更新：

- S1 只负责输出高层研究方向或 idea，本身就是传给 S2 的真实输入；这个原则现在适用于普通项目和 C2C 项目。
- 程序不再为了满足旧 schema 自动生成轻量方向卡或同方向 variant。
- S1 默认使用 Codex resume evidence agent：Codex 返回合法 JSON 就采用；JSON 不合法就把错误发回同一个 resume session 修复。
- S1 不再退回普通 GPT request-fetch-judge 或确定性 fallback；多次修复仍失败则 stage blocked。
- S2 才负责把 S1 的高层方向转成具体候选 patch / experiment variants。
- 普通项目写出 `literature/evidence_requests.json`、`literature/evidence_bundle.json`、`literature/direction_decision.json`、`literature/evidence_session.json`。
- C2C 项目写出对应的 `literature/c2c/...` 证据和方向产物，并保留 C2C 专属 contract/gate。

## Orchestration 层

新增能力：

- 新增 `OrchestrationStateManager`
- 每个项目生成 `orchestration/state.json`
- 记录当前阶段、运行状态、attempt 次数、judge retry 次数、last gate、失败原因、revision loop、failure feedback 路由状态
- 每个 stage 在 state 中记录 `contract_path`
- 支持中断后判断当前阶段、最后 gate、最后失败原因和 artifact 摘要

主要产物：

```text
orchestration/state.json
meta/registry.yaml
meta/session_log.jsonl
```

当前作用：

- 作为全局恢复账本
- 记录 S1-S3 是否完成、失败或被 failure feedback 路由
- 为后续 resume / skip / incremental rerun 提供基础

## Artifact Manifest 层

新增能力：

- 所有通过 `ArtifactManager` 写出的产物都会进入 `stage_manifest.json`
- manifest 从简单 artifact 列表升级为统一 schema
- 每个 artifact 记录：
  - `path`
  - `type`
  - `sha256`
  - `size_bytes`
  - `created_by`
  - `created_at`
  - `source_paths`
  - `status`
  - `validator`
  - `metadata`

主要产物：

```text
literature/stage_manifest.json
plan/stage_manifest.json
experiment/stage_manifest.json
paper/stage_manifest.json
review/stage_manifest.json
```

当前作用：

- 支持 artifact 溯源
- 支持去重和覆盖更新
- 支持后续缓存和增量重跑
- 支持审计某个结论依赖哪些上游文件

## Stage Contract 层

新增能力：

- 新增 `StageContractManager`
- 每个阶段生成显式输入输出合同
- contract schema 当前为 `stage_contract_v2`
- 区分：
  - `required_inputs`
  - `optional_inputs`
  - `conditional_inputs`
  - `required_outputs`
  - `optional_outputs`
  - `conditional_outputs`

条件当前支持：

```text
project.mode == c2c
iteration > 1
execution.collector == c2c_small_loop
```

主要产物：

```text
orchestration/stage_contracts/S1_literature.json
orchestration/stage_contracts/S2_plan.json
orchestration/stage_contracts/S3_experiment.json
orchestration/stage_contracts/S4_writing.json
orchestration/stage_contracts/S5_review.json
```

当前作用：

- 明确每个阶段理论输入和理论输出
- 记录实际解析到的输入、输出、hash、缺失项
- 避免普通非 C2C 项目被 C2C 专属输入误判
- 为后续 preflight contract 和 skip unchanged stage 打基础

## Stage Gate 层

新增能力：

- 将原来的自然语言 judge 升级为可执行 validator
- 新增：

```text
src/auto_research/validators/
  base.py
  s1_gate.py
  s2_gate.py
  s3_gate.py
  s4_gate.py
  s5_gate.py

src/auto_research/schemas/
  idea.schema.json
  plan.schema.json
  revision_dispatch.schema.json
```

gate 返回结构化状态：

```text
PASS
NEEDS_RETRY
FAIL
```

主要产物：

```text
literature/gate_report.json
plan/gate_report.json
experiment/gate_report.json
paper/gate_report.json
review/gate_report.json
```

当前作用：

- `PASS` 才能进入下一阶段
- `NEEDS_RETRY` 走 judge retry
- `FAIL` 进入失败、blocked 或 failure feedback 逻辑
- 不再只相信 agent 自报完成

## S1: 文献 / Repo / Rebuttal / Idea 阶段

最初问题：

- S1 生成较模板化
- 文献、rebuttal、repo 证据链不够明确
- 失败经验不能稳定回传
- GPT fallback 结果容易被误认为高质量 idea

现在更新：

- C2C 模式下自动读取目标 repo snapshot
- 自动导入 `ref_paper` 和 `ref_rebuttal`
- 构建结构化材料：
  - `repo_manifest.json`
  - `repo_card.json`
  - `historical_results.json`
  - `result_ledger.csv`
  - `baseline_evidence.json`
  - `paper_cards.json`
  - `paper_chunks.jsonl`
  - `bibliography.json`
  - `rebuttal_concern_matrix.json`
  - `rebuttal_chunks.jsonl`
  - `code_cards.json`
  - `code_chunks.jsonl`
  - `retrieval_plan.json`
  - `retrieval_followup.json`
  - `negative_result_memory.json`
- 新增多 agent reasoning：
  - `literature_scout`
  - `rebuttal_analyst`
  - `method_inventor`
  - `skeptic_reviewer`
  - `systems_feasibility`
  - `experiment_designer`
  - `meta_judge`
- S1 产物强调证据、反证、结论链
- idea 必须包含实验合同字段：
  - hypothesis
  - expected files
  - verification commands
  - evidence refs
  - counterevidence refs
  - code refs
  - reviewer risk response
- failure feedback 会进入下一轮 S1
- 若 GPT agent 超时 fallback，S1 gate 会标记 `NEEDS_RETRY`，不把 fallback 当作高质量 idea

主要产物：

```text
literature/ideas.json
literature/idea_debate.json
literature/idea_debate.md
literature/negative_constraints.json
literature/c2c/repo_card.json
literature/c2c/baseline_evidence.json
literature/c2c/rebuttal_concern_matrix.json
literature/c2c/code_chunks.jsonl
literature/gate_report.json
```

当前边界：

- PDF 不会无脑全文全部塞给 GPT，而是抽取成结构化 cards/chunks
- 参考文献区单独保留到 `bibliography.json`
- 当前最强能力在 C2C 专用路径，普通 topic 路径仍是简化版

## S2: 资源感知实验计划阶段

最初问题：

- GPU 使用和 plan 不一致
- 计划更多是文本描述，不够像可执行实验合同
- 没有把失败反馈稳定纳入下一轮计划

现在更新：

- S2 读取 S1 的 ideas、baseline、negative constraints、failure feedback
- 生成 C2C 小循环实验计划
- 新增 GPU 资源选择：
  - 支持显式 GPU 列表
  - 支持 `auto`
  - 最多选择 6 张卡
  - 写入真实 `selected_gpu_ids`
  - 写入 `resource_snapshot`
- `resource_budget.peak_concurrent_gpus` 与实际训练卡数保持一致
- eval 默认复用训练 GPU
- S2 写入 failure feedback 使用情况
- S2 gate 校验：
  - `plan.yaml` 是否存在且满足最低 schema
  - 是否有 hypotheses / baselines / datasets / task graph / resource budget
  - C2C 小循环是否有 `short_loop_plan.yaml`
  - acceptance threshold 是否存在
  - GPU budget 是否和 selected GPU 数一致

主要产物：

```text
plan/plan.yaml
plan/candidate_ideas.json
plan/short_loop_plan.yaml
plan/plan_feedback.json
plan/hypotheses.md
plan/task_graph.md
plan/resource_budget.md
plan/gate_report.json
```

当前边界：

- S2 负责计划和实验合同
- S2 不直接跑训练
- 代码修改被拆到 S2.5

## S2.5: Codex 代码修改与冻结 Patch 阶段

最初问题：

- 之前代码修改范围较窄
- 修改代码和训练阶段耦合，容易不确定
- 没有冻结 patch，难以复现
- Codex 生成失败或 noop 时缺少产品化状态

现在更新：

- 新增 `CodePatchAgent`
- 由 S2 调用，在 S2 和 S3 之间执行
- Codex 作为代码生成后端，框架负责边界、校验、归档
- Codex 在临时 C2C repo 副本中工作，不污染主 snapshot
- 动态白名单允许更大范围：

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

- 默认禁止：

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

- patch 使用冻结 schema：
  - `replace_file`
  - `add_file`
  - 默认不允许 delete
  - replace 必须匹配 `old_sha256`
- patch 冻结前运行：
  - path policy
  - py_compile
  - targeted tests
  - executable change 检查
- 支持 Codex sandbox fallback
- 噪声文件不会进入 patch：
  - `.pytest_cache`
  - `__pycache__`
  - `.coverage`
  - `htmlcov`

主要产物：

```text
plan/code_patches/patch_manifest.json
plan/code_patches/<idea_id>/patch.json
plan/code_patches/<idea_id>/patch.diff
plan/code_patches/<idea_id>/rationale.md
plan/code_patches/<idea_id>/validation.json
plan/code_patches/<idea_id>/implementation_contract.json
plan/code_patches/<idea_id>/codex_prompt.md
```

当前边界：

- S2.5 只生成和验证 patch
- 不跑完整训练
- 不允许修改数据、模型、checkpoint、历史结果
- S3 只应用冻结 patch，不再临时调用 LLM 改代码

## S3: 实验训练 / 评测 / Recovery 阶段

最初问题：

- 训练前环境问题容易中途暴露
- model symlink / HF cache / dataset cache / checkpoint 路径错误需要人工 recovery
- train 失败后 eval/metrics 续跑不够产品化
- 泛化的 `failed_no_metrics` 信息不足
- S3 期间可能不够 deterministic

现在更新：

- S3 训练执行阶段禁止 LLM 改代码或改配置
- S3 只读取 S2.5 冻结 patch
- 每个 candidate：
  - snapshot 当前 repo state
  - apply frozen patch
  - 保存 patched code snapshot
  - preflight
  - train
  - eval
  - collect metrics
  - restore repo state
- 新增 preflight：
  - env python 可执行
  - model path 存在
  - symlink resolve 正确
  - config/tokenizer 文件存在
  - checkpoint output parent 可写
  - eval checkpoint path 指向训练输出
  - dataset cache 可定位或离线可加载
- 新增 recovery/retry：
  - broken model symlink 自动从 HF cache snapshot 修复
  - train OOM 可重选 GPU 或降低并发重试
  - train failed 但 `checkpoints/final` 存在时跳过 train 继续 eval
  - eval 单数据集失败只重试该 dataset
  - metrics collect 最后总是执行，允许 partial metrics
- 新增 run state：

```text
local/auto_research_runs/<run_id>/run_state.json
```

- S3 输出状态更细：
  - `ok`
  - `partial`
  - `blocked`
  - `failed`
  - `not_viable`
- 训练结束后才允许 posthoc GPT 分析失败原因和下一轮建议
- failure feedback 会写回 S1/S2 可读位置

主要产物：

```text
experiment/results/main_results.json
experiment/results/ablation_results.json
experiment/results/hypothesis_verification.md
experiment/results/c2c_small_loop_results.json
experiment/results/posthoc_review.json
experiment/results/failure_analysis.md
experiment/results/failure_feedback.json
experiment/code_snapshots/<idea_id>/manifest.json
experiment/gate_report.json
```

当前边界：

- S3 acceptance 当前仍沿用规则：
  - mean 至少超过 baseline `+0.1`
  - 任一数据集回退不超过 `2.0`
- partial metrics 可以被记录，但不会通过 acceptance

## Failure Feedback 与自动迭代

新增能力：

- S3 未通过时，不一定直接停死
- 若开启 failure feedback，失败信息会路由回 S1
- iteration 增加
- S1/S2 下一轮读取失败记忆，避免重复 idea

主要产物：

```text
meta/negative_memory.jsonl
meta/iteration_trace.jsonl
experiment/results/failure_feedback.json
literature/feedback/failed_ideas_round_<N>.json
plan/plan_feedback.json
```

当前作用：

- 把失败 idea、metrics、dataset regression、stderr/posthoc reason 转成下一轮约束
- 使 S1/S2 能基于失败继续生成新 idea，而不是重复失败方案

## GPT / LLM 使用边界

现在边界是：

- S1 可以使用 GPT 做深度文献、rebuttal、repo、idea reasoning
- S2 可以使用 GPT 做 plan reasoning 和实验合同生成
- S2.5 可以使用 Codex 作为代码 patch 生成后端
- S3 训练/评测执行期间禁止 GPT 改代码或改配置
- S3 结束后可以使用 GPT 做 posthoc 分析

配置上支持：

```text
OPENAI_API_KEY
OPENAI_API_KEY_1
Responses API endpoint
model + reasoning effort
long timeout
key fallback
```

当前原则：

- 研究推理可以非确定
- 训练执行必须 deterministic
- posthoc GPT 只能分析原因和建议下一轮，不能改变 acceptance

## 当前测试状态

当前测试覆盖包括：

- artifact manifest schema
- orchestration state
- stage contract v2
- stage gate validator
- S1/S2/S3 pipeline
- C2C S1-S3 mock loop
- GPU selector
- preflight recovery
- frozen patch apply/restore
- S2.5 Codex patch agent
- deterministic S3 不调用 LLM
- failure feedback

最近一次验证：

```text
55 passed
```

## 当前还没有完全完成的部分

以下不是缺陷，而是下一阶段可以继续完善的方向：

- 还没有真正启用 “contract preflight 阻断阶段启动”
- 还没有实现基于 `input_hash/output_hash` 的自动 skip unchanged stage
- S4/S5 只是保留结构化写作和 review gate，论文写作闭环暂时不是本轮重点
- S1 的质量高度依赖 GPT 是否完成；fallback 已经会被 gate 标记为 `NEEDS_RETRY`
- C2C 路径最完整，普通非 C2C topic 路径仍较轻量

## 当前完整流程

现在框架的目标流程是：

```text
S1 文献 / Repo / Rebuttal / Idea reasoning
  -> S1 gate
  -> S2 资源感知实验计划
  -> S2 gate
  -> S2.5 Codex patch 生成、验证、冻结
  -> S3 deterministic apply patch + preflight + train + eval + metrics
  -> S3 gate
  -> 若失败，failure feedback 回传到 S1/S2
  -> 若通过，进入后续写作阶段
```

核心变化是：现在每一步都有明确输入、明确输出、artifact hash、可执行 gate 和可恢复状态。框架已经从“半自动 agent 产物生成器”升级成“可审计、可恢复、可迭代的 S1-S3 自动科研工作流”。

## 2026-05-24 C2C 五轮真实跑后审查

真实 C2C S1-S3 跑到 `max_iterations=5` 后阻塞，最佳候选为 `projector_ambiguous_span_weight_calibration_v1`：

- baseline mean: 50.82
- best mean: 49.7374
- delta: -1.0826
- 主要失败模式: mmlu-redux 回退 3.249，整体仍低于 baseline

审查结论：

- 框架执行链路可用：S1/S2/S2.5/S3 能完成真实 patch、训练、三项评测、baseline 对比和失败回流。
- 主要不足不是执行器，而是 idea 质量与失败归因强度：多轮候选仍集中在 confidence/fallback/gate 微调，未能形成稳定超过 v2.2 token-MLP baseline 的高杠杆机制。
- 结果审计不足：`c2c_small_loop_results.json` 只保留当前轮，跨轮趋势需要从分散文件手动拼。
- 无 GPT 或 posthoc 不可用时，失败反馈过弱，下一轮建议可能为空。
- 自动循环缺少可配置早停，可能在连续明显低于 baseline 时继续消耗算力。

本次补强：

- 新增 `experiment/results/c2c_iteration_history.json`，每轮 S3 追加记录 best candidate、delta、dataset regression、连续失败次数和历史最佳。
- S3 posthoc 在 GPT 不可用时也生成 deterministic failure feedback，不再返回空建议。
- 新增 `orchestration.failure_feedback.early_stop` 配置，可在连续多轮低于阈值时提前阻塞，默认关闭。

下一阶段优化重点：

- S1/S2 应强制候选保留 baseline 主路径，只允许一个局部可解释增益，减少“换机制式”低质量探索。
- 加入 cheap proxy / replay 预筛，避免每个 weak idea 都走完整 6 卡训练。
- 把 dataset-specific regression，尤其 mmlu-redux，作为 idea 生成和 S2.5 patch contract 的硬约束。

## 2026-05-24 Cheap Proxy 与失败归因增强

本次补强两条执行链路：

- 新增 `c2c.small_loop.proxy_screen`。默认开启；在 S3 preflight 通过后、完整训练前执行两级 cheap proxy gate：先静态 hard gate 检查 executable change、evaluator/test-only 改动和 patch 风险文件，再运行小样本 replay / validation。proxy replay 使用 128 train samples、三数据集各 64 eval limit，并缓存同子集 baseline 到 `experiment/results/c2c_proxy_baseline.json`。候选必须和同一 proxy baseline 做 paired comparison；明显低于阈值或单数据集回退过大时标记为 `proxy_rejected`，不会进入完整 train/eval。
- S3 候选结果新增 `failure_attribution`，记录 primary failure、dragging datasets、sample type family、mixed gain pattern、patch risk file/label，以及 proxy_screen 证据。`failure_feedback.json`、negative memory、feedback bundle 和下一轮 debate constraints 都会携带这些字段。

预期效果：

- 明显高风险或 proxy 低于 baseline 的候选可在完整 6 卡训练前淘汰。
- 下一轮 S1/S2 不再只看到 mean/delta，而是能引用 “mmlu-redux 拖后腿”、“openbookqa 提升但 mmlu 崩”、“evaluator/projector 改动风险” 这类证据。

## 2026-05-24 C2C 创新方向重构

本次把 C2C idea 生成从局部调参转向机制级创新：

- 默认候选池替换为 `utility_predicted_cache_routing`、`counterfactual_cache_dropout_objective`、`semantic_span_graph_alignment` 等机制候选；失败恢复池替换为 verifier/controller/latent bridge memory 方向。
- 新增 C2C novelty gate。每个候选必须包含 `mechanism_type`、`paper_claim`、`why_baseline_fails`、`expected_signature`、`ablation_plan`；纯 `top-k`、confidence floor、fallback、threshold 调参会被判为 `reject`。
- S1 debate prompt 明确禁止局部阈值调参；normalize 阶段会过滤低 novelty idea，并只允许已知 profile 继承机制字段，避免未知低质量 idea 被 fallback 字段“洗白”。
- S2 gate 新增 `c2c_mechanism_novelty_gate`，selected idea 如果缺少机制证据或看起来只是局部 tuning，会返回 `NEEDS_RETRY`。
- S2.5 implementation contract 新增 `mechanism_contract`，要求 Codex 实现机制级改动、暴露 ablation switch，并保留轻量 mechanism stats，不能只改配置参数。

验证：

```text
uv run pytest -q
73 passed
```

## 2026-05-24 Implementation Scope Gate

本次补强 S2.5 对“大创新 idea”的承接能力，避免创新机制因为需要新文件或较大改动而被简单判为编码不行：

- C2C idea 新增 `implementation_scope`，分为 `bounded`、`medium`、`large`。
- `medium` idea 可以新增 `rosetta/model/*.py` 等 helper module，但必须声明 `integration_points` 和 `smoke_tests`。
- `large` idea 不直接要求一次性重写训练/评测/架构，必须提供 `decomposition_plan` 和 `mvp_slice`；S2.5 只实现第一段可执行 MVP。
- S2 gate 新增 `c2c_implementation_scope_gate`。如果 selected idea 是 medium/large 但缺少接入点、smoke test、拆解计划，会返回 `NEEDS_RETRY`。
- S2.5 `implementation_contract` 新增 `implementation_scope`，Codex prompt 会按 scope 调整 patch 策略：bounded 原地改，medium 允许列出的新模块，large 只做首个 MVP slice。

验证：

```text
uv run pytest -q
75 passed
```

## 2026-05-24 P0 自动 Ablation 执行

本次补齐 S3 的机制可证伪闭环：

- 候选完整 eval 后，S3 会读取 `experiment_contract.ablation_switch` / `ablation_plan.switch`，自动生成 `ablation_disabled/eval_*.yaml`。
- disabled eval 复用同一个 candidate checkpoint，只在 `model.rosetta_config` 中打开 ablation switch，并把输出写到 `local/auto_research_runs/<run_id>/ablation_disabled/results/<dataset>`，避免污染 enabled results。
- `main_results.json` 中每个候选新增 `ablation` 字段，包含 disabled metrics、enabled-minus-disabled mean、逐数据集差值、命令尝试状态和 ablation 配置路径。
- `experiment/results/ablation_results.json` 从占位文件升级为真实汇总，记录 best candidate 的机制支持证据。
- `hypothesis_verification.md` 的 H2 不再固定 pending，而是基于 automatic ablation 结果给出 supported / not supported / inconclusive。

验证：

```text
uv run pytest -q tests/test_c2c.py tests/test_validators.py
56 passed
```

## 2026-05-26 S2.5 Contract-Aware Repair

本次针对真实 P0 运行暴露的问题补强 S2.5：

- 真实运行 `c2c_p0_ablation_live_20260525_000444` 跑到第 5 轮时失败在 S2：`S2.5 patch manifest status is no_valid_patch`。
- 原有 repair 只覆盖 py_compile / pytest validation failure；对 `config_activation_missing`、`blocked_no_executable_change`、backend failure、unauthorized patch delta 这类 contract failure，不能把失败证据立即回喂给 Codex。
- `CodePatchAgent` 现在会在同一候选内执行 contract-aware repair retry：
  - backend 失败或无可执行 diff：注入 `contract_failure_feedback`
  - config 参数未激活：注入 `validation_failure_feedback.activation_check`
  - 修复后仍必须重新生成 frozen patch，并通过原有 validation、activation check、S2 gate
- `patch_manifest.json` 新增 `candidate_count`、`valid_patch_count`、`failed_patch_count`，并保留 `patches` alias，方便 gate 和人工审查直接看有效 patch 数。

验证：

```text
uv run pytest -q
81 passed
```

## 2026-05-27 S2.5 Best-of-N + Repairable Proxy Routing

本次把 S2.5/S3 从“单次失败丢 idea”改成“先修 patch，再决定是否换 idea”：

- S2.5 默认每个 idea 最多生成 2 个 patch variant，第一版成功时路径不变；第一版失败时第二版会拿到上一版 validation / proxy risk 失败证据，并要求生成 materially different implementation。
- S2.5 repair loop 新增 proxy-risk contract check：`patch_too_broad`、`proxy_risk_repair_required`、`evaluation_code_changed`、test-only patch 会在进入 S3 前触发 contract-aware repair。
- S3 cheap proxy 新增 `repairable_proxy_risk` 分层。evaluator/test-only/过宽静态风险，以及接近阈值的小样本 proxy 失败，会标为 `proxy_repairable`，不进入完整训练，也不立刻丢弃 idea。
- Orchestrator 对 `repairable_proxy_risk` 优先回流 `S2_plan`，同一 iteration 内保留 idea，只重做 S2.5 patch；每轮默认最多回流一次，修不动再走原来的 S1 failure feedback。
- failure memory / debate failed decisions 识别 `proxy_repairable`，下一轮或修复轮能引用具体 proxy_screen、patch_risk、repair_hint。

验证：

```text
uv run pytest -q
85 passed
python -m py_compile src/auto_research/code_patch.py src/auto_research/agents/experiment.py src/auto_research/orchestrator.py src/auto_research/c2c.py src/auto_research/failure_log.py src/auto_research/agents/debate.py
```

## 2026-05-28 Two-Round Live Audit + Proxy/Ablation Hardening

真实两轮 C2C 全流程项目：`workspace/c2c_two_round_live_20260527_1`。

观测结果：

- 两轮均完整跑过 S1/S2/S2.5/S3，并自动停止在 `max_iterations=2`。
- S2.5 best-of-N 生效：两轮都能生成可执行机制级 patch，第一轮还出现了 variant/repair 挽救候选的情况。
- S3 cheap proxy 生效：高风险候选在完整 6 卡训练前被拦截。
- 两轮都没有超过真实 full baseline `v2.2_token_mlp_entropy050 mean=50.82`：
  - 第 1 轮 best=`counterfactual_cache_dropout_objective`，mean=48.4721，delta=-2.3479。
  - 第 2 轮 best=`verifier_guided_cache_acceptance`，mean=48.4721，delta=-2.3479。
- 关键失败分布：
  - 两个候选在 proxy 训练阶段以运行异常失败：dtype mismatch、dataset schema/list scalar mismatch。
  - 进入 full S3 的候选 enabled/disabled ablation 指标完全一致，`enabled_minus_disabled_mean=0.0`，说明机制开关没有实际改变评测路径。
  - proxy 子集允许 “零提升/软警告” 候选进入 full train，造成一次完整训练浪费。

本次补强：

- `repair_soft_proxy_fail` 默认开启。cheap proxy 只要没有正向证据、触发 soft threshold，就标为 `repairable_proxy_risk` 回 S2.5，而不是放行 full train。
- proxy command failure 改为可修复风险，记录 `command_failure.category` 和具体修复建议：
  - `dtype_mismatch`
  - `schema_shape_mismatch`
  - `resource_oom`
  - `runtime_exception`
- C2C acceptance 默认要求 ablation support。候选不仅要超过 full baseline，还要证明关闭 `ablation_switch` 会降低 enabled 指标；否则 reason=`mechanism ablation support not met`。
- failure attribution 新增 `ablation_evidence`，当 enabled/disabled 完全一致时 primary failure=`ablation_no_effect`，并写入 failure analysis / feedback / negative memory。

验证：

```text
python -m py_compile src/auto_research/agents/experiment.py src/auto_research/c2c.py src/auto_research/failure_log.py
uv run pytest -q tests/test_c2c.py tests/test_pipeline.py tests/test_validators.py
69 passed
uv run pytest -q
89 passed
```

## 2026-05-28 S1/S2 Coverage-Control Constraints

针对本轮真实运行里第三个候选触碰 evaluator、以及候选可能靠叠加硬 gate 降低 transfer coverage 取巧的问题，本次补强 S1/S2：

- S1 idea novelty gate 禁止把创新点退化为“再叠一层硬 accept/reject gate”。如果 idea 涉及 gate，必须同时给出 `coverage_diagnostics` 和 `matched_coverage_ablation`。
- 默认 C2C 机制候选现在都会携带 coverage 诊断字段，包括 candidate/control transfer coverage、reject/accept reason histogram、per-dataset coverage、matched coverage delta。
- S2 plan acceptance criteria 明确不接受“只靠降低 transfer coverage 得分”的候选，并要求 matched-coverage control ablation。
- S2 gate 新增 coverage-control 检查：计划必须声明 coverage diagnostics required，并在 ablation matrix 中包含 matched transfer coverage control。
- S2.5 patch contract 要求机制实现暴露 coverage diagnostics 和 matched-coverage ablation path，且不得通过修改 `script/evaluation/*` 伪造指标。
- evaluator-touched repair 增强：在 repair 前机械恢复 evaluator 文件，并把 `script/evaluation/` 写入 `forbidden_repair_files`；如果第三个候选继续污染 evaluator，会保留为明确失败样本而不是误放行。

验证：

```text
python -m py_compile src/auto_research/c2c.py src/auto_research/code_patch.py src/auto_research/agents/plan.py src/auto_research/agents/debate.py src/auto_research/validators/s2_gate.py
uv run pytest -q tests/test_c2c.py -k 'novelty_report or mechanism_contract or evaluator_proxy_risk or recontaminates or s2_gate_requires_coverage_controls'
6 passed, 61 deselected
uv run pytest -q tests/test_c2c.py
67 passed
uv run pytest -q tests/test_stage_contracts.py tests/test_pipeline.py
8 passed
```

## 2026-06-01 Effect-First Same-Direction S2/S3 Loop

本次把 C2C 失败反馈从“一次 cheap proxy hard reject 就回 S1”改成“S1 定基调，S2/S3 在同一机制方向内局部迭代”：

- S1 仍负责选择机制方向和方法层约束；单次 cheap proxy 失败不再触发 S1 重写。
- C2C 默认开启 `route_proxy_rejected_to_s2`，同一 S1 iteration 内最多记录 `max_same_direction_proxy_failures=5` 次 proxy hard reject。
- 前 4 次 proxy hard reject 会写入 `plan/performance_feedback.json`，然后 invalidate `S2_plan`，让 S2.5 基于同一方向做 patch 修复或同方向 variant。
- 第 5 次 hard reject 才升级为方向级失败，回到 S1 吸收方法层失败证据并重新定方向。
- performance feedback 记录 proxy mean delta、拖后腿数据集、是否全数据集崩、patch risk 文件/标签、changed files、runtime smoke / validation 状态，供 S2/S2.5 使用。
- S3 现在把 S2.5 patch validation artifact 一起带入 `patch_result`，避免回修时 runtime smoke / validation 证据丢失。

验证：

```text
python -m py_compile src/auto_research/orchestrator.py src/auto_research/orchestration_state.py src/auto_research/agents/experiment.py src/auto_research/c2c.py src/auto_research/agents/plan.py src/auto_research/stage_contracts.py
uv run pytest -q tests/test_pipeline.py tests/test_workspace.py tests/test_c2c.py -k 'pipeline or workspace or failure_feedback or proxy_rejected or repairable_proxy or pytest_timeout or runtime_smoke or validation_repair'
14 passed, 85 deselected
```
