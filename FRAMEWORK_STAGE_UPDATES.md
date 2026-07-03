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

## 2026-07-03 S1 Direction / S2 Variant Contract Upgrade

本次把文档里的“高层方向由 S1 产生、具体 variant 由 S2 产生”落到可执行 schema 和 gate：

- S1 主合同从旧 `literature/ideas.json` 切到 `literature/direction.json`、`literature/direction_scorecard.json`、`literature/evidence_bundle.json`、`literature/novelty_audit.json`。
- `literature/ideas.json`、`literature/direction_decision.json` 继续作为兼容镜像写出，但 S1 gate 不再用旧 idea list 的 `id/title/novelty_score/feasibility_score` 作为主通过条件。
- `direction.json` 显式记录 `direction_id`、`mechanism_axis`、`integration_point`、`control_signal`、`hypothesis`、baseline failure model、metric signature、证据/反证 refs、implementation surface refs、负记忆 refs，以及进入 S2 / 回 S1 条件。
- S2 新增 `plan/planner_decision.json`、`plan/variant_contract.json`、`plan/variant_fingerprint.json`，把原来的 `next_variant` 从 prompt 约定升级为可审计合同。
- S2 gate 现在校验 direction id 一致性、机制轴/入口/控制信号、expected files、metric signature、ablation switch/control、failure routing 条件、variant fingerprint 重复，以及 C2C allowed edit surface。
- `plan.yaml`、`plan/candidate_ideas.json`、`plan/next_variant.json` 仍保留给 S2.5/S3/报告路径兼容；新 artifact 是 source of truth，旧 artifact 是派生视图。

## 2026-07-03 C2C S1 Evidence Quality Gate

本次把 C2C S1 的“证据驱动选方向”从 prompt 约束升级为 deterministic hard gate，防止 JSON 结构合法但证据过薄的方向进入 S2：

- C2C S1 新增并写出 `literature/c2c/evidence_quality_score.json`、`literature/c2c/evidence_retrieval_trace.json`、`literature/c2c/direction_fingerprint.json`。
- `evidence_quality_score.json` 采用 `c2c_s1_evidence_quality_v1`，由现有 `evidence_bundle`、`resolve_s1_evidence_refs()` 报告、direction contract、novelty audit 和 direction fingerprint 计算；除 novelty score 外不依赖 GPT 判断。
- 硬通过规则固定为：`unresolved_ref_count == 0`、paper coverage >= 2、code coverage >= 2、counterevidence count >= 1、implementation surface coverage >= 0.6、novelty score >= 0.60。
- Codex evidence-agent 路径在 `_run_s1_codex_evidence_agent(mode="c2c")` 内先执行质量门；失败会作为 JSON repair reason 反馈给同一个 resume loop，而不是等 stage gate 才失败。
- legacy C2C debate 路径也写同样的 deterministic artifact；S1 gate 在检测到 C2C 项目时要求并 schema-check 三个 artifact，并用 `s1_evidence_quality_gate` 返回 `NEEDS_RETRY`。
- `stage_contracts.py` 将三个新 artifact 作为 C2C S1 conditional outputs，方便 manifest、orchestrator 和报告路径追踪。

## 2026-06-26 External S2.5 Worktree Storage

真实 C2C 迭代积累后发现 `workspace/<project_id>/plan/code_worktrees/<idea>/vN/repo` 会把完整 Git worktree 放进当前 VS Code workspace。打开项目根目录时 VS Code 会递归 watch 历史 artifact 和 worktree repo，当前本地 `workspace/` 已达到 109G、15 万以上文件，触发 20 万量级 watcher。

- 新增 `code_patch.worktree_storage_root`，也可用 `AUTO_RESEARCH_WORKTREE_ROOT` 覆盖；未配置时默认使用用户缓存目录 `~/.cache/auto-research/code-worktrees`。
- S2.5 session、events、metadata 仍写在 `workspace/<project_id>/plan/code_worktrees/...`，保持 artifact 审计和 resume 入口稳定。
- repo 路径选择顺序为 metadata 中已有 repo、legacy `workspace/.../repo`、外置缓存新 repo，避免打断已经跑到一半的 persistent Codex session。
- `.vscode/settings.json` 排除 `workspace/**` watcher/search/Pylance 扫描，立即降低打开项目根目录时的文件监听压力。

## 2026-06-21 Shared Method Memory Purity

本次收紧长期共享记忆池，目标是保证 `.auto-research/method_failure_memory.jsonl` 只保存可跨项目复用的方法级失败经验，不保存实现错误、资源错误、测试残留或短期反馈噪声。

- 默认全局共享池只允许 `workspace/<project_id>` 下的真实项目写入；单元测试或临时目录必须显式配置自定义 `shared_method_memory.path`。
- `append_shared_c2c_method_failure()` 只持久化满足以下至少一类证据的条目：
  - 候选方法的 proxy/full 指标；
  - 数据集回退或拖后腿数据集；
  - proxy/full false-positive 校准；
  - ablation 证据；
  - 明确的 method-level avoid-repeat 规则。
- 不再持久化：
  - implementation/resource/quota/OOM 失败；
  - direction-score-only 摘要；
  - 空 proxy calibration；
  - `patch_risk`、`feedback_items`、原始 `direction_scorecard` 大对象；
  - 没有候选结果的 repairable proxy 摘要。
- S0 的 `shared_method_memory` 入口改为 catalog-first：
  - `memory_catalog`
  - `quality_summary`
  - `high_quality_memory_ids`
  - `full_memory_access`
  - 不再把旧 `summary/feedback_items` 聚合层塞进 `negative_result_memory` 或 `evidence_brief`。
- `auto-research memory report` 支持压缩后的 `method_context` 和 proxy/full calibration 字段。
- 已清洗现有全局池：
  - 旧记录：336 条；
  - 删除 `proj_*` 测试残留：333 条；
  - 删除无候选方法结果的 repairable proxy 摘要：2 条；
  - 保留真实 C2C 方法失败：1 条；
  - 备份：`.auto-research/method_failure_memory.jsonl.bak_memory_purity_20260621`。

验证：

```text
python -m py_compile src/auto_research/method_memory.py src/auto_research/agents/intake.py src/auto_research/reporting.py
uv run pytest -q tests/test_failure_log.py -k 'shared_method_memory or feedback_loader_filters_retryable_resource_pause_noise'
uv run pytest -q tests/test_reporting.py -k memory_report
uv run pytest -q tests/test_c2c.py -k 'c2c_s0_reuses_cached_static_bundle_and_restores_sidecars'
uv run auto-research memory report
rg -n "feedback_items|direction_scorecard|patch_risk|resource|oom|implementation|no_metrics|proj_" .auto-research/method_failure_memory.jsonl .auto-research/method_failure_memory.md
```

## 2026-06-21 Neutral Proxy Full-S3 Exploration

真实 S3 运行显示：候选完整跑完 cheap proxy 和 activation smoke 后，proxy 三数据集与 paired proxy baseline 完全相同，`proxy_delta=0.0`、无 dataset regression，但因为 activation metric-neutral 被路由回 S2。这个行为对 effect-first discovery 偏保守：cheap proxy 应该主要挡明显坏 patch，而不是要求小样本阶段已经有正收益。

调整：

- 默认下调 cheap proxy soft threshold：
  - `soft_proxy_mean_delta=-0.1`
  - `soft_min_proxy_score=-0.3`
- 新增 neutral-proxy full S3 策略：
  - `allow_neutral_proxy_full_s3=true`
  - `neutral_proxy_min_delta=-0.1`
  - `neutral_proxy_max_dataset_regression=0.25`
- 如果 activation smoke 是 metric-neutral，但 mechanism trace 已经 `wired`，且 cheap proxy 处于近中性范围内，则不再回 S2/S2.5 repair，而是作为 warning 进入 full S3。
- 明显负收益、单数据集明显回退、proxy output health 异常、机制未接通仍会被挡住。
- 当前项目 `c2c_s1s3_three_iter_20260614_4` 的 `meta/project_config.yaml` 同步更新，后续 resume 会使用新阈值。

验证：

```text
python -m py_compile src/auto_research/c2c.py src/auto_research/agents/experiment.py
uv run pytest -q tests/test_c2c.py -k 'zero_delta_with_patch_risk_passes_lowered_effect_first_threshold or neutral_proxy_policy_allows_small_negative_but_blocks_clear_regression or proxy_activation_smoke_allows_metric_neutral_when_wiring_trace_passed'
```

## 2026-06-11 S3 S2.5 Artifact Lock

本次修复 S3 候选执行和 S2.5 `patch_manifest.json` 不同步的问题。新的执行原则是：`plan.yaml` 继续提供实验协议上下文，例如 baseline、datasets、GPU 和 acceptance rule；但 S3 具体执行哪个 candidate / patch，必须以 S2.5 实际产物为唯一真相。注意：这里的 GPU 上下文职责后来已被 `2026-06-15 S2 GPU Planning Removed` 覆盖，当前 S2 不再生成 GPU 资源决策。

- S3 启动时读取 `plan/code_patches/patch_manifest.json`。
- 如果存在 `selected_candidate_id`，S3 只执行这个 selected candidate，跳过 `plan.yaml` / `candidate_ideas` 里的其他候选。
- 如果 selected candidate 已不在 `plan.yaml.candidate_ideas` 里，S3 会从 `patch_manifest.selected_patch` / `patch_manifest.candidates` 恢复最小 candidate，而不是回退执行旧 plan candidate。
- S3 写出 `experiment/results/s3_candidate_selection.json`，记录：
  - `selected_candidate_id`
  - executed / skipped candidate ids
  - `patch_manifest` sha256
  - selected `patch.json` sha256
  - selected `implementation_contract.json` sha256
- S3 gate 新增锁校验：
  - `main_results.candidate_results[*].id` 必须和 `selected_candidate_id` 一致；
  - S3 selection 记录的 manifest / patch / contract sha256 必须和当前文件一致。
- S2.5 patch-only repair 现在会同步重写 `plan.yaml.selected_idea` 和 `plan.yaml.candidate_ideas`，避免 repair 后 `candidate_ideas.json` / `patch_manifest.json` 已更新但 `plan.yaml` 还停留在旧候选。

验证：

```text
python -m py_compile src/auto_research/agents/experiment.py src/auto_research/agents/plan.py src/auto_research/validators/s3_gate.py
uv run pytest -q tests/test_c2c.py -k "s2_5_patch_only_repair or c2c_small_loop_locks_to_selected_patch_manifest_candidate"
uv run pytest -q tests/test_validators.py -k "s3 or S3 or candidate_selection"
```

## 2026-06-11 Full S3 Readiness Hard Gate

本次把 `full_s3_readiness_report.json` 从监控 artifact 升级为 full train 前硬 gate，避免 report 已经显示 `full_train_allowed=false/status=not_ready` 时仍然启动 6 卡 full S3。

- S3 在 cheap proxy 和 activation smoke 后生成 `full_s3_readiness_report.json`。
- 如果 `full_train_allowed=true`，继续执行 full train / eval / ablation。
- 如果 `full_train_allowed=false`：
  - 不再调用 full `train`；
  - 不再调用 full `eval_*` 或 `ablation_eval_*`；
  - `command_status=proxy_repairable`；
  - `proxy_screen.status=repairable_proxy_risk`；
  - `proxy_screen.repair_mode=full_s3_readiness_repair`；
  - 写入 `recovery_actions[].action=block_full_train_until_readiness`；
  - `proxy_effect_repair_contract.source=full_s3_readiness`，并携带 readiness blockers / warnings。
- failure attribution 新增优先归因 `full_s3_readiness_not_ready`，保证 S2.5 repair 能看到真正的 full-train blocker，而不是误判成普通 method failure。

验证：

```text
python -m py_compile src/auto_research/agents/experiment.py
uv run pytest -q tests/test_c2c.py -k "full_s3_readiness_blocks_train or proxy_activation_smoke_blocks_no_effect or proxy_activation_smoke_allows_metric_neutral or proxy_activation_smoke_passes_metric_neutral_prediction_change or replay_proxy_runs_paired_baseline"
uv run pytest -q tests/test_reporting.py tests/test_pipeline.py
```

## 2026-06-11 Full Metrics Failure Classification

本次收紧 S3 失败分类边界，避免 full train/eval 已完成且 full metrics 明显失败时，仍被 activation / runtime / repairable proxy 信号误归为 `implementation_failure`。

- 新增强规则：只要 candidate 已产生 full S3 metrics、dataset metrics 或 `delta_vs_baseline`，该候选的失败就是 method-level evidence。
- `_classify_s3_failure_from_candidates()` 会先检查 full metrics evidence，再判断 implementation failure。
- 因此 cheap proxy 通过但 full metric 崩掉时：
  - `failure_class=method_failure`；
  - `does_not_consume_same_direction_attempt=false`；
  - 会计入同方向尝试次数；
  - 可进入 proxy/full false-positive calibration；
  - 不会走 `implementation_repair_routes` 的免计数 patch-only 修复通道。
- implementation failure 仍用于 full S3 前的 patch/runtime/eval-path 问题，例如 patch rejected、runtime smoke failed、proxy command failed、activation switch 未接通且没有 full metrics。

验证：

```text
python -m py_compile src/auto_research/orchestrator.py
uv run pytest -q tests/test_pipeline.py -k "implementation_failure_routes_to_s2 or full_metrics_failure_is_method_failure or full_metrics_repairable_does_not_use_implementation_repair_route or repairable_proxy_budget_returns_to_s1 or full_failure_records_proxy_calibration_shared_memory"
```

## 2026-06-10 Shared Memory Decision Refs

本次给 S1/S2 决策产物增加 `used_shared_memory_refs` 审计字段，用来记录哪些共享方法失败记忆实际影响了本轮方向或 variant 决策。

- S1 Codex evidence agent prompt 要求：如果 `shared_method_failure_memory` 影响了方向选择、forbidden patterns 或负约束，必须复制精确 `memory_id` 到 `used_shared_memory_refs`。
- S1 会把有效 memory refs 写入：
  - `literature/ideas.json`
  - `literature/idea_debate.json`
  - `literature/negative_constraints.json`
  - `literature/c2c/direction_decision.json`
  - `literature/c2c/evidence_session.json`
- S2 directional planner / Codex resume planner prompt 要求：如果共享记忆影响了 candidate、anti-repeat rule 或 forbidden pattern，必须在 top-level 和每个 variant 写出 `used_shared_memory_refs`。
- S2 会把 memory refs 传播到：
  - `plan/candidate_ideas.json`
  - `plan/next_variant.json`
  - `plan/plan.yaml.directional_planning`
  - `plan/s2_planner_memory.json`
- 程序侧新增 resolver：只接受共享记忆池中真实存在的 `memory_id`；如果模型没有显式填写，但 payload 文本引用了真实 `memory_id`，会自动补抓。
- S2 fallback / disabled planner 路径会继承 S1 的 refs，避免真实流程中审计字段断链。

验证：

```text
python -m py_compile src/auto_research/method_memory.py src/auto_research/agents/literature.py src/auto_research/agents/plan.py
uv run pytest -q tests/test_failure_log.py
uv run pytest -q tests/test_c2c.py -k 'pipeline_runs_to_s3_with_mock_small_loop or s1_codex_evidence_agent or s2_directional_planner_uses_direction_variants'
uv run pytest -q tests/test_c2c.py -k 's2_directional_planner or s2_resume_planner or s2_variant_scorer'
```

## 2026-06-10 Proxy/Full Calibration Memory Priority

本次把“cheap proxy 通过但 full S3 失败”的校准证据升级为共享方法记忆里的高优先级信号，避免 S1/S2 后续继续被错误 proxy 方向误导。

- `append_shared_c2c_method_failure()` 会读取 `experiment/results/proxy_calibration.json`，并提炼为共享池内的 `proxy_calibration`：
  - false positive proxy count / rate
  - proxy/full delta correlation
  - proxy 预测错的数据集
  - 容易 proxy 好但 full 差的 mechanism / integration point
  - false-positive candidate 摘要
- 每条共享记忆新增 `memory_quality`：
  - `priority`
  - `signals`
  - `proxy_full_false_positive` 和 `proxy_dataset_misprediction` 会显著提高 priority。
- `load_shared_method_memory()` 现在先按 `memory_quality.priority` 再按时间选取 top-k，S1/S2 默认优先看到高价值记忆。
- S1/S2 prompt 明确说明：`shared_method_failure_memory.recent_entries` 已按优先级排序，优先吸收 `proxy_full_false_positive` / `proxy_dataset_misprediction`。
- 普通 full S3 失败路由 `_route_s3_failure_to_s1()` 也会写入 shared method memory；即使 max-iteration blocked 或 early-stop，也先保留这轮 proxy/full 经验。
- 共享记忆 markdown 会显示 priority 和 proxy/full calibration 摘要，便于人工审计。

验证：

```text
python -m py_compile src/auto_research/method_memory.py src/auto_research/failure_log.py src/auto_research/orchestrator.py src/auto_research/agents/literature.py src/auto_research/agents/plan.py
uv run pytest -q tests/test_failure_log.py
uv run pytest -q tests/test_pipeline.py -k 'full_failure_records_proxy_calibration_shared_memory or early_stop or proxy_rejected'
```

## 2026-06-10 Memory Report CLI

本次新增 `auto-research memory report`，用于直接审计共享方法失败记忆池。

- `auto-research memory report` 显示全局共享池摘要：
  - method failure 总数
  - top failed mechanisms
  - top dragging datasets
  - 最近新增 memory
- `auto-research memory report --project-id <id>` 会额外显示当前项目配置下 S1/S2 实际会检索到的 memory：
  - retrieved count
  - memory_id
  - priority
  - memory_quality signals
- 支持 `--json` 输出结构化 JSON，便于后续 dashboard 或自动审计。
- 支持 `--limit N` 临时覆盖项目 prompt retrieval limit，只影响 report，不修改配置。
- memory report 总览读取共享池全量；project retrieval 才应用 prompt limit，避免把“池子里有多少经验”和“S1 会看到哪些经验”混在一起。

验证：

```text
python -m py_compile src/auto_research/cli.py src/auto_research/reporting.py
uv run pytest -q tests/test_reporting.py -k 'memory_report or project_report'
```

## 2026-06-10 Retryable Quota Pause

本次把 S2.5 Codex/backend 额度类失败从普通 `failed` 拆成可恢复暂停状态，避免把 429 / quota 误判成方法失败或流程失败。

- 当 `plan/code_patches/patch_manifest.json` 为 `retryable_no_valid_patch`，且 S2 gate 的失败来自 S2.5 patch manifest / executable patch retryable check 时，orchestrator 写入 `retryable_paused`。
- `retryable_paused` 保留当前 stage，例如 `S2_plan`，不推进阶段，不自动重启。
- 不调用 `increment_judge_retry()`，所以不消耗 S1/S2/S2.5 方法或实现尝试预算。
- registry / orchestration state / stage contract 都记录：
  - `status=retryable_paused`
  - `pause_type=codex_quota_or_rate_limit`
  - `resume_instruction=Wait for quota/rate limit recovery, then run auto-research resume --project-id ...`
- `auto-research status` 和 `auto-research report` 会显示 pause type 与 resume 指令；report 的 next route 为 `resume_after_quota_recovery`。
- 普通 S2 schema / plan contract 可自动修复问题仍走原有 judge retry，不会被错误暂停。

验证：

```text
python -m py_compile src/auto_research/registry.py src/auto_research/orchestration_state.py src/auto_research/orchestrator.py src/auto_research/reporting.py
uv run pytest -q tests/test_pipeline.py -k 'retryable_codex_limit or implementation_failure or repairable_proxy or proxy_rejected or proxy_feedback'
uv run pytest -q tests/test_reporting.py
```

## 2026-06-09 C2C Implementation vs Method Failure Routing

本次把 S3 前后的失败反馈拆成 implementation failure 和 method failure，避免 S2.5 代码实现问题消耗 S1/S2 方法方向预算。

- `implementation_failure` 只回 S2/S2.5 patch repair，不回 S1，不增加同方向 method attempt 计数。
- `method_failure` 必须满足 patch 已合法、runtime/eval path 接通、cheap proxy 已实际跑完并因为指标/数据集收益差失败，才进入同方向 5 次预算或最终回 S1。
- implementation failure 覆盖：
  - invalid / missing frozen `patch_json`
  - `code_patch.status != ok`
  - validation / runtime smoke / first batch 失败
  - evaluator / test-only / 过宽高风险 patch
  - proxy command failure
  - eval output health failure，例如全零、答案解析异常
  - activation smoke / ablation switch / eval wiring 未接通
- 新增 `implementation_repair_routes`，独立于原来的 `repair_routes` / `proxy_rejected_routes`。
- implementation repair 写出的 `plan/performance_feedback.json` 会标记：
  - `summary.failure_class=implementation_failure`
  - `summary.does_not_consume_same_direction_attempt=true`
  - `summary.recommended_s2_action=patch_repair`
- implementation failure 不更新 `plan/direction_scorecard.json`，避免把代码实现失败污染为方法方向失败。
- implementation failure 进入 `S2_plan` 只是复用当前代码里的阶段容器；`PlanAgent` 会检测 `failure_class=implementation_failure`，跳过 S2 resume planner / variant scorer / planner memory append，只复用 `plan/candidate_ideas.json` 直接运行 S2.5 patch repair。
- S2.5 patch-only repair 会写出 `plan/s2_5_patch_only_repair.json`，并把 `performance_feedback` 注入 `previous_patch_failure` / `previous_failure`，让 Codex 只修 patch eligibility。
- S2 planner prompt 也已明确：遇到 `implementation_failure` 时保持当前 S1/S2 方法假设，只生成 patch repair candidate，不花 token 想新机制方向。

验证：

```text
python -m py_compile src/auto_research/orchestrator.py src/auto_research/agents/plan.py
uv run pytest -q tests/test_pipeline.py -k 'implementation_failure or repairable_proxy or proxy_rejected or proxy_feedback'
9 passed
uv run pytest -q tests/test_c2c.py -k 's2_5_patch_only or s2_directional_planner or s2_variant_scorer or code_patch_runtime_smoke_repairs_missing_mechanism_activation_wiring'
5 passed
uv run pytest -q tests/test_c2c.py
134 passed
uv run pytest -q tests/test_stage_contracts.py tests/test_pipeline.py
17 passed
```

## 2026-06-09 C2C Mechanism Activation Wiring Smoke

本次把 activation smoke 从“proxy 分数/预测是否变化”前移并下沉到 S2.5 runtime smoke，目标是在进入 S3 cheap proxy / full train 前就发现 patch 的 ablation switch 没有真正接入 eval path。

- S2.5 validation 新增 `runtime_smoke:mechanism_activation_wiring`。
- 该检查会验证：
  - candidate 是否声明 `experiment_contract.ablation_switch`。
  - enabled eval config 是否带入 candidate 的 `model.rosetta_config` 机制字段。
  - disabled eval config 是否写入 `model.rosetta_config.<ablation_switch>=true`。
  - enabled eval config 是否错误地提前设置 disable switch。
  - `rosetta/model/aligner.py`、`projector.py`、`wrapper.py` 等 runtime 文件是否引用该 switch。
  - 如果存在 `forward` 函数，必须能在 `forward` 源码中读到该 switch；只在注释/常量里出现不会通过。
- S2.5 runtime smoke 不依赖 cheap proxy 已启用；即使 proxy_screen 关闭，也会从候选 eval config 合成最小 disabled eval config 来做接线检查。
- 如果 wiring smoke 失败，Codex repair prompt 会明确要求修 `rosetta_config` loading、eval config path、wrapper/projector/aligner forward switch，而不是重写 idea。
- S3 activation smoke 继续比较 metric / prediction / answer distribution，但现在会额外读取 eval 输出目录中的 tensor trace artifact：
  - `activation_trace.json/jsonl`
  - `mechanism_trace.json/jsonl`
  - `tensor_trace.json/jsonl`
- 如果 tensor trace 显示 enabled/disabled tensor 改变，即使 metric 不变，也判为“机制接通但效果中性”，允许进入后续 repair / full-train readiness 判断。
- 如果 proxy 分数变了但 tensor trace 完全没变，判为 `eval_noise_suspected`，不把它当作真实机制收益。
- 如果没有 tensor trace artifact，则保留 wiring-level 结果，并在 artifact 中记录 `tensor_trace.status=not_collected`。

验证：

```text
python -m py_compile src/auto_research/code_patch.py src/auto_research/c2c.py src/auto_research/agents/experiment.py
uv run pytest -q tests/test_c2c.py
134 passed
uv run pytest -q tests/test_stage_contracts.py tests/test_pipeline.py
16 passed
```

## 2026-06-05 C2C Proxy Eval-Path Activation Smoke

本次在 cheap proxy 通过后、full S3 训练前新增硬 gate，用来提前拦截“patch 能跑但机制没有真正接入 eval path / ablation switch 无效果”的候选。

- `DEFAULT_C2C_PROXY_SCREEN.activation_smoke` 默认启用。
- activation smoke 复用 proxy checkpoint，不重新训练；只额外跑 1 个数据集的 ablation-disabled eval。
- disabled eval config 写入 `local/auto_research_runs/<run_id>/proxy/activation_smoke_disabled/`。
- 程序比较 proxy enabled metrics 和 activation-smoke disabled metrics：
  - 有可观测 enabled-vs-disabled 差异：允许进入 full S3。
  - 无 metrics、disabled eval 输出健康失败、或 enabled/disabled 完全无差异：标记为 `proxy_repairable`，回 S2.5 修 patch，不进入 full train。
- failure attribution 新增 `proxy_activation_smoke_no_effect`，repair contract 会明确要求修复 eval-path activation / ablation switch wiring。
- 如果 `proxy_screen.enabled=false`，activation smoke 自动跳过，不影响原有 full train、OOM recovery、checkpoint recovery 测试路径。

后续补强：

- activation smoke 不再只看 metric 差异，还会比较 enabled/disabled prediction artifacts：
  - `prediction_diff_rate`
  - `answer_diff_rate`
  - mean output length delta
  - answer distribution change
- metric 没变但 prediction / answer / output profile 变化时，判定为“机制接通但效果中性”，允许进入 full S3。
- metric 和 prediction 都无变化时，才判定为强 no-op，回 S2.5 修 patch。
- S2.5 repair contract 会显式携带 activation smoke 失败证据：`ablation_switch`、disabled eval config path、enabled/disabled metrics、metric comparison、prediction comparison 和输出健康 red flags。
- S2.5 repair prompt 明确要求优先检查 `rosetta_config` loading、wrapper/projector/aligner forward path、train/eval config parity、ablation switch polarity；禁止通过换 idea、弱化 gate、改 evaluator 来绕过失败。

验证：

```text
python -m py_compile src/auto_research/c2c.py src/auto_research/agents/experiment.py src/auto_research/code_patch.py
uv run pytest -q tests/test_c2c.py
129 passed
uv run pytest -q tests/test_stage_contracts.py tests/test_pipeline.py
16 passed
```

## 2026-06-05 Proxy Calibration + S2 Variant Diversity Feedback

本次把真实 full S3 后的 cheap proxy / full train 相关性校准结果，接回 S2 variant search，降低 proxy 逐渐筛错方向的风险。

Proxy calibration 增强：

- `experiment/results/proxy_calibration.json` 继续按 iteration 记录 proxy passed 且有 full metrics 的候选。
- 每个候选记录 `mechanism_type`、`mechanism_axis`、`integration_point`、`control_signal`，并对比：
  - proxy delta
  - full delta
  - proxy_full_delta_error
  - proxy 是否 false positive
  - 哪个 dataset proxy 预测错
- summary 新增 `method_feedback`：
  - `risky_datasets`
  - `risky_mechanisms`
  - `risky_integration_points`
  - 面向 S1/S2 的 recommendations
- 机制级 summary 记录 `mispredicted_datasets`，不只是 false positive rate。

S2 variant scorer 增强：

- 同批 `variant_candidates` 如果重复 `mechanism_axis` / `integration_point` / `control_signal`，会被轻度降权，鼓励 3-5 个 variant 真正分散。
- planner memory 中连续失败的 integration point 会被降权。
- 最近失败过的 expected file group 会被降权。
- proxy calibration 标记的 risky mechanism / risky integration point 会被降权。
- 如果 variant 对 proxy-risky dataset 声称提升但没有在描述中解释该 dataset，会被降权。

验证：

```text
python -m py_compile src/auto_research/agents/plan.py src/auto_research/agents/experiment.py
uv run pytest -q tests/test_c2c.py -k 'variant_scorer or proxy_calibration'
4 passed
uv run pytest -q tests/test_pipeline.py -k 'proxy or feedback'
9 passed
```

## 2026-06-04 S2 Structured Variant Search

本次把 C2C S2 从“同方向再生成一个 candidate idea”升级为“同方向内有结构地搜索机制变体”，目标是在 5 次同方向迭代里真正探索不同机制实现，而不是局部重复同一个 patch。

- S2 planner 现在优先输出 `variant_candidates`，旧 `candidates` 仅作为兼容输入。
- 每个 variant 都会标准化为 `mechanism_axis`、`integration_point`、`control_signal`、`expected_dataset_tradeoff`、`risk_budget`、`anti_repeat`，并生成稳定 `variant_fingerprint`。
- 程序侧新增 diversity / risk / failure-target scorer：
  - 奖励新 fingerprint、新机制轴、新集成点、新控制信号。
  - 奖励针对 dragging dataset 的 variant，以及保留已有正收益 dataset 的 variant。
  - 惩罚重复历史 fingerprint、复用近期高风险文件、触碰 evaluator/result 文件、硬 gate 风险。
  - `plan/performance_feedback.json` 中嵌套的 `candidate_results` 也会被解析，用于识别拖后腿数据集和 patch risk 文件。
- S2 写出 `plan/variant_candidates.json`，记录所有 variant、score、selected_for_s2_5 和历史/反馈摘要。
- S2.5 只接收 scorer 选出的 1-2 个 selected variants；对应 `candidate_ideas` 内包含 `variant_fingerprint` 和 `s2_variant`。
- S2.5 implementation contract 明确要求实现该 `s2_variant.variant_fingerprint` 对应的机制变体，而不是泛泛按 S1 方向写代码。
- patch payload、`patch_manifest.json`、`selected_patch` 都记录 `variant_fingerprint` / `s2_variant`，方便判断 S2.5 是否真的实现了 S2 选中的 variant。
- 高风险 patch 仍优先在 S2.5 repair；如果连续复用高风险 integration point，S2 scorer 会降低该类 variant 排名，推动同方向内重新选择 integration point。

验证：

```text
python -m py_compile src/auto_research/agents/plan.py src/auto_research/code_patch.py
uv run --extra dev python -m pytest -q tests/test_c2c.py -k 's2_directional_planner_uses_direction_variants or s2_variant_scorer or code_patch_contract_includes_s2_variant_fingerprint or code_patch_contract_includes_c2c_mechanism'
4 passed, 114 deselected
```

## 2026-06-04 S0 Static Bundle Cache Fast Path

S0 现在把“已有静态证据就不要重复生成”作为阶段默认行为：

- C2C S0 启动时先检查 `intake/c2c/static_bundle.json`。
- 如果 bundle schema、chunk index、paper/rebuttal/code chunk 覆盖和关键字段完整，直接复用，不再调用 MinerU、tree-sitter code intake、DeepSeek semantic enrichment 或 reference import。
- 命中缓存时会重新注册已有 S0 artifacts 到 `intake/stage_manifest.json`，保证 stage manifest 和 contract 仍可追踪。
- 如果 sidecar artifact 缺失，例如 `chunk_index.jsonl`、`code_chunks.jsonl`、`code_repo_map.md`、`code_intake_report.md`，会从 `static_bundle` 轻量恢复，而不是重跑完整 S0。
- 只有设置 `intake.force_refresh=true` 或 `c2c.s0_force_refresh=true` 时，才强制重新生成 S0 静态证据。

验证：

```text
python -m py_compile src/auto_research/agents/intake.py
uv run --extra dev python -m pytest -q tests/test_c2c.py::test_c2c_s0_reuses_cached_static_bundle_and_restores_sidecars
1 passed
uv run --extra dev python -m pytest -q tests/test_c2c.py -k 's0_reuses_cached_static_bundle or c2c_pipeline_runs_to_s3 or missing_reference_path or c2c_s1_merges_s0_semantic'
4 passed, 115 deselected
uv run --extra dev python -m pytest -q tests/test_s0_enrichment.py tests/test_validators.py -k 's0 or S0 or enrichment'
12 passed, 5 deselected
```

## 2026-06-03 S0 Static Evidence Intake + S1 Enriched Catalog

本次把原来 S1 开始时重复做的静态证据整理拆到 S0，并让 S1 直接消费 S0 的完整证据目录：

- S0 负责项目级静态初始化：reference paper、rebuttal、target repo、历史结果、baseline、negative result memory、retrieval plan、followup bundle。
- PDF 输入改为 MinerU 优先解析，目标是生成结构化 `paper_full.md`，保留 `#` / `##` 层级和公式块；解析结果按 PDF sha / parser config 缓存。
- Code 输入改为 tree-sitter 解析，生成 file manifest、symbols、function-level code chunks、code edges、repo map、implementation surface map、code retrieval index，并按文件 sha / parser config 缓存。
- S0 可选 DeepSeek 语义增强，默认模型为 `deepseek-v4-flash`，为 paper / rebuttal / code chunks 添加：
  - `semantic_summary`
  - `mechanism_tags`
  - `failure_modes`
  - `retrieval_keywords`
  - S1/S2 utility 相关字段
- DeepSeek 增强记录写入 `intake/c2c/semantic_enrichment_sample.json/jsonl`，并按 chunk text sha / prompt version / model 缓存在 `.cache/auto_research/s0_semantic_enrichment/`。
- S1 C2C 启动时会把 static bundle 内嵌语义字段、后补生成的 `semantic_enrichment_sample.*`、本地 cache 里的全量增强记录合并回 `paper_chunks` / `rebuttal_chunks` / `code_chunks` / `chunk_index`。
- 对同一个 `chunk_id`，S1 merge 优先选择更新 prompt version、非 fallback、语义字段更完整的记录；code chunk 优先使用 `deepseek_s0_code_semantic_enrichment_v2`。
- S1 prompt 的 `chunk_catalog` 现在包含 `semantic_summary`、`mechanism_tags`、`failure_modes`、`retrieval_keywords`，Codex resume evidence agent 第一轮即可看到增强后的检索目录。
- S1 写出 `literature/c2c/semantic_enrichment_merge_report.json`，记录加载记录数、source type 覆盖、fallback 数、prompt versions、chunk_index enrichment 覆盖。

验证：

```text
python -m py_compile src/auto_research/agents/literature.py
uv run pytest -q tests/test_c2c.py::test_c2c_s1_merges_s0_semantic_enrichment_into_chunk_catalog tests/test_s0_enrichment.py
13 passed
uv run pytest -q tests/test_c2c.py
110 passed
uv run pytest -q
165 passed
```

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

## 2026-06-04 S0-S3 Live Resume Routing Fix

真实项目：`workspace/c2c_s0_deepseek_sample_20260603`。

观测结果：

- S0 静态证据缓存和 DeepSeek 语义增强可复用，resume 直接从已有 S0 bundle 进入后续阶段。
- S1 成功选择 `candidate_constrained_soft_span_alignment` 方向。
- S2/S2.5 在额度恢复后能生成可执行 patch，并通过 validation / runtime smoke / mechanism review。
- S3 cheap proxy 连续拦截同方向候选，未进入 full 6 卡训练，避免了明显无收益候选的完整训练成本。
- 真实运行暴露一个路由缺陷：`repairable_proxy_risk` 已写出 `return_to_s1_new_direction`，但 orchestrator 仍继续把流程送回 S2，导致多跑一次 S2.5。

本次补强：

- 新增 `repairable_proxy_risk` 预算耗尽后的 S1 路由：预算内回 S2.5 修 patch，耗尽后携带 `performance_feedback.json` 和 `direction_scorecard.json` 回 S1 换方向。
- `max_proxy_repair_routes_per_iteration` 语义保持为“最多回 S2.5 修复次数”；默认 3 次修复后，如果下一次 cheap proxy 仍是 repairable risk，才升级为方向级失败。
- `invalidate_from()` 现在会清理旧 `blocked_reason`，避免 registry 状态显示为 running 但仍残留 blocked reason。
- 已把真实项目恢复到 `iteration=2 / S1_literature`，由 S1 吸收上一方向失败证据继续选择新机制方向。

验证：

```text
python -m py_compile src/auto_research/orchestrator.py src/auto_research/registry.py
uv run pytest -q tests/test_pipeline.py -k 'repairable_proxy or proxy_rejected_routes_to_s1 or direction_scorecard or proxy_feedback'
7 passed, 6 deselected
uv run pytest -q tests/test_pipeline.py
13 passed
uv run pytest -q
165 passed
```

## 2026-06-04 Temporary OpenAI Key-2 Scoping

为继续真实 S0-S3 resume，本次临时把 OpenAI/Codex 相关调用限定到 `OPENAI_API_KEY_2`：

- `llm.api_key_env` 支持指定 OpenAI key 环境变量。
- `llm.disable_api_key_fallback=true` 时，主 OpenAI client 不再轮换 `OPENAI_API_KEY` / `OPENAI_API_KEY_1` / `OPENAI_API_TOKEN`。
- 所有独立 `codex exec` 入口都会收到同一套 scoped environment，把 `OPENAI_API_KEY` 映射到 `OPENAI_API_KEY_2`，并移除其它 OpenAI key。
- 该限制只作用于 OpenAI/Codex key，不屏蔽 DeepSeek、MinerU、Semantic Scholar 等非 OpenAI API。
- 删除 `.auto-research/config.yaml` 和项目 `meta/project_config.yaml` 里的 `api_key_env` / `disable_api_key_fallback` 两行即可恢复默认 key fallback。

验证：

```text
python -m py_compile src/auto_research/llm.py src/auto_research/agents/literature.py src/auto_research/agents/plan.py src/auto_research/code_patch.py
```

## 2026-06-04 S1 Evidence Ref Anchor Resolution

真实 S1 运行中，Codex 第一轮已选出可用方向，但使用了 `artifact.json#anchor` 形式的证据引用，例如 `experiment/results/failure_feedback.json#entry`。这些引用对人工可读，但旧 resolver 会按完整字符串查文件，容易触发一次额外 JSON repair。

本次优化：

- `resolve_s1_evidence_refs()` 解析 `source_path` / `source_label` / code target 时支持 `#anchor`。
- resolver 会先按 anchor 前的真实路径或 code target 校验，anchor 作为附加定位信息保留在原始 ref 中。
- 覆盖 `failure_feedback.json#entry`、`direction_scorecard.json#candidate...`、`negative_result_memory.json#blocked_idea_patterns` 这类 S1 常见 artifact anchor。
- 目标是减少 S1 因可追溯引用格式小问题触发的 Codex JSON repair，降低方向选择成本。

验证：

```text
python -m py_compile src/auto_research/evidence_refs.py
uv run --extra dev python -m pytest -q tests/test_c2c.py -k 's1_codex_evidence_agent'
3 passed, 110 deselected
uv run --extra dev python -m pytest -q tests/test_pipeline.py -k 'codex'
1 passed, 12 deselected
```

## 2026-06-04 C2C Proxy Eval Smoke Attribution

S3 cheap proxy 增加输出健康度归因，避免把全数据集 0 分直接当成方法失败：

- `C2CAdapter.collect_proxy_eval_smoke()` 会扫描 proxy result summary 和 prediction artifact。
- 记录 prediction 非空率、answer-like 率、答案解析率、平均输出长度、答案分布、dataset 级诊断和 red flags。
- all-zero summary 叠加低非空率、低解析率、答案分布塌缩或缺少 prediction artifact 时，记录 `proxy_eval_health_failure`。
- failure attribution 会把这类失败标成 `proxy_eval_output_health_failure`，S2.5 repair contract 会优先要求检查输出格式、answer parser、eval recipe 输出路径和机制开关，而不是直接换 idea。
- `eval_smoke.enabled=false` 时可显式跳过该诊断；默认开启。

验证：

```text
python -m py_compile src/auto_research/c2c.py src/auto_research/agents/experiment.py
uv run --extra dev python -m pytest -q tests/test_c2c.py -k 'eval_smoke or replay_proxy_runs_paired_baseline'
3 passed, 112 deselected
uv run --extra dev python -m pytest -q tests/test_c2c.py -k 'proxy_rejected or proxy_repairable or static_proxy or proxy_carries or proxy_reuse or s3_proxy'
6 passed, 109 deselected
uv run --extra dev python -m pytest -q tests/test_pipeline.py -k 'proxy or failure_feedback'
9 passed, 4 deselected
```

## 2026-06-04 Monitoring Artifact Readability

本次整理 S2.5 patch manifest 和 S3 proxy baseline artifact 的可读性，不改变实验判定逻辑：

- `patch_manifest.json` 新增 `selected_candidate_id`、`valid_patch_ids`、`selected_patch`，人工监控时不再出现 valid patch 存在但 selected 为空的情况。
- `plan.yaml` 的 `code_patch_manifest` 摘要和 `s2_planner_memory.json` 也同步写入 selected/valid patch 字段。
- `auto-research report` 优先使用与当前 best candidate 匹配的 patch；没有当前 candidate 时才使用 manifest selected patch，避免 report 指向错误文件。
- S3 cheap proxy 新增清晰 baseline 字段：
  - `full_baseline_mean` / `full_baseline`
  - `proxy_baseline_mean` / `proxy_baseline`
  - `comparison_baseline_mean` / `comparison_baseline`
  - `proxy_delta_vs_comparison_baseline`
  - `proxy_delta_vs_proxy_baseline` 只在 paired proxy baseline 存在时写入
- `proxy_screen.artifact_paths` 集中记录 run_state、candidate/proxy run root、proxy metrics、proxy baseline metrics、proxy train config，降低人工定位成本。
- 旧字段 `proxy_delta_vs_baseline`、`baseline_metrics` 保留兼容。

验证：

```text
python -m py_compile src/auto_research/code_patch.py src/auto_research/agents/plan.py src/auto_research/agents/experiment.py src/auto_research/reporting.py
uv run --extra dev python -m pytest -q tests/test_c2c.py -k 'code_patch_agent or replay_proxy_runs_paired_baseline or proxy_metric'
21 passed, 95 deselected
uv run --extra dev python -m pytest -q tests/test_reporting.py
3 passed
uv run --extra dev python -m pytest -q
175 passed
```

## 2026-06-04 S2.5 High-Quality Variant Early Stop

真实 S0-S3 监控中发现：同一候选已经生成一个 validation/risk/mechanism review/runtime smoke 全部通过的高分 patch 后，框架仍继续生成第二个 variant。第二个 variant 出现 9000+ 行 diff 和 evaluator touched 风险，消耗大量 Codex token，但不会改善 effect-first discovery。

本次优化：

- `code_patch.stop_after_first_ok_score` 新增为 S2.5 variant early-stop 阈值，默认 `100`。
- 同一 candidate 内，只要某个 variant 已经 `status=ok` 且 quality score 达到阈值，就停止后续 variant 生成，直接进入 best patch selection。
- 同一 S2 批次内，只要已有 candidate 产出高质量 `ok` patch，后续 candidate 会标记为 `skipped_high_quality_patch_found`，直接推进 S3 cheap proxy。
- 低质量 `ok` patch 不会提前停止；仍保留 best-of-N 继续搜索更好 variant。
- 该策略只减少明显浪费的 patch 生成，不降低“第一个 ok 但证据弱时继续探索”的能力。

验证：

```text
python -m py_compile src/auto_research/code_patch.py
uv run --extra dev python -m pytest -q tests/test_c2c.py -k 'second_variant_after_first_validation_failure or scores_all_ok_variants or stops_variants_after_high_quality or skips_later_candidates_after_high_quality'
```

## 2026-06-04 S2 Codex Resume Fallback Tightening

真实监控中发现：S2 Codex resume planner 因 duplicate output 触发 session reset 后，会继续调用普通 GPT directional planner。这个行为违背“同方向内用 Codex resume + memory 连续规划”的策略，也会在已有 S2 memory 的情况下额外消耗 OpenAI API。

本次优化：

- 真实 API 模式下，Codex resume planner 如果返回 invalid/duplicate/timeout，不再自动调用普通 GPT planner。
- 默认改为使用当前 S1 方向的 fallback candidate，并在 metadata 中记录 `fallback_resume_planner_unavailable` 和 resume planner 的 reset/health 信息。
- 非真实 API / simulate 测试模式仍保留原 fallback 行为，方便本地单元测试和无 Codex 环境演示。
- 如确实需要旧行为，可显式设置 `agents.s2_directional_planner.fallback_to_gpt_after_resume_failure=true`。

验证：

```text
python -m py_compile src/auto_research/agents/plan.py
uv run --extra dev python -m pytest -q tests/test_c2c.py -k 's2_resume_planner_resets_duplicate_session or s2_resume_planner_real_api_skips_gpt_fallback'
```

## 2026-06-04 Cheap Proxy Soft Warning Relaxation

真实 S0-S3 监控中发现：`utility_prior_departure_blend_alpha_cap_support_v2` 的 cheap proxy 有明显正向均值信号，但因为 OpenBookQA regression `0.7812` 轻微超过 soft threshold `0.75`，被回路由到 S2/S2.5，导致继续消耗 token 而没有进入 full S3。

本次调整 effect-first 策略：

- hard gate 保持不变：明显负收益、严重单数据集崩溃、proxy eval health failure、评测污染和运行失败仍然阻断 full S3。
- `repair_soft_proxy_fail` 默认改为 `false`。
- soft threshold 仍然记录 `soft_fail=true` 和 `soft_flags`，用于后续失败归因和 full/proxy 相关性校准。
- 对“proxy 均值正收益 + 单数据集轻微越线”的候选，默认进入 full S3，而不是先回 S2.5 修复。
- 如需要旧行为，可显式设置 `c2c.small_loop.proxy_screen.repair_soft_proxy_fail=true`。

验证：

```text
python -m py_compile src/auto_research/c2c.py src/auto_research/agents/experiment.py
uv run --extra dev python -m pytest -q tests/test_c2c.py -k 'proxy_metric or soft_warning or borderline_dataset_regression or soft_zero_delta'
uv run --extra dev python -m pytest -q tests/test_pipeline.py -k 'proxy or failure_feedback'
```

## 2026-06-05 Proxy-to-Full Readiness Report

真实监控中发现：C2C 进入 full S3 前的 gate 证据分散在 `proxy_screen`、`activation_smoke`、`eval_smoke`、patch risk 和 ablation switch 字段里。流程能跑，但人工判断“为什么值得花 full train”需要反复翻 artifact。

本次优化：

- C2C cheap proxy 和 activation smoke 都通过后、full train 启动前，生成 `experiment/results/full_s3_readiness_report.json`。
- readiness report 记录：
  - static patch risk 是否干净；
  - proxy delta / score / dataset deltas / soft warnings；
  - eval smoke 的 nonempty prediction rate、answer parse rate、answer distribution 和 red flags；
  - activation smoke 是否 passed、是否 activation no-op、prediction diff rate / answer diff rate；
  - ablation switch 是否声明、是否在 activation smoke 中表现为有效；
  - `worth_full_train`：为什么值得进入 full S3，以及 warnings/blockers。
- 同一份报告也写入候选 run 目录 `local/auto_research_runs/<run_id>/full_s3_readiness_report.json`，方便和具体 checkpoint/eval 追溯。
- `auto-research report` 新增监控字段：
  - activation no-op；
  - proxy / activation / full / ablation 四个状态；
  - readiness artifact path。

验证：

```text
python -m py_compile src/auto_research/agents/experiment.py src/auto_research/reporting.py
uv run pytest -q tests/test_c2c.py -k "proxy_replay_runs_paired_baseline or readiness"
uv run pytest -q tests/test_reporting.py
```

## 2026-06-05 S0 Semantic Enrichment Fail-Open

真实运行 `c2c_real4_20260605` 时发现：新项目第一次 S0 还没有 `intake/c2c/static_bundle.json`，如果 `intake.semantic_enrichment.enabled=true` 但 `DEEPSEEK_API_KEY` 缺失，S0 会在写出 static bundle 前抛异常。这个行为不符合 S0 的定位：S0 的核心职责是建立可复用静态证据 bundle，DeepSeek enrichment 只是可选语义增强，不应该阻断 raw chunk / static bundle 生成。

本次修复：

- 同一项目已有 `intake/c2c/static_bundle.json` 时，S0 仍优先复用缓存，不再重复跑 MinerU / tree-sitter / DeepSeek。
- 新项目第一次 S0 如果 DeepSeek semantic enrichment 缺 key 或失败，改为 fail-open：
  - 保留 paper/rebuttal/code raw chunks；
  - 继续生成 `static_bundle.json`、`chunk_index.json` 和所有 S0 sidecar artifacts；
  - 在 `static_bundle.semantic_enrichment` 中记录 `status=failed_open`、失败原因和 `fallback=raw_chunks_without_semantic_enrichment`。
- 这样 `OPENAI_API_KEY_2` 单 API 运行时不会因为 DeepSeek 可选增强中断 S0-S3 主流程。

验证：

```text
python -m py_compile src/auto_research/agents/intake.py
uv run pytest -q tests/test_c2c.py -k "cached_static_bundle or semantic_enrichment_missing_key"
```

## 2026-06-05 Restore Primary OpenAI Key

真实 S0-S3 运行前一度为了隔离额度，把 OpenAI/Codex 调用临时限定到 `OPENAI_API_KEY_2`。现在恢复为主 key：

- `.auto-research/config.yaml` 的 `llm.api_key_env` 改回 `OPENAI_API_KEY`。
- `llm.disable_api_key_fallback=true` 保持不变，因此 OpenAI/Codex 不会自动轮换到 `OPENAI_API_KEY_2`。
- 已经启动的 Codex 子进程不会自动继承新配置，需要重启当前 resume 进程后才生效。

## 2026-06-05 S3 Posthoc Respects Execution LLM Disable

真实 4 轮 C2C 运行中发现：项目配置已设置 `experiment.disable_llm_during_execution=true`，但 S3 失败收尾里的 posthoc review 仍可能调用 GPT。对于 effect-first discovery，这会在 cheap proxy / activation smoke 已经给出明确失败证据时额外消耗 token，并拖慢回 S2/S2.5 的路由。

本次调整：

- S3 posthoc review 现在会优先检查 `experiment.disable_llm_during_execution`。
- 该开关开启时，S3 失败分析直接使用 deterministic feedback，不再调用 GPT。
- proxy-only / activation-smoke repairable 失败仍优先走 deterministic proxy feedback。
- 这只影响 S3 执行期收尾分析，不影响 S1/S2 的 Codex/GPT 方向选择和规划生成。

验证：

```text
python -m py_compile src/auto_research/agents/experiment.py
uv run pytest -q tests/test_c2c.py -k "posthoc_review"
```

## 2026-06-09 S2.5 Patch Scope Auto-Prune

真实 resume `c2c_real4_20260605` 时发现：implementation_failure 已经正确留在 S2.5 repair 内，没有回 S1/S2 方法层；但 Codex patch repair 会反复“口头收窄 patch”，实际 diff 仍触碰 `script/evaluation/*`、`script/train/*`、`rosetta/train/*`、`wrapper.py` 等文件，导致 patch_too_broad / evaluator-touched repair 循环持续消耗 token。

本次修复：

- 在 S2.5 从 worktree freeze patch 之前增加 deterministic scope auto-prune。
- evaluator-like 文件无条件恢复到 source repo，避免评测污染进入 patch。
- 如果 changed files 超过 `validation.max_changed_files`，按优先级保留：
  - required new mechanism files；
  - expected files；
  - `rosetta/model/*` 机制文件；
  - focused smoke tests；
  - recipe 文件；
  - 最后才保留 train integration 文件。
- 被剔除的文件会记录到 patch/validation 的 `recovery_actions`，例如 `auto_prune_patch_scope_before_freeze`。
- 旧的 evaluator-touched contract repair 路径保留，可通过 `validation.auto_prune_scope=false` 测试或禁用。

验证：

```text
uv run pytest -q tests/test_c2c.py -k 'auto_prunes_evaluator or evaluator_proxy_risk or evaluator_repair_that_recontaminates'
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent or runtime_smoke or patch_manifest or s2_5'
uv run pytest -q tests/test_c2c.py
```

## 2026-06-09 S2.5 Worktree Scope Pre-Prune

继续真实 resume 后发现：仅在 patch build 成功后 auto-prune 还不够。Codex 有时会删除旧辅助文件或分析脚本，`_build_patch_from_repo_delta()` 会先返回 `patch_rejected: delete_file is not allowed`，导致流程又把这个实现噪音交给 Codex repair，继续消耗 token。

本次修复：

- 在每次 Codex 返回后、构建 patch 之前增加 deterministic worktree pre-prune。
- 被删除的 source 文件会先恢复，避免 delete_file 直接触发 contract repair。
- evaluator-like 文件仍然在 build 前恢复，不进入 patch freeze。
- 如果 worktree changed files 超过 `validation.max_changed_files`，先按 required / expected / model mechanism / focused test / recipe / train integration 优先级保留，低优先级文件恢复。
- patch build 后保留原有 auto-prune 作为二次兜底。
- recovery action 记录为 `auto_prune_worktree_scope_before_build` 或组合动作 `auto_prune_worktree_and_patch_scope`。

验证：

```text
uv run python -m py_compile src/auto_research/code_patch.py
uv run pytest -q tests/test_c2c.py -k 'auto_prunes_deleted_files_before_patch_build or auto_prunes_evaluator_and_low_priority_over_scope_files'
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent or runtime_smoke or patch_manifest or s2_5 or auto_prunes'
```

## 2026-06-09 S2.5 Fresh Patch Session For Implementation Repair

继续真实 resume 后又发现：即使程序能清理 worktree，implementation_failure patch-only repair 仍会复用旧 Codex patch session。旧 session 里模型已经反复声明“diff 已经收窄”，但实际 worktree 仍有噪音文件，导致后续多轮 repair 被旧上下文锚住。

2026-06-12 更新：这一 fresh-session 策略已被 S2.5 persistent-only same-session repair 取代；当前实现不再使用 `force_new_codex_session` 或 `discard_patch_codex_session_for_implementation_repair` 路由。

本次修复：

- `implementation_failure` 生成的 `previous_patch_failure.proxy_effect_repair_contract` 默认带 `force_new_codex_session=true`。
- `_prepare_code_worktree_workspace()` 收到 fresh-session 请求时，只删除该 candidate/version 的 `codex_session.json`，不删除 worktree 或历史 artifacts。
- 下一次 S2.5 Codex patch 仍会 preload 当前代码，但不会 resume 旧 patch 对话。
- recovery action 记录为 `discard_patch_codex_session_for_implementation_repair`，并合并进最终 patch artifact。
- 普通 S2.5 / best-of-N / 非 implementation_failure repair 仍保留 persistent resume 行为。

验证：

```text
uv run python -m py_compile src/auto_research/code_patch.py src/auto_research/agents/plan.py
uv run pytest -q tests/test_c2c.py -k 'fresh_patch_session or persistent_backend_uses_git_worktree_and_codex_resume or persistent_backend_resume_failure_falls_back_to_new_session or s2_5_patch_only'
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent or runtime_smoke or patch_manifest or s2_5 or auto_prunes or persistent_backend'
```

## 2026-06-09 S2.5 Patch Prompt Must Edit

fresh patch session 后继续真实监控发现：Codex 不再复读旧 session，但有时会退化成只输出 blueprint / intended files，没有落地代码。这个会让 S2.5 变成二次规划，仍然浪费 token。

本次修复：

- persistent patch turn 明确声明这是 implementation turn，不是 planning turn。
- `prompt_kind=preload` 仍可输出 blueprint；`prompt_kind=patch` 必须直接编辑仓库文件。
- `_codex_patch_prompt()` 增加硬约束：不能只返回 blueprint、patch plan、file list；结束时 filesystem 必须包含实现改动。
- 如果认为无需修改，也必须先验证 failing check 已经修复；否则要做最小代码改动。

验证：

```text
uv run python -m py_compile src/auto_research/code_patch.py
uv run pytest -q tests/test_c2c.py -k 'fresh_patch_session or persistent_backend or auto_prunes or code_patch_agent or runtime_smoke or patch_manifest or s2_5'
```

## 2026-06-09 S2.5 Pre-Codex Worktree Cleanup

真实 resume 继续暴露出一个成本问题：`patch_manifest.json` 可以通过 freeze 前剪枝得到干净 patch，但持久 code worktree 里仍可能保留上一轮 implementation repair 的 evaluator/train/helper 脏改动。这样 Codex 下一次进入 S2.5 时会先读到过宽 diff，再花 token 自我修复，虽然最终可能被 deterministic prune 剪掉，但成本偏高。

本次修复：

- 在调用 Codex patch backend 之前执行同一套 deterministic worktree scope prune。
- evaluator-like 文件、删除文件、超出 `validation.max_changed_files` 的低优先级文件会先恢复到 S0 snapshot/source repo。
- fresh implementation repair 仍只重置 patch Codex session，不回 S1/S2；这次补丁让 fresh session 看到的是干净实现上下文。
- patch artifact 的 `recovery_actions` 会记录 `auto_prune_worktree_scope_before_codex`，方便监控时确认 cleanup 生效。
- Codex 返回后的 pre-build prune 和 freeze 后 prune 仍保留，形成三层兜底。

验证：

```text
uv run python -m py_compile src/auto_research/code_patch.py
uv run pytest -q tests/test_c2c.py -k 'prunes_persistent_worktree_before_codex or fresh_patch_session or auto_prunes_deleted_files_before_patch_build or auto_prunes_evaluator_and_low_priority_over_scope_files'
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent or runtime_smoke or patch_manifest or s2_5 or auto_prunes or persistent_backend'
```

## 2026-06-10 Shared Method Failure Memory

真实多轮 C2C 运行后发现：项目内 `performance_feedback.json`、`direction_scorecard.json`、`negative_result_memory.json` 能服务当前项目，但换新项目或重新初始化时，过去多轮的 method-level 失败经验不会稳定复用，导致之前的失败代价沉没。

本次新增跨项目共享池：

- 新增 `.auto-research/method_failure_memory.jsonl` 和 `.auto-research/method_failure_memory.md`。
- 只写入 `method_failure` 证据：
  - cheap proxy / full train 的指标失败；
  - dataset regression / dragging dataset；
  - mixed dataset signal；
  - direction scorecard / budget exhausted；
  - proxy-full calibration 类方法证据。
- 明确排除 implementation noise：
  - `implementation_failure`
  - dtype/device/valid_mask/runtime smoke
  - patch too broad / evaluator touched / test-only
  - Codex 429 / backend failure
  - proxy command failure
- S3 method-level feedback 写出 `plan/performance_feedback.json` 和 `plan/direction_scorecard.json` 后，会同步 append 到共享池。
- S0 默认读取共享池，写入 `intake/shared_method_failure_memory.json`，并合并到：
  - `intake/c2c/negative_result_memory.json`
  - `intake/c2c/evidence_brief.json`
- S0 cache fast path 命中已有 `static_bundle` 时，不重跑 MinerU/tree-sitter/DeepSeek，但会轻量刷新共享池 sidecar，保证新启动能看到旧失败经验。
- S1 Codex evidence agent 的 context 现在包含 `shared_method_failure_memory`，作为跨项目方法级负证据。
- S2 resume planner 也能看到 `shared_method_failure_memory`，但只作为 method-level avoid-repeat 约束；implementation/runtime 修复仍只使用当前项目证据。

验证：

```text
uv run python -m py_compile src/auto_research/method_memory.py src/auto_research/failure_log.py src/auto_research/agents/intake.py src/auto_research/agents/literature.py src/auto_research/agents/plan.py src/auto_research/orchestrator.py
uv run pytest -q tests/test_failure_log.py
uv run pytest -q tests/test_c2c.py -k 's0_reuses_cached_static_bundle or c2c_s1_merges_s0_semantic or s2_resume_planner or s2_variant_scorer or s2_5_patch_only'
uv run pytest -q tests/test_pipeline.py -k 'implementation_failure or repairable_proxy or proxy_rejected or proxy_feedback'
```

## 2026-06-10 Shared Method Memory Quality Scoring

共享 method memory 之前只有粗粒度 `priority/signals`，无法区分普通 cheap proxy fail 和更有价值的 full-train false positive，也无法把跨项目重复失败机制抬高给 S1/S2 优先避开。

本次增强：

- `memory_quality` 扩展为可审计结构：
  - `priority`
  - `signals`
  - `score_components`
  - `evidence`
- 评分维度包括：
  - full train failure 高于 cheap proxy failure；
  - cheap proxy 通过但 full train 失败的 false positive 加权最高；
  - 同方向多次失败会按复现次数加权；
  - 有 ablation evidence 的失败高于只有指标均值的失败；
  - 明确 dataset regression / dragging dataset 高于只有 mean delta；
  - 同一 mechanism type 在多个 project 失败时自动加 `cross_project_mechanism_failure`。
- `load_shared_method_memory()` 和 append 写入时都会基于整个共享池重算质量分，因此旧 JSONL 不需要迁移，后续改评分规则也能在加载时生效。
- markdown 摘要现在写出 `Quality components`，方便人工监控为什么某条 memory 被排到前面。
- `shared_method_memory_for_prompt()` 现在额外提供：
  - `ranking_policy`
  - `quality_summary`
  - `retrieved_quality_summary`
  - `high_quality_memory_ids`
- S0 cache fast path 和普通 S0 都会把 `quality_summary / ranking_policy / high_quality_memory_ids` 写入 `intake/shared_method_failure_memory.json`、`intake/c2c/negative_result_memory.json` 和 `intake/c2c/evidence_brief.json`。
- S1/S2 prompt 明确要求按 `memory_quality.priority` 和 `ranking_policy` 使用共享池，优先吸收：
  - `proxy_full_false_positive`
  - `full_train_failure`
  - `proxy_dataset_misprediction`
  - `cross_project_mechanism_failure`
  - `ablation_evidence`
- 共享池进入 S0/S1/S2 prompt 前不再只是全局 top priority 截断，而是执行 top-k retrieval：
  - task/topic token overlap
  - dataset match，例如 `mmlu-redux`
  - `mechanism_type` match
  - failure mode / route / decision match
  - source repo fingerprint match
- 每条 prompt memory 带 `memory_retrieval`：
  - `quality_priority`
  - `relevance_score`
  - `combined_score`
  - `matched_fields`
  - `matched_values`
- 排序规则优先保留有 relevance match 的记录，再按 `combined_score` 排；没有检索上下文时才退回纯质量优先级。
- 新写入共享池的 method memory 会记录 `source_repo_fingerprint`，旧 memory 没有该字段时仍可通过 topic/dataset/mechanism/failure mode 检索。
- Prompt 视图改为 catalog-first：
  - `shared_method_memory_for_prompt()` 的 `recent_entries` 和 `memory_catalog` 只包含轻量错误目录。
  - 每条目录项包含 `one_line_summary`、`signals`、`datasets`、`mechanism_types`、`failure_modes`、`retrieval` 和 `read_hint`。
  - 完整 `proxy_calibration`、candidate entries、direction scorecard 不再直接塞进 S1/S2 prompt。
  - S1/S2 prompt 明确要求：如果某条 catalog item 影响决策，先按 `read_hint` / `full_memory_access` 自己读取完整 JSONL 或项目 snapshot，再引用 `memory_id`。
  - S0 的 `negative_result_memory` 和 `evidence_brief` 也只暴露 catalog，避免 brief 被完整失败证据污染。

验证：

```text
python -m py_compile src/auto_research/method_memory.py
uv run pytest -q tests/test_failure_log.py -k 'shared_method_memory'
uv run pytest -q tests/test_pipeline.py -k 'shared_method_memory'
uv run pytest -q tests/test_reporting.py -k 'memory_report'
uv run pytest -q tests/test_c2c.py -k 's0_reuses_cached_static_bundle'
uv run pytest -q tests/test_c2c.py -k 's1_codex_evidence_agent or pipeline_runs_to_s3_with_mock_small_loop or s2_directional_planner or s2_resume_planner'
```

## 2026-06-10 S1 Novelty Auditor

新增一个独立的 S1 后置 Codex auditor，用来比较新 S1 idea/direction 和历史失败/共享记忆的相似度，避免 S1 只是换名字重复旧方向。

- 配置入口：`agents.s1_novelty_auditor`。
- 默认关闭，需要显式设置 `enabled: true`，避免真实运行突然增加额外 Codex 成本。
- S1 Codex evidence agent 先生成方向 JSON。
- novelty auditor 使用独立 session key：
  - C2C: `s1:c2c_novelty_auditor`
  - generic: `s1:generic_novelty_auditor`
- auditor 只读：
  - 当前 `direction_decision`
  - 当前 selected idea
  - top-k shared method memory
  - local `performance_feedback`
  - `direction_scorecard`
  - `s2_planner_memory`
- auditor 返回：
  - `novelty_score`
  - `max_similarity_score`
  - `passed`
  - `most_similar_sources`
  - `repeated_patterns`
  - `revision_guidance`
- 如果 `novelty_score < threshold` 或 `passed=false`：
  - 不进入 S2；
  - 把 auditor feedback 回传到同一个 S1 Codex evidence session；
  - 要求 S1 改 core mechanism hypothesis 或 allowed variant family，而不是只改名；
  - 最多重试 `max_revision_rounds` 次。
- 通过后写入：
  - generic: `literature/novelty_audit.json`
  - C2C: `literature/c2c/novelty_audit.json`
  - idea card 内的 `s1_novelty_audit`
- S1 gate 如果发现 novelty audit artifact 存在，则最后一次 audit 必须 `passed=true`；否则 stage retry。

验证：

```text
python -m py_compile src/auto_research/agents/literature.py src/auto_research/validators/s1_gate.py
uv run pytest -q tests/test_c2c.py -k 's1_novelty_auditor_rejects_and_revises_direction'
uv run pytest -q tests/test_validators.py -k 's1_gate_retries_failed_novelty_audit'
uv run pytest -q tests/test_c2c.py -k 's1_codex_evidence_agent or pipeline_runs_to_s3_with_mock_small_loop or s1_codex_resets_duplicate_direction_session'
```

## 2026-06-10 S2.5 Soft Patch Scope Gate

S2.5 discovery 阶段放宽 `max_changed_files`，避免边界情况把可能有效的 patch 直接挡在 S3 前。

- `code_patch.validation.max_changed_files` 默认变成 soft risk：
  - 超文件数会记录 `risk_labels=["patch_too_broad"]` 和 warning；
  - 仍进入 quality score 扣分；
  - 默认不再导致 `no_valid_patch`。
- 仍然硬挡：
  - evaluator / evaluation-like 文件改动；
  - test-only patch；
  - syntax / validation / runtime smoke 失败；
  - 无 executable operations；
  - strict activation wiring 失败。
- 新增开关：
  - `strict_max_changed_files: true` 恢复文件数硬挡，适合后期收敛/论文版 patch。
  - `auto_prune_over_scope_files: true` 恢复自动剪枝低优先级超范围文件。
- 默认自动剪枝现在主要处理 deletions 和 evaluator contamination，不再因为机制文件稍多就剪掉可能有用的 runtime 接线。

验证：

```text
python -m py_compile src/auto_research/code_patch.py
uv run pytest -q tests/test_c2c.py -k 'soft_allows_over_scope_file_count_by_default or strict_max_changed_files_still_blocks or repairs_evaluator_proxy_risk or blocks_evaluator_repair_that_recontaminates or auto_prunes_evaluator_and_low_priority_over_scope_files or auto_prunes_deleted_files_before_patch_build'
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent_generates_artifacts or code_patch_agent_blocks_evaluator or code_patch_agent_marks_py_compile_failure or code_patch_agent_blocks_unactivated or code_patch_agent_repairs_unactivated or code_patch_agent_selects_best_variant or code_patch_agent_stops_after_high_quality or mechanism_self_review_keeps_diagnostics_soft or mechanism_self_review_can_be_strict or retryable_codex_patch_manifest'
uv run pytest -q tests/test_validators.py tests/test_pipeline.py -k 's2 or S2 or patch_manifest or retryable'
```

## 2026-06-10 Lightweight Discovery Gate Mode

S1-S3 当前目标是 effect-first discovery：先找到能跑、cheap proxy 有正收益、无评测污染的有效 patch。因此 S2/S2.5 gate 默认改成轻量 discovery 模式，只硬挡会污染结果或无法执行的错误，把论文质量/证据完整性问题记录为 `quality_debt`。

- 新增 `code_patch.validation.gate_mode`：
  - 默认 `discovery`。
  - 设置为 `strict` 可恢复后期收敛/论文版的硬 gate。
- S2.5 discovery 下仍然硬挡：
  - py_compile / focused pytest / first-batch runtime smoke 失败；
  - evaluator-like 文件改动；
  - test-only patch；
  - 无 executable operations；
  - retryable quota/backend pause。
- S2.5 discovery 下转为 soft debt：
  - 新增 config 参数但未显式接入 recipe/config；
  - ablation switch 静态接线证据不足；
  - coverage / matched-coverage 诊断证据不足；
  - core mechanism evidence 弱；
  - activation wiring smoke 失败但 `mechanism_activation.hard_gate` 未开启；
  - patch 文件数超过 soft limit。
- `patch.json` / `patch_manifest.json` / selected patch 摘要现在写入 `quality_debt`，best-of-N 会在能跑 patch 之间继续按质量债务和风险扣分排序。
- S2 gate 默认把 C2C coverage/matched/reviewer/novelty/scope 质量问题标为 PASS + `quality_debt`，不再阻断无人值守 discovery；strict 模式仍然 retry。
- activation wiring 可单独强制：
  - `code_patch.validation.runtime_smoke.mechanism_activation.hard_gate: true`

验证：

```text
python -m py_compile src/auto_research/code_patch.py src/auto_research/validators/s2_gate.py
uv run pytest -q tests/test_c2c.py -k 'unactivated_new_config_parameter or mechanism_activation_wiring or mechanism_self_review or coverage_controls or strict_max_changed_files or soft_allows_over_scope'
uv run pytest -q tests/test_validators.py -k 's2_gate'
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent or runtime_smoke or mechanism_self_review or patch_manifest or best_variant or executable_patch or over_scope or evaluator_proxy_risk or activation or unactivated or s2_gate'
uv run pytest -q tests/test_pipeline.py -k 'patch_manifest or retryable or implementation_failure or mechanism_activation'
```

## 2026-06-11 S2.5 Forward-Level Activation Probe

S2.5 runtime smoke 从“能跑首 batch / 静态接线”扩展到第三层：forward-level activation probe，用来提前发现“ablation switch 写进 config 但没有真正影响 wrapper/projector/aligner forward”的 no-op patch。

- `runtime_smoke:first_batch_train` 仍负责 dtype/device、valid_mask、训练首 batch 是否能跑。
- `runtime_smoke:mechanism_activation_wiring` 仍负责静态接线：
  - ablation switch 是否进入 disabled eval config；
  - eval `rosetta_config` 是否有开关；
  - wrapper/projector/aligner 代码是否引用该 switch。
- 新增 `runtime_smoke:mechanism_activation_forward_probe`：
  - 优先查找目标 repo 的 `script/auto_research/activation_forward_probe.py`；
  - 如果目标 repo 没有 probe 脚本，默认使用 auto-research 内置 C2C fallback probe：`src/auto_research/probes/c2c_activation_forward_probe.py`；
  - 内置 fallback 不加载大模型，只做 lightweight config/forward trace：disabled eval config 是否打开 switch、enabled/disabled rosetta_config 是否变化、wrapper/projector/aligner forward 路径是否读取该 switch/config；
  - 如果显式设置 `forward_probe.builtin_fallback=false` 且目标 repo 没有 probe 脚本，则跳过，不阻塞普通项目；
  - 有 repo 专用 probe 脚本时，enabled/disabled configs 必须让至少一个底层字段变化，例如 tensor checksum、routing score、cache weight、projector output；
  - 如果 metric 可能变化但底层 tensor 没变，判定为机制未接通或 eval 噪声，回 S2.5 repair；
  - 如果 tensor 变了但 metric 没变，说明机制接通但效果中性，后续可继续做效果 repair/variant，而不是 no-op。
- forward probe 默认 hard gate；repo 专用 probe 优先，内置 C2C fallback 兜底，避免真实 C2C 运行直接跳过 no-op 检查。
- `script/auto_research/activation_forward_probe.py` 被视为 validation/evaluator-like 文件，S2.5 repair 不允许通过修改 probe 脚本来“造通过”；必须修实际机制路径。
- validation repair prompt 现在会明确区分：
  - first-batch runtime 错误：修 dtype/device/valid_mask/训练首 batch；
  - activation wiring 错误：修 eval config -> rosetta_config -> forward switch 接线；
  - forward probe 错误：修实际 wrapper/projector/aligner forward，让 enabled/disabled 小 batch 的底层输出真的变化。

验证：

```text
python -m py_compile src/auto_research/code_patch.py src/auto_research/probes/c2c_activation_forward_probe.py
uv run pytest -q tests/test_c2c.py -k 'forward_probe or mechanism_activation_wiring or runtime_smoke_repairs_dtype_failure or evaluator_proxy_risk'
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent or runtime_smoke or mechanism_activation or patch_manifest or s2_5'
```

## 2026-06-11 S2.5 Config/Ablation/Forward Eligibility Gate

真实运行发现 light gate 放宽后，S2.5 能生成可运行 patch，但容易把核心执行闭环留成 quality debt：新增 config 参数没有进入 `config_overrides`/recipe，ablation switch 没有被 forward 读取，或 enabled/disabled 没有底层 forward trace 变化。这类问题不是论文质量债，而是 implementation eligibility，必须在进入 S3 前修掉。

- `code_patch.validation.require_config_activation` 默认 `true`：
  - 新增核心机制 config 参数必须被 `experiment_contract.config_overrides` 或允许的 recipe edit 激活；
  - 未激活时 `activation_check.status=config_activation_missing`，回 S2.5 validation repair；
  - repair prompt 明确要求闭合 `idea -> config -> constructor -> forward`，不能让 S3 静默跑默认值。
- `runtime_smoke.mechanism_activation.hard_gate` 默认 `true`：
  - ablation switch 必须进入 disabled eval config；
  - wrapper/projector/aligner forward 路径必须读取该 switch/config；
  - repo probe 或内置 C2C fallback probe 必须观察到 enabled/disabled forward trace 变化。
- 仍然保留显式 soft 模式：
  - `code_patch.validation.require_config_activation: "soft"` 可把新参数未激活降级为 quality debt；
  - 主要用于后期 paperization/diagnostic-only patch，不建议用于 effect-first discovery。
- 同方向次数语义不变：这些失败属于 `implementation_failure`，只回 S2.5 repair，不消耗 S1/S2 方法方向尝试。

验证：

```text
python -m py_compile src/auto_research/code_patch.py src/auto_research/probes/c2c_activation_forward_probe.py
uv run pytest -q tests/test_c2c.py -k 'unactivated_new_config_parameter or activation_repair or activated_param or mechanism_activation_wiring or forward_probe or builtin_c2c_activation'
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent or runtime_smoke or mechanism_activation or patch_manifest or s2_5 or unactivated or activation'
```

## 2026-06-11 S2.5 High-Freedom Truth-Gated Patch Policy

真实运行后确认：S2.5 最大浪费不应该靠静态“小 patch / MVP”约束解决。默认 discovery 阶段改成 `Codex 高自由度生成 + 强真实性验证 + 结果驱动迭代`。

- Codex patch prompt 不再要求 `smallest coherent code change` 或 `large -> first MVP slice`：
  - 可以改多文件、加 helper module、改 recipe/test，只要这些文件是机制接通所必需；
  - `implementation_scope` 只作为 integration surface 指引，不作为硬 patch-size cap；
  - large/medium/bounded 都要求实现一个 coherent executable mechanism slice，而不是人为压小。
- patch size / diff size 默认仍进入 quality score 和 risk label，但不作为 discovery 默认硬门槛：
  - 真正硬挡的是 evaluator contamination、无 executable diff、syntax/test/runtime 失败、config 未激活、ablation 未接通、forward trace 无变化；
  - `patch_too_broad` repair 现在要求去掉无关文件，同时保留真实机制必需文件，不再强制压到“一个核心文件 + 一个测试”。
- S2.5 repair 保持 implementation-only：
  - config activation 失败只要求修 `idea -> config -> constructor -> forward`；
  - activation wiring / forward probe 失败只修实际 runtime path；
  - 这些失败仍是 `implementation_failure`，不消耗同方向 S2/S3 方法尝试次数。
- 结果驱动迭代语义：
  - patch 通过真实性验证但 cheap proxy/full train 差，才算 method/variant failure；
  - patch 没通过真实性验证，只回 S2.5 继续修实现；
  - cheap/full 反馈进入后续同方向 variant 或共享 method memory。

验证：

```text
python -m py_compile src/auto_research/code_patch.py src/auto_research/probes/c2c_activation_forward_probe.py
uv run pytest -q tests/test_c2c.py -k 'code_patch_contract or s2_5 or patch_manifest or runtime_smoke or mechanism_activation or unactivated'
```

## 2026-06-11 C2C Repo-Specific Forward Activation Probe

本次在目标 C2C repo 落地专用 `script/auto_research/activation_forward_probe.py`，让 S2.5 runtime smoke 优先检查真实 C2C projector forward，而不是只依赖 auto-research 内置 static fallback。

- auto-research 调用逻辑不变：
  - 优先使用目标 repo 的 `script/auto_research/activation_forward_probe.py`；
  - 目标 repo 没有脚本时才回退到 `src/auto_research/probes/c2c_activation_forward_probe.py`。
- C2C repo probe 的行为：
  - 读取 enabled/disabled eval config 的 `model.rosetta_config`；
  - 检查 disabled config 是否设置 ablation switch；
  - 动态导入 `rosetta.model.projector`；
  - 优先实例化真实 `C2CProjector`，再尝试其他 `Projector` 子类；
  - 用合成 source/target KV、source weights、source confidence 跑真实 projector forward；
  - 比较 enabled/disabled 的 projector key/value 输出、source weight calibration、`last_*` 诊断字段；
  - 只有观察到真实 tensor/diagnostic 变化才返回 `mechanism_observed=true`。
- 失败语义更具体：
  - 代码里提到了 switch 但 tensor 输出完全一样，会返回 `enabled_disabled_projector_outputs_identical`；
  - import/constructor/forward 失败会进入 probe payload，S2.5 repair 继续修实现，不消耗方法方向次数。

验证：

```text
python -m py_compile script/auto_research/activation_forward_probe.py test/test_activation_forward_probe.py
/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest -q -o addopts='' test/test_activation_forward_probe.py
uv run pytest -q tests/test_c2c.py -k 'forward_probe'
```

## 2026-06-12 S2.5 Same-Session Codex Repair

真实运行发现 S2.5 patch 失败后，如果每次 repair 都重新理解上下文，会浪费 token，并且容易把 implementation failure 误当成新 idea。策略收敛为：S2.5 真实代码生成和 repair 只支持 `codex_persistent_cli`，必须有 Git worktree 和 Codex resume session；普通 `codex_cli`/新会话 repair 链路不再保留。

- `code_patch.backend` 默认且唯一真实实现是 `codex_persistent_cli`：
  - S2.5 patch/repair 必须通过同一个 persistent worktree/session；
  - 历史 `force_new_codex_session` / `repair_resume_session` 配置不再参与路由；
  - 非 persistent backend 配置会直接失败，不再悄悄走无上下文 repair。
- contract/validation repair 都会注入 `codex_repair_packet`：
  - failed command / trace；
  - changed files；
  - diff excerpt；
  - config activation / mechanism review / risk check；
  - activation wiring 和 forward probe evidence。
- persistent backend 会记录 session 复用审计：
  - `session_id_before`；
  - `session_id_after`；
  - `repair_previous_session_id`；
  - `same_session_reused`；
  - worktree metadata 写入 `session_policy=persistent_resume_required`。
- repair prompt 明确要求：
  - 继续当前 S2.5 implementation context；
  - 不重新规划 research direction；
  - activation/probe 失败必须修真实 `config -> constructor -> forward -> tensor` 路径；
  - 不编辑 validation/probe 代码绕过检查。

验证：

```text
python -m py_compile src/auto_research/code_patch.py
uv run pytest -q tests/test_c2c.py -k 'persistent_validation_repair_uses_same_codex_session or persistent_backend or implementation_failure_reuses_patch_session or prunes_persistent_worktree_before_codex or repairs_validation_failed_patch_once or repairs_blocked_no_executable_change'
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent or runtime_smoke or mechanism_activation or forward_probe or persistent_backend or s2_5 or patch_manifest or implementation_failure_reuses_patch_session or persistent_validation_repair'
```

## 2026-06-12 S3 Executes Frozen Patched Repo Snapshot

真实流程讨论后将 S3 的执行真相从 `patch.json` apply 主路径切换为 S2.5 验证过的 patched worktree snapshot。

- S2.5 仍然保留 `patch.json` / `patch.diff`：
  - `patch.json` 作为审计、回放和旧 artifact fallback；
  - `patch.diff` 作为人工 review 视图；
  - Codex 仍是在 persistent worktree 中原生改文件，不要求输出 patch。
- 新增 `patched_repo_snapshot`：
  - S2.5 在 validation/runtime smoke/activation probe 后，把通过验证的 worktree 复制到 `plan/code_patches/<candidate>/patched_repo_snapshot/`；
  - snapshot 过滤 `local/auto_research_runs`、checkpoint、cache、binary model 等运行产物；
  - 同目录写 `patched_repo_snapshot_manifest.json`，记录 sha、file_count、changed_files 和 execution_truth。
- S3 新执行顺序：
  - 优先从 `patch_manifest.selected_patch.patched_repo_snapshot` 创建 `experiment/execution_repos/<candidate>/`；
  - S3 的 train/eval/proxy/ablation 都在 execution repo 里运行；
  - `patch.json` 只在缺少 snapshot 的旧 artifact 上作为 fallback 应用到隔离 execution repo；
  - 原始 baseline snapshot repo 不再被 S3 修改/还原。
- 监控与锁定：
  - `s3_candidate_selection.json` 记录 selected snapshot 路径和 manifest sha；
  - S3 gate 校验已存在的 artifact locks，兼容未启用 S2.5 的模拟流程；
  - proxy cache fingerprint 纳入 execution source/snapshot sha，避免跨 snapshot 误复用。

验证：

```text
python -m py_compile src/auto_research/code_patch.py src/auto_research/agents/experiment.py src/auto_research/validators/s3_gate.py tests/test_c2c.py
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent or s2_5 or patch_manifest or s3_'
```

## 2026-06-12 S3 Execution Repo Pollution Audit

本次给 frozen patched repo execution 增加路径污染硬审计，防止 C2C train/eval/proxy 脚本通过硬编码路径把输出写回原始 baseline snapshot。

- S3 materialize `experiment/execution_repos/<candidate>/` 后，会做 path audit：
  - `run_root`
  - `train_config`
  - `eval_configs`
  - `proxy_run_root`
  - `proxy_train_config`
  - `proxy_eval_configs`
- 这些路径必须全部位于 execution repo 内，不能指向原始 `c2c.snapshot_path`。
- 候选执行前会记录原始 snapshot 的输出目录指纹：
  - `local/auto_research_runs/<run_id>/`
  - `local/auto_research_runs/proxy_baseline/`
- proxy baseline、proxy screen、activation smoke、full train/eval 后都会再次检查这些目录。
- 如果原始 snapshot 新增或改动运行输出文件：
  - `execution_repo_audit.status=failed`
  - `command_status=blocked`
  - `decision=blocked`
  - `failure_attribution.primary_failure=execution_repo_output_pollution`
  - `recovery_actions[].action=block_original_snapshot_output_pollution`
- 正常执行时，所有 C2C 输出必须只出现在 `experiment/execution_repos/<candidate>/local/auto_research_runs/...`。

验证：

```text
python -m py_compile src/auto_research/agents/experiment.py tests/test_c2c.py
uv run pytest -q tests/test_c2c.py -k 'execution_repo or s3_prefers_patched_repo_snapshot or s3_blocks_outputs_written_to_original_snapshot'
```

## 2026-06-13 S2.5 Single Selected Variant Implementation

真实运行后确认：S2 已经负责同方向内的结构化 variant 搜索和选择，S2.5 再做 best-of-N patch 生成会重复决策、增加 token 成本，并让 manifest/S3 执行真相更难审计。

本次把 S2.5 收窄为“实现 S2 选中的一个 variant，并在同一 persistent Codex session 内修到 eligible 或明确 blocked”：

- `CodePatchAgent.run()` 只实现一个候选：
  - 优先使用 `candidate.selected=true`；
  - 其次匹配 `plan.selected_idea`；
  - 再匹配 `plan.selected_variant_candidates` 的 `id` / `variant_fingerprint`；
  - 最后才 fallback 到第一个 candidate。
- 未选中的 candidate 不再进入 S2.5 patch 生成，标记为 `skipped_s2_5_single_candidate_mode`。
- `patch_manifest.json` 新增 `selection_policy.mode=single_s2_selected_variant`，同时记录：
  - `input_candidate_count`；
  - `implementation_candidate_count`；
  - `skipped_candidate_count`；
  - `ignored_legacy_config`。
- `code_patch.max_candidates` / `variants_per_candidate` 默认改为 `1`，旧字段仅作兼容记录，不再驱动 S2.5 搜索。
- 删除 S2.5 best-of-N variant feedback / quality-score selection 路径；质量分仍保留为 artifact 诊断，不用于二次选择。
- 保留 `variant_index=1` / `variant_attempts` 兼容字段，方便旧 report 和 artifact reader 继续工作。

新的职责边界：

- S2：同方向生成多个结构化 variant，并基于 diversity/risk/failure-target scorer 选中一个。
- S2.5：只实现这个选中 variant；语法、风险、activation、runtime smoke 失败只回同一 persistent repair loop。
- S3：继续以 S2.5 产物 `patch_manifest.selected_patch.patched_repo_snapshot` 为执行真相。

验证：

```text
python -m py_compile src/auto_research/code_patch.py tests/test_c2c.py
uv run pytest -q tests/test_c2c.py -k 'repairs_selected_variant_after_validation_failure or does_not_best_of_n_score_ok_variants or ignores_legacy_variant_budget_after_first_ok_patch or skips_later_candidates_after_high_quality_patch or implements_only_plan_selected_s2_variant'
```

## 2026-06-13 S3 Artifact Lock Compatibility

S3 gate 的 S2.5 artifact lock 校验原先只检查 `exists=true` 的 lock。旧 artifact 或轻量测试里常只有 `rel_path+sha256`，没有 `exists` 字段，这会导致 patch/contract 被改写后 gate 误判通过。

本次修复：

- `s3_s2_5_artifact_lock_sha256` 对所有带 `rel_path` 且未显式 `exists=false` 的 lock 做 sha256 校验；
- 兼容旧 `s3_candidate_selection.json`；
- 继续跳过明确记录为不存在的 optional lock，例如缺失 snapshot 的旧流程 fallback。

验证：

```text
uv run pytest -q tests/test_validators.py -k 's2_5_artifact_lock'
```

## 2026-06-13 C2C Repo Small-Batch Forward Activation Probe

真实运行暴露出一个高成本问题：静态 activation wiring 能证明 ablation switch 写进 config、forward 代码里也出现了字段名，但不能证明机制真的改变 projector/wrapper/aligner 的计算。S3 才发现 no-op 会浪费 cheap proxy/full train 成本。

本次把内置 C2C forward probe 从“静态接线检查”升级为优先执行 repo 专用小 batch forward：

- `src/auto_research/probes/c2c_activation_forward_probe.py` 现在优先：
  - 在临时 smoke repo 中导入 `rosetta.model.projector.create_projector`；
  - 从 enabled/disabled eval config 读取 `model.rosetta_config` 或 train recipe 风格的 `model`；
  - 构造轻量 projector，不加载 Qwen/tokenizer/完整 RosettaModel；
  - 用 synthetic KV batch 跑 enabled/disabled 两次 forward；
  - 比较 `projector_output.key` / `projector_output.value` 的 sha、max_abs_diff、mean_abs_diff；
  - 如果存在 `gate_logit` / `key_weight` / `value_weight` / `last_*` instrumentation，也纳入 tensor checks。
- 如果真实 forward 不可用，artifact 会明确记录：
  - `probe_type=repo_small_batch_forward_failed_static_trace`；
  - `fallback_reason`；
  - `forward_probe_error`；
  - `static_trace`。
- 默认 hard gate 下，静态接线不再等价于通过；必须观察到真实 forward tensor/routing 字段变化。
- S2.5 repair prompt 仍会收到 `runtime_smoke:mechanism_activation_forward_probe`，并要求修真实 `config -> constructor -> forward -> tensor` 路径。
- 实测 C2C env `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python` 可执行该 probe；未接入 switch 的默认 C2CProjector 会返回 `enabled_disabled_forward_tensors_identical`，符合预期。

验证：

```text
python -m py_compile src/auto_research/probes/c2c_activation_forward_probe.py src/auto_research/code_patch.py tests/test_c2c.py
uv run pytest -q tests/test_c2c.py -k 'builtin_c2c_activation_forward_probe or forward_probe or mechanism_activation'
uv run pytest -q tests/test_c2c.py -k 'code_patch_agent or runtime_smoke or patch_manifest or s2_5'
```

## 2026-06-13 C2C Wrapper-Level Cache Activation Probe

projector 小 batch forward 仍然比真实 S3 少一层：它能证明 projector 自身输出会变，但不能证明 `RosettaModel` wrapper 的 cache projection/injection 路径真的吃到了这个变化。

本次把内置 C2C probe 继续推进到 wrapper-level：

- 在不加载真实 Qwen/teacher 的前提下，构造 fake causal LM 和真实 `RosettaModel`；
- 通过真实 `set_projector_config` / `projector_dict` / `projector.forward` 路径执行一次轻量 cache projection；
- 比较 enabled/disabled 的 projected cache：
  - `wrapper_cache.layer*.key`
  - `wrapper_cache.layer*.value`
- artifact 新增关键字段：
  - `wrapper_probe`
  - `cache_key_diff`
  - `cache_value_diff`
  - `projector_called`
  - `switch_seen_by_forward`
- 当 wrapper probe 可执行时，以 wrapper cache 是否变化作为更强 eligibility 条件；projector 单体变化但 wrapper cache 不变不再等价于通过。
- S2.5 validation 归一化现在信任 probe 明确给出的 `mechanism_observed=false`，避免 artifact 层把 projector-only changed_fields 误判为 passed。

真实 C2C env 回归：

```text
/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python src/auto_research/probes/c2c_activation_forward_probe.py ...
status=1
projector_called=true
cache_key_diff=0.0
cache_value_diff=0.0
failures=[enabled_disabled_wrapper_cache_identical, enabled_disabled_forward_tensors_identical]
```

这符合预期：默认未接入 ablation switch 的 C2CProjector 会被 S2.5 runtime smoke 前置拦截，不再等到 S3。

## 2026-06-13 S2.5 Forward Probe Repair Packet Diagnostics

wrapper-level probe 能发现 no-op 后，还需要让 Codex repair 真正吃到这些证据。否则 Codex 只看到 `runtime_smoke:mechanism_activation_forward_probe failed`，容易做表面修复。

本次把 forward probe 的 tensor 证据压缩进 repair packet：

- `codex_repair_packet.activation_forward_probe_diagnostics` 新增：
  - `switch_config.enabled_value / disabled_value`
  - `projector_called`
  - `switch_seen_by_forward`
  - `cache_key_diff / cache_value_diff`
  - `projector_output_identical`
  - `wrapper_cache_identical`
  - `changed_tensors`
  - `identical_tensors`
  - `repair_focus`
- `validation_failure_feedback.failed_checks[*]` 也附带同一份 `forward_probe_diagnostics`，方便 Codex 从失败 check 摘要直接定位问题。
- `repair_focus` 会根据证据自动提示修复方向：
  - disabled switch 没进 config：修 config materialization / switch polarity；
  - forward 没看到 switch：修 projector/aligner/wrapper forward 分支；
  - projector output identical：修 constructor 参数或 projector forward；
  - projector 变了但 wrapper cache 不变：修 wrapper 传参/cache injection；
  - wrapper cache key/value identical：修真实 cache 写入路径。
- repair instruction 明确要求使用 tensor 名、enabled/disabled sha pair 和 repair_focus，而不是修改 validation probe。

验证：

```text
python -m py_compile src/auto_research/code_patch.py tests/test_c2c.py
uv run pytest -q tests/test_c2c.py::test_code_patch_repair_packet_includes_forward_probe_tensor_diagnostics
uv run pytest -q tests/test_c2c.py -k 'forward_probe or mechanism_activation'
```

## 2026-06-13 Activation Forward Failure Routing As Implementation-Only

forward probe 变严格后，activation no-op/env/import 类失败不能污染 S1/S2 方法尝试次数。它们说明 patch 还没有达到 cheap proxy/full S3 eligible 状态，不是方法已经失败。

本次明确把以下 S2.5/S3 activation failure 归为 `implementation_failure`：

- `repo_small_batch_forward_failed_static_trace`
- `enabled_disabled_forward_tensors_identical`
- `enabled_disabled_wrapper_cache_identical`
- `torch_import_failed`
- `projector_import_failed`
- `small_batch_forward_failed`
- `mechanism_activation_forward_probe_failed`
- `mechanism_activation_wiring_failed`

路由语义：

- 这些失败只回 S2.5/S2 repair；
- `same_direction_failure_count=0`；
- 不写 `plan/direction_scorecard.json`；
- 不进入 shared method memory；
- `performance_feedback.summary.does_not_consume_same_direction_attempt=true`；
- `performance_feedback.candidate_results[*].implementation_failure_signals` 会记录具体失败信号；
- 只有 patch eligible 且 cheap proxy/full train 已经跑出效果差，才算 `method_failure`。

同时保留优先级规则：如果 candidate 已经有 full S3 metrics 或 `delta_vs_baseline`，即使存在 activation/implementation 信号，也按 `method_failure` 处理，因为这时已经不是“没跑起来”，而是“跑起来但效果差”。

验证：

```text
python -m py_compile src/auto_research/orchestrator.py tests/test_pipeline.py
uv run pytest -q tests/test_pipeline.py -k 'implementation_failure or activation_forward_probe_failures or full_metrics_failure_is_method_failure'
uv run pytest -q tests/test_stage_contracts.py tests/test_pipeline.py
```

## 2026-06-13 S2.5 Forward Probe Environment Preflight

uv 测试环境可能没有 torch，但真实 C2C env 有 torch。forward probe 失败时，如果 artifact 不记录运行环境，很难判断是 patch no-op、repo import 问题，还是 Python/env 问题。

本次在 `runtime_smoke:mechanism_activation_forward_probe` 前增加同环境 preflight：

- 使用与 probe 完全相同的 `python_cmd`；
- 在临时 smoke repo 下运行；
- 设置 repo `PYTHONPATH`；
- artifact 新增 `probe_environment`：
  - `probe_python`
  - `using_c2c_env_python`
  - `torch_available`
  - `torch_version`
  - `repo_import_ok`
  - `repo_import_error`
  - `using_builtin_probe`
- `activation_probe_evidence.forward_probe` 和 `activation_forward_probe_diagnostics` 也会携带这份环境信息，方便 Codex repair 和人工审计。

真实 C2C env preflight：

```text
probe_python=/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python
using_c2c_env_python=true
torch_available=true
torch_version=2.6.0+cu124
repo_import_ok=true
```

验证：

```text
python -m py_compile src/auto_research/code_patch.py tests/test_c2c.py
uv run pytest -q tests/test_c2c.py -k 'forward_probe or mechanism_activation'
```

## 2026-06-14 S2.5-Only Implementation Repair Lane

implementation failure 现在不再语义上回到完整 S2 planner。S3 发现 patch 还没达到方法级实验资格时，会写出 `plan/s2_5_repair_dispatch.json`，并让 `S2_plan` 直接跳过 planner、只进入 S2.5 CodePatch repair。

关键约束：

- 同一个 candidate；
- 同一个 `variant_fingerprint`；
- 同一个 Codex persistent session/worktree；
- 输入 `activation_forward_probe_diagnostics`、`tensor_checks`、`patch_manifest`、`changed_files`；
- 只修 config/constructor/wrapper/projector/aligner forward 接通问题；
- 不消耗 5 次同方向方法尝试次数；
- 如果 dispatch 锁定的 candidate / `variant_fingerprint` 已经缺失，则写 `implementation_blocked` 和 `no_valid_patch`，不回完整 S2 planner；
- `patch_manifest.status == ok` 视为 `patch_eligible_for_s3=true`，否则记录 `implementation_blocked=true`。

验证：

```text
python -m py_compile src/auto_research/orchestrator.py src/auto_research/agents/plan.py src/auto_research/code_patch.py tests/test_pipeline.py tests/test_c2c.py
uv run pytest -q tests/test_pipeline.py -k 'implementation_failure or activation_forward_probe_failures'
uv run pytest -q tests/test_c2c.py -k 'implementation_failure_reruns_only_s2_5_patch_repair or implementation_failure_reuses_patch_session'
uv run pytest -q tests/test_stage_contracts.py tests/test_pipeline.py
```

## 2026-06-14 S2.5 Root-Cause Diagnosis Pre-Pass

S2.5-only implementation repair 现在会在真正改代码前，先用同一个 Codex persistent session 做一次根因诊断。这个 pre-pass 仍属于 S2.5，不回 S2 planner，也不改变方法方向。

诊断输入：

- `plan/s2_5_repair_dispatch.json`
- `plan/code_patches/patch_manifest.json`
- 当前 candidate 的 `implementation_contract.json`
- `changed_files`
- `activation_forward_probe_diagnostics`
- `tensor_checks`

诊断约束：

- 不编辑文件；
- 只允许轻量检查，例如 `py_compile`、targeted smoke、forward probe；
- 禁止 full train、大 proxy、distributed job；
- 所有 Python 命令必须优先使用 C2C conda 环境 `c2c.env_python`，例如 `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python`，不能默认用系统 Python；
- 输出写入 `plan/code_patches/<candidate>/repair_diagnosis.json`；
- 后续 patch prompt 会携带 `repair_diagnosis`，要求 Codex 按根因修 config -> rosetta_config -> constructor/wrapper/projector/aligner forward -> tensor/output 路径。

验证：

```text
python -m py_compile src/auto_research/code_patch.py tests/test_c2c.py
uv run pytest -q tests/test_c2c.py -k 'implementation_failure_reuses_patch_session or implementation_failure_reruns_only_s2_5_patch_repair or persistent_validation_repair_uses_same_codex_session'
uv run pytest -q tests/test_c2c.py -k 'repair_packet_includes_forward_probe_tensor_diagnostics or runtime_smoke_repairs_forward_probe_no_effect or runtime_smoke_repairs_missing_mechanism_activation_wiring'
```

## 2026-06-14 S2.5 Repeated Implementation Failure Detection

S2.5 implementation repair 现在会检测连续同类失败，避免 Codex 在同一个错误路径里反复小修。

重复指纹包括：

- 相同 `identical_tensors`，例如 `projector_output` 仍然 enabled/disabled identical；
- 相同 `cache_key_diff` / `cache_value_diff`；
- 连续 `switch_seen_by_forward=false`；
- 连续 `projector_output_identical=true` 或 `wrapper_cache_identical=true`；
- `changed_files` 高度相似。

如果检测到重复失败：

- `implementation_contract.repeated_failure_context.is_repeated=true`；
- diagnosis pre-pass prompt 要求显式说明 repeated signals；
- patch prompt 明确禁止继续普通 same-path repair；
- Codex 必须换修复目标或更底层 wiring boundary，同时保持同一个 candidate / variant；
- validation repair packet 也携带 `repeated_failure_context`，避免后续 repair retry 继续盲修。

验证：

```text
python -m py_compile src/auto_research/code_patch.py tests/test_c2c.py
uv run pytest -q tests/test_c2c.py -k 'implementation_failure_reuses_patch_session or implementation_failure_reruns_only_s2_5_patch_repair or persistent_validation_repair_uses_same_codex_session or repair_packet_includes_forward_probe_tensor_diagnostics or runtime_smoke_repairs_forward_probe_no_effect or runtime_smoke_repairs_missing_mechanism_activation_wiring'
uv run pytest -q tests/test_pipeline.py -k 'implementation_failure or activation_forward_probe_failures'
```

## 2026-06-15 Runtime Smoke GPU Resource Retry

S2.5 runtime smoke 现在不再默认固定 GPU 0。first-batch train smoke 会自动选择空闲显存最多且满足阈值的 GPU，并在 OOM 后换一张未尝试过的空闲 GPU 重试一次。

- 默认 `runtime_smoke.gpu_ids=auto`，`min_free_mb=8192`。
- 如果没有 GPU 满足 `min_free_mb`，会按 `runtime_smoke.resource_wait` 轮询等待，默认最多等 7200 秒，每 120 秒查一次。
- 等待期间不会调用 Codex repair，也不会消耗 S1/S2/S2.5 尝试次数。
- 等待超时或 OOM 换卡后仍无可用 GPU 时，patch manifest 标记为 `retryable_no_valid_patch`，candidate/check 标记：
  - `failure_category=runtime_smoke_resource_retry`
  - `resource_retry=true`
  - `retryable=true`
- Orchestrator 会进入 `retryable_paused`，`pause_type=runtime_smoke_resource_retry`，提示等待 GPU 显存恢复后 resume。
- 只有 dtype/device/valid_mask/first-batch runtime error 等真实代码问题才继续进入 S2.5 implementation repair。

验证：

```text
python -m py_compile src/auto_research/code_patch.py src/auto_research/orchestrator.py src/auto_research/validators/s2_gate.py src/auto_research/reporting.py tests/test_c2c.py tests/test_pipeline.py
uv run pytest -q tests/test_c2c.py -k 'runtime_smoke_gpu_attempts or runtime_smoke_oom_retry or patch_failure_retryable_treats_runtime_smoke_resource_retry'
uv run pytest -q tests/test_pipeline.py -k 'retryable_codex_limit_pauses or runtime_smoke_resource_retry_pauses'
uv run pytest -q tests/test_stage_contracts.py tests/test_pipeline.py
```

## 2026-06-15 C2C Resource-Aware GPU And Batch Policy

C2C S3 执行现在会在真正启动 train/proxy/eval 前重新读取当前 GPU 状态，而不是盲信 S2 `plan.yaml` 中较早生成的 `selected_gpu_ids`。

- 默认 GPU policy：
  - `max_gpus=6`
  - `min_free_mb=8192`
  - `max_utilization_gpu=40`
  - `respect_resource_filters=true`
- 如果 `c2c.small_loop.gpu_ids=auto`，S3 会实时选择空闲显存最多且利用率不高的 GPU。
- 如果用户显式指定 GPU ids，也可以开启 resource filter，只从这些卡里挑当前满足条件的卡。
- 当前真实机器状态下，新策略会避开正在忙的 GPU0/1/2/7，选择 GPU5/6/4/3 这类空闲卡。

训练 recipe 也会做资源感知调整：

- full train 根据选中 GPU 的最小空闲显存选择最大可行 `per_device_train_batch_size`。
- 默认 batch tiers：
  - >=22GB free: batch 4
  - >=16GB free: batch 3
  - >=10GB free: batch 2
  - otherwise: batch 1
- `gradient_accumulation_steps` 默认按 effective batch 反推，尽量保持原始 effective batch 不变。
- `learning_rate` 默认按 effective batch ratio 缩放；如果 effective batch 被保持不变，则 LR 保持不变。
- OOM recovery 继续生成 `train_recipe_memory_safe.json`，但现在会记录：
  - original/effective batch size
  - recovered batch size
  - LR adjustment
  - recovery GPU selection snapshot

这让资源问题不会被误判成 patch 差，同时也避免在别人任务跑到一半时强行抢满所有 GPU。

验证：

```text
python -m py_compile src/auto_research/adapters/runner.py src/auto_research/c2c.py src/auto_research/agents/experiment.py tests/test_c2c.py
uv run pytest -q tests/test_c2c.py -k 'select_gpus_filters_busy_cards or full_train_resource_policy or train_oom_uses_memory_safe_recipe or proxy_batch_auto_uses_gpu_memory'
uv run pytest -q tests/test_c2c.py -k 'c2c_train_oom or c2c_proxy_batch or materialized_train_configs_disable_wandb or runtime_smoke_gpu_attempts or runtime_smoke_oom_retry'
uv run pytest -q tests/test_stage_contracts.py tests/test_pipeline.py
```

## 2026-06-15 S2 GPU Planning Removed

S2 plan 不再选择 GPU，也不再把瞬时服务器资源状态写入 `plan.yaml` / `short_loop_plan.yaml`。

原因：

- GPU 占用是执行时状态，S2 生成计划时的 snapshot 很快过期；
- 旧设计让 `selected_gpu_ids` / `resource_snapshot` / `resource_budget.peak_concurrent_gpus` 在 S2 和 S3 之间容易不同步；
- S2 的职责应该是机制方向、实验协议、acceptance rule 和 failure feedback，不应该承担资源调度。

新的职责划分：

- S2:
  - 生成当前 next variant / candidate idea / hypotheses / acceptance criteria；
  - 生成不含 GPU 卡号、显存快照或资源调度说明的 task graph；
- S2.5:
  - runtime smoke 在真正执行前动态选卡；
  - resource wait / OOM retry 属于 S2.5 validation，不消耗方法尝试次数。
- S3:
  - train/proxy/eval 启动前重新读取当前 GPU 状态；
  - 按空闲显存和利用率选择资源；
  - batch size / gradient accumulation / LR 在 materialize config 时根据真实资源调整。

验证：

```text
python -m py_compile src/auto_research/agents/plan.py src/auto_research/validators/s2_gate.py src/auto_research/stage_contracts.py tests/test_c2c.py tests/test_pipeline.py tests/test_validators.py
uv run pytest -q tests/test_validators.py -k 's2_gate'
uv run pytest -q tests/test_pipeline.py -k 'retryable_codex_limit_pauses or runtime_smoke_resource_retry_pauses'
uv run pytest -q tests/test_c2c.py -k 's2_gate or code_patch_agent_marks_codex_429 or s2_directional_planner or pipeline_runs_to_s3_with_mock_small_loop or s3_executes_only_patch_manifest_selected_candidate'
```

## 2026-06-15 S2 Next-Variant Planner Simplification

S2 从“批量生成/打分多个 variant”简化为“同一 S1 方向内，每轮只提出一个 next variant”。

新的职责边界：

- S1 只定机制方向。
- S2 在同一个 persistent session 内，根据上一轮 method-level S3/proxy 反馈提出一个 `next_variant`。
- S2.5 只实现当前 `candidate_ideas[0]` / `plan.next_variant` 对应的候选。
- S3/proxy/full 结果回传后，下一轮 S2 继续在同一个 session 里生成第二个 next variant。
- implementation failure 不进入 S2 planner，只走 S2.5-only repair lane。
- 5 次同方向 method failure 后，才重置 S2 session 并回 S1 换方向。

删减内容：

- 不再要求 S2 一次性生成 `variant_candidates: [v1, v2, ...]`。
- 不再写 `plan/variant_candidates.json`。
- 不再把 rejected variant pool 作为核心 artifact 暴露给 S2.5。
- 保留 scorer 作为兼容兜底：如果旧模型返回多个 legacy candidates，程序只选一个 `next_variant` 执行。

新产物：

- `plan.next_variant`
- `plan/next_variant.json`
- `plan/candidate_ideas.json` 中只包含当前要进入 S2.5 的一个 candidate。

验证：

```text
python -m py_compile src/auto_research/agents/plan.py src/auto_research/code_patch.py
uv run pytest -q tests/test_c2c.py -k 's2_directional_planner or s2_variant_scorer or s2_resume_planner or implements_only_plan_selected_s2_variant'
```

## 2026-06-15 Runtime Smoke Legacy GPU Override Fix

S2.5 runtime smoke 默认强制按当前空闲显存自动选 GPU，避免旧项目的 `code_patch.validation.runtime_smoke.gpu_ids: [0]` 覆盖新策略。

原因：

- 旧项目配置可能在 S2.5 runtime smoke 里固定 GPU 0；
- 新框架已经把 GPU 选择下沉到执行时动态调度，但 resume 旧项目时 project config 会覆盖默认值；
- 当 GPU 0 忙而其他卡空闲时，runtime smoke 会进入长时间 resource wait，造成“代码修好了但流程不推进”的假象。

新规则：

- C2C 项目默认把 `runtime_smoke.gpu_ids` 归一化为 `auto`；
- 原始固定卡号记录到 `legacy_configured_gpu_ids`，方便审计；
- 只有显式设置 `runtime_smoke.respect_configured_gpu_ids: true` 时，才允许固定 GPU；
- runtime smoke 仍保留 `min_free_mb`、OOM retry 和 2 小时 resource wait。

验证：

```text
uv run pytest -q tests/test_c2c.py -k 'runtime_smoke_gpu_attempts or c2c_runtime_smoke_ignores_legacy_fixed_gpu or c2c_runtime_smoke_can_respect_explicit_fixed_gpu_opt_in or runtime_smoke_oom_retry'
uv run pytest -q tests/test_pipeline.py -k 'runtime_smoke_resource_retry_pauses_without_codex_repair'
```

## 2026-06-15 S3 Cheap Proxy GPU Isolation

S3 cheap proxy 不再复用 full-train 的宽 GPU policy。proxy/baseline/activation smoke 现在有独立 GPU 选择策略，默认只选一张资源干净的 GPU，并禁止回退到忙卡。

原因：

- 真实运行中 cheap proxy 曾选到 `[6,5,4,3,2]`，触发多卡 `torch.distributed` 子进程失败；
- 这种失败不是方法失败，也不应该让 S2.5 盲修 patch；
- cheap proxy 的目标是低成本预筛，默认单卡更稳定，full S3 仍可使用独立的多卡策略。

新规则：

- `c2c.small_loop.proxy_screen.gpu_policy` 默认 `max_gpus=1`；
- proxy GPU selection 使用 `min_free_mb` / `max_utilization_gpu` 过滤忙卡；
- proxy policy 默认 `disable_resource_fallback=true`，无干净 GPU 时不偷偷跑到忙卡；
- proxy GPU 不可用时进入 resource retry/blocking 状态，不消耗方法尝试次数，也不让 Codex 修代码；
- S3 artifact 同时记录 `gpu_selection` 和 `proxy_gpu_selection`；
- proxy command failure 能识别 `torch.distributed.elastic` / `ChildFailedError` / `local_rank`，并保留 rank、stdout/stderr tail。
- S3 preflight 不再继承 full/proxy GPU selection，并为 pytest 加 `--no-cov`，避免代码健康检查占用 GPU 或被 coverage 拖慢。

验证：

```text
python -m py_compile src/auto_research/c2c.py src/auto_research/agents/experiment.py src/auto_research/adapters/runner.py src/auto_research/orchestrator.py
uv run pytest -q tests/test_c2c.py -k 'runner_select_gpus_filters_busy_cards_when_requested or runner_select_gpus_can_disable_busy_card_fallback or c2c_proxy_gpu_policy_defaults_to_single_clean_gpu or c2c_proxy_command_failure_classifies_runtime_errors or c2c_proxy_command_failure_classifies_timeout or c2c_proxy_command_failure_classifies_distributed_child_failure or c2c_proxy_batch_auto_uses_gpu_memory or c2c_materializes_proxy_activation_smoke_ablation_config'
uv run pytest -q tests/test_c2c.py -k 'c2c_proxy_gpu_policy_defaults_to_single_clean_gpu or c2c_materialized_train_configs_disable_wandb_without_service_token or c2c_materializes_proxy_activation_smoke_ablation_config'
```

## 2026-06-16 Proxy Baseline Timeout Fallback

真实 S3 retry 暴露出一个路由 bug：paired proxy baseline 自身的 `proxy_baseline_eval_*` 可能在单卡 eval 上超时，但这不是候选 patch 的 implementation failure。旧逻辑会把它写成 `failed_no_metrics -> implementation_failure`，从而错误触发 S2.5 Codex repair，浪费 token 并污染 patch 修复上下文。

新规则：

- `proxy_baseline_eval_*` / baseline train / baseline preflight 失败时先生成结构化 `baseline_failure`。
- 如果 `allow_configured_baseline_fallback=true` 且项目 baseline 能覆盖 proxy dataset，使用 `configured_full_baseline_subset_fallback` 继续候选 cheap proxy。
- fallback 仍保留 baseline attempts 和 timeout/category 证据，方便后续审计。
- 如果 fallback 不可用，candidate 进入 `proxy_screen.status=baseline_blocked` / `command_status=blocked`。
- `baseline_blocked` 不再被 `_candidate_is_implementation_failure()` 归为 patch implementation failure，因此不会触发 S2.5-only repair。
- `proxy_screen` compact/report 保留 `baseline_status`、`baseline_failure`、`baseline_attempt_count`，人工监控能直接看到是 baseline eval timeout，而不是 patch 崩了。

验证：

```text
python -m py_compile src/auto_research/agents/experiment.py src/auto_research/orchestrator.py
uv run pytest -q tests/test_c2c.py -k 'proxy_baseline_eval_timeout_uses_configured_fallback or replay_proxy_runs_paired_baseline_before_full_training or proxy_command_failure_classifies_timeout'
uv run pytest -q tests/test_pipeline.py -k 'proxy_baseline_blocked_is_not_implementation_failure or s3_full_metrics_repairable_does_not_use_implementation_repair_route or s3_implementation_failure_routes_to_s2_without_consuming_direction_budget'
```

## 2026-06-16 Proxy Eval Timeout Extension

真实 S3 retry 显示 `mmlu-redux limit=64` 的 baseline/candidate proxy eval 在 1200 秒内可能无法完整写出 summary。这个 timeout 不是模型崩溃或 OOM，而是框架达到上限后主动终止 eval 进程，导致 cheap proxy 无法得到 metrics。

调整：

- `DEFAULT_C2C_PROXY_SCREEN.eval_timeout_seconds` 从 1200 秒调到 7200 秒。
- proxy train timeout 暂不改变，仍为 1800 秒。
- proxy 默认仍是单卡 `max_gpus=1`：
  - 之前真实运行中过宽 GPU selection 触发过多卡 `torch.distributed` 子进程失败；
  - 当前 C2C evaluator 命令本身不是 DDP，多给 `CUDA_VISIBLE_DEVICES` 不一定带来 eval 并行收益；
  - full S3 仍可使用独立的多卡策略，cheap proxy 先追求稳定和低成本。
- 如需实验性多卡 proxy，可显式覆盖 `c2c.small_loop.proxy_screen.gpu_policy.max_gpus`，但不作为默认。

验证：

```text
python -m py_compile src/auto_research/c2c.py
uv run pytest -q tests/test_c2c.py -k 'proxy_eval_timeout_default_allows_long_single_card_eval or c2c_proxy_gpu_policy_defaults_to_single_clean_gpu or proxy_command_failure_classifies_timeout'
```

## 2026-06-17 Proxy Metrics Method-Failure Routing

真实 S3 retry 在 timeout 拉长后完整跑完 cheap proxy，并产生了 paired proxy metrics。候选表现为 AI2-ARC 小涨，但 MMLU / OpenBookQA 明显回退，属于 method-level proxy 失败，而不是 implementation failure。

修复：

- 新增 `_candidate_has_proxy_metrics()`。
- 如果 candidate 已有 `proxy_screen.metrics.mean` 或 dataset metrics，且 `decision=proxy_rejected/not_viable`，则不再因为 patch risk labels / test files / training-loop changes 被归为 `implementation_failure`。
- 这类失败会走 method feedback：
  - 计入同方向 proxy/method 尝试；
  - 回同一 S2 session 生成下一 same-direction variant；
  - 不走 S2.5-only patch repair。

验证：

```text
python -m py_compile src/auto_research/orchestrator.py
uv run pytest -q tests/test_pipeline.py -k 'proxy_rejected_with_metrics_routes_as_method_feedback or proxy_baseline_blocked_is_not_implementation_failure or s3_full_metrics_repairable_does_not_use_implementation_repair_route or s3_implementation_failure_routes_to_s2_without_consuming_direction_budget'
```

## 2026-06-17 Per-Candidate S2.5 Repair Budget

真实三轮 S1-S3 跑法暴露出一个 S2.5 自动 repair 路由问题：`mechanism_activation_wiring_failed` 已经能被识别为 implementation failure，但项目之前累计了 8 次 implementation repair，达到 iteration 级预算后，新 variant 的第一次 S2.5 wiring failure 也被挡住，导致 stage 直接 failed。

修复：

- 新增 `implementation_repair_routes_by_candidate`。
- S2.5 validation failure 和 S3 implementation failure 的 repair budget 改为按 `iteration:candidate_id:variant_fingerprint` 计数。
- 旧的 `implementation_repair_routes` 继续保留为人工监控的累计计数，但不再作为新 candidate/variant 的阻断条件。
- feedback / trace / routed result 写入 `repair_route_key`，方便审计是哪一个 candidate 消耗了 repair budget。
- 新 candidate 不会被旧 candidate 的 repair 历史污染；同一个 candidate 仍受 `max_implementation_repair_routes_per_iteration` 保护，避免无限盲修。

验证：

```text
python -m py_compile src/auto_research/orchestrator.py src/auto_research/validators/s2_gate.py
uv run pytest -q tests/test_pipeline.py -k 's2_5_validation_failure_routes_to_patch_only_repair or s2_5_validation_repair_budget_is_per_candidate_variant or proxy_rejected_with_metrics_routes_as_method_feedback'
```

## 2026-06-21 S3 Proxy OOM Resource Retry Lane

真实 S3 cheap proxy 暴露出一个资源路由问题：proxy train 在 GPU 0 上 CUDA OOM 后，旧逻辑把它归为 `repairable_proxy_risk / implementation_failure`，导致 S2.5 Codex 被要求修代码并消耗 per-candidate repair budget。这个失败不是 patch 实现问题，而是 GPU 资源不可用。

修复：

- proxy command OOM 现在写成 `proxy_screen.status=resource_retry`。
- `failure_category=s3_proxy_resource_oom`，`repair_route=resource_retry`，明确不进入 Codex patch repair。
- S3 orchestrator 在看到 resource retry 时进入 `retryable_paused`，`pause_type=s3_proxy_resource_retry`。
- resource retry 不算 method failure，也不算 implementation failure，不消耗 S1/S2/S2.5 尝试次数。
- 旧 artifact 兼容：如果历史 `proxy_screen.status=repairable_proxy_risk` 但 `command_failure.category=resource_oom`，读取时自动 normalize 为 `resource_retry`。

验证：

```text
python -m py_compile src/auto_research/agents/experiment.py src/auto_research/orchestrator.py
uv run pytest -q tests/test_pipeline.py -k 's3_proxy_oom_pauses_as_resource_retry_not_s2_5_repair or s2_runtime_smoke_resource_retry_pauses_without_codex_repair or s3_implementation_failure_routes_to_s2_without_consuming_direction_budget'
uv run pytest -q tests/test_c2c.py -k 'proxy_command_failure_classifies_oom_as_resource_retry or normalize_c2c_proxy_screen_promotes_legacy_oom_to_resource_retry or proxy_command_failure_classifies_runtime_errors or proxy_command_failure_classifies_timeout'
```

## 2026-06-21 Retryable Pause Memory Isolation

真实 S3 resource wait timeout 暴露出一个记忆污染问题：虽然 orchestrator 已经把 GPU resource failure 标为 `retryable_paused / s3_proxy_resource_retry`，但 S3 failure feedback writer 仍会把它包装成 `failure_mode=no_metrics`，写入 `meta/failure.md`、`meta/failure.jsonl`、`meta/negative_memory.jsonl`、`literature/feedback/failed_ideas_round_*.json` 和 `experiment/results/failure_feedback.json`。这会让 S1/S2 误以为方法或实现失败。

修复：

- 新增 retryable/resource feedback detector，识别 `resource_retry`、`s3_proxy_gpu_resource_retry`、`runtime_smoke_resource_retry`、Codex quota/rate-limit pause。
- `build_c2c_feedback_bundle()` 读取历史 feedback 时过滤 retryable pause noise；混合 payload 中只剔除 retryable candidate，保留真实 method evidence。
- `_write_c2c_failure_feedback()` 遇到纯 retryable pause 时只写 `experiment/results/retryable_pause_feedback.json`，不写普通 failure feedback 和 negative memory。
- 旧 artifact 兼容：已经写入的 GPU wait / resource retry 条目不会继续进入 S1 method feedback 或 S2 implementation feedback。

验证：

```text
python -m py_compile src/auto_research/failure_log.py src/auto_research/agents/experiment.py
uv run pytest -q tests/test_failure_log.py -k 'feedback_loader_filters_retryable_resource_pause_noise or feedback_loader_splits_method_and_implementation_views or shared_method_memory_records_only_method_failures'
uv run pytest -q tests/test_pipeline.py -k 's3_proxy_oom_pauses_as_resource_retry_not_s2_5_repair or retryable_paused'
```
