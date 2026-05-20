# ToolRank

> 面向智能合约漏洞检测器的"确定性、可溯源"调度系统。

输入一份 Solidity 合约，ToolRank 会自动决定该跑 20 个检测器（Slither、Mythril、
Confuzzius、Vulhunter、Mando-HGT……）中的哪几个、**为什么**选这几个、以及
**哪些漏洞类别还是覆盖不到**。每一条推荐都附带具体证据 —— 要么是 pipeline 内部
计算出的指标，要么是从论文知识库检索到的段落 —— 并且必须通过 10 项校验才允许
真正去跑工具。

---

## 为什么做这个

智能合约检测工具之间存在显著的互补盲区：

- 全部 20 个工具一起跑，**算力浪费，报告冲突，谁对谁错难以判定**。
- 只跑一个，**漏掉整类漏洞**（比如 Slither 在 bad_randomness 上很弱，
  Confuzzius 在 access_control 上覆盖率不到 5%）。
- 凭经验选工具，**不可复现、不可审计、新人接不住**。

ToolRank 把"选工具"这件事变成一条 **显式、可追溯的流水线**：每一步都有名字、
有输入、有输出、有引用列表。

---

## 系统架构

```
                       contract.sol
                            │
                            ▼
       ┌───────────────────────────────────────────────────────────┐
       │                                                           │
  [1] Scene Match       ─→  匹配 5 个最相似的基准场景              │
  [2] Diagnostics       ─→  recall coverage / 认证 / 归属面板      │
  [3] Evidence Packet   ─→  主工具 + 弱项分区 + DACE 检索目标      │
  [4] DACE-RAG          ─→  枚举 3 个候选 action × 4 列证据槽      │
                            FOR / AGAINST / COMPARE / GAP，         │
                            填充内部证据 + 检索约 27 段论文 passage │
  [5] CEGO              ─→  LLM 在 26 条规则约束下挑一个 action    │
  [6] Checker           ─→  10 项子检查；不过 → 退回 CEGO 改写 ────┘
  [7] Execute + Fuse    ─→  跑选中的工具组合，合并报告
                            │
                            ▼
                    fused_report.json
                    （含完整审计链）
```

**除 CEGO 一步外全部确定性可复现。** CEGO 的输出必须通过 10 项 Checker
才允许进入执行阶段，LLM 错了也跑不起来。

---

## 快速开始

可编辑模式安装：

```bash
python -m pip install -e ".[dev]"
```

配置一个 LLM 端点（任选其一）：

```bash
export OPENAI_API_KEY=sk-...
# 或
export SILICONFLOW_API_KEY=sk-...
# 或
export WHATAI_API_KEY=...
```

一键端到端跑（推荐 + 执行 + 融合）：

```bash
toolrank recommend path/to/Contract.sol --execute --emit summary
```

融合后的报告写到 `LAKES_out/<合约名>/fused_report.json`。

---

## 演示 / 检视模式

加上 `-x`（或 `--explain`）会把**每个 stage 的中间状态**全部打印到终端。
适合演示、调试，也方便审视"这条 pipeline 不是黑盒"：

```bash
toolrank recommend path/to/Contract.sol -x --emit summary
```

实际输出片段（节选）：

```
[Evidence Packet]
focus=3
  primary_tool: confuzzius
  confirmed_weak: [access_control, bad_randomness, denial_of_service]
  low_support:    [short_addresses]
  dace_rag_focus:
    · vulhunter / access_control: hedge_for_confuzzius_gap_via_top_scene
    · solhint   / bad_randomness: hedge_for_confuzzius_gap_via_top_scene
    · mando-hgt / denial_of_service: hedge_for_confuzzius_gap_via_top_scene

[DACE-RAG]
actions=3 legal=3
  ✓ action=plan_tool_composition  type=PLAN_COMPOSITION
      FOR     (5):
        - confuzzius ranks #1 with S_scene=1.000  [refs: ev_scene_confuzzius]
        - No single whitelist tool detected all DASP categories... [refs: p_smartbug2.0 paper_3]
        ...
      AGAINST (3):  ...
      COMPARE (7):  ...
      GAP     (4):  ...

[Checker]
attempt=1 status=ACCEPT
  all 10 sub-checks passed

[LLM Reason]
selected_tools: confuzzius, vulhunter
gaps: bad_randomness, denial_of_service, short_addresses
```

---

## 目录结构

```
toolrank_project/
├── toolrank/                 主包（32 个模块）
│   ├── cli.py                Typer 命令面
│   ├── engine.py             主流程编排（run_recommendation）
│   ├── scene_pool.py · scene_scoring.py · contract_profile.py
│   │                         输入理解
│   ├── rcov.py · certification.py · precision_gate.py ·
│   │   ownership_evidence.py · assignment_evidence.py · stress.py
│   │                         诊断
│   ├── evidence_packet.py · category_candidates.py · feasibility.py
│   │                         Step1 证据包构建
│   ├── dace_rag.py           action 枚举 + 检索协调
│   ├── passage_store.py · vector_store.py · retrieval.py
│   │                         passage 存储 + 向量/词法混合检索
│   ├── cego.py               LLM 调度 prompt（26 条规则，约 1400 行）
│   ├── openai_compat.py      OpenAI 兼容 LLM 客户端
│   ├── checker.py            10 项校验子检查
│   ├── execution.py · runner.py · runner_adapter.py
│   │                         工具实际执行层
│   ├── fusion.py · report_parser.py
│   │                         多工具报告融合
│   ├── schemas.py · schemas_v2.py
│   │                         跨 stage 共享的 Pydantic 类型
│   ├── dataset_kb.py · solc_range.py
│   │                         KB 加载 + Solidity 版本工具
│   └── kb_extract/           Step 2：调度导向的 KB 抽取
│       │                     （15 个模块，论文 → passage_store）
│       ├── pipeline.py
│       ├── relation_first.py · passage_first.py
│       ├── scheduling_projection.py · layer1_projection.py
│       ├── materializer.py · normalizer.py
│       ├── alias_registry.py · tool_whitelist.py
│       └── models.py · llm_json.py · audit.py · ledger.py · assets.py
│
├── toolcards/                20 个工具卡 + performance_db.json +
│                             passage_store.json + vector_index/
└── pyproject.toml
```

---

## 核心概念

| 术语 | 含义 |
|---|---|
| **Scene** | 知识库里的一个基准切片；ToolRank 用它查找类似合约下工具的历史表现 |
| **R_hat** | 每个 (工具, 漏洞类) 在历史数据上的召回覆盖率 |
| **Tier-1 / Tier-2** | Tier-1 只看主场景判主工具弱不弱；主场景没数据 → Tier-2 退到跨数据集聚合 |
| **Confirmed-weak** | 主工具在主场景下 R_hat < 0.3 且样本量 ≥ 10，"确认弱" |
| **Low-support** | 主工具在主场景下样本量 < 10，"样本不足无法本地判定" |
| **DACE action** | 三选一：`run_robust_single` / `plan_tool_composition` / `stop_with_gaps` |
| **FOR / AGAINST / COMPARE / GAP** | 每个 action 的四列证据槽：赞成、反对、横向对比、已知缺口 |
| **Passage** | 论文段落证据，在 KB 抽取时就打好了 `owner_tool`、`category`、`relation_to_owner`、`knowledge_kind` 等结构化标签 |
| **Ownership panel** | 每类漏洞由哪个工具负责；找不到合适工具 → 标记为 `gap`（显式承认未解决） |
| **Hedge tier** | 补漏工具的选择按三级回退：top-scene → near-scenes → 全聚合，最终命中的层级写在 `dace_rag_focus.reason` 里 |

---

## CLI

```bash
toolrank recommend <合约文件.sol> [选项]      # 主流程
  --execute                  额外执行选中的工具并合并报告
  --emit summary|json        终端输出格式
  -x, --explain              打印每个 stage 的细节（演示友好）
  --no-retrieval             关闭 RAG 检索（消融）
  --jobs N                   工具并行度（默认全部核）

toolrank kb-extract <论文目录>    # Step 2：从论文目录抽取 KB
toolrank kb-audit <论文目录>      # 校验 KB 完整性
toolrank kb-vector-build         # 为 passage_store 构建向量索引
toolrank kb-vector-query <文本>   # 调试：查询向量索引
toolrank refresh-kb              # 用 raw 报告重建 performance_db
```

`toolrank <命令> --help` 查看完整选项。

---

## 直接调用 runner

跳过推荐器，手工指定工具组合直接执行：

```bash
python -m toolrank.runner path/to/Contract.sol \
  --tools slither,osiris \
  --primary_tool slither \
  --tool_categories osiris:ARITHMETIC
```

---

## 配置

| 环境变量 | 用途 |
|---|---|
| `OPENAI_API_KEY` / `SILICONFLOW_API_KEY` / `WHATAI_API_KEY` | LLM 端点（任意配一个即可） |
| `TOOLRANK_SMARTBUGS_DIR` | 显式指定 SmartBugs 位置；否则会自动发现 |
| `TOOLRANK_RUNNER_SCRIPT` | 覆盖默认的检测器 runner 脚本 |
| `TOOLRANK_RAG_STRICT_ERRORS` | 设为 `1` 时 RAG 检索失败直接抛错（默认静默降级） |

---

## 审计链

每次推荐都自带足够的字段供复盘和审计：

- `certification.reason_codes` —— 主工具为什么拿到当前认证状态
  （比如 `FEASIBLE_TOOL → STABLE_TOP1 → BIAS_RISK_ACCEPTABLE →
  INSUFFICIENT_EVIDENCE`）
- `evidence_packet.dace_rag_focus` —— 每个 hedge 工具是为哪类漏洞补漏，
  通过哪一层（`top_scene` / `near_scene` / `aggregate_fallback`）选出
- `action.evidence[slot]` —— 每条证据都有 `refs: [ev_*, p_*]` 指向内部证据卡
  或论文 passage ID
- `checker.sub_checks` —— 10 个 boolean 结果，逐项检查
- `category_decisions` —— 每个漏洞类的最终负责工具 + 支撑它的 ref ID 列表

`ev_*` 引用解析到 pipeline 确定性状态；`p_*` 引用解析到
`toolcards/passage_store.json` 里的具体 passage。**Checker 会拒绝任何引用了
prompt 里没出现过的 ref ID 的决策** —— LLM 想编 ID 编不出来。

---

## 关键数字

| | |
|---|---|
| 工具卡 | 20 个（feasibility 视合约 Solidity 版本而定） |
| 漏洞类别 | 10（DASP10） |
| 场景近邻 | 5（1 个主场景 + 4 个邻近场景） |
| CEGO prompt 区块 | 12 |
| CEGO prompt 规则 | 26 |
| Checker 子检查 | 10 |
| 测试用例 | 436 |
| 单次跑检索回来的 passage 数 | 约 27 |
