# ToolRank

> 面向智能合约漏洞检测器的"确定性、可溯源"调度系统。

输入一份 Solidity 合约，ToolRank 自动决定该跑 20 个检测器（Slither、Mythril、
Confuzzius、Vulhunter、Mando-HGT……）中的哪几个、**为什么**选、以及**哪些漏洞
类别仍覆盖不到**。每条推荐都附带证据——pipeline 内部计算的指标，或从论文知识库
检索到的段落——并且必须通过 10 项校验才允许真正去跑工具。

## 为什么做这个

- 全部 20 个工具一起跑：算力浪费、报告冲突、谁对谁错难判。
- 只跑一个：漏掉整类漏洞（Slither 弱于 bad_randomness，Confuzzius 在
  access_control 覆盖率不到 5%）。
- 凭经验选：不可复现、不可审计、新人接不住。

ToolRank 把"选工具"变成一条**显式、可追溯的流水线**：每步都有名字、有输入输出、
有引用列表。

## 系统架构

```
                       contract.sol
                            │
                            ▼
  [1] Scene Match       ─→  匹配 5 个最相似的基准场景
  [2] Diagnostics       ─→  recall coverage / 认证 / 归属面板
  [3] Evidence Packet   ─→  主工具 + 弱项分区 + DACE 检索目标
  [4] DACE-RAG          ─→  枚举 3 个候选 action × 4 列证据槽
                            （FOR/AGAINST/COMPARE/GAP），填内部证据
                            + 检索约 27 段论文 passage
  [5] CEGO              ─→  LLM 在 26 条规则约束下挑一个 action
  [6] Checker           ─→  10 项子检查；不过 → 退回 CEGO 改写
  [7] Execute + Fuse    ─→  跑选中的工具组合，合并报告
                            │
                            ▼
                    fused_report.json（含完整审计链）
```

**除 CEGO 一步外全部确定性可复现。** CEGO 的输出必须通过 10 项 Checker 才能进入
执行阶段，LLM 错了也跑不起来。

## 快速开始

```bash
python -m pip install -e ".[dev]"

# 配置你自己的 OpenAI 兼容端点（自行填入 URL 和 key）
export OPENAI_API_KEY=...              # chat LLM 的 key
export TOOLRANK_OPENAI_BASE_URL=...    # chat LLM 的 base URL
export TOOLRANK_EMBEDDING_API_KEY=... # embedding 的 key
export TOOLRANK_EMBEDDING_BASE_URL=...# embedding 的 base URL

# 一键端到端（推荐 + 执行 + 融合）
toolrank recommend path/to/Contract.sol --execute --emit summary
```

融合报告写到 `LAKES_out/<合约名>/fused_report.json`。加 `-x` / `--explain` 可把
每个 stage 的中间状态全部打印出来，便于演示和审视。

## CLI

```bash
toolrank recommend <合约.sol> [选项]   # 主流程
  --execute            额外执行选中的工具并合并报告
  --emit summary|json  终端输出格式
  -x, --explain        打印每个 stage 的细节
  --no-retrieval       关闭 RAG 检索（消融）
  --jobs N             工具并行度（默认全部核）

toolrank kb-extract <论文目录>    # 从论文目录抽取调度知识库
toolrank kb-audit <论文目录>      # 校验 KB 完整性
toolrank kb-vector-build         # 为 passage_store 构建向量索引
toolrank refresh-kb              # 用 raw 报告重建 performance_db
```

`toolrank <命令> --help` 查看完整选项。

## 核心概念

| 术语 | 含义 |
|---|---|
| **Scene** | 知识库里的一个基准切片，用于查找类似合约下工具的历史表现 |
| **R_hat** | 每个 (工具, 漏洞类) 在历史数据上的召回覆盖率 |
| **Confirmed-weak** | 主工具在主场景下 R_hat < 0.3 且样本量 ≥ 10，"确认弱" |
| **DACE action** | 三选一：`run_robust_single` / `plan_tool_composition` / `stop_with_gaps` |
| **FOR/AGAINST/COMPARE/GAP** | 每个 action 的四列证据槽：赞成、反对、横向对比、已知缺口 |
| **Passage** | 论文段落证据，KB 抽取时已打好 `owner_tool`、`category`、`relation_to_owner` 等结构化标签 |
| **Ownership panel** | 每类漏洞由哪个工具负责；找不到合适工具则标记 `gap`（显式承认未解决） |

## 配置

所有端点都是 OpenAI 兼容接口，**base URL 和 key 都由你自己提供**，仓库不内置任何
服务商地址。

| 环境变量 | 用途 |
|---|---|
| `OPENAI_API_KEY` | chat LLM 的 API key |
| `TOOLRANK_OPENAI_BASE_URL` | chat LLM 的 base URL（默认本地 `http://127.0.0.1:8317/v1`） |
| `TOOLRANK_EMBEDDING_API_KEY` | embedding 端点的 API key（缺省回退 `OPENAI_API_KEY`） |
| `TOOLRANK_EMBEDDING_BASE_URL` | embedding 端点的 base URL（必填，OpenAI 兼容） |
| `TOOLRANK_EMBEDDING_MODEL` | embedding 模型名（默认 `Qwen/Qwen3-Embedding-8B`，内置索引按此构建） |
| `TOOLRANK_SMARTBUGS_DIR` | 显式指定 SmartBugs 位置；否则自动发现 |
| `TOOLRANK_RAG_STRICT_ERRORS` | 设为 `1` 时 RAG 检索失败直接抛错（默认静默降级） |

外部分析器（Securify2、GPTScan、Sailfish、Smartian）的安装路径同样通过
`TOOLRANK_*` 环境变量指定，详见 `toolrank/runner.py`。

## Docker 真实执行

镜像内置 ToolRank + SmartBugs + 一个内部 Docker 守护进程（Docker-in-Docker），
可直接跑 SmartBugs 驱动的 16 个工具（slither、mythril、oyente、osiris、conkas、
confuzzius、honeybadger、maian、manticore、sfuzz、smartcheck、solhint、securify、
vandal、mando-hgt、vulhunter），无需在宿主手工安装它们。

构建：

```bash
docker build -t toolrank .
```

**快速验证真实执行**（只跑某个工具，不需要 LLM）：

```bash
mkdir -p out
docker run --rm --privileged \
  -v toolrank-docker-cache:/var/lib/docker \
  -v "$PWD/out:/work/out" \
  --entrypoint bash toolrank -lc '
    dockerd >/var/log/dockerd.log 2>&1 & \
    for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 1; done; \
    python -m toolrank.runner examples/Reentrancy.sol /work/out \
      --tools slither --primary_tool slither --timeout 600'
# 结果：out/LAKES_out/Reentrancy/raw/slither/result.json
```

**完整推荐 + 执行**（额外需要你自己的 LLM/embedding 端点，供 CEGO 与检索使用）：

```bash
docker run --rm --privileged \
  -v toolrank-docker-cache:/var/lib/docker \
  -v "$PWD/out:/work/out" \
  -e OPENAI_API_KEY=... -e TOOLRANK_OPENAI_BASE_URL=... \
  -e TOOLRANK_EMBEDDING_API_KEY=... -e TOOLRANK_EMBEDDING_BASE_URL=... \
  toolrank recommend examples/Reentrancy.sol --execute --emit summary --results-root /work/out
```

要点：

- **`--privileged` 必需**：镜像内运行自带的 dockerd 来拉起各工具容器（Docker-in-Docker）。
  这样 SmartBugs 与工具容器共享同一文件系统命名空间，避免挂宿主 socket 时的路径不匹配。
- **镜像缓存**：命名卷 `-v toolrank-docker-cache:/var/lib/docker` 让首次拉取的工具镜像
  （如 `smartbugs/slither:0.11.3`）跨运行持久化；首次运行会下载，较慢。
- **跨架构**：部分工具镜像为 `linux/amd64`；Apple Silicon 走 QEMU 模拟，可跑但较慢。
- **securify2 / gptscan / sailfish / smartian**：尚未纳入本镜像（见仓库 Phase 2 计划）。

## 审计链

每次推荐都自带可复盘字段：

- `certification.reason_codes` —— 主工具为何拿到当前认证状态
- `evidence_packet.dace_rag_focus` —— 每个 hedge 工具为哪类漏洞补漏、走哪一层选出
- `action.evidence[slot].refs` —— 每条证据指向内部证据卡 (`ev_*`) 或论文 passage (`p_*`)
- `checker.sub_checks` —— 10 个 boolean 子检查结果
- `category_decisions` —— 每个漏洞类的最终负责工具 + 支撑 ref ID 列表

`ev_*` 解析到 pipeline 确定性状态；`p_*` 解析到 `toolcards/passage_store.json` 里的
具体 passage。**Checker 会拒绝引用了 prompt 中未出现 ref ID 的决策**——LLM 编不出 ID。
