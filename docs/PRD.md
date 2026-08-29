# stellar-index PRD（v0.2 draft，待 review）

> 一句话：面向公版科幻小说的「深读助手」——每个答案带原文引用，并用 1M 长上下文检验 RAG 是否仍然必要。
> English: A citation-grounded deep-reading assistant for public-domain sci-fi novels; compares RAG, 1M-token long context, and self-routing retrieval.
> 状态：v0.2 草案，已补齐检索技术选型，尚未开始编码。

## 1. 背景与问题

DeepSeek V4 提供 1M token 上下文，单次可放入整本小说。这带来一个 2026 年非常真实的产品/架构问题：**RAG 是否还值得做？**

已有研究（arXiv:2407.16833, EMNLP 2024 Industry）发现：资源充足时长上下文平均表现更好，但成本显著更高；约 63% 的查询 RAG 与长上下文预测一致；Self-Route 可节省 39%–65% 成本并保持接近长上下文的性能。本项目把这个结论放到一个可感知的产品里验证：公版科幻小说深读助手。

通用问答在小说场景的典型失败：不引用原文、跨章节人物关系漂移、把同人二设当原作。本产品强制每个答案绑定原文证据。

## 2. 目标用户

1. **学生/研究者**：读科幻小说、写论文需要定位原文。
2. **同人创作者/设定考据党**：需要快速核查人物、时间线、地点与主题。
3. **AI 应用架构师**：关心 RAG vs 长上下文的成本/质量边界。

## 3. 目标 / 非目标

### 目标（v1）

- 收录 ≥ 20 本 Project Gutenberg 公版科幻小说，全部文本与索引可重建。
- 提供标准 hybrid RAG 检索栈（BM25 + dense + cross-encoder rerank）。
- 支持至少 4 种问答模式：naive RAG、hybrid RAG、full-book long-context、Self-Route 混合。
- 提供 ≥ 100 道黄金 QA（人工校验），覆盖单章事实、跨章关系、主题论证。
- 每个答案带 citation span，可跳转到原文。
- 产出 RAG vs Long-context 实验报告与可视化「星图」书库。

### 非目标（v1）

- 不微调生成模型，也不微调检索模型（微调边界属于 P1）。
- 不做多语种语料；v1 语料以英文公版书为主，界面中英双语。
- 不建设用户账号系统；local-first 单人部署。
- 不做版权存疑的当代小说/漫画/游戏文本。

## 4. 成功指标（v1 验收，M2 先导后校准一次）

| 指标 | 初始目标 |
|---|---|
| 语料规模 | ≥ 20 本，重建命令公开 |
| 检索质量（hybrid+rerank, test set） | Recall@10 ≥ 0.85；nDCG@10 ≥ 0.75；MRR@10 ≥ 0.70 |
| 黄金 QA | ≥ 100 道，每道含标准答案与证据章节；20 dev / 80 test 分离 |
| 引用正确率 | citation precision ≥ 0.85（正确引用 / 总引用） |
| 事实正确率 | 单章事实题 ≥ 0.85，跨章题 ≥ 0.70 |
| 实验完整性 | 至少 5 种检索/上下文配置 × 同一 test set 出报告 |
| Demo | 一个 GIF 展示「提问 → 证据高亮 → 星图跳转」 |
| 文档 | PRD、ADR、BENCHMARK.md 完整 |

## 5. 功能需求

### 5.1 语料与数据管线

- 数据源：Project Gutenberg Science Fiction bookshelf，仅收公版作品。
- 下载脚本带缓存与断点续传；仓库不直接提交整本书文本，只提交书目清单与 `make corpus` 命令。
- 每本书处理为：元数据、章节切分、chunk 索引、实体索引（人物/地点/概念）。
- 实体抽取：DeepSeek V4 结构化输出（strict JSON schema），人工抽检 ≥ 20 本中的 10%。

### 5.2 Chunking 方案（已确定）

**Chapter-aware + sentence-boundary + parent-child。**

| 项 | 决策 |
|---|---|
| 句子切分 | `spaCy` `en_core_web_sm`（约 12MB），只用于英文句子边界 |
| 章节约束 | chunk 永不跨章节 |
| child chunk | 完整句子合并至 **200–300 tokens（目标 256）**，不切句 |
| parent chunk | 同一上下文中合并至 **500–700 tokens**，作为生成上下文 |
| 检索-生成关系 | 用 child 检索，命中后把对应 parent 送入生成 |
| 元数据 | `book_id, chapter_id, parent_id, sentence_span, char_offsets, token_count` |
| chunk 消融 | M2 在 dev 20 题上比较 128 / 256 / 512 三档，以 Recall@10 + citation precision 选默认值 |
| 重叠 | child 之间不做滑动窗口重叠；依赖 parent 覆盖边界信息 |

### 5.3 检索技术选型（已确定）

| 层 | 默认方案 | 说明 |
|---|---|---|
| Sparse | BM25 | `SQLite FTS5`，零额外模型 |
| Dense | `BAAI/bge-small-en-v1.5` | 33M 参数，384 维，fp32 约 134MB；batch ≤ 32，CPU/MPS |
| 多语言查询 | 非英文 query 先用 DeepSeek 转写/翻译为英文 | 避免引入 2GB+ 多语言 embedding 模型 |
| 融合 | RRF（BM25 + dense 各取 top 100） | k=60，融合后取 top 50 进入 rerank |
| Rerank | `BAAI/bge-reranker-base` cross-encoder | 278M 参数，约 1.1GB；batch ≤ 16，CPU/MPS |
| Rerank 降级 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 约 90MB；配置项 `reranker=fast` |
| 实验对照臂 | DeepSeek LLM rerank（pointwise + 结构化理由） | 与 cross-encoder 对比质量/成本 |
| 向量库 | **LanceDB（embedded）** | 本地零服务；表：`chunks`、`books`、`qa`；向量 384 维 |
| 无模型降级 | BM25-only | CI 与 `--offline` 模式，不下载任何权重 |

**为什么 LanceDB 而不是 Chroma/Qdrant/FAISS**：20 本书规模不需要服务端；LanceDB embedded、Arrow-native、元数据过滤清晰、clone 后单进程可复现；Qdrant/Chroma 会引入服务与更多依赖，FAISS 不是数据库、元数据管理要自己写。

**为什么 bge-small-en-v1.5 而不是更大模型**：本机 M1 16GB，语料是英文；该模型 134MB、质量对书籍级检索足够。中文 query 通过 DeepSeek 转写为英文，绕开多语言 embedding 的内存成本。

**为什么 bge-reranker-base 而不是 LLM rerank 作为默认**：本地推理零 API 成本、确定性更强、延迟稳定；LLM rerank 作为实验臂回答「贵方案是否更准」。

### 5.4 问答模式

| 模式 | 实现 | 适用 |
|---|---|---|
| `rag-naive` | BM25 top-k 直接生成 | 廉价基线 |
| `rag-hybrid` | BM25 + dense + cross-encoder rerank（默认） | 默认生产路径 |
| `long-context` | 整本/多章直接入 1M 上下文 | 质量上界参照 |
| `chapter-scoped` | 先检索定位章节，再完整读该章 | 中间态 |
| `self-route` | 先走 hybrid RAG，模型自评 answerable 才提交，否则升级章节级或整本上下文 | 产品推荐 |

- 所有 RAG 模式输出必须带 `quote` 与 `chapter/paragraph` 定位；长上下文模式由 checker 回验。
- 查询改写/拆解：跨章多跳问题允许 agentic retrieval，v1 限制最多 3 次检索迭代。

### 5.5 黄金 QA 集

- 100 道 v0，分三类：single-chapter factual（40）、cross-chapter relation（40）、thematic argument（20）。
- **切分**：20 dev（用于 chunk/prompt 调参）/ 80 test（held-out，只跑最终报告）。
- 生成方式：DeepSeek 先基于原文起草，作者逐题人工校验；每题记录标准答案、证据章节、可接受答案列表。
- 事实题尽量可规则化判定；主题题用固定 rubric 的 DeepSeek judge。

### 5.6 应用 UI（v0：Streamlit）

- 首页：星图书库（Plotly 力导向图：书-人物-概念）。
- 问答页：答案、citation 高亮、证据跳转、模式选择与成本实时显示。
- 实验页：检索/上下文配置对比图表。
- 视觉主题：科幻星图，双语界面。

## 6. 评估指标（已确定）

### 6.1 检索质量（对 dev/test 的 evidence chunk 计算）

- Recall@5 / Recall@10
- Precision@5
- Hit@1（金标 evidence chunk 是否排第一）
- MRR@10
- nDCG@10

### 6.2 生成质量

- Factual exact-match / normalized contains / key-fact F1
- **Citation precision**：正确引用数 / 总引用数（quote 必须能在声明位置找到）
- **Citation recall**：答案所需证据点被引用的比例
- **Groundedness/faithfulness**：逐 claim 检查是否有引用支持（DeepSeek judge，0/1）
- **Answer relevance**（DeepSeek judge）
- 主题题 rubric 1–5：论证完整性、证据支撑、对原文的忠实度

### 6.3 效率与产品指标

- input/output tokens、估算成本（flash/pro 分档）
- latency：总时延、retrieval、rerank、generation 分段
- Self-Route：升级率、RAG answerable rate、因升级挽回的错误数
- 失败分类：no-evidence、wrong-evidence、quote-mismatch、reasoning error、judge-tie

### 6.4 引用回验器（citation checker）

1. 规范化后检查 quote 是否出现在模型声明的 chapter/paragraph。
2. 若声明位置不匹配但全书存在该 quote，记 `location-error`；完全不存在记 `quote-fabrication`。
3. 支持度：DeepSeek judge 判断 claim 是否被引用文本支持，与引用位置检查分开报告。

## 7. 技术架构（v0）

- Python 3.13 + `uv`；检索库：`rank_bm25`（可选）或 SQLite FTS5、`sentence-transformers`、`lancedb`、`spacy`。
- 本地模型总预算约 1.3GB（embedding 134MB + reranker 1.1GB + spaCy 12MB），峰值内存可容纳于 M1 16GB。
- 结果与索引用 LanceDB + SQLite；UI 用 Streamlit + Plotly。
- 实验记录：每次 QA 运行保存 JSONL traces + token 用量；报告自动生成 `docs/BENCHMARK.md`。
- CI 与 `--offline`：BM25-only，不下载权重，跑 fixtures 中的 2 本 mini 书 + 10 道 QA。

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| Project Gutenberg 下载限流 | 脚本串行下载、缓存、可指定镜像；仓库自带 2 本 mini fixtures 保证 CI |
| 1M 长上下文实验成本/时延 | 只对 ≤ 2 本书做全量 long-context 臂；按 flash/pro 分档记录 |
| bge-reranker-base 在 M1 上偏慢 | batch ≤ 16、只 rerank top 50；提供 90MB fast reranker 配置 |
| 引用定位作弊 | checker 独立回验 quote；不匹配按失败分类披露 |
| 本地 embedding 内存 | 模型与 batch 上限写死；默认 CPU，MPS 可切换 |
| DeepSeek judge 偏差 | 事实题规则判定优先；judge 单独列示并抽检 |
| 公版文本中的 PG 头尾说明污染 | 清洗脚本去除固定 header/footer，保留清洗规则与抽样报告 |
| dev 过拟合 | dev/test 分离；chunk 与 prompt 只用 dev，最终报告只跑 test |

## 9. 里程碑

- M0（当前）：PRD/ADR review。
- M1：书目确定、corpus 下载/清洗、chapter-aware parent-child chunking。
- M2：hybrid retrieval + cross-encoder rerank 闭环、citation checker、20 道 dev QA 先导、chunk 消融。
- M3：100 道 QA、四种问答模式跑通、完整指标与 BENCHMARK.md。
- M4：Streamlit 星图 UI、demo GIF、README/PRD 定稿、发布。

## 10. 仓库结构（草案）

```text
stellar-index/
  README.md
  LICENSE
  pyproject.toml
  docs/PRD.md docs/ADR/ docs/BENCHMARK.md
  src/stellar/
    corpus/       # download/clean/chunk
    retrieval/    # bm25, dense, hybrid, rerank
    qa/           # rag / longctx / self-route
    evals/        # golden QA runner, citation checker, metrics
    ui/           # Streamlit app
  data/books.toml # 书目清单
  fixtures/       # 2 本 mini 书 + 10 道 QA，CI 用
  experiments/    # 配置与结果
  tests/
  .github/workflows/ci.yml   # offline fixtures，不需要 API key
```

## 11. Definition of Done（v1）

- [ ] `make corpus` 能重建 20 本书索引（chunks 写入 LanceDB + SQLite）。
- [ ] 100 道黄金 QA 可一键运行，dev/test 分离，结果写入 `docs/BENCHMARK.md`。
- [ ] 检索指标六项、生成质量六项、效率指标全部产出。
- [ ] 每个 RAG 答案都有 checker 验证过的 citation。
- [ ] Streamlit demo 本地可跑，GIF 进 README。
- [ ] `--offline` 模式可跑通，不下载权重、不调 API。
- [ ] 不依赖本作品集任何其他仓库。
