# ADR-001：检索与上下文策略（v0.2）

- 状态：草案，待 review
- 日期：2026-08-30
- 版本变更：v0.2 补齐 embedding、reranker、向量库具体选型。

## 背景

DeepSeek V4 拥有 1M token 上下文，整本小说可以直接放入。项目需要决定产品默认检索策略，以及「RAG vs 长上下文」在 v0 的实现边界。

## 决策

1. **产品默认生产路径：`rag-hybrid`（BM25 + bge-small-en-v1.5 dense + bge-reranker-base cross-encoder）+ Self-Route 升级。**
2. **长上下文只作为对照臂，不作为默认产品路径**，v0 全量实验限制在 2 本书以内。
3. **CI/offline 路径为 BM25-only**，不下载任何权重。

理由：

- 已有证据（arXiv:2407.16833）显示：资源充足时长上下文质量更高，但 RAG 成本显著更低，且 63% 查询两者预测一致；Self-Route 可在质量接近长上下文的同时节省 39%–65% 成本。产品默认应当走「便宜路径优先、不确定升级」。
- 20 本书的小说 QA 中，单章事实题占多数；hybrid RAG 在成本、延迟、citation 可控性上优于整本长上下文。
- 本机 M1 16GB 可稳定承载默认检索栈（embedding 134MB + reranker 1.1GB + spaCy 12MB），无需 GPU 服务器。
- DeepSeek 官方没有 embedding API；使用本地开源 embedding/reranker 不违反「云端 LLM 只用 DeepSeek」的约束。

## 具体技术决策

| 层 | 决策 | 理由 |
|---|---|---|
| Chunking | chapter-aware + sentence-boundary + parent-child；child 200–300 tokens，parent 500–700 tokens | 不切句、不跨章；child 保证检索精度，parent 保证生成上下文完整 |
| Sparse | SQLite FTS5 BM25 | 零模型、可复现、易调试 |
| Dense | `BAAI/bge-small-en-v1.5`，384 维，约 134MB | 英文书籍检索够用；M1 友好；中文 query 由 DeepSeek 转写为英文 |
| Fusion | RRF k=60，各取 top 100 融合后 top 50 | 避免调权过拟合 dev |
| Rerank | `BAAI/bge-reranker-base` cross-encoder，batch ≤ 16；fast 降级 `cross-encoder/ms-marco-MiniLM-L-6-v2` | 本地确定性、零 API 成本；fast 路径应对低内存 |
| LLM rerank | DeepSeek pointwise 结构化评分，作为实验臂 | 回答「更贵是否更准」 |
| Vector DB | LanceDB embedded | 20 本书规模无需服务端；metadata 过滤、Arrow-native、单进程可复现 |
| 生成上下文 | 命中 child 的 parent chunk；不足时扩展到前后相邻 parent | 控制上下文，服务 citation |

## 选型依据（写入 README 的证据）

- RAG vs Long-Context：LC 平均 +7.6% / +13.1% / +3.6%（Gemini/GPT-4o/GPT-3.5），RAG 成本低；Self-Route 成本 -65% / -39%。
- Anthropic context engineering：context 是有限注意力预算；最小高信号 token 原则。
- 本项目自己的实验结果将作为第一手证据，决定 v1.1 是否调整默认路径。

## 技术后果

- 检索管线必须同时支持四种运行档位：`bm25-only`（CI）、`fast`（MiniLM reranker）、`full`（默认）、`llm-rerank`（实验）。
- Self-Route 的「answerable」判定由同一模型在 RAG 模式下输出 `answerable: true/false + confidence`，不可靠时升级章节级或整本上下文。
- 所有实验必须记录模式、token、cost、时延；报告不得只报准确率。
- 中文 query 先经 DeepSeek 转写为英文 retrieval query；转写前后 query 均记录，便于分析转写损失。

## 被否选项

- **默认 long-context 直读**：成本高、产品 demo 慢，且无法展示检索工程能力。
- **GraphRAG v0 引入**：构建实体图价值存在，但 v0 复杂度高、维护重；先做实体索引与星图可视化，图谱检索留 v1.1。
- **Chroma/Qdrant/FAISS**：Chroma 依赖较重；Qdrant 引入服务端；FAISS 不是数据库、元数据管理需自建。LanceDB 最匹配 local-first。
- **多语言 embedding 模型（如 bge-m3）**：约 2GB+，对英文语料收益有限；用 DeepSeek query 转写替代。
- **微调专用 reranker**：与「5 仓库中微调只在 P1」的边界冲突，且 100 道 QA 规模不足以支撑可靠微调。
