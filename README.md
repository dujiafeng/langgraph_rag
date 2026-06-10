# 高考志愿助手 — LangGraph RAG 问答系统

面向**高考志愿填报场景**的智能问答系统，基于 **LangGraph** 可编排 RAG 流水线，集成查询分类、混合检索、重排序与多样性筛选。内置 20 所高校、24 个热门专业、录取分数线、就业数据等知识库。

## langGraph流程图

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph LR;
	__start__([<p>__start__</p>]):::first
	classify(classify)
	rewrite(rewrite)
	split(split)
	retrieve(retrieve)
	rrf(rrf)
	rerank(rerank)
	mmr(mmr)
	generate(generate)
	__end__([<p>__end__</p>]):::last
	__start__ --> classify;
	classify -. &nbsp;chat&nbsp; .-> generate;
	classify -. &nbsp;rag&nbsp; .-> rewrite;
	mmr --> generate;
	rerank --> mmr;
	retrieve --> rrf;
	rewrite -. &nbsp;False&nbsp; .-> retrieve;
	rewrite -. &nbsp;True&nbsp; .-> split;
	rrf --> rerank;
	split --> retrieve;
	generate --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc


## 技术栈

| 组件 | 选型 |
|------|------|
| 图编排 | LangGraph, LangChain |
| LLM | DeepSeek (`deepseek-chat`) |
| Embedding | 阿里云百炼 `text-embedding-v3` |
| 稠密检索 | Chroma |
| 稀疏检索 | BM25 (`rank_bm25`) |
| 交叉编码器 | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 依赖管理 | uv |
| 测试 | pytest, pytest-mock |
| 评估 | ragas (faithfulness, answer_relevancy, context_recall, answer_correctness) |

## 知识库数据

| 内容 | 规模 |
|------|------|
| 高校介绍 | 20 所（985/211/省重点，覆盖北上广鄂川浙陕等地）|
| 专业介绍 | 24 个（计算机、口腔医学、临床医学、会计学、法学等）|
| 录取分数线 | 130+ 条（2023-2024 多省多专业，含分数和位次）|
| 就业数据 | 24 条（就业率、平均薪资、主要雇主、深造率）|
| 志愿填报技巧 | 冲稳保策略、时间节点、常见误区等 |
| 选科要求对照 | 物理类/历史类各专业选科要求 |

## 快速开始

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY

# 2. 上传文档构建索引
uv run python scripts/upload_docs.py --dir data/documents

# 3. 运行问答
uv run python src/main.py --question "华中科技大学计算机专业录取分数线？"

# 4. 启用多跳拆分（复杂问题自动拆解）
uv run python src/main.py --question "计算机和口腔医学哪个更好？" --multi-hop
```

## 项目结构

```
├── config/settings.py       # 配置管理（pydantic-settings，读取 .env）
├── data/documents/          # 高考知识库原始文档（7 个 Markdown 文件）
├── scripts/
│   ├── upload_docs.py       # 文档上传 → 分块 → 构建 Chroma + BM25 索引
│   └── generate_college_data.py  # 高考数据生成器
├── src/
│   ├── state.py             # RAGState（8 个节点共享的状态模型）
│   ├── main.py              # CLI 问答入口
│   ├── graph/
│   │   ├── builder.py       # LangGraph 图构建（8 节点 + 2 条件边）
│   │   └── agent.py         # 编译后的图实例（供 langgraph dev 使用）
│   ├── nodes/               # 8 个图节点
│   │   ├── classifier.py    # 分类：rag 走检索 / chat 直接回答
│   │   ├── rewrite.py       # 改写：优化查询 + 惰性 LLM 工厂
│   │   ├── split.py         # 拆分：多跳子问题分解
│   │   ├── retrieve.py      # 混合检索（稠密+稀疏）
│   │   ├── fusion.py        # RRF 融合
│   │   ├── rerank.py        # Cross-encoder 重排
│   │   ├── mmr.py           # MMR 多样性筛选
│   │   └── generate.py      # LLM 生成（闲聊/RAG 双路径）
│   ├── retrieval/           # 检索层
│   │   ├── dense_retriever.py   # Chroma 持久化向量检索
│   │   ├── sparse_retriever.py  # BM25 + jieba 分词
│   │   └── hybrid_retriever.py  # 并行调双路 + 去重
│   ├── rerankers/
│   │   └── cross_encoder.py     # Cross-encoder 精排
│   ├── api/                 # FastAPI 聊天后端
│   │   ├── server.py        # 应用入口
│   │   ├── routes.py        # API 路由（chat/session/upload）
│   │   ├── sessions.py      # 内存会话管理
│   │   └── static/          # Web 聊天前端（Markdown 渲染）
│   └── utils/
│       ├── embeddings.py    # DashScope Embeddings 惰性工厂
│       ├── deduplicate.py   # 文本去重
│       └── similarity.py    # 余弦相似度
├── tests/                   # 测试与评估模块
│   ├── conftest.py          # 公共 Fixtures（Mock LLM/Embedding/State）
│   ├── test_state.py        # RAGState 字段/序列化
│   ├── test_deduplicate.py  # 去重逻辑
│   ├── test_similarity.py   # 余弦相似度数学正确性
│   ├── test_fusion.py       # RRF 融合
│   ├── test_mmr.py          # MMR 选择（mock embedding）
│   ├── test_classifier.py   # 分类器（mock LLM）
│   ├── test_split.py        # 问题拆分（mock LLM）
│   ├── test_rerank.py       # 重排（mock cross-encoder）
│   ├── test_generate.py     # 生成（闲聊/RAG 路径）
│   ├── test_graph.py        # 图编译与路由
│   ├── test_hybrid_retrieval.py  # 混合检索
│   └── evaluation/          # ✅ 评估模块
│       ├── metrics.py       # Precision@k, Recall@k, MRR, MAP, NDCG
│       ├── evaluator.py     # RAGEvaluator：跑 pipeline → 算指标
│       ├── ragas_evaluator.py   # ragas 集成（faithfulness/relevancy/correctness）
│       ├── sample_qa.json   # 15 条高考领域标注 QA
│       ├── run_evaluation.py    # CLI 一键评估
│       └── test_metrics.py  # 指标函数验证
├── langgraph.json           # LangGraph API 服务配置
├── start_chat_server.cmd   # 一键启动 FastAPI 聊天（含 LangSmith 追踪）
├── start_dev_server.cmd    # 一键启动 LangGraph API 开发服务器
└── pyproject.toml           # 项目配置与依赖管理
```

### 节点说明

| 节点 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `classify` | original_question | query_type (rag/chat) | LLM 判断需要检索还是闲聊 |
| `rewrite` | original_question | rewritten_question | 优化查询语句适配检索 |
| `split` | rewritten_question | sub_questions | 分解复杂问题为 2-3 个子问题 |
| `retrieve` | rewritten_question / sub_questions | dense_results, sparse_results | 同时调 Chroma + BM25 |
| `rrf` | dense_results, sparse_results | fused_results | Reciprocal Rank Fusion 加权排序 |
| `rerank` | fused_results | reranked_results | Cross-encoder 对前 50 条精排 |
| `mmr` | reranked_results | final_docs | 兼顾相关性和多样性 |
| `generate` | final_docs | final_answer | DeepSeek 生成最终回答 |

## 索引构建

```bash
# 上传文档（自动分块后同时构建 Chroma + BM25 索引）
uv run python scripts/upload_docs.py --dir data/documents --chunk-size 512 --chunk-overlap 64

# 重置后重建
uv run python scripts/upload_docs.py --dir data/documents --reset
```

## Web 聊天界面（FastAPI）

内置的 FastAPI 应用提供完整的聊天 Web 界面，支持多会话、文档上传和 Multi-Hop 配置。

```bash
# 一键启动
start_chat_server.cmd

# 或手动启动：
PYTHONUTF8=1 .venv\Scripts\uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```

打开浏览器访问 `http://127.0.0.1:8000`。

**API 端点：**
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 聊天界面首页 |
| POST | `/api/chat` | 发送消息 `{ question, session_id?, use_multi_hop? }` |
| POST | `/api/session/create` | 创建新会话 |
| GET | `/api/session/{id}/history` | 获取历史消息 |
| POST | `/api/upload` | 上传文档（multipart/form-data） |

## LangGraph 可视化管理界面

本项目还支持通过 **LangSmith Studio** 和 **Agent Chat UI** 两种图形界面进行交互。

### LangSmith Studio（推荐）

```bash
# 启动开发服务器
start_dev_server.cmd
# 或手动运行：
PYTHONUTF8=1 .venv\Scripts\langgraph dev --port 2024
```

启动后访问：
- **Studio UI**：[`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)
- **API 文档**：[`http://127.0.0.1:2024/docs`](http://127.0.0.1:2024/docs)
- **API 端点**：`http://127.0.0.1:2024`

Studio 支持可视化图结构、单步调试、热重载等功能。

### Agent Chat UI（聊天界面）

```bash
start_chat_ui.cmd
# 或手动运行：
cd chat-ui
pnpm dev
```

启动后打开 `http://localhost:3000`，配置 `API URL` 为 `http://localhost:2024`，`Graph ID` 为 `rag_agent` 即可开始对话。

### 配置文件

项目根目录 `langgraph.json` 定义了图服务配置：

```json
{
  "dependencies": ["."],
  "graphs": {
    "rag_agent": "./src/graph/agent.py:graph"
  },
  "env": ".env"
}
```

## 追踪

集成 LangSmith，在 `.env` 设置 `LANGSMITH_API_KEY` 和 `LANGSMITH_PROJECT`，`start_chat_server.cmd` 已默认开启追踪。可在 [smith.langchain.com](https://smith.langchain.com) 查看每次对话的完整追踪。

## 测试

```bash
# 运行全部 98 个单元测试
uv run pytest tests/ -v

# 运行指定模块
uv run pytest tests/test_fusion.py tests/test_mmr.py -v
```

所有 LLM/Embedding 调用均 mock，无需真实 API key 即可运行。

## 评估

### 自定义指标评估

```bash
uv run python tests/evaluation/run_evaluation.py --custom-only
```

输出 Precision@5/10、Recall@5/10、MRR、NDCG@10 等检索指标。

### Ragas 评估

```bash
# 需要真实 API key（会调用 DeepSeek + DashScope）
uv run python tests/evaluation/run_evaluation.py --ragas-only
```

评估 faithfulness（忠实度）、answer_relevancy（相关性）、context_recall（召回）、answer_correctness（正确性）。

### 完整评估

```bash
uv run python tests/evaluation/run_evaluation.py
uv run python tests/evaluation/run_evaluation.py --num-questions 5 --output results.json
uv run python tests/evaluation/run_evaluation.py --use-multi-hop
```

## 配置文件

```env
# .env 完整配置项
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx
LANGSMITH_API_KEY=lsv2_xxx
LANGSMITH_PROJECT=rag-optimized-project
CHUNK_SIZE=512
TOP_K_HYBRID=20       # 混合检索召回数
TOP_K_RERANK=10       # 重排后保留数
TOP_K_MMR=5           # MMR 最终文档数
USE_MULTI_HOP=false   # 默认不启用问题拆分
```
