# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 LangGraph 的可编排 RAG 流水线，面向**高考志愿助手**场景。查询优化、混合检索、重排序与多样性筛选均由 LangGraph StateGraph 编排。

## 常用命令

```bash
# 上传文档构建索引（自动分块 → Chroma + BM25）
uv run python scripts/upload_docs.py --dir data/documents

# CLI 问答
uv run python src/main.py --question "北京大学录取分数线？"

# CLI 问答（启用多跳拆分）
uv run python src/main.py --question "计算机和金融哪个好？" --multi-hop

# 启动 FastAPI 聊天服务
PYTHONUTF8=1 .venv\Scripts\uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload

# 启动 LangGraph 开发服务器（Studio UI）
PYTHONUTF8=1 .venv\Scripts\langgraph dev --port 2024

# 一键启动脚本（Windows）
start_chat_server.cmd   # FastAPI Web 聊天
start_dev_server.cmd    # LangGraph API 开发服务器
```

本项目**没有测试套件**，也没有配置 pytest/lint 等工具。

## 核心架构

### LangGraph 流水线

```
classify → rewrite → [split if multi-hop] → retrieve → rrf → rerank → mmr → generate
```

- 所有节点共享 `RAGState`（Pydantic BaseModel，定义在 `src/state.py`），通过图逐节点传递、原地修改
- 条件边：`classify` 输出 `rag`/`chat` → 走检索或直接回答；`rewrite` 后根据 `config.use_multi_hop` 决定是否拆分
- 图构建在 `src/graph/builder.py`，编译后的全局实例在 `src/graph/agent.py`（供 `langgraph dev` 使用）

### 8 个节点 (`src/nodes/`)

| 节点 | 文件 | 职责 |
|------|------|------|
| classify | `classifier.py` | LLM 二分类：rag（需查库）/ chat（闲聊） |
| rewrite | `rewrite.py` | LLM 改写用户问题为更适合检索的查询 |
| split | `split.py` | LLM 将复杂问题拆为 2~3 个子问题（Multi-Hop） |
| retrieve | `retrieve.py` | 对每个 query 调用混合检索，结果去重后写入 state |
| rrf | `fusion.py` | RRF（Reciprocal Rank Fusion）融合稠密+稀疏两路排名 |
| rerank | `rerank.py` | Cross-encoder 精排融合结果的前 50 条 |
| mmr | `mmr.py` | MMR 算法从前 20 条中选 top_k 兼顾相关性和多样性 |
| generate | `generate.py` | LLM 用 final_docs 上下文生成最终回答 |

### 检索层 (`src/retrieval/`)

- **Dense**: ChromaDB 持久化存储 + 阿里云百炼 `text-embedding-v3`（通过 `DashScopeEmbeddings`）
- **Sparse**: BM25（`rank_bm25`）+ jieba 分词，索引序列化为 pickle 文件
- `hybrid_search()` 并行调两者，返回 `(dense_hits, sparse_hits)`，各自由 `deduplicate_by_text` 去重

### 重排序 (`src/rerankers/`)

- `cross-encoder/ms-marco-MiniLM-L-6-v2`，sentence-transformers 库

### LLM 层

- 所有 LLM 调用共享 `get_llm()`（`src/nodes/rewrite.py`）— DeepSeek 通过兼容 OpenAI 接口的 `langchain_openai.ChatOpenAI`
- 配置集中在 `config/settings.py`（pydantic-settings，从 `.env` 读取）

### FastAPI 聊天服务 (`src/api/`)

- `server.py` 创建 FastAPI app，挂载静态前端和 `/api` 路由
- `routes.py` 提供 `/api/chat`（同步调 graph.invoke，在线程池执行）、`/api/session/*`（会话管理）、`/api/upload`（文档上传+重建索引）
- `sessions.py` 内存存储会话，无持久化

### 索引构建 (`scripts/upload_docs.py`)

- 读取 `.txt/.md` 文件 → 按 chunk_size 分块（优先在换行处断） → 同时写入 Chroma 和 BM25

## 配置

`config/settings.py` 定义所有可调参数，自动读取 `.env`：
- `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY`（必需）
- `CHUNK_SIZE`、`TOP_K_HYBRID`、`TOP_K_RERANK`、`TOP_K_MMR`、`USE_MULTI_HOP`（流水线参数）

## 关键依赖

- **包管理**: uv（pyproject.toml）
- **图编排**: langgraph ≥ 0.4, langchain ≥ 0.3
- **向量存储**: chromadb ≥ 0.6
- **稀疏检索**: rank-bm25, jieba
- **重排序**: sentence-transformers
- **Web 服务**: fastapi, uvicorn
- **LLM/Embedding**: langchain-openai (ChatOpenAI), dashscope (DashScopeEmbeddings)
