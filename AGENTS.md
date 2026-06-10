# AGENTS.md

## 关键命令

```bash
uv run python scripts/upload_docs.py --dir data/documents --reset  # 重置并重建索引
uv run pytest tests/ -q                                            # 跑全部98个单元测试
uv run python src/main.py --question "..."                         # CLI问答
uv run python tests/evaluation/run_evaluation.py --custom-only     # 自定义评估
```

- Windows 上 `langgraph dev` 需 `PYTHONUTF8=1` 环境变量（否则 GBK 崩）
- 启动脚本在根目录：`start_chat_server.cmd` / `start_dev_server.cmd`
- 不直接写 `.venv\Scripts\python`，用 `uv run python`

## 包管理

```
uv sync            # 安装依赖（pyproject.toml + uv.lock）
uv add pkg         # 添加依赖
uv lock            # 生成 uv.lock
```

`pyproject.toml` 已配置清华源（`pypi.tuna.tsinghua.edu.cn`）。

## 架构要点

- **入口**：CLI `src/main.py` / FastAPI `src/api/server.py` / LangGraph 服务 `src/graph/agent.py`
- **图**：`src/graph/builder.py` 构建 8 节点 StateGraph，编译后在 `agent.py` 供 `langgraph dev` 消费
- **LLM 工厂** `get_llm()` 在 `src/nodes/rewrite.py`（惰性初始化 `ChatOpenAI` → DeepSeek）
- **Embedding 工厂** `get_embeddings()` 在 `src/utils/embeddings.py`（DashScopeEmbeddings）
- **全局单例**：`dense_retriever`（Chroma）、`sparse_retriever`（BM25）、`settings`，均模块级实例化
- **config/ 和 src/config/** 同时存在：`config/settings.py` 是真实配置，`src/config/` 是过期目录（不含 .git）
- **脚本与 API** 用 `settings` 或硬编码默认值，修改 pipeline 参数检查两处

## RAGState 陷阱

`config` 字段是普通 `dict`，非 Pydantic 字段覆盖——传递部分 dict 会**丢失**未提供的默认键：

```python
# 错误：top_k_hybrid 等信息会丢失
RAGState(original_question="...", config={"use_multi_hop": True})

# 正确：保留全部默认值
RAGState(original_question="...")
```

## 测试

- pytest 无需任何 API key：所有 LLM / Embedding / CrossEncoder 调用均 mock
- **mock 路径注意**：patch 要打在**导入方**（使用点），非定义方。例如 `src.nodes.retrieve.hybrid_search` 而非 `src.retrieval.hybrid_retriever.hybrid_search`
- 编译后的 `CompiledStateGraph` 没有 `entry_point()` 方法，通过 `list(graph.nodes.keys())[1]` 取首个业务节点
- `make_doc` 工具函数在 `tests/conftest.py`，带可变参数，keyword 会放在 dict 顶层而非 `metadata` 内

## 评估

自定义评估（纯本地，无需 API key）：
```bash
uv run python tests/evaluation/run_evaluation.py --custom-only
```

Ragas 评估（需真实 API key，因为会调 LLM + Embedding）：
```bash
uv run python tests/evaluation/run_evaluation.py --ragas-only
```

## 索引

- Chroma 持久化到 `data/vector_store/`，BM25 pickle 到 `data/bm25_index.pkl`（`.gitignore` 已排除）
- 先用 `upload_docs.py` 建索引，再用 QA 命令或 API
- 采样 QA 数据在 `tests/evaluation/sample_qa.json`（15 条高考志愿相关问题）

## 环境

- `.env` 必须含 `DEEPSEEK_API_KEY` 和 `DASHSCOPE_API_KEY`
- `.env.example` 列出全部可用配置项
- `start_chat_server.cmd` 开启 LangSmith 追踪，`start_dev_server.cmd` 关闭
- langsmith 配置：`LANGSMITH_API_KEY` + `LANGSMITH_PROJECT`（默认 `rag-optimized-project`）
