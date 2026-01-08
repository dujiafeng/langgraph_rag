@echo off
REM Start LangGraph Dev Server with Agent Chat UI
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set LANGSMITH_TRACING=false

.venv\Scripts\langgraph dev --port 2024 --no-browser
