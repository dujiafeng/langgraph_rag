@echo off
REM Start FastAPI Chat Server with LangSmith tracing
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set LANGSMITH_TRACING=true

.venv\Scripts\uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
