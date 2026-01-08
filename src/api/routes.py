import asyncio
import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from src.state import RAGState
from src.graph.builder import build_rag_graph
from src.utils.logger import get_logger
from src.api.sessions import session_manager, Session

logger = get_logger(__name__)

router = APIRouter(prefix="/api")
_executor = ThreadPoolExecutor(max_workers=4)
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    use_multi_hop: bool = False


class SessionResponse(BaseModel):
    session_id: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list


def _run_pipeline(question: str, use_multi_hop: bool) -> RAGState:
    initial = RAGState(
        original_question=question,
        config={
            "chunk_size": 512,
            "top_k_hybrid": 20,
            "top_k_rerank": 10,
            "top_k_mmr": 5,
            "use_multi_hop": use_multi_hop,
        },
    )
    graph = _get_graph()
    return graph.invoke(initial)


@router.post("/chat")
async def chat(req: ChatRequest):
    session = session_manager.get_or_create(req.session_id)
    session.add_message("user", req.question)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _run_pipeline,
            req.question,
            req.use_multi_hop,
        )
        answer = result.get("final_answer", "") or "无法生成答案。"
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        answer = f"处理问题时出错：{str(e)}"

    session.add_message("assistant", answer)

    return {
        "answer": answer,
        "session_id": session.session_id,
        "messages": [m.to_dict() for m in session.messages],
    }


@router.post("/session/create", response_model=SessionResponse)
async def create_session():
    session = session_manager.create_session()
    return SessionResponse(session_id=session.session_id)


@router.get("/session/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    session: Session | None = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return HistoryResponse(
        session_id=session.session_id,
        messages=[m.to_dict() for m in session.messages],
    )


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    from scripts.upload_docs import chunk_text, read_documents
    from src.retrieval.dense_retriever import dense_retriever
    from src.retrieval.sparse_retriever import sparse_retriever

    doc_dir = Path("data/documents")
    doc_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename:
            continue
        content = await f.read()
        text = content.decode("utf-8")
        target = doc_dir / f.filename
        target.write_text(text, encoding="utf-8")
        saved.append(f.filename)

    if not saved:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    all_chunks = []
    all_metadatas = []
    for name in saved:
        filepath = doc_dir / name
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text, 512, 64)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": name, "chunk_index": i})

    dense_retriever.add_documents(all_chunks, metadatas=all_metadatas)
    sparse_retriever.build_index(all_chunks)

    logger.info(f"Uploaded {len(saved)} files, built index with {len(all_chunks)} chunks")
    return {
        "files": saved,
        "chunks": len(all_chunks),
        "message": f"成功导入 {len(saved)} 个文件，共 {len(all_chunks)} 个文本块",
    }
