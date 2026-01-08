"""
文档上传脚本：读取本地文件（.txt/.md），分块后构建 Chroma 向量索引和 BM25 索引。

用法:
    uv run python scripts/upload_docs.py --dir data/documents
    uv run python scripts/upload_docs.py --dir data/documents --chunk-size 512 --chunk-overlap 64
"""

import argparse
import os
import glob
from dotenv import load_dotenv

load_dotenv()

from config.settings import settings
from src.retrieval.dense_retriever import dense_retriever
from src.retrieval.sparse_retriever import sparse_retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


def read_documents(doc_dir: str) -> list[tuple[str, str]]:
    files = []
    for ext in ("*.txt", "*.md"):
        files.extend(glob.glob(os.path.join(doc_dir, ext), recursive=True))
    if not files:
        logger.warning(f"No .txt or .md files found in {doc_dir}")
        return []

    docs = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
        basename = os.path.basename(fp)
        docs.append((basename, content))
        logger.info(f"Loaded: {basename} ({len(content)} chars)")
    return docs


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_newline = text.rfind("\n", start, end)
            if last_newline > start:
                end = last_newline + 1
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Build vector store and BM25 index from local documents")
    parser.add_argument("--dir", type=str, default="../data/documents", help="文档目录路径")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size, help="分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="分块重叠")
    parser.add_argument("--reset", action="store_true", help="重置已有索引")
    args = parser.parse_args()

    doc_dir = os.path.abspath(args.dir)
    if not os.path.isdir(doc_dir):
        logger.error(f"Directory not found: {doc_dir}")
        return

    if args.reset:
        logger.info("Resetting existing indices...")
        dense_retriever.reset()

    raw_docs = read_documents(doc_dir)
    if not raw_docs:
        return

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []

    for name, content in raw_docs:
        chunks = chunk_text(content, args.chunk_size, args.chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": name, "chunk_index": i})

    logger.info(f"Total chunks: {len(all_chunks)}")

    dense_retriever.add_documents(all_chunks, metadatas=all_metadatas)
    logger.info(f"Vector store built: {dense_retriever.count()} vectors")

    sparse_retriever.build_index(all_chunks)
    logger.info(f"BM25 index built: {sparse_retriever.count()} docs")


if __name__ == "__main__":
    main()
