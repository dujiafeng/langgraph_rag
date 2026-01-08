"""
快捷脚本：仅构建 BM25 索引。
用法: uv run python scripts/build_bm25_index.py --dir data/documents
"""
import argparse
import os
import glob
from dotenv import load_dotenv

load_dotenv()

from src.retrieval.sparse_retriever import sparse_retriever
from src.utils.logger import get_logger
from scripts.upload_docs import read_documents, chunk_text

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index from local documents")
    parser.add_argument("--dir", type=str, default="../data/documents", help="文档目录路径")
    parser.add_argument("--chunk-size", type=int, default=512, help="分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="分块重叠")
    args = parser.parse_args()

    doc_dir = os.path.abspath(args.dir)
    raw_docs = read_documents(doc_dir)

    all_chunks = []
    for name, content in raw_docs:
        all_chunks.extend(chunk_text(content, args.chunk_size, args.chunk_overlap))

    sparse_retriever.build_index(all_chunks)
    logger.info(f"BM25 index built: {sparse_retriever.count()} docs")


if __name__ == "__main__":
    main()
