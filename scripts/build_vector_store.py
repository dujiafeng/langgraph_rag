"""
快捷脚本：仅构建 Chroma 向量索引（不含 BM25）。
等价于 uv run python scripts/upload_docs.py --dir data/documents
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.upload_docs import main

if __name__ == "__main__":
    main()
