from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-optimized-project"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    dashscope_api_key: str = ""
    embedding_model: str = "text-embedding-v3"

    chunk_size: int = 512
    top_k_hybrid: int = 20
    top_k_rerank: int = 10
    top_k_mmr: int = 5
    use_multi_hop: bool = False

    vector_store_path: str = "./data/vector_store"
    bm25_index_path: str = "./data/bm25_index.pkl"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
