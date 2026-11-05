from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Local Hugging Face embeddings use a separate Chroma index.
    chroma_persist_directory: str = "./data/chroma_gpt4o_local_embeddings"
    upload_directory: str = "./data/uploads"

    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 4
    min_relevance_score: float = 0.05
    max_upload_mb: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
