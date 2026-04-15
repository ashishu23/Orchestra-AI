from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    embedding_provider: str = "google"    # "google" | "sentence-transformers"
    embedding_model: str = "models/gemini-embedding-001"
    google_api_key: str = ""

    chunk_size: int = 512
    chunk_overlap: int = 64

    collection_name: str = "research_docs"
    dense_dim: int = 768

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
