from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    tmdb_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "movies"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    default_k: int = 15
    llm_model: str = "claude-haiku-4-5-20251001"
    hybrid_search: bool = True
    qdrant_api_key: str = ""
    allowed_origins: str = "*"
    rate_limit_per_minute: int = 10


settings = Settings()
