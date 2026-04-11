from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vault_url: str = "http://localhost:8001"
    sandbox_url: str = "http://localhost:8002"

    llm_provider: str = "google"          # "google" | "anthropic"
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"

    max_retries: int = 3
    mcp_timeout: float = 30.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
