from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    max_cpu_seconds: int = 10
    max_memory_mb: int = 128
    timeout_seconds: int = 15

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
