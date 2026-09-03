"""
Centralized application settings.

This is the ONLY place environment variables are read from directly.
Every other file imports `settings` from here instead of calling
os.getenv() itself — that way there's one source of truth for config.
"""
import os
from dotenv import load_dotenv

# Loads variables from a .env file in the project root into the environment.
load_dotenv()


class Settings:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:Imaya789!@localhost:5432/leave_db",
    )
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "b79434b6c54af3ae5b905ec28537efa4ddc95d86543e8eab7b7dbb663755c265",
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    SERVICE_TOKEN_EXPIRE_MINUTES: int = 15

    PROJECT_NAME: str = "Leave Management API"
    API_V1_PREFIX: str = "/api/v1"

    # Optional. If unset, the AI assistant falls back to a deterministic,
    # template-based answer instead of a generated one — see app/rag/llm.py.
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")


settings = Settings()
