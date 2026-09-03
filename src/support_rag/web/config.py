from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WEB_", env_file=".env.web", extra="ignore")
    retrieval_url: str = "http://retrieval:8081"
    public_url: str = "https://tadpole-trimness-hunk.ngrok-free.dev"
    demo_mode: bool = False
    state_dir: Path = Path("state")
    session_seconds: int = Field(default=604800, ge=60)
    request_timeout_seconds: float = Field(default=180, ge=1)
    queue_timeout_seconds: float = Field(default=2, ge=0.01)
    max_chats: int = Field(default=50, ge=1)
    max_turns: int = Field(default=50, ge=1)
