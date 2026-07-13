"""
Request models.

Pydantic validates every incoming request automatically. A malformed payload is
rejected before our code runs — free input validation.

SECURITY: these objects hold credentials in memory for one request only. Never
written to disk, never logged, never persisted.
"""

from typing import Literal
from pydantic import BaseModel, Field


class UserConfig(BaseModel):
    """The keys a user brings (BYOK — Bring Your Own Keys)."""

    google_api_key: str = Field(..., description="Google Gemini API key")
    tavily_api_key: str = Field(..., description="Tavily search API key")
    wp_url: str = Field(..., description="WordPress site URL")
    wp_username: str = Field(..., description="WordPress username")
    wp_app_password: str = Field(..., description="WordPress application password")

    def clean_wp_url(self) -> str:
        """Strip a trailing slash so we can safely append API paths."""
        return self.wp_url.rstrip("/")


# "fast" leads with Flash-Lite: quickest, still Pro-derived.
# "quality" leads with 3.5 Flash: better factual grounding and prose.
SpeedMode = Literal["fast", "quality"]


class GenerateRequest(BaseModel):
    config: UserConfig
    topic: str = Field(..., min_length=3)
    category: str = Field(default="AI")
    mode: SpeedMode = Field(default="fast")


class ScoutRequest(BaseModel):
    config: UserConfig
    mode: SpeedMode = Field(default="fast")


class PublishRequest(BaseModel):
    config: UserConfig
    title: str
    body_markdown: str
    meta_description: str = ""
    category: str = "AI"