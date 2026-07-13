"""
Configuration models.

Pydantic models define the SHAPE of data coming into the API. FastAPI uses them
to validate every request automatically — if a user sends a malformed payload,
FastAPI rejects it before your code ever runs. This is free input validation.

SECURITY NOTE: These objects hold user credentials in memory for the duration of
a single request. They are never written to disk, never logged, never persisted.
"""

from pydantic import BaseModel, Field


class UserConfig(BaseModel):
    """The credentials a user brings (BYOK — Bring Your Own Keys)."""

    google_api_key: str = Field(..., description="Google Gemini API key")
    tavily_api_key: str = Field(..., description="Tavily search API key")
    wp_url: str = Field(..., description="WordPress site URL, e.g. https://example.com")
    wp_username: str = Field(..., description="WordPress username")
    wp_app_password: str = Field(..., description="WordPress application password")

    def clean_wp_url(self) -> str:
        """Strip any trailing slash so we can safely append API paths."""
        return self.wp_url.rstrip("/")


class GenerateRequest(BaseModel):
    """A request to generate an article."""

    config: UserConfig
    topic: str = Field(..., min_length=3, description="What to write about")
    category: str = Field(default="AI", description="WordPress category name")


class ScoutRequest(BaseModel):
    """A request to scout trending topics."""

    config: UserConfig


class PublishRequest(BaseModel):
    """A request to push a finished article to WordPress as a draft."""

    config: UserConfig
    title: str
    body_markdown: str
    meta_description: str = ""
    category: str = "AI"