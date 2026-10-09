import re
from datetime import datetime
from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import AfterValidator, BaseModel, Field

# A leading "scheme:" (e.g. "https:", "javascript:"), but not "host:port".
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:(?!\d)")


def _normalize_portfolio_url(value: str | None) -> str | None:
    """Trim the URL, default it to https, and reject non-http(s) or host-less URLs."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return value
    if not _SCHEME_RE.match(value):
        value = f"https://{value}"

    parts = urlsplit(value)
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError("portfolio_url must be an http or https URL")
    if not parts.hostname:
        raise ValueError("portfolio_url must include a hostname")
    return value


PortfolioUrl = Annotated[str | None, AfterValidator(_normalize_portfolio_url)]


class ProfileCreate(BaseModel):
    github_username: str | None = Field(default=None, max_length=255)
    portfolio_url: PortfolioUrl = Field(default=None, max_length=500)


class ProfileUpdate(BaseModel):
    github_username: str | None = Field(default=None, max_length=255)
    portfolio_url: PortfolioUrl = Field(default=None, max_length=500)


class ProfileResponse(BaseModel):
    id: UUID
    user_id: UUID
    github_username: str | None
    portfolio_url: str | None
    created_at: datetime
    resume_filename: str | None

    model_config = {"from_attributes": True}
