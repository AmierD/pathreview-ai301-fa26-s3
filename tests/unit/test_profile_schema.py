"""Tests for portfolio_url validation in api/schemas/profile.py"""

import pytest
from pydantic import ValidationError

from api.schemas.profile import ProfileCreate, ProfileUpdate


@pytest.mark.unit
@pytest.mark.parametrize("schema", [ProfileCreate, ProfileUpdate])
class TestPortfolioUrlValidation:
    """Test suite for the portfolio_url validator on create and update."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("portfolio.mieracle.com", "https://portfolio.mieracle.com"),
            ("  https://janedoe.dev/  ", "https://janedoe.dev/"),
            ("http://janedoe.dev", "http://janedoe.dev"),
            ("janedoe.dev:8080/work", "https://janedoe.dev:8080/work"),
        ],
    )
    def test_accepts_and_normalizes_web_urls(self, schema, raw, expected):
        """Test that http(s) URLs are kept and a missing scheme defaults to https."""
        assert schema(portfolio_url=raw).portfolio_url == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "javascript:alert(1)",
            "file:///etc/passwd",
            "ftp://janedoe.dev",
            "mailto:jane@janedoe.dev",
            "https://",
        ],
    )
    def test_rejects_non_web_urls(self, schema, raw):
        """Test that non-http(s) schemes and URLs without a hostname are rejected."""
        with pytest.raises(ValidationError):
            schema(portfolio_url=raw)

    def test_none_and_blank_are_left_alone(self, schema):
        """Test that a missing or blank URL is not turned into a URL."""
        assert schema().portfolio_url is None
        assert schema(portfolio_url=None).portfolio_url is None
        assert schema(portfolio_url="").portfolio_url == ""
