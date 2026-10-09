"""Tests for api/routes/profiles.py"""

import json
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from api.routes.profiles import create_profile_endpoint


@pytest.mark.unit
class TestCreateProfileEndpoint:
    """Test suite for create_profile_endpoint input validation."""

    @pytest.mark.asyncio
    async def test_invalid_portfolio_url_returns_422(self):
        """Test that an invalid portfolio_url is a 422 with a JSON-safe detail, not a 500."""
        db = AsyncMock()
        user = Mock(id=uuid4())

        with pytest.raises(HTTPException) as exc_info:
            await create_profile_endpoint(
                github_username=None,
                portfolio_url="javascript:alert(1)",
                resume_file=None,
                current_user=user,
                db=db,
            )

        assert exc_info.value.status_code == 422
        detail = exc_info.value.detail
        assert detail[0]["loc"] == ("portfolio_url",)
        assert "http or https" in detail[0]["msg"]
        json.dumps(detail)
        db.add.assert_not_called()
