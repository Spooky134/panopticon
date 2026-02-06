import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app

@pytest.mark.asyncio
async def test_get_streaming_session() -> None:
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as ac:
        response = await ac.get("/sessions/{streaming_session_id}")