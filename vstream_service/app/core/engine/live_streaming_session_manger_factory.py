from fastapi import Request

from core.engine.live_streaming_session_manager import LiveStreamingSessionManager


def get_streaming_session_manager(request: Request) -> LiveStreamingSessionManager:
    return request.app.state.session_manager

