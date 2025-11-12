from fastapi import Depends

from services.stream_service import StreamService
from utils.session_manager import SessionManager
from api.dependencies.session_manger_factory import get_session_manager


def get_stream_service(session_manager: SessionManager=Depends(get_session_manager)) -> StreamService:
    return StreamService(session_manager=session_manager)
