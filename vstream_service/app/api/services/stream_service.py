import jwt

from api.schemas.sdp import SDPData
from core.security.token import verify_token
from utils.session_manager import SessionManager
from core.logger import get_logger


logger = get_logger(__name__)

class StreamService:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    async def offer(self, token, sdp_data: SDPData) -> dict:
        payload = verify_token(token)
        user_id = payload["user_id"]
        session_id = payload["session_id"]
        logger.info(f"session: {session_id} - Authorized user {user_id} starting stream")

        answer = await self.session_manager.initiate_session(
            session_id=session_id,
            user_id=user_id,
            sdp_data=sdp_data,
        )

        return answer