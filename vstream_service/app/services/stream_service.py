import jwt

from schemas.sdp import SDPData
from core.security.token import verify_token
from utils.session_manager import SessionManager


class StreamService:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    async def offer(self, token, sdp_data: SDPData) -> dict:
        payload = verify_token(token)
        user_id = payload["user_id"]
        session_id = payload["session_id"]
        print(f"[StreamService] Authorized user {user_id} starting stream session {session_id}")

        answer = await self.session_manager.create_session(
            session_id=session_id,
            user_id=user_id,
            sdp_data=sdp_data,
        )

        return answer