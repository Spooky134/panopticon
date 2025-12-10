from aiortc import RTCPeerConnection, RTCConfiguration
from typing import Dict
from uuid import UUID

from core.logger import get_logger


logger = get_logger(__name__)


class ConnectionManager:
    def __init__(self, max_connections: int=1000):
        self.max_connections = max_connections
        self.peer_connections: Dict[UUID, RTCPeerConnection] = {}

    async def create_connection(self, session_id: UUID, rtc_config: RTCConfiguration) -> RTCPeerConnection:
        if len(self.peer_connections) >= self.max_connections:
            raise Exception("Server busy")
        pc =  RTCPeerConnection(rtc_config)
        self.peer_connections[session_id] = pc

        return pc

    async def get_connection(self, session_id: UUID) -> RTCPeerConnection:
        return self.peer_connections.get(session_id)

    async def close_connection(self, session_id: UUID):
        pc = self.peer_connections.pop(session_id, None)
        if pc:
            await pc.close()
        logger.info(f"peer_connection: {session_id} - is closed")