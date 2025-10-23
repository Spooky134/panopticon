from aiortc import RTCPeerConnection, RTCConfiguration
from typing import Dict


class ConnectionManager:
    def __init__(self, max_connections: int=1000):
        self.max_connections = max_connections
        self.peer_connections: Dict[str, RTCPeerConnection] = {}


    async def create_connection(self, session_id: str, rtc_config: RTCConfiguration) -> RTCPeerConnection:
        if len(self.peer_connections) >= self.max_connections:
            raise Exception("Server busy")
        pc =  RTCPeerConnection(rtc_config)
        self.peer_connections[session_id] = pc

        return pc

    async def get_connection(self, session_id: str) -> RTCPeerConnection:
        return self.peer_connections.get(session_id)

    async def close_connection(self, session_id: str):
        pc = self.peer_connections.pop(session_id, None)
        if pc:
            await pc.close()
        print("PeerConnection is closed")