import asyncio
from datetime import datetime
from aiortc import RTCSessionDescription, RTCPeerConnection

from grpc_client.base_processor import BaseProcessor
from utils.frame_collector import FrameCollector
from api.schemas.sdp import SDPData


class Session:
    def __init__(self, session_id:str, user_id:str, peer_connection: RTCPeerConnection,
                 video_processor: BaseProcessor, collector: FrameCollector):
        self.session_id = session_id
        self.user_id = user_id
        self.peer_connection = peer_connection
        self.grpc_processor = video_processor
        self.collector = collector
        self.started_at = None
        self.finished_at = None
        self._is_finalized = False

    async def start(self, sdp_data: SDPData) -> dict:
        offer = RTCSessionDescription(sdp_data.sdp, sdp_data.type)
        await self.peer_connection.setRemoteDescription(offer)
        await asyncio.sleep(1)
        answer = await self.peer_connection.createAnswer()
        await self.peer_connection.setLocalDescription(answer)

        self.started_at = datetime.now()
        print(f"[Session:{self.session_id}] started for user {self.user_id}")

        #TODO заменить на схему
        return {
            "sdp": self.peer_connection.localDescription.sdp,
            "type": self.peer_connection.localDescription.type,
        }

    async def finalize(self):
        if self._is_finalized:
            return

        print(f"[Session:{self.session_id}] Finalizing session resources...")
        self._is_finalized = True

        try:
            await self.collector.finalize()
        except Exception as e:
            print(f"[Session:{self.session_id}] Collector finalize error: {e}")

        self.finished_at = datetime.now()