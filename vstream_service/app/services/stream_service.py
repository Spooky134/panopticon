from aiortc import RTCSessionDescription, RTCConfiguration, RTCPeerConnection
import asyncio
from fastapi import BackgroundTasks
import uuid

from config import ice_servers

from utils.webrtc.connection_manager import ConnectionManager
from utils.grpc.processor_manager import ProcessorManager
from utils.webrtc.video_transform_track import VideoTransformTrack


class StreamService:
    def __init__(self,
                 connection_manager: ConnectionManager,
                 processor_manager: ProcessorManager):
        self.connection_manager = connection_manager
        self.processor_manager = processor_manager

    async def offer(self, sdp_data: dict, background_tasks: BackgroundTasks) -> dict:
        offer = RTCSessionDescription(sdp_data["sdp"], sdp_data["type"])
        rtc_config = RTCConfiguration(iceServers=ice_servers)

        # Создаем уникальную сессию для этого подключения
        session_id = str(uuid.uuid4())

        peer_connection = await self.connection_manager.create_connection(session_id=session_id, rtc_config=rtc_config)
        grpc_processor = await self.processor_manager.create_processor(session_id=session_id,)


        @peer_connection.on("track")
        async def on_track(track):
            print(f"Track received: {track.kind} for session {session_id}")
            if track.kind == "video":
                transformed_track = VideoTransformTrack(track, grpc_processor)  # Создаем измененный поток
                peer_connection.addTrack(transformed_track)  # Добавляем его в соединение

        @peer_connection.on("iceconnectionstatechange")
        async def on_ice_state_change():
            print(f"ICE connection state: {peer_connection.iceConnectionState} for session {session_id}")
            if peer_connection.iceConnectionState in ["failed", "closed", "disconnected"]:
                background_tasks.add_task(self.processor_manager.close_processor, session_id)
                background_tasks.add_task(self.connection_manager.close_connection, session_id)

        await peer_connection.setRemoteDescription(offer)
        # Ждем сбора ICE-кандидатов
        await asyncio.sleep(1)
        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)

        return {
            "sdp": peer_connection.localDescription.sdp,
            "type": peer_connection.localDescription.type,
            "iceServers": ice_servers
        }

    # async def _create_peer_connection(self, session_id: str) -> RTCPeerConnection:
    #     rtc_config = RTCConfiguration(iceServers=ice_servers)
    #     peer_connection = await self.connection_manager.create_connection(session_id=session_id, rtc_config=rtc_config)
    #
    #     return peer_connection
    #
    # async def _setup_event_handlers(self, peer_connection: RTCPeerConnection, processor, session_id: str, ):
    #     async def on_track(track):