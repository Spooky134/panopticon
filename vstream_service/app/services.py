from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
import asyncio
from fastapi import BackgroundTasks

from config import settings
from utils import GrpcVideoProcessor, VideoTransformTrack
import uuid




class StreamService:
    def __init__(self):
        self.pcs = set()
        self.session_processors = {}

    async def offer(self, sdp_data: dict, background_tasks: BackgroundTasks) -> dict:
        offer = RTCSessionDescription(sdp_data["sdp"], sdp_data["type"])
        ice_servers = [
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(urls=[settings.TURN_URL], username=settings.TURN_USERNAME, credential=settings.TURN_PASSWORD),
        ]

        rtc_config = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(rtc_config)

        self.pcs.add(pc)

        # Создаем уникальную сессию для этого подключения
        session_id = str(uuid.uuid4())
        grpc_processor = GrpcVideoProcessor(session_id)
        await grpc_processor.start()
        self.session_processors[session_id] = grpc_processor

        @pc.on("track")
        async def on_track(track):
            print(f"Получен трек: {track.kind} для сессии {session_id}")
            if track.kind == "video":
                transformed_track = VideoTransformTrack(track, grpc_processor)  # Создаем измененный поток
                pc.addTrack(transformed_track)  # Добавляем его в соединение

        @pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            print(f"ICE connection state: {pc.iceConnectionState} для сессии {session_id}")
            if pc.iceConnectionState in ["failed", "closed", "disconnected"]:
                # Очищаем ресурсы сессии
                if session_id in self.session_processors:
                    await self.session_processors[session_id].stop()
                    del self.session_processors[session_id]
                background_tasks.add_task(self.__close_peer_connection, pc)

        await pc.setRemoteDescription(offer)
        # Ждем сбора ICE-кандидатов
        await asyncio.sleep(1)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": [settings.TURN_URL], "username": settings.TURN_USERNAME, "credential": settings.TURN_PASSWORD}
            ]
        }

    async def __close_peer_connection(self, pc: RTCPeerConnection):
        if pc in self.pcs:
            self.pcs.discard(pc)
        await pc.close()
        print("PeerConnection закрыт")