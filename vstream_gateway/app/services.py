from aiortc import RTCPeerConnection, RTCSessionDescription
import asyncio
from fastapi import BackgroundTasks
from utils import VideoTransformTrack


class StreamService:
    def __init__(self):
        self.pcs = set()
    
    async def offer(self, sdp_data: dict, background_tasks: BackgroundTasks) -> dict:
        offer = RTCSessionDescription(sdp_data["sdp"], sdp_data["type"])
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("track")
        def on_track(track):
            print(f"Получен трек: {track.kind}")
            if track.kind == "video":
                transformed_track = VideoTransformTrack(track)  # Создаем измененный поток
                pc.addTrack(transformed_track)  # Добавляем его в соединение

        @pc.on("iceconnectionstatechange")
        def on_ice_state_change():
            print(f"ICE connection state: {pc.iceConnectionState}")
            if pc.iceConnectionState in ["failed", "closed", "disconnected"]:
                background_tasks.add_task(self.__close_peer_connection, pc)

        await pc.setRemoteDescription(offer)
        
        # Ждем сбора ICE-кандидатов
        await asyncio.sleep(1)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


    async def __close_peer_connection(self, pc: RTCPeerConnection):
        if pc in self.pcs:
            self.pcs.discard(pc)
        await pc.close()
        print("PeerConnection закрыт")