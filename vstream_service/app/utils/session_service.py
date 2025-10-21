from aiortc import RTCSessionDescription, RTCConfiguration, RTCIceServer
import asyncio
from fastapi import BackgroundTasks
import uuid


from config import settings
from utils.grpc_video_processor import GrpcVideoProcessor
from utils.video_transform_track import VideoTransformTrack
from utils.connection_manager import ConnectionManager


class SessionService:
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager

    async def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        grpc_processor = GrpcVideoProcessor(session_id)
        await grpc_processor.start()
        self.connection_manager.session_processors[session_id] = grpc_processor
        return session_id

    def get_processor(self, session_id: str) -> GrpcVideoProcessor:
        return self.connection_manager.session_processors.get(session_id)