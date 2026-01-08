from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.database import Base, engine
from core.logger import get_logger
from core.engine.live_streaming_session_manager import LiveStreamingSessionManager
from infrastructure.s3.s3_video_storage_factory import create_s3_video_storage
from infrastructure.video.frame_collector_factory import FrameCollectorFactory
from infrastructure.webrtc.connection_factory import ConnectionFactory
from infrastructure.grpc_client.video_processor_factory import VideoProcessorFactory
from core.events import EventManager
from config import settings

logger = get_logger(__name__)

#TODO просмотреть
@asynccontextmanager
async def lifespan(app: FastAPI):
    event_bus = EventManager()
    app.state.event_bus = event_bus

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_manager = LiveStreamingSessionManager(
        connection_factory=ConnectionFactory(ice_servers=settings.ice_servers),
        processor_factory=VideoProcessorFactory(service_url=settings.settings.ML_SERVICE_URL),
        collector_factory=FrameCollectorFactory(),
        max_sessions=1000
    )
    app.state.session_manager = session_manager

    s3_video_storage = await create_s3_video_storage()
    app.state.s3_video_storage = s3_video_storage

    yield

    logger.info("Shutting down: Disposing all active streaming sessions...")

    await session_manager.dispose_all_sessions()

    await s3_video_storage.close()

    await engine.dispose()



