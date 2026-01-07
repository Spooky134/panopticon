from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.database import Base, engine
from core.logger import get_logger
from core.engine.live_streaming_session_manager import LiveStreamingSessionManager
from infrastructure.grpc_client.get_processor_factory import get_processor_factory
from infrastructure.s3.s3_video_storage_factory import create_s3_video_storage
from infrastructure.webrtc.get_connection_factory import get_connection_factory
from utils.get_frame_collector_factory import get_frame_collector_factory

logger = get_logger(__name__)

#TODO просмотреть
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_manager = LiveStreamingSessionManager(
        connection_factory=get_connection_factory(),
        processor_factory=get_processor_factory(),
        collector_factory=get_frame_collector_factory(),
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



