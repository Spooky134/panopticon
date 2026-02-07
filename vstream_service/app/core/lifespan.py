from contextlib import asynccontextmanager
from fastapi import FastAPI
import tritonclient.grpc.aio as grpcclient

from app.core.database import Base, engine
from app.core.logger import get_logger
from app.stream.engine.streaming_manager import StreamingManager
from app.aws.s3_video_storage_factory import create_s3_video_storage
from app.streaming_video.recorder.frame_collector_factory import FrameCollectorFactory
from app.stream.webrtc.connection_factory import ConnectionFactory
from app.ml_client.video_processor_factory import VideoProcessorFactory
from app.config.settings import settings
from app.config.logging import setup_logging
from app.stream.utils.ice_servers import get_ice_servers
from app.streaming_video.models import StreamingVideoModel
from app.streaming_session.models import StreamingSessionModel

logger = get_logger(__name__)

#TODO просмотреть
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app.state.triton_client = grpcclient.InferenceServerClient(
        url=settings.ML_SERVICE_URL
    )

    streaming_manager = StreamingManager(
        connection_factory=ConnectionFactory(ice_servers_factory=get_ice_servers),
        processor_factory=VideoProcessorFactory(triton_client=app.state.triton_client),
        collector_factory=FrameCollectorFactory(),
        max_sessions=1000
    )
    app.state.streaming_manager = streaming_manager

    s3_video_storage = await create_s3_video_storage()
    app.state.s3_video_storage = s3_video_storage

    yield

    logger.info("Shutting down: Disposing all active streaming sessions...")

    await streaming_manager.dispose_all_sessions()
    await s3_video_storage.close()
    await engine.dispose()
    await app.state.triton_client.close()


