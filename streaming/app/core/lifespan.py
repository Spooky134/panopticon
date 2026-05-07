from contextlib import asynccontextmanager
from fastapi import FastAPI
import tritonclient.grpc.aio as grpcclient

from app.core.database import Base, engine
from app.core.logging import get_logger
from app.stream.engine.streaming_manager import StreamingManager
from app.video.recorder.frame_collector_factory import FrameCollectorFactory
from app.stream.webrtc.connection_factory import ConnectionFactory
from app.ml_processor.video_processor_factory import VideoProcessorFactory
from app.config.settings import settings
from app.config.logging import setup_logging
from app.stream.utils.ice_servers import get_ice_servers
from app.video.models import VideoModel
from app.session.models import SessionModel
from app.core.taskiq_broker import broker


logger = get_logger(__name__)

#TODO просмотреть
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # logger.info(type(settings.stream_cors.allowed_origins))
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)
    if not broker.is_worker_process:
        await broker.startup()

    app.state.triton_client = grpcclient.InferenceServerClient(
        url=settings.ml_processor.url
    )

    streaming_manager = StreamingManager(
        connection_factory=ConnectionFactory(ice_servers_factory=get_ice_servers),
        processor_factory=VideoProcessorFactory(triton_client=app.state.triton_client),
        collector_factory=FrameCollectorFactory(),
        max_sessions=1000
    )
    app.state.streaming_manager = streaming_manager

    # s3_video_storage = await create_s3_video_storage()
    # app.state.s3_video_storage = s3_video_storage

    yield

    logger.info("Shutting down: Disposing all active streaming sessions...")
    if not broker.is_worker_process:
        await broker.shutdown()
    await app.state.streaming_manager.dispose_all_sessions()
    # await app.state.s3_video_storage.close()
    await engine.dispose()
    await app.state.triton_client.close()


