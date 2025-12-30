from fastapi import Depends

from services.streaming_runtime_service import StreamingRuntimeService
from core.engine.live_streaming_session_manger_factory import get_streaming_session_manager
from infrastructure.s3.s3_service import S3Service
from infrastructure.s3.s3_service_factory import get_s3_service
from core.engine.live_streaming_session_manager import LiveStreamingSessionManager
from services.streaming_session_lifecycle_service import StreamingSessionLifecycleService
from services.streaming_session_lifecycle_service_factory import get_streaming_session_lifecycle_service


def get_streaming_runtime_service(streaming_session_manager: LiveStreamingSessionManager=Depends(get_streaming_session_manager),
                                  streaming_session_lifecycle_service: StreamingSessionLifecycleService=Depends(get_streaming_session_lifecycle_service),
                                  s3_storage: S3Service = Depends(get_s3_service),
                                  ) -> StreamingRuntimeService:
    return StreamingRuntimeService(streaming_session_manager=streaming_session_manager,
                                   streaming_session_lifecycle_service=streaming_session_lifecycle_service,
                                   s3_storage=s3_storage,
                                   )
