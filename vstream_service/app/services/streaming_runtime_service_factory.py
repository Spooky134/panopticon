from services.streaming_runtime_service import StreamingRuntimeService
from infrastructure.s3.s3_video_storage import S3VideoStorage


def create_streaming_runtime_service(streaming_session_manager,
                                  streaming_session_lifecycle_service,
                                  s3_video_storage: S3VideoStorage,
                                  ) -> StreamingRuntimeService:
    return StreamingRuntimeService(streaming_session_manager=streaming_session_manager,
                                   streaming_session_lifecycle_service=streaming_session_lifecycle_service,
                                   s3_video_storage=s3_video_storage)
