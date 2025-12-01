from fastapi import Depends

from api.services.stream import StreamService
from utils.streaming_session_manger_factory import get_streaming_session_manager
from storage.s3_storage import S3Storage
from storage.storage_factory import get_s3_storage
from db.repositories import StreamingVideoRepository, StreamingSessionRepository, repository_factory
from utils.streaming_session_manager import StreamingSessionManager


def get_stream_service(streaming_session_manager: StreamingSessionManager=Depends(get_streaming_session_manager),
                       s3_storage: S3Storage = Depends(get_s3_storage),
                       streaming_session_repository: StreamingSessionRepository = Depends(repository_factory(StreamingSessionRepository)),
                       streaming_video_repository: StreamingVideoRepository = Depends(repository_factory(StreamingVideoRepository))
                       ) -> StreamService:
    return StreamService(streaming_session_manager=streaming_session_manager,
                         s3_storage=s3_storage,
                         streaming_session_repository=streaming_session_repository,
                         streaming_video_repository=streaming_video_repository,
                         )
