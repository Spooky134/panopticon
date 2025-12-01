from fastapi import Depends

from api.services.streaming_session import StreamingSessionService
from db.repositories import StreamingVideoRepository, StreamingSessionRepository, repository_factory


def get_streaming_session_service(streaming_session_repository: StreamingSessionRepository = Depends(repository_factory(StreamingSessionRepository)),
                                  streaming_video_repository: StreamingVideoRepository = Depends(repository_factory(StreamingVideoRepository))
                                  ) -> StreamingSessionService:
    return StreamingSessionService(streaming_session_repository=streaming_session_repository,
                                   streaming_video_repository=streaming_video_repository,
                                   )
