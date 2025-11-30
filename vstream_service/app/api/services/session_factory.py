from fastapi import Depends

from api.services.session import SessionService
from utils.session_manger_factory import get_session_manager
from storage.s3_storage import S3Storage
from storage.storage_factory import get_s3_storage
from db.repositories import TestingVideoRepository, TestingSessionRepository, repository_factory
from utils.session_manager import SessionManager


def get_session_service(testing_session_repository: TestingSessionRepository = Depends(repository_factory(TestingSessionRepository)),
                        testing_video_repository: TestingVideoRepository = Depends(repository_factory(TestingVideoRepository))
                        ) -> SessionService:
    return SessionService(testing_session_repository=testing_session_repository,
                         testing_video_repository=testing_video_repository,
                         )
