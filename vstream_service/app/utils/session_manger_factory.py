from fastapi import Depends

from storage.s3_storage import S3Storage
from storage.storage_factory import get_s3_storage
from webrtc.connection_manager import ConnectionManager
from grpc_client.processor_manager import ProcessorManager
from config import settings
from db.repositories import TestingVideoRepository, TestingSessionRepository, repository_factory
from utils.session_manager import SessionManager


def get_session_manager(connection_manager: ConnectionManager = Depends(ConnectionManager),
                        processor_manager: ProcessorManager = Depends(ProcessorManager),
                        s3_storage: S3Storage = Depends(get_s3_storage),
                        testing_session_repository: TestingSessionRepository = Depends(repository_factory(TestingSessionRepository)),
                        testing_video_repository: TestingVideoRepository = Depends(repository_factory(TestingVideoRepository)),
                        ice_servers: list = Depends(lambda: settings.ice_servers)) -> SessionManager:
    return SessionManager(connection_manager=connection_manager,
                          processor_manager=processor_manager,
                          s3_storage=s3_storage,
                          testing_session_repository=testing_session_repository,
                          testing_video_repository=testing_video_repository,
                          ice_servers=ice_servers)

