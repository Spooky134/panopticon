from fastapi import Depends

from storage.s3_storage import S3Storage
from api.dependencies.storage_factory import get_s3_storage
from webrtc.connection_manager import ConnectionManager
from grpc_client.processor_manager import ProcessorManager
from config import settings
from db.repositories.testing_session import TestingSessionRepository
from api.dependencies.repository_factory import get_testing_session_repository
from utils.session_manager import SessionManager


def get_session_manager(connection_manager: ConnectionManager = Depends(ConnectionManager),
                        processor_manager: ProcessorManager = Depends(ProcessorManager),
                        s3_storage: S3Storage = Depends(get_s3_storage),
                        testing_session_repository: TestingSessionRepository = Depends(get_testing_session_repository),
                        ice_servers: list = Depends(lambda: settings.ice_servers)) -> SessionManager:
    return SessionManager(connection_manager=connection_manager,
                            processor_manager=processor_manager,
                             s3_storage=s3_storage,
                             testing_session_repository=testing_session_repository,
                             ice_servers=ice_servers)

