from typing import Type, TypeVar
from fastapi import Depends

from storage.s3_storage import S3Storage
from api.dependencies.storage_factory import get_s3_storage
from webrtc.connection_manager import ConnectionManager
from grpc_client.processor_manager import ProcessorManager
from config.settings import ice_servers


T = TypeVar('T')


def session_manager_factory(service_class: Type[T]) -> T:
    def _factory(connection_manager: ConnectionManager = Depends(ConnectionManager),
                 processor_manager: ProcessorManager = Depends(ProcessorManager),
                 s3_storage: S3Storage = Depends(get_s3_storage)) -> T:
        return service_class(connection_manager=connection_manager,
                             processor_manager=processor_manager,
                             s3_storage=s3_storage,
                             ice_servers=ice_servers)
    return _factory

