from typing import Type, TypeVar
from fastapi import Depends

from storage.s3_storage import S3Storage
from api.dependencies.storage_factory import get_s3_storage
from webrtc.connection_manager import ConnectionManager
from grpc_client.grpc_processor_manager import GrpcProcessorManager
from config.settings import ice_servers


T = TypeVar('T')


def session_manager_factory(service_class: Type[T]) -> T:
    def _factory(connection_manager: ConnectionManager = Depends(ConnectionManager),
                 processor_manager: GrpcProcessorManager = Depends(GrpcProcessorManager),
                 s3_storage: S3Storage = Depends(get_s3_storage)) -> T:
        return service_class(connection_manager, processor_manager, s3_storage, ice_servers)
    return _factory

