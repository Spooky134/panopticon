from typing import Type, TypeVar
from fastapi import Depends

from storage.s3_storage import S3Storage
from core.dependencies.storage_factory import get_s3_storage
from webrtc.connection_manager import ConnectionManager
from grpc_client.grpc_processor_manager import GrpcProcessorManager




T = TypeVar('T')

def service_factory(service_class: Type[T]) -> T:
    def _factory() -> T:
        return service_class()
    return _factory

def stream_service_factory(service_class: Type[T]) -> T:
    def _factory(connection_manager: ConnectionManager = Depends(ConnectionManager),
                 processor_manager: GrpcProcessorManager = Depends(GrpcProcessorManager),
                 s3_storage: S3Storage = Depends(get_s3_storage)) -> T:
        return service_class(connection_manager, processor_manager, s3_storage)
    return _factory

