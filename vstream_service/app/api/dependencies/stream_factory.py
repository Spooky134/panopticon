from typing import Type, TypeVar
from fastapi import Depends

from api.dependencies.session_manger_factory import session_manager_factory
from utils.session_manager import SessionManager


T = TypeVar('T')

def service_factory(service_class: Type[T]) -> T:
    def _factory() -> T:
        return service_class()
    return _factory

def stream_service_factory(service_class: Type[T]) -> T:
    def _factory(session_mng: SessionManager = Depends(session_manager_factory(SessionManager))) -> T:
        return service_class(session_mng)
    return _factory
