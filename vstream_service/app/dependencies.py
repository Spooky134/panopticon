from typing import Type, TypeVar
from fastapi import Depends
from fastapi import APIRouter, BackgroundTasks
# from sqlalchemy.ext.asyncio import AsyncSession
# from config.database import get_db


T = TypeVar('T')

def service_factory(service_class: Type[T]) -> T:
    def _factory() -> T:
        return service_class()
    return _factory


# from app.services.link_service import LinkService

# def service_factory(service_class: type[LinkService]) -> LinkService:
#     async def _factory(db: AsyncSession = Depends(get_db)) -> LinkService:
#         return service_class(db)
#     return _factory