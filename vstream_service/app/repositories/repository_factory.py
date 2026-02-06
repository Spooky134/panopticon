from typing import Type, TypeVar
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db


T = TypeVar('T')

def repository_factory(repository_class: Type[T]) -> T:
    def _factory(db: AsyncSession = Depends(get_db)) -> T:
        return repository_class(db)
    return _factory