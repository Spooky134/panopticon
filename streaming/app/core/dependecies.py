from typing import Annotated, AsyncIterator

from fastapi import Depends
from pydantic import BaseModel, Field


from app.core.unit_of_work import UnitOfWork


async def get_uow() -> AsyncIterator[UnitOfWork]:
    async with UnitOfWork() as uow:
        yield uow

class Pagination(BaseModel):
    skip: int = Field(default=0, ge=0, le=10000)
    limit: int = Field(default=10, ge=1, le=1000)


PaginationDep = Annotated[Pagination, Depends()]
