from typing import Dict
from sqlalchemy import delete, exists, select
from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    def __init__(self, model, db_session: AsyncSession):
        self._model = model
        self._db_session = db_session

    async def _get_by_filters(self, *filters):
        query = select(self._model).where(*filters)
        result = await self._db_session.execute(query)
        return result.scalar_one_or_none()

    async def _exists_by_filters(self, *filters):
        query = select(exists().where(*filters))
        result = await self._db_session.execute(query)
        return result.scalar() or False

    async def _list_by_filters(self, *filters, offset, limit):
        query = select(self._model).where(*filters).offset(offset).limit(limit)
        result = await self._db_session.execute(query)
        return result.scalars().all()

    async def _delete_by_filters(self, *filters):
        stmt = delete(self._model).where(*filters)
        res = await self._db_session.execute(stmt)
        await self._db_session.flush()
        return res.rowcount > 0 # type: ignore

    async def _update(self, orm_obj, update_data: Dict):
        for key, value in update_data.items():
            if hasattr(orm_obj, key):
                setattr(orm_obj, key, value)

        await self._db_session.flush()
        await self._db_session.refresh(orm_obj)

        return orm_obj
