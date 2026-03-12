from dataclasses import asdict
from typing import Any, Generic, List, Optional, Type, TypeVar

from sqlalchemy import delete, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.types import UNSET

ModelType = TypeVar("ModelType")
EntityType = TypeVar("EntityType")


class BaseRepository(Generic[ModelType, EntityType]):
    def __init__(self, model: Type[ModelType], async_session: AsyncSession):
        self.model = model
        self._async_session = async_session

    async def _get_by_filters(self, *filters) -> Optional[ModelType]:
        query = select(self.model).where(*filters)
        result = await self._async_session.execute(query)
        return result.scalar_one_or_none()

    async def _exists_by_filters(self, *filters) -> bool:
        query = select(exists().where(*filters))
        result = await self._async_session.execute(query)
        return result.scalar() or False

    async def _list_by_filters(self, *filters, offset=0, limit=100) -> List[ModelType]:
        query = select(self.model).where(*filters).offset(offset).limit(limit)
        result = await self._async_session.execute(query)
        return result.scalars().all()

    async def _delete_by_filters(self, *filters) -> bool:
        stmt = delete(self.model).where(*filters)
        res = await self._async_session.execute(stmt)
        await self._async_session.flush()
        return res.rowcount > 0

    async def _update_model(self, orm_obj, entity):
        update_data = asdict(entity)

        for key, value in update_data.items():
            if (value is not UNSET) and hasattr(orm_obj, key):
                setattr(orm_obj, key, value)

        await self._async_session.flush()
        await self._async_session.refresh(orm_obj)

        return orm_obj


class EntityRepository(BaseRepository[ModelType, EntityType]):
    def _to_entity(self, orm_obj: ModelType) -> EntityType:
        raise NotImplementedError

    def _to_entities(self, orm_objects: List[ModelType]) -> List[EntityType]:
        return [self._to_entity(obj) for obj in orm_objects]


class OwnedEntityRepository(EntityRepository[ModelType, EntityType]):
    async def add(self, user_id: int, entity: Any) -> EntityType:
        payload = asdict(entity)
        orm_obj = self.model(**payload)

        orm_obj.user_id = user_id

        self._async_session.add(orm_obj)
        await self._async_session.flush()
        await self._async_session.refresh(orm_obj)

        return self._to_entity(orm_obj)

    async def delete(self, user_id: int, entity_id: int) -> bool:
        return await self._delete_by_filters(
            self.model.id == entity_id, self.model.user_id == user_id
        )

    async def list(
        self, user_id: int, offset: int = 0, limit: int = 100
    ) -> List[EntityType]:
        orm_objects = await self._list_by_filters(
            self.model.user_id == user_id, offset=offset, limit=limit
        )

        return self._to_entities(orm_objects)

    async def update(
        self, user_id: int, entity_id: int, update_entity: Any
    ) -> Optional[EntityType]:
        orm_obj = await self._get_by_filters(
            self.model.user_id == user_id,
            self.model.id == entity_id,
        )
        if not orm_obj:
            return None

        await self._update_model(orm_obj=orm_obj, entity=update_entity)

        return self._to_entity(orm_obj)

    async def get(self, user_id: int, entity_id: int) -> Optional[EntityType]:
        orm_object = await self._get_by_filters(
            self.model.id == entity_id, self.model.user_id == user_id
        )
        if orm_object is None:
            return None
        return self._to_entity(orm_object)
