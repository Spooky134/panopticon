from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import exists, select, update, delete
from sqlalchemy.orm import selectinload
from db.models.session import TestingSession


class SessionRepository:
    def __init__(self, async_session: AsyncSession):
        self.async_session = async_session

    async def get(self, session_id: int) -> Optional[TestingSession]:
        result = await self.async_session.execute(
            select(TestingSession).
            where(TestingSession.id == session_id))
        return result.scalar_one_or_none()

    async def update(self, session_id: int, data: dict) -> Optional[TestingSession]:
        session = await self.get(session_id=session_id)

        for field, value in data.items():
            setattr(session, field, value)

        session.time_update = datetime.now()
        await self.async_session.commit()

        return await self.get(session_id=session_id)

    # async def create(self, link_data: dict) -> Link:
    #     new_link = Link(**link_data)
    #     new_link.collections = []
    #     self.async_session.add(new_link)
    #     await self.async_session.commit()
    #
    #     return await self.get(new_link.id)

    # async def get_all(self, skip: int = 0, limit: int = 100) -> list[Link]:
    #     result = await self.session.execute(
    #         select(Link).options(selectinload(Link.collections)).offset(skip).limit(limit)
    #     )
    #     return result.scalars().all()


    # async def delete(self, link_id: int) -> None:
    #     await self.session.execute(delete(Link).where(Link.id == link_id))
    #     await self.session.commit()

    # async def exists_by_url(self, url: str) -> bool:
    #     return await self.session.scalar(select(exists()
    #                                             .where(Link.url == url)))
    #
    # async def exists_by_id(self, link_id: int) -> bool:
    #     return await self.session.scalar(select(exists()
    #                                             .where(Link.id == link_id)))

    # async def get_by_type(self, link_type: Optional[LinkType] = None) -> list[Link]:
    #     query = select(Link).options(selectinload(Link.collections))
    #
    #     if link_type:
    #         query = query.where(Link.link_type == link_type.value)
    #
    #     result = await self.session.execute(query)
    #
    #     return result.scalars().all()

    # async def get_by_ids(self, link_ids: list[int], load_collection: bool = False) -> list[Link]:
    #     query = select(Link).where(Link.id.in_(link_ids))
    #
    #     if load_collection:
    #         query = query.options(selectinload(Link.collections))
    #
    #     result = await self.session.execute(query)
    #
    #     return result.scalars().all()

    # async def get_by_user_id(self, collection_id: int, skip: int = 0, limit: int = 100) -> list[Link]:
    #     result = await self.session.execute(
    #         select(Link)
    #         .join(Link.collections)
    #         .options(
    #             selectinload(Link.collections)  # Жадная загрузка коллекций
    #         )
    #         .where(Collection.id == collection_id)
    #         .offset(skip)
    #         .limit(limit)
    #     )
    #
    #     return result.scalars().all()
