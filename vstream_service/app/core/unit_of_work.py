from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_maker


class UnitOfWork:
    def __init__(self):
        self._session_factory = async_session_maker
        self.session: AsyncSession | None = None

    async def commit(self):
        if self.session:
            await self.session.commit()

    async def rollback(self):
        if self.session:
            await self.session.rollback()

    async def __aenter__(self):
        self.session = self._session_factory()
        await self.session.begin()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            await self.rollback()
        else:
            await self.commit()

        await self.session.close()
