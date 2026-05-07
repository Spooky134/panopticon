from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_maker


class UnitOfWork:
    def __init__(self):
        self._session_factory = async_session_maker
        self._session = None

    @property
    def session(self) -> AsyncSession:
        return self._session
    
    async def commit(self):
        if self._session:
            await self._session.commit()

    async def rollback(self):
        if self._session:
            await self._session.rollback()

    async def __aenter__(self):
        self._session = self._session_factory()
        await self._session.begin()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            await self.rollback()
        else:
            await self.commit()
        if self.session is not None:
            await self._session.close()
