# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
# from sqlalchemy.orm import declarative_base
# from typing import AsyncGenerator
# from config.settings import settings
#
#
# # Движок с настройками
# engine = create_async_engine(
#     settings.DB_URL,
#     echo=settings.DB_ECHO
# )
#
# # Фабрика асинхронных сессий
# AsyncSessionLocal = async_sessionmaker(
#     bind=engine,
#     autocommit=False,
#     autoflush=False,
#     expire_on_commit=False  # Чтобы объекты после commit() оставались доступны
# )
#
# Base = declarative_base()
#
# # Зависимость для FastAPI
# async def get_db() -> AsyncGenerator[AsyncSession, None]:
#     async with AsyncSessionLocal() as db:  # Автоматически закроется при выходе
#         yield db