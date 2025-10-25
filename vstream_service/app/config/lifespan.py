from contextlib import asynccontextmanager
from fastapi import FastAPI
from config.s3_client import get_s3_client, s3_client_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup: создаём клиент сразу ---
    # вызываем get_s3_client и "прокручиваем" генератор до yield, чтобы создать клиента
    client_gen = get_s3_client()
    await client_gen.__anext__()  # клиент создастся и сохранится в s3_client_instance

    yield  # приложение работает

    # --- shutdown: закрываем клиент ---
    if s3_client_instance:
        await s3_client_instance.close()
        # обнуляем глобальную переменную
        from config import s3_client
        s3_client.s3_client_instance = None
