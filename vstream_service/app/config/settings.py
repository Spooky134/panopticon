import json
from typing import Any

from aiortc import RTCIceServer
from pydantic import root_validator, model_validator, field_validator

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VSTREAM_SERVICE_NAME: str
    VSTREAM_DEBUG: bool
    VSTREAM_CORS_ALLOWED_ORIGINS: str
    SECRET_KEY: str
    VSTREAM_SERVICE_PORT: int
    VSTREAM_SERVICE_ALGORITHM: str
    VSTREAM_SERVICE_ACCESS_TOKEN_EXPIRE_MINUTES: int

    ML_SERVICE_URL: str

    STUN_SERVERS: Any
    TURN_SERVERS: Any
    TURN_SHARED_SECRET: str

    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    S3_URL: str
    S3_BUCKET_NAME: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    S3_REGION: str

    @field_validator("STUN_SERVERS", "TURN_SERVERS", mode="before")
    @classmethod
    def parse_json_servers(cls, value):
        if isinstance(value, str):
            value = [url.strip() for url in value.split(",") if url.strip()]
        return value

    @property
    def DB_URL(self):
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()