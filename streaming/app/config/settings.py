import base64
import hashlib
import hmac
from pathlib import Path
from typing import Any, List, Tuple
from uuid import UUID
from aiortc import RTCIceServer
from pydantic import field_validator, BaseModel, AmqpDsn

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent.parent.parent
ENV_FILE_PATH = ROOT_DIR / ".env"


class TaskiqConfig(BaseModel):
    url: AmqpDsn

class S3Config(BaseModel):
    url: str
    bucket_name: str
    access_key: str
    secret_key: str
    region: str

class DatabaseConfig(BaseModel):
    host: str
    port: int
    db: str
    user: str
    password: str

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )

class ServiceConfig(BaseModel):
    name: str
    port: int
    debug: bool

class AuthConfig(BaseModel):
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int

class CorsConfig(BaseModel):
    allowed_origins: List[str]

class MLProcessorConfig(BaseModel):
    url: str

class TURNServerConfig(BaseModel):
    url: str
    shared_secret: str

    def _generate_credentials(self, identifier: UUID, expiration_time: int, origin: str) -> Tuple[str, str]:
        username = f"{expiration_time}:{origin}:{identifier}"

        digester = hmac.new(
            self.shared_secret.encode(),
            username.encode(),
            hashlib.sha1
        )
        credential = base64.b64encode(digester.digest()).decode()

        return username, credential

    def build_ice(self, identifier: UUID, expiration_time: int, origin: str) -> RTCIceServer:
        username, credential = self._generate_credentials(identifier, expiration_time, origin)

        return RTCIceServer([self.url], username, credential)

class STUNServerConfig(BaseModel):
    url: str

    def build_ice(self) -> RTCIceServer:
        return RTCIceServer([self.url])

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )
    
    turn_servers: List[TURNServerConfig]
    stun_servers: List[STUNServerConfig]

    ml_processor: MLProcessorConfig
    stream_cors: CorsConfig
    stream_auth: AuthConfig
    stream_service: ServiceConfig
    s3: S3Config
    database: DatabaseConfig
    taskiq: TaskiqConfig



settings = Settings()