from aiortc import RTCIceServer
from dotenv import load_dotenv

from pydantic_settings import BaseSettings




class Settings(BaseSettings):
    VSTREAM_SERVICE_NAME: str
    VSTREAM_DEBUG: bool
    VSTREAM_CORS_ALLOWED_ORIGINS: str

    VSTREAM_SERVICE_PORT: int
    VSTREAM_SERVICE_SECRET_KEY: str
    VSTREAM_SERVICE_ALGORITHM: str
    VSTREAM_SERVICE_ACCESS_TOKEN_EXPIRE_MINUTES: int

    TURN_PASSWORD: str
    TURN_USERNAME: str
    TURN_URL: str
    TURN_PORT: str

    ML_SERVICE_HOST: str
    ML_SERVICE_PORT: int

    S3_ENDPOINT_URL: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET: str

    
settings = Settings()

ice_servers = [
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(urls=[settings.TURN_URL], username=settings.TURN_USERNAME, credential=settings.TURN_PASSWORD),
        ]