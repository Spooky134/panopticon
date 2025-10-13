from dotenv import load_dotenv

from pydantic_settings import BaseSettings

# TODO исправить на относительный
load_dotenv("/Users/andreychvankov/Projects/panopticon/.env")

class Settings(BaseSettings):
    VSTREAM_SERVICE_NAME: str
    VSTREAM_DEBUG: bool
    VSTREAM_CORS_ALLOWED_ORIGINS: str

    VSTREAM_SERVICE_SECRET_KEY: str
    VSTREAM_SERVICE_ALGORITHM: str
    VSTREAM_SERVICE_ACCESS_TOKEN_EXPIRE_MINUTES: int

    TURN_PASSWORD: str
    TURN_USERNAME: str
    TURN_URL: str
    TURN_PORT: str

    ML_SERVICE_HOST: str
    ML_SERVICE_PORT: int

    
settings = Settings()