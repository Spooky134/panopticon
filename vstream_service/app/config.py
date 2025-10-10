from dotenv import load_dotenv

from pydantic_settings import BaseSettings

# TODO исправить на относительный
load_dotenv("/Users/andreychvankov/Projects/panopticon/.env")

class Settings(BaseSettings):
    PROJECT_NAME: str
    VERSION: str
    DEBUG: bool
    CORS_ALLOWED_ORIGINS: str
    APP_HOST: str
    APP_PORT: int

    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    TURN_PASS: str
    TURN_USER: str
    TURN_URL: str
    TURN_PORT: str

    ML_WORKER_HOST: str
    ML_WORKER_PORT: int

    EXTERNAL_IP: str
    # ICE_SERVERS = [
    #     {"urls": ["stun:stun.l.google.com:19302"]},
    #     {"urls": [TURN_URL], "username": TURN_USER, "credential": TURN_PASS},
    # ]
    #
    # DB_ECHO: bool
    # DB_URL: str
    
settings = Settings()