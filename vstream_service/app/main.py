from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import streaming_session
from api.routes import stream
from config.settings import settings
from core.lifespan import lifespan
from config.logging import setup_logging

setup_logging()


app = FastAPI(title=settings.VSTREAM_SERVICE_NAME, debug=settings.VSTREAM_DEBUG, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_v1_router = APIRouter(prefix="/v1", tags=["v1"])

api_v1_router.include_router(streaming_session.router, tags=["streaming_sessions"])
api_v1_router.include_router(stream.router, tags=["stream"])

app.include_router(api_v1_router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=settings.VSTREAM_SERVICE_PORT, workers=1)