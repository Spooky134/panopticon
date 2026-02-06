from fastapi import APIRouter, FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from api.routes import streaming_session
from api.routes import streaming_runtime
from config.settings import settings
from core.lifespan import lifespan


from core.security.api_key import get_api_key

app = FastAPI(
    title=settings.VSTREAM_SERVICE_NAME,
    debug=settings.VSTREAM_DEBUG,
    lifespan=lifespan
)

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)


api_v1_router = APIRouter(prefix="/v1", tags=["v1"])
# TODO общая зависимость
# api_v1_router = APIRouter(prefix="/v1", tags=["v1"], dependencies=[Depends(get_api_key)])

api_v1_router.include_router(streaming_session.router, tags=["streaming_sessions"])
api_v1_router.include_router(streaming_runtime.router, tags=["stream"])

app.include_router(api_v1_router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=settings.VSTREAM_SERVICE_PORT, workers=1)