from fastapi import APIRouter, FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from app.session.routes import router as session_router
from app.stream.routes import router as stream_router
from app.config.settings import settings
from app.core.lifespan import lifespan
from app.core.logging import get_logger


from app.core.security.api_key import get_api_key

logger = get_logger(__name__)

app = FastAPI(
    title=settings.stream_service.name,
    debug=settings.stream_service.debug,
    lifespan=lifespan,
    root_path="/api"
)

@app.get("/health")
async def health():
    return {"status": "ok"}



app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

# @app.exception_handler(BaseAppException)
# async def app_exception_handler(request: Request, exc: BaseAppException):
#     logger.error(
#         f"Unhandled exception in {request.method} {request.url.path}: {str(exc)}",
#         exc_info=True
#     )
#     return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


v1_router = APIRouter(prefix="/v1")
# TODO общая зависимость
# api_v1_router = APIRouter(prefix="/v1", tags=["v1"], dependencies=[Depends(get_api_key)])

v1_router.include_router(session_router)
v1_router.include_router(stream_router)

app.include_router(v1_router)


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=8000, workers=1)