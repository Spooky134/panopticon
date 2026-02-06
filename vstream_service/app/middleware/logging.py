import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("app.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware для логирования всех входящих запросов"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(int(start_time * 1000))

        # Логируем входящий запрос
        logger.info(
            f"[{request_id}] → {request.method} {request.url}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else "unknown"
            }
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Логируем ответ
            logger.info(
                f"[{request_id}] ← {response.status_code} ({process_time:.3f}s)",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )

            # Добавляем заголовки для трассировки
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"

            return response

        except Exception as exc:
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] ← Ошибка: {str(exc)} ({process_time:.3f}s)",
                extra={
                    "request_id": request_id,
                    "error": str(exc),
                    "process_time": process_time
                },
                exc_info=True
            )
            raise