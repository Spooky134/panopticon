from fastapi import HTTPException, status

# TODO пересмотреть ошибки
# TODO пересмотреть расположение
class AppException(HTTPException):
    """Базовое исключение приложения"""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Произошла ошибка"

    def __init__(self, **kwargs):
        super().__init__(
            status_code=kwargs.get('status_code', self.status_code),
            detail=kwargs.get('detail', self.detail)
        )

class ValidationError(AppException):
    """Ошибки валидации (400)"""
    status_code = status.HTTP_400_BAD_REQUEST
    detail = "Ошибка валидации"

class NotFoundError(AppException):
    """Объект не найден (404)"""
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Объект не найден"

class PermissionDeniedError(AppException):
    """Доступ запрещен (403)"""
    status_code = status.HTTP_403_FORBIDDEN
    detail = "Доступ запрещен"