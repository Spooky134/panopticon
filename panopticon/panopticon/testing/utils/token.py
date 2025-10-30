import secrets

from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from datetime import datetime, timedelta

from django.http import JsonResponse


@login_required
def generate_webrtc_token(request):
    # Генерируем уникальный токен
    token = secrets.token_urlsafe(32)

    # Сохраняем в кеш с данными пользователя
    cache.set(
        f"webrtc_{token}",
        {
            'user_id': request.user.id,
            'username': request.user.username,
            'created_at': datetime.now().isoformat(),
            'used': False
        },
        timeout=300  # 5 минут достаточно для установки соединения
    )

    return JsonResponse({
        'token': token,
        'fastapi_url': settings.FASTAPI_WEBSOCKET_URL
    })