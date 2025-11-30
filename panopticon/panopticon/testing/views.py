import requests
from django.shortcuts import render
from django.views import View
from django.conf import settings
import jwt, datetime
from django.utils import timezone
from django.conf import settings
from django.views import View
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from .models import TestingSession
import uuid


#TODO можно задудосить стрим сервис если много запросов
@method_decorator(login_required, name='dispatch')
class WebStreamView(View):
    def get(self, request):
        user = request.user
        response = requests.post(f"{settings.VSTREAM_INTERNAL_URL}sessions/",
                                 json={"user_id": user.id},
                                 headers={"X-Api-Key": settings.SECRET_KEY})
                                 # timeout=3)
        data = response.json()

        context = {
            'vstream_service_url': settings.VSTREAM_WEBRTC_URL,
            'auth_token': data.get("token"),
            'turn_url': settings.TURN_URL,
            'turn_username': settings.TURN_USERNAME,
            'turn_password': settings.TURN_PASSWORD,
        }
        return render(request, 'testing/stream.html', context)

#TODO обработчик нажатия кнопки чтобы создать сессию в бд + в шаблоне тоже добавить обработку
#TODO переписать под шаблоны django

# class WebStreamView(View):
#     """View для отображения WebRTC клиента"""
#
#     def get(self, request):
#         context = {
#             'vstream_service_url': f'http://{settings.VSTREAM_SERVICE_HOST}/video-testing/test/stream/',  # например: 'localhost:8000'
#             'turn_url': settings.TURN_URL,
#             'turn_username': settings.TURN_USERNAME,
#             'turn_password': settings.TURN_PASSWORD,
#         }
#         return render(request, 'testing/stream.html', context)