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


@method_decorator(login_required, name='dispatch')
class WebStreamView(View):
    def get(self, request):
        user = request.user

        test_id = uuid.uuid4()
        # Уникальный session_id для этого теста
        testing_session_id = uuid.uuid4()

        testing_session = TestingSession.objects.create(
            id=testing_session_id,
            user_id=user.id,
            test_id=test_id,
            status="started",
        )

        payload = {
            "user_id": user.id,
            "session_id": str(testing_session_id),
            "exp": timezone.now() + datetime.timedelta(minutes=5)
        }
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

        context = {
            'vstream_service_url': f'http://{settings.VSTREAM_SERVICE_HOST}/video-testing/test/stream/',
            'auth_token': token,
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