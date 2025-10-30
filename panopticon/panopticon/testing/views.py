from django.shortcuts import render
from django.views import View
from django.conf import settings
import jwt, datetime
from django.conf import settings
from django.views import View
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
import uuid


@method_decorator(login_required, name='dispatch')
class WebStreamView(View):
    def get(self, request):
        user = request.user
        # Уникальный session_id для этого теста
        session_id = str(uuid.uuid4())

        payload = {
            "user_id": user.id,
            "session_id": session_id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
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