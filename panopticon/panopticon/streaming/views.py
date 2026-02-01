import requests
from django.conf import settings
from django.views import View
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.urls import reverse
import uuid
from django.http import JsonResponse
import json


#TODO можно задудосить стрим сервис если много запросов
@method_decorator(login_required, name='dispatch')
class WebStreamView(View):
    def get(self, request):
        user = request.user
        response = requests.post(f"{settings.VSTREAM_INTERNAL_URL}/sessions",
                                 json={"user_id": user.id,
                                       "test_id": str(uuid.uuid4())},
                                 headers={"X-Api-Key": settings.SECRET_KEY})
                                 # timeout=3)
        data = response.json()
        streaming_session_id = data.get("id")

        relative_offer_url = reverse(viewname='streaming:stream-offer',
                                     kwargs={'streaming_session_id': streaming_session_id})

        relative_stop_url = reverse(viewname='streaming:stream-stop',
                                    kwargs={'streaming_session_id': streaming_session_id})

        full_offer_url = request.build_absolute_uri(relative_offer_url)
        full_stop_url = request.build_absolute_uri(relative_stop_url)

        # print(full_stop_url)

        context = {
            'stop_url': full_stop_url,
            'offer_url': full_offer_url,
            'turn_url': settings.TURN_URL,
            'turn_username': settings.TURN_USERNAME,
            'turn_password': settings.TURN_PASSWORD
        }
        return render(request, 'streaming/streaming.html', context)


@method_decorator(login_required, name='dispatch')
class WebRTCOfferView(View):
    def post(self, request, streaming_session_id):
        try:
            body = json.loads(request.body)

            sdp = body.get("sdp")
            offer_type = body.get("type")

            if not sdp or not offer_type:
                return JsonResponse({"error": "Invalid SDP data"}, status=400)

            vstream_service_offer = f"{settings.VSTREAM_INTERNAL_URL}/stream/{streaming_session_id}/offer"
            # TODO отправка юзера в запросе если надо
            response = requests.post(
                vstream_service_offer,
                json={"sdp": sdp, "type": offer_type},
                headers={"Content-Type": "application/json",
                         "X-Api-Key": settings.SECRET_KEY}
                # timeout=3
            )

            if response.status_code != 200:
                return JsonResponse(
                    {"error": "Backend offer failed", "details": response.text},
                    status=500
                )
            data = response.json()

            return JsonResponse(data)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


class WebRTCStopView(View):
    def post(self, request, streaming_session_id):
        try:
            vstream_service_stop = f"{settings.VSTREAM_INTERNAL_URL}/stream/{streaming_session_id}/stop"

            response = requests.post(
                vstream_service_stop,
                headers={"X-Api-Key": settings.SECRET_KEY}
            )

            if response.status_code != 200:
                return JsonResponse(
                    {"error": "Backend stop failed", "details": response.text},
                    status=500
                )

            return JsonResponse({"status": "stopped"})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)



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