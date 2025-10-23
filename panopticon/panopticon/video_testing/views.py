from django.shortcuts import render
from django.views import View
from django.conf import settings


class WebStreamView(View):
    """View для отображения WebRTC клиента"""

    def get(self, request):
        context = {
            'vstream_service_url': f'http://{settings.VSTREAM_SERVICE_HOST}/video-testing/test/stream/',  # например: 'localhost:8000'
            'turn_url': settings.TURN_URL,
            'turn_username': settings.TURN_USERNAME,
            'turn_password': settings.TURN_PASSWORD,
        }
        return render(request, 'video_testing/stream.html', context)



# class VideoStreamView(TemplateView):
#     template_name = "video_testing/stream.html"
#     # login_url = '/account/login/'
#
#     # def get_context_data(self, **kwargs):
#     #     context = super().get_context_data(**kwargs)
#     #     user = self.request.user
#     #
#     #     profile, created = Profile.objects.get_or_create(user=user)
#     #     context['user'] = user
#     #     context['profile'] = profile
#     #
#     #     return context