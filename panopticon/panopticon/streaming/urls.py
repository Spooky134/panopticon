from django.conf import settings
from django.urls import path
from django.conf.urls.static import static
from .views import WebStreamView, WebRTCOfferView, WebRTCStopView


app_name = 'streaming'

urlpatterns = [
    # path('login/', AccountLoginView.as_view(), name='login'),
    # path('video/', VideoStreamView.as_view(), name='video_stream'),
    # path('', VideoStreamView.as_view(), name='video_stream'),
    path("stream/<uuid:streaming_session_id>/offer/", WebRTCOfferView.as_view(), name="stream-offer"),
    path("stream/<uuid:streaming_session_id>/stop/", WebRTCStopView.as_view(), name="stream-stop"),
    path('', WebStreamView.as_view(), name='streaming'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)