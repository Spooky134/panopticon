from django.conf import settings
from django.urls import path
from django.conf.urls.static import static
from .views import WebStreamView


app_name = 'testing'

urlpatterns = [
    # path('login/', AccountLoginView.as_view(), name='login'),
    # path('video/', VideoStreamView.as_view(), name='video_stream'),
    # path('', VideoStreamView.as_view(), name='video_stream'),
    path('', WebStreamView.as_view(), name='video_stream'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)