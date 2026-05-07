from django.urls import path
from django.conf.urls.static import static

from django.conf import settings
from .views import IndexView


app_name = 'core'

urlpatterns = [
    path('', IndexView.as_view(), name='dashboard'),
]


# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)