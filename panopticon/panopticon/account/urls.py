from django.urls import path
from django.conf.urls.static import static

from django.conf import settings
from .views import AccountRegisterView,\
                    AccountLoginView,\
                    AccountLogoutView,\
                    PasswordChangeView,\
                    ProfileView,\
                    ProfileUpdateView,\
                    PasswordResetView,\
                    PasswordResetDoneView,\
                    PasswordResetConfirmView,\
                    PasswordResetCompleteView


app_name = 'account'

urlpatterns = [
    path('login/', AccountLoginView.as_view(), name='login'),
    path('register/', AccountRegisterView.as_view(), name='register'),
    path('logout/', AccountLogoutView.as_view(), name='logout'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('profile/edit/', ProfileUpdateView.as_view(), name='profile_edit'),
    path('change-password/', PasswordChangeView.as_view(), name='change_password'),

    path('password-reset/', PasswordResetView.as_view(), name='password_reset'),
    path('password-reset/done/', PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uid64>/<token>/', PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset/done/', PasswordResetCompleteView.as_view(), name="password_reset_complete"),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)