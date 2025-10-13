from django.forms import BaseModelForm
from django.shortcuts import redirect, render
from django.http import HttpResponse, HttpRequest
from django.urls import reverse_lazy
from django.views import View
from .forms import ProfileEditForm, RegistrForm, LoginForm, UserEditForm, PasswordChangeForm, CustomPasswordResetForm, PasswordResetConfirmForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.views import LogoutView, LoginView, PasswordChangeView
from django.views.generic import TemplateView, CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import Profile
from django.views.generic.edit import UpdateView, FormView
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView




class AccountRegisterView(CreateView):
    form_class = RegistrForm
    template_name = 'account/authentication/register.html'
    success_url = reverse_lazy("account:profile")

    def form_valid(self, form: BaseModelForm) -> HttpResponse:
        response =  super().form_valid(form)

        username = form.cleaned_data.get("username")
        password = form.cleaned_data.get("password")

        user = authenticate(
            self.request,
            username=username,
            password=password,
        )
        
        login(request=self.request, user=user)

        return response

class AccountLoginView(LoginView):
    form_class = LoginForm
    template_name = "account/authentication/login.html"
    redirect_authenticated_user = True
    success_url = reverse_lazy("account:profile")

    def get_success_url(self):
        return self.success_url
    


class AccountLogoutView(LogoutView):
    next_page = reverse_lazy("account:login")


class ProfileView(LoginRequiredMixin, TemplateView):
    template_name = "account/profile/profile.html"
    login_url = '/account/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user

        profile, created = Profile.objects.get_or_create(user=user)
        context['user'] = user
        context['profile'] = profile

        return context



class ProfileUpdateView(LoginRequiredMixin, View):
    template_name = 'account/profile/profile_edit.html'
    success_url = '/account/profile/'

    def get(self, request, *args, **kwargs):
        user_form = UserEditForm(instance=request.user)

        profile, created = Profile.objects.get_or_create(user=request.user)
        profile_form = ProfileEditForm(instance=profile)

        return render(request, self.template_name, {
            'user_form': user_form,
            'profile_form': profile_form
        })

    def post(self, request, *args, **kwargs):
        user_form = UserEditForm(request.POST, instance=request.user)

        profile, created = Profile.objects.get_or_create(user=request.user)
        profile_form = ProfileEditForm(request.POST, request.FILES, instance=profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            return redirect(self.success_url)
        
        return render(request, self.template_name, {
            'user_form': user_form,
            'profile_form': profile_form
        })

class PasswordChangeView(PasswordChangeView):
    form_class = PasswordChangeForm
    template_name = "account/password/change_password.html"
    success_url = reverse_lazy("account:profile") 


class PasswordResetView(PasswordResetView):
    template_name = 'account/password/password_reset_form.html'
    form_class = CustomPasswordResetForm
    email_template_name = 'account/password/password_reset_email.html'
    success_url = reverse_lazy('password_reset_done')

class PasswordResetDoneView(PasswordResetDoneView):
    template_name = 'account/password/password_reset_done.html'

class PasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'account/password/password_reset_confirm.html'
    form_class = PasswordResetConfirmForm
    success_url = reverse_lazy('password_reset_complete')

class PasswordResetCompleteView(PasswordResetCompleteView):
    template_name = 'account/password/password_reset_complete.html'