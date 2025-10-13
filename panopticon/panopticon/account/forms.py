from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth.forms import PasswordChangeForm, PasswordResetForm
from .models import Profile


class LoginForm(AuthenticationForm):
        username = forms.CharField(
            widget=forms.TextInput(attrs={
                'class': 'form-control', 
                'id': 'username',
                'placeholder': "",
                'required': 'required'
            })
        )
        password = forms.CharField(
            widget=forms.PasswordInput(attrs={
                'class': 'form-control',
                'id': 'password',
                'placeholder': "",
                'required': 'required'
            })
        )


class RegistrForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password']

        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control', 
                'id': 'username',
                'placeholder': "",
                'required': 'required'}),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'id': 'email',
                'placeholder': "",
                'required': 'required'}),
            'password': forms.PasswordInput(attrs={
                'class': 'form-control',
                'id': 'password',
                'placeholder': "",
                'required': 'required'}),
        }

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("This username is already taken. Please choose a different one.")
        return username
    
    def clean_password(self):
        password = self.cleaned_data.get('password')
        try:
            validate_password(password)  # Проверка пароля через встроенные валидаторы
        except forms.ValidationError as e:
            raise forms.ValidationError(e.messages)
        return password

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password"])  # Ensure the password is hashed
        if commit:
            user.save()
            Profile.objects.create(user=user)
        return user
    
class UserEditForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control', 
                'id': 'username',
                'required': 'required'}),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'id': 'email',
                'required': 'required'}),
            'first_name': forms.TextInput(attrs={
                'class': 'form-control',
                'id': 'first_name',}),
            'last_name': forms.TextInput(attrs={
                'class': 'form-control',
                'id': 'last_name',}),
        }

class ProfileEditForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['photo', 'phone']
        widgets = {
            'photo': forms.FileInput(attrs={
                'class': 'form-control', 
                'id': 'uploadImage',
                'type': "file",
                'accept': "image/*",
                'onchange': "changePhoto(event)"}),
            'phone': forms.TextInput(attrs={
                'class': 'form-control',
                'id': 'phone',}),
        }



class PasswordChangeForm(PasswordChangeForm):
    old_password = forms.CharField(
        label="Current password",
        widget=forms.PasswordInput(attrs={"class": "form-control", "placeholder": ""}),
    )
    new_password1 = forms.CharField(
        label="New password",
        widget=forms.PasswordInput(attrs={"class": "form-control", "placeholder": ""}),
    )
    new_password2 = forms.CharField(
        label="Confirm password",
        widget=forms.PasswordInput(attrs={"class": "form-control", "placeholder": ""}),
    )


class CustomPasswordResetForm(PasswordResetForm):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control custom-input',
            'placeholder': '',
        })
    )

class PasswordResetConfirmForm(forms.Form):
    new_password1 = forms.CharField(
        label="New password",
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": ""}),
    )
    new_password2 = forms.CharField(
        label="Confirm password",
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": ""}),
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control custom-input',
            'placeholder': '',
        })
    )