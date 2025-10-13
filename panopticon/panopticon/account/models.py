from django.db import models

from django.conf import settings

# Create your models here.
class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, 
                                    on_delete=models.CASCADE)
    photo = models.ImageField(upload_to='avatars/%Y/%m/%d/', blank=True)
    phone = models.CharField(max_length=15, blank=True)

    def __str__(self):
        return f'Profile for user {self.user.username}'