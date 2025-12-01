import uuid
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class StreamingSession(models.Model):
    STATUS_CHOICES = [
        ("started", "Started"),
        ("running", "Running"),
        ("finished", "Finished"),
        ("interrupted", "Interrupted"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User,
                             on_delete=models.CASCADE,
                             related_name='streaming_sessions')
    test_id = models.UUIDField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True, )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="started")

    # video = models.OneToOneField("TestingVideo", on_delete=models.SET_NULL, null=True, blank=True, related_name="testing_session")


    # incidents = models.JSONField(null=True, blank=True)           # список инцидентов от ML
    # ml_metrics = models.JSONField(null=True, blank=True)          # любые метрики/лог
    # meta = models.JSONField(null=True, blank=True)                # свободное поле для доп.данных

    class Meta:
        db_table = "streaming_sessions"
        indexes = [
            models.Index(fields=["user_id"]),
            models.Index(fields=["status"]),
        ]



class StreamingVideo(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    s3_key = models.CharField(max_length=500)
    s3_bucket = models.CharField(max_length=255)
    duration = models.IntegerField(null=True, blank=True)
    file_size = models.IntegerField(null=True, blank=True)
    mime_type = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(null=True, blank=True)

    streaming_session = models.OneToOneField(StreamingSession,
                                             on_delete=models.CASCADE,
                                             db_column="streaming_session_id",
                                             related_name="video")

    class Meta:
        db_table = "streaming_videos"
        #TODO поменять флаг когда будет alembic
        # managed = False
        indexes = [
            models.Index(fields=["s3_key"]),
            models.Index(fields=["streaming_session_id"]),
        ]
