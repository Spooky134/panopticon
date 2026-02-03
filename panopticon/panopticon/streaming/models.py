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
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True, )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="started")

    class Meta:
        managed = False
        db_table = "streaming_session"
        indexes = [
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

    streaming_session = models.ForeignKey(
        StreamingSession,
        on_delete=models.CASCADE,
        db_column="streaming_session_id",
        related_name="video")


    class Meta:
        db_table = "streaming_video"
        #TODO поменять флаг когда будет alembic
        managed = False
        indexes = [
            models.Index(fields=["s3_key"]),
            models.Index(fields=["streaming_session_id"]),
        ]
