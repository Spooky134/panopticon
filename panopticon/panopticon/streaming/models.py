import uuid
from django.db import models
from django.db.models.functions import Now
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
    created_at = models.DateTimeField(db_default=Now())
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="started")

    class Meta:
        managed = False
        db_table = "session"
        managed = True
        indexes = [
            models.Index(fields=["status"]),
        ]



class StreamingVideo(models.Model):
    # id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    s3_key = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    duration = models.IntegerField(null=True, blank=True)
    fps = models.IntegerField(null=True, blank=True)
    file_size = models.IntegerField(null=True, blank=True)
    mime_type = models.CharField(max_length=100, null=True, blank=True)
    meta = models.JSONField(null=True, blank=True, default=dict)

    created_at = models.DateTimeField(db_default=Now())

    session = models.ForeignKey(
        StreamingSession,
        on_delete=models.CASCADE,
        db_column="session_id",
        related_name="videos")


    class Meta:
        db_table = "video"
        #TODO поменять флаг когда будет alembic
        managed = True
        indexes = [
            models.Index(fields=["s3_key"]),
            models.Index(fields=["session_id"]),
        ]
