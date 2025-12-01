import uuid
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class TestingSession(models.Model):
    STATUS_CHOICES = [
        ("started", "Started"),
        ("running", "Running"),
        ("finished", "Finished"),
        ("interrupted", "Interrupted"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,  # стандартная модель пользователя
        on_delete=models.CASCADE,
        related_name='testing_sessions'
    )
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
        db_table = "testing_sessions"
        indexes = [
            models.Index(fields=["user_id"]),
            models.Index(fields=["status"]),
        ]

    def mark_started(self):
        self.started_at = timezone.now()
        self.status = "running"
        self.save(update_fields=["started_at", "status"])

    def mark_finished(self, video=None, incidents: dict = None, ml_metrics: dict = None):
        self.ended_at = timezone.now()
        self.status = "finished"
        if video:
            self.video = video
        if incidents is not None:
            self.incidents = incidents
        if ml_metrics is not None:
            self.ml_metrics = ml_metrics
        self.save()


class TestingVideo(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    s3_key = models.CharField(max_length=500)
    s3_bucket = models.CharField(max_length=255)
    duration = models.IntegerField(null=True, blank=True)
    file_size = models.IntegerField(null=True, blank=True)
    mime_type = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(null=True, blank=True)

    testing_session = models.OneToOneField(TestingSession, on_delete=models.CASCADE, db_column="testing_session_id", related_name="video")

    class Meta:
        db_table = "testing_videos"
        #TODO поменять флаг когда будет alembic
        # managed = False
        indexes = [
            models.Index(fields=["s3_key"]),
            models.Index(fields=["testing_session_id"]),
        ]
