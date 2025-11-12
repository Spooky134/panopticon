import uuid
from django.db import models
from django.utils import timezone

class TestingSession(models.Model):
    STATUS_CHOICES = [
        ("started", "Started"),
        ("running", "Running"),
        ("finished", "Finished"),
        ("interrupted", "Interrupted"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.IntegerField()  # FK к users, для простоты как int
    test_id = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="started")
    video_url = models.TextField(null=True, blank=True)           # ссылка на MinIO/S3
    # incidents = models.JSONField(null=True, blank=True)           # список инцидентов от ML
    # ml_metrics = models.JSONField(null=True, blank=True)          # любые метрики/лог
    # meta = models.JSONField(null=True, blank=True)                # свободное поле для доп.данных

    class Meta:
        db_table = "testing_session"
        indexes = [
            models.Index(fields=["user_id"]),
            models.Index(fields=["status"]),
        ]

    def mark_started(self):
        self.started_at = timezone.now()
        self.status = "running"
        self.save(update_fields=["started_at", "status"])

    def mark_finished(self, video_url: str = None, incidents: dict = None, ml_metrics: dict = None):
        self.ended_at = timezone.now()
        self.status = "finished"
        if video_url:
            self.video_url = video_url
        if incidents is not None:
            self.incidents = incidents
        if ml_metrics is not None:
            self.ml_metrics = ml_metrics
        self.save()
