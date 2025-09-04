# app/celery_app.py
from __future__ import annotations
import os
from celery import Celery

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "bis_scenarios",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    timezone="Asia/Seoul",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    worker_hijack_root_logger=False,
    task_ignore_result=True,
    task_acks_late=True,
    task_time_limit=60 * 30,

    include=["app.tasks.scenario_tasks"],
)