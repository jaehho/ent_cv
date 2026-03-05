import json

from django.contrib.auth.models import User
from django.test import Client
import pytest


@pytest.fixture
def user(db):
    return User.objects.create_user(username="testuser", password="testpass123")


@pytest.fixture
def auth_client(user):
    client = Client()
    client.login(username="testuser", password="testpass123")
    return client


@pytest.fixture
def anon_client():
    return Client()


@pytest.fixture
def predictions_dir(tmp_path, monkeypatch):
    """Create a temporary predictions directory with sample data."""
    case_dir = tmp_path / "test_case"
    case_dir.mkdir()

    detections = [
        {
            "frame": 0,
            "class": 0,
            "name": "Forceps",
            "confidence": 0.95,
            "box": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
        },
        {
            "frame": 1,
            "class": 1,
            "name": "Suction",
            "confidence": 0.87,
            "box": {"x1": 50, "y1": 60, "x2": 150, "y2": 250},
        },
    ]
    (case_dir / "detections.json").write_text(json.dumps(detections))
    (case_dir / "metadata.json").write_text(json.dumps({"fps": 30, "total_frames": 2}))

    monkeypatch.setattr("api.views.PREDICTIONS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """Create a temporary raw directory with a dummy video file."""
    case_dir = tmp_path / "test_case"
    case_dir.mkdir(exist_ok=True)
    # Create a small dummy file to serve as a "video"
    dummy_video = case_dir / "part1.mp4"
    dummy_video.write_bytes(b"\x00" * 1024)

    monkeypatch.setattr("api.views.RAW_DIR", tmp_path)
    return tmp_path
