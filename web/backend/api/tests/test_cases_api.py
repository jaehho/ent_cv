import json

import pytest


@pytest.mark.django_db
class TestListCases:
    def test_unauthenticated_returns_401(self, anon_client):
        resp = anon_client.get("/api/cases/")
        assert resp.status_code == 401

    def test_empty_predictions_dir(self, auth_client, predictions_dir):
        # Remove the test_case so dir is empty of valid cases
        import shutil

        shutil.rmtree(predictions_dir / "test_case")
        resp = auth_client.get("/api/cases/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_cases_with_detections(self, auth_client, predictions_dir):
        resp = auth_client.get("/api/cases/")
        assert resp.status_code == 200
        data = resp.json()
        assert "test_case" in data


@pytest.mark.django_db
class TestDetections:
    def test_unauthenticated_returns_401(self, anon_client):
        resp = anon_client.get("/api/cases/test_case/detections/")
        assert resp.status_code == 401

    def test_invalid_case_name(self, auth_client):
        resp = auth_client.get("/api/cases/bad.name/detections/")
        assert resp.status_code == 400

    def test_nonexistent_case(self, auth_client, predictions_dir):
        resp = auth_client.get("/api/cases/nonexistent/detections/")
        assert resp.status_code == 404


@pytest.mark.django_db
class TestRawVideo:
    def test_unauthenticated_returns_401(self, anon_client):
        resp = anon_client.get("/api/cases/test_case/raw/part1.mp4")
        assert resp.status_code == 401

    def test_invalid_case_name(self, auth_client):
        resp = auth_client.get("/api/cases/bad.name/raw/part1.mp4")
        assert resp.status_code == 400

    def test_nonexistent_file(self, auth_client, raw_dir):
        resp = auth_client.get("/api/cases/test_case/raw/nonexistent.mp4")
        assert resp.status_code == 404

    def test_path_traversal_blocked(self, auth_client, raw_dir):
        resp = auth_client.get("/api/cases/bad..name/raw/part1.mp4")
        assert resp.status_code == 400

    def test_serves_video(self, auth_client, raw_dir):
        resp = auth_client.get("/api/cases/test_case/raw/part1.mp4")
        assert resp.status_code == 200
        assert resp["Content-Type"] == "video/mp4"

    def test_range_request(self, auth_client, raw_dir):
        resp = auth_client.get(
            "/api/cases/test_case/raw/part1.mp4",
            HTTP_RANGE="bytes=0-99",
        )
        assert resp.status_code == 206
        assert "Content-Range" in resp


@pytest.mark.django_db
class TestPostprocess:
    def test_unauthenticated_returns_401(self, anon_client):
        resp = anon_client.post(
            "/api/cases/test_case/postprocess/",
            data=json.dumps({"method": "run_length"}),
            content_type="application/json",
        )
        assert resp.status_code == 401

    def test_invalid_method(self, auth_client, predictions_dir):
        resp = auth_client.post(
            "/api/cases/test_case/postprocess/",
            data=json.dumps({"method": "invalid_method"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_invalid_json(self, auth_client, predictions_dir):
        resp = auth_client.post(
            "/api/cases/test_case/postprocess/",
            data="not json",
            content_type="application/json",
        )
        assert resp.status_code == 400


@pytest.mark.django_db
class TestFilteredSummary:
    def test_unauthenticated_returns_401(self, anon_client):
        resp = anon_client.get("/api/cases/test_case/filtered-summary/")
        assert resp.status_code == 401

    def test_no_summary_returns_404(self, auth_client, predictions_dir):
        resp = auth_client.get("/api/cases/test_case/filtered-summary/")
        assert resp.status_code == 404

    def test_returns_summary(self, auth_client, predictions_dir):
        summary = {"instruments": ["Forceps"], "total_time": 10.0}
        (predictions_dir / "test_case" / "filtered_summary.json").write_text(json.dumps(summary))
        resp = auth_client.get("/api/cases/test_case/filtered-summary/")
        assert resp.status_code == 200
        assert resp.json()["instruments"] == ["Forceps"]
