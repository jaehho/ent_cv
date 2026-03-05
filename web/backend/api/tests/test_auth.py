import json

import pytest


@pytest.mark.django_db
class TestSessionView:
    def test_unauthenticated_returns_401(self, anon_client):
        resp = anon_client.get("/auth/session/")
        assert resp.status_code == 401
        data = resp.json()
        assert data["authenticated"] is False

    def test_authenticated_returns_200(self, auth_client):
        resp = auth_client.get("/auth/session/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is True
        assert data["username"] == "testuser"

    def test_sets_csrf_cookie(self, anon_client):
        resp = anon_client.get("/auth/session/")
        assert "entcv_csrftoken" in resp.cookies


@pytest.mark.django_db
class TestLogin:
    def test_valid_credentials(self, anon_client, user):
        resp = anon_client.post(
            "/auth/login/",
            data=json.dumps({"username": "testuser", "password": "testpass123"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.json()["authenticated"] is True

    def test_invalid_credentials(self, anon_client, user):
        resp = anon_client.post(
            "/auth/login/",
            data=json.dumps({"username": "testuser", "password": "wrong"}),
            content_type="application/json",
        )
        assert resp.status_code == 403

    def test_missing_fields(self, anon_client):
        resp = anon_client.post(
            "/auth/login/",
            data=json.dumps({"username": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_invalid_json(self, anon_client):
        resp = anon_client.post(
            "/auth/login/",
            data="not json",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_get_not_allowed(self, anon_client):
        resp = anon_client.get("/auth/login/")
        assert resp.status_code == 405


@pytest.mark.django_db
class TestLogout:
    def test_logout_clears_session(self, auth_client):
        # Verify authenticated first
        resp = auth_client.get("/auth/session/")
        assert resp.status_code == 200

        # Logout
        resp = auth_client.post("/auth/logout/")
        assert resp.status_code == 200
        assert resp.json()["authenticated"] is False

        # Verify no longer authenticated
        resp = auth_client.get("/auth/session/")
        assert resp.status_code == 401
