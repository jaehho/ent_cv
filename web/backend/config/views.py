import json

from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.http import require_GET, require_POST


@require_GET
def session_view(request):
    """Return current auth status + CSRF token."""
    get_token(request)  # ensure CSRF cookie is set
    if request.user.is_authenticated:
        return JsonResponse({"authenticated": True, "username": request.user.username})
    return JsonResponse({"authenticated": False}, status=401)


@require_POST
def login_view(request):
    """Authenticate with username/password (JSON body)."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    username = body.get("username", "").strip()
    password = body.get("password", "")

    if not username or not password:
        return JsonResponse({"error": "Username and password required"}, status=400)

    user = authenticate(request, username=username, password=password)
    if user is None:
        return JsonResponse({"error": "Invalid credentials"}, status=403)

    login(request, user)
    return JsonResponse({"authenticated": True, "username": user.username})


@require_POST
def logout_view(request):
    logout(request)
    return JsonResponse({"authenticated": False})
