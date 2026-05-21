import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

# Add project root to sys.path so `from ent_cv.modeling...` works
_PROJECT_ROOT = str(BASE_DIR.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    if os.environ.get("DJANGO_DEBUG", "True").lower() in ("true", "1", "yes"):
        SECRET_KEY = "django-insecure-dev-only-key-do-not-use-in-production"
    else:
        raise ValueError("DJANGO_SECRET_KEY environment variable is required when DEBUG=False")

DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() in ("true", "1", "yes")

ALLOWED_HOSTS = os.environ.get(
    "DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,entcv.jaehho.com"
).split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "api",
]

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MIDDLEWARE = [
    # JSON-only gzip. The detection JSON for a 1h surgery is 50-100MB; gzip
    # compresses it ~5-10×, cutting case-switch wire time accordingly. The
    # vanilla GZipMiddleware also compresses video streams, which breaks
    # range requests and stalls playback into buffering — see api.middleware.
    "api.middleware.JsonGZipMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {}
_DATABASE_URL = os.environ.get("DATABASE_URL")
if _DATABASE_URL:
    # PostgreSQL: DATABASE_URL=postgres://user:pass@host:port/dbname
    import re

    m = re.match(
        r"postgres(?:ql)?://(?P<user>[^:]+):(?P<pass>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)/(?P<name>.+)",
        _DATABASE_URL,
    )
    if m:
        DATABASES["default"] = {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": m.group("name"),
            "USER": m.group("user"),
            "PASSWORD": m.group("pass"),
            "HOST": m.group("host"),
            "PORT": m.group("port"),
        }
    else:
        raise ValueError(f"Cannot parse DATABASE_URL: {_DATABASE_URL}")
else:
    DATABASES["default"] = {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

# Session settings — cookie shared with the Vite dev server on the same host
SESSION_COOKIE_NAME = "entcv_sessionid"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_AGE = 86400 * 7  # 1 week

# CSRF — trust the Vite dev server origin
CSRF_COOKIE_NAME = "entcv_csrftoken"
CSRF_COOKIE_HTTPONLY = False  # JS needs to read the token
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_TRUSTED_ORIGINS = os.environ.get(
    "CSRF_TRUSTED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,https://entcv.jaehho.com",
).split(",")

# @login_required returns 401 JSON instead of redirecting to a login page
LOGIN_URL = "/auth/session/"

# No static files needed — the Vite app handles the frontend
STATIC_URL = "static/"

# ── Production security hardening ────────────────────────────────────────
if not DEBUG:
    # Trust X-Forwarded-Proto from the reverse proxy so Django knows the
    # original request was HTTPS (prevents infinite SECURE_SSL_REDIRECT loop).
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = True
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
