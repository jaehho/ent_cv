from django.contrib import admin
from django.urls import include, path

from . import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("auth/session/", views.session_view),
    path("auth/login/", views.login_view),
    path("auth/logout/", views.logout_view),
    path("", include("api.urls")),
]
