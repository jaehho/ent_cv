from django.urls import path

from . import views

urlpatterns = [
    path("auth/session/", views.session_view),
    path("auth/login/", views.login_view),
    path("auth/logout/", views.logout_view),
]
