from django.urls import path

from . import views

urlpatterns = [
    path("api/cases/", views.list_cases),
    path("api/cases/<str:case>/detections/", views.detections),
    path("api/cases/<str:case>/predictions/<path:path>", views.predictions_file),
    path("api/cases/<str:case>/raw/<str:filename>", views.raw_video),
    path("api/cases/<str:case>/postprocess/", views.postprocess_case),
    path("api/cases/<str:case>/filtered-summary/", views.filtered_summary),
]
