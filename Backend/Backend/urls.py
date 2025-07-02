"""
URL configuration for the PronunciationApp application of this project.
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("PronunciationApp.urls")),
]