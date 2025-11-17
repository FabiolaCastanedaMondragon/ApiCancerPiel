# Archivo: api_cancer/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),

    # ← CORREGIDO
    path('api/', include('cancerpiel.api_cancer.detector.urls')),
]
