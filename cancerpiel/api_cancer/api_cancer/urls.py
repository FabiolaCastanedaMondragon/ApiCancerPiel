# Archivo: api_cancer/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # Conecta la aplicación detector bajo /api/
    path('api/', include('detector.urls')), 
]
# (Guardar y cerrar)