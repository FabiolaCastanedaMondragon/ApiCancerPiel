# Archivo: detector/urls.py

from django.urls import path
from .views import MetricsView, PredictView # Importaremos estas vistas después

urlpatterns = [
    # URL para ver el F1 Score y Accuracy
    path('metrics/', MetricsView.as_view(), name='metrics'),
    
    # URL para subir la imagen y obtener la predicción
    path('predict/', PredictView.as_view(), name='predict'),
]