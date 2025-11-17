"""
WSGI config for api_cancer project.
"""

import os
from django.core.wsgi import get_wsgi_application

# Render NECESITA la ruta 100% correcta al settings.py
os.environ.setdefault(
    'DJANGO_SETTINGS_MODULE',
    'cancerpiel.api_cancer.api_cancer.settings'
)

application = get_wsgi_application()
