"""
WSGI config for api_cancer project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application

# ← CORREGIDO
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cancerpiel.api_cancer.api_cancer.settings')

application = get_wsgi_application()
