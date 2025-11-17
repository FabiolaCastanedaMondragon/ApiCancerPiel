"""
Django settings for api_cancer project.
"""

from pathlib import Path
import os

# BASE_DIR apunta a /cancerpiel/api_cancer/api_cancer
BASE_DIR = Path(__file__).resolve().parent.parent

# PROJECT_ROOT apunta a /cancerpiel/
PROJECT_ROOT = BASE_DIR.parent.parent

SECRET_KEY = 'django-insecure-z(^3p0^y31g!n)5$j2%i@876c0*qj5!s&m5!%6n0!50=i'

# Mantienes DEBUG=True (tanto local como Render)
DEBUG = True

# Render acepta '*'
ALLOWED_HOSTS = ['*']

# Necesario en Render para formularios POST
CSRF_TRUSTED_ORIGINS = [
    'https://*.onrender.com',
    'http://localhost',
    'http://127.0.0.1'
]


# -----------------------------------------------------------
# APPS
# -----------------------------------------------------------
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'rest_framework',

    # app exacta
    'cancerpiel.api_cancer.detector',
]


# -----------------------------------------------------------
# MIDDLEWARE
# -----------------------------------------------------------
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


# -----------------------------------------------------------
# ROOT URL
# -----------------------------------------------------------
ROOT_URLCONF = 'cancerpiel.api_cancer.api_cancer.urls'


# -----------------------------------------------------------
# TEMPLATES
# -----------------------------------------------------------
# Tus templates están en /cancerpiel/templates/
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(PROJECT_ROOT, 'templates'),
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]


# -----------------------------------------------------------
# WSGI
# -----------------------------------------------------------
WSGI_APPLICATION = 'cancerpiel.api_cancer.api_cancer.wsgi.application'


# -----------------------------------------------------------
# DATABASE
# -----------------------------------------------------------
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(PROJECT_ROOT, 'db.sqlite3'),  # ← corregido
    }
}


# -----------------------------------------------------------
# PASSWORD VALIDATORS
# -----------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# -----------------------------------------------------------
# LANG - TIMEZONE
# -----------------------------------------------------------
LANGUAGE_CODE = 'es-mx'
TIME_ZONE = 'America/Mexico_City'

USE_I18N = True
USE_TZ = True


# -----------------------------------------------------------
# STATICFILES (Render obligatorio)
# -----------------------------------------------------------
STATIC_URL = '/static/'

# Render recolecta aquí:
STATIC_ROOT = os.path.join(PROJECT_ROOT, 'staticfiles')
os.makedirs(STATIC_ROOT, exist_ok=True)

# Carpeta static opcional local:
STATICFILES_DIRS = [
    os.path.join(PROJECT_ROOT, 'static'),
]


DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
