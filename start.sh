#!/bin/sh

# Salir inmediatamente si falla un comando
set -e

# Activar el entorno virtual (opcional pero bueno para establecer rutas)
. .venv/bin/activate || true

# EJECUTAR EL CÓDIGO CON PYTHON
# Usamos python -m gunicorn para que Python sepa dónde encontrar el paquete,
# y 'sh' se encargará de encontrar el ejecutable 'python'.
exec python -m gunicorn cancerpiel.wsgi:application --bind 0.0.0.0:"$PORT"