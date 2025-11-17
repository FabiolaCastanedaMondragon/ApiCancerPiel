#!/bin/sh

# ... (otras líneas)

# EJECUTAR EL CÓDIGO CON PYTHON (Línea Corregida)
exec python -m gunicorn **api_cancer**.wsgi:application --bind 0.0.0.0:"$PORT"