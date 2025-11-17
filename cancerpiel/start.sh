#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Try activating the venv in case the path is slightly different
source .venv/bin/activate || echo "No .venv found in root; proceeding."

# Execute the Gunicorn command using the Python interpreter
exec python -m gunicorn cancerpiel.wsgi:application --bind 0.0.0.0:$PORT