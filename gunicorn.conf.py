"""Conservative serving configuration for a small demo instance."""

# pylint: disable=invalid-name
# Gunicorn requires these lowercase setting names.
import os

bind = "0.0.0.0:" + os.environ.get("PORT", "8000")
workers = 1
threads = 2
timeout = 60
accesslog = "-"
errorlog = "-"
