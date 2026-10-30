"""Gunicorn and local Flask entry point."""

from spark_credit.web import create_app

app = create_app()
