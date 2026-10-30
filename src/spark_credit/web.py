"""Flask application factory, HTML screens and bounded JSON/CSV endpoints."""

import csv
import io
import logging
import os
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from spark_credit.schema import FEATURES
from spark_credit.service import PredictionService

ROOT = Path.cwd()


def create_app(config=None):
    """Create an application with explicit model location and request limits."""
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=1024 * 1024,
        MODEL_DIR=os.environ.get("MODEL_DIR", str(ROOT / "models")),
    )
    app.config.update(config or {})
    logging.basicConfig(level=logging.INFO)
    service = PredictionService(app.config["MODEL_DIR"])
    app.extensions["prediction_service"] = service

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.get("/")
    def home():
        return render_template(
            "index.html",
            features=FEATURES,
            values=service.sample,
            result=None,
            error=None,
            report=service.report,
        )

    @app.post("/predict")
    def predict_form():
        values = {field: request.form.get(field, "") for field in FEATURES}
        try:
            result = service.predict([values], request.form.get("threshold"))[0]
        except ValueError as exc:
            return (
                render_template(
                    "index.html",
                    features=FEATURES,
                    values=values,
                    result=None,
                    error=str(exc),
                    report=service.report,
                ),
                400,
            )
        return render_template(
            "index.html",
            features=FEATURES,
            values=values,
            result=result,
            error=None,
            report=service.report,
        )

    @app.get("/health/ready")
    def ready():
        return jsonify(
            status="ready",
            version=service.report["version"],
            revision=os.environ.get("RENDER_GIT_COMMIT", "local"),
        )

    @app.get("/api/metrics")
    def report():
        return jsonify(service.report)

    @app.post("/api/predict")
    def predict_json():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "records" not in payload:
            raise ValueError('Expected JSON object with a "records" list.')
        return jsonify(
            predictions=service.predict(payload["records"], payload.get("threshold")),
            version=service.report["version"],
        )

    @app.get("/sample.csv")
    def sample_csv():
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(FEATURES))
        writer.writeheader()
        writer.writerow(service.sample)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=sample.csv"},
        )

    @app.post("/predict-csv")
    def predict_csv():
        uploaded = request.files.get("file")
        if uploaded is None:
            raise ValueError("Upload a CSV in the file field.")
        try:
            frame = pd.read_csv(uploaded, nrows=501)
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            raise ValueError(
                "CSV could not be read. Use UTF-8 with a header row."
            ) from exc
        records = (
            frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
        )
        predictions = service.predict(records, request.form.get("threshold"))
        # Export only generated values: never echo untrusted spreadsheet formulas.
        rows = [
            {
                "row": idx,
                "category": item["category"],
                "threshold": item["threshold"],
                **item["scores"],
            }
            for idx, item in enumerate(predictions, 1)
        ]
        return Response(
            pd.DataFrame(rows).to_csv(index=False),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    @app.errorhandler(ValueError)
    def invalid_input(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.error("Unhandled application error", exc_info=error)
        return jsonify(error="Unexpected server error. Please try again."), 500

    return app
