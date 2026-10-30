"""Portable inference from a verified Spark pipeline export; no Java at runtime."""

import hashlib
import json
from pathlib import Path
import numpy as np
from spark_credit.schema import FEATURES, LABELS, validate_records


def probabilities(values: object, model: dict) -> np.ndarray:
    """Match Spark imputation, centering/scaling and the stable logistic sigmoid."""
    array = np.asarray(values, dtype=float)
    medians = np.asarray(model["medians"])
    mean = np.asarray(model["mean"])
    deviation = np.asarray(model["std"])
    array = np.where(np.isnan(array), medians, array)
    # Spark StandardScaler returns zero for a zero-variance coordinate.
    scaled = np.divide(
        array - mean, deviation, out=np.zeros_like(array), where=deviation != 0
    )
    margin = scaled @ np.asarray(model["coefficients"]) + model["intercept"]
    return np.exp(-np.logaddexp(0, -margin))


class PredictionService:
    """Load only finite, schema-matched numeric JSON from a reviewed release."""

    def __init__(self, model_dir: str | Path):
        folder = Path(model_dir)
        self.report = json.loads((folder / "metrics.json").read_text())
        artifact = folder / "portable-model.json"
        if (
            hashlib.sha256(artifact.read_bytes()).hexdigest()
            != self.report["artifact_sha256"]
        ):
            raise ValueError("Model integrity check failed.")
        self.model = json.loads(artifact.read_text())
        if self.model["features"] != list(FEATURES):
            raise ValueError("Model feature order is incompatible.")
        for key in ["medians", "mean", "std", "coefficients"]:
            array = np.asarray(self.model[key], dtype=float)
            if array.shape != (len(FEATURES),) or not np.isfinite(array).all():
                raise ValueError("Invalid numeric model parameters.")
        if (np.asarray(self.model["std"]) < 0).any() or not np.isfinite(
            self.model["intercept"]
        ):
            raise ValueError("Invalid scaling or intercept.")
        self.sample = json.loads((folder / "sample.json").read_text())

    def predict(self, records: list, threshold: float | None = None) -> list[dict]:
        """Return uncalibrated historical-source scores, never a lending decision."""
        if threshold is None:
            threshold = self.report["threshold"]
        try:
            if isinstance(threshold, bool):
                raise ValueError
            threshold = float(threshold)
        except (ValueError, TypeError) as exc:
            raise ValueError("Threshold must be numeric.") from exc
        if not 0.01 <= threshold <= 0.99:
            raise ValueError("Threshold must be between 0.01 and 0.99.")
        frame = validate_records(records)
        scores = probabilities(frame.to_numpy(), self.model)
        results = []
        for score in scores:
            label = "1" if score >= threshold else "0"
            results.append(
                {
                    "category": label,
                    "description": LABELS[label],
                    "threshold": threshold,
                    "scores": {
                        "0": round(float(1 - score), 6),
                        "1": round(float(score), 6),
                    },
                }
            )
        return results
