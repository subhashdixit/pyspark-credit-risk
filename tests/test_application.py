"""Exercise real exported-model inference and the shared HTTP contract."""

import json
import shutil
import numpy as np
import pytest
from spark_credit.schema import FEATURES, validate_records
from spark_credit.service import PredictionService, probabilities
from spark_credit.web import create_app


@pytest.fixture
def service():
    """Read the actual released Spark parameters."""
    return PredictionService("models")


@pytest.fixture
def client():
    """Use real model-backed routes."""
    return create_app({"TESTING": True}).test_client()


def test_export_verified_against_spark(service):
    """The training release must pass independent Spark-transform parity."""
    assert service.report["export_max_abs_error"] < 1e-9
    assert service.report["missing_probe_abs_error"] < 1e-9
    assert "UCI 350" in service.report["dataset"]
    assert "SEX" not in FEATURES and "AGE" not in FEATURES


def test_sample_and_missing_values(service):
    """Synthetic median input and an explicitly missing field remain scoreable."""
    result = service.predict([service.sample])[0]
    assert abs(sum(result["scores"].values()) - 1) < 1e-6
    sample = dict(service.sample, LIMIT_BAL=None)
    assert service.predict([sample])[0]["category"] in ["0", "1"]


@pytest.mark.parametrize("value", [True, "nan", "inf", -1e12, "abc"])
def test_bad_feature_values(service, value):
    """Reject nonnumeric, boolean, nonfinite and extreme amounts."""
    with pytest.raises(ValueError):
        service.predict([dict(service.sample, LIMIT_BAL=value)])


@pytest.mark.parametrize("threshold", [True, 0, 1, "nan", "abc"])
def test_invalid_threshold(service, threshold):
    """Thresholds are an explicit bounded operating choice."""
    with pytest.raises(ValueError):
        service.predict([service.sample], threshold)


def test_missing_columns_and_batch_bound(service):
    """Malformed rows and unbounded batches cannot trigger inference."""
    for rows in [[], [{}], [service.sample] * 501, "text"]:
        with pytest.raises(ValueError):
            service.predict(rows)
    with pytest.raises(ValueError):
        validate_records([{key: None for key in FEATURES}])


def test_integer_status_code(service):
    """A repayment code is a categorical code, not a fractional amount."""
    with pytest.raises(ValueError):
        service.predict([dict(service.sample, PAY_0=0.5)])


def test_stable_sigmoid_and_zero_std():
    """Extreme margins remain finite and zero variance contributes zero."""
    model = {
        "medians": [0, 0],
        "mean": [0, 0],
        "std": [1, 0],
        "coefficients": [1000, 1000],
        "intercept": 0,
    }
    score = probabilities([[1, 1], [-1, 1], [np.nan, 1]], model)
    assert np.allclose(score, [1, 0, 0.5])


def test_model_tampering_rejected(tmp_path):
    """A mismatched JSON artifact must fail before serving."""
    shutil.copytree("models", tmp_path / "models")
    target = tmp_path / "models/portable-model.json"
    target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        PredictionService(tmp_path / "models")


def test_http_single_csv_and_health(client, service):
    """All user entry points use one real source-label model."""
    assert client.get("/").status_code == 200
    response = client.post("/api/predict", json={"records": [service.sample]})
    assert response.status_code == 200
    assert response.json["predictions"] == service.predict([service.sample])
    assert client.get("/sample.csv").status_code == 200
    assert client.get("/health/ready").json["status"] == "ready"
    assert client.post("/predict", data=service.sample).status_code == 200
    assert client.post("/api/predict", json={"records": [{}]}).status_code == 400
    assert client.post("/predict-csv").status_code == 400
    assert client.get("/").headers["X-Content-Type-Options"] == "nosniff"


def test_csv_scoring(client):
    """Downloaded sample scores through the multipart endpoint."""
    import io

    sample = client.get("/sample.csv").data
    response = client.post(
        "/predict-csv", data={"file": (io.BytesIO(sample), "sample.csv")}
    )
    assert response.status_code == 200 and b"category" in response.data


def test_metrics_dont_claim_home_credit(client):
    """Expose provenance and measured results in the machine-readable card."""
    report = client.get("/api/metrics").json
    assert "not Home Credit" in report["dataset"]
    assert report["test"]["rows"] > 1000
