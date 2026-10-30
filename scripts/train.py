"""Train real Spark pipelines on UCI 350 and verify a portable JSON export."""

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from spark_credit.schema import FEATURES, validate_records
from spark_credit.service import probabilities

ROOT = Path(__file__).resolve().parents[1]


def metrics(labels: object, scores: object, threshold: float) -> dict:
    """Describe source-label performance without claiming current lending utility."""
    predicted = (np.asarray(scores) >= threshold).astype(int)
    return {
        "rows": len(labels),
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro"),
        "average_precision": average_precision_score(labels, scores),
        "roc_auc": roc_auc_score(labels, scores),
        "classification_report": classification_report(
            labels, predicted, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def export_model(pipeline: object) -> dict:
    """Preserve the exact stage order, sample standard deviation and coefficients."""
    imputer, _, scaler, classifier = pipeline.stages
    row = imputer.surrogateDF.first().asDict()
    return {
        "features": list(FEATURES),
        "medians": [row[name] for name in FEATURES],
        "mean": scaler.mean.toArray().tolist(),
        "std": scaler.std.toArray().tolist(),
        "coefficients": classifier.coefficients.toArray().tolist(),
        "intercept": float(classifier.intercept),
    }


def train() -> None:
    """Fit preprocessing only on train, select on validation and test once."""
    source = ROOT / "data/default-credit-clients.xls"
    if not source.exists():
        raise SystemExit("Run python scripts/download_data.py first.")
    original = pd.read_excel(source, header=1)
    features = validate_records_for_training(original)
    labels = original.iloc[:, -1].astype(int).to_numpy()
    if not set(np.unique(labels)) == {0, 1}:
        raise ValueError("Expected binary source target.")
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    split = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42)
    train_idx, remainder = next(split.split(features, groups=groups))
    val_part, test_part = next(
        GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=43).split(
            features.iloc[remainder], groups=groups[remainder]
        )
    )
    val_idx, test_idx = remainder[val_part], remainder[test_part]
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("uci-credit-training")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        columns = list(FEATURES)
        schema = (
            ", ".join(f"{name} double" for name in columns)
            + ", label double, row_id int"
        )
        rows = [
            tuple(float(value) for value in values) + (float(labels[index]), int(index))
            for index, values in enumerate(features.to_numpy())
        ]
        full = spark.createDataFrame(rows, schema=schema)
        training = full.where(full.row_id.isin([int(i) for i in train_idx])).cache()
        imputed = [name + "_filled" for name in columns]
        candidates, exports, pipelines = {}, {}, {}
        for penalty in [0.01, 0.1]:
            name = "spark_logistic_reg_" + str(penalty)
            pipeline = Pipeline(
                stages=[
                    Imputer(
                        strategy="median",
                        inputCols=columns,
                        outputCols=imputed,
                        relativeError=0,
                    ),
                    VectorAssembler(inputCols=imputed, outputCol="assembled"),
                    StandardScaler(
                        inputCol="assembled",
                        outputCol="features",
                        withMean=True,
                        withStd=True,
                    ),
                    LogisticRegression(
                        featuresCol="features",
                        labelCol="label",
                        regParam=penalty,
                        maxIter=80,
                        tol=1e-7,
                    ),
                ]
            ).fit(training)
            exported = export_model(pipeline)
            scores = probabilities(features.iloc[val_idx].to_numpy(), exported)
            candidates[name] = metrics(labels[val_idx], scores, 0.5)
            exports[name], pipelines[name] = exported, pipeline
        selected = max(
            candidates, key=lambda name: candidates[name]["average_precision"]
        )
        model = exports[selected]
        validation_scores = probabilities(features.iloc[val_idx].to_numpy(), model)
        threshold = float(
            max(
                np.linspace(0.05, 0.95, 91),
                key=lambda value: f1_score(
                    labels[val_idx], validation_scores >= value, zero_division=0
                ),
            )
        )
        # Independent Spark transform vs portable inference on all held-out rows.
        probes = full.where(~full.row_id.isin([int(i) for i in train_idx]))
        transformed = (
            pipelines[selected]
            .transform(probes)
            .select("row_id", "probability")
            .collect()
        )
        ids = [row.row_id for row in transformed]
        reference = np.array([row.probability[1] for row in transformed])
        difference = float(
            np.max(
                np.abs(reference - probabilities(features.iloc[ids].to_numpy(), model))
            )
        )
        if difference > 1e-9:
            raise ValueError("Portable inference differs from Spark.")
        # Also verify missing-value imputation on a synthetic probe.
        probe_values = features.iloc[train_idx[:1]].to_numpy().copy()
        probe_values[0, 0] = np.nan
        probe_row = tuple(float(v) for v in probe_values[0]) + (0.0, -1)
        probe_score = (
            pipelines[selected]
            .transform(spark.createDataFrame([probe_row], schema=schema))
            .first()
            .probability[1]
        )
        missing_difference = abs(
            float(probe_score) - float(probabilities(probe_values, model)[0])
        )
        if missing_difference > 1e-9:
            raise ValueError("Missing-value export parity failed.")
        folder = ROOT / "models"
        folder.mkdir(exist_ok=True)
        artifact = folder / "portable-model.json"
        artifact.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
        sample = {
            name: float(features.iloc[train_idx][name].median()) for name in columns
        }
        # Status-code fields must remain valid integers for the form.
        for name in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
            sample[name] = int(round(sample[name]))
        (folder / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")
        report = {
            "version": "1.0.0",
            "dataset": (
                "UCI 350 Default of Credit Card Clients; "
                "Taiwan 2005; not Home Credit"
            ),
            "source_url": (
                "https://archive.ics.uci.edu/dataset/350/"
                "default+of+credit+card+clients"
            ),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_rows": len(original),
            "classes": ["0", "1"],
            "selected_model": selected,
            "threshold": round(threshold, 2),
            "split": {
                key: {"rows": len(idx), "positive_rows": int(labels[idx].sum())}
                for key, idx in [
                    ("train", train_idx),
                    ("validation", val_idx),
                    ("test", test_idx),
                ]
            },
            "split_strategy": (
                "Grouped numeric profiles 60/20/20; " "seeds 42 and 43; not temporal"
            ),
            "validation_candidates": candidates,
            "test": metrics(
                labels[test_idx],
                probabilities(features.iloc[test_idx].to_numpy(), model),
                threshold,
            ),
            "dummy_test": metrics(
                labels[test_idx],
                np.repeat(float(labels[train_idx].mean()), len(test_idx)),
                0.5,
            ),
            "export_max_abs_error": difference,
            "missing_probe_abs_error": missing_difference,
            "spark_version": spark.version,
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        (folder / "metrics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in [
                        "selected_model",
                        "threshold",
                        "split",
                        "test",
                        "export_max_abs_error",
                    ]
                },
                indent=2,
            )
        )
    finally:
        spark.stop()


def validate_records_for_training(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the runtime contract in bounded batches; do not fit on full data."""
    return pd.concat(
        [
            validate_records(frame.iloc[start : start + 500].to_dict("records"))
            for start in range(0, len(frame), 500)
        ],
        ignore_index=True,
    )


if __name__ == "__main__":
    train()
