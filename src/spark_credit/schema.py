"""Fixed UCI historical financial features; exclude demographics and identifiers."""

import math
import pandas as pd

FEATURES = {"LIMIT_BAL": ("Credit limit (source NTD)", 0, 100000000)}
for month, column in enumerate(
    ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], 1
):
    FEATURES[column] = (
        f"Repayment-status code, month {month} (Sep to Apr 2005)",
        -2,
        9,
    )
for prefix, label, minimum in [
    ("BILL_AMT", "Statement amount", -100000000),
    ("PAY_AMT", "Payment amount", 0),
]:
    for month in range(1, 7):
        FEATURES[f"{prefix}{month}"] = (
            f"{label}, month {month} (source NTD)",
            minimum,
            100000000,
        )
INTEGER_FIELDS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
LABELS = {
    "0": "No next-month default source label",
    "1": "Next-month default source label",
}


def validate_records(records: list) -> pd.DataFrame:
    """Enforce bounded numeric input; explicit null means training imputation."""
    if not isinstance(records, list) or not 1 <= len(records) <= 500:
        raise ValueError("Provide 1 to 500 records.")
    rows = []
    for record in records:
        if not isinstance(record, dict) or not set(FEATURES) <= set(record):
            raise ValueError("Each record must contain every named feature.")
        row = {}
        for name, (_, lower, upper) in FEATURES.items():
            value = record[name]
            if value is None or value == "":
                row[name] = float("nan")
                continue
            if isinstance(value, bool):
                raise ValueError("Boolean features are not supported.")
            try:
                number = float(value)
            except (ValueError, TypeError) as exc:
                raise ValueError("Features must be numeric or null.") from exc
            if not math.isfinite(number) or not lower <= number <= upper:
                raise ValueError(f"{name} is outside the supported range.")
            if name in INTEGER_FIELDS and number != int(number):
                raise ValueError("Repayment-status codes must be whole numbers.")
            row[name] = number
        if all(math.isnan(value) for value in row.values()):
            raise ValueError("Provide at least one observed feature.")
        rows.append(row)
    return pd.DataFrame(rows, columns=list(FEATURES))
