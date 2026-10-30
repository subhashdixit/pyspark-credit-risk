"""Check live readiness and a real CSV-derived JSON prediction."""

import argparse
import csv
from http.client import HTTPException
import io
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def check(base_url, attempts, expected_revision=None):
    """Retry cold starts, then verify the deployed source and inference contract."""
    base = base_url.rstrip("/")
    last_error = None
    for attempt in range(attempts):
        try:
            with urlopen(base + "/health/ready", timeout=20) as response:
                ready = json.load(response)
            if ready["status"] != "ready":
                raise ValueError("Service not ready")
            if expected_revision and ready["revision"] != expected_revision:
                raise ValueError("Expected revision is not deployed")
            with urlopen(base + "/sample.csv", timeout=20) as response:
                records = list(csv.DictReader(io.StringIO(response.read().decode())))
            request = Request(
                base + "/api/predict",
                data=json.dumps({"records": records}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=20) as response:
                result = json.load(response)
            prediction = result["predictions"][0]
            if prediction["category"] not in ["0", "1"]:
                raise ValueError("Invalid prediction category")
            if abs(sum(prediction["scores"].values()) - 1) > 0.00001:
                raise ValueError("Invalid class scores")
            print("Readiness and real HTTP prediction passed:", base)
            return
        except (
            HTTPError,
            URLError,
            TimeoutError,
            ValueError,
            ConnectionError,
            HTTPException,
        ) as exc:
            last_error = str(exc)
            if attempt + 1 < attempts:
                time.sleep(10)
    raise SystemExit(f"Smoke test failed: {last_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--expected-revision")
    args = parser.parse_args()
    check(args.url, args.attempts, args.expected_revision)
