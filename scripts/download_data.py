"""Fetch the official UCI 350 dataset, preserving the source spreadsheet."""

import hashlib
from pathlib import Path
from urllib.request import urlopen
import zipfile
import io

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def download() -> None:
    """Download from UCI, not a third-party mirror or gated competition."""
    with urlopen(URL, timeout=60) as response:
        content = response.read(15 * 1024 * 1024)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        payload = archive.read("default of credit card clients.xls")
    target = ROOT / "data/default-credit-clients.xls"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(payload)
    print("UCI source SHA-256:", hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    download()
