"""Verify exact dependency pins and reviewed requirements-file identities."""

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Reject an accidental floating package or an unreviewed pin-file edit."""
    manifest = json.loads((ROOT / "requirements-versions.json").read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Update the reviewed dependency manifest for {path.name}")
        if path.suffix != ".txt":
            continue
        for line in path.read_text().splitlines():
            clean = line.split("#", 1)[0].strip()
            if not clean or clean == "-e .":
                continue
            if not re.fullmatch(
                r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s;*<>=~]+(?:\s*;.*)?",
                clean,
            ):
                raise ValueError(
                    f"Use an exact package==version pin: {path.name}: {clean}"
                )
    print("Exact requirements versions and reviewed file hashes verified")


if __name__ == "__main__":
    main()
