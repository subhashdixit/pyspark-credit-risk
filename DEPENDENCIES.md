# Exact dependency versions

`requirements.txt` lists the complete runtime dependency resolution, including transitive packages, with exact `package==version` pins. Use a fresh virtual environment; the recorded local interpreter is **CPython 3.11.11**. Platform/Python markers select compatible wheels on supported systems.

```bash
python3.11 -m venv .venv
# Activate: source .venv/bin/activate (Windows: .venv\Scripts\activate)
python -m pip install -r requirements.txt
python -m pip check
```

`requirements-dev.txt` similarly pins the development/test environment. Where present, `requirements-training.txt` pins extra training tools separately. Do not install training-only tooling in the web-serving environment unnecessarily. The two Streamlit projects also use `-e .` to install this local repository after its external packages; its own version is recorded in `pyproject.toml`.

Direct inputs are preserved in `requirements*.in`. The complete generated `.txt` files are the installation entry points. The package build backend is pinned to `setuptools==80.9.0` in `pyproject.toml`. `requirements-versions.json` records reviewed file hashes, and `python scripts/check_requirements.py` rejects floating pins or unreviewed edits. Existing CI still runs dependency checks, tests, lint and supported container checks.

For an intentional update, edit the appropriate `.in` input and regenerate with **uv==0.13.6**:

```bash
python -m pip install uv==0.13.6
uv pip compile requirements.in --python-version 3.10 --universal --no-header --no-annotate -o requirements.txt
uv pip compile requirements-dev.in --python-version 3.10 --universal --no-header --no-annotate -o requirements-dev.txt
```

Where used, compile `requirements-training.in` the same way. For Streamlit keep `-e .` at the end of each generated install file. Update the manifest hashes and run the existing CI checks before merging. Exact package versions do not guarantee identical native libraries or floating-point output across hardware. External datasets and server/database credentials are separate setup requirements documented in the README.
