FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-deps . && useradd --create-home app
COPY models ./models
COPY wsgi.py gunicorn.conf.py ./
USER app
EXPOSE 8000
CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:app"]
