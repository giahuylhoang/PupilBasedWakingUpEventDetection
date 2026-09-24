# Sleep-state labeling tool — container for Google Cloud Run.
#
# Trace data is NOT baked into the image; it is read from a GCS bucket at
# runtime (DATA_BUCKET env var). The image only carries the app code.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HOST=0.0.0.0 \
    APP_COOKIE_SECURE=1

WORKDIR /app

# Install dependencies first (better layer caching).
COPY deploy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code only (traces/labels live in GCS).
COPY scripts/labeling_server.py /app/scripts/labeling_server.py
COPY scripts/labeling_frontend.html /app/scripts/labeling_frontend.html

# The server reads scripts/labeling_frontend.html relative to itself and
# computes ROOT as parents[1]; with /app/scripts/... that makes ROOT=/app,
# which is fine because in GCS mode no local data/ dir is used.
EXPOSE 8080

# gunicorn serves the Flask `app` object. 2 workers x 8 threads comfortably
# handles a lab's worth of annotators; --timeout covers slow GCS reads.
CMD exec gunicorn --bind "$HOST:$PORT" --workers 2 --threads 8 \
    --timeout 120 scripts.labeling_server:app
