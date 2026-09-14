# Vasomotion PSD tool -- container for Google Cloud Run.
#
# Self-contained: this image only needs the vasomotion_app/ directory. No
# database, no persisted data, no authentication (see the design spec's
# "Out of scope" section). Scales to zero on Cloud Run between uses.
#
# Build context is this directory (vasomotion_app/), e.g.:
#   docker build -t vasomotion-psd-tool vasomotion_app
# This is a separate, independent image from the root Dockerfile (the
# sleep-state labeling tool) -- two separate Cloud Run services.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8080

WORKDIR /app

# Install dependencies first (better layer caching): this layer only
# invalidates when requirements.txt changes, not on every code edit.
# All six pins (numpy, pandas, scipy, matplotlib, openpyxl, streamlit) have
# published cp312 wheels (manylinux for the compiled ones, py3-none-any for
# openpyxl/streamlit), so this is a pure wheel install -- no compiler and no
# system build dependencies needed.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code, copied into an /app/vasomotion_app/ package directory -- NOT
# flattened into /app/ directly. app.py imports its sibling modules with
# absolute package imports, e.g. `from vasomotion_app import bands`, so the
# `vasomotion_app` package name must resolve at import time. Streamlit's
# `streamlit run` inserts the *script's own* directory (/app/vasomotion_app)
# at the front of sys.path, which is not enough by itself; PYTHONPATH=/app
# above adds the package's *parent* directory too, so
# `import vasomotion_app` finds /app/vasomotion_app/__init__.py. This is the
# same pattern already used by the sibling root Dockerfile, which copies
# scripts/labeling_server.py to /app/scripts/labeling_server.py rather than
# flattening it to /app/labeling_server.py.
COPY __init__.py io_excel.py preprocess.py psd_methods.py bands.py figures.py bundle.py app.py /app/vasomotion_app/

EXPOSE 8080

# Cloud Run injects $PORT at runtime (falls back to the ENV default above for
# local `docker run`); Streamlit must bind to it on 0.0.0.0, and headless
# mode skips the "would you like to send usage stats" prompt and the
# auto-opened browser, neither of which make sense in a container.
# `sh -c` (rather than exec-form CMD) is required so that `$PORT` is
# expanded by the shell at container start, not baked in at build time.
CMD ["sh", "-c", "streamlit run vasomotion_app/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]
