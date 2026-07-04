# =============================================================================
# ECG Arrhythmia Detection — reproducible container
# -----------------------------------------------------------------------------
# Builds a slim image with the full training/evaluation stack. By default it
# runs the test suite; override the command to train, evaluate or serve the
# Streamlit dashboard.
#
#   docker build -t ecg-detect .
#   docker run --rm -it ecg-detect                         # run tests
#   docker run --rm -it ecg-detect python -m src.train     # train
#   docker run --rm -it -p 8501:8501 ecg-detect \
#       streamlit run app/streamlit_app.py --server.address 0.0.0.0
# =============================================================================
FROM python:3.12-slim

# All dependencies (numpy, scipy, torch, scikit-learn, matplotlib, ...) ship
# prebuilt manylinux wheels for CPython 3.12, so no compiler toolchain is needed
# — keeping the image small.

WORKDIR /app

# Keep Python lean and unbuffered for clean container logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies first so Docker can cache this layer independently of
# the source code.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the project (see .dockerignore for what is excluded from the context).
COPY . .

# Document the Streamlit dashboard port (used when running the app command).
EXPOSE 8501

# Default command: verify the build by running the unit tests.
CMD ["pytest", "-q"]
