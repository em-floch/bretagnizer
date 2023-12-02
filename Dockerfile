# syntax=docker/dockerfile:1.4
FROM python:3.9-slim-bullseye

# Install dependencies:
RUN pip install --upgrade pip && \
    # --no-cache-dir to prevent pip creating a cache
    pip install poetry --no-cache-dir && \
    # `virtualenvs.create false` to install dependencies globally
    poetry config virtualenvs.create false

ENTRYPOINT ["/bin/bash", "-c"]
