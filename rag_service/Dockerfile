# Dockerfile for the rag service
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory and create user
WORKDIR /app
RUN useradd -m agentuser && chown -R agentuser /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install base requirements
COPY --chown=agentuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install extra requirements separately
COPY --chown=agentuser requirements-extra.txt ./
RUN pip install --no-cache-dir -r requirements-extra.txt

# Copy entrypoint and set permissions
COPY --chown=agentuser entrypoint.sh ./
RUN chmod 755 /app/entrypoint.sh

# Copy application code
USER agentuser
COPY --chown=agentuser . .

ENTRYPOINT ["/app/entrypoint.sh"]
