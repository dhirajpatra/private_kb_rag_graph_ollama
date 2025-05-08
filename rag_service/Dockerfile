FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install only required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    libffi-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory and create user
WORKDIR /app
RUN useradd -m agentuser && chown -R agentuser /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy and install all requirements in one layer to reduce image layers
COPY --chown=agentuser requirements.txt requirements-extra.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-extra.txt

# Copy entrypoint and set permissions
COPY --chown=agentuser entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# Copy app code
USER agentuser
COPY --chown=agentuser . .

ENTRYPOINT ["/app/entrypoint.sh"]
