# Dockerfile for the agent service
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy and fix permissions BEFORE anything else
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod 755 /app/entrypoint.sh  # ← more reliable than  +x

# Now copy the rest of the app
COPY . .

ENTRYPOINT ["/app/entrypoint.sh"]
