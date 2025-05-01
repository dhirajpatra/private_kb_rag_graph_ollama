# Use official Ollama image as base
FROM ollama/ollama:latest

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Set env to disable pip's threaded progress bar
ENV PIP_NO_PROGRESS_BAR=off

# Set working directory
WORKDIR /app

# Copy files before installing requirements
COPY requirements.txt ollama_server.py entrypoint.sh /app/ollama_server/

# Ensure entrypoint script is executable
RUN chmod +x /app/ollama_server/entrypoint.sh

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/ollama_server/requirements.txt

# Define entrypoint
ENTRYPOINT ["/bin/bash", "/app/ollama_server/entrypoint.sh"]
