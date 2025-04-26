# Dockerfile for Ollama server
# Start from the official ollama image
# Use Python as the base image instead of Ollama
# Use official Ollama image
FROM ollama/ollama:latest

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files from build context (./ollama_server)
COPY requirements.txt ollama_server.py entrypoint.sh /app/ollama_server/

# Ensure entrypoint script is executable
RUN chmod +x /app/ollama_server/entrypoint.sh

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/ollama_server/requirements.txt

# Define entrypoint
ENTRYPOINT ["/bin/bash", "/app/ollama_server/entrypoint.sh"]


