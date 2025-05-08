#!/bin/bash

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

# Ensure model directory exists
mkdir -p /root/.ollama/models

# Define models
# MODEL_NAME="gemma3:1b"
MODEL_NAME="qwen3:0.6b"
EMBEDDING_MODEL="mxbai-embed-large"

# Wait for the Ollama server to start accepting connections
echo "Waiting for the Ollama server to start..."
until curl -s http://localhost:11434 > /dev/null; do
  echo "Ollama server is still starting..."
  sleep 2
done
echo "Ollama server is running."

# Pull models if not already present
if ! ollama list | grep -q "$MODEL_NAME"; then
  echo "Model $MODEL_NAME not found. Pulling..."
  until ollama pull "$MODEL_NAME"; do
    echo "Retrying pull for $MODEL_NAME..."
    sleep 5
  done
else
  echo "Model $MODEL_NAME already exists. Skipping pull."
fi

if ! ollama list | grep -q "$EMBEDDING_MODEL"; then
  echo "Embedding model $EMBEDDING_MODEL not found. Pulling..."
  until ollama pull "$EMBEDDING_MODEL"; do
    echo "Retrying pull for $EMBEDDING_MODEL..."
    sleep 5
  done
else
  echo "Embedding model $EMBEDDING_MODEL already exists. Skipping pull."
fi

# Keep container alive by waiting for Ollama process
wait $OLLAMA_PID
