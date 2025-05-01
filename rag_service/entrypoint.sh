#!/bin/bash

# Wait for Neo4j to be ready
host="neo4j"
port=7687

echo "Waiting for Neo4j at $host:$port..."

# Check if Neo4j is up using Python
until python -c "import socket; socket.create_connection(('$host', $port), timeout=5)" > /dev/null 2>&1; do
  echo "Waiting for Neo4j..."
  sleep 3
done

echo "Neo4j is up. Starting rag_service..."

# Start service
uvicorn app:app --host 0.0.0.0 --port 5000

# Optionally run langgraph later
# sleep 5
# python -m langgraph dev --port 5000
