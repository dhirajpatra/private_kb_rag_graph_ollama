#!/bin/bash

# Wait for Neo4j (Bolt at neo4j:7687)
until python -c "
import socket
host = 'neo4j'
port = 7687
with socket.create_connection((host, port), timeout=5):
    pass
" >/dev/null 2>&1; do
  echo 'Waiting for Neo4j on bolt://neo4j:7687...'
  sleep 3
done

echo 'Neo4j is up. Starting services...'

# Wait for Ollama (HTTP at ollama_server:11434)
until curl -s http://ollama_server:11434 > /dev/null; do
  echo "Waiting for Ollama on http://ollama_server:11434..."
  sleep 3
done

echo "Ollama is up. Proceeding to start services..."

# Start Knowledge Graph Service
uvicorn knowledge_graph.score:app --host 0.0.0.0 --port 8000 &

# Health check
check_service() {
  timeout 30 bash -c "until curl -fs http://localhost:$1/health >/dev/null; do sleep 1; done"
}

echo "Services ready. Monitoring..."

# Monitor processes
while sleep 30; do
  for port in 8000; do
    if ! curl -fs http://localhost:$port/health >/dev/null; then
      echo "Service on port $port down, restarting..."
      # pkill -f "uvicorn.*:$port"
      if [ $port -eq 8000 ]; then
        uvicorn knowledge_graph.score:app --host 0.0.0.0 --port 8000 &
      fi
    fi
  done
done

check_service 8000 || { echo "Knowledge Graph service failed"; exit 1; }