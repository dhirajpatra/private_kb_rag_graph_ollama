#!/bin/bash

# Start both commands
uvicorn app:app --host 0.0.0.0 --port 5000

# sleep 5

# python -m langgraph dev --port 5000
