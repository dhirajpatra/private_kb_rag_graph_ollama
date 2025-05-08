# ollama_server.py
import subprocess
import time
import os

OLLAMA_PATH = "/usr/local/bin/ollama"

def start_ollama_server():
    """Starts the Ollama server as a subprocess."""
    print("Starting Ollama server...")
    
    if not os.path.exists(OLLAMA_PATH):
        print(f"Error: Ollama binary not found at {OLLAMA_PATH}")
        return None

    process = subprocess.Popen([OLLAMA_PATH, "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # Give some time for the server to initialize
    return process

if __name__ == "__main__":
    server_process = start_ollama_server()
    if server_process:
        print("Ollama server is running.")
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("Stopping Ollama server...")
            server_process.terminate()
    else:
        print("Failed to start Ollama server.")

