#!/bin/bash

# Configure environment
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1
export OMP_NUM_THREADS=1

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RAG API Server...${NC}"
echo -e "${YELLOW}API Documentation: ${NC}http://localhost:8000/docs"
echo -e "${YELLOW}API Endpoints:${NC}"
echo "  - POST /search: Search and generate responses"
echo "  - GET  /health: Health check"
echo -e "\n${YELLOW}Press Ctrl+C to stop the server${NC}\n"

# Check if Python and required packages are installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    exit 1
fi

# Verify required packages
python3 -c "import fastapi; import uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install fastapi uvicorn
fi

# Start the server with uvicorn
python3 -c '
import uvicorn
import webbrowser
import threading
import time

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)
    webbrowser.open("http://localhost:8000/docs")

# Start browser in background
threading.Thread(target=open_browser).start()

# Start server
uvicorn.run(
    "query:app",
    host="0.0.0.0",
    port=8000,
    reload=True,
    reload_excludes=["*.pyc", "*.log"],
    log_level="info"
)
'
