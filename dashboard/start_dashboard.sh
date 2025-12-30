#!/bin/bash

# RAG System - Streamlit Dashboard Launcher
# Modern, user-friendly interface for the RAG system

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}   RAG System - Dashboard Launcher${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Python virtual environment exists in project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found. Creating...${NC}"
    cd "$PROJECT_ROOT"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -e .
    pip install -r dashboard/streamlit_app/requirements.txt
else
    source "$PROJECT_ROOT/venv/bin/activate"
fi

echo -e "${GREEN}✓ Environment ready${NC}"
echo ""

# Create a temporary file to store process ID
PID_FILE="$SCRIPT_DIR/.dashboard_pid"

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down dashboard...${NC}"

    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null; then
            kill $pid 2>/dev/null || true
        fi
        rm "$PID_FILE"
    fi

    echo -e "${GREEN}✓ Dashboard stopped${NC}"
    exit 0
}

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start Streamlit app
echo -e "${BLUE}Starting RAG Dashboard...${NC}"
echo ""

cd "$SCRIPT_DIR/streamlit_app"
streamlit run app.py --server.port 8501 --server.address localhost &
STREAMLIT_PID=$!
echo $STREAMLIT_PID > "$PID_FILE"

# Wait a moment for Streamlit to start
sleep 3

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Dashboard is running!${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "  Dashboard:  ${BLUE}http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the dashboard${NC}"
echo ""

# Wait for process
wait
