#!/bin/bash

# Orchard Development Startup Script
# This script starts all services for local development

set -e

echo "🌳 Starting Orchard Distributed LLM Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}Error: Node.js is required but not installed.${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}Error: npm is required but not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"

# Install dependencies if not already installed
echo -e "${BLUE}Installing dependencies...${NC}"

# Backend dependencies
if [ ! -d "packages/backend/.venv" ]; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    cd packages/backend
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cd ../..
else
    echo -e "${GREEN}✓ Backend dependencies already installed${NC}"
fi

# Device agent dependencies  
if [ ! -d "packages/device-agent/.venv" ]; then
    echo -e "${YELLOW}Installing device agent dependencies...${NC}"
    cd packages/device-agent
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cd ../..
else
    echo -e "${GREEN}✓ Device agent dependencies already installed${NC}"
fi

# Frontend dependencies
if [ ! -d "packages/frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd packages/frontend
    npm install
    cd ../..
else
    echo -e "${GREEN}✓ Frontend dependencies already installed${NC}"
fi

echo -e "${GREEN}✓ All dependencies installed${NC}"

# Create logs directory
mkdir -p logs

# Start services in background
echo -e "${BLUE}Starting services...${NC}"

# Start backend
echo -e "${YELLOW}Starting backend server...${NC}"
cd packages/backend
source .venv/bin/activate
python main.py > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

# Wait for backend to start
sleep 3

# Check if backend is running
if ! curl -s http://localhost:8000/api/models > /dev/null; then
    echo -e "${RED}Error: Backend failed to start. Check logs/backend.log${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"

# Start device agents (simulate 2 devices for demo)
echo -e "${YELLOW}Starting device agents...${NC}"

cd packages/device-agent
source .venv/bin/activate

python agent.py --port 8001 > ../../logs/agent1.log 2>&1 &
AGENT1_PID=$!

python agent.py --port 8002 > ../../logs/agent2.log 2>&1 &
AGENT2_PID=$!

cd ../..

sleep 2

echo -e "${GREEN}✓ Device agents started (PIDs: $AGENT1_PID, $AGENT2_PID)${NC}"

# Start frontend
echo -e "${YELLOW}Starting frontend...${NC}"
cd packages/frontend
npm run dev > ../../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

sleep 3

echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"

# Save PIDs for cleanup
echo "$BACKEND_PID $AGENT1_PID $AGENT2_PID $FRONTEND_PID" > .dev_pids

echo ""
echo -e "${GREEN}🎉 Orchard is now running!${NC}"
echo ""
echo -e "${BLUE}Services:${NC}"
echo -e "  Backend:    http://localhost:8000"
echo -e "  Frontend:   http://localhost:3000"
echo -e "  Agent 1:    http://localhost:8001"
echo -e "  Agent 2:    http://localhost:8002"
echo ""
echo -e "${BLUE}Logs:${NC}"
echo -e "  Backend:    logs/backend.log"
echo -e "  Frontend:   logs/frontend.log"
echo -e "  Agent 1:    logs/agent1.log"
echo -e "  Agent 2:    logs/agent2.log"
echo ""
echo -e "${YELLOW}To stop all services, run: ./scripts/stop-dev.sh${NC}"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}" 