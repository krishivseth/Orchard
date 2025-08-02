#!/bin/bash

# Orchard Development Stop Script
# This script stops all running development services

set -e

echo "🛑 Stopping Orchard services..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Read PIDs if they exist
if [ -f ".dev_pids" ]; then
    echo -e "${BLUE}Reading service PIDs...${NC}"
    PIDS=$(cat .dev_pids)
    
    # Kill each PID
    for PID in $PIDS; do
        if kill -0 $PID 2>/dev/null; then
            echo -e "${YELLOW}Stopping process $PID...${NC}"
            kill $PID
            
            # Wait for graceful shutdown
            sleep 2
            
            # Force kill if still running
            if kill -0 $PID 2>/dev/null; then
                echo -e "${RED}Force killing process $PID...${NC}"
                kill -9 $PID 2>/dev/null || true
            fi
        else
            echo -e "${GREEN}Process $PID already stopped${NC}"
        fi
    done
    
    # Clean up PID file
    rm .dev_pids
else
    echo -e "${YELLOW}No PID file found, attempting to kill by port...${NC}"
    
    # Kill processes by port (fallback method)
    echo -e "${BLUE}Stopping services on known ports...${NC}"
    
    # Kill backend (port 8000)
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    
    # Kill device agents (ports 8001, 8002)
    lsof -ti:8001 | xargs kill -9 2>/dev/null || true
    lsof -ti:8002 | xargs kill -9 2>/dev/null || true
    
    # Kill frontend dev server (port 3000)
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
fi

# Additional cleanup - kill any remaining python processes with our scripts
echo -e "${BLUE}Cleaning up any remaining processes...${NC}"

# Kill any remaining main.py processes
pkill -f "python.*main.py" 2>/dev/null || true

# Kill any remaining agent.py processes  
pkill -f "python.*agent.py" 2>/dev/null || true

# Kill any remaining npm dev processes
pkill -f "npm.*run.*dev" 2>/dev/null || true

echo -e "${GREEN}✓ All services stopped${NC}"

# Clean up log files (optional)
read -p "Do you want to clean up log files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "logs" ]; then
        echo -e "${BLUE}Cleaning up log files...${NC}"
        rm -rf logs/*
        echo -e "${GREEN}✓ Log files cleaned${NC}"
    fi
fi

echo ""
echo -e "${GREEN}🏁 Orchard stopped successfully!${NC}"
echo ""
echo -e "${BLUE}To start again, run: ./scripts/start-dev.sh${NC}" 