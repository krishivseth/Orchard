#!/bin/bash

# Orchard Dependency Installation Script
# This script installs all required dependencies for the project

set -e

echo "📦 Installing Orchard dependencies..."

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

echo -e "${BLUE}Checking system requirements...${NC}"

# Check Python
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3.8+ is required but not installed.${NC}"
    echo -e "${YELLOW}Please install Python 3.8 or later from https://python.org${NC}"
    exit 1
else
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
fi

# Check Node.js
if ! command_exists node; then
    echo -e "${RED}Error: Node.js 18+ is required but not installed.${NC}"
    echo -e "${YELLOW}Please install Node.js from https://nodejs.org${NC}"
    exit 1
else
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓ Node.js $NODE_VERSION found${NC}"
fi

# Check npm
if ! command_exists npm; then
    echo -e "${RED}Error: npm is required but not installed.${NC}"
    echo -e "${YELLOW}npm should come with Node.js installation${NC}"
    exit 1
else
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓ npm $NPM_VERSION found${NC}"
fi

# Check pip
if ! command_exists pip3; then
    echo -e "${RED}Error: pip3 is required but not installed.${NC}"
    echo -e "${YELLOW}Please install pip3 with your package manager${NC}"
    exit 1
else
    PIP_VERSION=$(pip3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓ pip $PIP_VERSION found${NC}"
fi

echo -e "${GREEN}✓ All system requirements satisfied${NC}"
echo ""

# Install backend dependencies
echo -e "${BLUE}Installing backend dependencies...${NC}"
cd packages/backend

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv .venv
fi

echo -e "${YELLOW}Activating virtual environment and installing packages...${NC}"
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✓ Backend dependencies installed${NC}"
cd ../..

# Install device agent dependencies
echo -e "${BLUE}Installing device agent dependencies...${NC}"
cd packages/device-agent

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv .venv
fi

echo -e "${YELLOW}Activating virtual environment and installing packages...${NC}"
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✓ Device agent dependencies installed${NC}"
cd ../..

# Install frontend dependencies
echo -e "${BLUE}Installing frontend dependencies...${NC}"
cd packages/frontend

echo -e "${YELLOW}Installing npm packages...${NC}"
npm install

echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
cd ../..

# Create necessary directories
echo -e "${BLUE}Creating project directories...${NC}"
mkdir -p logs
mkdir -p data

# Make scripts executable
echo -e "${BLUE}Making scripts executable...${NC}"
chmod +x scripts/*.sh

echo ""
echo -e "${GREEN}🎉 All dependencies installed successfully!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Start development: ${YELLOW}./scripts/start-dev.sh${NC}"
echo -e "  2. Open browser: ${YELLOW}http://localhost:3000${NC}"
echo -e "  3. Stop services: ${YELLOW}./scripts/stop-dev.sh${NC}"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}" 