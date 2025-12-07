#!/bin/bash
# ============================================
# Legal Document OCR - Quick Start Script
# For RunPod GPU deployment
# ============================================

set -e

echo "============================================"
echo "Legal Document OCR - Starting Services"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root${NC}"
fi

# Create directories
echo -e "${GREEN}Creating directories...${NC}"
mkdir -p uploads output logs models

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cat > .env << EOF
# Gemini API Key (optional, for AI text correction)
GEMINI_API_KEY=

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8888
DEBUG=false

# Storage
UPLOAD_DIR=./uploads
OUTPUT_DIR=./output

# Model Paths
GOT_MODEL_PATH=./models/GOT-OCR2_0
YOLO_MODEL_PATH=./models/yolov8_layout.pt
EOF
    echo -e "${GREEN}.env file created. Please edit it to add your GEMINI_API_KEY.${NC}"
fi

# Function to check if Redis is running
check_redis() {
    if command -v redis-cli &> /dev/null; then
        redis-cli ping &> /dev/null
        return $?
    fi
    return 1
}

# Start Redis if not running
if ! check_redis; then
    echo -e "${YELLOW}Starting Redis...${NC}"
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes
        sleep 2
    else
        echo -e "${RED}Redis not installed. Installing...${NC}"
        apt-get update && apt-get install -y redis-server
        redis-server --daemonize yes
        sleep 2
    fi
fi

# Check Redis connection
if check_redis; then
    echo -e "${GREEN}Redis is running${NC}"
else
    echo -e "${RED}Failed to start Redis${NC}"
    exit 1
fi

# Start Celery worker in background
echo -e "${GREEN}Starting Celery worker...${NC}"
pkill -f "celery -A tasks worker" || true
celery -A tasks worker --loglevel=info --concurrency=1 &
CELERY_PID=$!
echo "Celery worker started (PID: $CELERY_PID)"

# Wait for worker to initialize
sleep 3

# Start FastAPI server
echo -e "${GREEN}Starting FastAPI server on port 8888...${NC}"
pkill -f "uvicorn main:app" || true

echo ""
echo "============================================"
echo -e "${GREEN}Server starting...${NC}"
echo "API: http://localhost:8888"
echo "Docs: http://localhost:8888/docs"
echo "============================================"
echo ""

# Run server (blocking)
uvicorn main:app --host 0.0.0.0 --port 8888 --workers 1
