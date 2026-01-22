#!/bin/bash
#
# GPU Agent Installation Script
# Run with: curl -sSL https://your-domain.com/install.sh | bash
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "================================================"
echo "       GPU Cloud Orchestrator - Agent Setup     "
echo "================================================"
echo -e "${NC}"

# Configuration
BRAIN_URL="${BRAIN_URL:-http://localhost:8000}"
AGENT_DIR="/opt/gpu-agent"
CONFIG_DIR="/etc/gpu-agent"
LOG_DIR="/var/log/gpu-agent"
DATA_DIR="/var/lib/gpu-agent"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

# Check for NVIDIA GPU
echo -e "${YELLOW}Checking for NVIDIA GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}nvidia-smi not found. Please install NVIDIA drivers first.${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo -e "${GREEN}NVIDIA GPU detected!${NC}"

# Check for Docker
echo -e "${YELLOW}Checking for Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Installing Docker...${NC}"
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi

docker --version
echo -e "${GREEN}Docker is installed!${NC}"

# Check for NVIDIA Container Toolkit
echo -e "${YELLOW}Checking for NVIDIA Container Toolkit...${NC}"
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo -e "${YELLOW}Installing NVIDIA Container Toolkit...${NC}"

    # Add NVIDIA repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update
    apt-get install -y nvidia-container-toolkit

    # Configure Docker to use NVIDIA runtime
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# Verify GPU access in Docker
echo -e "${YELLOW}Testing GPU access in Docker...${NC}"
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi || {
    echo -e "${RED}Failed to access GPU from Docker container${NC}"
    exit 1
}
echo -e "${GREEN}GPU is accessible from Docker!${NC}"

# Check for Python 3.12+
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_CMD=""
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$(echo "$PYTHON_VERSION >= 3.11" | bc)" -eq 1 ]; then
        PYTHON_CMD="python3"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Python 3.11+ required. Installing Python 3.12...${NC}"
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.12 python3.12-venv python3.12-dev
    PYTHON_CMD="python3.12"
fi

$PYTHON_CMD --version
echo -e "${GREEN}Python is ready!${NC}"

# Create directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p $AGENT_DIR $CONFIG_DIR $LOG_DIR $DATA_DIR

# Create virtual environment
echo -e "${YELLOW}Setting up Python environment...${NC}"
$PYTHON_CMD -m venv $AGENT_DIR/venv
source $AGENT_DIR/venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install httpx pydantic pydantic-settings docker

# Copy agent files (in production, these would be downloaded)
echo -e "${YELLOW}Installing agent...${NC}"
# For now, just create a placeholder
cat > $AGENT_DIR/run_agent.py << 'EOF'
#!/usr/bin/env python3
"""Agent runner script."""
import sys
sys.path.insert(0, '/opt/gpu-agent')
from agent.agent import main
import asyncio
asyncio.run(main())
EOF
chmod +x $AGENT_DIR/run_agent.py

# Create systemd service
echo -e "${YELLOW}Creating systemd service...${NC}"
cat > /etc/systemd/system/gpu-agent.service << EOF
[Unit]
Description=GPU Cloud Orchestrator Agent
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=$AGENT_DIR
Environment="GPU_AGENT_BRAIN_URL=$BRAIN_URL"
Environment="GPU_AGENT_CONFIG_FILE=$CONFIG_DIR/config.json"
Environment="GPU_AGENT_LOG_FILE=$LOG_DIR/agent.log"
ExecStart=$AGENT_DIR/venv/bin/python $AGENT_DIR/run_agent.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

echo -e "${GREEN}"
echo "================================================"
echo "           Installation Complete!               "
echo "================================================"
echo -e "${NC}"
echo ""
echo "To start the agent:"
echo "  systemctl start gpu-agent"
echo ""
echo "To enable auto-start on boot:"
echo "  systemctl enable gpu-agent"
echo ""
echo "To view logs:"
echo "  journalctl -u gpu-agent -f"
echo ""
echo "Configuration file: $CONFIG_DIR/config.json"
echo ""

# Ask if user wants to start now
read -p "Start the agent now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    systemctl start gpu-agent
    echo -e "${GREEN}Agent started!${NC}"
    echo "Checking status..."
    sleep 2
    systemctl status gpu-agent --no-pager
fi
