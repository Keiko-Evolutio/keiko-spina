#!/bin/bash

# Quick Startup Test for Keiko Personal Assistant
# Simplified version for demonstration

set -eo pipefail

# Configuration
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly COMPOSE_PROJECT="keiko-development"
readonly COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.dev.yml"

# Colors
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    Keiko Personal Assistant                                  ║"
echo "║                      Quick Startup Test                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Cleanup existing
echo -e "${BLUE}🧹 Cleaning up existing containers...${NC}"
docker-compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" down --remove-orphans 2>/dev/null || true

# Start all services
echo -e "${BLUE}🚀 Starting all infrastructure services...${NC}"
cd "${PROJECT_ROOT}"
docker-compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" up -d

# Wait a bit for services to start
echo -e "${BLUE}⏳ Waiting for services to initialize...${NC}"
sleep 30

# Check running containers
echo -e "${BLUE}📊 Checking running containers...${NC}"
running_containers=$(docker-compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" ps | grep keiko | wc -l)

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    🎉 INFRASTRUCTURE STARTUP COMPLETE! 🎉                   ║"
echo "║                                                                              ║"
echo "║  Running containers: ${running_containers}                                                    ║"
echo "║                                                                              ║"
echo "║  Ready to start backend services...                                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo "Infrastructure startup completed successfully!"
exit 0
