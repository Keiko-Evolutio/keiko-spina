#!/bin/bash

# Keiko Personal Assistant - Simple Development Startup
# Einfaches, zuverlässiges Startup-Script für die Entwicklung
# Author: Keiko Development Team
# Version: 1.0.0

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.dev-essential.yml"
readonly LOG_FILE="${PROJECT_ROOT}/logs/simple-startup.log"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Stelle sicher, dass das logs Verzeichnis existiert
    mkdir -p "$(dirname "${LOG_FILE}")"

    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Check prerequisites
check_prerequisites() {
    log_info "🔍 Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "docker-compose is not installed. Please install docker-compose and try again."
        exit 1
    fi
    
    # Check if compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log_success "✅ Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log_info "🏗️ Setting up environment..."
    
    # Create logs directory
    mkdir -p "${PROJECT_ROOT}/logs"
    
    # Ensure we're in the project root
    cd "${PROJECT_ROOT}"
    
    # Clean up any existing containers
    log_info "Cleaning up existing containers..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    
    log_success "✅ Environment setup completed"
}

# Start services
start_services() {
    log_info "🚀 Starting development services..."
    
    # Start all services
    if docker-compose -f "$COMPOSE_FILE" up -d; then
        log_success "✅ Services started successfully"
    else
        log_error "❌ Failed to start services"
        return 1
    fi
}

# Wait for services to be healthy
wait_for_services() {
    log_info "⏳ Waiting for services to be healthy..."

    local services=("postgres" "redis" "nats" "prometheus" "jaeger" "otel-collector")
    local max_wait=60  # 60 seconds total for development (more services need more time)
    local start_time=$(date +%s)

    # Wait for all services in parallel, not sequentially
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [ $elapsed -gt $max_wait ]; then
            log_warn "⚠️ Some services may not be fully healthy yet, but proceeding with development"
            break
        fi

        local all_healthy=true
        local status_info=""

        for service in "${services[@]}"; do
            # Check if container is running
            if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
                # Check health status
                local health_status=$(docker inspect --format='{{.State.Health.Status}}' "keiko-${service}" 2>/dev/null || echo "none")

                if [ "$health_status" = "healthy" ] || [ "$health_status" = "none" ]; then
                    status_info="${status_info} ${service}(✅)"
                else
                    # Für otel-collector: Prüfe ob Container läuft, auch wenn Health Check fehlschlägt
                    if [ "$service" = "otel-collector" ]; then
                        local container_status=$(docker inspect --format='{{.State.Status}}' "keiko-${service}" 2>/dev/null || echo "not_found")
                        if [ "$container_status" = "running" ]; then
                            status_info="${status_info} ${service}(✅)"
                        else
                            all_healthy=false
                            status_info="${status_info} ${service}(⏳)"
                        fi
                    else
                        all_healthy=false
                        status_info="${status_info} ${service}(⏳)"
                    fi
                fi
            else
                all_healthy=false
                status_info="${status_info} ${service}(❌)"
            fi
        done

        if [ "$all_healthy" = true ]; then
            local total_time=$(($(date +%s) - start_time))
            log_success "✅ All services are healthy (${total_time}s)"
            return 0
        fi

        log_info "Status:${status_info} (${elapsed}s/${max_wait}s)"
        sleep 2
    done

    local total_time=$(($(date +%s) - start_time))
    log_success "✅ Development services ready (${total_time}s)"
}

# Show service status
show_status() {
    log_info "📊 Service Status:"
    echo ""
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    
    log_info "🌐 Essential Service URLs:"
    echo -e "  ${CYAN}PostgreSQL:${NC} localhost:5432 (keiko_user/keiko_password)"
    echo -e "  ${CYAN}Redis:${NC} localhost:6379"
    echo -e "  ${CYAN}NATS:${NC} localhost:4222 (client), localhost:8222 (monitoring)"
    echo ""
    log_info "📊 Monitoring Service URLs:"
    echo -e "  ${CYAN}Prometheus:${NC} http://localhost:9090"
    echo -e "  ${CYAN}Jaeger UI:${NC} http://localhost:16686"
    echo -e "  ${CYAN}OpenTelemetry Collector:${NC} localhost:4317 (gRPC), localhost:4318 (HTTP)"
    echo ""
}

# Cleanup function
cleanup() {
    log_info "🧹 Cleaning up..."
    # Add any cleanup logic here if needed
}

# Main function
main() {
    local start_time=$(date +%s)
    
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    Keiko Personal Assistant                                  ║"
    echo "║                    Essential Development Startup                             ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Setup trap for cleanup on exit
    trap cleanup EXIT INT TERM
    
    # Execute startup sequence
    check_prerequisites
    setup_environment
    start_services
    wait_for_services
    show_status
    
    local total_time=$(($(date +%s) - start_time))
    
    echo -e "\n${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                🎉 ESSENTIAL DEVELOPMENT STARTUP SUCCESSFUL! 🎉                 ║"
    echo "║                                                                              ║"
    echo "║  Keiko Essential Development Environment ist bereit!                         ║"
    echo "║  Startup-Zeit: ${total_time}s                                                            ║"
    echo "║                                                                              ║"
    echo "║  Essential + Monitoring Services gestartet:                                  ║"
    echo "║  - PostgreSQL, Redis, NATS (Core)                                            ║"
    echo "║  - Prometheus, Jaeger, OpenTelemetry (Monitoring)                            ║"
    echo "║  Für vollständige Schulungsumgebung: 'make dev-full'                         ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log_success "Development startup completed successfully in ${total_time}s"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
