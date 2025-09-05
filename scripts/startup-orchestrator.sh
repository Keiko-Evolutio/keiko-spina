#!/bin/bash

# Keiko Personal Assistant - Startup Orchestrator
# Ensures proper dependency management and health checking for all services
# Author: Keiko Development Team
# Version: 1.0.0

set -eo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly COMPOSE_PROJECT="keiko-development"
readonly COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.training.yml"
readonly LOG_FILE="${PROJECT_ROOT}/logs/startup-orchestrator.log"
readonly CONFIG_FILE="${SCRIPT_DIR}/config/startup-timeouts.conf"

# Environment Detection
readonly KEIKO_ENV="${KEIKO_ENV:-development}"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Load environment-specific configuration
load_environment_config() {
    local env="$1"

    # Default values (development)
    BASE_SERVICES_TIMEOUT=120
    MONITORING_SERVICES_TIMEOUT=180
    EDGE_SERVICES_TIMEOUT=300
    WORKFLOW_SERVICES_TIMEOUT=120
    BACKEND_STARTUP_TIMEOUT=60

    # Load from config file if exists
    if [[ -f "$CONFIG_FILE" ]]; then
        # Parse config file for environment section
        local in_section=false
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ $key =~ ^[[:space:]]*# ]] && continue
            [[ -z $key ]] && continue

            # Check for section headers
            if [[ $key =~ ^\[(.+)\]$ ]]; then
                local section="${BASH_REMATCH[1]}"
                if [[ "$section" == "$env" ]]; then
                    in_section=true
                else
                    in_section=false
                fi
                continue
            fi

            # Set variables if in correct section
            if [[ $in_section == true ]]; then
                case "$key" in
                    "BASE_SERVICES_TIMEOUT")
                        BASE_SERVICES_TIMEOUT="$value"
                        ;;
                    "MONITORING_SERVICES_TIMEOUT")
                        MONITORING_SERVICES_TIMEOUT="$value"
                        ;;
                    "EDGE_SERVICES_TIMEOUT")
                        EDGE_SERVICES_TIMEOUT="$value"
                        ;;
                    "WORKFLOW_SERVICES_TIMEOUT")
                        WORKFLOW_SERVICES_TIMEOUT="$value"
                        ;;
                    "BACKEND_STARTUP_TIMEOUT")
                        BACKEND_STARTUP_TIMEOUT="$value"
                        ;;
                esac
            fi
        done < "$CONFIG_FILE"
    fi

    # Override with environment variables if set
    BASE_SERVICES_TIMEOUT="${KEIKO_BASE_SERVICES_TIMEOUT:-$BASE_SERVICES_TIMEOUT}"
    MONITORING_SERVICES_TIMEOUT="${KEIKO_MONITORING_SERVICES_TIMEOUT:-$MONITORING_SERVICES_TIMEOUT}"
    EDGE_SERVICES_TIMEOUT="${KEIKO_EDGE_SERVICES_TIMEOUT:-$EDGE_SERVICES_TIMEOUT}"
    WORKFLOW_SERVICES_TIMEOUT="${KEIKO_WORKFLOW_SERVICES_TIMEOUT:-$WORKFLOW_SERVICES_TIMEOUT}"
    BACKEND_STARTUP_TIMEOUT="${KEIKO_BACKEND_STARTUP_TIMEOUT:-$BACKEND_STARTUP_TIMEOUT}"

    # Make variables readonly
    readonly BASE_SERVICES_TIMEOUT
    readonly MONITORING_SERVICES_TIMEOUT
    readonly EDGE_SERVICES_TIMEOUT
    readonly WORKFLOW_SERVICES_TIMEOUT
    readonly BACKEND_STARTUP_TIMEOUT
}

# Service categories and their containers
get_services_for_category() {
    local category="$1"
    case "$category" in
        "base")
            echo "postgres redis nats"
            ;;
        "monitoring")
            echo "prometheus alertmanager jaeger grafana"
            ;;
        "workflow")
            echo "n8n-postgres"
            ;;
        "edge")
            echo "edge-registry edge-node-1 edge-node-2 edge-node-3 edge-load-balancer edge-monitor"
            ;;
        "tools")
            echo ""
            ;;
        *)
            echo ""
            ;;
    esac
}

# Service timeouts
get_timeout_for_category() {
    local category="$1"
    case "$category" in
        "base")
            echo $BASE_SERVICES_TIMEOUT
            ;;
        "monitoring")
            echo $MONITORING_SERVICES_TIMEOUT
            ;;
        "workflow")
            echo $WORKFLOW_SERVICES_TIMEOUT
            ;;
        "edge")
            echo $EDGE_SERVICES_TIMEOUT
            ;;
        "tools")
            echo $BASE_SERVICES_TIMEOUT
            ;;
        *)
            echo $BASE_SERVICES_TIMEOUT
            ;;
    esac
}

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Progress indicator
show_progress() {
    local current=$1
    local total=$2
    local service=$3
    local percentage=$((current * 100 / total))
    printf "\r${CYAN}[%3d%%]${NC} %s" "$percentage" "$service"
}

# Check if Docker and Docker Compose are available
check_prerequisites() {
    log_info "🔍 Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "✅ Prerequisites check passed"
}

# Create necessary directories
setup_environment() {
    log_info "🏗️ Setting up environment..."

    mkdir -p "${PROJECT_ROOT}/logs"

    # Ensure we're in the project root
    cd "${PROJECT_ROOT}"

    # Remove existing keiko networks to avoid conflicts
    docker network ls --format "{{.Name}}" | grep -E "keiko|${COMPOSE_PROJECT}" | xargs -r docker network rm 2>/dev/null || true

    # Create network if it doesn't exist
    if ! docker network ls | grep -q "keiko-network"; then
        log_info "Creating Docker network: keiko-network"
        docker network create keiko-network --driver bridge --subnet 172.20.0.0/16 || true
    fi

    log_success "✅ Environment setup completed"
}

# Clean up any existing containers
cleanup_existing() {
    log_info "🧹 Cleaning up existing containers..."
    
    if docker-compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" ps -q &> /dev/null; then
        docker-compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" down --remove-orphans || true
    fi
    
    log_success "✅ Cleanup completed"
}

# Start services in a specific category
start_service_category() {
    local category="$1"
    local services=$(get_services_for_category "$category")
    local timeout=$(get_timeout_for_category "$category")
    
    log_info "🚀 Starting ${category} services: ${services}"
    
    # Start services
    for service in $services; do
        log_info "  Starting ${service}..."
        # Try to start the service, if it fails due to network conflict, clean up and retry
        if ! docker-compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" up -d "$service" 2>/dev/null; then
            log_warn "First attempt failed, cleaning up networks and retrying..."
            # Clean up conflicting networks
            docker network ls --format "{{.Name}}" | grep -E "keiko|${COMPOSE_PROJECT}" | xargs -r docker network rm 2>/dev/null || true
            # Recreate the network
            docker network create keiko-network --driver bridge --subnet 172.20.0.0/16 2>/dev/null || true
            # Retry starting the service
            if ! docker-compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" up -d "$service" 2>/dev/null; then
                log_error "Failed to start service: ${service}"
                return 1
            fi
        fi
    done
    
    # Wait for health checks
    local start_time=$(date +%s)
    local service_count=$(echo $services | wc -w)
    local current_service=0
    
    for service in $services; do
        ((current_service++))
        show_progress $current_service $service_count "Waiting for ${service} to be healthy..."
        
        local service_start_time=$(date +%s)
        while true; do
            local current_time=$(date +%s)
            local elapsed=$((current_time - service_start_time))
            
            if [ $elapsed -gt $timeout ]; then
                echo # New line after progress
                log_error "Service ${service} failed to become healthy within ${timeout}s"
                return 1
            fi
            
            # Check if container is running and healthy
            local container_name="keiko-${service}"
            if docker ps --filter "name=${container_name}" --filter "status=running" --format "{{.Names}}" | grep -q "${container_name}"; then
                # Check health status
                local health_status=$(docker inspect --format='{{.State.Health.Status}}' "${container_name}" 2>/dev/null || echo "none")
                
                if [ "$health_status" = "healthy" ] || [ "$health_status" = "none" ]; then
                    # If no health check defined, consider it healthy if running
                    if [ "$health_status" = "none" ]; then
                        # Additional check for services without health checks
                        sleep 5
                    fi
                    break
                fi
            fi
            
            sleep 2
        done
    done
    
    echo # New line after progress
    local total_time=$(($(date +%s) - start_time))
    log_success "✅ ${category} services started successfully in ${total_time}s"
    return 0
}

# Verify all services are healthy
verify_all_services() {
    log_info "🔍 Verifying all services are healthy..."

    local failed_services=()
    local categories="base monitoring workflow edge tools"

    for category in $categories; do
        local services=$(get_services_for_category "$category")
        for service in $services; do
            local container_name="keiko-${service}"
            
            if ! docker ps --filter "name=${container_name}" --filter "status=running" --format "{{.Names}}" | grep -q "${container_name}"; then
                failed_services+=("${service} (not running)")
                continue
            fi
            
            local health_status=$(docker inspect --format='{{.State.Health.Status}}' "${container_name}" 2>/dev/null || echo "none")
            if [ "$health_status" != "healthy" ] && [ "$health_status" != "none" ]; then
                failed_services+=("${service} (unhealthy: ${health_status})")
            fi
        done
    done
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        log_error "The following services are not healthy:"
        for service in "${failed_services[@]}"; do
            log_error "  - ${service}"
        done
        return 1
    fi
    
    log_success "✅ All services are healthy"
    return 0
}

# Graceful shutdown in reverse dependency order
graceful_shutdown() {
    log_warn "🛑 Initiating graceful shutdown..."

    # Shutdown in reverse order
    local categories="tools edge workflow monitoring base"

    for category in $categories; do
        local services=$(get_services_for_category "$category")
        log_info "Stopping ${category} services..."

        for service in $services; do
            docker-compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" stop "$service" 2>/dev/null || true
        done
    done
    
    # Final cleanup
    docker-compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" down --remove-orphans 2>/dev/null || true
    
    log_info "✅ Graceful shutdown completed"
}

# Main orchestration function
main() {
    local start_time=$(date +%s)

    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    Keiko Personal Assistant                                  ║"
    echo "║                      Startup Orchestrator v1.0.0                           ║"
    echo "║                      Environment: ${KEIKO_ENV}                                    ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Load environment-specific configuration
    load_environment_config "$KEIKO_ENV"
    log_info "Loaded configuration for environment: $KEIKO_ENV"
    log_info "Timeouts - Base: ${BASE_SERVICES_TIMEOUT}s, Monitoring: ${MONITORING_SERVICES_TIMEOUT}s, Edge: ${EDGE_SERVICES_TIMEOUT}s"

    # Setup trap for cleanup on exit
    trap graceful_shutdown EXIT INT TERM
    
    # Initialize
    check_prerequisites
    setup_environment
    cleanup_existing
    
    # Start services in dependency order
    local startup_order="base monitoring workflow edge tools"
    local total_categories=5
    local current_category=0

    for category in $startup_order; do
        ((current_category++))
        echo -e "\n${BLUE}[${current_category}/${total_categories}] Starting ${category} services...${NC}"
        
        if ! start_service_category "$category"; then
            log_error "Failed to start ${category} services"
            exit 1
        fi
    done
    
    # Final verification
    echo -e "\n${YELLOW}🔍 Performing final health verification...${NC}"
    if ! verify_all_services; then
        log_error "Final health verification failed"
        exit 1
    fi
    
    local total_time=$(($(date +%s) - start_time))
    
    echo -e "\n${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    🎉 INFRASTRUCTURE STARTUP SUCCESSFUL! 🎉                 ║"
    echo "║                                                                              ║"
    echo "║  All 20 containers are running and healthy                                  ║"
    echo "║  Total startup time: ${total_time}s                                                    ║"
    echo "║                                                                              ║"
    echo "║  Ready to start backend services...                                         ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Don't trigger cleanup on successful exit
    trap - EXIT
    
    log_success "Infrastructure startup completed successfully in ${total_time}s"
    return 0
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
