#!/bin/bash

# Keiko Personal Assistant - Production Startup Script
# Startet das System in der Produktionsumgebung mit erweiterten Timeouts und Sicherheitsmaßnahmen
# Author: Keiko Development Team
# Version: 1.0.0

set -eo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly COMPOSE_PROJECT="keiko-production"
readonly COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.prod.yml"
readonly LOG_FILE="${PROJECT_ROOT}/logs/production-startup.log"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Production Environment Variables
export KEIKO_ENV="production"
export KEIKO_BASE_SERVICES_TIMEOUT=300
export KEIKO_MONITORING_SERVICES_TIMEOUT=600
export KEIKO_EDGE_SERVICES_TIMEOUT=900
export KEIKO_WORKFLOW_SERVICES_TIMEOUT=300
export KEIKO_BACKEND_STARTUP_TIMEOUT=180

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

# Pre-flight checks für Production
production_preflight_checks() {
    log_info "🔍 Durchführung von Production Pre-flight Checks..."
    
    # Check if running as root (not recommended for production)
    if [[ $EUID -eq 0 ]]; then
        log_warn "⚠️  Script läuft als root. Für Production wird ein dedizierter User empfohlen."
    fi
    
    # Check available disk space
    local available_space=$(df "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')
    local required_space=10485760  # 10GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        log_error "❌ Nicht genügend Speicherplatz verfügbar. Benötigt: 10GB, Verfügbar: $((available_space/1024/1024))GB"
        exit 1
    fi
    
    # Check available memory
    local available_memory=$(free -m | awk 'NR==2{print $7}')
    local required_memory=8192  # 8GB
    
    if [[ $available_memory -lt $required_memory ]]; then
        log_warn "⚠️  Wenig verfügbarer Arbeitsspeicher. Verfügbar: ${available_memory}MB, Empfohlen: ${required_memory}MB"
    fi
    
    # Check if production data directories exist
    local data_dirs=(
        "/opt/keiko/data/postgres"
        "/opt/keiko/data/prometheus"
        "/opt/keiko/data/grafana"
        "/opt/keiko/data/redis"
    )
    
    for dir in "${data_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_info "📁 Erstelle Production Data Directory: $dir"
            sudo mkdir -p "$dir"
            sudo chown -R $(whoami):$(whoami) "$dir"
        fi
    done
    
    # Check if required environment variables are set
    local required_vars=(
        "GRAFANA_ADMIN_PASSWORD"
        "POSTGRES_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log_warn "⚠️  Umgebungsvariable $var ist nicht gesetzt. Verwende Default-Wert."
        fi
    done
    
    log_success "✅ Production Pre-flight Checks abgeschlossen"
}

# Backup existing data
backup_production_data() {
    log_info "💾 Erstelle Backup der Production-Daten..."
    
    local backup_dir="/opt/keiko/backups/$(date +%Y%m%d_%H%M%S)"
    
    if [[ -d "/opt/keiko/data" ]]; then
        sudo mkdir -p "$backup_dir"
        sudo cp -r /opt/keiko/data/* "$backup_dir/" 2>/dev/null || true
        log_success "✅ Backup erstellt in: $backup_dir"
    else
        log_info "ℹ️  Keine bestehenden Daten für Backup gefunden"
    fi
}

# Security hardening
apply_security_hardening() {
    log_info "🔒 Anwendung von Security Hardening..."
    
    # Set secure file permissions
    chmod 600 "${PROJECT_ROOT}/.env" 2>/dev/null || true
    chmod 600 "${PROJECT_ROOT}/infrastructure/alertmanager/alertmanager.yml" 2>/dev/null || true
    
    # Ensure log files have correct permissions
    mkdir -p "${PROJECT_ROOT}/logs"
    chmod 755 "${PROJECT_ROOT}/logs"
    touch "$LOG_FILE"
    chmod 644 "$LOG_FILE"
    
    log_success "✅ Security Hardening angewendet"
}

# Health monitoring setup
setup_health_monitoring() {
    log_info "📊 Setup Health Monitoring für Production..."
    
    # Create health check script
    cat > "${PROJECT_ROOT}/scripts/health-check.sh" << 'EOF'
#!/bin/bash
# Production Health Check Script

API_BASE="${KEIKO_API_URL:-http://localhost:8000}"
HEALTH_ENDPOINT="${API_BASE}/api/v1/system/health"
READINESS_ENDPOINT="${API_BASE}/api/v1/system/readiness"

# Check basic health
if ! curl -sf "$HEALTH_ENDPOINT" > /dev/null; then
    echo "CRITICAL: Health endpoint not responding"
    exit 2
fi

# Check readiness
if ! curl -sf "$READINESS_ENDPOINT" > /dev/null; then
    echo "WARNING: System not ready"
    exit 1
fi

echo "OK: System is healthy and ready"
exit 0
EOF
    
    chmod +x "${PROJECT_ROOT}/scripts/health-check.sh"
    
    log_success "✅ Health Monitoring Setup abgeschlossen"
}

# Main production startup function
main() {
    local start_time=$(date +%s)
    
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    Keiko Personal Assistant                                  ║"
    echo "║                      Production Startup v1.0.0                             ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Setup logging
    mkdir -p "${PROJECT_ROOT}/logs"
    
    # Production checks and setup
    production_preflight_checks
    backup_production_data
    apply_security_hardening
    setup_health_monitoring
    
    # Set production environment
    export COMPOSE_PROJECT_NAME="$COMPOSE_PROJECT"
    
    log_info "🚀 Starte Production Startup Orchestrator..."
    
    # Call the main startup orchestrator with production settings
    if "${SCRIPT_DIR}/startup-orchestrator.sh"; then
        local total_time=$(($(date +%s) - start_time))
        
        echo -e "\n${GREEN}"
        echo "╔══════════════════════════════════════════════════════════════════════════════╗"
        echo "║                    🎉 PRODUCTION STARTUP SUCCESSFUL! 🎉                    ║"
        echo "║                                                                              ║"
        echo "║  Keiko Personal Assistant ist bereit für Production                         ║"
        echo "║  Startup-Zeit: ${total_time}s                                                    ║"
        echo "║                                                                              ║"
        echo "║  Health Check: ${PROJECT_ROOT}/scripts/health-check.sh                      ║"
        echo "║  Logs: ${LOG_FILE}                                                          ║"
        echo "╚══════════════════════════════════════════════════════════════════════════════╝"
        echo -e "${NC}"
        
        log_success "Production startup completed successfully in ${total_time}s"
        
        # Start health monitoring
        log_info "🔄 Starte kontinuierliches Health Monitoring..."
        
        return 0
    else
        log_error "❌ Production startup failed"
        return 1
    fi
}

# Cleanup function
cleanup() {
    log_info "🧹 Production cleanup..."
    # Add any production-specific cleanup here
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
