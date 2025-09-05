#!/bin/bash

# Edge Load Balancer Startup Script
# Startet den intelligenten Load Balancer für Edge-Node-Routing

set -euo pipefail

# =============================================================================
# Konfiguration
# =============================================================================

SERVICE_NAME="edge-load-balancer"
BALANCER_PORT="${BALANCER_PORT:-8088}"
HEALTH_PORT="${HEALTH_PORT:-8089}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ENVIRONMENT="${ENVIRONMENT:-development}"
REGISTRY_URL="${REGISTRY_URL:-http://edge-registry:8080}"
STRATEGY="${STRATEGY:-adaptive}"

# =============================================================================
# Logging Setup
# =============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$SERVICE_NAME] $1"
}

log_info() {
    log "INFO: $1"
}

log_error() {
    log "ERROR: $1"
}

log_warn() {
    log "WARN: $1"
}

# =============================================================================
# Environment Validation
# =============================================================================

validate_environment() {
    log_info "Validiere Umgebungsvariablen..."
    
    if [[ -z "$REGISTRY_URL" ]]; then
        log_error "REGISTRY_URL ist erforderlich"
        exit 1
    fi
    
    # Strategy validieren
    case "$STRATEGY" in
        "round-robin"|"least-connections"|"latency-based"|"capacity-based"|"adaptive")
            log_info "Load-Balancing-Strategy validiert: $STRATEGY"
            ;;
        *)
            log_error "Ungültige Load-Balancing-Strategy: $STRATEGY"
            exit 1
            ;;
    esac
    
    log_info "Umgebungsvariablen validiert"
}

# =============================================================================
# Registry Connectivity Check
# =============================================================================

check_registry_connectivity() {
    log_info "Prüfe Verbindung zur Edge Registry..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$REGISTRY_URL/health" > /dev/null 2>&1; then
            log_info "Edge Registry erreichbar"
            return 0
        fi
        
        log_warn "Edge Registry nicht erreichbar (Versuch $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    log_error "Edge Registry nach $max_attempts Versuchen nicht erreichbar"
    exit 1
}

# =============================================================================
# Service Initialization
# =============================================================================

initialize_service() {
    log_info "Initialisiere Load Balancer Service..."
    
    # Logs-Verzeichnis erstellen
    mkdir -p /app/logs
    
    # Konfigurationsdatei prüfen
    if [[ ! -f "$EDGE_CONFIG_PATH" ]]; then
        log_error "Konfigurationsdatei nicht gefunden: $EDGE_CONFIG_PATH"
        exit 1
    fi
    
    # Python-Pfad setzen
    export PYTHONPATH="/app:$PYTHONPATH"
    
    log_info "Service-Initialisierung abgeschlossen"
}

# =============================================================================
# Health Check Server
# =============================================================================

start_health_server() {
    log_info "Starte Health Check Server auf Port $HEALTH_PORT..."
    
    /app/.venv/bin/python -c "
import asyncio
import aiohttp
from aiohttp import web
import json
import os
import psutil
from datetime import datetime

async def health_check(request):
    try:
        # System-Metriken sammeln
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        health_data = {
            'service': '$SERVICE_NAME',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'environment': '$ENVIRONMENT',
            'metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'strategy': '$STRATEGY',
                'registry_url': '$REGISTRY_URL'
            },
            'ports': {
                'balancer': $BALANCER_PORT,
                'health': $HEALTH_PORT
            }
        }
        
        return web.json_response(health_data)
        
    except Exception as e:
        error_data = {
            'service': '$SERVICE_NAME',
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }
        return web.json_response(error_data, status=500)

async def readiness_check(request):
    try:
        # Prüfe ob Load Balancer bereit ist
        registry_reachable = True
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('$REGISTRY_URL/health', timeout=5) as resp:
                    registry_reachable = resp.status == 200
        except:
            registry_reachable = False
            
        if registry_reachable:
            return web.json_response({'status': 'ready'})
        else:
            return web.json_response({
                'status': 'not_ready',
                'registry_reachable': registry_reachable
            }, status=503)
            
    except Exception as e:
        return web.json_response({'status': 'not_ready', 'error': str(e)}, status=503)

app = web.Application()
app.router.add_get('/health', health_check)
app.router.add_get('/ready', readiness_check)

web.run_app(app, host='0.0.0.0', port=$HEALTH_PORT)
" &
    
    HEALTH_PID=$!
    echo $HEALTH_PID > /tmp/health-server.pid
    
    log_info "Health Check Server gestartet (PID: $HEALTH_PID)"
}

# =============================================================================
# Main Service
# =============================================================================

start_load_balancer() {
    log_info "Starte Edge Load Balancer Service auf Port $BALANCER_PORT..."
    
    # Edge Load Balancer Service starten
    cd /app
    exec /app/.venv/bin/python backend/edge/load_balancer.py
}

# =============================================================================
# Signal Handlers
# =============================================================================

cleanup() {
    log_info "Fahre Services herunter..."
    
    # Health Server stoppen
    if [[ -f /tmp/health-server.pid ]]; then
        HEALTH_PID=$(cat /tmp/health-server.pid)
        if kill -0 "$HEALTH_PID" 2>/dev/null; then
            log_info "Stoppe Health Server (PID: $HEALTH_PID)"
            kill "$HEALTH_PID"
        fi
        rm -f /tmp/health-server.pid
    fi
    
    log_info "Cleanup abgeschlossen"
    exit 0
}

# Signal Handlers registrieren
trap cleanup SIGTERM SIGINT

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_info "Starte $SERVICE_NAME..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Log Level: $LOG_LEVEL"
    log_info "Balancer Port: $BALANCER_PORT"
    log_info "Health Port: $HEALTH_PORT"
    log_info "Registry URL: $REGISTRY_URL"
    log_info "Strategy: $STRATEGY"
    
    # Initialisierung
    validate_environment
    check_registry_connectivity
    initialize_service
    
    # Services starten
    start_health_server
    
    # Kurz warten bis Health Server läuft
    sleep 2
    
    # Hauptservice starten
    start_load_balancer
}

# Script ausführen
main "$@"
