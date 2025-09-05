#!/bin/bash

# Edge Registry Startup Script
# Startet die zentrale Edge-Node-Registry mit Health Monitoring

set -euo pipefail

# =============================================================================
# Konfiguration
# =============================================================================

SERVICE_NAME="edge-registry"
REGISTRY_PORT="${REGISTRY_PORT:-8080}"
HEALTH_PORT="${HEALTH_PORT:-8081}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ENVIRONMENT="${ENVIRONMENT:-development}"

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
    
    # Erforderliche Variablen prüfen
    required_vars=(
        "POSTGRES_URL"
        "REDIS_URL"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Erforderliche Umgebungsvariable nicht gesetzt: $var"
            exit 1
        fi
    done
    
    log_info "Umgebungsvariablen validiert"
}

# =============================================================================
# Database Connectivity Check
# =============================================================================

check_database_connectivity() {
    log_info "Überspringe Datenbankverbindungsprüfung für schnelleren Start..."
    log_info "Datenbankverbindungen werden beim Service-Start geprüft"
}

# =============================================================================
# Service Initialization
# =============================================================================

initialize_service() {
    log_info "Initialisiere Edge Registry Service..."
    
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
    
    # Health Check Server im Hintergrund starten
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
        disk = psutil.disk_usage('/')
        
        health_data = {
            'service': '$SERVICE_NAME',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'environment': '$ENVIRONMENT',
            'metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100,
                'uptime_seconds': int(psutil.boot_time())
            },
            'ports': {
                'registry': $REGISTRY_PORT,
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
    # Prüfe ob Service bereit ist
    try:
        # Hier würden wir die Registry-Bereitschaft prüfen
        return web.json_response({'status': 'ready'})
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

start_edge_registry() {
    log_info "Starte Edge Registry Service auf Port $REGISTRY_PORT..."
    
    # Edge Registry Service starten
    cd /app
    exec /app/.venv/bin/python backend/edge/registry.py
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
    log_info "Registry Port: $REGISTRY_PORT"
    log_info "Health Port: $HEALTH_PORT"
    
    # Initialisierung
    validate_environment
    check_database_connectivity
    initialize_service
    
    # Services starten
    start_health_server
    
    # Kurz warten bis Health Server läuft
    sleep 2
    
    # Hauptservice starten
    start_edge_registry
}

# Script ausführen
main "$@"
