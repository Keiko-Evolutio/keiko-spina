#!/bin/bash

# Edge Performance Monitor Startup Script
# Startet das Performance-Monitoring und Alerting für Edge-Infrastructure

set -euo pipefail

# =============================================================================
# Konfiguration
# =============================================================================

SERVICE_NAME="edge-monitor"
MONITOR_PORT="${MONITOR_PORT:-8090}"
HEALTH_PORT="${HEALTH_PORT:-8091}"
METRICS_PORT="${METRICS_PORT:-9091}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ENVIRONMENT="${ENVIRONMENT:-development}"
REGISTRY_URL="${REGISTRY_URL:-http://edge-registry:8080}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://prometheus:9090}"
ALERTMANAGER_URL="${ALERTMANAGER_URL:-http://alertmanager:9093}"

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
    
    log_info "Umgebungsvariablen validiert"
}

# =============================================================================
# Dependencies Connectivity Check
# =============================================================================

check_dependencies_connectivity() {
    log_info "Prüfe Verbindung zu Dependencies..."
    
    # Edge Registry prüfen
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$REGISTRY_URL/health" > /dev/null 2>&1; then
            log_info "Edge Registry erreichbar"
            break
        fi
        
        log_warn "Edge Registry nicht erreichbar (Versuch $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
        
        if [[ $attempt -gt $max_attempts ]]; then
            log_error "Edge Registry nach $max_attempts Versuchen nicht erreichbar"
            exit 1
        fi
    done
    
    # Prometheus prüfen (optional)
    if curl -f -s "$PROMETHEUS_URL/api/v1/status/config" > /dev/null 2>&1; then
        log_info "Prometheus erreichbar"
    else
        log_warn "Prometheus nicht erreichbar - Monitoring eingeschränkt"
    fi
    
    # Alertmanager prüfen (optional)
    if curl -f -s "$ALERTMANAGER_URL/api/v1/status" > /dev/null 2>&1; then
        log_info "Alertmanager erreichbar"
    else
        log_warn "Alertmanager nicht erreichbar - Alerting eingeschränkt"
    fi
}

# =============================================================================
# Service Initialization
# =============================================================================

initialize_service() {
    log_info "Initialisiere Performance Monitor Service..."
    
    # Logs-Verzeichnis erstellen
    mkdir -p /app/logs
    
    # Konfigurationsdatei prüfen
    if [[ ! -f "$EDGE_CONFIG_PATH" ]]; then
        log_error "Konfigurationsdatei nicht gefunden: $EDGE_CONFIG_PATH"
        exit 1
    fi
    
    # Alert-Rules prüfen
    if [[ ! -f "$ALERT_RULES_PATH" ]]; then
        log_warn "Alert-Rules-Datei nicht gefunden: $ALERT_RULES_PATH"
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
                'registry_url': '$REGISTRY_URL',
                'prometheus_url': '$PROMETHEUS_URL',
                'alertmanager_url': '$ALERTMANAGER_URL'
            },
            'ports': {
                'monitor': $MONITOR_PORT,
                'health': $HEALTH_PORT,
                'metrics': $METRICS_PORT
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
        # Prüfe ob Monitor bereit ist
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
# Prometheus Metrics Server
# =============================================================================

start_metrics_server() {
    log_info "Starte Prometheus Metrics Server auf Port $METRICS_PORT..."
    
    /app/.venv/bin/python -c "
import time
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import psutil
import threading

# Prometheus Metriken definieren
edge_node_count = Gauge('edge_node_count', 'Anzahl der Edge-Nodes')
edge_node_latency = Histogram('edge_node_latency_seconds', 'Edge-Node-Latenz')
edge_task_duration = Histogram('edge_task_duration_seconds', 'Task-Verarbeitungszeit')
edge_cache_hit_rate = Gauge('edge_cache_hit_rate', 'Cache-Hit-Rate')
edge_requests_total = Counter('edge_requests_total', 'Gesamt-Requests')
edge_errors_total = Counter('edge_errors_total', 'Gesamt-Fehler')

def collect_metrics():
    while True:
        try:
            # System-Metriken sammeln
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Mock Edge-Metriken (in echter Implementation würden diese von Registry kommen)
            edge_node_count.set(3)  # 3 Edge-Nodes
            edge_cache_hit_rate.set(0.85)  # 85% Hit-Rate
            
            time.sleep(10)  # Alle 10 Sekunden
        except Exception as e:
            print(f'Fehler beim Sammeln der Metriken: {e}')
            time.sleep(10)

# Metrics-Collection im Hintergrund starten
metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
metrics_thread.start()

# Prometheus HTTP Server starten
start_http_server($METRICS_PORT)
print('Prometheus Metrics Server gestartet auf Port $METRICS_PORT')

# Server am Leben halten
while True:
    time.sleep(1)
" &
    
    METRICS_PID=$!
    echo $METRICS_PID > /tmp/metrics-server.pid
    
    log_info "Prometheus Metrics Server gestartet (PID: $METRICS_PID)"
}

# =============================================================================
# Main Service
# =============================================================================

start_performance_monitor() {
    log_info "Starte Edge Performance Monitor Service auf Port $MONITOR_PORT..."
    
    # Edge Performance Monitor Service starten
    cd /app
    exec /app/.venv/bin/python backend/edge/performance_monitor.py
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
    
    # Metrics Server stoppen
    if [[ -f /tmp/metrics-server.pid ]]; then
        METRICS_PID=$(cat /tmp/metrics-server.pid)
        if kill -0 "$METRICS_PID" 2>/dev/null; then
            log_info "Stoppe Metrics Server (PID: $METRICS_PID)"
            kill "$METRICS_PID"
        fi
        rm -f /tmp/metrics-server.pid
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
    log_info "Monitor Port: $MONITOR_PORT"
    log_info "Health Port: $HEALTH_PORT"
    log_info "Metrics Port: $METRICS_PORT"
    log_info "Registry URL: $REGISTRY_URL"
    
    # Initialisierung
    validate_environment
    check_dependencies_connectivity
    initialize_service
    
    # Services starten
    start_health_server
    start_metrics_server
    
    # Kurz warten bis Server laufen
    sleep 3
    
    # Hauptservice starten
    start_performance_monitor
}

# Script ausführen
main "$@"
