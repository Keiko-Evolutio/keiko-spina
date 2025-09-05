#!/bin/bash

# Edge Node Startup Script
# Startet einen Edge-Node für Audio-Processing oder AI-Inferenz

set -euo pipefail

# =============================================================================
# Konfiguration
# =============================================================================

SERVICE_NAME="edge-node"
NODE_ID="${NODE_ID:-edge-node-$(hostname)}"
NODE_TYPE="${NODE_TYPE:-audio-processor}"
NODE_REGION="${NODE_REGION:-us-east-1}"
NODE_PORT="${NODE_PORT:-8080}"
HEALTH_PORT="${HEALTH_PORT:-8081}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ENVIRONMENT="${ENVIRONMENT:-development}"
REGISTRY_URL="${REGISTRY_URL:-http://edge-registry:8080}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-10}"
CACHE_SIZE_MB="${CACHE_SIZE_MB:-256}"

# =============================================================================
# Logging Setup
# =============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$SERVICE_NAME:$NODE_ID] $1"
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
    
    # Node-spezifische Validierung
    if [[ -z "$NODE_ID" ]]; then
        log_error "NODE_ID ist erforderlich"
        exit 1
    fi
    
    if [[ -z "$REGISTRY_URL" ]]; then
        log_error "REGISTRY_URL ist erforderlich"
        exit 1
    fi
    
    # Node-Type validieren
    case "$NODE_TYPE" in
        "audio-processor"|"ai-inference"|"cache-node"|"load-balancer")
            log_info "Node-Type validiert: $NODE_TYPE"
            ;;
        *)
            log_error "Ungültiger Node-Type: $NODE_TYPE"
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
# Models and Cache Initialization
# =============================================================================

initialize_models_and_cache() {
    log_info "Initialisiere Modelle und Cache..."
    
    # Cache-Verzeichnis erstellen
    mkdir -p "$CACHE_PATH"
    
    # Modelle-Verzeichnis prüfen
    if [[ ! -d "$MODELS_PATH" ]]; then
        log_warn "Modelle-Verzeichnis nicht gefunden: $MODELS_PATH"
        mkdir -p "$MODELS_PATH"
    fi
    
    # Verfügbare Modelle auflisten
    if [[ -d "$MODELS_PATH" ]]; then
        log_info "Verfügbare Modelle:"
        find "$MODELS_PATH" -name "*.onnx" -o -name "*.pt" -o -name "*.pkl" | while read -r model; do
            log_info "  - $(basename "$model")"
        done
    fi
    
    # Cache-Größe konfigurieren
    log_info "Cache-Größe: ${CACHE_SIZE_MB}MB"
    
    log_info "Modelle und Cache initialisiert"
}

# =============================================================================
# Node Registration
# =============================================================================

register_with_registry() {
    log_info "Registriere Node bei Edge Registry..."
    
    # Node-Informationen sammeln
    local cpu_count=$(nproc)
    local memory_mb=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    local disk_gb=$(df -BG "$CACHE_PATH" | awk 'NR==2{print $2}' | sed 's/G//')
    
    # Unterstützte Capabilities basierend auf Node-Type
    local capabilities=""
    case "$NODE_TYPE" in
        "audio-processor")
            capabilities="vad,noise-reduction,audio-enhancement"
            ;;
        "ai-inference")
            capabilities="speaker-identification,emotion-detection,language-detection"
            ;;
        "cache-node")
            capabilities="model-caching,result-caching"
            ;;
    esac
    
    # Registrierungs-Payload erstellen
    local registration_data=$(cat <<EOF
{
    "node_id": "$NODE_ID",
    "node_type": "$NODE_TYPE",
    "endpoint": "http://$(hostname -i):$NODE_PORT",
    "region": "$NODE_REGION",
    "capabilities": "$capabilities",
    "supported_models": "$SUPPORTED_MODELS",
    "max_concurrent_tasks": $MAX_CONCURRENT_TASKS,
    "metadata": {
        "cpu_count": $cpu_count,
        "memory_mb": $memory_mb,
        "disk_gb": $disk_gb,
        "cache_size_mb": $CACHE_SIZE_MB,
        "environment": "$ENVIRONMENT"
    }
}
EOF
)
    
    # Bei Registry registrieren
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$registration_data" \
        "$REGISTRY_URL/nodes/register" || echo "ERROR")
    
    if [[ "$response" == "ERROR" ]]; then
        log_error "Registrierung bei Edge Registry fehlgeschlagen"
        exit 1
    fi
    
    log_info "Node erfolgreich bei Edge Registry registriert"
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
        disk = psutil.disk_usage('$CACHE_PATH')
        
        # Cache-Statistiken
        cache_files = len([f for f in os.listdir('$CACHE_PATH') if os.path.isfile(os.path.join('$CACHE_PATH', f))])
        cache_size_bytes = sum(os.path.getsize(os.path.join('$CACHE_PATH', f)) 
                              for f in os.listdir('$CACHE_PATH') 
                              if os.path.isfile(os.path.join('$CACHE_PATH', f)))
        
        health_data = {
            'service': '$SERVICE_NAME',
            'node_id': '$NODE_ID',
            'node_type': '$NODE_TYPE',
            'region': '$NODE_REGION',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'environment': '$ENVIRONMENT',
            'metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100,
                'cache_files': cache_files,
                'cache_size_mb': cache_size_bytes / (1024 * 1024),
                'max_concurrent_tasks': $MAX_CONCURRENT_TASKS,
                'current_tasks': 0  # Würde von echtem Service gesetzt
            },
            'capabilities': '$SUPPORTED_CAPABILITIES'.split(','),
            'models': '$SUPPORTED_MODELS'.split(','),
            'ports': {
                'node': $NODE_PORT,
                'health': $HEALTH_PORT
            }
        }
        
        return web.json_response(health_data)
        
    except Exception as e:
        error_data = {
            'service': '$SERVICE_NAME',
            'node_id': '$NODE_ID',
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }
        return web.json_response(error_data, status=500)

async def readiness_check(request):
    try:
        # Prüfe ob Node bereit ist
        models_available = os.path.exists('$MODELS_PATH') and len(os.listdir('$MODELS_PATH')) > 0
        cache_writable = os.access('$CACHE_PATH', os.W_OK)
        
        if models_available and cache_writable:
            return web.json_response({'status': 'ready'})
        else:
            return web.json_response({
                'status': 'not_ready',
                'models_available': models_available,
                'cache_writable': cache_writable
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

start_edge_node() {
    log_info "Starte Edge Node Service auf Port $NODE_PORT..."
    
    # Edge Node Service starten
    cd /app
    exec /app/.venv/bin/python backend/edge/node.py
}

# =============================================================================
# Signal Handlers
# =============================================================================

cleanup() {
    log_info "Fahre Services herunter..."
    
    # Bei Registry abmelden
    log_info "Melde Node bei Registry ab..."
    curl -s -X DELETE "$REGISTRY_URL/nodes/$NODE_ID" || true
    
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
    log_info "Node ID: $NODE_ID"
    log_info "Node Type: $NODE_TYPE"
    log_info "Region: $NODE_REGION"
    log_info "Environment: $ENVIRONMENT"
    log_info "Registry URL: $REGISTRY_URL"
    log_info "Node Port: $NODE_PORT"
    log_info "Health Port: $HEALTH_PORT"
    
    # Initialisierung
    validate_environment
    check_registry_connectivity
    initialize_models_and_cache
    
    # Services starten
    start_health_server
    
    # Kurz warten bis Health Server läuft
    sleep 2
    
    # Bei Registry registrieren
    register_with_registry
    
    # Hauptservice starten
    start_edge_node
}

# Script ausführen
main "$@"
