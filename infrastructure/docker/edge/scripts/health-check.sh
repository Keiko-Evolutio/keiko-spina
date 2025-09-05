#!/bin/bash

# Universal Health Check Script für Edge Services
# Prüft Health und Readiness von Edge-Services

set -euo pipefail

# =============================================================================
# Konfiguration
# =============================================================================

SERVICE_NAME="${1:-unknown}"
HEALTH_PORT="${2:-8081}"
TIMEOUT="${3:-10}"

# =============================================================================
# Health Check Functions
# =============================================================================

check_health_endpoint() {
    local service="$1"
    local port="$2"
    local timeout="$3"
    
    # Health Endpoint prüfen
    if curl -f -s --max-time "$timeout" "http://localhost:$port/health" > /dev/null; then
        return 0
    else
        return 1
    fi
}

check_readiness_endpoint() {
    local service="$1"
    local port="$2"
    local timeout="$3"
    
    # Readiness Endpoint prüfen
    if curl -f -s --max-time "$timeout" "http://localhost:$port/ready" > /dev/null; then
        return 0
    else
        return 1
    fi
}

get_health_details() {
    local service="$1"
    local port="$2"
    local timeout="$3"
    
    # Detaillierte Health-Informationen abrufen
    curl -s --max-time "$timeout" "http://localhost:$port/health" 2>/dev/null || echo '{"status":"unknown"}'
}

# =============================================================================
# Service-spezifische Health Checks
# =============================================================================

check_edge_registry() {
    local port="$1"
    local timeout="$2"
    
    # Basis Health Check
    if ! check_health_endpoint "edge-registry" "$port" "$timeout"; then
        echo "UNHEALTHY: Health endpoint nicht erreichbar"
        return 1
    fi
    
    # Registry-spezifische Checks
    local health_data=$(get_health_details "edge-registry" "$port" "$timeout")
    local status=$(echo "$health_data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    
    if [[ "$status" != "healthy" ]]; then
        echo "UNHEALTHY: Service-Status ist $status"
        return 1
    fi
    
    echo "HEALTHY: Edge Registry läuft normal"
    return 0
}

check_edge_node() {
    local port="$1"
    local timeout="$2"
    
    # Basis Health Check
    if ! check_health_endpoint "edge-node" "$port" "$timeout"; then
        echo "UNHEALTHY: Health endpoint nicht erreichbar"
        return 1
    fi
    
    # Node-spezifische Checks
    local health_data=$(get_health_details "edge-node" "$port" "$timeout")
    local status=$(echo "$health_data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    local cpu_percent=$(echo "$health_data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('metrics', {}).get('cpu_percent', 0))")
    local memory_percent=$(echo "$health_data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('metrics', {}).get('memory_percent', 0))")
    
    if [[ "$status" != "healthy" ]]; then
        echo "UNHEALTHY: Service-Status ist $status"
        return 1
    fi
    
    # Ressourcen-Checks
    if (( $(echo "$cpu_percent > 90" | bc -l) )); then
        echo "WARNING: Hohe CPU-Auslastung ($cpu_percent%)"
    fi
    
    if (( $(echo "$memory_percent > 90" | bc -l) )); then
        echo "WARNING: Hohe Memory-Auslastung ($memory_percent%)"
    fi
    
    # Readiness Check
    if ! check_readiness_endpoint "edge-node" "$port" "$timeout"; then
        echo "UNHEALTHY: Node nicht bereit"
        return 1
    fi
    
    echo "HEALTHY: Edge Node läuft normal"
    return 0
}

check_edge_load_balancer() {
    local port="$1"
    local timeout="$2"
    
    # Basis Health Check
    if ! check_health_endpoint "edge-load-balancer" "$port" "$timeout"; then
        echo "UNHEALTHY: Health endpoint nicht erreichbar"
        return 1
    fi
    
    # Load Balancer-spezifische Checks
    local health_data=$(get_health_details "edge-load-balancer" "$port" "$timeout")
    local status=$(echo "$health_data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    
    if [[ "$status" != "healthy" ]]; then
        echo "UNHEALTHY: Service-Status ist $status"
        return 1
    fi
    
    echo "HEALTHY: Edge Load Balancer läuft normal"
    return 0
}

check_edge_monitor() {
    local port="$1"
    local timeout="$2"
    
    # Basis Health Check
    if ! check_health_endpoint "edge-monitor" "$port" "$timeout"; then
        echo "UNHEALTHY: Health endpoint nicht erreichbar"
        return 1
    fi
    
    # Monitor-spezifische Checks
    local health_data=$(get_health_details "edge-monitor" "$port" "$timeout")
    local status=$(echo "$health_data" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    
    if [[ "$status" != "healthy" ]]; then
        echo "UNHEALTHY: Service-Status ist $status"
        return 1
    fi
    
    echo "HEALTHY: Edge Monitor läuft normal"
    return 0
}

# =============================================================================
# Main Health Check Logic
# =============================================================================

perform_health_check() {
    local service="$1"
    local port="$2"
    local timeout="$3"
    
    case "$service" in
        "edge-registry")
            check_edge_registry "$port" "$timeout"
            ;;
        "edge-node")
            check_edge_node "$port" "$timeout"
            ;;
        "edge-load-balancer")
            check_edge_load_balancer "$port" "$timeout"
            ;;
        "edge-monitor")
            check_edge_monitor "$port" "$timeout"
            ;;
        *)
            # Generischer Health Check
            if check_health_endpoint "$service" "$port" "$timeout"; then
                echo "HEALTHY: $service läuft"
                return 0
            else
                echo "UNHEALTHY: $service nicht erreichbar"
                return 1
            fi
            ;;
    esac
}

# =============================================================================
# Utility Functions
# =============================================================================

show_usage() {
    cat << EOF
Usage: $0 <service-name> [health-port] [timeout]

Service Names:
  edge-registry       - Edge Node Registry
  edge-node          - Edge Processing Node
  edge-load-balancer - Edge Load Balancer
  edge-monitor       - Edge Performance Monitor

Parameters:
  health-port        - Health check port (default: 8081)
  timeout           - Timeout in seconds (default: 10)

Examples:
  $0 edge-registry
  $0 edge-node 8081 5
  $0 edge-load-balancer 8081 10
EOF
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Parameter validieren
    if [[ $# -lt 1 ]]; then
        echo "ERROR: Service-Name ist erforderlich"
        show_usage
        exit 1
    fi
    
    local service="$1"
    local port="${2:-8081}"
    local timeout="${3:-10}"
    
    # bc für Floating-Point-Arithmetik installieren falls nötig
    if ! command -v bc &> /dev/null; then
        echo "WARNING: bc nicht verfügbar, überspringe numerische Checks"
    fi
    
    # Health Check durchführen
    if perform_health_check "$service" "$port" "$timeout"; then
        exit 0
    else
        exit 1
    fi
}

# Script ausführen
main "$@"
