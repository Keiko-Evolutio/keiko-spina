# Edge Computing Docker Infrastructure

## 📋 Übersicht

Diese Docker-Infrastructure stellt eine vollständige Edge Computing-Umgebung für das Keiko Voice-Service-System bereit.
Sie umfasst Edge-Nodes, Load Balancing, Performance-Monitoring und automatische Service-Discovery.

## 🏗️ Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    Edge Computing Infrastructure            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │  Edge Registry  │    │ Load Balancer   │                 │
│  │   Port: 8080    │    │   Port: 8088    │                 │
│  └─────────────────┘    └─────────────────┘                 │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────┐ │
│  │   Edge Node 1   │    │   Edge Node 2   │    │Edge Node3│ │
│  │ Audio Processor │    │ Audio Processor │    │AI Inference│ 
│  │   Port: 8082    │    │   Port: 8084    │    │Port: 8086│ │
│  └─────────────────┘    └─────────────────┘    └──────────┘ │
│                                                             │
│  ┌─────────────────┐                                        │
│  │ Performance     │                                        │
│  │ Monitor         │                                        │
│  │   Port: 8090    │                                        │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Edge-Infrastructure starten

```bash
# Alle Edge-Services starten
make edge-up

# Oder manuell mit Docker Compose
docker-compose -f docker-compose.dev.yml up -d edge-registry edge-node-1 edge-node-2 edge-node-3 edge-load-balancer edge-monitor
```

### 2. Status prüfen

```bash
# Service-Status anzeigen
make edge-status

# Health Checks durchführen
make edge-health

# Logs anzeigen
make edge-logs
```

### 3. Edge-Services verwenden

```bash
# Edge Registry API
curl http://localhost:8080/api/v1/nodes

# Edge Node 1 Health
curl http://localhost:8083/health

# Load Balancer Status
curl http://localhost:8089/health
```

## 📦 Services

### Edge Registry (Port 8080)

**Zweck**: Zentrale Registry für Edge-Node-Management und Service-Discovery

**Endpoints**:

- `GET /api/v1/nodes` - Liste aller registrierten Nodes
- `POST /api/v1/nodes/register` - Node registrieren
- `DELETE /api/v1/nodes/{node_id}` - Node abmelden
- `GET /health` - Health Check (Port 8081)

**Konfiguration**: `docker/edge/config/edge-registry.yml`

### Edge Node 1 (Port 8082)

**Zweck**: Audio-Processing-Node für US-East Region

**Capabilities**:

- Voice Activity Detection (VAD)
- Noise Reduction
- Audio Enhancement

**Endpoints**:

- `POST /api/v1/process` - Audio-Verarbeitung
- `GET /api/v1/capabilities` - Verfügbare Capabilities
- `GET /health` - Health Check (Port 8083)

### Edge Node 2 (Port 8084)

**Zweck**: Audio-Processing-Node für US-West Region

**Capabilities**:

- Voice Activity Detection (VAD)
- Noise Reduction
- Audio Enhancement
- Spectral Analysis

**Endpoints**:

- `POST /api/v1/process` - Audio-Verarbeitung
- `GET /api/v1/models` - Verfügbare Modelle
- `GET /health` - Health Check (Port 8085)

### Edge Node 3 (Port 8086)

**Zweck**: AI-Inference-Node für EU-Central Region

**Capabilities**:

- Speaker Identification
- Emotion Detection
- Language Detection

**Endpoints**:

- `POST /api/v1/inference` - AI-Inferenz
- `GET /api/v1/models` - Verfügbare AI-Modelle
- `GET /health` - Health Check (Port 8087)

### Edge Load Balancer (Port 8088)

**Zweck**: Intelligente Lastverteilung zwischen Edge-Nodes

**Features**:

- Adaptive Routing-Algorithmen
- Latenz-basierte Entscheidungen
- Kapazitäts-Monitoring
- Failover-Mechanismen

**Endpoints**:

- `POST /api/v1/route` - Request-Routing
- `GET /api/v1/metrics` - Load Balancing-Metriken
- `GET /health` - Health Check (Port 8089)

### Edge Performance Monitor (Port 8090)

**Zweck**: Real-time Performance-Monitoring und Alerting

**Features**:

- Latenz-Monitoring
- Durchsatz-Metriken
- Resource-Utilization
- Prometheus-Integration

**Endpoints**:

- `GET /api/v1/metrics` - Performance-Metriken
- `GET /api/v1/alerts` - Aktive Alerts
- `GET /metrics` - Prometheus-Metriken (Port 9091)
- `GET /health` - Health Check (Port 8091)

## 🔧 Konfiguration

### Umgebungsvariablen

```bash
# Edge Registry
REGISTRY_PORT=8080
HEALTH_PORT=8081
POSTGRES_URL=postgresql://keiko:keiko_dev_password@postgres:5432/keiko
REDIS_URL=redis://redis:6379/2

# Edge Node 1
NODE_ID=edge-node-us-east-1
NODE_TYPE=audio-processor
NODE_REGION=us-east-1
REGISTRY_URL=http://edge-registry:8080
MAX_CONCURRENT_TASKS=20
SUPPORTED_CAPABILITIES=vad,noise-reduction,audio-enhancement
CACHE_SIZE_MB=512

# Edge Node 2
NODE_ID=edge-node-us-west-1
NODE_TYPE=audio-processor
NODE_REGION=us-west-1
MAX_CONCURRENT_TASKS=15
SUPPORTED_CAPABILITIES=vad,noise-reduction,audio-enhancement,spectral-analysis
CACHE_SIZE_MB=256

# Edge Node 3
NODE_ID=edge-node-eu-central-1
NODE_TYPE=ai-inference
NODE_REGION=eu-central-1
MAX_CONCURRENT_TASKS=10
SUPPORTED_CAPABILITIES=speaker-identification,emotion-detection,language-detection
CACHE_SIZE_MB=1024

# Load Balancer
STRATEGY=adaptive
LATENCY_WEIGHT=0.4
CAPACITY_WEIGHT=0.3
RELIABILITY_WEIGHT=0.2
COST_WEIGHT=0.1

# Performance Monitor
METRICS_INTERVAL=10
ALERT_EVALUATION_INTERVAL=30
LATENCY_THRESHOLD_MS=100
CPU_THRESHOLD_PERCENT=80
MEMORY_THRESHOLD_PERCENT=85
```

### Konfigurationsdateien

- **Edge Registry**: `docker/edge/config/edge-registry.yml`
- **Edge Nodes**: `docker/edge/config/edge-node.yml`
- **Load Balancer**: `docker/edge/config/edge-load-balancer.yml`
- **Performance Monitor**: `docker/edge/config/edge-monitor.yml`

## 🧪 Testing

### Unit Tests

```bash
# Frontend Edge Tests
npm test -- tests/edge/

# Backend Edge Tests
pytest tests/edge/
```

### Integration Tests

```bash
# Edge Integration Tests
make edge-test-integration

# Performance Tests
make edge-test-performance
```

### Performance Benchmarks

```bash
# WASM vs JavaScript Performance
make edge-benchmark

# Latenz-Benchmarks
npm run benchmark -- edge-node-latency

# Cache Performance
npm run benchmark -- edge-cache-performance
```

## 📊 Monitoring

### Prometheus Metriken

```bash
# Edge-spezifische Metriken anzeigen
curl http://localhost:9091/metrics

# Prometheus UI öffnen
make edge-prometheus
```

**Verfügbare Metriken**:

- `edge_node_latency_seconds` - Node-Latenz
- `edge_task_duration_seconds` - Task-Verarbeitungszeit
- `edge_cache_hit_rate` - Cache-Hit-Rate
- `edge_node_cpu_usage` - CPU-Auslastung
- `edge_node_memory_usage` - Memory-Auslastung
- `edge_requests_total` - Gesamt-Requests
- `edge_errors_total` - Gesamt-Fehler

### Grafana Dashboards

```bash
# Grafana öffnen
make edge-grafana
```

**Verfügbare Dashboards**:

- Edge Computing Overview
- Edge Node Performance
- Edge Load Balancing
- Edge Cache Performance

### Health Monitoring

```bash
# Alle Health Checks
make edge-health

# Einzelne Service Health Checks
curl http://localhost:8081/health  # Registry
curl http://localhost:8083/health  # Node 1
curl http://localhost:8085/health  # Node 2
curl http://localhost:8087/health  # Node 3
curl http://localhost:8089/health  # Load Balancer
curl http://localhost:8091/health  # Monitor
```

## 🔧 Development

### Container-Shell-Zugriff

```bash
# Edge Registry Shell
make edge-shell-registry

# Edge Node Shells
make edge-shell-node1
make edge-shell-node2
make edge-shell-node3

# Load Balancer Shell
make edge-shell-balancer

# Performance Monitor Shell
make edge-shell-monitor
```

### Logs

```bash
# Alle Edge-Service-Logs
make edge-logs

# Spezifische Service-Logs
make edge-logs-registry
make edge-logs-nodes
make edge-logs-balancer
make edge-logs-monitor
```

### Hot Reload

Die Development-Container unterstützen Hot Reload:

```bash
# Code-Änderungen werden automatisch erkannt
# Services werden automatisch neu geladen
```

## 🚨 Troubleshooting

### Häufige Probleme

#### 1. Edge-Nodes registrieren sich nicht

```bash
# Registry-Logs prüfen
make edge-logs-registry

# Node-Logs prüfen
make edge-logs-nodes

# Netzwerk-Konnektivität testen
docker-compose -f docker-compose.dev.yml exec edge-node-1 curl http://edge-registry:8080/health
```

#### 2. Hohe Latenz

```bash
# Performance-Metriken prüfen
make edge-metrics

# Load Balancer-Status prüfen
curl http://localhost:8088/api/v1/metrics

# Node-Performance prüfen
curl http://localhost:8083/health
```

#### 3. Cache-Probleme

```bash
# Cache-Statistiken prüfen
curl http://localhost:8083/health | jq '.metrics.cache_files'

# Cache-Verzeichnis prüfen
make edge-shell-node1
ls -la /app/cache/
```

### Debug-Modus

```bash
# Debug-Logs aktivieren
export LOG_LEVEL=DEBUG

# Services neu starten
make edge-restart

# Debug-Logs anzeigen
make edge-logs
```

## 🧹 Cleanup

### Services stoppen

```bash
# Edge-Services stoppen
make edge-down

# Vollständige Bereinigung
make edge-clean

# Kompletter Reset
make edge-reset
```

### Volumes bereinigen

```bash
# Edge-Volumes entfernen
docker volume rm keiko_edge-node-1-cache
docker volume rm keiko_edge-node-2-cache
docker volume rm keiko_edge-node-3-cache
```

## 📚 Weitere Dokumentation

- **Edge Computing Integration**: `docs/EDGE_COMPUTING_INTEGRATION.md`
- **Frontend Edge API**: `frontend/src/edge/README.md`
- **Backend Edge API**: `backend/edge/README.md`
- **Performance Benchmarks**: `docs/EDGE_PERFORMANCE_BENCHMARKS.md`

## 🔗 Nützliche Links

- **Edge Registry API**: http://localhost:8080/docs
- **Edge Node APIs**: http://localhost:8082/docs, http://localhost:8084/docs, http://localhost:8086/docs
- **Load Balancer API**: http://localhost:8088/docs
- **Performance Monitor**: http://localhost:8090/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001
