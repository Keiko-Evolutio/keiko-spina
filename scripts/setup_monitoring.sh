#!/bin/bash
# Monitoring and Observability Setup Script
# Sets up comprehensive monitoring for the separated repositories

set -e

echo "📊 Setting up Monitoring and Observability Infrastructure"

# Configuration
MONITORING_ROOT="$(pwd)"
MONITORING_DATE=$(date +"%Y%m%d_%H%M%S")

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create monitoring directory structure
create_monitoring_structure() {
    log_info "Creating monitoring directory structure..."
    
    mkdir -p monitoring/{prometheus,grafana,jaeger,alertmanager}
    mkdir -p monitoring/grafana/{dashboards,provisioning/{dashboards,datasources}}
    mkdir -p monitoring/prometheus/rules
    mkdir -p monitoring/configs
    
    log_success "Monitoring directory structure created"
}

# Create Prometheus configuration
create_prometheus_config() {
    log_info "Creating Prometheus configuration..."
    
    cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Backend service monitoring
  - job_name: 'keiko-backend'
    static_configs:
      - targets: ['keiko-backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    scrape_timeout: 5s

  # Frontend monitoring (if metrics endpoint exists)
  - job_name: 'keiko-frontend'
    static_configs:
      - targets: ['keiko-frontend:3000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    scrape_timeout: 5s

  # Infrastructure monitoring
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nats-exporter'
    static_configs:
      - targets: ['nats:7777']

  # Node exporter for system metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # API Gateway/Load Balancer (if applicable)
  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
EOF
    
    log_success "Prometheus configuration created"
}

# Create Prometheus alerting rules
create_prometheus_rules() {
    log_info "Creating Prometheus alerting rules..."
    
    cat > monitoring/prometheus/rules/keiko-alerts.yml << 'EOF'
groups:
  - name: keiko-backend-alerts
    rules:
      - alert: BackendHighErrorRate
        expr: rate(http_requests_total{job="keiko-backend",code=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
          service: keiko-backend
        annotations:
          summary: "High error rate in Keiko Backend"
          description: "Error rate is {{ $value }} errors per second"

      - alert: BackendHighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="keiko-backend"}[5m])) > 1
        for: 5m
        labels:
          severity: warning
          service: keiko-backend
        annotations:
          summary: "High latency in Keiko Backend"
          description: "95th percentile latency is {{ $value }}s"

      - alert: BackendDown
        expr: up{job="keiko-backend"} == 0
        for: 1m
        labels:
          severity: critical
          service: keiko-backend
        annotations:
          summary: "Keiko Backend is down"
          description: "Keiko Backend has been down for more than 1 minute"

  - name: keiko-frontend-alerts
    rules:
      - alert: FrontendDown
        expr: up{job="keiko-frontend"} == 0
        for: 2m
        labels:
          severity: warning
          service: keiko-frontend
        annotations:
          summary: "Keiko Frontend is down"
          description: "Keiko Frontend has been down for more than 2 minutes"

  - name: infrastructure-alerts
    rules:
      - alert: PostgreSQLDown
        expr: up{job="postgres-exporter"} == 0
        for: 1m
        labels:
          severity: critical
          service: postgresql
        annotations:
          summary: "PostgreSQL is down"
          description: "PostgreSQL database has been down for more than 1 minute"

      - alert: RedisDown
        expr: up{job="redis-exporter"} == 0
        for: 1m
        labels:
          severity: critical
          service: redis
        annotations:
          summary: "Redis is down"
          description: "Redis cache has been down for more than 1 minute"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
          service: system
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 90%"

      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
          service: system
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is above 80%"
EOF
    
    log_success "Prometheus alerting rules created"
}

# Create Grafana datasource configuration
create_grafana_datasources() {
    log_info "Creating Grafana datasource configuration..."
    
    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: true
EOF
    
    log_success "Grafana datasource configuration created"
}

# Create Grafana dashboard provisioning
create_grafana_dashboards() {
    log_info "Creating Grafana dashboard provisioning..."
    
    cat > monitoring/grafana/provisioning/dashboards/dashboards.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /var/lib/grafana/dashboards
EOF
    
    # Create backend dashboard
    cat > monitoring/grafana/dashboards/keiko-backend-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Keiko Backend Dashboard",
    "tags": ["keiko", "backend"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"keiko-backend\"}[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"keiko-backend\",code=~\"5..\"}[5m])",
            "legendFormat": "5xx Errors"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"keiko-backend\"}[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{job=\"keiko-backend\"}[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Active Agents",
        "type": "stat",
        "targets": [
          {
            "expr": "registered_agents_total{job=\"keiko-backend\"}",
            "legendFormat": "Active Agents"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF
    
    log_success "Grafana dashboards created"
}

# Create AlertManager configuration
create_alertmanager_config() {
    log_info "Creating AlertManager configuration..."
    
    cat > monitoring/alertmanager/alertmanager.yml << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@keiko.ai'

route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        service: keiko-backend
      receiver: 'backend-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://localhost:5001/'
        title: 'Keiko Alert'
        text: 'Alert: {{ .GroupLabels.alertname }} - {{ .CommonAnnotations.summary }}'

  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-critical'
        title: 'CRITICAL: {{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: 'backend-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-backend'
        title: 'Backend Alert: {{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'service']
EOF
    
    log_success "AlertManager configuration created"
}

# Create Jaeger configuration
create_jaeger_config() {
    log_info "Creating Jaeger configuration..."
    
    cat > monitoring/jaeger/jaeger.yml << 'EOF'
# Jaeger configuration for distributed tracing
collector:
  zipkin:
    host-port: 9411

storage:
  type: memory
  memory:
    max-traces: 10000

query:
  base-path: /
EOF
    
    log_success "Jaeger configuration created"
}

# Create comprehensive docker-compose for monitoring
create_monitoring_docker_compose() {
    log_info "Creating monitoring docker-compose..."
    
    cat > monitoring/docker-compose.monitoring.yml << 'EOF'
version: '3.9'

services:
  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:latest
    container_name: keiko-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    restart: unless-stopped
    networks:
      - monitoring

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: keiko-grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    restart: unless-stopped
    networks:
      - monitoring

  # AlertManager for alerting
  alertmanager:
    image: prom/alertmanager:latest
    container_name: keiko-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    restart: unless-stopped
    networks:
      - monitoring

  # Jaeger for distributed tracing
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: keiko-jaeger
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Jaeger collector HTTP
      - "14250:14250"  # Jaeger collector gRPC
      - "6831:6831/udp"  # Jaeger agent
      - "6832:6832/udp"  # Jaeger agent
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    volumes:
      - jaeger_data:/badger
    restart: unless-stopped
    networks:
      - monitoring

  # Node Exporter for system metrics
  node-exporter:
    image: prom/node-exporter:latest
    container_name: keiko-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped
    networks:
      - monitoring

  # Redis Exporter
  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: keiko-redis-exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped
    networks:
      - monitoring

  # PostgreSQL Exporter
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: keiko-postgres-exporter
    ports:
      - "9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://keiko:keiko@postgres:5432/keiko?sslmode=disable
    depends_on:
      - postgres
    restart: unless-stopped
    networks:
      - monitoring

  # Infrastructure services (shared with main application)
  postgres:
    image: postgres:15-alpine
    container_name: keiko-postgres
    environment:
      POSTGRES_DB: keiko
      POSTGRES_USER: keiko
      POSTGRES_PASSWORD: keiko
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - monitoring

  redis:
    image: redis:7-alpine
    container_name: keiko-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
  jaeger_data:
  postgres_data:
  redis_data:

networks:
  monitoring:
    driver: bridge
    name: keiko-monitoring
EOF
    
    log_success "Monitoring docker-compose created"
}

# Create monitoring middleware for backend
create_monitoring_middleware() {
    log_info "Creating monitoring middleware for backend..."
    
    mkdir -p backend/monitoring
    
    cat > backend/monitoring/prometheus_middleware.py << 'EOF'
"""
Prometheus Monitoring Middleware for Keiko Backend
Provides metrics collection for the separated backend service
"""

import time
from typing import Callable
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Metrics definitions
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code', 'repository']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint', 'repository']
)

API_ERRORS = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type', 'repository']
)

AGENT_METRICS = Counter(
    'agent_operations_total',
    'Total agent operations',
    ['operation', 'agent_type', 'status']
)

REGISTERED_AGENTS = Counter(
    'registered_agents_total',
    'Total registered agents',
    ['agent_type', 'status']
)

async def prometheus_middleware(request: Request, call_next: Callable) -> Response:
    """Prometheus metrics collection middleware"""
    start_time = time.time()
    
    # Extract request information
    method = request.method
    endpoint = request.url.path
    repository = "keiko-backend"
    
    try:
        # Process request
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint,
            repository=repository
        ).observe(duration)
        
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=response.status_code,
            repository=repository
        ).inc()
        
        return response
        
    except Exception as e:
        # Record error metrics
        API_ERRORS.labels(
            endpoint=endpoint,
            error_type=type(e).__name__,
            repository=repository
        ).inc()
        raise

async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Agent-specific metrics functions
def record_agent_operation(operation: str, agent_type: str, status: str):
    """Record agent operation metrics"""
    AGENT_METRICS.labels(
        operation=operation,
        agent_type=agent_type,
        status=status
    ).inc()

def record_agent_registration(agent_type: str, status: str):
    """Record agent registration metrics"""
    REGISTERED_AGENTS.labels(
        agent_type=agent_type,
        status=status
    ).inc()
EOF
    
    log_success "Monitoring middleware created"
}

# Create health check endpoints
create_health_endpoints() {
    log_info "Creating health check endpoints..."
    
    cat > backend/monitoring/health_checks.py << 'EOF'
"""
Health Check Endpoints for Keiko Backend
Provides comprehensive health monitoring for the separated backend service
"""

from typing import Dict, Any
from datetime import datetime
import asyncio
import psutil
import asyncpg
import redis.asyncio as aioredis
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

router = APIRouter()

class HealthChecker:
    """Comprehensive health checker for backend services"""
    
    def __init__(self):
        self.startup_time = datetime.utcnow()
    
    async def check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity"""
        try:
            conn = await asyncpg.connect(
                host="localhost",
                port=5432,
                user="keiko",
                password="keiko",
                database="keiko",
                timeout=5.0
            )
            
            # Test query
            await conn.fetchval("SELECT 1")
            await conn.close()
            
            return {
                "status": "healthy",
                "response_time_ms": 0,
                "details": "Database connection successful"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "response_time_ms": 5000,
                "details": f"Database connection failed: {str(e)}"
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            redis_client = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
            
            # Test ping
            await redis_client.ping()
            await redis_client.close()
            
            return {
                "status": "healthy",
                "response_time_ms": 0,
                "details": "Redis connection successful"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "response_time_ms": 1000,
                "details": f"Redis connection failed: {str(e)}"
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            details = []
            
            if cpu_percent > 80:
                status = "degraded"
                details.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 85:
                status = "degraded" 
                details.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > 90:
                status = "degraded"
                details.append(f"High disk usage: {disk.percent:.1f}%")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "details": "; ".join(details) if details else "System resources normal"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": f"System resource check failed: {str(e)}"
            }
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start_time = datetime.utcnow()
        
        # Run all health checks concurrently
        db_check, redis_check = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            return_exceptions=True
        )
        
        system_check = self.check_system_resources()
        
        # Determine overall health
        checks = {
            "database": db_check if not isinstance(db_check, Exception) else {"status": "unhealthy", "details": str(db_check)},
            "redis": redis_check if not isinstance(redis_check, Exception) else {"status": "unhealthy", "details": str(redis_check)},
            "system": system_check
        }
        
        # Calculate overall status
        all_healthy = all(check["status"] == "healthy" for check in checks.values())
        any_unhealthy = any(check["status"] == "unhealthy" for check in checks.values())
        
        if all_healthy:
            overall_status = "healthy"
        elif any_unhealthy:
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        end_time = datetime.utcnow()
        check_duration = (end_time - start_time).total_seconds() * 1000
        
        return {
            "status": overall_status,
            "timestamp": end_time.isoformat(),
            "uptime_seconds": (end_time - self.startup_time).total_seconds(),
            "check_duration_ms": check_duration,
            "version": "1.0.0",
            "services": checks
        }

# Global health checker instance
health_checker = HealthChecker()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    result = await health_checker.comprehensive_health_check()
    
    status_code = status.HTTP_200_OK
    if result["status"] == "degraded":
        status_code = status.HTTP_200_OK  # Still return 200 for degraded
    elif result["status"] == "unhealthy":
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(content=result, status_code=status_code)

@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    result = await health_checker.comprehensive_health_check()
    
    status_code = status.HTTP_200_OK
    if result["status"] == "unhealthy":
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        content={"status": result["status"], "ready": result["status"] != "unhealthy"}, 
        status_code=status_code
    )
EOF
    
    log_success "Health check endpoints created"
}

# Create monitoring README
create_monitoring_readme() {
    log_info "Creating monitoring documentation..."
    
    cat > monitoring/README.md << 'EOF'
# Keiko Monitoring and Observability

Comprehensive monitoring stack for the Keiko Personal Assistant separated repositories.

## Components

### Core Monitoring Stack
- **Prometheus** - Metrics collection and storage
- **Grafana** - Metrics visualization and dashboards
- **AlertManager** - Alert routing and notification
- **Jaeger** - Distributed tracing

### Exporters
- **Node Exporter** - System metrics
- **PostgreSQL Exporter** - Database metrics
- **Redis Exporter** - Cache metrics

## Quick Start

1. **Start monitoring stack:**
   ```bash
   cd monitoring
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

2. **Access services:**
   - Grafana: http://localhost:3001 (admin/admin)
   - Prometheus: http://localhost:9090
   - AlertManager: http://localhost:9093
   - Jaeger: http://localhost:16686

## Metrics

### Backend Metrics
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration histogram
- `api_errors_total` - API error counter
- `agent_operations_total` - Agent operation counter
- `registered_agents_total` - Active agent counter

### System Metrics
- CPU usage
- Memory usage
- Disk usage
- Network I/O

### Database Metrics
- Connection pool stats
- Query performance
- Database size
- Transaction rates

## Alerts

### Critical Alerts
- Service down (Backend, Database, Redis)
- High error rates (>10% 5xx errors)
- High system resource usage (>90%)

### Warning Alerts
- High latency (>1s 95th percentile)
- Moderate resource usage (>80%)
- Agent registration failures

## Dashboards

### Backend Dashboard
- Request rate and error rate
- Response time percentiles
- Active agent count
- API endpoint performance

### Infrastructure Dashboard
- System resource usage
- Database performance
- Cache hit rates
- Network metrics

## Health Checks

### Endpoints
- `/health` - Comprehensive health check
- `/health/live` - Kubernetes liveness probe
- `/health/ready` - Kubernetes readiness probe

### Health Check Components
- Database connectivity
- Redis connectivity
- System resource usage
- Service dependencies

## Configuration

### Environment Variables
- `PROMETHEUS_RETENTION` - Metrics retention period (default: 15d)
- `GRAFANA_ADMIN_PASSWORD` - Grafana admin password
- `ALERTMANAGER_SLACK_URL` - Slack webhook for alerts

### Custom Metrics
Add custom metrics in your application:

```python
from prometheus_client import Counter, Histogram

# Custom counter
custom_operations = Counter('custom_operations_total', 'Custom operations', ['operation_type'])
custom_operations.labels(operation_type='my_operation').inc()

# Custom histogram
custom_duration = Histogram('custom_duration_seconds', 'Custom operation duration')
with custom_duration.time():
    # Your operation here
    pass
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing in Prometheus**
   - Check if service is exposing metrics on `/metrics` endpoint
   - Verify Prometheus scrape configuration
   - Check network connectivity between services

2. **Grafana dashboard not loading data**
   - Verify Prometheus datasource configuration
   - Check metric names and labels in queries
   - Ensure time range is appropriate

3. **Alerts not firing**
   - Check AlertManager configuration
   - Verify alert rule syntax in Prometheus
   - Test notification channels (Slack, email)

### Logs
- Prometheus logs: `docker logs keiko-prometheus`
- Grafana logs: `docker logs keiko-grafana`
- AlertManager logs: `docker logs keiko-alertmanager`

## Scaling

### High Availability
- Use Prometheus federation for multiple instances
- Deploy Grafana with shared storage
- Configure AlertManager clustering

### Performance
- Adjust scrape intervals based on needs
- Use recording rules for complex queries
- Implement metric retention policies

## Security

### Authentication
- Change default Grafana credentials
- Enable Prometheus authentication
- Secure AlertManager webhook endpoints

### Network Security
- Use TLS for inter-service communication
- Restrict access to monitoring ports
- Implement proper firewall rules

## Backup

### Critical Data
- Grafana dashboards and configuration
- Prometheus configuration and rules
- AlertManager routing configuration

### Backup Strategy
```bash
# Backup Grafana dashboards
docker exec keiko-grafana grafana-cli admin export-dashboard > grafana-backup.json

# Backup Prometheus configuration
docker cp keiko-prometheus:/etc/prometheus prometheus-backup/
```
EOF
    
    log_success "Monitoring documentation created"
}

# Main execution function
main() {
    log_info "Starting monitoring setup"
    log_info "Working directory: $MONITORING_ROOT"
    
    create_monitoring_structure
    create_prometheus_config
    create_prometheus_rules
    create_grafana_datasources
    create_grafana_dashboards
    create_alertmanager_config
    create_jaeger_config
    create_monitoring_docker_compose
    create_monitoring_middleware
    create_health_endpoints
    create_monitoring_readme
    
    log_success "Monitoring and observability setup completed!"
    echo ""
    log_info "Next steps:"
    log_info "1. Start monitoring stack: cd monitoring && docker-compose -f docker-compose.monitoring.yml up -d"
    log_info "2. Access Grafana: http://localhost:3001 (admin/admin)"
    log_info "3. Access Prometheus: http://localhost:9090"
    log_info "4. Access Jaeger: http://localhost:16686"
    log_info "5. Configure alerting channels in AlertManager"
    echo ""
    log_info "Integration:"
    log_info "- Add prometheus_middleware to your FastAPI backend"
    log_info "- Include health_checks router in your API routes"
    log_info "- Configure your services to expose metrics on /metrics"
}

# Check if script is being run from the correct directory
if [ ! -f "Repository-Trennung-Migrationsplan.md" ]; then
    log_error "Please run this script from the root of the keiko-personal-assistant repository"
    exit 1
fi

# Execute main function
main "$@"