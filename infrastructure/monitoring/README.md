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
