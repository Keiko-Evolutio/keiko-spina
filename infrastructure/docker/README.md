# Backend API
http://localhost:8000

Vollständige Container-Übersicht (20 Container):
Base Services (5):
✅ keiko-postgres (PostgreSQL Database)
✅ keiko-redis (Redis Cache)
✅ keiko-nats (NATS Message Broker)
✅ keiko-pgadmin (PostgreSQL Admin Interface) - http://localhost:5050
✅ keiko-redis-insight (Redis Management Interface) - http://localhost:8002
Monitoring & Observability (5):
✅ keiko-prometheus (Metrics Collection) - http://localhost:9090
✅ keiko-grafana (Dashboard) - http://localhost:3001
✅ keiko-jaeger (Distributed Tracing) - http://localhost:16686
✅ keiko-alertmanager (Alert Management) - http://localhost:9093
✅ keiko-otel-collector (OpenTelemetry Collector)
Edge Computing (6):
✅ keiko-edge-registry (Container Registry) - http://localhost:8080
✅ keiko-edge-node-1 (Edge Node 1) - http://localhost:8082
✅ keiko-edge-node-2 (Edge Node 2) - http://localhost:8084
✅ keiko-edge-node-3 (Edge Node 3) - http://localhost:8086
✅ keiko-edge-load-balancer (Load Balancer) - http://localhost:8088
✅ keiko-edge-monitor (Edge Monitor) - http://localhost:8090
Workflow & Tools (4):
✅ keiko-n8n (Workflow Automation) - http://localhost:5678
✅ keiko-n8n-postgres (n8n Database)
✅ keiko-mailhog (Email Testing) - http://localhost:8025
✅ keiko-otel-healthcheck (Health Check)