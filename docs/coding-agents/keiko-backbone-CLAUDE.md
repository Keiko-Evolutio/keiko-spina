# CLAUDE.md - keiko-backbone Infrastructure Team

This file provides comprehensive guidance for the **keiko-backbone Infrastructure Team** and their Claude Code agents when working on the central infrastructure services of the Keiko Multi-Agent Platform.

## Projektkontext

**keiko-backbone** ist der **Master Enterprise Infrastructure Orchestrator** des Kubernetes-basierten Multi-Agent-Systems. Als zentraler Infrastructure Services Container stellt backbone die fundamentale Infrastruktur bereit, die alle anderen Komponenten des Systems benötigen, um effektiv zu funktionieren.

**Kernverantwortung:** Ausschließlich zentrale Infrastrukturdienste - Authentication, Service Registry, Event Streams, Monitoring, Orchestration und Security Infrastructure. Das Team implementiert KEINE UI-, Contract- oder SDK-spezifische Business Logic.

**System-Abgrenzung:**
- ✅ **WAS backbone MACHT:** Infrastructure Services bereitstellen
- ❌ **WAS backbone NICHT MACHT:** UI-Logic, Contract-Definitionen, SDK-Tools entwickeln

## Architektonische Prinzipien

### 1. **Resilience by Design**
- **Circuit Breaker Pattern:** Automatische Isolation fehlerhafter Services
- **Bulkhead Pattern:** Ressourcen-Isolation kritischer Komponenten
- **Graceful Degradation:** Kontrollierte Funktionsreduktion bei Ressourcenknappheit
- **Self-Healing:** Automatische Recovery-Mechanismen

### 2. **Scalability First**
- **Horizontal Scaling:** Automatische Skalierung basierend auf Metriken
- **Predictive Scaling:** ML-basierte Skalierungsvorhersagen
- **Multi-Region Deployment:** Geografisch verteilte Deployments
- **Resource Optimization:** Intelligente Ressourcenallokation

### 3. **Security First Approach**
- **Zero-Trust Architecture:** Jede Kommunikation authentifiziert und verschlüsselt
- **Defense in Depth:** Mehrschichtige Sicherheitsmaßnahmen
- **Principle of Least Privilege:** Minimale Berechtigungen für alle Komponenten
- **Continuous Security Monitoring:** Real-Time Threat Detection

### 4. **Observability Excellence**
- **Three Pillars:** Metrics, Logs, Traces für vollständige Observability
- **eBPF-based Monitoring:** Kernel-level Monitoring ohne Performance-Impact
- **AIOps Integration:** KI-gestützte Anomalie-Erkennung und Auto-Remediation
- **Business Metrics:** Integration von Technical- und Business-Metriken

## Technische Kernkomponenten

### **Agent/MCP/Tool Registry System**
```python
# Beispiel: Service Registration Interface
@dataclass
class ServiceRegistration:
    service_id: str
    capabilities: List[str]
    health_check_endpoint: str
    sla_requirements: SLARequirements
    
class ServiceRegistry:
    async def register_service(self, registration: ServiceRegistration) -> bool
    async def discover_services(self, capability: str) -> List[ServiceInfo]
    async def health_check_all(self) -> Dict[str, HealthStatus]
```

**Verantwortlichkeiten:**
- Dynamisches Service Discovery mit Raft-basiertem Consensus
- Intelligent Load Balancing mit kognitiver Last-Berücksichtigung
- Automated Service Health Monitoring
- Capability-based Service Matching

### **Orchestrator-Agent System**
```python
class WorkflowOrchestrator:
    async def create_workflow(self, intent: BusinessIntent) -> WorkflowPlan
    async def execute_workflow(self, plan: WorkflowPlan) -> ExecutionResult
    async def handle_failure(self, workflow_id: str, error: Exception)
    async def optimize_workflow(self, metrics: PerformanceMetrics)
```

**Verantwortlichkeiten:**
- Intention-based Workflow Orchestration
- Saga-Pattern für verteilte Transaktionen
- Swarm Intelligence für Agent-Koordination
- Dynamic Workflow Adaptation

### **Event Store und Monitoring System**
```python
class EventStore:
    async def append_event(self, stream: str, event: DomainEvent)
    async def read_stream(self, stream: str, from_version: int)
    async def create_snapshot(self, stream: str, version: int)
    
class MonitoringSystem:
    async def collect_metrics(self, source: str, metrics: Dict[str, Any])
    async def trigger_alert(self, condition: AlertCondition)
    async def generate_health_report(self) -> SystemHealthReport
```

**Verantwortlichkeiten:**
- Event Sourcing mit unveränderlichen Event-Streams
- Multi-dimensional Metrics Collection (Resource, Application, Business, Security)
- OpenTelemetry-basiertes Distributed Tracing
- Intelligent Alerting mit ML-basierter Alert-Korrelation

## Schnittstellen zu anderen Subsystemen

### **Interface zu keiko-face (Human Interface Layer)**
```yaml
Provided Services:
  - Central Authentication Service (SSO/OAuth2)
  - Real-Time Event Streaming (WebSocket/SSE)
  - Agent Orchestration Gateway
  - System Health Aggregation

API Endpoints:
  - POST /auth/validate-token
  - GET /events/stream/{user_id}
  - POST /agents/execute-request
  - GET /health/system-status
```

### **Interface zu keiko-contracts (API Authority)**
```yaml
Consumed Services:
  - Infrastructure Contract Definitions
  - Service Registration Contract Templates
  - Event Schema Authority
  - Protocol Translation Rules

Integration Points:
  - Contract Validation Pipeline
  - Schema Evolution Notifications
  - Protocol Gateway Configuration
  - SLA Monitoring Contract Definitions
```

### **Interface zu keiko-agent-py-sdk (Development Gateway)**
```yaml
Provided Services:
  - Token-based Authentication Infrastructure
  - Event Stream Infrastructure
  - Service Registry Infrastructure (Read-Only)
  - Monitoring Data Aggregation

Security Boundaries:
  - Agent Token Validation
  - Permission-based Event Access
  - Resource Quota Enforcement
  - Health Check Aggregation
```

## Entwicklungsrichtlinien

### **Coding Standards**
```python
# Python Code Style
- Use Python 3.11+ with type hints
- Follow PEP 8 with black formatting
- Use dataclasses for data structures
- Implement async/await for I/O operations

# Error Handling
class InfrastructureError(Exception):
    def __init__(self, message: str, error_code: str, context: Dict[str, Any]):
        super().__init__(message)
        self.error_code = error_code
        self.context = context

# Logging Standards
import structlog
logger = structlog.get_logger()
await logger.ainfo("Service registered", 
                   service_id=service_id, 
                   capabilities=capabilities,
                   correlation_id=correlation_id)
```

### **Dependency Management**
- **Package Manager:** Use `uv` for fast dependency management
- **Requirements:** Separate dev and runtime dependencies
- **Pinning:** Pin exact versions for production stability
- **Security:** Regular CVE scanning of dependencies

### **Code Organization**
```
keiko-backbone/
├── src/
│   ├── core/           # Core infrastructure services
│   ├── registry/       # Service discovery and registration
│   ├── orchestrator/   # Workflow orchestration
│   ├── monitoring/     # Observability stack
│   ├── security/       # Authentication and authorization
│   ├── events/         # Event store and streaming
│   └── interfaces/     # External system interfaces
├── tests/              # Comprehensive test suite
├── config/             # Configuration templates
└── deployments/        # Kubernetes manifests
```

### **Quality Gates**
- **Test Coverage:** Minimum 85% code coverage
- **Type Checking:** MyPy with strict configuration
- **Security Scanning:** Automated SAST/DAST in CI/CD
- **Performance:** Latency P95 < 200ms, P99 < 500ms

## Sicherheitsanforderungen

### **Authentication & Authorization**
```python
# JWT Token Validation
class TokenValidator:
    def __init__(self, jwt_secret: str, algorithms: List[str]):
        self.jwt_secret = jwt_secret
        self.algorithms = algorithms
    
    async def validate_token(self, token: str) -> TokenClaims:
        # Implement JWT validation with rotation support
        pass
    
    async def check_permissions(self, token: TokenClaims, resource: str) -> bool:
        # Implement RBAC permission checking
        pass
```

### **Network Security**
- **mTLS Everywhere:** Mutual TLS for all service-to-service communication
- **Network Policies:** Kubernetes Network Policies for micro-segmentation
- **DDoS Protection:** Automatic DDoS detection and mitigation
- **Certificate Management:** Automatic certificate rotation with cert-manager

### **Data Protection**
- **Encryption at Rest:** All persistent data encrypted with AES-256
- **Encryption in Transit:** TLS 1.3 for all network communication
- **Key Management:** Integration with HashiCorp Vault or cloud KMS
- **Audit Logging:** Immutable audit trails with cryptographic proof

### **Zero-Trust Implementation**
```python
class ZeroTrustValidator:
    async def authenticate_request(self, request: Request) -> AuthResult:
        # Validate client identity
        # Check request source and destination
        # Verify permissions and policies
        # Log all access attempts
        pass
```

## Performance-Ziele

### **Service Level Objectives (SLOs)**
```yaml
Tier 0 Services (Critical):
  Availability: 99.99% (52.6 min downtime/year)
  Response Time: P95 < 100ms, P99 < 200ms
  Throughput: 50,000 requests/second/node
  Error Rate: < 0.01%

Tier 1 Services (Important):
  Availability: 99.95% (4.4 hours downtime/year)
  Response Time: P95 < 200ms, P99 < 500ms
  Throughput: 25,000 requests/second/node
  Error Rate: < 0.05%

Recovery Objectives:
  MTTR: < 5 minutes (automated), < 30 minutes (manual)
  MTBF: > 720 hours for critical services
  RPO: < 15 minutes for critical data
  RTO: < 15 minutes for critical services
```

### **Resource Optimization**
```python
# Example: Resource-aware scheduling
class ResourceOptimizer:
    async def schedule_workload(self, workload: Workload) -> NodeAssignment:
        # Consider CPU, memory, network, and storage
        # Implement bin-packing algorithms
        # Account for anti-affinity rules
        # Optimize for cost and performance
        pass
```

### **Scaling Strategies**
- **HPA:** Horizontal Pod Autoscaler with custom metrics
- **VPA:** Vertical Pod Autoscaler for right-sizing
- **Predictive Scaling:** ML-based capacity forecasting
- **Cluster Autoscaler:** Automatic node scaling

## Testing-Strategien

### **Testing Pyramid**
```python
# Unit Tests (70%)
class TestServiceRegistry:
    async def test_service_registration(self):
        registry = ServiceRegistry()
        result = await registry.register_service(mock_service)
        assert result.success
        assert result.service_id in registry.active_services

# Integration Tests (20%)
class TestEventStreamIntegration:
    async def test_end_to_end_event_flow(self):
        # Test complete event flow from producer to consumer
        pass

# System Tests (10%)
class TestSystemResilience:
    async def test_chaos_engineering_scenario(self):
        # Simulate node failures, network partitions
        pass
```

### **Test Infrastructure**
```yaml
Test Environment:
  - Kubernetes cluster with test namespace isolation
  - Database fixtures with realistic data volumes
  - Message queue with test topics
  - Monitoring stack for test observability

Test Data Management:
  - Factory pattern for test data generation
  - Database migrations for test schemas
  - Cleanup strategies for test isolation
```

### **Performance Testing**
```python
# Load Testing
import asyncio
import aiohttp

class LoadTester:
    async def simulate_concurrent_users(self, user_count: int, duration: int):
        # Simulate realistic user behavior
        # Measure latency, throughput, error rates
        # Generate performance reports
        pass
```

## Deployment-Überlegungen

### **Kubernetes Configuration**
```yaml
# Deployment Example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: keiko-backbone-registry
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: registry
        image: keiko/backbone-registry:v1.0.0
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
```

### **Configuration Management**
```python
# Environment-specific configuration
class BackboneConfig:
    database_url: str
    redis_url: str
    jwt_secret: str
    monitoring_enabled: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "BackboneConfig":
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            redis_url=os.getenv("REDIS_URL"),
            jwt_secret=os.getenv("JWT_SECRET"),
        )
```

### **Observability Setup**
```yaml
Monitoring Stack:
  - Prometheus for metrics collection
  - Grafana for visualization and alerting
  - Jaeger for distributed tracing
  - ELK/Loki for log aggregation

Key Metrics to Monitor:
  - Request latency and throughput
  - Error rates and types
  - Resource utilization (CPU, memory, disk, network)
  - Business metrics (agent registrations, workflow executions)
  - Security events (authentication failures, permission denials)
```

### **Deployment Patterns**
- **Blue-Green Deployments:** Zero-downtime deployments with instant rollback
- **Canary Releases:** Gradual rollout with automatic rollback on errors
- **Feature Flags:** Runtime feature toggling for risk mitigation
- **Database Migrations:** Backward-compatible schema changes

## Development Commands

### **Core Development Workflow**
```bash
# Setup Development Environment
make install                    # Install all dependencies with uv
make dev                       # Start essential containers (PostgreSQL, Redis, NATS, etc.)
make dev-backend-only          # Start only backbone services

# Code Quality
make lint                      # Run ruff linting
make format                    # Format code with ruff
make type-check               # Run MyPy type checking
make quality                  # Run all quality checks

# Testing
make test                     # Run all tests with 85%+ coverage
make test-cov                 # Generate HTML coverage report
make test-integration         # Run integration tests
make test-performance         # Run performance benchmarks

# Container Management
make build                    # Build Docker images
make push                     # Push images to registry
make deploy-staging           # Deploy to staging environment
make deploy-production        # Deploy to production
```

### **Troubleshooting Commands**
```bash
# Health Checks
kubectl get pods -n keiko-backbone
kubectl describe pod <pod-name> -n keiko-backbone
kubectl logs -f <pod-name> -n keiko-backbone

# Metrics and Monitoring
kubectl port-forward svc/prometheus 9090:9090
kubectl port-forward svc/grafana 3000:3000
kubectl port-forward svc/jaeger 16686:16686

# Database Operations
kubectl exec -it <postgres-pod> -- psql -U keiko -d backbone
```

## Important Notes

### **Multi-System Coordination**
- **Startup Sequence:** contracts → backbone → face → sdk
- **Cross-System Events:** backbone orchestrates system-wide events
- **Disaster Recovery:** backbone coordinates multi-system failover
- **Compliance:** backbone aggregates audit trails from all systems

### **Performance Considerations**
- **Connection Pooling:** Use connection pooling for all database connections
- **Caching Strategy:** Implement multi-level caching (in-memory, Redis, CDN)
- **Async Operations:** Use async/await for all I/O-bound operations
- **Resource Limits:** Set appropriate resource requests and limits

### **Security Best Practices**
- **Never log sensitive data:** Sanitize logs to prevent data leakage
- **Rotate secrets regularly:** Implement automatic secret rotation
- **Validate all inputs:** Use JSON Schema validation for all API inputs
- **Rate limiting:** Implement rate limiting to prevent abuse

The backbone team is responsible for the **foundational infrastructure** that enables the entire Keiko Multi-Agent Platform to operate reliably, securely, and at scale.