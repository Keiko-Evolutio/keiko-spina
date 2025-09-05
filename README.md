# 🤖 Keiko Personal Assistant

[![CI/CD Pipeline](https://github.com/oscharko/keiko-personal-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/oscharko/keiko-personal-assistant/actions/workflows/ci.yml)
[![Security Scanning](https://github.com/oscharko/keiko-personal-assistant/actions/workflows/security.yml/badge.svg)](https://github.com/oscharko/keiko-personal-assistant/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/oscharko/keiko-personal-assistant/branch/main/graph/badge.svg)](https://codecov.io/gh/oscharko/keiko-personal-assistant)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Enterprise-grade Personal Assistant with MCP (Model Context Protocol) Integration**

Keiko is a sophisticated personal assistant platform that leverages the Model Context Protocol (MCP) to integrate with external tools and services, providing a unified interface for AI-powered automation and assistance.

## ✨ Features

### 🔌 **MCP Integration**
- **External Server Management**: Register and manage external MCP servers
- **Tool Discovery & Execution**: Automatic discovery and execution of external tools
- **Resource Access**: Seamless access to external resources and data
- **Prompt Management**: Dynamic prompt discovery and execution

### 🛡️ **Enterprise Security**
- **Multi-Authentication**: Bearer Token, mTLS, OIDC support
- **Rate Limiting**: Configurable rate limits with Redis backend
- **Audit Logging**: Comprehensive audit trails for compliance
- **Schema Validation**: JSON Schema validation for all tool parameters

### 📊 **Observability**
- **Prometheus Metrics**: Comprehensive metrics for monitoring
- **Distributed Tracing**: OpenTelemetry integration
- **Circuit Breakers**: Fault tolerance for external services
- **Health Checks**: Multi-level health monitoring

### ⚡ **Performance**
- **Async Architecture**: FastAPI with async/await throughout
- **Connection Pooling**: Optimized database and Redis connections
- **Caching**: Multi-layer caching for improved performance
- **Load Balancing**: Support for horizontal scaling

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Redis (for caching and rate limiting)
- PostgreSQL (optional, for persistent storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/oscharko/keiko-personal-assistant.git
cd keiko-personal-assistant

# Install dependencies
cd backend
pip install -e ".[dev]"

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Run the application
uvicorn main:app --reload
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

## 📋 API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Key Endpoints

#### Server Management
```bash
# Register MCP Server
POST /api/v1/mcp/external/servers/register
{
  "server_name": "weather-service",
  "base_url": "https://weather.example.com",
  "api_key": "your-api-key"
}

# List Servers
GET /api/v1/mcp/external/servers

# Remove Server
DELETE /api/v1/mcp/external/servers/{server_name}
```

#### Tool Operations
```bash
# Discover Tools
GET /api/v1/mcp/external/tools

# Invoke Tool
POST /api/v1/mcp/external/tools/invoke
{
  "server_name": "weather-service",
  "tool_name": "get_weather",
  "parameters": {
    "location": "München",
    "units": "metric"
  }
}
```

#### Monitoring
```bash
# Health Check
GET /health

# Prometheus Metrics
GET /metrics

# Circuit Breaker Status
GET /circuit-breakers
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run all tests
make test

# Run specific test categories
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-e2e           # End-to-end tests only
make test-conformance   # Protocol conformance tests
make test-security      # Security tests only

# Run with coverage
make test-cov

# Lint code
ruff check .
ruff format .

# Type checking
mypy .

# Security scan
bandit -r .
```

### Code Quality Standards

- **Linting**: Ruff (replaces flake8, black, isort)
- **Type Checking**: MyPy with strict configuration
- **Security**: Bandit security scanning
- **Testing**: Pytest with 85%+ coverage requirement
- **Documentation**: German comments, English identifiers

### Testing Strategy

Our comprehensive testing strategy is organized into distinct categories for better maintainability and CI/CD optimization:

#### 🧪 Test Categories

```bash
# Unit Tests - Fast, isolated component tests
make test-unit
cd backend && pytest tests/unit/ -v

# Integration Tests - Component interaction tests
make test-integration
cd backend && pytest tests/integration/ -v

# System Tests - Full system behavior tests
make test-system
cd backend && pytest tests/system/ -v

# End-to-End Tests - Complete user journey tests
make test-e2e
cd backend && pytest tests/e2e/ -v

# Conformance Tests - Protocol compliance tests
make test-conformance
cd backend && pytest tests/e2e/conformance/ -v

# Security Tests - Security validation tests
make test-security
cd backend && pytest tests/e2e/security/ -v

# Performance Tests - Latency and throughput tests
make test-performance
cd backend && pytest tests/perf/ tests/e2e/performance/ -v

# SDK Tests - Client SDK validation
make test-sdk
cd backend && pytest tests/sdk/ -v
```

#### 📁 Test Structure

```
backend/tests/
├── unit/                    # Fast, isolated unit tests
├── integration/             # Component integration tests
├── system/                  # System-level tests
├── perf/                    # Performance benchmarks
├── reliability/             # Fault tolerance tests
├── api/                     # API contract tests
├── e2e/                     # End-to-end tests
│   ├── conformance/         # Protocol conformance
│   ├── security/            # Security validation
│   ├── api/                 # API E2E tests
│   ├── performance/         # E2E performance tests
│   ├── observability/       # Monitoring tests
│   ├── middleware/          # Middleware tests
│   ├── audit/               # Audit logging tests
│   ├── selfverify/          # Self-verification tests
│   └── services/            # Service integration tests
├── sdk/                     # SDK client tests
└── plugins/                 # Test plugins and utilities
```

#### 🎯 Test Execution Guidelines

- **Unit Tests**: Run frequently during development (< 30 seconds)
- **Integration Tests**: Run before commits (< 2 minutes)
- **E2E Tests**: Run in CI/CD pipelines (< 10 minutes)
- **Performance Tests**: Run nightly or on-demand
- **Security Tests**: Run on every PR and release

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Keiko API     │    │  External MCP   │
│                 │◄──►│                 │◄──►│    Servers      │
│ • Web UI        │    │ • FastAPI       │    │ • Weather       │
│ • Mobile App    │    │ • Authentication│    │ • Calendar      │
│ • CLI Tools     │    │ • Rate Limiting │    │ • Email         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Infrastructure │
                    │ • Redis Cache   │
                    │ • PostgreSQL    │
                    │ • Prometheus    │
                    └─────────────────┘
```

### Key Components

- **API Layer**: FastAPI with automatic OpenAPI documentation
- **MCP Integration**: Client for Model Context Protocol servers
- **Authentication**: Multi-method auth with JWT and mTLS
- **Caching**: Redis-based caching for performance
- **Monitoring**: Prometheus metrics and health checks
- **Security**: Comprehensive security scanning and audit logging

## 📊 Monitoring & Observability

### Metrics

The application exposes Prometheus metrics at `/metrics`:

- `kei_mcp_servers_total`: Number of registered MCP servers
- `kei_mcp_tool_invocations_total`: Tool invocation counter
- `kei_mcp_tool_duration_seconds`: Tool execution time histogram
- `kei_mcp_circuit_breaker_state`: Circuit breaker status
- `kei_mcp_rate_limit_hits_total`: Rate limit hits
- `kei_mcp_auth_attempts_total`: Authentication attempts

### Health Checks

Multi-level health monitoring:

```bash
# Basic health
GET /health

# Detailed health with dependencies
GET /health?detailed=true
```

### Logging

Structured logging with:
- **Application Logs**: Standard application events
- **Audit Logs**: Security and compliance events
- **Access Logs**: HTTP request/response logging
- **Error Logs**: Exception and error tracking

## 🔒 Security

### Authentication Methods

1. **Bearer Token**: JWT-based authentication
2. **mTLS**: Mutual TLS with client certificates
3. **OIDC**: OpenID Connect integration

### Security Features

- **Rate Limiting**: Configurable per-endpoint limits
- **Input Validation**: JSON Schema validation
- **Audit Logging**: Comprehensive audit trails
- **Security Scanning**: Automated vulnerability detection
- **Secrets Management**: Secure credential handling

### Security Scanning

```bash
# Dependency vulnerabilities
safety check
pip-audit

# Code security issues
bandit -r .

# Secrets detection
detect-secrets scan
```

## 🚀 Deployment

### Production Deployment

```bash
# Build production image
docker build -t keiko-backend:latest .

# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/
```

### Environment Configuration

Key environment variables:

```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0

# Authentication
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# MCP Configuration
MCP_TIMEOUT_SECONDS=30
MCP_MAX_RETRIES=3

# Monitoring
PROMETHEUS_ENABLED=true
TRACING_ENABLED=true
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write tests for new features
- Update documentation as needed
- Ensure all CI checks pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Model Context Protocol](https://modelcontextprotocol.io/) - Tool integration standard
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [OpenTelemetry](https://opentelemetry.io/) - Observability framework

---

**Built with ❤️ by the Keiko Team**
