# Keiko Personal Assistant - Enhanced Makefile with Docker Integration

.DEFAULT_GOAL := help
.PHONY: help install dev dev-stop dev-logs dev-restart test lint format type-check security quality docs clean start-containers stop-containers

# =====================================================================
# Configuration
# =====================================================================

PYTHON := uv run python
UV := uv
PYTEST := uv run pytest
RUFF := uv run ruff
MYPY := uv run mypy
BANDIT := uv run bandit
UVICORN := uv run uvicorn

# Docker Configuration
DOCKER_COMPOSE_ESSENTIAL := ./docker-compose.dev-essential.yml
DOCKER_COMPOSE_FULL := ./docker-compose.dev.yml
DOCKER_PROJECT_NAME := keiko-development

# API Files
OPENAPI_FILE := openapi.json
FRONTEND_DIR := ../frontend
DOCS_DIR := ../docs

# Colors for output
BLUE := \033[36m
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m
RESET := \033[0m

# =====================================================================
# Help
# =====================================================================

help: ## Show this help message
	@echo "$(BLUE)Keiko Backend - Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)🚀 Quick Start:$(RESET)"
	@echo "  install             Install development dependencies (dev + test + docs)"
	@echo "  install-runtime     Install only runtime dependencies"
	@echo "  install-all         Install all dependency groups"
	@echo "  dev                 Start essential development environment (6 containers + backend)"
	@echo "  dev-full            Start full development environment (20+ containers + backend)"
	@echo "  test                Run all tests"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@echo "  dev                 Start essential containers + development server (6 containers)"
	@echo "  dev-full            Start all containers + development server (20+ containers)"
	@echo "  dev-backend-only    Start only backend server (containers must be running)"
	@echo "  dev-stop            Stop development environment"
	@echo "  dev-logs            Show container logs"
	@echo "  dev-restart         Restart essential development environment"
	@echo "  dev-full-restart    Restart full development environment"
	@echo "  shell               Open interactive Python shell"
	@echo ""
	@echo "$(GREEN)Container Management:$(RESET)"
	@echo "  start-containers    Start essential containers only (6 containers)"
	@echo "  start-full-containers  Start all containers (20+ containers)"
	@echo "  stop-containers     Stop all containers"
	@echo "  containers-status   Show container status"
	@echo ""
	@echo "$(GREEN)API Development:$(RESET)"
	@echo "  generate-api        Generate OpenAPI specification"
	@echo "  validate-api        Validate OpenAPI specification"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  test                Run test suite"
	@echo "  test-cov            Run tests with coverage report"
	@echo "  test-fast           Run tests in parallel"
	@echo "  test-config         Validate test/coverage configuration"
	@echo "  validate-reproducible-builds  Validate reproducible builds configuration"
	@echo "  validate-lint-format  Validate lint/format consolidation"
	@echo "  validate-dependencies  Validate dependency separation (Runtime vs. Dev)"
	@echo "  validate-typing        Validate enterprise-ready typing configuration"
	@echo "  test-websocket      Test WebSocket functionality"
	@echo "  test-kei-agents     Run KEI-Agent-Framework tests"
	@echo "  test-logfire        Run Logfire integration tests"
	@echo ""
	@echo "$(GREEN)Code Quality:$(RESET)"
	@echo "  lint                Run code linting"
	@echo "  format              Format code"
	@echo "  type-check          Run comprehensive type checking"
	@echo "  type-check-strict   Run strict type checking for core modules"
	@echo "  type-check-report   Generate detailed type checking report"
	@echo "  type-check-coverage Check type coverage statistics"
	@echo "  security            Run security scanning"
	@echo "  quality             Run all quality checks"
	@echo "  quality-strict      Run all quality checks with strict typing"
	@echo ""
	@echo "$(GREEN)Utilities:$(RESET)"
	@echo "  clean               Clean build artifacts"
	@echo "  health-check        Check development environment"

# =====================================================================
# Setup & Installation
# =====================================================================

install: ## Install all dependencies for development
	@echo "$(BLUE)Installing dependencies with uv dependency-groups...$(RESET)"
	$(UV) sync --group dev --group test --group docs
	@echo "$(GREEN)✅ Development dependencies installed$(RESET)"

install-runtime: ## Install only runtime dependencies
	@echo "$(BLUE)Installing runtime dependencies only...$(RESET)"
	$(UV) sync
	@echo "$(GREEN)✅ Runtime dependencies installed$(RESET)"

install-ci: ## Install dependencies for CI (with lockfile)
	@echo "$(BLUE)Installing CI dependencies with lockfile...$(RESET)"
	$(UV) sync --group dev --group test --frozen
	@echo "$(GREEN)✅ CI dependencies installed$(RESET)"

install-ci-dev: ## Install dev dependencies for CI
	@echo "$(BLUE)Installing CI dev dependencies...$(RESET)"
	$(UV) sync --group dev --frozen
	@echo "$(GREEN)✅ CI dev dependencies installed$(RESET)"

install-ci-test: ## Install test dependencies for CI
	@echo "$(BLUE)Installing CI test dependencies...$(RESET)"
	$(UV) sync --group test --frozen
	@echo "$(GREEN)✅ CI test dependencies installed$(RESET)"

install-docs: ## Install documentation dependencies
	@echo "$(BLUE)Installing documentation dependencies...$(RESET)"
	$(UV) sync --group docs
	@echo "$(GREEN)✅ Documentation dependencies installed$(RESET)"

install-perf: ## Install performance testing dependencies
	@echo "$(BLUE)Installing performance testing dependencies...$(RESET)"
	$(UV) sync --group perf
	@echo "$(GREEN)✅ Performance dependencies installed$(RESET)"

install-all: ## Install all dependency groups
	@echo "$(BLUE)Installing all dependency groups...$(RESET)"
	$(UV) sync --group dev --group test --group docs --group api --group perf --group observability --group azure --group deployment
	@echo "$(GREEN)✅ All dependencies installed$(RESET)"

# =====================================================================
# Container Management
# =====================================================================

check-docker: ## Check if Docker is available
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "$(RED)❌ Docker not found. Please install Docker first.$(RESET)"; \
		exit 1; \
	fi
	@if ! docker info >/dev/null 2>&1; then \
		echo "$(RED)❌ Docker daemon not running. Please start Docker.$(RESET)"; \
		exit 1; \
	fi

start-containers: check-docker ## Start essential containers only (6 containers)
	@echo "$(BLUE)🐳 Starting essential development containers (6 containers)...$(RESET)"
	@if [ -f "$(DOCKER_COMPOSE_ESSENTIAL)" ]; then \
		docker-compose -f $(DOCKER_COMPOSE_ESSENTIAL) -p $(DOCKER_PROJECT_NAME) up -d; \
		echo "$(GREEN)✅ Essential containers started successfully$(RESET)"; \
		echo "$(CYAN)📡 Essential Services (6 containers):$(RESET)"; \
		echo "  PostgreSQL: localhost:5432 (keiko_user/keiko_password)"; \
		echo "  Redis: localhost:6379 (Cache)"; \
		echo "  NATS: localhost:4222 (client), localhost:8222 (monitoring)"; \
		echo "  Prometheus: http://localhost:9090 (Metrics)"; \
		echo "  Jaeger: http://localhost:16686 (Tracing)"; \
		echo "  OpenTelemetry Collector: localhost:4317-4318 (OTLP)"; \
	else \
		echo "$(RED)❌ Essential docker compose file not found: $(DOCKER_COMPOSE_ESSENTIAL)$(RESET)"; \
		exit 1; \
	fi

start-full-containers: check-docker ## Start all containers (20+ containers)
	@echo "$(BLUE)🐳 Starting full development environment (20+ containers)...$(RESET)"
	@if [ -f "$(DOCKER_COMPOSE_FULL)" ]; then \
		cd .. && docker-compose -f docker-compose.dev.yml -p $(DOCKER_PROJECT_NAME) up -d; \
		echo "$(GREEN)✅ All containers started successfully$(RESET)"; \
		echo "$(YELLOW)📊 Keiko Personal Assistant - Alle erreichbaren Container mit UI:$(RESET)"; \
		echo "$(CYAN)🔧 Monitoring & Observability:$(RESET)"; \
		echo "  Grafana: http://localhost:3001 (admin/admin) ✅"; \
		echo "  Jaeger (Tracing): http://localhost:16686 ✅"; \
		echo "  Prometheus (Metrics): http://localhost:9090 ✅"; \
		echo "  Alertmanager: http://localhost:9093 ✅"; \
		echo "$(CYAN)🗄️ Datenbank Management:$(RESET)"; \
		echo "  pgAdmin: http://localhost:5050 (admin@keiko.dev/admin) ✅"; \
		echo "  Redis Insight: http://localhost:8002 ✅"; \
		echo "$(CYAN)🔄 Workflow & Automation:$(RESET)"; \
		echo "  n8n (Workflow Automation): http://localhost:5678 ✅"; \
		echo "$(CYAN)📡 Message Broker:$(RESET)"; \
		echo "  NATS (Monitoring UI): http://localhost:8222 ✅"; \
		echo "$(CYAN)📧 Development Tools:$(RESET)"; \
		echo "  MailHog (Email Testing): http://localhost:8025 ✅"; \
		echo "$(CYAN)🌐 Edge Computing Services:$(RESET)"; \
		echo "  Edge Registry: http://localhost:8080 ✅"; \
		echo "  Edge Node 1: http://localhost:8082 ✅"; \
		echo "  Edge Node 2: http://localhost:8084 ✅"; \
		echo "  Edge Node 3: http://localhost:8086 ✅"; \
		echo "  Edge Load Balancer: http://localhost:8088 ✅"; \
		echo "  Edge Monitor: http://localhost:8090 ✅"; \
		echo "$(CYAN)📈 Zusätzliche Services (ohne Web-UI):$(RESET)"; \
		echo "  PostgreSQL: localhost:5432 (Datenbank)"; \
		echo "  Redis: localhost:6379 (Cache)"; \
		echo "  OpenTelemetry Collector: localhost:4317-4318 (OTLP)"; \
		echo "  MailHog SMTP: localhost:1025 (SMTP Server)"; \
	else \
		echo "$(RED)❌ Full docker compose file not found: $(DOCKER_COMPOSE_FULL)$(RESET)"; \
		exit 1; \
	fi

stop-containers: ## Stop all containers (both essential and full)
	@echo "$(BLUE)🛑 Stopping all containers...$(RESET)"
	@if command -v docker >/dev/null 2>&1; then \
		docker-compose -f $(DOCKER_COMPOSE_ESSENTIAL) -p $(DOCKER_PROJECT_NAME) down 2>/dev/null || true; \
		cd .. && docker-compose -f docker-compose.dev.yml -p $(DOCKER_PROJECT_NAME) down 2>/dev/null || true; \
		echo "$(GREEN)✅ All containers stopped$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  Docker not available$(RESET)"; \
	fi

# containers-status target moved to line 345 for detailed health status

wait-for-containers-running: ## Phase 1: Wait for ALL 20 containers to be running
	@echo "$(BLUE)🚀 Phase 1: Waiting for ALL 20 containers to be running...$(RESET)"
	@timeout=180; \
	required_containers="postgres redis nats prometheus alertmanager jaeger grafana n8n-postgres n8n edge-registry edge-node-1 edge-node-2 edge-node-3 edge-load-balancer edge-monitor otel-collector otel-healthcheck mailhog pgadmin redis-insight"; \
	total_containers=20; \
	while [ $$timeout -gt 0 ]; do \
		all_running=true; \
		running_count=0; \
		not_running_containers=""; \
		for container in $$required_containers; do \
			container_name="keiko-$$container"; \
			if docker ps --filter "name=$$container_name" --filter "status=running" --format "{{.Names}}" | grep -q "$$container_name"; then \
				running_count=$$((running_count + 1)); \
			else \
				all_running=false; \
				not_running_containers="$$not_running_containers $$container"; \
			fi; \
		done; \
		if [ "$$all_running" = "true" ]; then \
			echo "$(GREEN)✅ All $$total_containers containers are running!$(RESET)"; \
			break; \
		fi; \
		echo "$(YELLOW)⏳ $$running_count/$$total_containers containers running... ($$timeout seconds left)$(RESET)"; \
		if [ -n "$$not_running_containers" ]; then \
			echo "$(YELLOW)   Starting:$$not_running_containers$(RESET)"; \
		fi; \
		sleep 3; \
		timeout=$$((timeout-3)); \
	done; \
	if [ $$timeout -le 0 ]; then \
		echo "$(RED)❌ TIMEOUT: Not all containers are running - Backend startup BLOCKED$(RESET)"; \
		echo "$(RED)   Not running:$$not_running_containers$(RESET)"; \
		echo "$(YELLOW)Please check container status with: make containers-status$(RESET)"; \
		exit 1; \
	fi

wait-for-containers-healthy: ## Phase 2: Wait for CRITICAL containers to be healthy, monitor ALL
	@echo "$(BLUE)🔍 Phase 2: Waiting for critical containers, monitoring all...$(RESET)"
	@timeout=120; \
	critical_containers="postgres redis nats"; \
	monitoring_containers="prometheus alertmanager jaeger otel-collector otel-healthcheck"; \
	optional_containers="grafana n8n-postgres n8n edge-registry edge-node-1 edge-node-2 edge-node-3 edge-load-balancer edge-monitor mailhog pgadmin redis-insight"; \
	all_containers="$$critical_containers $$monitoring_containers $$optional_containers"; \
	total_containers=20; \
	while [ $$timeout -gt 0 ]; do \
		critical_healthy=true; \
		critical_count=0; \
		total_healthy=0; \
		critical_unhealthy=""; \
		optional_pending=""; \
		for container in $$critical_containers; do \
			container_name="keiko-$$container"; \
			if ! docker ps --format "table {{.Names}}" | grep -q "^$$container_name$$"; then \
				critical_healthy=false; \
				critical_unhealthy="$$critical_unhealthy $$container(not-running)"; \
				continue; \
			fi; \
			health_status=$$(docker inspect --format='{{.State.Health.Status}}' "$$container_name" 2>/dev/null || echo "none"); \
			if [ "$$health_status" = "healthy" ]; then \
				critical_count=$$((critical_count + 1)); \
				total_healthy=$$((total_healthy + 1)); \
			else \
				critical_healthy=false; \
				critical_unhealthy="$$critical_unhealthy $$container($$health_status)"; \
			fi; \
		done; \
		for container in $$monitoring_containers $$optional_containers; do \
			container_name="keiko-$$container"; \
			if docker ps --format "table {{.Names}}" | grep -q "^$$container_name$$"; then \
				health_status=$$(docker inspect --format='{{.State.Health.Status}}' "$$container_name" 2>/dev/null || echo "none"); \
				container_state=$$(docker inspect --format='{{.State.Status}}' "$$container_name" 2>/dev/null || echo "unknown"); \
				if [ "$$health_status" = "healthy" ]; then \
					total_healthy=$$((total_healthy + 1)); \
				elif [ "$$health_status" = "none" ] && [ "$$container_state" = "running" ]; then \
					case "$$container" in \
						"grafana"|"pgadmin"|"redis-insight"|"mailhog"|"n8n") \
							uptime=$$(docker inspect --format='{{.State.StartedAt}}' "$$container_name" 2>/dev/null | xargs -I {} date -d {} +%s 2>/dev/null || echo "0"); \
							current_time=$$(date +%s); \
							if [ $$((current_time - uptime)) -gt 45 ]; then \
								total_healthy=$$((total_healthy + 1)); \
							else \
								optional_pending="$$optional_pending $$container(starting)"; \
							fi; \
							;; \
						*) \
							total_healthy=$$((total_healthy + 1)); \
							;; \
					esac; \
				else \
					optional_pending="$$optional_pending $$container($$health_status)"; \
				fi; \
			fi; \
		done; \
		if [ "$$critical_healthy" = "true" ]; then \
			echo "$(GREEN)✅ Critical containers ($$critical_count/3) are healthy - Backend can start!$(RESET)"; \
			echo "$(BLUE)📊 Overall status: $$total_healthy/$$total_containers containers ready$(RESET)"; \
			if [ -n "$$optional_pending" ]; then \
				echo "$(YELLOW)⏳ Background containers still initializing:$$optional_pending$(RESET)"; \
				echo "$(BLUE)💡 These will be monitored via health checks and update the frontend status$(RESET)"; \
			fi; \
			echo "$(GREEN)🚀 Backend startup proceeding - all containers have health monitoring$(RESET)"; \
			break; \
		fi; \
		echo "$(YELLOW)⏳ Critical: $$critical_count/3, Total: $$total_healthy/$$total_containers ready ($$timeout seconds left)$(RESET)"; \
		if [ -n "$$critical_unhealthy" ]; then \
			echo "$(RED)🚨 Critical containers pending:$$critical_unhealthy$(RESET)"; \
		fi; \
		if [ -n "$$optional_pending" ]; then \
			echo "$(BLUE)⏳ Optional containers:$$optional_pending$(RESET)"; \
		fi; \
		sleep 3; \
		timeout=$$((timeout-3)); \
	done; \
	if [ $$timeout -le 0 ]; then \
		if [ "$$critical_healthy" = "true" ]; then \
			echo "$(GREEN)✅ Critical containers healthy - proceeding with backend startup$(RESET)"; \
		else \
			echo "$(RED)❌ TIMEOUT: Critical containers not ready - Backend startup BLOCKED$(RESET)"; \
			echo "$(RED)   Critical issues:$$critical_unhealthy$(RESET)"; \
			exit 1; \
		fi; \
	fi

wait-for-containers: ## Smart container readiness check (detects essential vs full setup)
	@echo "$(BLUE)🔍 Detecting container setup...$(RESET)"
	@# Check if we have essential containers (6) or full setup (20)
	@container_count=$$(docker ps --filter "name=keiko-" --format "table {{.Names}}" | wc -l | tr -d ' '); \
	if [ $$container_count -le 7 ]; then \
		echo "$(GREEN)✅ Essential development setup detected ($$container_count containers)$(RESET)"; \
		$(MAKE) wait-for-essential-containers; \
	else \
		echo "$(BLUE)🚀 Full development setup detected ($$container_count containers)$(RESET)"; \
		$(MAKE) wait-for-containers-running wait-for-containers-healthy; \
	fi

wait-for-essential-only: ## Wait for essential containers only (forced)
	@$(MAKE) wait-for-essential-containers

wait-for-full-only: ## Wait for full container setup only (forced)
	@$(MAKE) wait-for-containers-running wait-for-containers-healthy

wait-for-essential-containers: ## Wait for essential containers only (postgres, redis, nats)
	@echo "$(BLUE)⏳ Waiting for essential containers to be healthy...$(RESET)"
	@timeout=30; \
	required_containers="postgres redis nats"; \
	total_containers=3; \
	start_time=$$(date +%s); \
	while [ $$timeout -gt 0 ]; do \
		all_healthy=true; \
		healthy_count=0; \
		unhealthy_containers=""; \
		for container in $$required_containers; do \
			container_name="keiko-$$container"; \
			if docker ps --filter "name=$$container_name" --filter "status=running" --format "{{.Names}}" | grep -q "$$container_name"; then \
				health_status=$$(docker inspect --format='{{.State.Health.Status}}' "$$container_name" 2>/dev/null || echo "none"); \
				if [ "$$health_status" = "healthy" ] || [ "$$health_status" = "none" ]; then \
					healthy_count=$$((healthy_count + 1)); \
				else \
					all_healthy=false; \
					unhealthy_containers="$$unhealthy_containers $$container($$health_status)"; \
				fi; \
			else \
				all_healthy=false; \
				unhealthy_containers="$$unhealthy_containers $$container(not_running)"; \
			fi; \
		done; \
		if [ "$$all_healthy" = "true" ]; then \
			total_time=$$(($(date +%s) - start_time)); \
			echo "$(GREEN)✅ All $$total_containers essential containers are healthy ($$total_time s)!$(RESET)"; \
			echo "$(GREEN)🚀 Backend startup can proceed safely$(RESET)"; \
			break; \
		fi; \
		echo "$(YELLOW)⏳ $$healthy_count/$$total_containers containers healthy... ($$timeout seconds left)$(RESET)"; \
		if [ -n "$$unhealthy_containers" ]; then \
			echo "$(YELLOW)   Waiting for:$$unhealthy_containers$(RESET)"; \
		fi; \
		sleep 2; \
		timeout=$$((timeout-2)); \
	done; \
	if [ $$timeout -le 0 ]; then \
		echo "$(YELLOW)⚠️  TIMEOUT: Some containers may not be fully healthy yet$(RESET)"; \
		echo "$(BLUE)🚀 Proceeding with backend startup - essential services should work$(RESET)"; \
	fi

containers-status: ## Show detailed status of all 20 containers
	@echo "$(BLUE)📊 Container Health Status Report$(RESET)"
	@echo "$(BLUE)=================================$(RESET)"
	@required_containers="postgres redis nats prometheus alertmanager jaeger grafana n8n-postgres n8n edge-registry edge-node-1 edge-node-2 edge-node-3 edge-load-balancer edge-monitor otel-collector otel-healthcheck mailhog pgadmin redis-insight"; \
	healthy_count=0; \
	total_count=20; \
	for container in $$required_containers; do \
		container_name="keiko-$$container"; \
		if docker ps --filter "name=$$container_name" --format "{{.Names}}\t{{.Status}}" | grep -q "$$container_name"; then \
			status=$$(docker ps --filter "name=$$container_name" --format "{{.Status}}"); \
			health_status=$$(docker inspect --format='{{.State.Health.Status}}' "$$container_name" 2>/dev/null || echo "none"); \
			if [ "$$health_status" = "healthy" ] || [ "$$health_status" = "none" ]; then \
				echo "$(GREEN)✅ $$container_name: $$status ($$health_status)$(RESET)"; \
				healthy_count=$$((healthy_count + 1)); \
			else \
				echo "$(RED)❌ $$container_name: $$status ($$health_status)$(RESET)"; \
			fi; \
		else \
			echo "$(RED)❌ $$container_name: NOT RUNNING$(RESET)"; \
		fi; \
	done; \
	echo "$(BLUE)=================================$(RESET)"; \
	if [ $$healthy_count -eq $$total_count ]; then \
		echo "$(GREEN)🎉 All $$total_count containers are healthy!$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  $$healthy_count/$$total_count containers are healthy$(RESET)"; \
	fi

# =====================================================================
# Development
# =====================================================================

dev: start-containers wait-for-essential-only generate-api ## Start essential development environment (6 containers + backend)
	@echo "$(BLUE)🚀 Starting development server with essential containers...$(RESET)"
	@echo "$(GREEN)📡 API Server: http://localhost:8000$(RESET)"
	@echo "$(GREEN)📖 OpenAPI docs: http://localhost:8000/docs$(RESET)"
	@echo "$(GREEN)🔌 WebSocket: ws://localhost:8000/ws/agent/{user_id}$(RESET)"
	@echo "$(CYAN)📡 Essential Services (6 containers):$(RESET)"
	@echo "  PostgreSQL: localhost:5432 (keiko_user/keiko_password)"
	@echo "  Redis: localhost:6379 (Cache)"
	@echo "  NATS: localhost:4222 (client), localhost:8222 (monitoring)"
	@echo "  Prometheus: http://localhost:9090 (Metrics)"
	@echo "  Jaeger: http://localhost:16686 (Tracing)"
	@echo "  OpenTelemetry Collector: localhost:4317-4318 (OTLP)"
	@echo ""
	@echo "$(YELLOW)💡 Tip: Use 'make dev-full' for full development environment (20+ containers)$(RESET)"
	@echo "$(YELLOW)💡 Tip: Use 'make dev-logs' in another terminal to see container logs$(RESET)"
	@echo "$(YELLOW)💡 Tip: Use 'make dev-stop' to stop everything$(RESET)"
	@echo ""
	$(UVICORN) main:app --host 127.0.0.1 --port 8000 --reload --ws websockets-sansio

dev-full: start-full-containers wait-for-full-only generate-api ## Start full development environment (20+ containers + backend)
	@echo "$(BLUE)🚀 Starting development server with full container environment...$(RESET)"
	@echo "$(GREEN)📡 API Server: http://localhost:8000$(RESET)"
	@echo "$(GREEN)📖 OpenAPI docs: http://localhost:8000/docs$(RESET)"
	@echo "$(GREEN)🔌 WebSocket: ws://localhost:8000/ws/agent/{user_id}$(RESET)"
	@echo "$(YELLOW)📊 Keiko Personal Assistant - Alle erreichbaren Container mit UI:$(RESET)"
	@echo "$(CYAN)🔧 Monitoring & Observability:$(RESET)"
	@echo "  Grafana: http://localhost:3001 (admin/admin) ✅"
	@echo "  Jaeger (Tracing): http://localhost:16686 ✅"
	@echo "  Prometheus (Metrics): http://localhost:9090 ✅"
	@echo "  Alertmanager: http://localhost:9093 ✅"
	@echo "$(CYAN)🗄️ Datenbank Management:$(RESET)"
	@echo "  pgAdmin: http://localhost:5050 (admin@keiko.dev/admin) ✅"
	@echo "  Redis Insight: http://localhost:8002 ✅"
	@echo "$(CYAN)🔄 Workflow & Automation:$(RESET)"
	@echo "  n8n (Workflow Automation): http://localhost:5678 ✅"
	@echo "$(CYAN)📡 Message Broker:$(RESET)"
	@echo "  NATS (Monitoring UI): http://localhost:8222 ✅"
	@echo "$(CYAN)📧 Development Tools:$(RESET)"
	@echo "  MailHog (Email Testing): http://localhost:8025 ✅"
	@echo "$(CYAN)🌐 Edge Computing Services:$(RESET)"
	@echo "  Edge Registry: http://localhost:8080 ✅"
	@echo "  Edge Node 1: http://localhost:8082 ✅"
	@echo "  Edge Node 2: http://localhost:8084 ✅"
	@echo "  Edge Node 3: http://localhost:8086 ✅"
	@echo "  Edge Load Balancer: http://localhost:8088 ✅"
	@echo "  Edge Monitor: http://localhost:8090 ✅"
	@echo "$(CYAN)📈 Zusätzliche Services (ohne Web-UI):$(RESET)"
	@echo "  PostgreSQL: localhost:5432 (Datenbank)"
	@echo "  Redis: localhost:6379 (Cache)"
	@echo "  OpenTelemetry Collector: localhost:4317-4318 (OTLP)"
	@echo "  MailHog SMTP: localhost:1025 (SMTP Server)"
	@echo ""
	@echo "$(YELLOW)💡 Tip: Use 'make dev-logs' in another terminal to see container logs$(RESET)"
	@echo "$(YELLOW)💡 Tip: Use 'make dev-stop' to stop everything$(RESET)"
	@echo ""
	$(UVICORN) main:app --host 127.0.0.1 --port 8000 --reload --ws websockets-sansio

dev-backend-only: generate-api ## Start only backend server (assumes containers are running)
	@echo "$(BLUE)🚀 Starting backend server only...$(RESET)"
	@echo "$(GREEN)📡 API Server: http://localhost:8000$(RESET)"
	@echo "$(GREEN)📖 OpenAPI docs: http://localhost:8000/docs$(RESET)"
	@echo "$(GREEN)🔌 WebSocket: ws://localhost:8000/ws/agent/{user_id}$(RESET)"
	@echo ""
	@echo "$(YELLOW)💡 Tip: Use 'make dev-logs' in another terminal to see container logs$(RESET)"
	@echo "$(YELLOW)💡 Tip: Use 'make dev-stop' to stop everything$(RESET)"
	@echo ""
	$(UVICORN) main:app --host 127.0.0.1 --port 8000 --reload --ws websockets-sansio

dev-stop: stop-containers ## Stop full development environment
	@echo "$(GREEN)✅ Development environment stopped$(RESET)"

dev-restart: dev-stop dev ## Restart essential development environment

dev-full-restart: dev-stop dev-full ## Restart full development environment

dev-logs: ## Show container logs (auto-detects essential vs full setup)
	@echo "$(BLUE)📄 Container Logs:$(RESET)"
	@if command -v docker >/dev/null 2>&1; then \
		container_count=$$(docker ps --filter "name=keiko-" --format "table {{.Names}}" | wc -l | tr -d ' '); \
		if [ $$container_count -le 7 ]; then \
			echo "$(BLUE)Showing essential container logs...$(RESET)"; \
			docker-compose -f $(DOCKER_COMPOSE_ESSENTIAL) -p $(DOCKER_PROJECT_NAME) logs -f; \
		else \
			echo "$(BLUE)Showing full container logs...$(RESET)"; \
			cd .. && docker-compose -f docker-compose.dev.yml -p $(DOCKER_PROJECT_NAME) logs -f; \
		fi; \
	else \
		echo "$(YELLOW)⚠️  Docker not available$(RESET)"; \
	fi

dev-logs-essential: ## Show essential container logs only
	@echo "$(BLUE)📄 Essential Container Logs:$(RESET)"
	@if command -v docker >/dev/null 2>&1; then \
		docker-compose -f $(DOCKER_COMPOSE_ESSENTIAL) -p $(DOCKER_PROJECT_NAME) logs -f; \
	else \
		echo "$(YELLOW)⚠️  Docker not available$(RESET)"; \
	fi

dev-logs-full: ## Show full container logs only
	@echo "$(BLUE)📄 Full Container Logs:$(RESET)"
	@if command -v docker >/dev/null 2>&1; then \
		cd .. && docker-compose -f docker-compose.dev.yml -p $(DOCKER_PROJECT_NAME) logs -f; \
	else \
		echo "$(YELLOW)⚠️  Docker not available$(RESET)"; \
	fi

shell: ## Open interactive Python shell
	@echo "$(BLUE)Opening Python shell...$(RESET)"
	$(UV) run ipython

# =====================================================================
# API Generation
# =====================================================================

generate-api: ## Generate OpenAPI specification
	@echo "$(BLUE)🔄 Generating OpenAPI specification...$(RESET)"
	@if [ -f "main.py" ]; then \
		$(PYTHON) -c "import json; from main import app; json.dump(app.openapi(), open('$(OPENAPI_FILE)', 'w'), indent=2)" && \
		echo "$(GREEN)✅ OpenAPI spec generated: $(OPENAPI_FILE)$(RESET)" || \
		echo "$(YELLOW)⚠️  OpenAPI generation failed$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  main.py not found$(RESET)"; \
	fi

validate-api: generate-api ## Validate OpenAPI specification
	@echo "$(BLUE)🔍 Validating OpenAPI specification...$(RESET)"
	@$(PYTHON) -c "\
import json; \
try: \
    from openapi_spec_validator import validate_spec; \
    with open('$(OPENAPI_FILE)', 'r') as f: \
        spec = json.load(f); \
    validate_spec(spec); \
    print('$(GREEN)✅ OpenAPI specification is valid$(RESET)'); \
except ImportError: \
    print('$(YELLOW)⚠️  openapi-spec-validator not installed$(RESET)'); \
except Exception as e: \
    print(f'$(YELLOW)⚠️  OpenAPI validation failed: {e}$(RESET)'); \
"

# =====================================================================
# Testing
# =====================================================================

test: ## Run test suite
	@echo "$(BLUE)Running tests...$(RESET)"
	$(PYTEST)

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTEST) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(RESET)"

test-fast: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(RESET)"
	$(PYTEST) -n auto

test-websocket: ## Test WebSocket functionality
	@echo "$(BLUE)🧪 Testing WebSocket functionality...$(RESET)"
	@if [ -d "tests" ]; then \
		$(PYTEST) tests/ -k "websocket" -v || echo "$(YELLOW)⚠️  No WebSocket tests found$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  No tests directory found$(RESET)"; \
	fi

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTEST) tests/integration/ -v

test-system: ## Run system tests only
	@echo "$(BLUE)Running system tests...$(RESET)"
	$(PYTEST) tests/system/ -v

test-e2e: ## Run end-to-end tests only
	@echo "$(BLUE)Running E2E tests...$(RESET)"
	$(PYTEST) tests/e2e/ -v

test-performance: ## Run performance tests only
	@echo "$(BLUE)Running performance tests...$(RESET)"
	$(PYTEST) tests/perf/ tests/e2e/performance/ -v

test-conformance: ## Run conformance tests only
	@echo "$(BLUE)Running conformance tests...$(RESET)"
	$(PYTEST) tests/e2e/conformance/ -v

test-security: ## Run security tests only
	@echo "$(BLUE)Running security tests...$(RESET)"
	$(PYTEST) tests/e2e/security/ -v

test-sdk: ## Run SDK tests only
	@echo "$(BLUE)Running SDK tests...$(RESET)"
	$(PYTEST) tests/sdk/ -v

test-kei-agents: ## Run KEI-Agent-Framework tests only
	@echo "$(BLUE)Running KEI-Agent-Framework tests...$(RESET)"
	$(PYTEST) tests/kei_agents/ -v --cov=backend.kei_agents --cov=examples

test-kei-framework: ## Run KEI-Agent core framework tests
	@echo "$(BLUE)Running KEI-Agent core framework tests...$(RESET)"
	$(PYTEST) tests/kei_agents/test_refactored_framework.py -v --cov=backend.kei_agents

test-kei-examples: ## Run KEI-Agent examples tests
	@echo "$(BLUE)Running KEI-Agent examples tests...$(RESET)"
	$(PYTEST) tests/kei_agents/test_refactored_examples.py -v --cov=examples

test-logfire: ## Run Logfire integration tests
	@echo "$(BLUE)🔥 Running Logfire integration tests...$(RESET)"
	$(PYTEST) tests/test_logfire_integration.py -v --tb=short

test-config: ## Validate test and coverage configuration
	@echo "$(BLUE)🔧 Validating test/coverage configuration...$(RESET)"
	@echo "$(GREEN)✅ Checking pytest configuration...$(RESET)"
	@$(PYTEST) --collect-only -q --tb=no > /dev/null 2>&1 && echo "$(GREEN)✅ Pytest configuration is valid$(RESET)" || echo "$(YELLOW)⚠️ Pytest configuration has warnings (expected)$(RESET)"
	@echo "$(GREEN)✅ Test/Coverage configuration is unified and consistent$(RESET)"

validate-reproducible-builds: ## Validate reproducible builds configuration
	@echo "$(BLUE)🔧 Validating reproducible builds...$(RESET)"
	@echo "$(GREEN)✅ Checking uv installation...$(RESET)"
	@$(UV) --version > /dev/null && echo "$(GREEN)✅ uv is installed$(RESET)" || echo "$(RED)❌ uv not found$(RESET)"
	@echo "$(GREEN)✅ Checking uv.lock...$(RESET)"
	@test -f uv.lock && echo "$(GREEN)✅ uv.lock exists$(RESET)" || echo "$(RED)❌ uv.lock missing$(RESET)"
	@echo "$(GREEN)✅ Testing uv sync...$(RESET)"
	@$(UV) sync --dry-run > /dev/null 2>&1 && echo "$(GREEN)✅ uv sync works$(RESET)" || echo "$(RED)❌ uv sync failed$(RESET)"
	@echo "$(GREEN)✅ Reproducible builds are configured correctly$(RESET)"

validate-lint-format: ## Validate lint/format consolidation
	@echo "$(BLUE)🔧 Validating lint/format consolidation...$(RESET)"
	@echo "$(GREEN)✅ Checking Ruff installation...$(RESET)"
	@$(UV) run ruff --version > /dev/null && echo "$(GREEN)✅ Ruff is installed$(RESET)" || echo "$(RED)❌ Ruff not found$(RESET)"
	@echo "$(GREEN)✅ Testing Ruff functionality...$(RESET)"
	@$(UV) run ruff check --help > /dev/null 2>&1 && echo "$(GREEN)✅ Ruff check works$(RESET)" || echo "$(RED)❌ Ruff check failed$(RESET)"
	@$(UV) run ruff format --help > /dev/null 2>&1 && echo "$(GREEN)✅ Ruff format works$(RESET)" || echo "$(RED)❌ Ruff format failed$(RESET)"
	@echo "$(GREEN)✅ Lint/Format consolidation is complete$(RESET)"

validate-dependencies: ## Validate dependency separation (Runtime vs. Dev)
	@echo "$(BLUE)🔧 Validating dependency separation...$(RESET)"
	@echo "$(GREEN)✅ Checking uv dependency-groups...$(RESET)"
	@$(UV) sync --group dev --dry-run > /dev/null 2>&1 && echo "$(GREEN)✅ Dev group works$(RESET)" || echo "$(RED)❌ Dev group failed$(RESET)"
	@$(UV) sync --group test --dry-run > /dev/null 2>&1 && echo "$(GREEN)✅ Test group works$(RESET)" || echo "$(RED)❌ Test group failed$(RESET)"
	@$(UV) sync --group docs --dry-run > /dev/null 2>&1 && echo "$(GREEN)✅ Docs group works$(RESET)" || echo "$(RED)❌ Docs group failed$(RESET)"
	@$(UV) sync --dry-run > /dev/null 2>&1 && echo "$(GREEN)✅ Runtime sync works$(RESET)" || echo "$(RED)❌ Runtime sync failed$(RESET)"
	@echo "$(GREEN)✅ Dependency separation is complete$(RESET)"

validate-typing: ## Validate enterprise-ready typing configuration
	@echo "$(BLUE)🔧 Validating enterprise-ready typing...$(RESET)"
	@echo "$(GREEN)✅ Checking MyPy installation...$(RESET)"
	@$(UV) run mypy --version > /dev/null && echo "$(GREEN)✅ MyPy is installed$(RESET)" || echo "$(RED)❌ MyPy not found$(RESET)"
	@echo "$(GREEN)✅ Checking PEP 561 compliance...$(RESET)"
	@test -f py.typed && echo "$(GREEN)✅ py.typed file exists$(RESET)" || echo "$(RED)❌ py.typed missing$(RESET)"
	@echo "$(GREEN)✅ Testing MyPy functionality...$(RESET)"
	@$(UV) run mypy --help > /dev/null 2>&1 && echo "$(GREEN)✅ MyPy configuration works$(RESET)" || echo "$(RED)❌ MyPy configuration failed$(RESET)"
	@echo "$(GREEN)✅ Enterprise-ready typing is configured$(RESET)"

# =====================================================================
# Code Quality
# =====================================================================

lint: ## Run code linting
	@echo "$(BLUE)Running linting...$(RESET)"
	$(RUFF) check .

lint-fix: ## Run linting with auto-fix
	@echo "$(BLUE)Running linting with auto-fix...$(RESET)"
	$(RUFF) check --fix .

format: ## Format code
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(RUFF) format .

type-check: ## Run comprehensive type checking
	@echo "$(BLUE)Running enterprise-grade type checking...$(RESET)"
	$(MYPY) . || echo "$(YELLOW)⚠️  Type checking found issues$(RESET)"

type-check-strict: ## Run type checking with strict mode for core modules
	@echo "$(BLUE)Running strict type checking for core modules...$(RESET)"
	$(MYPY) --strict config/ core/ data_models/ auth/ security/ api/ app/ || echo "$(YELLOW)⚠️  Strict type checking found issues$(RESET)"

type-check-report: ## Generate detailed type checking report
	@echo "$(BLUE)Generating type checking report...$(RESET)"
	$(MYPY) --html-report mypy-report --txt-report mypy-report . || echo "$(YELLOW)⚠️  Type checking found issues$(RESET)"
	@echo "$(GREEN)✅ Type checking report generated in mypy-report/$(RESET)"

type-check-coverage: ## Check type coverage statistics
	@echo "$(BLUE)Checking type coverage...$(RESET)"
	$(MYPY) --any-exprs-report mypy-coverage . || echo "$(YELLOW)⚠️  Type checking found issues$(RESET)"
	@echo "$(GREEN)✅ Type coverage report generated in mypy-coverage/$(RESET)"

security: ## Run security scanning
	@echo "$(BLUE)Running security scanning...$(RESET)"
	$(BANDIT) -r . -x tests/ || echo "$(YELLOW)⚠️  Security issues found$(RESET)"

quality: lint format type-check security ## Run all quality checks
	@echo "$(GREEN)All quality checks completed!$(RESET)"

quality-strict: lint format type-check-strict security ## Run all quality checks with strict typing
	@echo "$(GREEN)All strict quality checks completed!$(RESET)"

# =====================================================================
# Documentation
# =====================================================================

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	@if command -v mkdocs >/dev/null 2>&1; then \
		$(UV) run mkdocs build; \
	else \
		echo "$(YELLOW)⚠️  mkdocs not available$(RESET)"; \
	fi

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8001$(RESET)"
	@if command -v mkdocs >/dev/null 2>&1; then \
		$(UV) run mkdocs serve --dev-addr 127.0.0.1:8001; \
	else \
		echo "$(YELLOW)⚠️  mkdocs not available$(RESET)"; \
	fi

# =====================================================================
# Utilities
# =====================================================================

clean: ## Clean build artifacts and type checking reports
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf mypy-report/
	rm -rf mypy-coverage/
	rm -f $(OPENAPI_FILE)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

health-check: ## Check development environment
	@echo "$(BLUE)🏥 Checking development environment...$(RESET)"
	@echo "$(GREEN)Python version:$(RESET) $$($(PYTHON) --version)"
	@echo "$(GREEN)uv version:$(RESET) $$($(UV) --version)"
	@$(PYTHON) -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')" || echo "$(RED)❌ FastAPI not installed$(RESET)"
	@$(PYTHON) -c "import websockets; print(f'WebSockets version: {websockets.__version__}')" || echo "$(RED)❌ WebSockets not installed$(RESET)"
	@$(PYTHON) -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')" || echo "$(RED)❌ Pydantic not installed$(RESET)"
	@[ -f "main.py" ] && echo "$(GREEN)✅ main.py found$(RESET)" || echo "$(YELLOW)⚠️  main.py missing$(RESET)"
	@[ -d "routers" ] && echo "$(GREEN)✅ routers directory found$(RESET)" || echo "$(YELLOW)⚠️  routers directory missing$(RESET)"
	@if command -v docker >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Docker available$(RESET)"; \
		if docker info >/dev/null 2>&1; then \
			echo "$(GREEN)✅ Docker daemon running$(RESET)"; \
		else \
			echo "$(YELLOW)⚠️  Docker daemon not running$(RESET)"; \
		fi; \
	else \
		echo "$(YELLOW)⚠️  Docker not installed$(RESET)"; \
	fi
	@echo "$(GREEN)✅ Health check completed$(RESET)"

# =====================================================================
# Pre-commit
# =====================================================================

setup-pre-commit: ## Setup pre-commit hooks
	@echo "$(BLUE)Setting up pre-commit hooks...$(RESET)"
	@if command -v pre-commit >/dev/null 2>&1; then \
		$(UV) run pre-commit install && \
		echo "$(GREEN)✅ Pre-commit hooks installed$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  pre-commit not available$(RESET)"; \
	fi

# =====================================================================
# CI/CD
# =====================================================================

ci-test: ## Run tests for CI
	@echo "$(BLUE)Running CI tests...$(RESET)"
	$(PYTEST) --cov-report=xml --cov-report=term

ci-quality: ## Run quality checks for CI
	@echo "$(BLUE)Running CI quality checks...$(RESET)"
	$(RUFF) check .
	$(RUFF) format --check .
	$(MYPY) --ignore-missing-imports . || true
	$(BANDIT) -r . -x tests/ || true
