#!/bin/bash
# Repository Migration Script
# Implements the repository separation plan from Repository-Trennung-Migrationsplan.md

set -e

echo "🚀 Starting Repository Migration..."

# Configuration
MIGRATION_ROOT="$(pwd)"
MIGRATION_DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="${MIGRATION_ROOT}/migration_backup_${MIGRATION_DATE}"

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

# Create backup of current state
create_backup() {
    log_info "Creating backup of current repository state..."
    mkdir -p "$BACKUP_DIR"
    
    # Backup critical directories
    cp -r backend "$BACKUP_DIR/"
    cp -r frontend "$BACKUP_DIR/"
    cp -r kei-agent-py-sdk "$BACKUP_DIR/" 2>/dev/null || log_warning "kei-agent-py-sdk not found, skipping backup"
    cp -r api-contracts "$BACKUP_DIR/" 2>/dev/null || log_warning "api-contracts not found, skipping backup"
    
    # Backup important files
    cp Repository-Trennung-Migrationsplan.md "$BACKUP_DIR/"
    cp README.md "$BACKUP_DIR/"
    cp docker-compose*.yml "$BACKUP_DIR/" 2>/dev/null || true
    cp Makefile "$BACKUP_DIR/" 2>/dev/null || true
    
    log_success "Backup created at: $BACKUP_DIR"
}

# Phase 1: Repository Structure Creation
create_repository_structures() {
    log_info "Phase 1: Creating repository structures..."
    
    # Create keiko-backend repository structure
    log_info "Creating keiko-backend repository structure..."
    mkdir -p keiko-backend/src/{api,agents,auth,core,data_models,grpc_services}
    mkdir -p keiko-backend/src/{kei_logging,messaging,middleware,monitoring}
    mkdir -p keiko-backend/src/{security,services,storage,task_management,utils}
    mkdir -p keiko-backend/src/{voice,workflows}
    mkdir -p keiko-backend/{tests,docs,infrastructure,scripts,api-contracts,kubernetes}
    
    # Create keiko-frontend repository structure
    log_info "Creating keiko-frontend repository structure..."
    mkdir -p keiko-frontend/src/{components,pages,hooks,services,stores,types,utils,assets}
    mkdir -p keiko-frontend/{public,tests,docs,api-contracts,docker}
    
    # Create keiko-api-contracts repository structure
    log_info "Creating keiko-api-contracts repository structure..."
    mkdir -p keiko-api-contracts/{openapi,asyncapi,protobuf,schemas}
    
    log_success "Repository structures created"
}

# Phase 2: Backend Migration
migrate_backend() {
    log_info "Phase 2: Migrating backend code..."
    
    if [ ! -d "backend" ]; then
        log_error "Backend directory not found!"
        return 1
    fi
    
    # Copy backend directories with rsync (excluding problematic files)
    log_info "Copying backend source code..."
    
    # Core directories
    rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' backend/api/ keiko-backend/src/api/
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/agents/ keiko-backend/src/agents/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/auth/ keiko-backend/src/auth/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/core/ keiko-backend/src/core/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/data_models/ keiko-backend/src/data_models/
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/grpc_services/ keiko-backend/src/grpc_services/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/kei_logging/ keiko-backend/src/kei_logging/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/messaging/ keiko-backend/src/messaging/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/middleware/ keiko-backend/src/middleware/
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/monitoring/ keiko-backend/src/monitoring/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/security/ keiko-backend/src/security/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/services/ keiko-backend/src/services/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/storage/ keiko-backend/src/storage/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/task_management/ keiko-backend/src/task_management/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/utils/ keiko-backend/src/utils/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/voice/ keiko-backend/src/voice/ 2>/dev/null || true
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/workflows/ keiko-backend/src/workflows/ 2>/dev/null || true
    
    # Copy tests and other directories
    rsync -av --exclude='__pycache__' --exclude='*.pyc' backend/tests/ keiko-backend/tests/ 2>/dev/null || true
    
    # Copy configuration files
    cp backend/pyproject.toml keiko-backend/ 2>/dev/null || log_warning "pyproject.toml not found in backend"
    cp backend/.env.example keiko-backend/ 2>/dev/null || true
    
    log_success "Backend code migration completed"
}

# Phase 3: Frontend Migration
migrate_frontend() {
    log_info "Phase 3: Migrating frontend code..."
    
    if [ ! -d "frontend" ]; then
        log_error "Frontend directory not found!"
        return 1
    fi
    
    # Copy frontend source code
    log_info "Copying frontend source code..."
    rsync -av --exclude='node_modules' --exclude='.next' --exclude='dist' --exclude='build' frontend/src/ keiko-frontend/src/ 2>/dev/null || true
    rsync -av frontend/public/ keiko-frontend/public/ 2>/dev/null || true
    
    # Copy configuration files
    cp frontend/package.json keiko-frontend/ 2>/dev/null || log_warning "package.json not found in frontend"
    cp frontend/package-lock.json keiko-frontend/ 2>/dev/null || true
    cp frontend/yarn.lock keiko-frontend/ 2>/dev/null || true
    cp frontend/vite.config.ts keiko-frontend/ 2>/dev/null || true
    cp frontend/tsconfig.json keiko-frontend/ 2>/dev/null || true
    cp frontend/tailwind.config.js keiko-frontend/ 2>/dev/null || true
    cp frontend/.env.example keiko-frontend/ 2>/dev/null || true
    
    log_success "Frontend code migration completed"
}

# Phase 4: API Contracts Migration
migrate_api_contracts() {
    log_info "Phase 4: Migrating API contracts..."
    
    if [ -d "api-contracts" ]; then
        rsync -av api-contracts/ keiko-api-contracts/
        log_success "Existing API contracts migrated"
    else
        log_info "No existing API contracts found, creating basic structure..."
        # Create basic API contract templates - these will be created by other scripts
        touch keiko-api-contracts/openapi/backend-frontend-api-v1.yaml
        touch keiko-api-contracts/asyncapi/backend-frontend-events-v1.yaml
        touch keiko-api-contracts/protobuf/agent_service.proto
        touch keiko-api-contracts/versions.yaml
    fi
}

# Phase 5: Cross-Dependency Analysis and Reporting
analyze_cross_dependencies() {
    log_info "Phase 5: Analyzing cross-dependencies..."
    
    # Search for problematic imports
    log_info "Searching for kei_agent imports in backend..."
    AGENT_IMPORTS=$(find keiko-backend -name "*.py" -exec grep -l "from kei_agent" {} \; 2>/dev/null || true)
    
    if [ ! -z "$AGENT_IMPORTS" ]; then
        log_warning "Found kei_agent imports in backend (these need to be resolved):"
        echo "$AGENT_IMPORTS"
        echo "$AGENT_IMPORTS" > migration_issues_${MIGRATION_DATE}.log
    fi
    
    # Search for kei_agents orchestrator imports
    log_info "Searching for kei_agents.orchestrator imports in backend..."
    ORCHESTRATOR_IMPORTS=$(find keiko-backend -name "*.py" -exec grep -l "from kei_agents.orchestrator" {} \; 2>/dev/null || true)
    
    if [ ! -z "$ORCHESTRATOR_IMPORTS" ]; then
        log_warning "Found kei_agents.orchestrator imports in backend (these need to be resolved):"
        echo "$ORCHESTRATOR_IMPORTS"
        echo "$ORCHESTRATOR_IMPORTS" >> migration_issues_${MIGRATION_DATE}.log
    fi
    
    log_success "Cross-dependency analysis completed"
}

# Phase 6: Create Docker configurations
create_docker_configs() {
    log_info "Phase 6: Creating Docker configurations..."
    
    # Backend Dockerfile
    cat > keiko-backend/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy source code
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
    
    # Backend docker-compose.yml
    cat > keiko-backend/docker-compose.yml << 'EOF'
version: '3.9'
services:
  keiko-backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://keiko:keiko@postgres:5432/keiko
      - REDIS_URL=redis://redis:6379
      - NATS_URL=nats://nats:4222
    depends_on:
      - postgres
      - redis
      - nats
    volumes:
      - ./src:/app/src
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: keiko
      POSTGRES_USER: keiko
      POSTGRES_PASSWORD: keiko
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nats:
    image: nats:2.10-alpine
    ports:
      - "4222:4222"
      - "8222:8222"
    command: ["--jetstream"]
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
EOF
    
    # Frontend Dockerfile
    cat > keiko-frontend/Dockerfile << 'EOF'
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
EOF
    
    # Frontend docker-compose.yml
    cat > keiko-frontend/docker-compose.yml << 'EOF'
version: '3.9'
services:
  keiko-frontend:
    build: .
    ports:
      - "3000:3000"
    environment:
      - VITE_API_BASE_URL=http://localhost:8000
      - VITE_WS_URL=ws://localhost:8000/ws
    restart: unless-stopped
EOF
    
    log_success "Docker configurations created"
}

# Phase 7: Create master development docker-compose
create_master_docker_compose() {
    log_info "Phase 7: Creating master development docker-compose..."
    
    cat > docker-compose.dev-multi-repo.yml << 'EOF'
version: '3.9'
services:
  # Infrastructure
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: keiko_dev
      POSTGRES_USER: keiko
      POSTGRES_PASSWORD: keiko
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nats:
    image: nats:2.10-alpine
    ports:
      - "4222:4222"
      - "8222:8222"
    command: ["--jetstream"]
    restart: unless-stopped

  # Backend Service
  keiko-backend:
    build: ./keiko-backend
    ports:
      - "8000:8000"
    volumes:
      - ./keiko-backend/src:/app/src
    environment:
      - DATABASE_URL=postgresql://keiko:keiko@postgres:5432/keiko_dev
      - REDIS_URL=redis://redis:6379
      - NATS_URL=nats://nats:4222
      - DEV_MODE=true
    depends_on:
      - postgres
      - redis
      - nats
    restart: unless-stopped

  # Frontend Application
  keiko-frontend:
    build: ./keiko-frontend
    ports:
      - "3000:3000"
    volumes:
      - ./keiko-frontend/src:/app/src
    environment:
      - VITE_API_BASE_URL=http://localhost:8000
      - VITE_WS_URL=ws://localhost:8000/ws
      - NODE_ENV=development
    depends_on:
      - keiko-backend
    restart: unless-stopped

  # SDK Development Environment
  kei-agent-sdk-dev:
    build: ./kei-agent-py-sdk
    volumes:
      - ./kei-agent-py-sdk:/app
    environment:
      - KEI_BACKEND_URL=http://keiko-backend:8000
      - KEI_GRPC_URL=keiko-backend:9000
    depends_on:
      - keiko-backend
    command: ["tail", "-f", "/dev/null"]  # Keep container running for development

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    name: keiko-dev-network
EOF
    
    log_success "Master development docker-compose created"
}

# Phase 8: Create README files for each repository
create_readme_files() {
    log_info "Phase 8: Creating README files..."
    
    # Backend README
    cat > keiko-backend/README.md << 'EOF'
# Keiko Backend Services

This is the backend platform for the Keiko Personal Assistant, providing REST/gRPC APIs, agent management, and core business services.

## Architecture

- **Microservices-based Backend Platform**
- **REST/gRPC APIs** for Frontend/SDK Integration
- **Event-driven Architecture** with NATS JetStream
- **API-first Communication** (no cross-imports)

## Quick Start

```bash
# Start development environment
docker-compose up -d

# Install dependencies
pip install -e .

# Run development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

- **OpenAPI Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **gRPC Services**: See `api-contracts/protobuf/`

## Services

- `src/api/` - REST API endpoints
- `src/agents/` - Agent management
- `src/auth/` - Authentication/Authorization
- `src/core/` - Core business logic
- `src/grpc_services/` - gRPC services
- `src/messaging/` - Event handling (NATS)
- `src/security/` - Security components
- `src/storage/` - Database layer
- `src/voice/` - Voice processing

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

## Deployment

```bash
# Build image
docker build -t keiko/backend .

# Deploy with kubernetes
kubectl apply -f kubernetes/
```
EOF
    
    # Frontend README
    cat > keiko-frontend/README.md << 'EOF'
# Keiko Frontend Application

React-based frontend application for the Keiko Personal Assistant platform.

## Architecture

- **React 18** with TypeScript
- **Vite** Build System
- **Auto-generated API Clients** from OpenAPI specs
- **Zustand** for state management
- **API-first Integration** with backend

## Quick Start

```bash
# Install dependencies
npm install

# Generate API clients from contracts
npm run generate:api-clients

# Start development server
npm run dev
```

## Development

### API Client Generation

```bash
# Generate clients from API contracts
npm run generate:api-clients
```

This will generate:
- REST API clients in `src/api/generated/`
- WebSocket event types in `src/api/events/`
- TypeScript types in `src/types/api.ts`

### Project Structure

- `src/components/` - React components
- `src/pages/` - Page components
- `src/hooks/` - Custom React hooks
- `src/services/` - API services
- `src/stores/` - State management (Zustand)
- `src/types/` - TypeScript type definitions

## Building

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Testing

```bash
# Run unit tests
npm test

# Run with watch mode
npm run test:watch
```
EOF
    
    # API Contracts README
    cat > keiko-api-contracts/README.md << 'EOF'
# Keiko API Contracts

Shared API contracts and specifications for the Keiko Personal Assistant platform.

## Structure

- `openapi/` - REST API specifications
- `asyncapi/` - Real-time event specifications
- `protobuf/` - gRPC protocol definitions
- `schemas/` - JSON Schema definitions
- `versions.yaml` - Version management

## Usage

### Backend Integration

The backend automatically validates requests/responses against these contracts.

### Frontend Client Generation

```bash
# In keiko-frontend/
npm run generate:api-clients
```

### SDK Integration

```bash
# In kei-agent-py-sdk/
python scripts/generate_grpc_clients.py
```

## Versioning

All API contracts follow semantic versioning:
- Major version: Breaking changes
- Minor version: New features (backward compatible)
- Patch version: Bug fixes

See `versions.yaml` for current versions.

## Development Workflow

1. **Define API Contract** in appropriate directory
2. **Validate** contract syntax
3. **Generate Clients** for frontend/SDK
4. **Implement** in backend
5. **Test** integration
EOF
    
    log_success "README files created"
}

# Main execution function
main() {
    log_info "Repository Migration Script Started"
    log_info "Migration Date: $MIGRATION_DATE"
    log_info "Working Directory: $MIGRATION_ROOT"
    
    # Execute migration phases
    create_backup
    create_repository_structures
    migrate_backend
    migrate_frontend
    migrate_api_contracts
    analyze_cross_dependencies
    create_docker_configs
    create_master_docker_compose
    create_readme_files
    
    # Final summary
    log_success "Repository migration completed!"
    echo ""
    log_info "Created repositories:"
    log_info "  - keiko-backend/     (Backend services)"
    log_info "  - keiko-frontend/    (Frontend application)"
    log_info "  - keiko-api-contracts/ (API specifications)"
    echo ""
    log_info "Development:"
    log_info "  - Start all services: docker-compose -f docker-compose.dev-multi-repo.yml up -d"
    log_info "  - Backend API: http://localhost:8000"
    log_info "  - Frontend: http://localhost:3000"
    echo ""
    log_warning "Next steps:"
    log_warning "  1. Review migration_issues_${MIGRATION_DATE}.log for dependency issues"
    log_warning "  2. Run dependency resolution scripts"
    log_warning "  3. Implement API contracts"
    log_warning "  4. Test integration between services"
    echo ""
    log_info "Backup location: $BACKUP_DIR"
}

# Check if script is being run from the correct directory
if [ ! -f "Repository-Trennung-Migrationsplan.md" ]; then
    log_error "Please run this script from the root of the keiko-personal-assistant repository"
    exit 1
fi

# Execute main function
main "$@"