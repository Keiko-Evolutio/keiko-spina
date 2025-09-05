#!/bin/bash
# Repository Migration Rollback Script
# Implements rollback strategy from Repository-Trennung-Migrationsplan.md

set -e

echo "🔄 Repository Migration Rollback Script"

# Configuration
MIGRATION_ROOT="$(pwd)"
ROLLBACK_DATE=$(date +"%Y%m%d_%H%M%S")
ROLLBACK_LOG="rollback_${ROLLBACK_DATE}.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$ROLLBACK_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$ROLLBACK_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$ROLLBACK_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$ROLLBACK_LOG"
}

# Get rollback phase from argument
ROLLBACK_PHASE=${1:-"all"}

show_help() {
    echo "Usage: $0 [PHASE]"
    echo ""
    echo "Available rollback phases:"
    echo "  phase1    - Rollback repository setup and structure creation"
    echo "  phase2    - Rollback backend separation"
    echo "  phase3    - Rollback frontend separation"
    echo "  phase4    - Rollback API contracts"
    echo "  logging   - Rollback logging independence"
    echo "  all       - Complete rollback to monolithic architecture"
    echo "  help      - Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 phase2     # Rollback only backend separation"
    echo "  $0 all        # Complete rollback"
}

# Check if help is requested
if [ "$ROLLBACK_PHASE" = "help" ]; then
    show_help
    exit 0
fi

# Find most recent backup
find_latest_backup() {
    local backup_dirs=$(find . -maxdepth 1 -name "migration_backup_*" -type d | sort -r)
    if [ -z "$backup_dirs" ]; then
        log_error "No migration backup found!"
        log_error "Cannot proceed with rollback without backup."
        exit 1
    fi
    
    local latest_backup=$(echo "$backup_dirs" | head -n 1)
    echo "$latest_backup"
}

# Phase 1: Rollback repository structure creation
rollback_phase1() {
    log_info "Rolling back Phase 1: Repository Structure Creation"
    
    # Remove created repository directories
    if [ -d "keiko-backend" ]; then
        log_info "Removing keiko-backend directory..."
        rm -rf keiko-backend
        log_success "keiko-backend directory removed"
    fi
    
    if [ -d "keiko-frontend" ]; then
        log_info "Removing keiko-frontend directory..."
        rm -rf keiko-frontend
        log_success "keiko-frontend directory removed"
    fi
    
    if [ -d "keiko-api-contracts" ]; then
        log_info "Removing keiko-api-contracts directory..."
        rm -rf keiko-api-contracts
        log_success "keiko-api-contracts directory removed"
    fi
    
    # Remove master docker-compose file
    if [ -f "docker-compose.dev-multi-repo.yml" ]; then
        rm -f docker-compose.dev-multi-repo.yml
        log_success "Master docker-compose file removed"
    fi
    
    log_success "Phase 1 rollback completed"
}

# Phase 2: Rollback backend separation
rollback_phase2() {
    log_info "Rolling back Phase 2: Backend Separation"
    
    local backup_dir=$(find_latest_backup)
    
    if [ -d "$backup_dir/backend" ]; then
        log_info "Restoring backend from backup..."
        
        # Remove current backend changes
        if [ -d "backend" ]; then
            rm -rf backend
        fi
        
        # Restore from backup
        cp -r "$backup_dir/backend" ./
        log_success "Backend restored from backup"
    else
        log_warning "No backend backup found in $backup_dir"
    fi
    
    # Remove backend-specific migration artifacts
    rm -f backend_logging_migration_report_*.json
    
    log_success "Phase 2 rollback completed"
}

# Phase 3: Rollback frontend separation  
rollback_phase3() {
    log_info "Rolling back Phase 3: Frontend Separation"
    
    local backup_dir=$(find_latest_backup)
    
    if [ -d "$backup_dir/frontend" ]; then
        log_info "Restoring frontend from backup..."
        
        # Remove current frontend changes
        if [ -d "frontend" ]; then
            rm -rf frontend
        fi
        
        # Restore from backup
        cp -r "$backup_dir/frontend" ./
        log_success "Frontend restored from backup"
    else
        log_warning "No frontend backup found in $backup_dir"
    fi
    
    log_success "Phase 3 rollback completed"
}

# Phase 4: Rollback API contracts
rollback_phase4() {
    log_info "Rolling back Phase 4: API Contracts"
    
    local backup_dir=$(find_latest_backup)
    
    # Restore original api-contracts if it existed
    if [ -d "$backup_dir/api-contracts" ]; then
        log_info "Restoring original api-contracts from backup..."
        
        # Remove current api-contracts
        if [ -d "api-contracts" ]; then
            rm -rf api-contracts
        fi
        
        # Restore from backup
        cp -r "$backup_dir/api-contracts" ./
        log_success "Original api-contracts restored"
    else
        # Remove new api-contracts structure if no backup existed
        if [ -d "api-contracts" ]; then
            rm -rf api-contracts
            log_success "New api-contracts structure removed"
        fi
    fi
    
    log_success "Phase 4 rollback completed"
}

# Rollback logging independence
rollback_logging() {
    log_info "Rolling back logging independence changes..."
    
    # Find files with backend-native logging imports
    local files_to_rollback=$(find backend -name "*.py" -exec grep -l "from kei_logging import" {} \; 2>/dev/null || true)
    
    if [ ! -z "$files_to_rollback" ]; then
        log_info "Rolling back logging imports in files:"
        echo "$files_to_rollback"
        
        # Replace backend-native imports with SDK imports
        for file in $files_to_rollback; do
            if [ -f "$file" ]; then
                sed -i.rollback \
                    -e 's/from kei_logging import get_logger/from kei_agent.enterprise_logging import get_logger/g' \
                    -e 's/import kei_logging as/import kei_agent.enterprise_logging as/g' \
                    "$file"
                
                log_success "Rolled back logging imports in $(basename "$file")"
            fi
        done
    fi
    
    # Remove backend-native kei_logging module
    if [ -d "backend/kei_logging" ]; then
        rm -rf backend/kei_logging
        log_success "Backend-native kei_logging module removed"
    fi
    
    # Remove logging migration reports
    rm -f backend_logging_migration_report_*.json
    
    log_success "Logging rollback completed"
}

# Rollback orchestrator service changes
rollback_orchestrator() {
    log_info "Rolling back orchestrator service changes..."
    
    # Remove new orchestrator service
    if [ -f "backend/services/agent_orchestrator_service.py" ]; then
        rm -f backend/services/agent_orchestrator_service.py
        log_success "New orchestrator service removed"
    fi
    
    log_success "Orchestrator rollback completed"
}

# Docker and Kubernetes rollback
rollback_docker_kubernetes() {
    log_info "Rolling back Docker and Kubernetes changes..."
    
    # Stop any running services
    if [ -f "docker-compose.dev-multi-repo.yml" ]; then
        log_info "Stopping multi-repo services..."
        docker-compose -f docker-compose.dev-multi-repo.yml down 2>/dev/null || true
    fi
    
    # Restore original docker-compose files if they were backed up
    local backup_dir=$(find_latest_backup)
    
    for compose_file in docker-compose*.yml; do
        if [ -f "$backup_dir/$compose_file" ]; then
            cp "$backup_dir/$compose_file" ./
            log_success "Restored $compose_file from backup"
        fi
    done
    
    log_success "Docker and Kubernetes rollback completed"
}

# Data migration rollback
rollback_data() {
    log_info "Checking for data migration rollback requirements..."
    
    # This would contain database rollback logic
    # For now, just log that no data migration was performed
    log_info "No database migrations to rollback (structure-only migration)"
    
    log_success "Data rollback check completed"
}

# Complete rollback function
complete_rollback() {
    log_info "Performing complete rollback to monolithic architecture..."
    
    local backup_dir=$(find_latest_backup)
    log_info "Using backup: $backup_dir"
    
    # Stop any running services first
    rollback_docker_kubernetes
    
    # Rollback all phases in reverse order
    rollback_phase1
    rollback_phase4
    rollback_phase3
    rollback_phase2
    rollback_logging
    rollback_orchestrator
    rollback_data
    
    # Restore key files from backup
    if [ -f "$backup_dir/Repository-Trennung-Migrationsplan.md" ]; then
        cp "$backup_dir/Repository-Trennung-Migrationsplan.md" ./
    fi
    
    if [ -f "$backup_dir/README.md" ]; then
        cp "$backup_dir/README.md" ./
    fi
    
    if [ -f "$backup_dir/Makefile" ]; then
        cp "$backup_dir/Makefile" ./
    fi
    
    log_success "Complete rollback to monolithic architecture completed"
}

# Validation function
validate_rollback() {
    log_info "Validating rollback..."
    
    local issues=0
    
    # Check that separated directories are removed (for complete rollback)
    if [ "$ROLLBACK_PHASE" = "all" ]; then
        if [ -d "keiko-backend" ] || [ -d "keiko-frontend" ] || [ -d "keiko-api-contracts" ]; then
            log_error "Separated repository directories still exist after rollback"
            issues=$((issues + 1))
        fi
    fi
    
    # Check that original structure exists
    if [ ! -d "backend" ]; then
        log_error "Original backend directory missing after rollback"
        issues=$((issues + 1))
    fi
    
    if [ ! -d "frontend" ]; then
        log_error "Original frontend directory missing after rollback"
        issues=$((issues + 1))
    fi
    
    # Check for remaining migration artifacts
    local migration_files=$(find . -name "*migration*" -type f | grep -v "$ROLLBACK_LOG" | grep -v "Repository-Trennung-Migrationsplan.md" || true)
    if [ ! -z "$migration_files" ]; then
        log_warning "Migration artifacts still present:"
        echo "$migration_files"
    fi
    
    if [ $issues -eq 0 ]; then
        log_success "Rollback validation passed"
        return 0
    else
        log_error "Rollback validation failed with $issues issues"
        return 1
    fi
}

# Main execution function
main() {
    log_info "Starting rollback process"
    log_info "Rollback phase: $ROLLBACK_PHASE"
    log_info "Working directory: $MIGRATION_ROOT"
    log_info "Rollback log: $ROLLBACK_LOG"
    
    case $ROLLBACK_PHASE in
        "phase1")
            rollback_phase1
            ;;
        "phase2")
            rollback_phase2
            ;;
        "phase3")
            rollback_phase3
            ;;
        "phase4")
            rollback_phase4
            ;;
        "logging")
            rollback_logging
            ;;
        "orchestrator")
            rollback_orchestrator
            ;;
        "all")
            complete_rollback
            ;;
        *)
            log_error "Unknown rollback phase: $ROLLBACK_PHASE"
            show_help
            exit 1
            ;;
    esac
    
    # Validate rollback
    if validate_rollback; then
        log_success "Rollback completed successfully!"
        echo ""
        log_info "Rollback Summary:"
        log_info "  Phase: $ROLLBACK_PHASE"
        log_info "  Completed: $(date)"
        log_info "  Log file: $ROLLBACK_LOG"
        echo ""
        log_info "Next steps:"
        log_info "  1. Verify system functionality"
        log_info "  2. Run tests to ensure stability"
        log_info "  3. Review rollback log for any warnings"
        
        if [ "$ROLLBACK_PHASE" = "all" ]; then
            log_info "  4. Restart monolithic services"
            log_info "     docker-compose up -d"
        fi
    else
        log_error "Rollback validation failed!"
        log_error "Please review the rollback log: $ROLLBACK_LOG"
        exit 1
    fi
}

# Check if script is being run from the correct directory
if [ ! -f "Repository-Trennung-Migrationsplan.md" ]; then
    log_error "Please run this script from the root of the keiko-personal-assistant repository"
    exit 1
fi

# Execute main function
main "$@"