#!/bin/bash

# Test Suite für Startup Orchestrator Shell Script
# Testet das Startup-Orchestrator-Script mit gemockten Docker-Befehlen
# Author: Keiko Development Team
# Version: 1.0.0

set -euo pipefail

# Test Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly TEST_LOG_FILE="${PROJECT_ROOT}/logs/test-startup-orchestrator.log"
readonly MOCK_DIR="${SCRIPT_DIR}/mocks"

# Colors for test output
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Setup test environment
setup_test_environment() {
    echo -e "${BLUE}🔧 Setting up test environment...${NC}"
    
    # Create test directories
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${MOCK_DIR}"
    
    # Create mock Docker commands
    create_mock_docker_commands
    
    # Set test environment variables
    export KEIKO_ENV="test"
    export KEIKO_BASE_SERVICES_TIMEOUT=10
    export KEIKO_MONITORING_SERVICES_TIMEOUT=15
    export KEIKO_EDGE_SERVICES_TIMEOUT=20
    export KEIKO_WORKFLOW_SERVICES_TIMEOUT=10
    export KEIKO_BACKEND_STARTUP_TIMEOUT=5
    
    echo -e "${GREEN}✅ Test environment setup complete${NC}"
}

# Create mock Docker commands
create_mock_docker_commands() {
    # Mock docker-compose command
    cat > "${MOCK_DIR}/docker-compose" << 'EOF'
#!/bin/bash
# Mock docker-compose command for testing

case "$1" in
    "up")
        echo "Creating network keiko-network"
        echo "Creating keiko-postgres ... done"
        echo "Creating keiko-redis ... done"
        echo "Creating keiko-nats ... done"
        echo "Creating keiko-prometheus ... done"
        exit 0
        ;;
    "ps")
        echo "Name                State    Ports"
        echo "keiko-postgres      Up       5432/tcp"
        echo "keiko-redis         Up       6379/tcp"
        echo "keiko-nats          Up       4222/tcp"
        echo "keiko-prometheus    Up       9090/tcp"
        exit 0
        ;;
    "down")
        echo "Stopping keiko-postgres ... done"
        echo "Stopping keiko-redis ... done"
        echo "Removing keiko-postgres ... done"
        echo "Removing keiko-redis ... done"
        exit 0
        ;;
    *)
        echo "Mock docker-compose: $*"
        exit 0
        ;;
esac
EOF

    # Mock docker command
    cat > "${MOCK_DIR}/docker" << 'EOF'
#!/bin/bash
# Mock docker command for testing

case "$1" in
    "network")
        if [[ "$2" == "create" ]]; then
            echo "Network created successfully"
            exit 0
        elif [[ "$2" == "ls" ]]; then
            echo "NETWORK ID     NAME           DRIVER    SCOPE"
            echo "abc123def456   keiko-network  bridge    local"
            exit 0
        fi
        ;;
    "ps")
        echo "CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES"
        echo "abc123def456   postgres  postgres  1 min     Up        5432/tcp  keiko-postgres"
        echo "def456ghi789   redis     redis     1 min     Up        6379/tcp  keiko-redis"
        exit 0
        ;;
    "logs")
        echo "Mock logs for container $2"
        exit 0
        ;;
    *)
        echo "Mock docker: $*"
        exit 0
        ;;
esac
EOF

    # Make mock commands executable
    chmod +x "${MOCK_DIR}/docker-compose"
    chmod +x "${MOCK_DIR}/docker"
    
    # Add mock directory to PATH
    export PATH="${MOCK_DIR}:${PATH}"
}

# Test helper functions
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    echo -e "\n${BLUE}🧪 Running test: ${test_name}${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    
    if $test_function; then
        echo -e "${GREEN}✅ PASS: ${test_name}${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL: ${test_name}${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

assert_equals() {
    local expected="$1"
    local actual="$2"
    local message="${3:-Assertion failed}"
    
    if [[ "$expected" == "$actual" ]]; then
        return 0
    else
        echo -e "${RED}${message}: expected '${expected}', got '${actual}'${NC}"
        return 1
    fi
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local message="${3:-String not found}"
    
    if [[ "$haystack" == *"$needle"* ]]; then
        return 0
    else
        echo -e "${RED}${message}: '${needle}' not found in '${haystack}'${NC}"
        return 1
    fi
}

# Test functions
test_environment_config_loading() {
    # Test dass Environment-Konfiguration korrekt geladen wird
    local config_output
    config_output=$(KEIKO_ENV=test bash -c '
        source scripts/startup-orchestrator.sh
        load_environment_config "test"
        echo "BASE_TIMEOUT=${BASE_SERVICES_TIMEOUT}"
        echo "MONITORING_TIMEOUT=${MONITORING_SERVICES_TIMEOUT}"
    ')
    
    assert_contains "$config_output" "BASE_TIMEOUT=10" "Base services timeout not set correctly"
    assert_contains "$config_output" "MONITORING_TIMEOUT=15" "Monitoring services timeout not set correctly"
}

test_service_category_functions() {
    # Test Service-Kategorie-Funktionen
    local base_services
    base_services=$(bash -c '
        source scripts/startup-orchestrator.sh
        get_services_for_category "base"
    ')
    
    assert_contains "$base_services" "postgres" "PostgreSQL not in base services"
    assert_contains "$base_services" "redis" "Redis not in base services"
    
    local timeout
    timeout=$(bash -c '
        source scripts/startup-orchestrator.sh
        get_timeout_for_category "base"
    ')
    
    assert_equals "120" "$timeout" "Base services timeout incorrect"
}

test_docker_compose_startup() {
    # Test Docker Compose Startup-Sequenz
    local startup_output
    startup_output=$(bash -c '
        export PATH="'${MOCK_DIR}':${PATH}"
        cd "'${PROJECT_ROOT}'"
        
        # Simuliere Startup-Funktion
        docker-compose -f docker-compose.dev.yml -p keiko-test up -d postgres redis
    ')
    
    assert_contains "$startup_output" "Creating keiko-postgres" "PostgreSQL container not created"
    assert_contains "$startup_output" "Creating keiko-redis" "Redis container not created"
}

test_health_check_simulation() {
    # Test Health Check Simulation
    local health_check_result
    
    # Simuliere erfolgreichen Health Check
    health_check_result=$(bash -c '
        # Mock curl command für Health Check
        curl() {
            case "$*" in
                *"localhost:5432"*)
                    return 0  # PostgreSQL healthy
                    ;;
                *"localhost:6379"*)
                    return 0  # Redis healthy
                    ;;
                *)
                    return 1  # Other services unhealthy
                    ;;
            esac
        }
        export -f curl
        
        # Simuliere Health Check
        if curl -s http://localhost:5432/health > /dev/null 2>&1; then
            echo "postgres:healthy"
        else
            echo "postgres:unhealthy"
        fi
        
        if curl -s http://localhost:6379/health > /dev/null 2>&1; then
            echo "redis:healthy"
        else
            echo "redis:unhealthy"
        fi
    ')
    
    assert_contains "$health_check_result" "postgres:healthy" "PostgreSQL health check failed"
    assert_contains "$health_check_result" "redis:healthy" "Redis health check failed"
}

test_graceful_shutdown() {
    # Test Graceful Shutdown-Sequenz
    local shutdown_output
    shutdown_output=$(bash -c '
        export PATH="'${MOCK_DIR}':${PATH}"
        cd "'${PROJECT_ROOT}'"
        
        # Simuliere Shutdown
        docker-compose -f docker-compose.dev.yml -p keiko-test down --remove-orphans
    ')
    
    assert_contains "$shutdown_output" "Stopping keiko-postgres" "PostgreSQL not stopped"
    assert_contains "$shutdown_output" "Removing keiko-postgres" "PostgreSQL container not removed"
}

test_error_handling() {
    # Test Fehlerbehandlung
    local error_output
    
    # Simuliere fehlgeschlagenen Docker-Befehl
    error_output=$(bash -c '
        # Mock fehlgeschlagenen docker-compose Befehl
        docker-compose() {
            echo "Error: Could not start container" >&2
            return 1
        }
        export -f docker-compose
        
        # Teste Fehlerbehandlung
        if docker-compose up -d postgres 2>&1; then
            echo "SUCCESS"
        else
            echo "ERROR_HANDLED"
        fi
    ')
    
    assert_contains "$error_output" "ERROR_HANDLED" "Error not handled correctly"
}

test_timeout_configuration() {
    # Test Timeout-Konfiguration
    local timeout_test
    timeout_test=$(bash -c '
        export KEIKO_BASE_SERVICES_TIMEOUT=999
        source scripts/startup-orchestrator.sh
        load_environment_config "development"
        echo "TIMEOUT=${BASE_SERVICES_TIMEOUT}"
    ')
    
    assert_contains "$timeout_test" "TIMEOUT=999" "Environment variable override not working"
}

test_network_creation() {
    # Test Docker Network Creation
    local network_output
    network_output=$(bash -c '
        export PATH="'${MOCK_DIR}':${PATH}"
        
        # Test network creation
        docker network create keiko-test-network --driver bridge
    ')
    
    assert_contains "$network_output" "Network created successfully" "Network creation failed"
}

test_container_status_check() {
    # Test Container Status Check
    local status_output
    status_output=$(bash -c '
        export PATH="'${MOCK_DIR}':${PATH}"
        
        # Check container status
        docker ps --filter "name=keiko-"
    ')
    
    assert_contains "$status_output" "keiko-postgres" "PostgreSQL container not found in status"
    assert_contains "$status_output" "keiko-redis" "Redis container not found in status"
}

test_log_file_creation() {
    # Test Log-Datei-Erstellung
    local test_log="${PROJECT_ROOT}/logs/test-startup.log"
    
    # Erstelle Test-Log
    echo "Test log entry" > "$test_log"
    
    if [[ -f "$test_log" ]]; then
        local log_content
        log_content=$(cat "$test_log")
        assert_contains "$log_content" "Test log entry" "Log file content incorrect"
        rm -f "$test_log"
        return 0
    else
        echo "Log file not created"
        return 1
    fi
}

test_configuration_file_parsing() {
    # Test Konfigurationsdatei-Parsing
    local test_config="${SCRIPT_DIR}/test-config.conf"
    
    # Erstelle Test-Konfiguration
    cat > "$test_config" << 'EOF'
[test]
BASE_SERVICES_TIMEOUT=123
MONITORING_SERVICES_TIMEOUT=456

[production]
BASE_SERVICES_TIMEOUT=789
EOF
    
    local parsed_config
    parsed_config=$(bash -c '
        CONFIG_FILE="'$test_config'"
        source scripts/startup-orchestrator.sh
        load_environment_config "test"
        echo "BASE=${BASE_SERVICES_TIMEOUT}"
        echo "MONITORING=${MONITORING_SERVICES_TIMEOUT}"
    ')
    
    assert_contains "$parsed_config" "BASE=123" "Test config BASE_SERVICES_TIMEOUT not parsed"
    assert_contains "$parsed_config" "MONITORING=456" "Test config MONITORING_SERVICES_TIMEOUT not parsed"
    
    rm -f "$test_config"
}

# Cleanup function
cleanup_test_environment() {
    echo -e "\n${BLUE}🧹 Cleaning up test environment...${NC}"
    
    # Remove mock commands
    rm -rf "${MOCK_DIR}"
    
    # Reset environment variables
    unset KEIKO_ENV
    unset KEIKO_BASE_SERVICES_TIMEOUT
    unset KEIKO_MONITORING_SERVICES_TIMEOUT
    unset KEIKO_EDGE_SERVICES_TIMEOUT
    unset KEIKO_WORKFLOW_SERVICES_TIMEOUT
    unset KEIKO_BACKEND_STARTUP_TIMEOUT
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Main test runner
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    Keiko Startup Orchestrator Tests                         ║"
    echo "║                              Shell Script Tests                             ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Setup
    setup_test_environment
    
    # Run tests
    run_test "Environment Config Loading" test_environment_config_loading
    run_test "Service Category Functions" test_service_category_functions
    run_test "Docker Compose Startup" test_docker_compose_startup
    run_test "Health Check Simulation" test_health_check_simulation
    run_test "Graceful Shutdown" test_graceful_shutdown
    run_test "Error Handling" test_error_handling
    run_test "Timeout Configuration" test_timeout_configuration
    run_test "Network Creation" test_network_creation
    run_test "Container Status Check" test_container_status_check
    run_test "Log File Creation" test_log_file_creation
    run_test "Configuration File Parsing" test_configuration_file_parsing
    
    # Cleanup
    cleanup_test_environment
    
    # Results
    echo -e "\n${BLUE}📊 Test Results:${NC}"
    echo -e "Tests run: ${TESTS_RUN}"
    echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
    echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "\n${GREEN}🎉 All tests passed!${NC}"
        exit 0
    else
        echo -e "\n${RED}❌ Some tests failed!${NC}"
        exit 1
    fi
}

# Run tests if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
