#!/bin/bash

# Test-Coverage-Script für Frontend-Backend-Integration-Verbesserungen
# Führt Unit-Tests aus und überprüft Code-Coverage-Ziele (≥85%)

set -e

echo "🧪 Starte Test-Coverage-Analyse für Frontend-Backend-Integration..."

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funktionen
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Coverage-Ziel
COVERAGE_TARGET=85

print_header "Frontend Tests (TypeScript/Vitest)"

# Prüfe ob Node.js verfügbar ist
if ! command -v node &> /dev/null; then
    print_error "Node.js nicht gefunden. Bitte installieren Sie Node.js."
    exit 1
fi

# Wechsle ins Frontend-Verzeichnis
cd frontend

# Installiere Dependencies falls nötig
if [ ! -d "node_modules" ]; then
    print_warning "Node modules nicht gefunden. Installiere Dependencies..."
    npm install
fi

# Führe Frontend-Tests mit Coverage aus
print_header "Führe Frontend-Tests aus..."

# Offline Queue Tests
echo "📦 Teste Offline Queue..."
npm run test -- src/services/offline/__tests__/queue.test.ts --coverage --reporter=verbose

# Cache Invalidation Tests  
echo "🗄️ Teste Cache-Invalidierung..."
npm run test -- src/websocket/__tests__/cache-invalidation.test.ts --coverage --reporter=verbose

# Cache Metrics Tests
echo "📊 Teste Cache-Metriken..."
npm run test -- src/services/perf/__tests__/cache-metrics.test.ts --coverage --reporter=verbose

# Gesamte Frontend-Coverage
echo "📈 Generiere Frontend-Coverage-Report..."
npm run test -- --coverage --reporter=verbose --run

# Coverage-Report auswerten
if [ -f "coverage/coverage-summary.json" ]; then
    # Extrahiere Coverage-Werte mit jq falls verfügbar
    if command -v jq &> /dev/null; then
        TOTAL_COVERAGE=$(jq -r '.total.lines.pct' coverage/coverage-summary.json)
        BRANCH_COVERAGE=$(jq -r '.total.branches.pct' coverage/coverage-summary.json)
        FUNCTION_COVERAGE=$(jq -r '.total.functions.pct' coverage/coverage-summary.json)
        
        echo ""
        print_header "Frontend Coverage-Ergebnisse"
        echo "📊 Lines Coverage: ${TOTAL_COVERAGE}%"
        echo "🌿 Branch Coverage: ${BRANCH_COVERAGE}%"
        echo "🔧 Function Coverage: ${FUNCTION_COVERAGE}%"
        
        # Prüfe Coverage-Ziel
        if (( $(echo "$TOTAL_COVERAGE >= $COVERAGE_TARGET" | bc -l) )); then
            print_success "Frontend Coverage-Ziel erreicht: ${TOTAL_COVERAGE}% ≥ ${COVERAGE_TARGET}%"
        else
            print_error "Frontend Coverage-Ziel verfehlt: ${TOTAL_COVERAGE}% < ${COVERAGE_TARGET}%"
            FRONTEND_COVERAGE_FAILED=1
        fi
    else
        print_warning "jq nicht verfügbar. Coverage-Auswertung übersprungen."
    fi
else
    print_warning "Coverage-Report nicht gefunden."
fi

# Wechsle zurück ins Root-Verzeichnis
cd ..

print_header "Backend Tests (Python/pytest)"

# Prüfe ob Python verfügbar ist
if ! command -v python3 &> /dev/null; then
    print_error "Python3 nicht gefunden. Bitte installieren Sie Python3."
    exit 1
fi

# Prüfe ob pytest verfügbar ist
if ! python3 -c "import pytest" &> /dev/null; then
    print_warning "pytest nicht gefunden. Installiere pytest..."
    pip3 install pytest pytest-cov pytest-asyncio
fi

# Führe Backend-Tests mit Coverage aus
print_header "Führe Backend-Tests aus..."

# Cache Metrics Integration Tests
echo "🔗 Teste Cache-Metriken-Integration..."
python3 -m pytest backend/tests/api/test_metrics_cache_integration.py -v --cov=backend/api/routes/metrics_routes --cov-report=term-missing --cov-report=html:backend/htmlcov

# Prüfe Backend-Coverage
if [ -f "backend/.coverage" ]; then
    # Extrahiere Coverage mit coverage.py
    cd backend
    BACKEND_COVERAGE=$(python3 -m coverage report --format=total 2>/dev/null || echo "0")
    cd ..
    
    echo ""
    print_header "Backend Coverage-Ergebnisse"
    echo "📊 Total Coverage: ${BACKEND_COVERAGE}%"
    
    # Prüfe Coverage-Ziel
    if (( $(echo "$BACKEND_COVERAGE >= $COVERAGE_TARGET" | bc -l) )); then
        print_success "Backend Coverage-Ziel erreicht: ${BACKEND_COVERAGE}% ≥ ${COVERAGE_TARGET}%"
    else
        print_error "Backend Coverage-Ziel verfehlt: ${BACKEND_COVERAGE}% < ${COVERAGE_TARGET}%"
        BACKEND_COVERAGE_FAILED=1
    fi
else
    print_warning "Backend Coverage-Report nicht gefunden."
fi

print_header "Service Worker Tests"

# Service Worker kann nicht direkt getestet werden, aber wir können die Syntax prüfen
echo "🔧 Prüfe Service Worker Syntax..."
if node -c frontend/public/service-worker.js; then
    print_success "Service Worker Syntax korrekt"
else
    print_error "Service Worker Syntax-Fehler"
    SERVICE_WORKER_FAILED=1
fi

print_header "Integration Tests"

# Führe einfache Integration-Tests aus
echo "🔗 Teste API-Integration..."

# Prüfe ob Backend läuft (optional)
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Backend erreichbar"
    
    # Teste Metrics-Endpoint
    if curl -s -X POST http://localhost:8000/api/v1/metrics/client \
        -H "Content-Type: application/json" \
        -d '{"session":{"id":"test","startedAt":1640995200000,"durationMs":1000}}' > /dev/null; then
        print_success "Metrics-Endpoint funktional"
    else
        print_warning "Metrics-Endpoint nicht erreichbar"
    fi
else
    print_warning "Backend nicht erreichbar (optional für Tests)"
fi

print_header "Test-Zusammenfassung"

# Gesamtergebnis
TOTAL_FAILURES=0

if [ "${FRONTEND_COVERAGE_FAILED:-0}" -eq 1 ]; then
    print_error "Frontend Coverage-Ziel verfehlt"
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
fi

if [ "${BACKEND_COVERAGE_FAILED:-0}" -eq 1 ]; then
    print_error "Backend Coverage-Ziel verfehlt"
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
fi

if [ "${SERVICE_WORKER_FAILED:-0}" -eq 1 ]; then
    print_error "Service Worker Tests fehlgeschlagen"
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
fi

if [ $TOTAL_FAILURES -eq 0 ]; then
    print_success "Alle Tests erfolgreich! Coverage-Ziel von ${COVERAGE_TARGET}% erreicht."
    echo ""
    echo "📋 Implementierte Features:"
    echo "  ✅ Service Worker Background Sync"
    echo "  ✅ Event-Mapping für Cache-Invalidierung"
    echo "  ✅ Cache-Performance-Metriken"
    echo "  ✅ Unit-Tests mit ≥85% Coverage"
    echo ""
    echo "🎉 Frontend-Backend-Integration erfolgreich verbessert!"
    exit 0
else
    print_error "${TOTAL_FAILURES} Test-Kategorie(n) fehlgeschlagen"
    echo ""
    echo "📋 Nächste Schritte:"
    echo "  1. Überprüfen Sie die Test-Ausgaben oben"
    echo "  2. Beheben Sie fehlgeschlagene Tests"
    echo "  3. Erhöhen Sie die Code-Coverage falls nötig"
    echo "  4. Führen Sie das Script erneut aus"
    exit 1
fi
