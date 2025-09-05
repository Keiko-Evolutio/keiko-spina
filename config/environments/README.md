# Environment-Konfigurationssystem - Keiko Personal Assistant

**Version:** 2.0  
**Letzte Aktualisierung:** 2025-08-29  
**Status:** ✅ Produktionsreif

## 📋 Überblick

### Das neue konsolidierte System

Das Keiko Personal Assistant Backend verwendet jetzt ein **konsolidiertes Environment-Konfigurationssystem**, das die Komplexität drastisch reduziert und die Wartbarkeit erhöht.

#### Vorher vs. Nachher

| **Vorher (v1.0)** | **Nachher (v2.0)** |
|-------------------|---------------------|
| 12 verschiedene .env Dateien | 7 strukturierte Dateien |
| Massive Redundanz | Klare Trennung |
| Inkonsistente Variablennamen | Standardisierte Konventionen |
| Schwer wartbar | Entwicklerfreundlich |

#### Neue Struktur

```
config/
├── environments/          # Hauptumgebungen
│   ├── .env.example      # Template für alle Umgebungen
│   ├── .env.development  # Lokale Entwicklung
│   ├── .env.staging      # Pre-Production Testing
│   └── .env.production   # Live-Deployment
└── features/             # Feature-spezifische Konfigurationen
    ├── logfire.dev.env   # Logfire Development
    ├── logfire.prod.env  # Logfire Production
    ├── monitoring.env    # Monitoring-System
    ├── voice_performance.env
    ├── voice_rate_limiting.env
    └── agent_circuit_breaker.env
```

### Vorteile des neuen Systems

- **🎯 Klarheit:** Jede Umgebung hat ihre eigene Datei
- **🔒 Sicherheit:** Klare Trennung von Development/Production-Credentials
- **🛠️ Wartbarkeit:** Standardisierte Variablennamen und Struktur
- **📚 Dokumentation:** Vollständig dokumentierte Konfigurationsoptionen
- **🚀 Skalierbarkeit:** Einfache Erweiterung für neue Umgebungen

## 🚀 Setup-Anleitung

### 1. Development-Umgebung (Lokale Entwicklung)

#### Schritt 1: Basis-Konfiguration erstellen
```bash
# Im backend/ Verzeichnis
cp config/environments/.env.example config/environments/.env.development
```

#### Schritt 2: Development-spezifische Anpassungen
```bash
# .env.development bearbeiten
ENVIRONMENT=development
LOG_LEVEL=DEBUG
KEIKO_ENABLE_CLICKABLE_LINKS=true

# Sicherheit (Development)
VERIFY_SSL=false  # Nur für lokale Entwicklung!
WEBSOCKET_AUTH_ENABLED=false
WEBSOCKET_AUTH_MODE=optional

# Development-Tokens (NICHT in Production verwenden!)
KEIKO_DEV_TOKEN=dev-token-12345
KEI_JWT_SECRET=dev-jwt-secret-please-change
```

#### Schritt 3: Azure-Credentials konfigurieren
```bash
# Ihre Azure AI Services Credentials
AZURE_OPENAI_ENDPOINT=https://your-dev-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-development-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

#### Schritt 4: Lokale Services
```bash
# Redis (lokal)
REDIS_URL=redis://localhost:6379/0

# PostgreSQL (lokal)
DATABASE_URL=postgresql://user:password@localhost:5432/keiko_dev
```

### 2. Staging-Umgebung (Pre-Production Testing)

#### Schritt 1: Staging-Konfiguration erstellen
```bash
cp config/environments/.env.example config/environments/.env.staging
```

#### Schritt 2: Staging-spezifische Anpassungen
```bash
# .env.staging bearbeiten
ENVIRONMENT=staging
LOG_LEVEL=INFO
KEIKO_ENABLE_CLICKABLE_LINKS=false

# Sicherheit (Staging - Production-ähnlich)
VERIFY_SSL=true
WEBSOCKET_AUTH_ENABLED=true
WEBSOCKET_AUTH_MODE=required
WEBSOCKET_AUTH_STRICT_MODE=true

# Staging-Credentials (separate von Production!)
KEI_JWT_SECRET=${STAGING_JWT_SECRET}
AZURE_OPENAI_API_KEY=${STAGING_AZURE_API_KEY}
```

### 3. Production-Umgebung (Live-Deployment)

#### Schritt 1: Production-Konfiguration erstellen
```bash
cp config/environments/.env.example config/environments/.env.production
```

#### Schritt 2: Production-Sicherheitskonfiguration
```bash
# .env.production bearbeiten
ENVIRONMENT=production
LOG_LEVEL=WARNING
KEIKO_ENABLE_CLICKABLE_LINKS=false

# Maximale Sicherheit
VERIFY_SSL=true
WEBSOCKET_AUTH_ENABLED=true
WEBSOCKET_AUTH_MODE=required
WEBSOCKET_AUTH_STRICT_MODE=true
WEBSOCKET_AUTH_FALLBACK_TO_BYPASS=false

# Production-Credentials (über Environment-Variablen setzen!)
KEI_JWT_SECRET=${PRODUCTION_JWT_SECRET}
KEI_JWT_ALGORITHM=RS256  # Sicherer Algorithmus für Production
AZURE_OPENAI_API_KEY=${PRODUCTION_AZURE_API_KEY}
```

## 📖 Konfigurationsreferenz

### Core Application Settings

| Variable | Development | Staging | Production | Beschreibung |
|----------|-------------|---------|------------|--------------|
| `ENVIRONMENT` | `development` | `staging` | `production` | Umgebungstyp |
| `LOG_LEVEL` | `DEBUG` | `INFO` | `WARNING` | Logging-Level |
| `KEIKO_ENABLE_CLICKABLE_LINKS` | `true` | `false` | `false` | Clickable Links in Logs |

### Sicherheitskonfiguration

| Variable | Development | Production | Beschreibung |
|----------|-------------|------------|--------------|
| `VERIFY_SSL` | `false` | `true` | SSL-Zertifikat-Verifikation |
| `WEBSOCKET_AUTH_ENABLED` | `false` | `true` | WebSocket-Authentifizierung |
| `WEBSOCKET_AUTH_MODE` | `optional` | `required` | Auth-Modus |
| `WEBSOCKET_AUTH_STRICT_MODE` | `false` | `true` | Strenger Auth-Modus |

### JWT-Konfiguration

| Variable | Development | Production | Beschreibung |
|----------|-------------|------------|--------------|
| `KEI_JWT_SECRET` | `dev-jwt-secret` | `${JWT_SECRET}` | JWT-Signatur-Schlüssel |
| `KEI_JWT_ALGORITHM` | `HS256` | `RS256` | JWT-Algorithmus |
| `KEI_JWT_ISSUER` | `keiko-platform` | `keiko-platform-prod` | JWT-Aussteller |

### Azure AI Services

| Variable | Beschreibung | Beispiel |
|----------|--------------|----------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI Endpoint | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | API-Schlüssel | `your-api-key-here` |
| `AZURE_OPENAI_API_VERSION` | API-Version | `2024-02-15-preview` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Deployment-Name | `gpt-4` |

### Rate Limiting

| Variable | Development | Production | Beschreibung |
|----------|-------------|------------|--------------|
| `KEI_DEFAULT_RATE_LIMIT` | `5000` | `1000` | Standard Rate Limit |
| `KEI_BURST_RATE_LIMIT` | `10000` | `2000` | Burst Rate Limit |

## 💡 Praktische Beispiele

### Development-Setup (Vollständig)

```bash
# config/environments/.env.development
ENVIRONMENT=development
LOG_LEVEL=DEBUG
KEIKO_ENABLE_CLICKABLE_LINKS=true

# Sicherheit (Development)
VERIFY_SSL=false
WEBSOCKET_AUTH_ENABLED=false
WEBSOCKET_AUTH_MODE=optional
WEBSOCKET_AUTH_FALLBACK_TO_BYPASS=true
WEBSOCKET_AUTH_STRICT_MODE=false

# Development-Credentials
KEIKO_DEV_TOKEN=dev-token-12345
KEI_API_TOKEN=dev-api-token
KEI_MCP_API_TOKEN=dev-mcp-token
KEI_JWT_SECRET=dev-jwt-secret-please-change
KEI_JWT_ALGORITHM=HS256

# Azure AI (Development)
AZURE_OPENAI_ENDPOINT=https://keiko-dev.openai.azure.com/
AZURE_OPENAI_API_KEY=your-dev-api-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4

# Lokale Services
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql://keiko:password@localhost:5432/keiko_dev

# Rate Limiting (Development - entspannt)
KEI_DEFAULT_RATE_LIMIT=5000
KEI_BURST_RATE_LIMIT=10000

# Feature Flags
ENABLE_VOICE_FEATURES=true
ENABLE_WEBRTC=true
ENABLE_MONITORING=true
```

### Production-Setup (Sicherheitsfokus)

```bash
# config/environments/.env.production
ENVIRONMENT=production
LOG_LEVEL=WARNING
KEIKO_ENABLE_CLICKABLE_LINKS=false

# Maximale Sicherheit
VERIFY_SSL=true
WEBSOCKET_AUTH_ENABLED=true
WEBSOCKET_AUTH_MODE=required
WEBSOCKET_AUTH_METHODS=jwt,mtls
WEBSOCKET_AUTH_FALLBACK_TO_BYPASS=false
WEBSOCKET_AUTH_STRICT_MODE=true

# Production-Credentials (über Environment-Variablen!)
KEI_API_TOKEN=${PRODUCTION_API_TOKEN}
KEI_MCP_API_TOKEN=${PRODUCTION_MCP_TOKEN}
KEI_JWT_SECRET=${PRODUCTION_JWT_SECRET}
KEI_JWT_ALGORITHM=RS256
KEI_JWT_ISSUER=keiko-platform-prod
KEI_JWT_AUDIENCE=keiko-api-prod

# Azure AI (Production)
AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-prod

# Production Services
REDIS_URL=${REDIS_CONNECTION_STRING}
DATABASE_URL=${DATABASE_CONNECTION_STRING}

# Rate Limiting (Production - restriktiv)
KEI_DEFAULT_RATE_LIMIT=1000
KEI_BURST_RATE_LIMIT=2000
```

## 🔧 Feature-spezifische Konfigurationen

### Logfire-Konfiguration

#### Development (`config/features/logfire.dev.env`)
```bash
# Logfire Development
LOGFIRE_TOKEN=your-dev-logfire-token
LOGFIRE_PROJECT_NAME=keiko-dev
LOGFIRE_ENVIRONMENT=development
LOGFIRE_SEND_TO_LOGFIRE=false  # Lokal testen
LOGFIRE_CONSOLE_LOGS=true
LOGFIRE_CONSOLE_SPAN_EVENTS=true
```

#### Production (`config/features/logfire.prod.env`)
```bash
# Logfire Production
LOGFIRE_TOKEN=${LOGFIRE_PRODUCTION_TOKEN}
LOGFIRE_PROJECT_NAME=keiko-production
LOGFIRE_ENVIRONMENT=production
LOGFIRE_SEND_TO_LOGFIRE=true
LOGFIRE_CONSOLE_LOGS=false
LOGFIRE_CONSOLE_SPAN_EVENTS=false
LOGFIRE_SAMPLING_RATE=0.1  # 10% Sampling für Performance
```

### Monitoring-Konfiguration (`config/features/monitoring.env`)
```bash
# Monitoring System
MONITORING_ENABLED=true
METRICS_COLLECTION_INTERVAL=30
HEALTH_CHECK_INTERVAL=60
ALERT_WEBHOOK_URL=${MONITORING_WEBHOOK_URL}
PROMETHEUS_METRICS_ENABLED=true
GRAFANA_DASHBOARD_ENABLED=true
```

### Voice-Performance (`config/features/voice_performance.env`)
```bash
# Voice Performance Optimization
VOICE_PERFORMANCE_ENABLED=true
VOICE_CACHE_TTL=3600
VOICE_MAX_CONCURRENT_REQUESTS=10
VOICE_TIMEOUT_SECONDS=30
VOICE_QUALITY_OPTIMIZATION=true
```

## ⚠️ Sicherheitshinweise

### 🔒 Production-Credentials

**NIEMALS** echte Production-Credentials in .env-Dateien speichern!

#### ✅ Richtig (über Environment-Variablen):
```bash
# In .env.production
KEI_JWT_SECRET=${PRODUCTION_JWT_SECRET}
AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
```

#### ❌ Falsch (Hardcoded):
```bash
# NIEMALS so machen!
KEI_JWT_SECRET=super-secret-production-key-123
AZURE_OPENAI_API_KEY=sk-real-production-key-here
```

### 🛡️ Deployment-Best-Practices

1. **Secrets Management:** Verwenden Sie Azure Key Vault, AWS Secrets Manager oder ähnliche Services
2. **Environment-Variablen:** Setzen Sie Credentials über CI/CD-Pipeline
3. **Rotation:** Rotieren Sie Secrets regelmäßig
4. **Monitoring:** Überwachen Sie Zugriffe auf Credentials

### 🔐 JWT-Sicherheit

| Umgebung | Algorithmus | Schlüssel-Typ | Rotation |
|----------|-------------|----------------|----------|
| Development | HS256 | Shared Secret | Nicht nötig |
| Staging | RS256 | RSA Key Pair | Monatlich |
| Production | RS256 | RSA Key Pair | Wöchentlich |

## 🐛 Troubleshooting

### Häufige Probleme und Lösungen

#### Problem 1: "Konfigurationsdatei nicht gefunden"
```bash
# Fehler
FileNotFoundError: .env.development not found

# Lösung
cp config/environments/.env.example config/environments/.env.development
```

#### Problem 2: "JWT-Token ungültig"
```bash
# Fehler
JWT signature verification failed

# Lösung - Prüfen Sie:
1. KEI_JWT_SECRET ist gesetzt
2. KEI_JWT_ALGORITHM stimmt überein
3. Token ist nicht abgelaufen
```

#### Problem 3: "Azure OpenAI Verbindungsfehler"
```bash
# Fehler
Azure OpenAI API connection failed

# Lösung - Prüfen Sie:
1. AZURE_OPENAI_ENDPOINT ist korrekt
2. AZURE_OPENAI_API_KEY ist gültig
3. AZURE_OPENAI_DEPLOYMENT_NAME existiert
4. Netzwerk-Konnektivität
```

#### Problem 4: "WebSocket-Authentifizierung fehlgeschlagen"
```bash
# Fehler
WebSocket authentication failed

# Development-Lösung:
WEBSOCKET_AUTH_ENABLED=false
WEBSOCKET_AUTH_MODE=optional

# Production-Lösung:
1. Gültigen JWT-Token verwenden
2. WEBSOCKET_AUTH_METHODS prüfen
3. Zertifikate für mTLS validieren
```

### 🔍 Konfiguration validieren

#### Automatische Validierung
```bash
# Im backend/ Verzeichnis
python3 -c "
from config.settings import get_settings
settings = get_settings()
print(f'✅ Konfiguration geladen: {settings.environment}')
print(f'✅ Azure Endpoint: {settings.azure_openai_endpoint[:30]}...')
print(f'✅ JWT konfiguriert: {bool(settings.kei_jwt_secret)}')
"
```

#### Manuelle Checkliste

**Development:**
- [ ] `.env.development` existiert
- [ ] `ENVIRONMENT=development`
- [ ] Azure-Credentials gesetzt
- [ ] Lokale Services erreichbar
- [ ] JWT-Secret gesetzt

**Production:**
- [ ] `.env.production` existiert
- [ ] `ENVIRONMENT=production`
- [ ] Alle Credentials über Environment-Variablen
- [ ] SSL-Verifikation aktiviert
- [ ] WebSocket-Auth aktiviert
- [ ] Rate Limiting konfiguriert

## 📚 Weiterführende Dokumentation

- **[Azure AI Services Setup](../docs/azure-setup.md)**
- **[WebSocket-Authentifizierung](../docs/websocket-auth.md)**
- **[Monitoring-Konfiguration](../docs/monitoring-setup.md)**
- **[Deployment-Guide](../docs/deployment.md)**

## 🆘 Support

Bei Problemen mit der Konfiguration:

1. **Dokumentation prüfen:** Diese README und verlinkte Docs
2. **Logs analysieren:** `LOG_LEVEL=DEBUG` für detaillierte Ausgaben
3. **Validierung ausführen:** Automatische Konfigurationsprüfung
4. **Team kontaktieren:** Bei persistenten Problemen

---

**Letzte Aktualisierung:** 2025-08-29
**Version:** 2.0
**Status:** ✅ Produktionsreif
