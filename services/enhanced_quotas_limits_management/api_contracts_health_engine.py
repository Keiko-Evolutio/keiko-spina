# backend/services/enhanced_quotas_limits_management/api_contracts_health_engine.py
"""API Contracts & Health Checks Engine.

Implementiert umfassende API-Contract-Validierung und Health-Checks
für alle Service-Komponenten mit SLA-Monitoring.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

import aiohttp

from kei_logging import get_logger

from .data_models import APIContract, HealthCheckResult, HealthStatus

logger = get_logger(__name__)


class APIContractsHealthEngine:
    """API Contracts & Health Checks Engine für Service-Monitoring."""

    def __init__(self):
        """Initialisiert API Contracts & Health Engine."""
        # Health-Check-Konfiguration
        self.enable_health_checks = True
        self.enable_sla_monitoring = True
        self.enable_contract_validation = True
        self.default_timeout_seconds = 30

        # API-Contracts-Storage
        self._api_contracts: dict[str, APIContract] = {}
        self._health_check_results: dict[str, list[HealthCheckResult]] = {}
        self._max_health_history = 1000

        # SLA-Tracking
        self._sla_violations: list[dict[str, Any]] = []
        self._service_availability: dict[str, dict[str, Any]] = {}

        # Performance-Tracking
        self._health_check_count = 0
        self._total_health_check_time_ms = 0.0
        self._contract_validation_count = 0

        # Background-Tasks
        self._health_check_task: asyncio.Task | None = None
        self._sla_monitoring_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("API Contracts & Health Engine initialisiert")

    async def start(self) -> None:
        """Startet API Contracts & Health Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._sla_monitoring_task = asyncio.create_task(self._sla_monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Initialisiere Standard-API-Contracts
        await self._initialize_default_contracts()

        logger.info("API Contracts & Health Engine gestartet")

    async def stop(self) -> None:
        """Stoppt API Contracts & Health Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._sla_monitoring_task:
            self._sla_monitoring_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        await asyncio.gather(
            self._health_check_task,
            self._sla_monitoring_task,
            self._cleanup_task,
            return_exceptions=True
        )

        logger.info("API Contracts & Health Engine gestoppt")

    async def register_api_contract(
        self,
        service_name: str,
        version: str,
        endpoint_path: str,
        http_method: str = "GET",
        max_response_time_ms: int = 1000,
        min_success_rate: float = 0.95,
        max_error_rate: float = 0.05,
        health_check_interval_seconds: int = 60
    ) -> str:
        """Registriert neuen API-Contract.

        Args:
            service_name: Service-Name
            version: Service-Version
            endpoint_path: Endpoint-Pfad
            http_method: HTTP-Method
            max_response_time_ms: Max Response-Zeit
            min_success_rate: Min Success-Rate
            max_error_rate: Max Error-Rate
            health_check_interval_seconds: Health-Check-Intervall

        Returns:
            Contract-ID
        """
        try:
            import uuid
            contract_id = str(uuid.uuid4())

            contract = APIContract(
                contract_id=contract_id,
                service_name=service_name,
                version=version,
                endpoint_path=endpoint_path,
                http_method=http_method,
                max_response_time_ms=max_response_time_ms,
                min_success_rate=min_success_rate,
                max_error_rate=max_error_rate,
                health_check_interval_seconds=health_check_interval_seconds
            )

            # Speichere Contract
            self._api_contracts[contract_id] = contract

            # Initialisiere Service-Availability-Tracking
            self._service_availability[service_name] = {
                "total_checks": 0,
                "successful_checks": 0,
                "failed_checks": 0,
                "avg_response_time_ms": 0.0,
                "current_status": HealthStatus.UNKNOWN,
                "last_check": None,
                "uptime_percentage": 100.0
            }

            logger.info({
                "event": "api_contract_registered",
                "contract_id": contract_id,
                "service_name": service_name,
                "endpoint_path": endpoint_path,
                "max_response_time_ms": max_response_time_ms
            })

            return contract_id

        except Exception as e:
            logger.error(f"API contract registration fehlgeschlagen: {e}")
            raise

    async def perform_health_check(
        self,
        contract_id: str,
        base_url: str | None = None
    ) -> HealthCheckResult:
        """Führt Health-Check für API-Contract durch.

        Args:
            contract_id: Contract-ID
            base_url: Base-URL für Health-Check

        Returns:
            Health-Check-Result
        """
        start_time = time.time()

        try:
            contract = self._api_contracts.get(contract_id)
            if not contract:
                raise ValueError(f"API Contract {contract_id} nicht gefunden")

            logger.debug({
                "event": "health_check_started",
                "contract_id": contract_id,
                "service_name": contract.service_name,
                "endpoint_path": contract.endpoint_path
            })

            # Konstruiere Health-Check-URL
            if base_url:
                health_check_url = f"{base_url.rstrip('/')}{contract.endpoint_path}"
            else:
                # Fallback: Lokaler Health-Check
                health_check_url = f"http://localhost:8000{contract.endpoint_path}"

            # Führe HTTP-Request durch
            async with aiohttp.ClientSession() as session:
                try:
                    request_start = time.time()

                    async with session.request(
                        method=contract.http_method,
                        url=health_check_url,
                        timeout=aiohttp.ClientTimeout(total=contract.health_check_timeout_seconds)
                    ) as response:
                        request_time_ms = (time.time() - request_start) * 1000

                        # Bestimme Health-Status
                        if response.status == 200:
                            if request_time_ms <= contract.max_response_time_ms:
                                status = HealthStatus.HEALTHY
                            else:
                                status = HealthStatus.DEGRADED
                        elif 200 <= response.status < 300:
                            status = HealthStatus.DEGRADED
                        else:
                            status = HealthStatus.UNHEALTHY

                        # Berechne Success/Error-Rates (basierend auf Historie)
                        success_rate, error_rate = await self._calculate_service_rates(contract.service_name)

                        # Prüfe SLA-Compliance
                        quota_compliance = True  # TODO: Implementiere echte Quota-Compliance-Prüfung - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
                        rate_limit_compliance = True  # TODO: Implementiere echte Rate-Limit-Compliance-Prüfung - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114

                        # Erstelle Health-Check-Result
                        import uuid
                        result = HealthCheckResult(
                            check_id=str(uuid.uuid4()),
                            contract_id=contract_id,
                            service_name=contract.service_name,
                            status=status,
                            response_time_ms=request_time_ms,
                            success_rate=success_rate,
                            error_rate=error_rate,
                            quota_compliance=quota_compliance,
                            rate_limit_compliance=rate_limit_compliance,
                            details={
                                "http_status": response.status,
                                "endpoint": health_check_url,
                                "method": contract.http_method
                            }
                        )

                        # Prüfe SLA-Violations
                        await self._check_sla_violations(contract, result)

                except TimeoutError:
                    # Timeout
                    result = HealthCheckResult(
                        check_id=str(uuid.uuid4()),
                        contract_id=contract_id,
                        service_name=contract.service_name,
                        status=HealthStatus.UNHEALTHY,
                        response_time_ms=contract.health_check_timeout_seconds * 1000,
                        success_rate=0.0,
                        error_rate=1.0,
                        errors=["Health check timeout"]
                    )

                except Exception as e:
                    # Connection-Error
                    result = HealthCheckResult(
                        check_id=str(uuid.uuid4()),
                        contract_id=contract_id,
                        service_name=contract.service_name,
                        status=HealthStatus.CRITICAL,
                        response_time_ms=0.0,
                        success_rate=0.0,
                        error_rate=1.0,
                        errors=[f"Connection error: {e!s}"]
                    )

            # Speichere Health-Check-Result
            if contract_id not in self._health_check_results:
                self._health_check_results[contract_id] = []

            self._health_check_results[contract_id].append(result)

            # Limitiere History-Größe
            if len(self._health_check_results[contract_id]) > self._max_health_history:
                self._health_check_results[contract_id] = self._health_check_results[contract_id][-self._max_health_history:]

            # Aktualisiere Service-Availability
            await self._update_service_availability(contract.service_name, result)

            # Performance-Tracking
            check_duration_ms = (time.time() - start_time) * 1000
            self._update_health_check_performance_stats(check_duration_ms)

            logger.debug({
                "event": "health_check_completed",
                "contract_id": contract_id,
                "status": result.status.value,
                "response_time_ms": result.response_time_ms,
                "check_duration_ms": check_duration_ms
            })

            return result

        except Exception as e:
            logger.error(f"Health check fehlgeschlagen für Contract {contract_id}: {e}")

            # Fallback-Result
            import uuid
            return HealthCheckResult(
                check_id=str(uuid.uuid4()),
                contract_id=contract_id,
                service_name="unknown",
                status=HealthStatus.CRITICAL,
                response_time_ms=0.0,
                success_rate=0.0,
                error_rate=1.0,
                errors=[f"Health check error: {e!s}"]
            )

    async def validate_api_contract_compliance(
        self,
        contract_id: str,
        actual_response_time_ms: float,
        success: bool
    ) -> dict[str, Any]:
        """Validiert API-Contract-Compliance.

        Args:
            contract_id: Contract-ID
            actual_response_time_ms: Tatsächliche Response-Zeit
            success: Erfolg-Status

        Returns:
            Compliance-Result
        """
        try:
            contract = self._api_contracts.get(contract_id)
            if not contract:
                return {"compliant": False, "error": f"Contract {contract_id} nicht gefunden"}

            violations = []
            warnings = []

            # Prüfe Response-Zeit
            if actual_response_time_ms > contract.max_response_time_ms:
                violations.append(f"Response time {actual_response_time_ms:.1f}ms exceeds limit {contract.max_response_time_ms}ms")

            # Berechne aktuelle Success/Error-Rates
            success_rate, error_rate = await self._calculate_service_rates(contract.service_name)

            # Prüfe Success-Rate
            if success_rate < contract.min_success_rate:
                violations.append(f"Success rate {success_rate:.2f} below minimum {contract.min_success_rate}")

            # Prüfe Error-Rate
            if error_rate > contract.max_error_rate:
                violations.append(f"Error rate {error_rate:.2f} exceeds maximum {contract.max_error_rate}")

            # Bestimme Compliance-Status
            compliant = len(violations) == 0

            self._contract_validation_count += 1

            result = {
                "compliant": compliant,
                "contract_id": contract_id,
                "service_name": contract.service_name,
                "violations": violations,
                "warnings": warnings,
                "metrics": {
                    "response_time_ms": actual_response_time_ms,
                    "success_rate": success_rate,
                    "error_rate": error_rate,
                    "success": success
                }
            }

            if not compliant:
                logger.warning({
                    "event": "api_contract_violation",
                    "contract_id": contract_id,
                    "violations": violations
                })

            return result

        except Exception as e:
            logger.error(f"API contract compliance validation fehlgeschlagen: {e}")
            return {"compliant": False, "error": str(e)}

    async def _calculate_service_rates(self, service_name: str) -> tuple[float, float]:
        """Berechnet Success/Error-Rates für Service."""
        try:
            availability = self._service_availability.get(service_name)
            if not availability or availability["total_checks"] == 0:
                return 1.0, 0.0  # Default: 100% success, 0% error

            success_rate = availability["successful_checks"] / availability["total_checks"]
            error_rate = availability["failed_checks"] / availability["total_checks"]

            return success_rate, error_rate

        except Exception as e:
            logger.error(f"Service rates calculation fehlgeschlagen: {e}")
            return 1.0, 0.0

    async def _check_sla_violations(self, contract: APIContract, result: HealthCheckResult) -> None:
        """Prüft SLA-Violations."""
        try:
            violations = []

            # Response-Zeit-SLA
            if result.response_time_ms > contract.max_response_time_ms:
                violations.append({
                    "type": "response_time_sla",
                    "contract_id": contract.contract_id,
                    "service_name": contract.service_name,
                    "expected": contract.max_response_time_ms,
                    "actual": result.response_time_ms,
                    "timestamp": result.check_timestamp
                })

            # Success-Rate-SLA
            if result.success_rate < contract.min_success_rate:
                violations.append({
                    "type": "success_rate_sla",
                    "contract_id": contract.contract_id,
                    "service_name": contract.service_name,
                    "expected": contract.min_success_rate,
                    "actual": result.success_rate,
                    "timestamp": result.check_timestamp
                })

            # Error-Rate-SLA
            if result.error_rate > contract.max_error_rate:
                violations.append({
                    "type": "error_rate_sla",
                    "contract_id": contract.contract_id,
                    "service_name": contract.service_name,
                    "expected": contract.max_error_rate,
                    "actual": result.error_rate,
                    "timestamp": result.check_timestamp
                })

            # Speichere SLA-Violations
            self._sla_violations.extend(violations)

            # Limitiere SLA-Violations-History
            if len(self._sla_violations) > 1000:
                self._sla_violations = self._sla_violations[-1000:]

            if violations:
                logger.warning({
                    "event": "sla_violations_detected",
                    "service_name": contract.service_name,
                    "violations": len(violations)
                })

        except Exception as e:
            logger.error(f"SLA violations check fehlgeschlagen: {e}")

    async def _update_service_availability(self, service_name: str, result: HealthCheckResult) -> None:
        """Aktualisiert Service-Availability-Metriken."""
        try:
            if service_name not in self._service_availability:
                self._service_availability[service_name] = {
                    "total_checks": 0,
                    "successful_checks": 0,
                    "failed_checks": 0,
                    "avg_response_time_ms": 0.0,
                    "current_status": HealthStatus.UNKNOWN,
                    "last_check": None,
                    "uptime_percentage": 100.0
                }

            availability = self._service_availability[service_name]

            # Aktualisiere Counters
            availability["total_checks"] += 1

            if result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                availability["successful_checks"] += 1
            else:
                availability["failed_checks"] += 1

            # Aktualisiere Average Response Time
            total_response_time = availability["avg_response_time_ms"] * (availability["total_checks"] - 1)
            availability["avg_response_time_ms"] = (total_response_time + result.response_time_ms) / availability["total_checks"]

            # Aktualisiere Status
            availability["current_status"] = result.status
            availability["last_check"] = result.check_timestamp

            # Berechne Uptime-Percentage
            availability["uptime_percentage"] = (availability["successful_checks"] / availability["total_checks"]) * 100

        except Exception as e:
            logger.error(f"Service availability update fehlgeschlagen: {e}")

    async def _initialize_default_contracts(self) -> None:
        """Initialisiert Standard-API-Contracts."""
        try:
            # Standard-Contracts für Core-Services
            default_contracts = [
                {
                    "service_name": "enhanced_quotas_limits_management",
                    "version": "1.0.0",
                    "endpoint_path": "/health",
                    "max_response_time_ms": 100,
                    "min_success_rate": 0.99,
                    "max_error_rate": 0.01
                },
                {
                    "service_name": "enhanced_security_integration",
                    "version": "1.0.0",
                    "endpoint_path": "/health",
                    "max_response_time_ms": 150,
                    "min_success_rate": 0.98,
                    "max_error_rate": 0.02
                },
                {
                    "service_name": "policy_aware_selection",
                    "version": "1.0.0",
                    "endpoint_path": "/health",
                    "max_response_time_ms": 200,
                    "min_success_rate": 0.97,
                    "max_error_rate": 0.03
                },
                {
                    "service_name": "orchestrator_service",
                    "version": "1.0.0",
                    "endpoint_path": "/health",
                    "max_response_time_ms": 300,
                    "min_success_rate": 0.95,
                    "max_error_rate": 0.05
                }
            ]

            for contract_config in default_contracts:
                await self.register_api_contract(**contract_config)

            logger.info(f"Standard-API-Contracts initialisiert: {len(default_contracts)} Contracts")

        except Exception as e:
            logger.error(f"Standard-API-Contracts initialization fehlgeschlagen: {e}")

    async def _health_check_loop(self) -> None:
        """Background-Loop für Health-Checks."""
        while self._is_running:
            try:
                # Führe Health-Checks für alle Contracts durch
                for contract_id, contract in self._api_contracts.items():
                    if contract.enabled and self._is_running:
                        await self.perform_health_check(contract_id)
                        await asyncio.sleep(1)  # Kurze Pause zwischen Checks

                # Warte bis zum nächsten Check-Zyklus
                await asyncio.sleep(60)  # Standard-Intervall

            except Exception as e:
                logger.error(f"Health check loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _sla_monitoring_loop(self) -> None:
        """Background-Loop für SLA-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # SLA-Monitoring alle 5 Minuten

                if self._is_running:
                    await self._analyze_sla_trends()

            except Exception as e:
                logger.error(f"SLA monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(300)

    async def _cleanup_loop(self) -> None:
        """Background-Loop für Cleanup."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Cleanup alle Stunde

                if self._is_running:
                    await self._cleanup_old_health_results()

            except Exception as e:
                logger.error(f"Cleanup loop fehlgeschlagen: {e}")
                await asyncio.sleep(3600)

    async def _analyze_sla_trends(self) -> None:
        """Analysiert SLA-Trends."""
        try:
            # TODO: Implementiere SLA-Trend-Analysis - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
            logger.debug("SLA trends analysiert")

        except Exception as e:
            logger.error(f"SLA trend analysis fehlgeschlagen: {e}")

    async def _cleanup_old_health_results(self) -> None:
        """Bereinigt alte Health-Check-Results."""
        try:
            from datetime import datetime, timedelta

            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            for contract_id in list(self._health_check_results.keys()):
                original_count = len(self._health_check_results[contract_id])
                self._health_check_results[contract_id] = [
                    r for r in self._health_check_results[contract_id]
                    if r.check_timestamp > cutoff_time
                ]

                cleaned_count = original_count - len(self._health_check_results[contract_id])
                if cleaned_count > 0:
                    logger.debug(f"Health results cleanup für {contract_id}: {cleaned_count} alte Results entfernt")

        except Exception as e:
            logger.error(f"Health results cleanup fehlgeschlagen: {e}")

    def _update_health_check_performance_stats(self, check_duration_ms: float) -> None:
        """Aktualisiert Health-Check-Performance-Statistiken."""
        self._health_check_count += 1
        self._total_health_check_time_ms += check_duration_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_health_check_time = (
            self._total_health_check_time_ms / self._health_check_count
            if self._health_check_count > 0 else 0.0
        )

        return {
            "total_health_checks": self._health_check_count,
            "avg_health_check_time_ms": avg_health_check_time,
            "contract_validations": self._contract_validation_count,
            "registered_contracts": len(self._api_contracts),
            "monitored_services": len(self._service_availability),
            "sla_violations": len(self._sla_violations),
            "health_checks_enabled": self.enable_health_checks,
            "sla_monitoring_enabled": self.enable_sla_monitoring,
            "contract_validation_enabled": self.enable_contract_validation
        }

    def get_service_dashboard_data(self) -> dict[str, Any]:
        """Holt Service-Dashboard-Daten."""
        try:
            dashboard_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "services": {},
                "global_stats": {
                    "total_services": len(self._service_availability),
                    "healthy_services": 0,
                    "degraded_services": 0,
                    "unhealthy_services": 0,
                    "avg_uptime_percentage": 0.0
                }
            }

            total_uptime = 0.0

            for service_name, availability in self._service_availability.items():
                dashboard_data["services"][service_name] = {
                    "status": availability["current_status"].value,
                    "uptime_percentage": availability["uptime_percentage"],
                    "avg_response_time_ms": availability["avg_response_time_ms"],
                    "total_checks": availability["total_checks"],
                    "successful_checks": availability["successful_checks"],
                    "failed_checks": availability["failed_checks"],
                    "last_check": availability["last_check"].isoformat() if availability["last_check"] else None
                }

                # Aktualisiere Global-Stats
                if availability["current_status"] == HealthStatus.HEALTHY:
                    dashboard_data["global_stats"]["healthy_services"] += 1
                elif availability["current_status"] == HealthStatus.DEGRADED:
                    dashboard_data["global_stats"]["degraded_services"] += 1
                else:
                    dashboard_data["global_stats"]["unhealthy_services"] += 1

                total_uptime += availability["uptime_percentage"]

            # Berechne Average Uptime
            if len(self._service_availability) > 0:
                dashboard_data["global_stats"]["avg_uptime_percentage"] = total_uptime / len(self._service_availability)

            return dashboard_data

        except Exception as e:
            logger.error(f"Service dashboard data generation fehlgeschlagen: {e}")
            return {"error": str(e)}
