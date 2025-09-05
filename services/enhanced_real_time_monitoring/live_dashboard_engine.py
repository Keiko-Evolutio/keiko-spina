# backend/services/enhanced_real_time_monitoring/live_dashboard_engine.py
"""Live Dashboard Engine für Real-time Monitoring.

Implementiert Enterprise-Grade Live-Dashboards mit Real-time Metrics,
Alerting und Performance-Visualisierung für alle Services.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger
from services.enhanced_security_integration import SecurityContext

from .data_models import AlertSeverity, LiveDashboardData, MonitoringAlert, PerformanceMetrics

logger = get_logger(__name__)


class LiveDashboardEngine:
    """Live Dashboard Engine für Enterprise-Grade Real-time Dashboards."""

    def __init__(self):
        """Initialisiert Live Dashboard Engine."""
        # Dashboard-Storage
        self._dashboard_data: dict[str, LiveDashboardData] = {}
        self._dashboard_subscriptions: dict[str, set[str]] = defaultdict(set)  # dashboard_id -> client_ids
        self._client_connections: dict[str, dict[str, Any]] = {}  # client_id -> connection_info

        # Real-time Data
        self._real_time_metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._alert_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._performance_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Dashboard-Konfiguration
        self.refresh_interval_seconds = 5
        self.data_retention_points = 100
        self.enable_real_time_updates = True
        self.enable_alert_notifications = True

        # Performance-Tracking
        self._dashboard_performance_stats = {
            "total_dashboard_updates": 0,
            "avg_dashboard_generation_time_ms": 0.0,
            "total_client_connections": 0,
            "active_subscriptions": 0,
            "data_points_processed": 0
        }

        # Background-Tasks
        self._dashboard_tasks: list[asyncio.Task] = []
        self._is_running = False

        # WebSocket-Simulation (in Realität würde hier ein echter WebSocket-Server verwendet)
        self._websocket_clients: dict[str, dict[str, Any]] = {}

        logger.info("Live Dashboard Engine initialisiert")

    async def start(self) -> None:
        """Startet Live Dashboard Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._dashboard_tasks = [
            asyncio.create_task(self._dashboard_update_loop()),
            asyncio.create_task(self._real_time_data_processing_loop()),
            asyncio.create_task(self._alert_notification_loop()),
            asyncio.create_task(self._client_connection_monitoring_loop())
        ]

        logger.info("Live Dashboard Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Live Dashboard Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._dashboard_tasks:
            task.cancel()

        await asyncio.gather(*self._dashboard_tasks, return_exceptions=True)
        self._dashboard_tasks.clear()

        logger.info("Live Dashboard Engine gestoppt")

    async def create_dashboard(
        self,
        dashboard_name: str,
        dashboard_config: dict[str, Any],
        security_context: SecurityContext | None = None
    ) -> str:
        """Erstellt neues Live-Dashboard.

        Args:
            dashboard_name: Name des Dashboards
            dashboard_config: Dashboard-Konfiguration
            security_context: Security-Context

        Returns:
            Dashboard-ID
        """
        try:
            import uuid

            dashboard_id = str(uuid.uuid4())

            dashboard_data = LiveDashboardData(
                dashboard_id=dashboard_id,
                dashboard_name=dashboard_name,
                metadata=dashboard_config
            )

            self._dashboard_data[dashboard_id] = dashboard_data

            logger.info({
                "event": "dashboard_created",
                "dashboard_id": dashboard_id,
                "dashboard_name": dashboard_name,
                "user_id": security_context.user_id if security_context else None
            })

            return dashboard_id

        except Exception as e:
            logger.error(f"Dashboard creation fehlgeschlagen: {e}")
            raise

    async def subscribe_to_dashboard(
        self,
        dashboard_id: str,
        client_id: str,
        connection_info: dict[str, Any],
        security_context: SecurityContext | None = None
    ) -> bool:
        """Abonniert Client für Dashboard-Updates.

        Args:
            dashboard_id: Dashboard-ID
            client_id: Client-ID
            connection_info: Connection-Informationen
            security_context: Security-Context

        Returns:
            Erfolg-Status
        """
        try:
            if dashboard_id not in self._dashboard_data:
                logger.warning(f"Dashboard {dashboard_id} nicht gefunden für Subscription")
                return False

            # Füge Client zu Subscriptions hinzu
            self._dashboard_subscriptions[dashboard_id].add(client_id)
            self._client_connections[client_id] = {
                "dashboard_id": dashboard_id,
                "connection_info": connection_info,
                "subscribed_at": datetime.utcnow(),
                "security_context": security_context,
                "last_update": datetime.utcnow()
            }

            # Performance-Tracking
            self._dashboard_performance_stats["total_client_connections"] += 1
            self._dashboard_performance_stats["active_subscriptions"] = len(self._client_connections)

            logger.debug({
                "event": "dashboard_subscription_added",
                "dashboard_id": dashboard_id,
                "client_id": client_id,
                "user_id": security_context.user_id if security_context else None
            })

            return True

        except Exception as e:
            logger.error(f"Dashboard subscription fehlgeschlagen: {e}")
            return False

    async def unsubscribe_from_dashboard(
        self,
        dashboard_id: str,
        client_id: str
    ) -> bool:
        """Entfernt Client-Subscription.

        Args:
            dashboard_id: Dashboard-ID
            client_id: Client-ID

        Returns:
            Erfolg-Status
        """
        try:
            # Entferne Client aus Subscriptions
            if dashboard_id in self._dashboard_subscriptions:
                self._dashboard_subscriptions[dashboard_id].discard(client_id)

            if client_id in self._client_connections:
                del self._client_connections[client_id]

            # Performance-Tracking
            self._dashboard_performance_stats["active_subscriptions"] = len(self._client_connections)

            logger.debug({
                "event": "dashboard_subscription_removed",
                "dashboard_id": dashboard_id,
                "client_id": client_id
            })

            return True

        except Exception as e:
            logger.error(f"Dashboard unsubscription fehlgeschlagen: {e}")
            return False

    async def update_dashboard_data(
        self,
        dashboard_id: str,
        metrics: dict[str, Any],
        alerts: list[MonitoringAlert],
        performance_data: dict[str, PerformanceMetrics]
    ) -> None:
        """Aktualisiert Dashboard-Daten.

        Args:
            dashboard_id: Dashboard-ID
            metrics: Aktuelle Metriken
            alerts: Aktuelle Alerts
            performance_data: Performance-Daten
        """
        start_time = time.time()

        try:
            dashboard = self._dashboard_data.get(dashboard_id)
            if not dashboard:
                logger.warning(f"Dashboard {dashboard_id} nicht gefunden für Update")
                return

            # Update Dashboard-Daten
            dashboard.generated_at = datetime.utcnow()

            # Update System-Health
            dashboard.system_health = self._calculate_system_health(alerts)
            dashboard.active_alerts = len([alert for alert in alerts if alert.status == "active"])

            # Update Performance-Übersicht
            if performance_data:
                all_metrics = list(performance_data.values())
                dashboard.avg_response_time_ms = sum(m.avg_response_time_ms for m in all_metrics) / len(all_metrics)
                dashboard.overall_error_rate = sum(m.error_rate for m in all_metrics) / len(all_metrics)
                dashboard.overall_success_rate = sum(m.success_rate for m in all_metrics) / len(all_metrics)

                # Update Service-Counts
                dashboard.total_services = len(performance_data)
                dashboard.healthy_services = len([
                    m for m in all_metrics
                    if m.error_rate < 0.05  # < 5% Error-Rate = healthy
                ])

            # Update Service-Details
            dashboard.service_metrics = performance_data

            # Speichere Real-time Data
            await self._store_real_time_data(dashboard_id, metrics, alerts, performance_data)

            # Benachrichtige Subscribers
            if self.enable_real_time_updates:
                await self._notify_dashboard_subscribers(dashboard_id, dashboard)

            # Performance-Tracking
            update_time_ms = (time.time() - start_time) * 1000
            self._update_dashboard_performance_stats(update_time_ms)

            logger.debug({
                "event": "dashboard_data_updated",
                "dashboard_id": dashboard_id,
                "system_health": dashboard.system_health,
                "active_alerts": dashboard.active_alerts,
                "total_services": dashboard.total_services,
                "update_time_ms": update_time_ms
            })

        except Exception as e:
            logger.error(f"Dashboard data update fehlgeschlagen: {e}")

    async def get_dashboard_data(
        self,
        dashboard_id: str,
        include_history: bool = False
    ) -> dict[str, Any] | None:
        """Holt Dashboard-Daten.

        Args:
            dashboard_id: Dashboard-ID
            include_history: History einschließen

        Returns:
            Dashboard-Daten oder None
        """
        try:
            dashboard = self._dashboard_data.get(dashboard_id)
            if not dashboard:
                return None

            dashboard_dict = {
                "dashboard_id": dashboard.dashboard_id,
                "dashboard_name": dashboard.dashboard_name,
                "system_health": dashboard.system_health,
                "active_alerts": dashboard.active_alerts,
                "total_services": dashboard.total_services,
                "healthy_services": dashboard.healthy_services,
                "avg_response_time_ms": dashboard.avg_response_time_ms,
                "overall_error_rate": dashboard.overall_error_rate,
                "overall_success_rate": dashboard.overall_success_rate,
                "service_metrics": {
                    service_id: {
                        "avg_response_time_ms": metrics.avg_response_time_ms,
                        "error_rate": metrics.error_rate,
                        "success_rate": metrics.success_rate,
                        "total_requests": metrics.total_requests
                    }
                    for service_id, metrics in dashboard.service_metrics.items()
                },
                "generated_at": dashboard.generated_at.isoformat(),
                "metadata": dashboard.metadata
            }

            if include_history:
                dashboard_dict["history"] = {
                    "metrics": list(self._real_time_metrics.get(dashboard_id, [])),
                    "alerts": list(self._alert_history.get(dashboard_id, [])),
                    "performance": list(self._performance_history.get(dashboard_id, []))
                }

            return dashboard_dict

        except Exception as e:
            logger.error(f"Dashboard data retrieval fehlgeschlagen: {e}")
            return None

    async def get_real_time_metrics(
        self,
        dashboard_id: str,
        metric_names: list[str] | None = None,
        time_range_minutes: int = 60
    ) -> dict[str, list[dict[str, Any]]]:
        """Holt Real-time Metriken.

        Args:
            dashboard_id: Dashboard-ID
            metric_names: Spezifische Metrik-Namen
            time_range_minutes: Zeitraum in Minuten

        Returns:
            Real-time Metriken
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_range_minutes)
            metrics_data = defaultdict(list)

            # Hole Metriken aus History
            for data_point in self._real_time_metrics.get(dashboard_id, []):
                if data_point.get("timestamp"):
                    timestamp = datetime.fromisoformat(data_point["timestamp"])
                    if timestamp >= cutoff_time:
                        for metric_name, metric_value in data_point.get("metrics", {}).items():
                            if not metric_names or metric_name in metric_names:
                                metrics_data[metric_name].append({
                                    "timestamp": data_point["timestamp"],
                                    "value": metric_value
                                })

            return dict(metrics_data)

        except Exception as e:
            logger.error(f"Real-time metrics retrieval fehlgeschlagen: {e}")
            return {}

    async def _store_real_time_data(
        self,
        dashboard_id: str,
        metrics: dict[str, Any],
        alerts: list[MonitoringAlert],
        performance_data: dict[str, PerformanceMetrics]
    ) -> None:
        """Speichert Real-time Daten."""
        try:
            timestamp = datetime.utcnow().isoformat()

            # Speichere Metriken
            metrics_entry = {
                "timestamp": timestamp,
                "metrics": metrics
            }
            self._real_time_metrics[dashboard_id].append(metrics_entry)

            # Speichere Alerts
            for alert in alerts:
                alert_entry = {
                    "timestamp": timestamp,
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "scope": alert.scope.value,
                    "scope_id": alert.scope_id
                }
                self._alert_history[dashboard_id].append(alert_entry)

            # Speichere Performance-Daten
            performance_entry = {
                "timestamp": timestamp,
                "performance_data": {
                    service_id: {
                        "avg_response_time_ms": metrics.avg_response_time_ms,
                        "error_rate": metrics.error_rate,
                        "success_rate": metrics.success_rate
                    }
                    for service_id, metrics in performance_data.items()
                }
            }
            self._performance_history[dashboard_id].append(performance_entry)

            # Performance-Tracking
            self._dashboard_performance_stats["data_points_processed"] += 1

        except Exception as e:
            logger.error(f"Real-time data storage fehlgeschlagen: {e}")

    async def _notify_dashboard_subscribers(
        self,
        dashboard_id: str,
        dashboard_data: LiveDashboardData
    ) -> None:
        """Benachrichtigt Dashboard-Subscribers."""
        try:
            subscribers = self._dashboard_subscriptions.get(dashboard_id, set())

            if not subscribers:
                return

            # Erstelle Update-Message
            update_message = {
                "type": "dashboard_update",
                "dashboard_id": dashboard_id,
                "data": {
                    "system_health": dashboard_data.system_health,
                    "active_alerts": dashboard_data.active_alerts,
                    "total_services": dashboard_data.total_services,
                    "healthy_services": dashboard_data.healthy_services,
                    "avg_response_time_ms": dashboard_data.avg_response_time_ms,
                    "overall_error_rate": dashboard_data.overall_error_rate,
                    "overall_success_rate": dashboard_data.overall_success_rate,
                    "generated_at": dashboard_data.generated_at.isoformat()
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            # Sende an alle Subscribers (simuliert)
            for client_id in subscribers:
                if client_id in self._client_connections:
                    connection = self._client_connections[client_id]
                    connection["last_update"] = datetime.utcnow()

                    # In Realität würde hier WebSocket-Send stattfinden
                    logger.debug({
                        "event": "dashboard_update_sent",
                        "dashboard_id": dashboard_id,
                        "client_id": client_id,
                        "message_type": update_message["type"],
                        "data_size": len(str(update_message["data"]))
                    })

        except Exception as e:
            logger.error(f"Dashboard subscribers notification fehlgeschlagen: {e}")

    async def _dashboard_update_loop(self) -> None:
        """Background-Loop für Dashboard-Updates."""
        while self._is_running:
            try:
                await asyncio.sleep(self.refresh_interval_seconds)

                if self._is_running:
                    await self._refresh_all_dashboards()

            except Exception as e:
                logger.error(f"Dashboard update loop fehlgeschlagen: {e}")
                await asyncio.sleep(self.refresh_interval_seconds)

    async def _real_time_data_processing_loop(self) -> None:
        """Background-Loop für Real-time Data-Processing."""
        while self._is_running:
            try:
                await asyncio.sleep(1)  # Jede Sekunde

                if self._is_running:
                    await self._process_real_time_data()

            except Exception as e:
                logger.error(f"Real-time data processing loop fehlgeschlagen: {e}")
                await asyncio.sleep(1)

    async def _alert_notification_loop(self) -> None:
        """Background-Loop für Alert-Notifications."""
        while self._is_running:
            try:
                await asyncio.sleep(10)  # Alle 10 Sekunden

                if self._is_running and self.enable_alert_notifications:
                    await self._process_alert_notifications()

            except Exception as e:
                logger.error(f"Alert notification loop fehlgeschlagen: {e}")
                await asyncio.sleep(10)

    async def _client_connection_monitoring_loop(self) -> None:
        """Background-Loop für Client-Connection-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Jede Minute

                if self._is_running:
                    await self._monitor_client_connections()

            except Exception as e:
                logger.error(f"Client connection monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _refresh_all_dashboards(self) -> None:
        """Aktualisiert alle Dashboards."""
        try:
            for dashboard_id in self._dashboard_data.keys():
                # Simuliere Dashboard-Refresh
                # In Realität würde hier die echte Datensammlung stattfinden
                logger.debug(f"Refreshing dashboard: {dashboard_id}")

        except Exception as e:
            logger.error(f"All dashboards refresh fehlgeschlagen: {e}")

    async def _process_real_time_data(self) -> None:
        """Verarbeitet Real-time Daten."""
        try:
            # Simuliere Real-time Data-Processing
            # In Realität würde hier die echte Datenverarbeitung stattfinden
            pass

        except Exception as e:
            logger.error(f"Real-time data processing fehlgeschlagen: {e}")

    async def _process_alert_notifications(self) -> None:
        """Verarbeitet Alert-Notifications."""
        try:
            # Prüfe auf neue Alerts in allen Dashboards
            for dashboard_id, alert_history in self._alert_history.items():
                if alert_history:
                    latest_alert = alert_history[-1]

                    # Prüfe ob Alert kritisch ist
                    if latest_alert.get("severity") in ["critical", "emergency"]:
                        await self._send_critical_alert_notification(dashboard_id, latest_alert)

        except Exception as e:
            logger.error(f"Alert notifications processing fehlgeschlagen: {e}")

    async def _send_critical_alert_notification(
        self,
        dashboard_id: str,
        alert_data: dict[str, Any]
    ) -> None:
        """Sendet kritische Alert-Notification."""
        try:
            subscribers = self._dashboard_subscriptions.get(dashboard_id, set())

            notification_message = {
                "type": "critical_alert",
                "dashboard_id": dashboard_id,
                "alert": alert_data,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Sende an alle Subscribers
            for client_id in subscribers:
                if client_id in self._client_connections:
                    logger.warning({
                        "event": "critical_alert_notification_sent",
                        "dashboard_id": dashboard_id,
                        "client_id": client_id,
                        "alert_severity": alert_data.get("severity"),
                        "alert_message": alert_data.get("message"),
                        "notification_type": notification_message["type"]
                    })

        except Exception as e:
            logger.error(f"Critical alert notification fehlgeschlagen: {e}")

    async def _monitor_client_connections(self) -> None:
        """Monitort Client-Connections."""
        try:
            current_time = datetime.utcnow()
            stale_clients = []

            for client_id, connection in self._client_connections.items():
                last_update = connection.get("last_update", current_time)
                age = (current_time - last_update).total_seconds()

                # Entferne stale Connections (> 5 Minuten ohne Update)
                if age > 300:
                    stale_clients.append(client_id)

            # Cleanup stale Clients
            for client_id in stale_clients:
                connection = self._client_connections[client_id]
                dashboard_id = connection.get("dashboard_id")

                if dashboard_id:
                    await self.unsubscribe_from_dashboard(dashboard_id, client_id)

                logger.debug({
                    "event": "stale_client_removed",
                    "client_id": client_id,
                    "age_seconds": age
                })

        except Exception as e:
            logger.error(f"Client connections monitoring fehlgeschlagen: {e}")

    def _calculate_system_health(self, alerts: list[MonitoringAlert]) -> str:
        """Berechnet System-Health basierend auf Alerts."""
        try:
            active_alerts = [alert for alert in alerts if alert.status == "active"]

            if not active_alerts:
                return "healthy"

            critical_alerts = len([
                alert for alert in active_alerts
                if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            ])

            if critical_alerts > 0:
                return "unhealthy"
            if len(active_alerts) > 5:
                return "degraded"
            return "healthy"

        except Exception as e:
            logger.error(f"System health calculation fehlgeschlagen: {e}")
            return "unknown"

    def _update_dashboard_performance_stats(self, update_time_ms: float) -> None:
        """Aktualisiert Dashboard-Performance-Statistiken."""
        try:
            self._dashboard_performance_stats["total_dashboard_updates"] += 1

            current_avg = self._dashboard_performance_stats["avg_dashboard_generation_time_ms"]
            total_count = self._dashboard_performance_stats["total_dashboard_updates"]
            new_avg = ((current_avg * (total_count - 1)) + update_time_ms) / total_count
            self._dashboard_performance_stats["avg_dashboard_generation_time_ms"] = new_avg

        except Exception as e:
            logger.error(f"Dashboard performance stats update fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            stats = self._dashboard_performance_stats.copy()

            # Current State
            stats["total_dashboards"] = len(self._dashboard_data)
            stats["active_subscriptions"] = len(self._client_connections)
            stats["total_subscriptions"] = sum(len(subs) for subs in self._dashboard_subscriptions.values())

            # Configuration
            stats["refresh_interval_seconds"] = self.refresh_interval_seconds
            stats["data_retention_points"] = self.data_retention_points
            stats["real_time_updates_enabled"] = self.enable_real_time_updates
            stats["alert_notifications_enabled"] = self.enable_alert_notifications

            return stats

        except Exception as e:
            logger.error(f"Dashboard performance stats retrieval fehlgeschlagen: {e}")
            return {}
