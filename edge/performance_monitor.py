"""Edge Performance Monitor - Monitoring and alerting for edge infrastructure
"""

import asyncio
import logging
from datetime import datetime, timedelta

import psutil

logger = logging.getLogger(__name__)

class EdgePerformanceMonitor:
    """Performance monitor for edge computing infrastructure"""

    def __init__(self):
        self.metrics: dict[str, list[dict]] = {}
        self.alerts: list[dict] = []
        self.thresholds = {
            "latency_ms": 100,
            "cpu_percent": 80,
            "memory_percent": 85,
            "error_rate": 0.05
        }
        self.monitoring_active = False
        self._is_running = False

    async def start(self):
        """Start the performance monitor service"""
        if self._is_running:
            logger.warning("Performance monitor is already running")
            return

        try:
            self._is_running = True
            await self.start_monitoring()
            logger.info("Edge Performance Monitor started successfully")
        except Exception as e:
            logger.error(f"Failed to start performance monitor: {e}")
            self._is_running = False
            raise

    async def stop(self):
        """Stop the performance monitor service"""
        if not self._is_running:
            logger.warning("Performance monitor is not running")
            return

        try:
            await self.stop_monitoring()
            self._is_running = False
            logger.info("Edge Performance Monitor stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop performance monitor: {e}")
            raise

    def is_running(self) -> bool:
        """Check if performance monitor is running"""
        return self._is_running

    async def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        logger.info("Edge performance monitoring started")

        # Start background monitoring tasks
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._evaluate_alerts())

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("Edge performance monitoring stopped")

    async def record_metric(self, node_id: str, metric_type: str, value: float, metadata: dict = None):
        """Record a performance metric"""
        if node_id not in self.metrics:
            self.metrics[node_id] = []

        metric = {
            "timestamp": datetime.utcnow(),
            "metric_type": metric_type,
            "value": value,
            "metadata": metadata or {}
        }

        self.metrics[node_id].append(metric)

        # Keep only last 1000 metrics per node
        if len(self.metrics[node_id]) > 1000:
            self.metrics[node_id] = self.metrics[node_id][-1000:]

        # Check for threshold violations
        await self._check_thresholds(node_id, metric_type, value)

    async def _check_thresholds(self, node_id: str, metric_type: str, value: float):
        """Check if metric violates thresholds"""
        threshold_key = metric_type.lower()
        if threshold_key in self.thresholds:
            threshold = self.thresholds[threshold_key]

            if value > threshold:
                alert = {
                    "timestamp": datetime.utcnow(),
                    "node_id": node_id,
                    "metric_type": metric_type,
                    "value": value,
                    "threshold": threshold,
                    "severity": "warning" if value < threshold * 1.2 else "critical",
                    "message": f"{metric_type} on {node_id} is {value} (threshold: {threshold})"
                }

                self.alerts.append(alert)
                logger.warning(f"Threshold violation: {alert['message']}")

                # Keep only last 100 alerts
                if len(self.alerts) > 100:
                    self.alerts = self.alerts[-100:]

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()

                await self.record_metric(
                    "system",
                    "cpu_percent",
                    cpu_percent,
                    {"cores": psutil.cpu_count()}
                )

                await self.record_metric(
                    "system",
                    "memory_percent",
                    memory.percent,
                    {"total_gb": memory.total / (1024**3)}
                )

                await asyncio.sleep(10)  # Collect every 10 seconds

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(10)

    async def _evaluate_alerts(self):
        """Evaluate alert conditions"""
        while self.monitoring_active:
            try:
                # Check for node health issues
                current_time = datetime.utcnow()

                for node_id, metrics in self.metrics.items():
                    if not metrics:
                        continue

                    # Check if node has been silent for too long
                    last_metric_time = max(m["timestamp"] for m in metrics)
                    if (current_time - last_metric_time) > timedelta(minutes=5):
                        alert = {
                            "timestamp": current_time,
                            "node_id": node_id,
                            "metric_type": "node_silence",
                            "value": (current_time - last_metric_time).total_seconds(),
                            "threshold": 300,  # 5 minutes
                            "severity": "critical",
                            "message": f"Node {node_id} has been silent for {current_time - last_metric_time}"
                        }

                        self.alerts.append(alert)
                        logger.critical(f"Node silence alert: {alert['message']}")

                await asyncio.sleep(30)  # Evaluate every 30 seconds

            except Exception as e:
                logger.error(f"Error evaluating alerts: {e}")
                await asyncio.sleep(30)

    async def get_metrics(self, node_id: str = None, metric_type: str = None,
                         since: datetime = None) -> list[dict]:
        """Get performance metrics"""
        results = []

        # Filter by time
        if since is None:
            since = datetime.utcnow() - timedelta(hours=1)

        # Get metrics for specific node or all nodes
        nodes_to_check = [node_id] if node_id else self.metrics.keys()

        for nid in nodes_to_check:
            if nid not in self.metrics:
                continue

            for metric in self.metrics[nid]:
                if metric["timestamp"] < since:
                    continue

                if metric_type and metric["metric_type"] != metric_type:
                    continue

                results.append({
                    "node_id": nid,
                    **metric
                })

        return sorted(results, key=lambda x: x["timestamp"])

    async def get_alerts(self, severity: str = None, since: datetime = None) -> list[dict]:
        """Get alerts"""
        if since is None:
            since = datetime.utcnow() - timedelta(hours=24)

        results = []
        for alert in self.alerts:
            if alert["timestamp"] < since:
                continue

            if severity and alert["severity"] != severity:
                continue

            results.append(alert)

        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    async def get_summary(self) -> dict:
        """Get monitoring summary"""
        current_time = datetime.utcnow()
        one_hour_ago = current_time - timedelta(hours=1)

        # Count metrics in last hour
        recent_metrics = await self.get_metrics(since=one_hour_ago)
        recent_alerts = await self.get_alerts(since=one_hour_ago)

        return {
            "monitoring_active": self.monitoring_active,
            "total_nodes": len(self.metrics),
            "metrics_last_hour": len(recent_metrics),
            "alerts_last_hour": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a["severity"] == "critical"]),
            "thresholds": self.thresholds,
            "timestamp": current_time
        }

# Factory function
def create_performance_monitor(config=None) -> EdgePerformanceMonitor:
    """Create a new EdgePerformanceMonitor instance"""
    monitor = EdgePerformanceMonitor()
    if config:
        # Apply configuration if provided
        if hasattr(config, "thresholds"):
            monitor.thresholds.update(config.thresholds)
    return monitor

# Global performance monitor instance
edge_performance_monitor = EdgePerformanceMonitor()

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI, Query

    app = FastAPI(title="Edge Performance Monitor", version="1.0.0")

    @app.on_event("startup")
    async def startup():
        await edge_performance_monitor.start_monitoring()

    @app.on_event("shutdown")
    async def shutdown():
        await edge_performance_monitor.stop_monitoring()

    @app.post("/metrics")
    async def record_metric(node_id: str, metric_type: str, value: float, metadata: dict = None):
        await edge_performance_monitor.record_metric(node_id, metric_type, value, metadata)
        return {"success": True}

    @app.get("/metrics")
    async def get_metrics(
        node_id: str = Query(None),
        metric_type: str = Query(None),
        hours: int = Query(1)
    ):
        since = datetime.utcnow() - timedelta(hours=hours)
        metrics = await edge_performance_monitor.get_metrics(node_id, metric_type, since)
        return {"metrics": metrics}

    @app.get("/alerts")
    async def get_alerts(
        severity: str = Query(None),
        hours: int = Query(24)
    ):
        since = datetime.utcnow() - timedelta(hours=hours)
        alerts = await edge_performance_monitor.get_alerts(severity, since)
        return {"alerts": alerts}

    @app.get("/summary")
    async def get_summary():
        return await edge_performance_monitor.get_summary()

    @app.get("/health")
    async def health():
        summary = await edge_performance_monitor.get_summary()
        return {"status": "healthy" if summary["monitoring_active"] else "unhealthy"}

    uvicorn.run(app, host="0.0.0.0", port=8090, ws="websockets-sansio")
