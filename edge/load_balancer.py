"""Edge Load Balancer - Intelligent routing for edge nodes
"""

import logging
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class EdgeLoadBalancer:
    """Intelligent load balancer for edge nodes"""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.routing_strategy = "adaptive"
        self.request_count = 0
        self._is_running = False

    async def register_node(self, node_info: dict):
        """Register an edge node with the load balancer"""
        node_id = node_info["node_id"]
        self.nodes[node_id] = {
            **node_info,
            "requests_routed": 0,
            "avg_response_time": 0.0,
            "last_used": datetime.utcnow()
        }
        logger.info(f"Node registered with load balancer: {node_id}")

    async def start(self):
        """Start the load balancer service"""
        if self._is_running:
            logger.warning("Load balancer is already running")
            return

        try:
            self._is_running = True
            logger.info("Edge Load Balancer started successfully")
        except Exception as e:
            logger.error(f"Failed to start load balancer: {e}")
            self._is_running = False
            raise

    async def stop(self):
        """Stop the load balancer service"""
        if not self._is_running:
            logger.warning("Load balancer is not running")
            return

        try:
            self._is_running = False
            self.nodes.clear()
            self.request_count = 0
            logger.info("Edge Load Balancer stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop load balancer: {e}")
            raise

    def is_running(self) -> bool:
        """Check if load balancer is running"""
        return self._is_running

    async def route_request(self, task_type: str, requirements: dict = None) -> dict | None:
        """Route request to best available node"""
        suitable_nodes = []

        # Filter nodes by capability
        for node_id, node_info in self.nodes.items():
            if task_type in node_info.get("capabilities", []):
                suitable_nodes.append(node_info)

        if not suitable_nodes:
            logger.warning(f"No suitable nodes found for task type: {task_type}")
            return None

        # Select best node based on strategy
        selected_node = await self._select_node(suitable_nodes)

        if selected_node:
            # Update routing statistics
            selected_node["requests_routed"] += 1
            selected_node["last_used"] = datetime.utcnow()
            self.request_count += 1

            logger.info(f"Routed {task_type} request to node {selected_node['node_id']}")

        return selected_node

    async def _select_node(self, nodes: list[dict]) -> dict | None:
        """Select best node based on routing strategy"""
        if not nodes:
            return None

        if self.routing_strategy == "round_robin":
            return nodes[self.request_count % len(nodes)]

        if self.routing_strategy == "least_connections":
            return min(nodes, key=lambda n: n.get("current_load", 0))

        if self.routing_strategy == "latency_based":
            return min(nodes, key=lambda n: n.get("avg_response_time", float("inf")))

        if self.routing_strategy == "adaptive":
            # Adaptive strategy considering multiple factors
            best_node = None
            best_score = float("inf")

            for node in nodes:
                # Calculate composite score
                load_factor = node.get("current_load", 0) * 0.4
                latency_factor = node.get("avg_response_time", 0) * 0.3
                capacity_factor = (1.0 - node.get("cpu_usage", 0) / 100.0) * 0.2
                reliability_factor = node.get("success_rate", 1.0) * 0.1

                score = load_factor + latency_factor - capacity_factor - reliability_factor

                if score < best_score:
                    best_score = score
                    best_node = node

            return best_node

        # Default to random selection
        return random.choice(nodes)

    async def update_node_metrics(self, node_id: str, metrics: dict):
        """Update node performance metrics"""
        if node_id in self.nodes:
            self.nodes[node_id].update(metrics)
            logger.debug(f"Updated metrics for node {node_id}")

    async def get_routing_stats(self) -> dict:
        """Get load balancer statistics"""
        return {
            "total_requests": self.request_count,
            "active_nodes": len(self.nodes),
            "routing_strategy": self.routing_strategy,
            "nodes": {
                node_id: {
                    "requests_routed": info["requests_routed"],
                    "avg_response_time": info["avg_response_time"],
                    "current_load": info.get("current_load", 0)
                }
                for node_id, info in self.nodes.items()
            }
        }

# Factory function
def create_load_balancer(config=None) -> EdgeLoadBalancer:
    """Create a new EdgeLoadBalancer instance"""
    balancer = EdgeLoadBalancer()
    if config:
        # Apply configuration if provided
        if hasattr(config, "load_balancing_strategy"):
            balancer.routing_strategy = config.load_balancing_strategy
    return balancer

# Global load balancer instance
edge_load_balancer = EdgeLoadBalancer()

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Edge Load Balancer", version="1.0.0")

    @app.post("/register")
    async def register_node(node_info: dict):
        await edge_load_balancer.register_node(node_info)
        return {"success": True}

    @app.post("/route")
    async def route_request(task_type: str, requirements: dict = None):
        node = await edge_load_balancer.route_request(task_type, requirements)
        if node:
            return {"node": node}
        raise HTTPException(status_code=404, detail="No suitable node found")

    @app.post("/metrics/{node_id}")
    async def update_metrics(node_id: str, metrics: dict):
        await edge_load_balancer.update_node_metrics(node_id, metrics)
        return {"success": True}

    @app.get("/stats")
    async def get_stats():
        return await edge_load_balancer.get_routing_stats()

    @app.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": datetime.utcnow()}

    uvicorn.run(app, host="0.0.0.0", port=8088, ws="websockets-sansio")
