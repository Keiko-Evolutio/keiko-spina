"""Edge Registry - Central registry for edge nodes
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EdgeRegistry:
    """Central registry for managing edge nodes"""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.last_heartbeat: dict[str, datetime] = {}
        self.heartbeat_timeout = timedelta(minutes=2)

    async def register_node(self, node_id: str, node_info: dict) -> bool:
        """Register a new edge node"""
        try:
            self.nodes[node_id] = {
                **node_info,
                "registered_at": datetime.utcnow(),
                "status": "active"
            }
            self.last_heartbeat[node_id] = datetime.utcnow()
            logger.info(f"Edge node registered: {node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")
            return False

    async def heartbeat(self, node_id: str) -> bool:
        """Process heartbeat from edge node"""
        if node_id in self.nodes:
            self.last_heartbeat[node_id] = datetime.utcnow()
            self.nodes[node_id]["status"] = "active"
            return True
        return False

    async def get_healthy_nodes(self) -> list[dict]:
        """Get list of healthy edge nodes"""
        healthy_nodes = []
        current_time = datetime.utcnow()

        for node_id, node_info in self.nodes.items():
            last_heartbeat = self.last_heartbeat.get(node_id)
            if last_heartbeat and (current_time - last_heartbeat) < self.heartbeat_timeout:
                healthy_nodes.append({
                    "node_id": node_id,
                    **node_info
                })

        return healthy_nodes

    async def get_node_by_capability(self, capability: str) -> dict | None:
        """Get best node for specific capability"""
        healthy_nodes = await self.get_healthy_nodes()

        for node in healthy_nodes:
            if capability in node.get("capabilities", []):
                return node

        return None

# Global registry instance
edge_registry = EdgeRegistry()

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="Edge Registry", version="1.0.0")

    @app.post("/register")
    async def register_node(node_id: str, node_info: dict):
        success = await edge_registry.register_node(node_id, node_info)
        return {"success": success}

    @app.post("/heartbeat/{node_id}")
    async def heartbeat(node_id: str):
        success = await edge_registry.heartbeat(node_id)
        return {"success": success}

    @app.get("/nodes")
    async def get_nodes():
        nodes = await edge_registry.get_healthy_nodes()
        return {"nodes": nodes}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": datetime.utcnow()}

    uvicorn.run(app, host="0.0.0.0", port=8080, ws="websockets-sansio")
