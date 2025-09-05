"""
Locust Load-Testing-Konfiguration für KEI-MCP API.

Diese Datei definiert realistische Load-Testing-Szenarien für verschiedene
API-Endpunkte und Benutzerverhalten.
"""

import random
import time

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


# ============================================================================
# LOAD-TESTING-KONFIGURATION
# ============================================================================

class KEIMCPUser(HttpUser):
    """Basis-User-Klasse für KEI-MCP API Load-Tests."""

    wait_time = between(1, 3)  # 1-3 Sekunden zwischen Requests

    def __init__(self, *args, **kwargs):
        """Initialisiert KEI-MCP User."""
        super().__init__(*args, **kwargs)
        # Initialisiere Attribute für Load-Testing
        self.auth_token = "Bearer test-load-token"
        self.headers = {
            "Authorization": self.auth_token,
            "Content-Type": "application/json",
            "User-Agent": "Locust-LoadTest/1.0"
        }

    def on_start(self):
        """Setup für jeden User beim Start."""
        # Registriere Test-Server für Load-Tests
        self.setup_test_servers()

    def setup_test_servers(self):
        """Registriert Test-Server für Load-Tests."""
        test_servers = [
            {
                "server_name": f"load-test-server-{i}",
                "base_url": f"https://test-server-{i}.example.com",
                "api_key": f"test-api-key-{i}",
                "timeout_seconds": 30
            }
            for i in range(1, 4)  # 3 Test-Server
        ]
        
        for server_config in test_servers:
            response = self.client.post(
                "/api/v1/mcp/external/servers/register",
                json=server_config,
                headers=self.headers,
                name="setup_server_registration"
            )
            if response.status_code != 200:
                print(f"Failed to register server {server_config['server_name']}: {response.status_code}")


class ToolInvocationUser(KEIMCPUser):
    """User-Klasse für Tool-Invocation Load-Tests."""
    
    weight = 3  # 30% der User führen Tool-Invocations aus
    
    @task(5)
    def invoke_simple_tool(self):
        """Führt einfache Tool-Invocation aus."""
        server_name = f"load-test-server-{random.randint(1, 3)}"
        
        payload = {
            "server_name": server_name,
            "tool_name": "simple_tool",
            "parameters": {
                "input": f"test_input_{random.randint(1, 1000)}",
                "count": random.randint(1, 100)
            }
        }
        
        with self.client.post(
            "/api/v1/mcp/external/tools/invoke",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="tool_invocation_simple"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 422:
                # Validierungsfehler sind bei Load-Tests akzeptabel
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(3)
    def invoke_complex_tool(self):
        """Führt komplexe Tool-Invocation mit großen Parametern aus."""
        server_name = f"load-test-server-{random.randint(1, 3)}"
        
        # Größere Parameter-Payload
        payload = {
            "server_name": server_name,
            "tool_name": "complex_tool",
            "parameters": {
                "large_text": "Lorem ipsum " * 100,  # ~1.2KB Text
                "data_array": [
                    {"id": i, "value": f"item_{i}", "metadata": {"type": "test"}}
                    for i in range(50)
                ],
                "configuration": {
                    "timeout": random.randint(5, 30),
                    "retries": random.randint(1, 5),
                    "options": {
                        f"option_{j}": random.choice(["enabled", "disabled"])
                        for j in range(10)
                    }
                }
            }
        }
        
        with self.client.post(
            "/api/v1/mcp/external/tools/invoke",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="tool_invocation_complex"
        ) as response:
            if response.status_code in [200, 422]:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def invoke_slow_tool(self):
        """Simuliert langsame Tool-Invocation."""
        server_name = f"load-test-server-{random.randint(1, 3)}"
        
        payload = {
            "server_name": server_name,
            "tool_name": "slow_tool",
            "parameters": {
                "delay_seconds": random.randint(1, 5),
                "operation": "heavy_computation"
            }
        }
        
        with self.client.post(
            "/api/v1/mcp/external/tools/invoke",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="tool_invocation_slow"
        ) as response:
            if response.status_code in [200, 408, 422]:  # 408 = Timeout
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


class ResourceAccessUser(KEIMCPUser):
    """User-Klasse für Resource-Access Load-Tests."""
    
    weight = 2  # 20% der User greifen auf Resources zu
    
    @task(4)
    def list_resources(self):
        """Listet verfügbare Resources auf."""
        server_name = f"load-test-server-{random.randint(1, 3)}"
        
        with self.client.get(
            "/api/v1/mcp/external/resources",
            params={"server_name": server_name},
            headers=self.headers,
            catch_response=True,
            name="resource_list"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(6)
    def access_resource(self):
        """Greift auf einzelne Resource zu."""
        server_name = f"load-test-server-{random.randint(1, 3)}"
        resource_id = f"resource-{random.randint(1, 100)}"
        
        with self.client.get(
            f"/api/v1/mcp/external/resources/{server_name}/{resource_id}",
            headers=self.headers,
            catch_response=True,
            name="resource_access"
        ) as response:
            if response.status_code in [200, 404]:  # 404 ist bei zufälligen IDs normal
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(2)
    def access_large_resource(self):
        """Greift auf große Resource zu."""
        server_name = f"load-test-server-{random.randint(1, 3)}"
        
        with self.client.get(
            f"/api/v1/mcp/external/resources/{server_name}/large-dataset",
            headers=self.headers,
            catch_response=True,
            name="resource_access_large"
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


class DiscoveryUser(KEIMCPUser):
    """User-Klasse für Discovery-Operations Load-Tests."""
    
    weight = 2  # 20% der User führen Discovery-Operations aus
    
    @task(3)
    def discover_tools(self):
        """Entdeckt verfügbare Tools."""
        server_name = f"load-test-server-{random.randint(1, 3)}"
        
        with self.client.get(
            "/api/v1/mcp/external/tools",
            params={"server_name": server_name},
            headers=self.headers,
            catch_response=True,
            name="tool_discovery"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(2)
    def discover_prompts(self):
        """Entdeckt verfügbare Prompts."""
        server_name = f"load-test-server-{random.randint(1, 3)}"
        
        with self.client.get(
            "/api/v1/mcp/external/prompts",
            params={"server_name": server_name},
            headers=self.headers,
            catch_response=True,
            name="prompt_discovery"
        ) as response:
            if response.status_code in [200, 404]:  # 404 wenn keine Prompts verfügbar
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def list_servers(self):
        """Listet registrierte Server auf."""
        with self.client.get(
            "/api/v1/mcp/external/servers",
            headers=self.headers,
            catch_response=True,
            name="server_list"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


class ManagementUser(KEIMCPUser):
    """User-Klasse für Management-Operations Load-Tests."""
    
    weight = 1  # 10% der User führen Management-Operations aus
    
    @task(2)
    def check_server_health(self):
        """Prüft Server-Health."""
        server_name = f"load-test-server-{random.randint(1, 3)}"
        
        with self.client.get(
            f"/api/v1/mcp/external/servers/{server_name}/stats",
            headers=self.headers,
            catch_response=True,
            name="server_health_check"
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def get_metrics(self):
        """Ruft Prometheus-Metriken ab."""
        with self.client.get(
            "/metrics",
            headers={"Accept": "text/plain"},
            catch_response=True,
            name="metrics_access"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def check_circuit_breakers(self):
        """Prüft Circuit-Breaker-Status."""
        with self.client.get(
            "/circuit-breakers",
            headers=self.headers,
            catch_response=True,
            name="circuit_breaker_status"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


class MixedWorkloadUser(KEIMCPUser):
    """User-Klasse für realistische Mixed-Workload-Tests."""
    
    weight = 2  # 20% der User führen gemischte Workloads aus
    
    @task
    def realistic_user_session(self):
        """Simuliert realistische User-Session."""
        # 1. Server-Liste abrufen
        self.client.get(
            "/api/v1/mcp/external/servers",
            headers=self.headers,
            name="session_list_servers"
        )
        
        # Kurze Pause
        time.sleep(random.uniform(0.5, 1.5))
        
        # 2. Tools für zufälligen Server entdecken
        server_name = f"load-test-server-{random.randint(1, 3)}"
        self.client.get(
            "/api/v1/mcp/external/tools",
            params={"server_name": server_name},
            headers=self.headers,
            name="session_discover_tools"
        )
        
        # Kurze Pause
        time.sleep(random.uniform(0.2, 0.8))
        
        # 3. Tool ausführen
        payload = {
            "server_name": server_name,
            "tool_name": "session_tool",
            "parameters": {
                "session_id": f"session_{random.randint(1000, 9999)}",
                "action": random.choice(["read", "write", "process", "analyze"])
            }
        }
        
        self.client.post(
            "/api/v1/mcp/external/tools/invoke",
            json=payload,
            headers=self.headers,
            name="session_invoke_tool"
        )
        
        # Längere Pause zwischen Sessions
        time.sleep(random.uniform(2, 5))


# ============================================================================
# LOAD-TESTING-EVENTS UND METRIKEN
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Event-Handler für Test-Start."""
    print("🚀 KEI-MCP Load-Test gestartet")
    print(f"Target Host: {environment.host}")
    
    if isinstance(environment.runner, MasterRunner):
        print(f"Master-Runner mit {environment.runner.worker_count} Workers")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Event-Handler für Test-Ende."""
    print("🏁 KEI-MCP Load-Test beendet")
    
    # Performance-Metriken ausgeben
    stats = environment.runner.stats
    
    print("\n📊 Performance-Zusammenfassung:")
    print(f"Total Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min Response Time: {stats.total.min_response_time:.2f}ms")
    print(f"Max Response Time: {stats.total.max_response_time:.2f}ms")
    print(f"Requests per Second: {stats.total.current_rps:.2f}")
    
    # SLA-Validierung
    print("\n🎯 SLA-Validierung:")
    
    critical_endpoints = [
        ("tool_invocation_simple", 200),  # P95 < 200ms
        ("resource_list", 100),           # P95 < 100ms
        ("tool_discovery", 150),          # P95 < 150ms
    ]
    
    for endpoint_name, sla_target in critical_endpoints:
        if endpoint_name in stats.entries:
            endpoint_stats = stats.entries[endpoint_name]
            p95_time = endpoint_stats.get_response_time_percentile(0.95)
            
            status = "✅ PASS" if p95_time <= sla_target else "❌ FAIL"
            print(f"{endpoint_name}: P95={p95_time:.1f}ms (Target: {sla_target}ms) {status}")


# ============================================================================
# CUSTOM LOAD-TESTING-SHAPES
# ============================================================================

from locust import LoadTestShape

class StepLoadShape(LoadTestShape):
    """
    Stufenweise Lasterhöhung für Performance-Testing.
    
    Erhöht die Last schrittweise um Performance-Grenzen zu identifizieren.
    """
    
    step_time = 60  # 60 Sekunden pro Stufe
    step_load = 50   # 50 User pro Stufe
    spawn_rate = 10  # 10 User pro Sekunde spawnen
    time_limit = 600 # 10 Minuten Gesamtzeit
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = run_time // self.step_time
        return current_step * self.step_load, self.spawn_rate


class SpikeLoadShape(LoadTestShape):
    """
    Spike-Load-Testing für Stress-Tests.
    
    Simuliert plötzliche Lastspitzen.
    """
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < 60:
            return 10, 2  # Warm-up: 10 User
        elif run_time < 120:
            return 200, 50  # Spike: 200 User schnell
        elif run_time < 180:
            return 50, 10   # Cool-down: 50 User
        elif run_time < 240:
            return 500, 100 # Größerer Spike: 500 User
        elif run_time < 300:
            return 25, 5    # Final cool-down
        else:
            return None
