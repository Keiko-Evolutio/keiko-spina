"""Konstanten für Workflow-Module.

Dieses Modul definiert alle Magic Numbers, Hard-coded Strings und
Konfigurationswerte, die in den Workflow-Modulen verwendet werden.
"""


# =============================================================================
# Workflow-Konfiguration
# =============================================================================

DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_DYNAMIC_NODES = 10
DEFAULT_MERMAID_THEME = "default"
DEFAULT_N8N_MODE = "rest"

# Photo Workflow Konfiguration
DEFAULT_PHOTO_MAX_ATTEMPTS = 3

# Tuple-Längen für Validierung
EDGE_TUPLE_LENGTH = 2
TASK_TUPLE_LENGTH = 2

# =============================================================================
# Workflow-Node-Namen
# =============================================================================

WORKFLOW_NODES: dict[str, str] = {
    "ANALYZE_INTENT": "analyze_intent",
    "ROUTE": "route",
    "INVOKE_AGENT": "invoke_agent",
    "END": "end",
    "HUMAN_REVIEW": "human_review",
    "PARALLEL_INVOKE": "parallel_invoke",
    "AGGREGATE": "aggregate",
    "RETRY_NODE": "retry_node",
    "LOOP_DECIDE": "loop_decide",
    "POST_PROCESS": "post_process",
    "DECIDE": "decide",
    "SUBGRAPH_AGGREGATE": "subgraph_aggregate",
    "DYNAMIC_NOOP": "dynamic_noop",
    # Photo Workflow Nodes
    "PHOTO_DETECT": "detect",
    "PHOTO_ASK_READY": "ask_ready",
    "PHOTO_CAPTURE": "capture",
    "PHOTO_AWAIT_DECISION": "await_decision",
    "PHOTO_FINISH": "finish",
}

# =============================================================================
# Branch-Namen
# =============================================================================

BRANCH_NAMES: dict[str, str] = {
    "AUTO": "auto",
    "HUMAN": "human",
    "APPROVED": "approved",
    "ITERATE": "iterate",
    "DONE": "done",
    "RETRY": "retry",
    "GIVEUP": "giveup",
    "WAIT": "wait",
    "EXIT": "exit",
    "ROUTE": "route",
    # Photo Workflow Decisions
    "PHOTO_READY": "ready",
    "PHOTO_OK": "ok",
    "PHOTO_RETAKE": "retake",
    "PHOTO_ABORT": "abort",
}

# =============================================================================
# Error-Messages
# =============================================================================

ERROR_MESSAGES: dict[str, str] = {
    "NO_AGENT_FOUND": "no_agent_found",
    "NO_WORKFLOW": "no_workflow",
    "DYNAMIC_INVOKE_FAILED": "dynamic_invoke_failed",
    "LANGGRAPH_NOT_INSTALLED": "LangGraph nicht installiert",
    "REGISTRY_START_FAILED": "Registry-Start fehlgeschlagen",
    "AGENT_FILTER_FAILED": "Agent-Filter fehlgeschlagen",
    "NO_MATCHING_AGENTS": "no_matching_agents",
    "CANNOT_RUN_COROUTINE": "Kann Coroutine nicht im laufenden Event Loop ausführen",
}

# =============================================================================
# Node-Typen für Visualisierung
# =============================================================================

NODE_TYPES: dict[str, str] = {
    "START": "start",
    "END": "end",
    "TASK": "task",
    "DECISION": "decision",
    "PARALLEL": "parallel",
    "HUMAN": "human",
    "RETRY": "retry",
    "LOOP": "loop",
    "ROUTER": "router",
}

# =============================================================================
# Visualisierung-Konfiguration (DOT)
# =============================================================================

DOT_COLORS: dict[str, tuple[str, str]] = {
    "start": ("circle", "#E3F2FD"),
    "end": ("doublecircle", "#E8F5E9"),
    "task": ("box", "#FFFDE7"),
    "decision": ("diamond", "#F3E5F5"),
    "parallel": ("hexagon", "#E0F7FA"),
    "human": ("oval", "#FFECB3"),
    "retry": ("box", "#FFEBEE"),
    "loop": ("box", "#EDE7F6"),
    "router": ("parallelogram", "#E1F5FE"),
}

# Fallback-Farbe für unbekannte Node-Typen
DOT_FALLBACK_COLOR: tuple[str, str] = ("box", "#FFFFFF")

# =============================================================================
# Mermaid-Konfiguration
# =============================================================================

MERMAID_FLOWCHART_DIRECTION = "LR"
MERMAID_CDN_URL = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"

# =============================================================================
# HTML-Template-Konfiguration
# =============================================================================

HTML_TEMPLATE_TITLE = "Keiko Workflow Preview"
HTML_TEMPLATE_CHARSET = "utf-8"
HTML_TEMPLATE_VIEWPORT = "width=device-width, initial-scale=1"
HTML_TEMPLATE_MAX_WIDTH = "1200px"

# =============================================================================
# Capability-Mappings
# =============================================================================

DEFAULT_CAPABILITIES: list[str] = ["assistant"]
PHOTO_CAPABILITIES: list[str] = ["camera", "photo"]
IMAGE_CAPABILITIES: list[str] = ["image_generation"]

# =============================================================================
# State-Manipulation-Keys
# =============================================================================

STATE_KEYS: dict[str, str] = {
    "EXTRAS": "extras",
    "ERROR": "error",
    "RETRY_COUNT": "retry_count",
    "BRANCH": "branch",
    "MESSAGE": "message",
    "LAST_OUTPUT": "last_output",
    "TARGET_AGENT_ID": "target_agent_id",
    "HUMAN_REQUIRED": "human_required",
    "HUMAN_FEEDBACK": "human_feedback",
    "PARALLEL_RESULTS": "parallel_results",
    "REQUIRED_CAPABILITIES": "required_capabilities",
    "PARALLEL_TARGETS": "parallel_targets",
    "SHOULD_ITERATE": "should_iterate",
    "DYNAMIC_FALLBACK": "dynamic_fallback",
    "DYNAMIC_REASON": "dynamic_reason",
}

# =============================================================================
# Node-ID-Präfixe
# =============================================================================

NODE_PREFIXES: dict[str, str] = {
    "INVOKE": "invoke__",
    "RESULT": "result__",
    "RETRY_DECIDE": "subgraph_retry_decide__",
}

# =============================================================================
# Heuristik-Keywords für Node-Typ-Inferenz
# =============================================================================

NODE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "start": ["start", "entry"],
    "end": ["end", "finish", "exit"],
    "router": ["route", "router"],
    "human": ["human", "review"],
    "retry": ["retry", "error"],
    "loop": ["loop", "iterate"],
    "decision": ["cond", "branch", "decision"],
    "parallel": ["parallel", "fanout"],
}

# =============================================================================
# Photo Workflow Keywords
# =============================================================================

PHOTO_INTENT_KEYWORDS: list[str] = [
    "photo", "picture", "fotograf", "bild", "foto", "photograph",
    "selfie", "aufnahme", "knips", "schieß", "capture", "snap"
]

# =============================================================================
# Regex-Patterns
# =============================================================================

WORKFLOW_NAME_PATTERN = r"^[a-zA-Z0-9_]+$"

# =============================================================================
# Performance-Limits
# =============================================================================

MAX_NODES_FOR_VISUALIZATION = 100
MAX_EDGES_FOR_VISUALIZATION = 200
MAX_HTML_TEMPLATE_SIZE = 1024 * 1024  # 1MB

# =============================================================================
# Logging-Kategorien
# =============================================================================

LOG_CATEGORIES: dict[str, str] = {
    "WORKFLOW_BUILD": "workflow.build",
    "WORKFLOW_VISUALIZE_DOT": "workflow.visualize.dot",
    "WORKFLOW_VISUALIZE_MERMAID": "workflow.visualize.mermaid",
    "WORKFLOW_VISUALIZE_HTML": "workflow.visualize.html",
    "WORKFLOW_DECIDE": "workflow.decide",
    "WORKFLOW_HUMAN_REVIEW": "workflow.human_review",
    "WORKFLOW_PARALLEL_INVOKE": "workflow.parallel_invoke",
    "WORKFLOW_AGGREGATE": "workflow.aggregate",
    "WORKFLOW_POST_PROCESS": "workflow.post_process",
    "AGENTS_N8N_BRIDGE_SELECT": "agents.n8n_bridge.select_workflow",
    "AGENTS_N8N_BRIDGE_TRIGGER": "agents.n8n_bridge.trigger",
    "AGENTS_N8N_BRIDGE_POLL": "agents.n8n_bridge.poll_status",
}

# =============================================================================
# Dependency-Namen
# =============================================================================

DEPENDENCIES: dict[str, str] = {
    "LANGGRAPH": "langgraph",
    "CORE_EXCEPTIONS": "core.exceptions",
    "AGENTS_OPERATIONS": "agents.common.operations",
    "AGENTS_INTENT": "agents.orchestrator.intent_recognition",
}

# =============================================================================
# Pragma-Kommentare
# =============================================================================

PRAGMA_COMMENTS: dict[str, str] = {
    "NO_COVER_OPTIONAL": "# pragma: no cover - optional zur Laufzeit",
    "NO_COVER_TEST": "# pragma: no cover - test/runtime ohne langgraph",
    "NO_COVER_DEFENSIV": "# pragma: no cover - defensiv",
    "NO_COVER": "# pragma: no cover",
}

__all__ = [
    "BRANCH_NAMES",
    # Capabilities
    "DEFAULT_CAPABILITIES",
    "DEFAULT_MAX_DYNAMIC_NODES",
    # Konfiguration
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MERMAID_THEME",
    "DEFAULT_N8N_MODE",
    "DEFAULT_PHOTO_MAX_ATTEMPTS",
    # Dependencies
    "DEPENDENCIES",
    # Visualisierung
    "DOT_COLORS",
    "DOT_FALLBACK_COLOR",
    "EDGE_TUPLE_LENGTH",
    # Error-Messages
    "ERROR_MESSAGES",
    "HTML_TEMPLATE_CHARSET",
    "HTML_TEMPLATE_MAX_WIDTH",
    # HTML-Template
    "HTML_TEMPLATE_TITLE",
    "HTML_TEMPLATE_VIEWPORT",
    "IMAGE_CAPABILITIES",
    # Logging
    "LOG_CATEGORIES",
    "MAX_EDGES_FOR_VISUALIZATION",
    "MAX_HTML_TEMPLATE_SIZE",
    # Performance
    "MAX_NODES_FOR_VISUALIZATION",
    "MERMAID_CDN_URL",
    "MERMAID_FLOWCHART_DIRECTION",
    "NODE_PREFIXES",
    "NODE_TYPES",
    # Heuristik
    "NODE_TYPE_KEYWORDS",
    "PHOTO_CAPABILITIES",
    "PHOTO_INTENT_KEYWORDS",
    # Pragma
    "PRAGMA_COMMENTS",
    # State-Keys
    "STATE_KEYS",
    "TASK_TUPLE_LENGTH",
    "WORKFLOW_NAME_PATTERN",
    # Node-Namen und Branches
    "WORKFLOW_NODES",
]
