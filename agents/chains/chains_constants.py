"""Konstanten für Chain-Module.

Zentrale Definition aller Konfigurationswerte
für konsistente Verwendung in allen Chain-Implementierungen.
"""

from __future__ import annotations

# LLM-Konfiguration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TEMPERATURE_DETERMINISTIC = 0.0

# Text-Verarbeitung
DEFAULT_TEXT_TRUNCATION = 120
DEFAULT_MAX_HISTORY_MESSAGES = 30
DEFAULT_MAX_HISTORY_MESSAGES_QA = 20
DEFAULT_TARGET_SUMMARY_POINTS = 5

# Timeouts und Limits
DEFAULT_TOP_K_RETRIEVAL = 5
DEFAULT_MAX_WORD_COUNT_FOR_SUMMARY = 180

# Prompt-Templates
ANALYSIS_PROMPT_TEMPLATE = """
Du bist ein hilfreicher Assistent. Analysiere die Nutzeranfrage und die Historie knapp.
Nutze stichpunktartige interne Überlegungen, gib sie aber NICHT an den Nutzer aus.

Historie (kurz):
{history}
Anfrage: {input}

Interne Analyse (ein Satz):
""".strip()

ANSWER_PROMPT_TEMPLATE = """
Du bist ein präziser Assistent. Erzeuge eine kurze, klare Antwort auf die Anfrage.
Beziehe relevante Historie mit ein. Antworte in der Sprache der Anfrage.

Historie (kurz):
{history}
Anfrage: {input}
""".strip()

QA_PROMPT_TEMPLATE = """
Du bist ein präziser Enterprise-Assistent. Beantworte die Frage kurz
und zitierfähig basierend auf den bereitgestellten Kontextauszügen.
Antworte in der Sprache der Frage. Nutze KEINE Halluzinationen.

Historie (kurz):
{history}
Frage: {question}
Kontext-Auszüge:
{context}
""".strip()

SUMMARIZATION_PROMPT_TEMPLATE = """
Erstelle eine prägnante Zusammenfassung in {n} Stichpunkten.
Sprache beibehalten, keine Halluzinationen.

Text:
{text}
""".strip()

TRANSFORM_PROMPT_TEMPLATE = """
Wende folgende Instruktion strikt auf den Eingabetext an:
Instruktion: {instruction}

Eingabe:
{input}
""".strip()

# Router-Klassifikation
QA_MARKERS = [
    "wer", "was", "wie", "warum", "wieso", "wo", "wann", "?",
    "explain", "what", "why"
]

SUMMARY_MARKERS = [
    "zusammenfassen", "summary", "fass", "bullet", "stichpunkt", "tl;dr"
]

ROUTER_CLASSIFICATION_PROMPT = (
    "Klassifiziere den folgenden Text als 'QA', 'SUMMARIZE' oder 'TRANSFORM'. "
    "Nur das Wort ausgeben.\n"
)

# Fallback-Nachrichten
ECHO_FALLBACK_PREFIX = "Echo"
FALLBACK_ANSWER_PREFIX = "Antwort (Fallback)"
FALLBACK_SUMMARY_PREFIX = "Zusammenfassung (Fallback)"
FALLBACK_TRANSFORM_PREFIX = "Fallback"
NO_RESULTS_MESSAGE = "Keine Treffer"

__all__ = [
    "ANALYSIS_PROMPT_TEMPLATE",
    "ANSWER_PROMPT_TEMPLATE",
    "DEFAULT_MAX_HISTORY_MESSAGES",
    "DEFAULT_MAX_HISTORY_MESSAGES_QA",
    "DEFAULT_MAX_WORD_COUNT_FOR_SUMMARY",
    "DEFAULT_MODEL",
    "DEFAULT_TARGET_SUMMARY_POINTS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TEMPERATURE_DETERMINISTIC",
    "DEFAULT_TEXT_TRUNCATION",
    "DEFAULT_TOP_K_RETRIEVAL",
    "ECHO_FALLBACK_PREFIX",
    "FALLBACK_ANSWER_PREFIX",
    "FALLBACK_SUMMARY_PREFIX",
    "FALLBACK_TRANSFORM_PREFIX",
    "NO_RESULTS_MESSAGE",
    "QA_MARKERS",
    "QA_PROMPT_TEMPLATE",
    "ROUTER_CLASSIFICATION_PROMPT",
    "SUMMARIZATION_PROMPT_TEMPLATE",
    "SUMMARY_MARKERS",
    "TRANSFORM_PROMPT_TEMPLATE",
]
