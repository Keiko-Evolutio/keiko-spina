"""Retrieval-basierte Q&A Chain.

Kombiniert asynchronen Retriever mit optionalem Chat-Memory
und LLM, fällt bei fehlenden Abhängigkeiten robust auf
deterministische Antworten zurück.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

from .chains_base import BaseKeikoChain
from .chains_common import (
    LANGCHAIN_CORE_AVAILABLE,
    LANGCHAIN_OPENAI_AVAILABLE,
    AsyncRetriever,
    BaseChainConfig,
    ChatMemory,
    PromptTemplate,
    Runnable,
    RunnableLambda,
    StrOutputParser,
    create_retrieval_fallback,
    create_safe_llm,
)
from .chains_constants import (
    DEFAULT_MAX_HISTORY_MESSAGES_QA,
    DEFAULT_TOP_K_RETRIEVAL,
    QA_PROMPT_TEMPLATE,
)
from .chains_utils import (
    create_fallback_runnable,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievalQAConfig(BaseChainConfig):
    """Konfiguration für Retrieval-Q&A Chain."""

    top_k: int = DEFAULT_TOP_K_RETRIEVAL
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES_QA


class KeikoRetrievalQAChain(BaseKeikoChain):
    """Retrieval-basierte Frage-Antwort-Kette für Enterprise-Use-Cases.

    Nutzt optional Chat-Memory (Cosmos) und OpenAI-kompatibles LLM. Bei fehlenden
    Abhängigkeiten liefert die Kette eine deterministische Antwort, die die
    besten Treffer extrahiert und zurückgibt.
    """

    def __init__(
        self,
        *,
        config: RetrievalQAConfig,
        retriever: AsyncRetriever,
        memory: ChatMemory | None = None,
    ) -> None:
        """Initialisiert die Retrieval-Q&A Kette.

        Args:
            config: Kettenkonfiguration
            retriever: Asynchroner Retriever
            memory: Optionaler Chat-Speicher
        """
        super().__init__(config=config, memory=memory)
        self._retriever = retriever

    def _get_trace_name(self) -> str:
        """Gibt Trace-Name zurück."""
        return "keiko_chains.retrieval_qa.ainvoke"

    def _create_fallback_response(self, input_data: Any) -> str:
        """Erstellt Retrieval-Fallback-Antwort."""
        if isinstance(input_data, dict) and "docs" in input_data:
            return create_retrieval_fallback(input_data["docs"])
        return create_retrieval_fallback([])

    async def _safe_retrieve_docs(self, question: str) -> list[dict[str, Any]]:
        """Sicher Dokumente abrufen mit Error-Handling."""
        try:
            return await self._retriever.aretrieve(question, top_k=self._config.top_k)
        except Exception as exc:  # pragma: no cover - defensiv
            logger.warning(f"Retriever-Fehler, Fallback aktiv: {exc}")
            return []

    async def ainvoke(self, question: str) -> str:
        """Beantwortet eine Frage basierend auf abgerufenen Kontextdokumenten."""
        # Nutze Tracing-Mixin für einheitliches Tracing
        traced_method = self._create_traced_method(self._execute_retrieval_qa)
        return await traced_method(question)

    async def _execute_retrieval_qa(self, question: str) -> str:
        """Führt die Retrieval-QA-Logik aus.

        Args:
            question: Frage des Benutzers.

        Returns:
            Antwort basierend auf abgerufenen Dokumenten.
        """
        docs = await self._safe_retrieve_docs(question)
        history = await self._load_history_for_session(
            self._config.session_id,
            self._config.max_history_messages,
        )

        runnable_params = {
            "question": question,
            "history": history,
            "docs": docs,
        }

        result = await self._execute_with_fallback_wrapper(runnable_params, runnable_params)
        await self._persist_conversation(question, result)
        return result

    def _build_context_from_docs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Baut formatierten Kontext aus Dokumenten auf."""
        docs: list[dict[str, Any]] = inputs.get("docs", [])
        snippets = []

        for i, doc in enumerate(docs[: self._config.top_k]):
            txt = str(doc.get("text", ""))
            score = doc.get("score")
            meta = doc.get("metadata") or {}

            prefix = f"[{i+1}]"
            if score is not None:
                prefix += f" (score={score:.3f})"

            source = meta.get("source")
            if source:
                prefix += f" {source}"

            snippets.append(f"{prefix}: {txt}")

        context = "\n".join(snippets)
        return {**inputs, "context": context}

    async def _create_runnable(self) -> Runnable:
        """Erstellt die Runnable-Pipeline."""
        if not (LANGCHAIN_CORE_AVAILABLE and LANGCHAIN_OPENAI_AVAILABLE):
            return self._create_fallback_runnable()

        llm = create_safe_llm(self._config.model, self._config.temperature)
        if llm is None:
            return self._create_fallback_runnable()

        qa_prompt = PromptTemplate.from_template(QA_PROMPT_TEMPLATE)
        parser = StrOutputParser()

        return RunnableLambda(self._build_context_from_docs) | qa_prompt | llm | parser

    @staticmethod
    def _create_fallback_runnable() -> Runnable:
        """Erstellt Fallback-Runnable ohne LLM.

        Returns:
            Runnable das Retrieval-Fallback implementiert
        """
        def _fallback(inputs: dict[str, Any]) -> str:
            return create_retrieval_fallback(inputs.get("docs", []))

        return create_fallback_runnable(_fallback)


__all__ = [
    "KeikoRetrievalQAChain",
    "RetrievalQAConfig",
]
