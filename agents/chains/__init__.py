"""Chain-Module für Enterprise-AI-Anwendungen.

Dieses Package stellt produktionsreife, leichtgewichtige Chain-Implementierungen
auf Basis von langchain-core bereit. Alle Chains sind defensiv ausgelegt und
funktionieren auch bei fehlenden optionalen Abhängigkeiten.

Verfügbare Chain-Typen:
- ConversationChain: Mehrschritt-Konversation mit Tool-Support
- RetrievalQAChain: Retrieval-basierte Q&A für Enterprise-Integrationen
- RouterChain: Intelligentes Chain-Routing mit Klassifikation
- SummarizationChain: Dokumenten-Zusammenfassung mit Fallback
- TransformChain: Generische Daten-Transformation
- SequentialChain: Sequentielle Verkettung von Operationen

Gemeinsame Infrastruktur:
- BaseChain: Abstrakte Basis-Klasse mit Mixins
- ChatMemory: Protokoll für Chat-Speicher-Integration
- AsyncRetriever: Protokoll für asynchrone Retriever
- chains_utils: Utility-Funktionen und Mixins
"""

from __future__ import annotations

# Basis-Infrastruktur
from .chains_base import BaseKeikoChain, SimpleChain
from .chains_common import AsyncRetriever, ChatMemory
from .chains_constants import *

# Chain-Implementierungen
from .keiko_conversation_chain import ChainConfig, KeikoConversationChain, KeikoSequentialChain
from .keiko_retrieval_qa_chain import KeikoRetrievalQAChain, RetrievalQAConfig
from .keiko_router_chain import KeikoRouterChain, RouterConfig, RouteType
from .keiko_summarization_chain import KeikoSummarizationChain, SummarizationConfig
from .keiko_transform_chain import KeikoTransformChain, TransformConfig

__all__ = [
    "AsyncRetriever",
    # Basis-Infrastruktur
    "BaseKeikoChain",
    # Conversation Chain
    "ChainConfig",
    "ChatMemory",
    "KeikoConversationChain",
    "KeikoRetrievalQAChain",
    "KeikoRouterChain",
    "KeikoSequentialChain",
    "KeikoSummarizationChain",
    "KeikoTransformChain",
    # Retrieval QA Chain
    "RetrievalQAConfig",
    # Router Chain
    "RouteType",
    "RouterConfig",
    "SimpleChain",
    # Summarization Chain
    "SummarizationConfig",
    # Transform Chain
    "TransformConfig",
]
