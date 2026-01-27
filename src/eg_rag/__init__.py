"""
EG-RAG: Retrieval-Augmented Generation with Evidence Graph
for Reliable Multi-Document Reasoning

AAMAS 2026
"""

from .eg_rag import EGRAG
from .graph import build_evidence_graph
from .retrieval import extract_key_sentences
from .nli import NLIClassifier

__version__ = "1.0.0"
__all__ = ["EGRAG", "build_evidence_graph", "extract_key_sentences", "NLIClassifier"]
