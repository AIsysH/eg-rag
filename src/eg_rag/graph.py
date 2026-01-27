"""
Evidence Graph construction and analysis module.
"""

import networkx as nx
from typing import Optional

from .nli import NLIClassifier


def build_evidence_graph(
    sentences: list[str],
    scores: list[float],
    nli_classifier: Optional[NLIClassifier] = None
) -> nx.Graph:
    """
    Build an evidence graph from key sentences.

    Nodes: Each sentence with attributes (text, score)
    Edges: NLI relationships with attributes (label, weight)
           weight = nli_probability * score_i * score_j

    Args:
        sentences: List of key sentences
        scores: Relevance scores for each sentence
        nli_classifier: NLI classifier instance (created if None)

    Returns:
        NetworkX Graph representing the evidence graph
    """
    if nli_classifier is None:
        nli_classifier = NLIClassifier()

    G = nx.Graph()

    # Add nodes
    for i, (sent, score) in enumerate(zip(sentences, scores)):
        G.add_node(i, text=sent, score=score)

    # Add edges based on NLI classification
    n = len(sentences)
    for i in range(n):
        for j in range(i + 1, n):
            label, prob = nli_classifier.classify(sentences[i], sentences[j])
            weight = prob * scores[i] * scores[j]
            G.add_edge(i, j, label=label, weight=weight)

    return G


def find_subgraphs_by_label(G: nx.Graph, label: str) -> list[nx.Graph]:
    """
    Find connected subgraphs containing only edges with the specified label.

    Args:
        G: Evidence graph
        label: Edge label to filter ('contradiction', 'neutral', 'support')

    Returns:
        List of subgraphs with 2+ connected nodes
    """
    H = nx.Graph(
        (u, v, d) for u, v, d in G.edges(data=True) if d["label"] == label
    )
    return [G.subgraph(c).copy() for c in nx.connected_components(H) if len(c) > 1]


def find_contradiction_subgraphs(G: nx.Graph) -> list[nx.Graph]:
    """Find subgraphs connected by contradiction edges."""
    return find_subgraphs_by_label(G, "contradiction")


def find_neutral_subgraphs(G: nx.Graph) -> list[nx.Graph]:
    """Find subgraphs connected by neutral edges."""
    return find_subgraphs_by_label(G, "neutral")


def find_support_subgraphs(G: nx.Graph) -> list[nx.Graph]:
    """Find subgraphs connected by support/entailment edges."""
    return find_subgraphs_by_label(G, "support")


def extract_clusters(G: nx.Graph, subgraphs: list[nx.Graph]) -> list[list[str]]:
    """
    Extract sentence clusters from subgraphs.

    Args:
        G: Original evidence graph (contains node attributes)
        subgraphs: List of subgraphs

    Returns:
        List of clusters, where each cluster is a list of sentence texts
    """
    return [
        [G.nodes[node]["text"] for node in subg.nodes()]
        for subg in subgraphs
    ]
