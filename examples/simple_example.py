#!/usr/bin/env python3
"""
Simple example demonstrating EG-RAG usage.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eg_rag import EGRAG


def main():
    # Example documents with conflicting information
    documents = """
Document: The Eiffel Tower, completed in 1889, was designed by Gustave Eiffel for the World's Fair held in Paris. It stands at 330 meters tall and was the tallest man-made structure in the world until 1930.

Document: The Eiffel Tower was finished in 1887 as a temporary structure for a local exhibition. Designed by Alexandre Gustave Bonickhausen, it was originally planned to be dismantled after 20 years.
"""

    query = "When was the Eiffel Tower completed and who designed it?"

    # Initialize EG-RAG
    print("Initializing EG-RAG...")
    eg_rag = EGRAG(
        nli_model="roberta-large-mnli",
        llm_model="gpt-4o-mini",
        top_k=3
    )

    # Run the pipeline
    print(f"\nQuery: {query}")
    print("-" * 50)

    answer, evidence_graph = eg_rag.run(
        query=query,
        documents=documents,
        verbose=True
    )

    print("-" * 50)
    print(f"Final Answer: {answer}")

    # Show graph info
    print(f"\nEvidence Graph:")
    print(f"  Nodes: {evidence_graph.number_of_nodes()}")
    print(f"  Edges: {evidence_graph.number_of_edges()}")

    # Show edge labels
    print("\nEdge relationships:")
    for u, v, data in evidence_graph.edges(data=True):
        print(f"  {u} <-> {v}: {data['label']} (weight: {data['weight']:.3f})")


if __name__ == "__main__":
    main()
