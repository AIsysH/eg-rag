"""
Main EG-RAG class for Retrieval-Augmented Generation with Evidence Graph.
"""

from typing import Optional
import networkx as nx

from .utils import separate_passages
from .retrieval import extract_key_sentences, get_device
from .nli import NLIClassifier
from .graph import (
    build_evidence_graph,
    find_contradiction_subgraphs,
    find_neutral_subgraphs,
    find_support_subgraphs,
    extract_clusters
)
from .prompts import create_qa_prompt, create_simple_prompt
from .llm import LLMClient


class EGRAG:
    """
    EG-RAG: Evidence Graph based Retrieval-Augmented Generation.

    This class implements the full EG-RAG pipeline:
    1. Extract key sentences from documents using semantic similarity
    2. Build evidence graph with NLI-based edge classification
    3. Identify contradiction/neutral/support clusters
    4. Generate answer using LLM with evidence-aware prompting
    """

    def __init__(
        self,
        nli_model: str = "roberta-large-mnli",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        top_k: int = 3,
        device: Optional[str] = None
    ):
        """
        Initialize EG-RAG.

        Args:
            nli_model: HuggingFace model for NLI classification
            embed_model: SentenceTransformer model for embeddings
            llm_provider: LLM provider ("openai", "anthropic", "ollama")
            llm_model: Model name (defaults based on provider)
            llm_api_key: API key (uses env if None)
            llm_base_url: Base URL for API (for Ollama: http://localhost:11434)
            top_k: Number of key sentences per document
            device: Device string ('cuda', 'mps', 'cpu') or None for auto
        """
        self.embed_model = embed_model
        self.top_k = top_k

        # Set device
        if device:
            import torch
            self.device = torch.device(device)
        else:
            self.device = get_device()

        # Initialize NLI classifier
        self.nli_classifier = NLIClassifier(model_name=nli_model, device=self.device)

        # Initialize LLM client
        self.llm_client = LLMClient(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url
        )

        print(f"EG-RAG initialized with device: {self.device}, LLM: {llm_provider}/{self.llm_client.model}")

    def run(
        self,
        query: str,
        documents: str,
        verbose: bool = False
    ) -> tuple[str, nx.Graph]:
        """
        Run the EG-RAG pipeline.

        Args:
            query: User question
            documents: Document string (may contain multiple 'Document:' sections)
            verbose: Print intermediate results

        Returns:
            Tuple of (answer_string, evidence_graph)
        """
        # 1. Separate passages
        passages = separate_passages(documents)

        # 2. Extract key sentences from each passage
        key_sents = []
        key_scores = []
        key_idxs = []

        for idx, passage in enumerate(passages):
            sents, scores = extract_key_sentences(
                [passage],
                query,
                model_name=self.embed_model,
                top_k=self.top_k,
                device=self.device
            )
            key_sents.extend(sents)
            key_scores.extend(scores)
            key_idxs.extend([idx] * len(sents))

        # Format key sentences with document index and score
        formatted_key_sents = [
            f"[{idx}] ({score:.3f}) {sent}"
            for idx, sent, score in zip(key_idxs, key_sents, key_scores)
        ]

        # 3. Build evidence graph
        G = build_evidence_graph(key_sents, key_scores, self.nli_classifier)

        # 4. Find clusters
        contradiction_subgraphs = find_contradiction_subgraphs(G)
        neutral_subgraphs = find_neutral_subgraphs(G)
        support_subgraphs = find_support_subgraphs(G)

        contradictory_clusters = extract_clusters(G, contradiction_subgraphs)
        neutral_clusters = extract_clusters(G, neutral_subgraphs)
        support_clusters = extract_clusters(G, support_subgraphs)

        if verbose:
            print("Key Sentences:")
            print("\n".join(formatted_key_sents))

            if contradictory_clusters:
                print("\nContradictory clusters:")
                for i, cluster in enumerate(contradictory_clusters, 1):
                    print(f"  Cluster {i}:")
                    for s in cluster:
                        print(f"   - {s}")

            if neutral_clusters:
                print("\nNeutral clusters:")
                for i, cluster in enumerate(neutral_clusters, 1):
                    print(f"  Cluster {i}:")
                    for s in cluster:
                        print(f"   - {s}")

            if support_clusters:
                print("\nSupport clusters:")
                for i, cluster in enumerate(support_clusters, 1):
                    print(f"  Cluster {i}:")
                    for s in cluster:
                        print(f"   - {s}")

        # 5. Generate answer using LLM
        messages = create_qa_prompt(
            query,
            passages,
            formatted_key_sents,
            contradictory_clusters,
            neutral_clusters,
            support_clusters
        )

        answer = self.llm_client.generate(messages)

        if verbose:
            print(f"\nAnswer: {answer}")

        return answer, G

    def run_batch(
        self,
        queries: list[str],
        documents_list: list[str],
        verbose: bool = False
    ) -> list[tuple[str, nx.Graph]]:
        """
        Run EG-RAG on multiple examples.

        Args:
            queries: List of questions
            documents_list: List of document strings
            verbose: Print progress

        Returns:
            List of (answer, graph) tuples
        """
        from tqdm import tqdm

        results = []
        for i, (query, docs) in enumerate(tqdm(
            zip(queries, documents_list),
            total=len(queries),
            desc="Processing"
        )):
            answer, G = self.run(query, docs, verbose=False)
            if verbose:
                print(f"[{i}] {answer}")
            results.append((answer, G))

        return results
