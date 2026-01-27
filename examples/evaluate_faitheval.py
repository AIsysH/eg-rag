#!/usr/bin/env python3
"""
Example: Evaluate EG-RAG on FaithEval-Inconsistent dataset.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
from tqdm import tqdm

from src.eg_rag import EGRAG
from src.eg_rag.evaluation import compute_metrics


def main():
    # Load FaithEval-Inconsistent dataset
    print("Loading FaithEval-Inconsistent dataset...")
    dataset = load_dataset("Salesforce/FaithEval-inconsistent-v1.0", split="test")

    # Limit samples for quick testing
    num_samples = 10  # Set to None for full evaluation
    if num_samples:
        dataset = dataset.select(range(num_samples))

    print(f"Evaluating on {len(dataset)} samples")

    # Initialize EG-RAG
    eg_rag = EGRAG(
        nli_model="roberta-large-mnli",
        llm_model="gpt-4o-mini",
        top_k=3
    )

    # Run evaluation
    predictions = []
    ground_truths = []

    for example in tqdm(dataset, desc="Evaluating"):
        answer, _ = eg_rag.run(
            query=example["question"],
            documents=example["context"],
            verbose=False
        )
        predictions.append(answer)
        ground_truths.append(example["answers"])

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)

    print("\n" + "=" * 50)
    print("RESULTS - FaithEval-Inconsistent")
    print("=" * 50)
    print(f"Samples: {metrics['total_items']}")
    print(f"Exact Match (Accuracy): {metrics['exact_match']:.2%}")
    print(f"Element F1: {metrics['element_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
