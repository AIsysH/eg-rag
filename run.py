#!/usr/bin/env python3
"""
EG-RAG: Run evaluation on benchmark datasets.

Usage:
    python run.py --dataset faitheval --top_k 3 --output results.json
    python run.py --dataset hotpotqa --nli_model roberta-large-mnli
"""

import argparse
import json
import os
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from src.eg_rag import EGRAG
from src.eg_rag.evaluation import compute_metrics


# Dataset configurations
DATASET_CONFIGS = {
    "faitheval": {
        "name": "Salesforce/FaithEval-inconsistent-v1.0",
        "split": "test",
        "context_key": "context",
        "question_key": "question",
        "answer_key": "answers",
    },
    "hotpotqa": {
        "name": "hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "context_key": "context",
        "question_key": "question",
        "answer_key": "answer",
    },
}


def load_benchmark_dataset(dataset_name: str, num_samples: int = None):
    """Load a benchmark dataset."""
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if "config" in config:
        dataset = load_dataset(config["name"], config["config"], split=config["split"])
    else:
        dataset = load_dataset(config["name"], split=config["split"])

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset, config


def format_context(example, config):
    """Format context from dataset example."""
    context_key = config["context_key"]

    if isinstance(example[context_key], list):
        # HotpotQA format: list of (title, sentences)
        passages = []
        for title, sentences in zip(example[context_key]["title"], example[context_key]["sentences"]):
            text = " ".join(sentences)
            passages.append(f"Document: {title}\n{text}")
        return "\n\n".join(passages)
    else:
        return example[context_key]


def main():
    parser = argparse.ArgumentParser(description="Run EG-RAG evaluation")
    parser.add_argument("--dataset", type=str, default="faitheval",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Benchmark dataset to evaluate on")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of key sentences per document")
    parser.add_argument("--nli_model", type=str, default="roberta-large-mnli",
                        help="NLI model name")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                        help="LLM model for answer generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize EG-RAG
    print(f"Initializing EG-RAG with top_k={args.top_k}")
    eg_rag = EGRAG(
        nli_model=args.nli_model,
        llm_model=args.llm_model,
        top_k=args.top_k
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset, config = load_benchmark_dataset(args.dataset, args.num_samples)
    print(f"Loaded {len(dataset)} samples")

    # Run evaluation
    predictions = []
    ground_truths = []

    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        context = format_context(example, config)
        question = example[config["question_key"]]

        answer, _ = eg_rag.run(question, context, verbose=args.verbose)
        predictions.append(answer)

        # Get ground truth
        gt = example[config["answer_key"]]
        if isinstance(gt, str):
            gt = [gt]
        ground_truths.append(gt)

        if args.verbose:
            tqdm.write(f"[{i}] Pred: {answer}, GT: {gt}")

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths, verbose=False)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {metrics['total_items']}")
    print(f"top_k: {args.top_k}")
    print("-" * 50)
    print(f"Exact Match (Accuracy): {metrics['exact_match']:.2%}")
    print(f"Element F1: {metrics['element_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("=" * 50)

    # Save results
    if args.output:
        results = {
            "config": {
                "dataset": args.dataset,
                "num_samples": len(dataset),
                "top_k": args.top_k,
                "nli_model": args.nli_model,
                "llm_model": args.llm_model,
            },
            "metrics": metrics,
            "predictions": predictions,
            "ground_truths": ground_truths,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
