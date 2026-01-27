"""
Evaluation metrics for EG-RAG.
"""

import re
import json
import ast
from typing import Optional


def normalize_answer(s: str) -> str:
    """Lowercase, strip, and remove punctuation."""
    return re.sub(r'[^\w\s]', '', s.lower()).strip()


def count_matches(true_list: list[str], pred_list: list[str]) -> int:
    """
    Count how many true answers are matched by predictions.
    Uses case-insensitive, punctuation-agnostic, and substring matching.

    Args:
        true_list: Ground truth answers
        pred_list: Predicted answers

    Returns:
        Number of matched answers
    """
    norm_true = [normalize_answer(t) for t in true_list]
    norm_pred = [normalize_answer(p) for p in pred_list]
    matched_indices = set()

    for i, t in enumerate(norm_true):
        for p in norm_pred:
            if p and (p in t or t in p):
                matched_indices.add(i)
                break
    return len(matched_indices)


def parse_prediction(pred_str: str) -> list[str]:
    """
    Parse prediction string to list of answers.

    Args:
        pred_str: Raw prediction string (JSON list or 'unknown')

    Returns:
        List of predicted answers
    """
    pred_str = (pred_str or '').strip()

    if pred_str.lower() == 'unknown':
        return []

    try:
        return json.loads(pred_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(pred_str)
        except Exception:
            return []


def compute_metrics(
    predictions: list[str],
    ground_truths: list[list[str]],
    verbose: bool = False
) -> dict:
    """
    Compute evaluation metrics.

    Args:
        predictions: List of prediction strings
        ground_truths: List of ground truth answer lists
        verbose: Print per-item metrics

    Returns:
        Dictionary with metrics:
        - element_precision, element_recall, element_f1
        - exact_match (accuracy)
        - macro_precision, macro_recall, macro_f1
    """
    total_matched = 0
    total_true = 0
    total_pred = 0
    fully_correct = 0

    item_precisions = []
    item_recalls = []
    item_f1s = []

    for i, (pred, true_list) in enumerate(zip(predictions, ground_truths)):
        pred_list = parse_prediction(pred)
        matched = count_matches(true_list, pred_list)

        n_true = len(true_list)
        n_pred = len(pred_list)

        # Exact match
        if matched == n_true and n_pred == n_true:
            fully_correct += 1

        # Item-level metrics
        recall = matched / n_true if n_true > 0 else 0.0
        precision = matched / n_pred if n_pred > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        item_precisions.append(precision)
        item_recalls.append(recall)
        item_f1s.append(f1)

        if verbose:
            print(f"Item {i}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")

        total_matched += matched
        total_true += n_true
        total_pred += n_pred

    num_items = len(predictions)

    # Micro-average (element-level)
    element_precision = total_matched / total_pred if total_pred > 0 else 0.0
    element_recall = total_matched / total_true if total_true > 0 else 0.0
    element_f1 = (
        2 * element_precision * element_recall / (element_precision + element_recall)
        if (element_precision + element_recall) > 0 else 0.0
    )

    # Macro-average
    macro_precision = sum(item_precisions) / num_items if num_items > 0 else 0.0
    macro_recall = sum(item_recalls) / num_items if num_items > 0 else 0.0
    macro_f1 = sum(item_f1s) / num_items if num_items > 0 else 0.0

    # Exact match accuracy
    exact_match = fully_correct / num_items if num_items > 0 else 0.0

    return {
        "element_precision": element_precision,
        "element_recall": element_recall,
        "element_f1": element_f1,
        "exact_match": exact_match,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "total_items": num_items,
        "fully_correct": fully_correct,
    }
