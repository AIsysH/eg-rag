"""
Natural Language Inference (NLI) module for sentence relationship classification.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .retrieval import get_device


class NLIClassifier:
    """
    NLI classifier for determining sentence relationships.

    Supports multiple NLI models:
    - roberta-large-mnli (default): RoBERTa fine-tuned on MNLI
    - facebook/bart-large-mnli: BART fine-tuned on MNLI
    - microsoft/deberta-v3-large-mnli: DeBERTa v3
    """

    # Label mapping for different model architectures
    LABEL_MAPS = {
        "roberta-large-mnli": {0: "contradiction", 1: "neutral", 2: "entailment"},
        "facebook/bart-large-mnli": {0: "contradiction", 1: "neutral", 2: "entailment"},
        "microsoft/deberta-v3-large-mnli": {0: "contradiction", 1: "neutral", 2: "entailment"},
    }

    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: torch.device = None
    ):
        """
        Initialize NLI classifier.

        Args:
            model_name: HuggingFace model name for NLI
            device: Torch device (auto-detected if None)
        """
        self.device = device if device else get_device()
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )

        # Get label mapping
        self.label_map = self.LABEL_MAPS.get(
            model_name,
            {0: "contradiction", 1: "neutral", 2: "entailment"}
        )

    def classify(self, sentence_a: str, sentence_b: str) -> tuple[str, float]:
        """
        Classify the relationship between two sentences.

        Args:
            sentence_a: First sentence (premise)
            sentence_b: Second sentence (hypothesis)

        Returns:
            Tuple of (label, probability) where label is one of:
            'contradiction', 'neutral', 'entailment' (mapped to 'support')
        """
        inputs = self.tokenizer(
            sentence_a,
            sentence_b,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        label = self.label_map[idx]

        # Map 'entailment' to 'support' for consistency
        if label == "entailment":
            label = "support"
        elif label == "contradiction":
            label = "contradiction"
        else:
            label = "neutral"

        return label, float(probs[idx])
