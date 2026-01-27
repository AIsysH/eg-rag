"""
Utility functions for EG-RAG.
"""

import re
import string


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"'", u"'", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(s))))
    ).strip()


def simple_sent_tokenize(text: str) -> list[str]:
    """
    Split text into sentences based on period, question mark, or exclamation mark
    followed by whitespace.
    """
    return re.split(r'(?<=[\.\?\!])\s+', text.strip())


def separate_passages(document: str) -> list[str]:
    """
    Split a document by 'Document:' delimiter.
    """
    raw_passages = document.split("Document:")
    return [p.strip() for p in raw_passages if p.strip()]


def eliminate_duplicates(items: list) -> list:
    """Remove duplicates while preserving order."""
    seen = set()
    unique = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
