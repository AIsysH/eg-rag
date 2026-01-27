"""
Prompt templates for LLM-based answer generation.
"""


def create_qa_prompt(
    question: str,
    documents: list[str],
    formatted_key_sents: list[str],
    contradictory_clusters: list[list[str]],
    neutral_clusters: list[list[str]],
    support_clusters: list[list[str]]
) -> list[dict]:
    """
    Create a full QA prompt with evidence graph analysis.

    Args:
        question: User question
        documents: List of document passages
        formatted_key_sents: Formatted key sentences with scores
        contradictory_clusters: Clusters of contradicting sentences
        neutral_clusters: Clusters of neutral sentences
        support_clusters: Clusters of supporting sentences

    Returns:
        List of message dicts for chat completion
    """
    prompt = f"""You are an expert retrieval-based QA system.

Follow these steps for each input:

You will receive a question together with a set of key passages.

Each key passage includes a document number, a confidence score, and a key sentence.

Focusing on the highest-scoring sentence in each document, consult all of the key sentences to determine the correct answer step-by-step.

How to Answer
For every document, start with the key sentence that has the highest confidence score.
(ex. [0] (0.712) While studying law and philosophy in England and Germany, Iqbal became a member of the London branch of the All India Muslim League.,
[1] (0.698) While studying culinary arts and music in England and Germany, Iqbal became a member of the London branch of the All India Muslim League.)

To find the answer, focus on the highest-scoring sentences in each document in key passages, and refer to the clusters.

It's important to find answers by comparing the key sentences extracted from each document.

Use this evidence to craft your final answer to the question.

* ANSWER EXTRACTION RULES
- Extract the *exact* noun phrase(s) or term(s) that answer the question.
- No explanations, no punctuation, no extra text - just the terms.

Reference Information as follows:

Question: {question}
Documents:
{documents}
Key passages:
{formatted_key_sents}
Contradictory clusters:
{contradictory_clusters}
Neutral clusters:
{neutral_clusters}
Support clusters:
{support_clusters}

You must provide the final answer as follows:

["answer1", "answer2"]

"""

    return [
        {"role": "system", "content": "You are a helpful and logical assistant."},
        {"role": "user", "content": prompt}
    ]


def create_simple_prompt(
    question: str,
    formatted_key_sents: list[str],
    contradictory_clusters: list[list[str]],
    neutral_clusters: list[list[str]],
    support_clusters: list[list[str]]
) -> list[dict]:
    """
    Create a simplified QA prompt without full documents.

    Args:
        question: User question
        formatted_key_sents: Formatted key sentences with scores
        contradictory_clusters: Clusters of contradicting sentences
        neutral_clusters: Clusters of neutral sentences
        support_clusters: Clusters of supporting sentences

    Returns:
        List of message dicts for chat completion
    """
    prompt = f"""
You are an expert retrieval-based QA system.

Follow these steps for each input:

1. INPUTS (they are already pre-formatted for you)
    - Question: {question}
    - Key passages: {formatted_key_sents}
    - Contradictory clusters: {contradictory_clusters}
    - Neutral clusters: {neutral_clusters}
    - Support clusters: {support_clusters}

2. READ & REASON
    a. Carefully skim *all* key passages and clusters.
    b. The key passages consist of the document number, the confidence score, and the key sentence.
       Please refer to each document to find the correct answer.

3. ANSWER EXTRACTION RULES
    - Extract the *exact* noun phrase(s) or term(s) that answer the question.
    - No explanations, no punctuation, no extra text - just the terms.
    - If no answer exists, reply with exactly unknown.

4. OUTPUT FORMAT
    - Please provide the final answer as follows:
    ["answer1", "answer2"]

"""

    return [
        {"role": "system", "content": "You are a helpful and logical assistant."},
        {"role": "user", "content": prompt}
    ]
