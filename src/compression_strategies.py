"""
Four compression strategies for prompt compression experiments.

Strategy A: Random Word Dropout
Strategy B: Stopword-Only Removal
Strategy C: Entity/Number-Preserving Dropout
Strategy D: LLM-Guided Compression
"""

import random
import re
import os
import json
import hashlib

# Common English stopwords (avoids NLTK dependency)
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "although", "though",
    "that", "which", "who", "whom", "this", "these", "those", "am",
    "it", "its", "itself", "i", "me", "my", "myself", "we", "our",
    "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves", "what",
}


def random_dropout(prompt: str, rate: float, seed: int = None) -> str:
    """Strategy A: Randomly remove X% of words. Fully random — no protection."""
    if rate <= 0:
        return prompt
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    words = prompt.split()
    result = [w for w in words if rng.random() > rate]
    return " ".join(result) if result else prompt


def stopword_removal(prompt: str, rate: float, seed: int = None) -> str:
    """Strategy B: Remove only stopwords up to target compression rate."""
    if rate <= 0:
        return prompt
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    words = prompt.split()
    n_total = len(words)
    n_remove = int(n_total * rate)

    # Find indices of stopwords
    stop_idxs = [i for i, w in enumerate(words) if w.lower().strip(".,!?;:\"'()[]") in STOPWORDS]
    rng.shuffle(stop_idxs)

    to_remove = set(stop_idxs[:min(n_remove, len(stop_idxs))])
    result = [w for i, w in enumerate(words) if i not in to_remove]
    return " ".join(result) if result else prompt


def entity_preserving_dropout(prompt: str, rate: float, protected_words: set = None, seed: int = None) -> str:
    """Strategy C: Random dropout but protect digits, proper nouns, label words, ALL CAPS."""
    if rate <= 0:
        return prompt
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    if protected_words is None:
        protected_words = set()

    words = prompt.split()
    n_total = len(words)
    n_remove = int(n_total * rate)

    # Identify droppable indices
    droppable = []
    for i, w in enumerate(words):
        clean = w.strip(".,!?;:\"'()[]")
        if clean.lower() in {pw.lower() for pw in protected_words}:
            continue
        if re.search(r'\d', w):  # contains digit
            continue
        if clean.isupper() and len(clean) > 1:  # ALL CAPS
            continue
        if clean and clean[0].isupper() and not (i == 0 or words[i-1].endswith(('.', '!', '?', ':'))):
            # Proper noun (capitalized not at sentence start)
            continue
        droppable.append(i)

    rng.shuffle(droppable)
    to_remove = set(droppable[:min(n_remove, len(droppable))])
    result = [w for i, w in enumerate(words) if i not in to_remove]
    return " ".join(result) if result else prompt


def llm_compress(prompt: str, target_rate: float, model: str = "claude-sonnet-4-20250514") -> str:
    """Strategy D: Use Claude to intelligently compress the prompt."""
    import anthropic

    if target_rate <= 0:
        return prompt

    # Cache to avoid redundant API calls
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "llm_compress_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_key = hashlib.md5(f"{prompt}|{target_rate}|{model}".encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)["compressed"]

    word_count = len(prompt.split())
    target_words = int(word_count * (1 - target_rate))

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system="You are a prompt compressor. Rewrite the given prompt to be shorter while preserving ALL information needed to complete the task correctly. Do not change labels, rules, examples, or their mappings. Only remove redundant/filler words and shorten phrasing. Return ONLY the compressed prompt, nothing else.",
        messages=[{
            "role": "user",
            "content": f"Compress this prompt to approximately {target_words} words (currently {word_count} words). Remove filler but keep all labels, rules, and examples intact:\n\n{prompt}"
        }],
    )

    compressed = response.content[0].text.strip()

    # Cache result
    with open(cache_path, "w") as f:
        json.dump({"original": prompt, "compressed": compressed, "rate": target_rate}, f)

    return compressed


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

ALC_PROTECTED = {"zorp", "bleem", "quiff", "Label", "Input", "Classify", "classifier", "categories"}
SLR_PROTECTED = {"ALPHA", "BETA", "GAMMA", "DELTA", "Output", "Input", "Rules", "Rule", "output"}


def get_protected_words(task: str) -> set:
    if task == "alc":
        return ALC_PROTECTED
    elif task == "slr":
        return SLR_PROTECTED
    return set()


def compress(prompt: str, strategy: str, rate: float, task: str = "alc", seed: int = None) -> str:
    """Unified compression interface."""
    if strategy == "random_dropout":
        return random_dropout(prompt, rate, seed=seed)
    elif strategy == "stopword_removal":
        return stopword_removal(prompt, rate, seed=seed)
    elif strategy == "entity_preserving":
        return entity_preserving_dropout(prompt, rate, protected_words=get_protected_words(task), seed=seed)
    elif strategy == "llm_guided":
        return llm_compress(prompt, rate)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def compression_ratio(original: str, compressed: str) -> float:
    """Compute actual compression ratio (fraction of words removed)."""
    orig_words = len(original.split())
    comp_words = len(compressed.split())
    if orig_words == 0:
        return 0.0
    return 1.0 - (comp_words / orig_words)
