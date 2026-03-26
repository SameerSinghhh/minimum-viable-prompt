"""
Evaluation harness for prompt compression experiments.

Sends compressed prompts to Claude, parses responses, scores exact match.
Supports async batched calls with caching and retry logic.
"""

import asyncio
import json
import os
import hashlib
import time
import re
from typing import Optional

import anthropic

# ──────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────

ALC_PROMPT_TEMPLATE = """You are a classifier. Classify each sentence into one of three categories:
- "zorp": sentences expressing happiness or excitement
- "bleem": sentences expressing sadness or disappointment
- "quiff": sentences expressing anger or frustration

Respond with ONLY the label (zorp, bleem, or quiff). No explanation.

Examples:
Input: "I just got promoted at work!"
Label: zorp

Input: "My dog passed away yesterday."
Label: bleem

Input: "I can't believe they cancelled my flight again."
Label: quiff

Input: "We're going on vacation to Hawaii tomorrow!"
Label: zorp

Input: "I failed my driving test for the third time."
Label: bleem

Input: "Someone scratched my brand new car in the parking lot."
Label: quiff

Input: "My best friend surprised me with concert tickets!"
Label: zorp

Input: "The doctor said the treatment didn't work."
Label: bleem

Now classify this:
Input: "{test_input}"
Label:"""

SLR_PROMPT_TEMPLATE = """You are a logic engine. Apply the following rules to classify inputs.

Rules:
1. If the input contains a color AND a number greater than 5, output "ALPHA"
2. If the input contains an animal AND no numbers, output "BETA"
3. If the input contains a city name AND a negative word, output "GAMMA"
4. If none of the above apply, output "DELTA"
5. If multiple rules match, apply the first matching rule only.

Respond with ONLY the label (ALPHA, BETA, GAMMA, or DELTA). No explanation.

Examples:
Input: "The red 7 is here."
Output: ALPHA

Input: "A dog ran through the park."
Output: BETA

Input: "Tokyo is a terrible place lately."
Output: GAMMA

Input: "The old bridge was quiet today."
Output: DELTA

Input: "I saw 12 blue lights flickering."
Output: ALPHA

Input: "The cat sat on the windowsill."
Output: BETA

Now classify:
Input: "{test_input}"
Output:"""

# Section boundaries for ablation experiments
ALC_SECTIONS = {
    "instructions": (0, 3),   # Lines 0-3 (system instruction + label defs)
    "examples": (4, -2),      # Lines 4 to second-to-last (few-shot examples)
    "task": (-2, None),       # Last 2 lines (the actual test input)
}

SLR_SECTIONS = {
    "instructions": (0, 7),
    "examples": (7, -3),
    "task": (-3, None),
}


def get_prompt_template(task: str) -> str:
    if task == "alc":
        return ALC_PROMPT_TEMPLATE
    elif task == "slr":
        return SLR_PROMPT_TEMPLATE
    raise ValueError(f"Unknown task: {task}")


def build_prompt(task: str, test_input: str) -> str:
    template = get_prompt_template(task)
    return template.replace("{test_input}", test_input)


def get_prompt_sections(task: str, test_input: str) -> dict:
    """Split prompt into instructions, examples, and task sections."""
    prompt = build_prompt(task, test_input)
    lines = prompt.strip().split("\n")

    if task == "alc":
        # Instructions: first block (classifier + label defs)
        # Find where examples start
        ex_start = None
        task_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Examples:"):
                ex_start = i
            if line.strip().startswith("Now classify"):
                task_start = i
                break

        instructions = "\n".join(lines[:ex_start]) if ex_start else ""
        examples = "\n".join(lines[ex_start:task_start]) if ex_start and task_start else ""
        task_section = "\n".join(lines[task_start:]) if task_start else ""

    elif task == "slr":
        ex_start = None
        task_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Examples:"):
                ex_start = i
            if line.strip().startswith("Now classify"):
                task_start = i
                break

        instructions = "\n".join(lines[:ex_start]) if ex_start else ""
        examples = "\n".join(lines[ex_start:task_start]) if ex_start and task_start else ""
        task_section = "\n".join(lines[task_start:]) if task_start else ""
    else:
        raise ValueError(f"Unknown task: {task}")

    return {
        "instructions": instructions,
        "examples": examples,
        "task": task_section,
    }


def reassemble_prompt(sections: dict) -> str:
    return "\n".join([sections["instructions"], sections["examples"], sections["task"]])


# ──────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────

ALC_LABELS = {"zorp", "bleem", "quiff"}
SLR_LABELS = {"ALPHA", "BETA", "GAMMA", "DELTA"}


def parse_response(response_text: str, task: str) -> Optional[str]:
    """Extract the predicted label from model response."""
    text = response_text.strip()

    if task == "alc":
        valid = ALC_LABELS
        # Try exact first word match
        first_word = text.lower().split()[0].strip(".,!?;:\"'()[]") if text else ""
        if first_word in valid:
            return first_word

        # Look for label patterns like "Label: zorp" or "**zorp**" or ": zorp"
        for label in valid:
            # Check for "Label: label" or "output: label" patterns
            if re.search(rf'(?:label|answer|classification|output)\s*[:=]\s*{label}', text, re.IGNORECASE):
                return label
            # Check for bold/emphasized labels
            if re.search(rf'\*\*{label}\*\*', text, re.IGNORECASE):
                return label

        # Find last occurrence of any valid label in the text (models often reason then conclude)
        last_pos = -1
        last_label = None
        for label in valid:
            pos = text.lower().rfind(label)
            if pos > last_pos:
                last_pos = pos
                last_label = label
        if last_label:
            return last_label

        return first_word if first_word else None

    elif task == "slr":
        valid = SLR_LABELS
        # Try exact first word match
        first_word = text.upper().split()[0].strip(".,!?;:\"'()[]") if text else ""
        if first_word in valid:
            return first_word

        # Look for output patterns like "Output: ALPHA" or "**ALPHA**"
        for label in valid:
            if re.search(rf'(?:output|answer|result|classification)\s*[:=]\s*{label}', text, re.IGNORECASE):
                return label
            if re.search(rf'\*\*{label}\*\*', text):
                return label

        # Find last occurrence of any valid label (conclusion usually at end)
        last_pos = -1
        last_label = None
        for label in valid:
            pos = text.upper().rfind(label)
            if pos > last_pos:
                last_pos = pos
                last_label = label
        if last_label:
            return last_label

        return first_word if first_word else None

    return None


# ──────────────────────────────────────────────
# API calling with cache and retry
# ──────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "api_cache")


def _cache_key(prompt: str, model: str) -> str:
    return hashlib.md5(f"{model}|{prompt}".encode()).hexdigest()


def _get_cached(prompt: str, model: str) -> Optional[str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{_cache_key(prompt, model)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["response"]
    return None


def _set_cache(prompt: str, model: str, response: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{_cache_key(prompt, model)}.json")
    with open(path, "w") as f:
        json.dump({"prompt": prompt[:200], "model": model, "response": response}, f)


async def call_model_async(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 3,
    semaphore: asyncio.Semaphore = None,
) -> str:
    """Call Claude API with caching and retry logic."""
    # Check cache
    cached = _get_cached(prompt, model)
    if cached is not None:
        return cached

    if semaphore:
        await semaphore.acquire()

    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text.strip()
            _set_cache(prompt, model, result)
            if semaphore:
                semaphore.release()
            return result

        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  Rate limited, waiting {wait}s...")
            await asyncio.sleep(wait)
        except anthropic.APIError as e:
            wait = 2 ** attempt * 2
            print(f"  API error ({e}), retry in {wait}s...")
            await asyncio.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt * 2
            print(f"  Unexpected error ({type(e).__name__}: {e}), retry in {wait}s...")
            await asyncio.sleep(wait)

    if semaphore:
        semaphore.release()
    return ""


async def eval_batch_async(
    prompts: list[str],
    task: str,
    labels: list[str],
    model: str = "claude-sonnet-4-20250514",
    concurrency: int = 20,
) -> list[dict]:
    """Evaluate a batch of prompts. Returns list of {prompt, predicted, expected, correct}."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    async def eval_one(prompt, expected):
        response = await call_model_async(client, prompt, model, semaphore=semaphore)
        predicted = parse_response(response, task)
        return {
            "predicted": predicted,
            "expected": expected,
            "correct": predicted == expected,
            "raw_response": response,
        }

    tasks = [eval_one(p, l) for p, l in zip(prompts, labels)]
    results = await asyncio.gather(*tasks)
    return list(results)


def eval_batch(prompts, task, labels, model="claude-sonnet-4-20250514", concurrency=20):
    """Synchronous wrapper for eval_batch_async."""
    return asyncio.run(eval_batch_async(prompts, task, labels, model, concurrency))
