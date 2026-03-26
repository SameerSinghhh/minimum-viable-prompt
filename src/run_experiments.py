"""
Main experiment loop for prompt compression research.

Runs all 80 conditions (4 strategies × 2 tasks × 10 compression levels)
with 5 random seeds for stochastic strategies, plus section-level ablation.

Saves results incrementally with checkpointing.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from compression_strategies import compress, compression_ratio
from eval_harness import (
    build_prompt,
    get_prompt_sections,
    reassemble_prompt,
    eval_batch_async,
    parse_response,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

STRATEGIES = ["random_dropout", "stopword_removal", "entity_preserving", "llm_guided"]
COMPRESSION_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TASKS = ["alc", "slr"]
SEEDS = [42, 123, 456, 789, 1024]
MODEL = "claude-sonnet-4-20250514"
CONCURRENCY = 20
MAX_EXAMPLES = 200  # per condition for main experiments

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.json")
RAW_RESULTS_PATH = os.path.join(RESULTS_DIR, "raw_results.jsonl")


def load_dataset(task: str) -> list[dict]:
    filename = "alc_test_set.jsonl" if task == "alc" else "slr_test_set.jsonl"
    path = os.path.join(DATA_DIR, filename)
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_checkpoint() -> set:
    """Return set of completed condition keys."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(completed: set):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(list(completed), f)


def append_results(results: list[dict]):
    with open(RAW_RESULTS_PATH, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def condition_key(task, strategy, rate, seed=None):
    return f"{task}|{strategy}|{rate}|{seed}"


async def run_condition(
    task: str,
    strategy: str,
    rate: float,
    seed: int,
    dataset: list[dict],
    completed: set,
) -> list[dict]:
    """Run a single experimental condition."""
    key = condition_key(task, strategy, rate, seed)
    if key in completed:
        print(f"  [SKIP] {key} (already completed)")
        return []

    examples = dataset[:MAX_EXAMPLES]
    prompts = []
    labels = []

    for ex in examples:
        # Build the full prompt
        full_prompt = build_prompt(task, ex["text"])
        # Compress it (don't compress the test input line itself)
        sections = get_prompt_sections(task, ex["text"])
        compressed_instr = compress(sections["instructions"], strategy, rate, task, seed)
        compressed_examples = compress(sections["examples"], strategy, rate, task, seed)
        # Keep task section intact
        compressed_prompt = reassemble_prompt({
            "instructions": compressed_instr,
            "examples": compressed_examples,
            "task": sections["task"],
        })
        prompts.append(compressed_prompt)
        labels.append(ex["label"])

    # Compute actual compression ratio on first prompt for logging
    orig_prompt = build_prompt(task, examples[0]["text"])
    actual_ratio = compression_ratio(orig_prompt, prompts[0])

    print(f"  Evaluating {len(prompts)} examples... (actual compression: {actual_ratio:.1%})")

    import anthropic
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    from eval_harness import call_model_async

    results = []
    batch_size = 50
    correct = 0
    total = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        tasks_list = []
        for p, l in zip(batch_prompts, batch_labels):
            tasks_list.append(call_model_async(client, p, MODEL, semaphore=semaphore))

        responses = await asyncio.gather(*tasks_list)

        for j, (resp, expected) in enumerate(zip(responses, batch_labels)):
            predicted = parse_response(resp, task)
            is_correct = predicted == expected
            correct += int(is_correct)
            total += 1

            results.append({
                "task": task,
                "strategy": strategy,
                "compression_rate": rate,
                "seed": seed,
                "example_idx": i + j,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "raw_response": resp,
                "actual_compression": actual_ratio,
                "tokens_remaining_frac": 1 - actual_ratio,
            })

    acc = correct / total if total > 0 else 0
    print(f"  → Accuracy: {acc:.1%} ({correct}/{total})")

    # Save results and checkpoint
    append_results(results)
    completed.add(key)
    save_checkpoint(completed)

    return results


async def run_ablation_condition(
    task: str,
    section_to_compress: str,  # "instructions", "examples", or "both"
    rate: float,
    seed: int,
    dataset: list[dict],
    completed: set,
) -> list[dict]:
    """Run section-level ablation: compress only one section."""
    key = f"ablation|{task}|{section_to_compress}|{rate}|{seed}"
    if key in completed:
        print(f"  [SKIP] {key}")
        return []

    strategy = "entity_preserving"  # Use entity-preserving for ablation
    examples = dataset[:MAX_EXAMPLES]
    prompts = []
    labels = []

    for ex in examples:
        sections = get_prompt_sections(task, ex["text"])

        if section_to_compress == "instructions":
            sections["instructions"] = compress(sections["instructions"], strategy, rate, task, seed)
        elif section_to_compress == "examples":
            sections["examples"] = compress(sections["examples"], strategy, rate, task, seed)
        elif section_to_compress == "both":
            sections["instructions"] = compress(sections["instructions"], strategy, rate, task, seed)
            sections["examples"] = compress(sections["examples"], strategy, rate, task, seed)

        prompts.append(reassemble_prompt(sections))
        labels.append(ex["label"])

    print(f"  Ablation [{section_to_compress}] evaluating {len(prompts)} examples...")

    import anthropic
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    from eval_harness import call_model_async

    results = []
    correct = 0
    total = 0

    for i in range(0, len(prompts), 50):
        batch = prompts[i:i+50]
        batch_labels = labels[i:i+50]
        tasks_list = [call_model_async(client, p, MODEL, semaphore=semaphore) for p in batch]
        responses = await asyncio.gather(*tasks_list)

        for j, (resp, expected) in enumerate(zip(responses, batch_labels)):
            predicted = parse_response(resp, task)
            is_correct = predicted == expected
            correct += int(is_correct)
            total += 1
            results.append({
                "task": task,
                "strategy": f"ablation_{section_to_compress}",
                "compression_rate": rate,
                "seed": seed,
                "example_idx": i + j,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "raw_response": resp,
            })

    acc = correct / total if total > 0 else 0
    print(f"  → Ablation [{section_to_compress}] accuracy: {acc:.1%}")

    append_results(results)
    completed.add(key)
    save_checkpoint(completed)
    return results


async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    completed = load_checkpoint()

    total_conditions = len(TASKS) * len(STRATEGIES) * len(COMPRESSION_RATES) * len(SEEDS)
    print(f"Prompt Compression Experiment")
    print(f"{'='*50}")
    print(f"Total conditions: {total_conditions}")
    print(f"Already completed: {len(completed)}")
    print(f"Model: {MODEL}")
    print(f"Examples per condition: {MAX_EXAMPLES}")
    print()

    # Load datasets
    datasets = {}
    for task in TASKS:
        datasets[task] = load_dataset(task)
        print(f"Loaded {task}: {len(datasets[task])} examples")

    start_time = time.time()

    # ── Main experiments ──
    for task in TASKS:
        for strategy in STRATEGIES:
            for rate in COMPRESSION_RATES:
                seeds_to_run = [None] if strategy == "llm_guided" else SEEDS

                for seed in seeds_to_run:
                    key = condition_key(task, strategy, rate, seed)
                    if key in completed:
                        continue

                    seed_str = f"seed={seed}" if seed else "deterministic"
                    print(f"\n[{task.upper()}] {strategy} @ {rate:.0%} ({seed_str})")

                    await run_condition(
                        task, strategy, rate, seed,
                        datasets[task], completed
                    )

    # ── Section ablation ──
    print(f"\n{'='*50}")
    print("Section-Level Ablation (50% compression)")
    print(f"{'='*50}")

    ablation_rate = 0.5
    for task in TASKS:
        for section in ["instructions", "examples", "both"]:
            for seed in SEEDS:
                print(f"\n[{task.upper()}] Ablation: compress {section} @ {ablation_rate:.0%} (seed={seed})")
                await run_ablation_condition(
                    task, section, ablation_rate, seed,
                    datasets[task], completed
                )

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"All experiments complete! ({elapsed:.0f}s)")
    print(f"Results saved to {RAW_RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
