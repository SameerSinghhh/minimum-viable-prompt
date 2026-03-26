Prompt Compression Research: Finding the Minimum Viable Prompt
Project Goal
Quantitatively determine where and how prompt performance cliffs as prompts are progressively compressed. We test multiple compression strategies across two tasks designed so the model cannot rely on parametric knowledge — it must read the prompt to succeed. Results should be publication-quality: hundreds of trials, statistical significance, clean graphs, and clear takeaways.

Tasks
Task 1: Arbitrary Label Classification (ALС)
Create a sentiment/topic classification task where labels are nonsense words (e.g. "zorp" vs "bleem" vs "quiff"). The model has never seen these labels — it must read the prompt to know what they mean. If the prompt is compressed enough that the label definitions are lost, performance collapses to random chance (33% for 3 classes). This gives us a clean floor and ceiling.
Prompt structure:
You are a classifier. Classify each sentence into one of three categories:
- "zorp": sentences expressing happiness or excitement
- "bleem": sentences expressing sadness or disappointment  
- "quiff": sentences expressing anger or frustration

Examples:
Input: "I just got promoted at work!"
Label: zorp

Input: "My dog passed away yesterday."
Label: bleem

Input: "I can't believe they cancelled my flight again."
Label: quiff

[5-8 more examples]

Now classify this:
Input: "{test_sentence}"
Label:
Scoring: Exact match on label. Baseline (random) = 33%. Full prompt ceiling should be ~90%+.
Dataset: Generate 500 test sentences programmatically (100 per compression level × 5 levels, or better: run all 500 at every compression level for a true curve). Use GPT/Claude to generate diverse sentences in each category, then verify labels manually or with a reference model.

Task 2: Synthetic Logic Rules (SLR)
Give the model a set of made-up logical rules that exist nowhere in its training data. Ask it to apply them to novel inputs. Compression destroys rule fidelity — we can measure exactly which rules survive longest.
Prompt structure:
You are a logic engine. Apply the following rules to classify inputs.
Rules:
1. If the input contains a color AND a number greater than 5, output "ALPHA"
2. If the input contains an animal AND no numbers, output "BETA"
3. If the input contains a city name AND a negative word, output "GAMMA"
4. If none of the above apply, output "DELTA"
5. If multiple rules match, apply the first matching rule only.

Examples:
Input: "The red 7 is here."
Output: ALPHA  (color: red, number: 7 > 5)

Input: "A dog ran through the park."
Output: BETA  (animal: dog, no numbers)

[4-6 more examples covering all rules]

Now classify:
Input: "{test_input}"
Output:
Scoring: Exact match. Baseline = 25% (4 classes). Full prompt ceiling should be ~85-95%.
Dataset: Programmatically generate 600 test inputs with known ground truth labels (150 per class). This is fully synthetic so generation is cheap and labels are guaranteed correct.

Compression Strategies
Test ALL four strategies on BOTH tasks. This is the core experimental contribution — not just "compression hurts" but "which compression strategy preserves performance longest."
Strategy A: Random Word Dropout
Randomly remove X% of words from the prompt, excluding punctuation. Fully random — no token is protected.
pythondef random_dropout(prompt, rate):
    words = prompt.split()
    return ' '.join(w for w in words if random.random() > rate)
Strategy B: Stopword-Only Removal
Remove only stopwords ("the", "a", "an", "is", "are", "of", "and", "or", etc.) up to the target compression rate. If stopwords are exhausted before hitting target rate, stop there.
pythonfrom nltk.corpus import stopwords
STOPS = set(stopwords.words('english'))

def stopword_removal(prompt, rate):
    words = prompt.split()
    stop_idxs = [i for i, w in enumerate(words) if w.lower() in STOPS]
    n_remove = int(len(words) * rate)
    to_remove = set(random.sample(stop_idxs, min(n_remove, len(stop_idxs))))
    return ' '.join(w for i, w in enumerate(words) if i not in to_remove)
Strategy C: Entity/Number-Preserving Dropout
Like random dropout but never removes: digits, numbers spelled out, proper nouns, label words (zorp/bleem/quiff/ALPHA/BETA etc.), or words in ALL CAPS. Everything else is fair game.
pythonimport re
def entity_preserving_dropout(prompt, rate, protected_words):
    words = prompt.split()
    droppable = [i for i, w in enumerate(words) 
                 if w not in protected_words 
                 and not re.search(r'\d', w)
                 and not w.isupper()]
    n_remove = int(len(words) * rate)
    to_remove = set(random.sample(droppable, min(n_remove, len(droppable))))
    return ' '.join(w for i, w in enumerate(words) if i not in to_remove)
Strategy D: LLM-Guided Compression
Ask Claude/GPT to rewrite the prompt to be shorter while preserving meaning, iteratively. This is the "smart" baseline — how does a model compress vs. random?
pythondef llm_compress(prompt, target_tokens, model="claude-sonnet-4-20250514"):
    system = "You are a prompt compressor. Rewrite the given prompt to be as short as possible while preserving all information needed to complete the task correctly. Do not change labels, rules, or examples — only remove redundant words."
    user = f"Compress this prompt to approximately {target_tokens} tokens:\n\n{prompt}"
    # Call API here

Compression Levels
Test at these 10 compression rates for every strategy × task combination:
0% (original), 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%
That's: 4 strategies × 2 tasks × 10 compression levels = 80 conditions

Sample Size & Statistical Rigor

Minimum 200 test examples per condition for reliable accuracy estimates
Run each compression level with 5 different random seeds (for stochastic strategies) and report mean ± std
For LLM-guided compression (deterministic), run the full test set once
Report 95% confidence intervals on all accuracy numbers
Flag conditions where std > 5% as high-variance

Total API calls estimate: ~80 conditions × 200 examples = ~16,000 inference calls. Use batching and async calls to keep runtime reasonable.

Section-Level Ablation (Bonus Experiment)
For both tasks, the prompt has three sections:

System instructions (what labels/rules mean)
Few-shot examples (demonstrated I/O pairs)
Task input (the actual question)

Run a targeted ablation: compress ONLY one section at a time at 50% rate, leave others untouched. This answers: which section matters most?
Conditions:

Compress instructions only
Compress examples only
Compress both equally
Full prompt (control)


Output Requirements
Numerical Results Table
For every condition, report:
TaskStrategyCompression %Mean AccuracyStd95% CITokens Remaining
Graphs (generate with matplotlib/seaborn, save as PNG + include in writeup)

Main curve plot: Accuracy vs. compression rate, one line per strategy, two subplots (one per task). This is the hero figure.
Cliff detection plot: First derivative of accuracy vs. compression rate — visually shows WHERE performance drops sharpest
Strategy comparison bar chart: At 50% compression, accuracy by strategy side-by-side
Section ablation bar chart: Accuracy when compressing each section independently
Variance plot: Std dev across seeds vs. compression rate (shows which strategies are unstable)

Statistical Tests

Run pairwise t-tests between strategies at each compression level
Report which strategy differences are statistically significant (p < 0.05)
Identify the "cliff point" for each strategy: first compression level where accuracy drops >10% from baseline


File Structure
prompt-compression/
├── CLAUDE.md                  # This file
├── README.md                  # Public-facing writeup
├── data/
│   alc_test_set.jsonl         # 500 ALC test examples
│   slr_test_set.jsonl         # 600 SLR test examples
├── src/
│   generate_datasets.py       # Script to generate both datasets
│   compression_strategies.py  # All 4 compression functions
│   eval_harness.py            # Runs a prompt through the model and scores it
│   run_experiments.py         # Main experiment loop — all 80 conditions
│   analyze_results.py         # Stats, CI computation, t-tests
│   plot_results.py            # All 5 graphs
├── results/
│   raw_results.jsonl          # Every single trial result
│   aggregated_results.csv     # Mean/std/CI per condition
├── figures/
│   main_curve.png
│   cliff_detection.png
│   strategy_comparison.png
│   section_ablation.png
│   variance_plot.png
└── writeup/
    blog_post.md               # The actual public writeup

Implementation Order

generate_datasets.py — build and verify both test sets first, spot-check 20 examples manually
compression_strategies.py — implement and unit test all 4 strategies with toy prompts
eval_harness.py — single-example eval function with retry logic and response parsing
run_experiments.py — full experiment loop with progress logging, checkpoint saving (in case of API failures)
analyze_results.py — stats and table generation
plot_results.py — all figures
blog_post.md — writeup once results are in


Key Hypotheses to Test (state these upfront in the writeup)

Entity-preserving dropout will outperform random dropout — protecting numbers and label words should preserve more task-relevant information
LLM-guided compression will outperform all random strategies — a model compressing intelligently should find the minimum viable prompt better than random deletion
Compressing examples hurts more than compressing instructions — few-shot examples carry more signal than prose instructions for novel label tasks
The cliff will be sharper for SLR than ALC — logic rules are more brittle than label definitions
Stopword removal will be nearly free — removing stopwords at low rates should barely hurt performance

These are falsifiable. Report whether each was confirmed or rejected.

Notes on API Usage

Use claude-sonnet-4-20250514 as the primary eval model (fast, cheap, strong)
Optionally re-run a subset on GPT-4o for cross-model comparison (great for the writeup)
Use async Python (asyncio + anthropic.AsyncAnthropic) for parallelism
Cache all API responses to disk so you can re-run analysis without re-calling the API
Set temperature=0 for all eval calls for reproducibility


What Makes This Publishable

Novel framing: "minimum viable prompt" is an actionable concept practitioners care about
Controlled tasks: no parametric knowledge leakage, clean baselines
Multiple strategies compared head-to-head with stats
Section-level ablation adds a second layer of insight
Cross-model replication (if done) makes it generalizable
All code and data public on GitHub for reproducibility