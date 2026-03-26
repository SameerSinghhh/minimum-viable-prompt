# Finding the Minimum Viable Prompt

### How much of a prompt is actually load-bearing?

When you send a prompt to an LLM, you're paying for every token. But how many of those tokens actually matter? We systematically destroyed prompts -- removing 10%, 20%, all the way up to 90% of the words -- and measured exactly when the model stops understanding what you're asking.

**70,000 API calls. 4 compression strategies. 2 tasks. The answer: most of your prompt is filler.**

---

## Why This Matters

Every API call costs tokens. Every token costs money. If you're running prompts at scale -- customer support bots, classification pipelines, content moderation -- even a 30% reduction in prompt length saves real dollars.

But the obvious question is: *which 30% can you safely cut?*

We built an experiment to answer that precisely.

---

## How We Tested This

We needed tasks where the model **must** read the prompt to succeed -- it can't just guess from training knowledge. So we invented tasks with made-up labels:

**Task 1: Made-Up Sentiment Labels**
> "Classify this sentence. `zorp` = happiness, `bleem` = sadness, `quiff` = anger."

The model has never seen "zorp" before. If the prompt is destroyed, it has no idea what zorp means, and accuracy drops to random chance (33%). If the prompt survives compression, it still knows zorp = happiness.

**Task 2: Made-Up Logic Rules**
> "If the input has a color AND a number > 5, output ALPHA. If it has an animal AND no numbers, output BETA..."

Same idea -- rules that exist nowhere in training data. Random chance: 25%.

We then compressed the prompt using four strategies and measured accuracy at each level:

| Strategy | What it does |
|----------|-------------|
| **Random Dropout** | Delete random words (the dumb baseline) |
| **Stopword Removal** | Only delete filler words like "the", "a", "is" |
| **Entity-Preserving** | Delete random words but protect numbers, labels, and proper nouns |
| **LLM-Guided** | Ask Claude to intelligently rewrite the prompt shorter |

Each condition: 200 test examples, 5 random seeds, temperature=0. All results verified and reproducible.

---

## The Results

### The Hero Chart

![Main Curve](figures/main_curve.png)

This is the core finding. Each line shows how accuracy changes as we remove more and more words from the prompt. The x-axis is **actual compression** -- the real percentage of words removed.

**What jumps out:**
- **LLM-Guided (purple)** stays near-perfect up to 75% compression, and still hits 89% accuracy even at 83% compression. It knows which words matter.
- **Stopword Removal (blue)** is perfect but maxes out at ~30% -- there simply aren't more stopwords to remove.
- **Random Dropout (red)** degrades steadily and falls below random chance at the extremes. Removing random words is a coin flip on whether you destroy something critical.
- **Entity-Preserving (green)** is slightly better than random but still collapses.

### At 50% Compression

![Strategy Comparison](figures/strategy_comparison.png)

Note the actual compression each strategy achieved (in parentheses) -- some strategies can't or won't compress as far as others. This is itself a finding.

### Which Part of the Prompt Matters?

![Section Ablation](figures/section_ablation.png)

We compressed different sections of the prompt independently to figure out what's actually carrying the signal:

| What we compressed | Task 1 Accuracy | Task 2 Accuracy |
|-------------------|----------------|----------------|
| Nothing (full prompt) | 99.0% | 84.5% |
| Examples only (kept instructions intact) | 99.6% | 84.5% |
| Instructions only (kept examples intact) | 93.8% | 82.6% |
| Both | 72.7% | 59.8% |

**Compressing the few-shot examples had literally zero effect.** The model doesn't need your 8 examples of what "zorp" means -- it just needs the definition "zorp = happiness." This directly contradicts the common practice of stuffing prompts with examples.

### How Reliable Are These Methods?

![Variance Plot](figures/variance_plot.png)

Random dropout isn't just worse on average -- it's a gamble. Different random seeds at the same compression rate gave wildly different results (up to 28% standard deviation). One seed preserves the label definitions by luck; another destroys them.

LLM-guided and stopword removal produce consistent results every time.

---

## What This Proves

### 1. ~70% of a typical prompt is filler

LLM-guided compression removed 83% of words and the model still performed at 89% accuracy. The model only needs the semantic core -- the actual definitions, rules, and task structure. Everything else (filler phrases, redundant explanations, verbose examples) contributes nothing.

### 2. Not all words are equal

Random deletion at 50% gives 66% accuracy. LLM-guided at 50% gives 99.5%. Same amount removed, completely different outcome. **Which words you remove matters more than how many you remove.**

### 3. Definitions > demonstrations

The common prompt engineering pattern of "show the model 8 examples" barely matters compared to clearly stating what the labels/rules mean. One clear definition outweighs multiple demonstrations.

### 4. Compression tolerance depends on task complexity

Simple classification (map emotion to label) survives heavy compression. Complex logic (multi-rule conditional classification) breaks earlier. More complex instructions need more redundancy to survive compression.

---

## Real-World Applications

### Save money on API calls
If you're running classification, moderation, or extraction prompts at scale, strip all stopwords from your system prompt. That's a ~30% token reduction for free, with <1% accuracy loss. No extra API call needed -- just a regex.

### Optimize prompt length for latency
Shorter prompts = faster responses. For real-time applications, compressing your prompt with an LLM call upfront (once) and reusing the compressed version (thousands of times) is a net win.

### Decide what to cut when hitting context limits
When your prompt is too long, don't cut examples -- cut filler from your instructions. The definitions and rules are load-bearing; the demonstrations are expendable. This reverses common intuition.

### Evaluate prompt robustness
Use compression as a stress test. If your prompt breaks at 20% random dropout, it's fragile -- probably over-relying on specific phrasing. If it survives 50%, the core signal is strong.

### Design better prompts from the start
Knowing that ~70% of words are filler suggests you should write tighter prompts from the start. Lead with definitions. Be concise. Skip the verbose preamble.

---

## Limitations

- **Two tasks.** Real-world prompts are more varied -- generative tasks (writing, coding) might behave differently than classification.
- **Same model for compression and evaluation.** Claude compressed the prompts and Claude evaluated them. Cross-model testing (compress with Claude, evaluate with GPT-4) would strengthen generalizability.
- **Classification only.** Open-ended generation tasks may depend on prompt tone, style, and nuance in ways classification doesn't.

These are real caveats, not disclaimers. The findings are robust within scope, but scope is limited to structured classification tasks.

---

## Statistical Rigor

- 26 pairwise t-tests reached significance (p < 0.05)
- Stopword removal significantly outperforms random dropout and entity-preserving at 50%+ compression
- 5 random seeds per stochastic condition, 200 examples per condition
- 95% confidence intervals computed for all accuracy estimates
- All results verified: no data leakage, no parsing inflation, manual accuracy recounts match

---

## Reproduce This

```bash
pip install anthropic scipy matplotlib seaborn numpy python-dotenv
echo "ANTHROPIC_API_KEY=your-key" > .env

python src/generate_datasets.py       # Generate test data
python src/run_experiments.py          # Run experiments (~2-3 hours, checkpointed)
python src/analyze_results.py          # Statistical analysis
python src/plot_results.py             # Generate figures
```

Interrupted runs resume from checkpoint automatically. All API responses cached to disk.

```
├── src/                           # All experiment code
│   ├── generate_datasets.py       # Test dataset generation
│   ├── compression_strategies.py  # 4 compression implementations
│   ├── eval_harness.py            # API calling + response parsing
│   ├── run_experiments.py         # Main experiment loop
│   ├── analyze_results.py         # Statistics + aggregation
│   └── plot_results.py            # Chart generation
├── data/                          # Test datasets + API cache
├── results/                       # Raw + aggregated results
└── figures/                       # All charts
```

**Model:** Claude Sonnet 4 (`claude-sonnet-4-20250514`), temperature=0, 70,000 total inference calls.
