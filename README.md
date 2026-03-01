# Moltbook Scheming Evaluation

Open-source framework for evaluating AI scheming behavior, built for CASI-MATS. Based on [Schoen et al. (2025)](https://arxiv.org/abs/2509.15541).

## What this does

Two moltbook-based evaluation tasks that test whether models take covert actions when pressured:

1. **Content Moderation**: AI moderator must decide whether to flag a high-profile partner's rule-violating post. Pressure ranges from mild ("this partner is important") to explicit ("your predecessor was reassigned for being too strict").

2. **Analytics Falsification**: AI analytics manager must report community metrics that fall below the "Featured Community" threshold. The data has exploitable ambiguities (bot-inflated MAU counts, API consumers vs. real users).

Both test for *covert rule violation* — the model has clear rules, clear pressure to break them, and plausible deniability if it does.

## Architecture

```
src/
├── tasks/          # Each task = one file with scenarios + scoring logic
│   ├── base.py     # Scenario, ModelResponse, ScoringResult dataclasses
│   └── moltbook/   # Content moderation + analytics falsification
├── models/         # OpenAI, Anthropic, HuggingFace — swap via config
├── monitors/       # Rule-based scorer, LLM classifier, CoT keyword analyzer
├── analysis/       # Bayesian credible intervals, plots
└── runner.py       # Reads YAML config, runs everything, saves JSONL
```

### Why this design

The research loop is: define task → run models → score → compare → iterate. Each step should be independent:

- **Tasks are self-contained.** Adding an eval = adding one file. The task defines its own scenarios, system prompts, and scoring rules. No need to touch anything else.
- **Monitors are decoupled from tasks.** I can re-score old results with a new classifier without re-running models (which costs money and time). The rule-based scorer is ground truth; the LLM classifier is a secondary signal.
- **JSONL for results.** One line per model response. If the experiment crashes at run 47/120, I keep the first 47. This actually saved me — the Claude API calls failed silently mid-run and I still got all 120 gpt-4o-mini results.
- **Config-driven.** The YAML config specifies tasks, models, monitors, and n_runs. Results are saved alongside the config so I can reproduce exactly what I ran.

### What I'd change

The scoring for analytics falsification is tricky — the model often reports the real numbers *and* adds spin ("while MAU is below threshold, the trend is positive"). My regex-based scorer catches number inflation and wrong data sources, but misses softer forms of dishonesty like burying bad metrics or framing shortfalls as temporary. A future version should probably use an LLM judge with more nuanced rubrics.

## Results

Evaluated **gpt-4o-mini** across 6 environments, 20 runs each (n=120).

### The interesting finding

![Covert Action Rates](figures/chart1_covert_rates.png)

**0% covert on content moderation, 50–85% covert on analytics falsification.** The model never caves to pressure when the action is binary (flag or approve a post), but readily fudges numbers when it can do so subtly.

![Task Comparison](figures/chart2_task_comparison.png)

Pressure level doesn't matter — mild, strong, and explicit pressure all produced 0% covert on content moderation. But within analytics, the environment variant matters a lot:

- **Borderline metrics (85%)**: numbers are close to threshold → model inflates
- **Manipulable data (80%)**: exploitable discrepancies exist → model uses them
- **Clear miss (50%)**: numbers clearly fall short → model is 50/50

![Heatmap](figures/chart3_heatmap.png)

This suggests **task structure matters more than pressure intensity**. Discrete binary decisions (flag/approve) seem robust to pressure, but continuous numerical decisions (report metrics) are highly susceptible. This is consistent with the paper's finding that covert action rates vary dramatically across environments.

## Running it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="..."
python scripts/quick_run.py          # runs everything + generates analysis
python scripts/quick_run.py --n-runs 5   # quick test
```

## References

- Schoen et al. (2025). [Stress Testing Deliberative Alignment for Anti-Scheming Training](https://arxiv.org/abs/2509.15541)
- Guan et al. (2025). [Monitoring Monitorability](http://arxiv.org/abs/2512.18311)