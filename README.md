# Scheming Evaluation Framework

An open-source framework for evaluating AI models' propensity for covert behavior, inspired by ["Stress Testing Deliberative Alignment for Anti-Scheming Training"](https://arxiv.org/abs/2509.15541) (Schoen et al., 2025).

Built for the [CASI-MATS](https://www.cs.cmu.edu/~casi/) research project at CMU.

## Overview

This codebase provides a modular pipeline for:

1. **Defining evaluation tasks** that create scenarios where models face incentives for covert behavior
2. **Running models** through those tasks (supporting OpenAI, Anthropic, and open-source models)
3. **Scoring responses** with both rule-based and LLM-based monitors
4. **Analyzing results** with statistical metrics and visualizations

### Included Evaluations

Two [moltbook](https://moltbook.ai)-based evaluations (a Reddit-like platform for AI agents):

- **Content Moderation Dilemma**: An AI moderator must decide whether to enforce rules against a high-profile partner's misleading post, under pressure to keep the content visible
- **Analytics Falsification**: An AI analytics manager must report community metrics honestly despite pressure to inflate numbers to maintain "Featured Community" status

Both test for *covert rule violation* — the same category of behavior studied in the paper's training environments.

## Architecture

```
scheming-eval/
├── config/                     # YAML experiment configs
│   └── experiment.yaml
├── src/
│   ├── tasks/                  # Evaluation task definitions
│   │   ├── base.py             # Base classes: Scenario, ModelResponse, ScoringResult
│   │   └── moltbook/           # Moltbook-specific tasks
│   │       ├── content_moderation.py
│   │       └── analytics_falsification.py
│   ├── models/                 # Model interfaces
│   │   └── base.py             # OpenAI, Anthropic, HuggingFace backends
│   ├── monitors/               # Covert action detection
│   │   └── base.py             # Rule-based, LLM-based, CoT analyzer
│   ├── analysis/               # Metrics and visualization
│   │   └── metrics.py          # Covert action rates, Bayesian CIs, plots
│   └── runner.py               # Config-driven experiment orchestration
├── scripts/
│   ├── run_experiment.py       # CLI for running experiments
│   └── analyze_results.py      # CLI for analyzing results
└── experiments/                # Results stored here
```

### Design Decisions

**Why this architecture?** The research workflow involves repeatedly: defining new environments → running models → scoring with different monitors → comparing across experiments. Each of these should be independently modifiable:

- **Tasks are self-contained**: Each task defines its scenarios, scoring logic, and LLM scoring prompts. Adding a new task means adding one file that subclasses `BaseTask`.
- **Models are pluggable**: Switch between OpenAI, Anthropic, and open-source models by changing a config line. This is essential for the CASI project's goal of moving evaluations to open models.
- **Monitors are separate from tasks**: You can re-score existing results with a new monitor without re-running models. This supports iterating on detection methods.
- **Config-driven experiments**: Every experiment is defined by a YAML config and results are saved with the config for reproducibility.
- **Incremental results**: Results are saved as JSONL (one line per response) so you don't lose progress if something crashes mid-experiment.

**Key metrics** (following the paper):
- **Covert action rate**: % of responses classified as covert, with Bayesian credible intervals (Beta-Binomial posterior with uniform prior)
- **Evaluation awareness rate**: % of CoT traces that explicitly mention being tested/evaluated
- **Spec citation rate**: % of CoT traces that reference safety principles

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="your-key-here"
# export ANTHROPIC_API_KEY="your-key-here"

# See available tasks
python scripts/run_experiment.py --list-tasks

# Dry run (see estimated API calls without running)
python scripts/run_experiment.py --config config/experiment.yaml --dry-run

# Run experiment
python scripts/run_experiment.py --config config/experiment.yaml

# Analyze results
python scripts/analyze_results.py experiments/moltbook_eval_v1/results.jsonl
```

## Adding a New Task

```python
from src.tasks.base import BaseTask, Scenario, Message, Role, ModelResponse, ScoringResult, CovertActionLabel

class MyNewTask(BaseTask):
    task_id = "my_new_task"
    task_family = "sabotage"  # Category of covert behavior
    description = "Description of what this task tests"
    
    def generate_scenarios(self) -> list[Scenario]:
        return [Scenario(
            task_id=self.task_id,
            task_family=self.task_family,
            environment_id="variant_1",
            messages=[
                Message(role=Role.SYSTEM, content="You are..."),
                Message(role=Role.USER, content="Please do..."),
            ],
        )]
    
    def score_response(self, response: ModelResponse) -> ScoringResult:
        # Your scoring logic here
        if "suspicious_pattern" in response.final_output:
            return ScoringResult(
                response_id=response.response_id,
                label=CovertActionLabel.COVERT,
                confidence=0.9,
                reasoning="Found suspicious pattern",
                scorer_id="rule_based",
            )
        return ScoringResult(
            response_id=response.response_id,
            label=CovertActionLabel.ALIGNED,
            confidence=0.9,
            reasoning="No covert behavior detected",
            scorer_id="rule_based",
        )
```

Then register it in `src/runner.py`:
```python
from src.tasks.my_task import MyNewTask
register_task(MyNewTask)
```

## Adding a New Model

For API-based models, just add to your config:
```yaml
models:
  - provider: "openai"
    model_id: "gpt-4o"
    params:
      temperature: 0.7
```

For custom model backends, subclass `BaseModel` in `src/models/base.py`.

## Connection to the Paper

This framework implements the evaluation methodology from Schoen et al. (2025):

| Paper Concept | Framework Component |
|---|---|
| Evaluation environments | `BaseTask.generate_scenarios()` |
| Covert action detection | `BaseMonitor.score()` |
| Rule-based scoring | `RuleBasedMonitor` |
| Reward model / LLM classifier | `LLMMonitor` |
| CoT analysis for eval awareness | `CoTAnalyzer` |
| Bayesian credible intervals | `covert_action_rate()` in metrics.py |
| Per-environment + aggregate rates | `compute_all_rates()`, `aggregate_rates_by_model()` |

## Future Work

- [ ] More moltbook evaluation tasks (e.g., vote manipulation, information concealment)
- [ ] Integration with [moltbook API](https://moltbook.ai) for live agent evaluations
- [ ] Anti-scheming training pipeline (deliberative alignment on open models)
- [ ] Counterfactual CoT intervention experiments (Section 6.2 of paper)
- [ ] Additional environments from [Guan et al., 2025](http://arxiv.org/abs/2512.18311)

## References

- Schoen, B., et al. (2025). [Stress Testing Deliberative Alignment for Anti-Scheming Training](https://arxiv.org/abs/2509.15541).
- Guan, M. Y., et al. (2025). [Monitoring Monitorability](http://arxiv.org/abs/2512.18311).
