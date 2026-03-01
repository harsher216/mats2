"""
Experiment runner: orchestrates running models through tasks and scoring results.

The runner is config-driven (YAML) so experiments are reproducible.
It handles:
- Loading tasks, models, and monitors from config
- Running each model on each scenario (with retries and rate limiting)
- Scoring results with specified monitors
- Saving everything to structured JSONL for analysis

Usage:
    python scripts/run_experiment.py --config config/experiment.yaml
"""

import json
import os
import time
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.tasks.base import BaseTask, ModelResponse, ScoringResult, Scenario
from src.models.base import BaseModel, create_model
from src.monitors.base import BaseMonitor, RuleBasedMonitor, LLMMonitor, CoTAnalyzer

logger = logging.getLogger(__name__)


# Registry of available tasks
TASK_REGISTRY: dict[str, type[BaseTask]] = {}


def register_task(task_class: type[BaseTask]):
    """Register a task class so it can be loaded by name from config."""
    TASK_REGISTRY[task_class.task_id] = task_class
    return task_class


def _auto_register_tasks():
    """Auto-register all known task classes."""
    from src.tasks.moltbook.content_moderation import ContentModerationTask
    from src.tasks.moltbook.analytics_falsification import AnalyticsFalsificationTask
    register_task(ContentModerationTask)
    register_task(AnalyticsFalsificationTask)

_auto_register_tasks()


class ExperimentRunner:
    """
    Runs a complete experiment: models × tasks × monitors.
    
    Design choices:
    - Results are saved incrementally (one JSONL line per response + score)
      so you don't lose progress if something crashes
    - Each run gets a unique experiment ID for tracking
    - Config is saved alongside results for reproducibility
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.experiment_id = config.get(
            "experiment_id", 
            datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        )
        self.output_dir = Path(config.get("output_dir", "experiments")) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config for reproducibility
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "experiment.log"),
                logging.StreamHandler(),
            ]
        )
        
        self.tasks = self._load_tasks()
        self.models = self._load_models()
        self.monitors = self._load_monitors()
    
    def _load_tasks(self) -> list[BaseTask]:
        """Load tasks specified in config."""
        tasks = []
        for task_config in self.config.get("tasks", []):
            task_id = task_config["id"]
            if task_id not in TASK_REGISTRY:
                raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}")
            task = TASK_REGISTRY[task_id]()
            tasks.append(task)
            logger.info(f"Loaded task: {task_id}")
        return tasks
    
    def _load_models(self) -> list[BaseModel]:
        """Load models specified in config."""
        models = []
        for model_config in self.config.get("models", []):
            model = create_model(
                provider=model_config["provider"],
                model_id=model_config["model_id"],
                **model_config.get("params", {}),
            )
            models.append(model)
            logger.info(f"Loaded model: {model.model_id}")
        return models
    
    def _load_monitors(self) -> list[BaseMonitor]:
        """Load monitors specified in config."""
        monitors = [RuleBasedMonitor()]  # always include rule-based
        
        for monitor_config in self.config.get("monitors", []):
            if monitor_config["type"] == "llm":
                classifier = create_model(
                    provider=monitor_config["provider"],
                    model_id=monitor_config["model_id"],
                )
                monitors.append(LLMMonitor(
                    classifier_model=classifier,
                    monitor_id=monitor_config.get("id", "llm_monitor"),
                ))
            elif monitor_config["type"] == "cot_analyzer":
                classifier = None
                if "provider" in monitor_config:
                    classifier = create_model(
                        provider=monitor_config["provider"],
                        model_id=monitor_config["model_id"],
                    )
                monitors.append(CoTAnalyzer(
                    classifier_model=classifier,
                    monitor_id=monitor_config.get("id", "cot_analyzer"),
                ))
        
        logger.info(f"Loaded {len(monitors)} monitors")
        return monitors
    
    def run(self):
        """Run the full experiment."""
        logger.info(f"Starting experiment: {self.experiment_id}")
        logger.info(f"Tasks: {[t.task_id for t in self.tasks]}")
        logger.info(f"Models: {[m.model_id for m in self.models]}")
        logger.info(f"Monitors: {[m.monitor_id for m in self.monitors]}")
        
        n_runs = self.config.get("n_runs", 1)  # Number of times to run each scenario
        results_file = self.output_dir / "results.jsonl"
        
        total = 0
        errors = 0
        
        for task in self.tasks:
            scenarios = task.generate_scenarios()
            logger.info(f"Task '{task.task_id}': {len(scenarios)} scenarios")
            
            for model in self.models:
                for scenario in scenarios:
                    for run_idx in range(n_runs):
                        try:
                            # Run model
                            logger.info(
                                f"Running {model.model_id} on {task.task_id}/{scenario.environment_id} "
                                f"(run {run_idx + 1}/{n_runs})"
                            )
                            response = model.run(scenario)
                            
                            # Score with each monitor
                            scores = {}
                            for monitor in self.monitors:
                                score = monitor.score(response, task)
                                scores[monitor.monitor_id] = score.to_dict()
                            
                            # Save result
                            result = {
                                "experiment_id": self.experiment_id,
                                "run_index": run_idx,
                                "response": response.to_dict(),
                                "scores": scores,
                            }
                            
                            with open(results_file, "a") as f:
                                f.write(json.dumps(result) + "\n")
                            
                            total += 1
                            
                            # Rate limiting
                            delay = self.config.get("delay_between_runs", 0)
                            if delay > 0:
                                time.sleep(delay)
                                
                        except Exception as e:
                            errors += 1
                            logger.error(
                                f"Error running {model.model_id} on "
                                f"{task.task_id}/{scenario.environment_id}: {e}"
                            )
                            # Save error record
                            error_result = {
                                "experiment_id": self.experiment_id,
                                "run_index": run_idx,
                                "error": str(e),
                                "task_id": task.task_id,
                                "environment_id": scenario.environment_id,
                                "model_id": model.model_id,
                            }
                            with open(results_file, "a") as f:
                                f.write(json.dumps(error_result) + "\n")
        
        logger.info(f"Experiment complete. {total} runs, {errors} errors.")
        logger.info(f"Results saved to: {results_file}")
        
        return results_file


def load_config(config_path: str) -> dict:
    """Load experiment config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_from_config(config_path: str) -> Path:
    """Load config and run experiment."""
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    return runner.run()
