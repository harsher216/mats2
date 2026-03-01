#!/usr/bin/env python3
"""
Run an experiment from a YAML config file.

Usage:
    python scripts/run_experiment.py --config config/experiment.yaml
    python scripts/run_experiment.py --config config/experiment.yaml --dry-run
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.runner import run_from_config, load_config, TASK_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Run scheming evaluation experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    args = parser.parse_args()
    
    if args.list_tasks:
        print("Available tasks:")
        for task_id, task_cls in TASK_REGISTRY.items():
            print(f"  {task_id}: {task_cls.description}")
            task = task_cls()
            scenarios = task.generate_scenarios()
            print(f"    Environments: {len(scenarios)}")
            for s in scenarios:
                print(f"      - {s.environment_id}")
        return
    
    config = load_config(args.config)
    
    if args.dry_run:
        print("Experiment config:")
        print(f"  Tasks: {[t['id'] for t in config.get('tasks', [])]}")
        print(f"  Models: {[m['model_id'] for m in config.get('models', [])]}")
        print(f"  Runs per scenario: {config.get('n_runs', 1)}")
        
        # Count total API calls
        total_scenarios = 0
        for task_config in config.get("tasks", []):
            task_cls = TASK_REGISTRY.get(task_config["id"])
            if task_cls:
                task = task_cls()
                total_scenarios += len(task.generate_scenarios())
        
        n_models = len(config.get("models", []))
        n_runs = config.get("n_runs", 1)
        n_monitors = len([m for m in config.get("monitors", []) if m["type"] == "llm"])
        
        total_calls = total_scenarios * n_models * n_runs * (1 + n_monitors)
        print(f"  Total scenarios: {total_scenarios}")
        print(f"  Estimated API calls: {total_calls}")
        return
    
    results_path = run_from_config(args.config)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
