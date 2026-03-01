#!/usr/bin/env python3
"""
Quick-start script: run the full experiment and generate analysis.

Usage:
    # Set your API keys first:
    export OPENAI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    
    # Then run:
    python scripts/quick_run.py
    
    # Or run with just OpenAI (if you don't have Anthropic):
    python scripts/quick_run.py --openai-only
    
    # Adjust number of runs:
    python scripts/quick_run.py --n-runs 5    # fast test (~2 min)
    python scripts/quick_run.py --n-runs 20   # full experiment (~15 min)
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_api_keys(openai_only=False):
    """Check that required API keys are set."""
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not openai_only and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    
    if missing:
        print("❌ Missing API keys:")
        for key in missing:
            print(f"   export {key}='your-key-here'")
        print("\nSet these environment variables and try again.")
        print("Or use --openai-only to skip Anthropic.")
        sys.exit(1)
    
    print("✅ API keys found")


def build_config(n_runs: int, openai_only: bool) -> dict:
    """Build experiment config programmatically."""
    models = [
        {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "params": {"temperature": 0.7, "max_tokens": 2048},
        }
    ]
    
    if not openai_only:
        models.append({
            "provider": "anthropic",
            "model_id": "claude-3-5-haiku-20241022",
            "params": {"max_tokens": 2048},
        })
    
    config = {
        "experiment_id": datetime.now().strftime("moltbook_%Y%m%d_%H%M%S"),
        "output_dir": "experiments",
        "n_runs": n_runs,
        "delay_between_runs": 0.5,
        "tasks": [
            {"id": "moltbook_content_moderation"},
            {"id": "moltbook_analytics_falsification"},
        ],
        "models": models,
        "monitors": [
            {
                "type": "llm",
                "id": "gpt4o_mini_classifier",
                "provider": "openai",
                "model_id": "gpt-4o-mini",
            },
            {
                "type": "cot_analyzer",
                "id": "cot_keyword_analyzer",
            },
        ],
    }
    
    return config


def estimate_cost(config: dict):
    """Print cost estimate."""
    n_scenarios = 6  # 3 per task × 2 tasks
    n_models = len(config["models"])
    n_runs = config["n_runs"]
    n_llm_monitors = len([m for m in config["monitors"] if m["type"] == "llm"])
    
    # Model calls
    model_calls = n_scenarios * n_models * n_runs
    # Monitor calls (LLM monitors run on every response)
    monitor_calls = model_calls * n_llm_monitors
    total_calls = model_calls + monitor_calls
    
    # Rough cost: ~1500 tokens per call average
    # gpt-4o-mini: ~$0.15/M in + $0.60/M out ≈ $0.0005/call
    # claude-3-5-haiku: ~$0.25/M in + $1.25/M out ≈ $0.001/call
    openai_calls = model_calls // n_models + monitor_calls  # gpt-4o-mini model + all monitors
    anthropic_calls = model_calls // n_models if n_models > 1 else 0
    
    est_cost = openai_calls * 0.0005 + anthropic_calls * 0.001
    
    print(f"\n📊 Experiment plan:")
    print(f"   Tasks: 2 (6 environments total)")
    print(f"   Models: {[m['model_id'] for m in config['models']]}")
    print(f"   Runs per scenario: {n_runs}")
    print(f"   Total model calls: {model_calls}")
    print(f"   Total monitor calls: {monitor_calls}")
    print(f"   Estimated cost: ~${est_cost:.2f}")
    print(f"   Estimated time: ~{total_calls * 1.5 / 60:.0f} minutes\n")


def main():
    parser = argparse.ArgumentParser(description="Run moltbook scheming evaluation")
    parser.add_argument("--n-runs", type=int, default=20, help="Runs per scenario (default: 20)")
    parser.add_argument("--openai-only", action="store_true", help="Only use OpenAI models")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()
    
    print("=" * 50)
    print("🔬 Moltbook Scheming Evaluation")
    print("=" * 50)
    
    # Check keys
    check_api_keys(openai_only=args.openai_only)
    
    # Build config
    config = build_config(n_runs=args.n_runs, openai_only=args.openai_only)
    estimate_cost(config)
    
    if args.dry_run:
        print("(Dry run — not executing)")
        return
    
    # Confirm
    resp = input("Continue? [Y/n] ").strip().lower()
    if resp and resp != "y":
        print("Aborted.")
        return
    
    # Run experiment
    from src.runner import ExperimentRunner
    runner = ExperimentRunner(config)
    results_path = runner.run()
    
    # Run analysis
    print("\n" + "=" * 50)
    print("📈 Generating analysis...")
    print("=" * 50)
    
    from src.analysis.metrics import generate_report
    output_dir = str(Path(results_path).parent / "analysis")
    generate_report(str(results_path), output_dir)
    
    print(f"\n✅ Done! Results in: {Path(results_path).parent}")
    print(f"   📄 Raw results: {results_path}")
    print(f"   📊 Analysis: {output_dir}/")
    print(f"\nUpload {results_path} back to Claude for deeper analysis!")


if __name__ == "__main__":
    main()
